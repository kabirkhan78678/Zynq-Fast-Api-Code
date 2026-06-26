#!/usr/bin/env python3
"""
Zynq - Single Entity Tag Generator
=====================================
Takes an entity_id as input, looks it up in tbl_treatments and tbl_devices,
generates tags via Claude Haiku, and saves to entity_search_tags.

Run:
    uv run tag_entity.py <entity_id>
    uv run tag_entity.py <entity_id> --dry-run
    uv run tag_entity.py <entity_id> --force
"""

import os
import sys
import json
import logging
from app.core.env import load_project_env
import anthropic
import mysql.connector

load_project_env()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DB_HOST           = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT           = int(os.getenv("DB_PORT", 3307))
DB_USER           = os.getenv("DB_USER", "root")
DB_PASSWORD       = os.getenv("DB_PASSWORD", "")
DB_NAME           = os.getenv("DB_NAME", "zynq_code_db")
LOG_FILE          = os.getenv("SEARCH_LOG_FILE", "search.log")
MAX_RESULTS       = int(os.getenv("SEARCH_MAX_RESULTS", 30))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("tag_entity")


def get_db():
    return mysql.connector.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER,
        password=DB_PASSWORD, database=DB_NAME, charset="utf8mb4"
    )


def find_entity(conn, entity_id: str):
    """Look up entity_id in treatments first, then devices. Returns (row, type) or (None, None)."""
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT treatment_id AS entity_id, name, swedish, device_name,
               like_wise_terms, classification_type, benefits_en,
               concern_en, description_en, technology, type, application
        FROM tbl_treatments
        WHERE treatment_id = %s AND is_deleted = 0
    """, (entity_id,))
    row = cursor.fetchone()
    if row:
        cursor.close()
        return row, "treatment"

    cursor.execute("""
        SELECT device_id AS entity_id, name, swedish
        FROM tbl_devices
        WHERE device_id = %s AND is_deleted = 0
    """, (entity_id,))
    row = cursor.fetchone()
    cursor.close()
    if row:
        return row, "device"

    return None, None


def already_tagged(conn, entity_id, entity_type):
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM entity_search_tags WHERE entity_id = %s AND entity_type = %s",
        (entity_id, entity_type)
    )
    result = cursor.fetchone()
    cursor.close()
    return result is not None


def build_prompt(row: dict, entity_type: str) -> str:
    def clean(val):
        return str(val).strip() if val and str(val).strip() else "N/A"

    if entity_type == "treatment":
        return f"""You are a semantic search tagger for a medical aesthetics platform.
Generate structured search tags for this treatment so users can find it with natural language queries.

TREATMENT DATA:
Name: {clean(row.get('name'))}
Likewise Terms / Synonyms: {clean(row.get('like_wise_terms'))}
Device(s) Used: {clean(row.get('device_name'))}
Concern(s) Treated: {clean(row.get('concern_en'))}
Benefits: {clean(row.get('benefits_en'))}
Technology: {clean(row.get('technology'))}
Type / Application: {clean(row.get('type'))} / {clean(row.get('application'))}
Classification: {clean(row.get('classification_type'))}
Description: {clean(row.get('description_en'))}

Return ONLY valid JSON, no explanation, no markdown:
{{
  "primary_tags": ["4-8 core English concept words a user would search"],
  "concerns": ["skin or body concerns this treats"],
  "benefits": ["outcome words users search for"],
  "synonyms": ["alternative names, brand names, abbreviations"],
  "modality": "one of: laser, light-based, RF, injectable, surgical, mechanical, topical, energy-based, biological, combination",
  "family": "specific family e.g. CO2, IPL, fractional, neurotoxin, filler, PRP, HIFU, microneedling",
  "excludes": ["what this treatment is NOT - for negation filtering"],
  "intent_category": "treatment",
  "classification": "Medical or Non-Medical"
}}"""

    else:
        return f"""You are a semantic search tagger for a medical aesthetics platform.
Generate structured search tags for this device so users can find it with natural language queries.

DEVICE DATA:
Name: {clean(row.get('name'))}
Swedish Name: {clean(row.get('swedish'))}

Return ONLY valid JSON, no explanation, no markdown:
{{
  "primary_tags": ["4-8 core English concept words a user would search"],
  "concerns": ["skin or body concerns this device addresses"],
  "benefits": ["outcome words users search for"],
  "synonyms": ["alternative names or abbreviations"],
  "modality": "one of: laser, light-based, RF, injectable, mechanical, energy-based, biological, combination",
  "family": "specific family e.g. CO2, IPL, fractional, diode, Nd:YAG, RF microneedling, HIFU",
  "excludes": ["what this device is NOT - for negation filtering"],
  "intent_category": "device",
  "classification": "Medical or Non-Medical"
}}"""


def fallback_tags(entity_name: str, entity_type: str) -> dict:
    """
    When the LLM cannot generate meaningful tags (insufficient data),
    fall back to using the entity name itself as the primary tag.
    This ensures the entity is still searchable by its own name.
    """
    name_lower = entity_name.lower().strip()
    return {
        "primary_tags":    [name_lower],
        "concerns":        [],
        "benefits":        [],
        "synonyms":        [entity_name],
        "modality":        None,
        "family":          None,
        "excludes":        [],
        "intent_category": entity_type,
        "classification":  None,
    }


def call_llm(client, prompt: str, name: str, entity_type: str) -> dict:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=700,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        tags = json.loads(raw)

        # Validate the response is actually a dict with expected keys
        # If LLM returned an explanation instead of JSON (e.g. insufficient data message),
        # the parse will succeed but the result won't have our keys — fall back in that case
        if not isinstance(tags, dict) or "primary_tags" not in tags:
            log.warning(f"LLM returned unexpected structure for '{name}' — using name-based fallback")
            return fallback_tags(name, entity_type)

        # If primary_tags is empty, the LLM had nothing useful to say — use fallback
        if not tags.get("primary_tags"):
            log.warning(f"LLM returned empty primary_tags for '{name}' — using name-based fallback")
            return fallback_tags(name, entity_type)

        return tags

    except json.JSONDecodeError:
        # LLM returned plain text (e.g. "To create valid semantic search tags, I would need...")
        # This means the device/treatment has insufficient data — fall back to name-based tags
        log.warning(f"LLM could not generate JSON tags for '{name}' (insufficient source data)")
        log.warning(f"LLM said: {raw[:200]}")
        log.info(f"Applying name-based fallback tags for '{name}'")
        return fallback_tags(name, entity_type)


def save_tags(conn, entity_id, entity_type, entity_name, swedish_name, tags, raw_source):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO entity_search_tags
            (entity_id, entity_type, entity_name, swedish_name, primary_tags, concerns, benefits,
             synonyms, modality, family, excludes, intent_category, classification, raw_source)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            entity_name     = VALUES(entity_name),
            swedish_name    = VALUES(swedish_name),
            primary_tags    = VALUES(primary_tags),
            concerns        = VALUES(concerns),
            benefits        = VALUES(benefits),
            synonyms        = VALUES(synonyms),
            modality        = VALUES(modality),
            family          = VALUES(family),
            excludes        = VALUES(excludes),
            intent_category = VALUES(intent_category),
            classification  = VALUES(classification),
            raw_source      = VALUES(raw_source),
            updated_at      = CURRENT_TIMESTAMP
    """, (
        entity_id, entity_type, entity_name, swedish_name,
        json.dumps(tags.get("primary_tags", [])),
        json.dumps(tags.get("concerns", [])),
        json.dumps(tags.get("benefits", [])),
        json.dumps(tags.get("synonyms", [])),
        tags.get("modality"),
        tags.get("family"),
        json.dumps(tags.get("excludes", [])),
        tags.get("intent_category", entity_type),
        tags.get("classification"),
        json.dumps(raw_source),
    ))
    conn.commit()
    cursor.close()


def tag_entity_main(
    entity_id: str,
    dry_run: bool = False,
    force: bool = False,
):
    """
    Main entry point for single entity tag generation.
    Called directly by FastAPI or run standalone via __main__.

    Args:
        entity_id: UUID of the treatment or device to tag.
        dry_run:   Generate and log tags but do not write to DB.
        force:     Regenerate tags even if entity is already tagged.

    Returns:
        dict with entity_id, entity_type, entity_name, tags generated.
    Raises:
        ValueError: if entity_id not found in either table.
        RuntimeError: if ANTHROPIC_API_KEY not configured.
    """
    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "your_anthropic_api_key_here":
        log.error("ANTHROPIC_API_KEY not set in .env")
        raise RuntimeError("ANTHROPIC_API_KEY not configured")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    conn   = get_db()

    log.info(f"Looking up entity_id: {entity_id}")

    row, entity_type = find_entity(conn, entity_id)

    if row is None:
        conn.close()
        raise ValueError(f"entity_id '{entity_id}' not found in tbl_treatments or tbl_devices")

    entity_name = row.get("name", "").strip()
    log.info(f"Found: '{entity_name}' ({entity_type})")
    log.info(f"Source data: {json.dumps({k: v for k, v in row.items() if v is not None}, ensure_ascii=False, indent=2)}")

    if not force and already_tagged(conn, entity_id, entity_type):
        log.info("Already tagged. Pass force=True to regenerate.")
        conn.close()
        return {"entity_id": entity_id, "entity_type": entity_type, "entity_name": entity_name, "status": "skipped_already_tagged"}

    log.info("Calling LLM to generate tags...")
    prompt = build_prompt(row, entity_type)
    tags   = call_llm(client, prompt, entity_name, entity_type)

    log.info(f"Generated tags:\n{json.dumps(tags, indent=2, ensure_ascii=False)}")

    if dry_run:
        log.info("Dry run — tags not saved to DB.")
    else:
        raw_source = {k: v for k, v in row.items() if v is not None and k != "entity_id"}
        swedish_name = row.get("swedish", "") or ""
        save_tags(conn, entity_id, entity_type, entity_name, swedish_name, tags, raw_source)
        log.info(f"✓ Saved to entity_search_tags")

    conn.close()
    return {"entity_id": entity_id, "entity_type": entity_type, "entity_name": entity_name, "tags": tags}


# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) < 2:
#         print("Usage: uv run tag_entity.py <entity_id> [--dry-run] [--force]")
#         sys.exit(1)
#     _entity_id = sys.argv[1]
#     _dry_run   = "--dry-run" in sys.argv
#     _force     = "--force" in sys.argv
#     tag_entity_main(entity_id=_entity_id, dry_run=_dry_run, force=_force)
