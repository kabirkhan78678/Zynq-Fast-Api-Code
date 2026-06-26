#!/usr/bin/env python3
"""
Zynq Search - Batch Tag Generator
==================================
Reads tbl_treatments and tbl_devices, generates semantic search tags
via Claude Haiku for each row, and stores them in entity_search_tags.

Run with:
    uv run batch_tag_generator.py

Flags:
    --dry-run       Print tags to console without writing to DB
    --entity-type   'treatment' | 'device' | 'all' (default: all)
    --limit         Max rows to process per table (default: all)
    --force         Re-generate tags even if row already has tags
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from app.core.env import load_project_env
import anthropic
import mysql.connector

# ── Load env ──────────────────────────────────────────────────────────────────
load_project_env()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DB_HOST           = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT           = int(os.getenv("DB_PORT", 3307))
DB_USER           = os.getenv("DB_USER", "root")
DB_PASSWORD       = os.getenv("DB_PASSWORD", "")
DB_NAME           = os.getenv("DB_NAME", "zynq_code_db")
LOG_FILE          = os.getenv("SEARCH_LOG_FILE", "search.log")
MAX_RESULTS       = int(os.getenv("SEARCH_MAX_RESULTS", 30))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("batch_tag_generator")


# ── DB helpers ────────────────────────────────────────────────────────────────
def get_db_connection():
    return mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4",
    )


def fetch_treatments(conn, limit=None):
    """Fetch APPROVED, non-deleted treatments with searchable content."""
    cursor = conn.cursor(dictionary=True)
    sql = """
        SELECT
            treatment_id   AS entity_id,
            name,
            swedish,
            device_name,
            like_wise_terms,
            classification_type,
            benefits_en,
            concern_en,
            description_en,
            technology,
            type,
            application
        FROM tbl_treatments
        WHERE is_deleted = 0
          AND approval_status = 'APPROVED'
          AND (name IS NOT NULL AND TRIM(name) != '')
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    cursor.execute(sql)
    rows = cursor.fetchall()
    cursor.close()
    return rows


def fetch_devices(conn, limit=None):
    """Fetch APPROVED, non-deleted devices with searchable content."""
    cursor = conn.cursor(dictionary=True)
    sql = """
        SELECT
            device_id   AS entity_id,
            name,
            swedish
        FROM tbl_devices
        WHERE is_deleted = 0
          AND approval_status = 'APPROVED'
          AND (name IS NOT NULL AND TRIM(name) != '')
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    cursor.execute(sql)
    rows = cursor.fetchall()
    cursor.close()
    return rows


def already_tagged(conn, entity_id, entity_type):
    """Check if entity already has tags in entity_search_tags."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM entity_search_tags WHERE entity_id = %s AND entity_type = %s",
        (entity_id, entity_type),
    )
    result = cursor.fetchone()
    cursor.close()
    return result is not None


def save_tags(conn, entity_id, entity_type, entity_name, swedish_name, tags, raw_source, dry_run=False):
    """Upsert generated tags into entity_search_tags."""
    if dry_run:
        log.info(f"[DRY RUN] Would save tags for {entity_type}:{entity_name} (swedish: {swedish_name})")
        return

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
        entity_id,
        entity_type,
        entity_name,
        swedish_name,
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


# ── LLM tag generation ────────────────────────────────────────────────────────
def build_treatment_prompt(row: dict) -> str:
    # Parse comma-separated fields into clean lists for context
    def clean(val):
        return val.strip() if val and val.strip() else "N/A"

    return f"""You are a semantic search tagger for a medical aesthetics platform.
Your job is to generate structured search tags for a treatment so users can find it
with natural language queries in English or Swedish.

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

Return ONLY a valid JSON object. No explanation. No markdown. No extra text.
Schema:
{{
  "primary_tags": ["4-8 core English concept words a user would search to find this treatment"],
  "concerns": ["skin or body concerns this treats, e.g. pigmentation, acne, wrinkles, hair loss"],
  "benefits": ["outcome words users search for, e.g. tighter skin, even tone, hair regrowth"],
  "synonyms": ["alternative names, brand names, abbreviations for this treatment"],
  "modality": "single string: one of laser, light-based, RF, injectable, surgical, mechanical, topical, energy-based, biological, combination",
  "family": "single string: specific family e.g. CO2, IPL, fractional, neurotoxin, filler, PRP, HIFU, microneedling",
  "excludes": ["what this treatment is NOT - critical for negation filtering e.g. if non-surgical write surgical"],
  "intent_category": "treatment",
  "classification": "Medical or Non-Medical"
}}"""


def build_device_prompt(row: dict) -> str:
    def clean(val):
        return val.strip() if val and val.strip() else "N/A"

    return f"""You are a semantic search tagger for a medical aesthetics platform.
Your job is to generate structured search tags for a device so users can find it
with natural language queries in English or Swedish.

DEVICE DATA:
Name: {clean(row.get('name'))}
Swedish Name: {clean(row.get('swedish'))}

Based on your knowledge of this device in the medical aesthetics industry,
generate accurate semantic tags. If this looks like test/dummy data (random names,
numbers only, wavelength fragments like "900-1200nm"), return empty arrays.

Return ONLY a valid JSON object. No explanation. No markdown. No extra text.
Schema:
{{
  "primary_tags": ["4-8 core English concept words a user would search to find this device"],
  "concerns": ["skin or body concerns this device addresses"],
  "benefits": ["outcome words users search for"],
  "synonyms": ["alternative names or abbreviations for this device"],
  "modality": "single string: one of laser, light-based, RF, injectable, mechanical, energy-based, biological, combination",
  "family": "single string: specific family e.g. CO2, IPL, fractional, diode, Nd:YAG, RF microneedling, HIFU",
  "excludes": ["what this device is NOT - for negation filtering"],
  "intent_category": "device",
  "classification": "Medical or Non-Medical"
}}"""


def fallback_tags(entity_name: str, entity_type: str) -> dict:
    """
    When the LLM cannot generate meaningful tags (insufficient data),
    fall back to using the entity name itself as the primary tag.
    Ensures the entity remains searchable by its own name.
    """
    return {
        "primary_tags":    [entity_name.lower().strip()],
        "concerns":        [],
        "benefits":        [],
        "synonyms":        [entity_name],
        "modality":        None,
        "family":          None,
        "excludes":        [],
        "intent_category": entity_type,
        "classification":  None,
    }


def call_llm(client: anthropic.Anthropic, prompt: str, entity_name: str, entity_type: str) -> dict:
    """Call Claude Haiku and parse JSON response. Always returns a dict — uses name fallback if needed."""
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=700,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.content[0].text.strip()

        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        tags = json.loads(raw_text)

        if not isinstance(tags, dict) or "primary_tags" not in tags:
            log.warning(f"  ⚠ Unexpected LLM response structure for '{entity_name}' — using name fallback")
            return fallback_tags(entity_name, entity_type)

        if not tags.get("primary_tags"):
            log.warning(f"  ⚠ Empty primary_tags for '{entity_name}' — using name fallback")
            return fallback_tags(entity_name, entity_type)

        return tags

    except json.JSONDecodeError:
        log.warning(f"  ⚠ LLM returned plain text for '{entity_name}' (insufficient source data) — using name fallback")
        return fallback_tags(entity_name, entity_type)
    except anthropic.APIError as e:
        log.error(f"  ✗ Anthropic API error for '{entity_name}': {e}")
        return fallback_tags(entity_name, entity_type)
    except Exception as e:
        log.error(f"  ✗ Unexpected error for '{entity_name}': {e}")
        return fallback_tags(entity_name, entity_type)


# ── Main processing ───────────────────────────────────────────────────────────
def process_treatments(conn, client, dry_run=False, entity_type='all', limit=None, force=False):
    log.info("=" * 60)
    log.info("PROCESSING TREATMENTS")
    log.info("=" * 60)

    rows = fetch_treatments(conn, limit=limit)
    log.info(f"Fetched {len(rows)} approved treatments from tbl_treatments")

    success, skipped, failed = 0, 0, 0

    for i, row in enumerate(rows, 1):
        entity_id   = row["entity_id"]
        entity_name = row.get("name", "").strip()

        log.info(f"\n[{i}/{len(rows)}] Treatment: '{entity_name}' (id: {entity_id})")

        # Skip if already tagged and not forcing
        if not force and already_tagged(conn, entity_id, "treatment"):
            log.info(f"  → SKIPPED (already tagged, use --force to regenerate)")
            skipped += 1
            continue

        # Build raw source snapshot
        raw_source = {
            "name":             row.get("name"),
            "like_wise_terms":  row.get("like_wise_terms"),
            "concern_en":       row.get("concern_en"),
            "benefits_en":      row.get("benefits_en"),
            "technology":       row.get("technology"),
            "classification":   row.get("classification_type"),
        }

        log.info(f"  → Calling LLM for tag generation...")
        prompt = build_treatment_prompt(row)
        tags = call_llm(client, prompt, entity_name, "treatment")

        log.info(f"  ✓ Tags generated:")
        log.info(f"    primary_tags : {tags.get('primary_tags', [])}")
        log.info(f"    concerns     : {tags.get('concerns', [])}")
        log.info(f"    benefits     : {tags.get('benefits', [])}")
        log.info(f"    synonyms     : {tags.get('synonyms', [])}")
        log.info(f"    modality     : {tags.get('modality')}")
        log.info(f"    family       : {tags.get('family')}")
        log.info(f"    excludes     : {tags.get('excludes', [])}")

        save_tags(conn, entity_id, "treatment", entity_name, row.get("swedish", "") or "", tags, raw_source, dry_run=dry_run)

        if not dry_run:
            log.info(f"  ✓ Saved to entity_search_tags")

        success += 1
        time.sleep(BATCH_SLEEP)

    log.info(f"\nTREATMENTS COMPLETE → Success: {success} | Skipped: {skipped} | Failed: {failed}")
    return success, skipped, failed


def process_devices(conn, client, dry_run=False, entity_type='all', limit=None, force=False):
    log.info("=" * 60)
    log.info("PROCESSING DEVICES")
    log.info("=" * 60)

    rows = fetch_devices(conn, limit=limit)
    log.info(f"Fetched {len(rows)} approved devices from tbl_devices")

    success, skipped, failed = 0, 0, 0

    for i, row in enumerate(rows, 1):
        entity_id   = row["entity_id"]
        entity_name = row.get("name", "").strip()

        log.info(f"\n[{i}/{len(rows)}] Device: '{entity_name}' (id: {entity_id})")

        # Skip dummy/fragment rows early — no point calling LLM
        if len(entity_name) <= 4 or entity_name.replace("-", "").replace(" ", "").isdigit():
            log.info(f"  → SKIPPED (looks like test/fragment data: '{entity_name}')")
            skipped += 1
            continue

        if not force and already_tagged(conn, entity_id, "device"):
            log.info(f"  → SKIPPED (already tagged, use --force to regenerate)")
            skipped += 1
            continue

        raw_source = {
            "name":    row.get("name"),
            "swedish": row.get("swedish"),
        }

        log.info(f"  → Calling LLM for tag generation...")
        prompt = build_device_prompt(row)
        tags = call_llm(client, prompt, entity_name, "device")

        log.info(f"  ✓ Tags generated:")
        log.info(f"    primary_tags : {tags.get('primary_tags', [])}")
        log.info(f"    concerns     : {tags.get('concerns', [])}")
        log.info(f"    benefits     : {tags.get('benefits', [])}")
        log.info(f"    synonyms     : {tags.get('synonyms', [])}")
        log.info(f"    modality     : {tags.get('modality')}")
        log.info(f"    family       : {tags.get('family')}")
        log.info(f"    excludes     : {tags.get('excludes', [])}")

        save_tags(conn, entity_id, "device", entity_name, row.get("swedish", "") or "", tags, raw_source, dry_run=dry_run)

        if not dry_run:
            log.info(f"  ✓ Saved to entity_search_tags")

        success += 1
        time.sleep(BATCH_SLEEP)

    log.info(f"\nDEVICES COMPLETE → Success: {success} | Skipped: {skipped} | Failed: {failed}")
    return success, skipped, failed


# ── Entry point ───────────────────────────────────────────────────────────────
def batch_tag_generator_main(
    dry_run: bool = False,
    entity_type: str = "all",
    limit: int = None,
    force: bool = False,
):
    """
    Main entry point for the batch tag generator.
    Called directly by FastAPI or run standalone via __main__.

    Args:
        dry_run:     Generate tags but do not write to DB.
        entity_type: 'treatment' | 'device' | 'all' (default: all).
        limit:       Max rows to process per table (default: all).
        force:       Re-generate tags even if row already has tags.

    Returns:
        dict with total_success, total_skipped, total_failed counts.
    """
    log.info("=" * 60)
    log.info("ZYNQ BATCH TAG GENERATOR")
    log.info(f"Started  : {datetime.now().isoformat()}")
    log.info(f"Dry run  : {dry_run}")
    log.info(f"Entity   : {entity_type}")
    log.info(f"Limit    : {limit or 'all'}")
    log.info(f"Force    : {force}")
    log.info("=" * 60)

    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "your_anthropic_api_key_here":
        log.error("ANTHROPIC_API_KEY not set in .env — aborting")
        raise RuntimeError("ANTHROPIC_API_KEY not configured")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    try:
        conn = get_db_connection()
        log.info(f"Connected to MySQL: {DB_HOST}/{DB_NAME}")
    except mysql.connector.Error as e:
        log.error(f"DB connection failed: {e}")
        raise

    total_success = total_skipped = total_failed = 0

    try:
        if entity_type in ("treatment", "all"):
            s, sk, f = process_treatments(conn, client, dry_run=dry_run, entity_type=entity_type, limit=limit, force=force)
            total_success += s; total_skipped += sk; total_failed += f

        if entity_type in ("device", "all"):
            s, sk, f = process_devices(conn, client, dry_run=dry_run, entity_type=entity_type, limit=limit, force=force)
            total_success += s; total_skipped += sk; total_failed += f

    finally:
        conn.close()

    log.info("\n" + "=" * 60)
    log.info("BATCH COMPLETE")
    log.info(f"Total success : {total_success}")
    log.info(f"Total skipped : {total_skipped}")
    log.info(f"Total failed  : {total_failed}")
    log.info(f"Finished : {datetime.now().isoformat()}")
    log.info("=" * 60)

    return {
        "total_success": total_success,
        "total_skipped": total_skipped,
        "total_failed":  total_failed,
    }


# if __name__ == "__main__":
#     batch_tag_generator_main()
