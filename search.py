#!/usr/bin/env python3
"""
Zynq Search - Search Engine Script
====================================
Classifies user query via Claude Haiku, fetches candidates from
entity_search_tags joined to tbl_treatments and tbl_devices,
scores and groups results.

Run interactively:
    uv run search.py

Run with a query directly:
    uv run search.py --query "pigmentation laser"

Flags:
    --query     Search query string (if omitted, enters interactive mode)
    --debug     Print extra scoring detail per candidate
    --limit     Max candidates to return per category (default: 5)
"""

import os
import sys
import json
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
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("zynq_search")


# ── DB ─────────────────────────────────────────────────────────────────────────
def get_db_connection():
    return mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4",
    )


# ── Step 1: Query Classification ──────────────────────────────────────────────
def classify_query(client: anthropic.Anthropic, raw_query: str) -> dict:
    log.info("─" * 50)
    log.info(f"STEP 1 — CLASSIFY QUERY: '{raw_query}'")

    prompt = f"""You are a search query classifier for a medical aesthetics platform selling
treatments and devices. Users search in English or Swedish with 1-4 word queries.

User searched: "{raw_query}"

Analyze and return ONLY this JSON. No explanation. No markdown.

{{
  "normalized": "core concept in English, single word or short phrase",
  "intent": "one of: treatment | device | concern | clinic | location | exclusion",
  "concepts": ["2-5 related English concept words to search the DB for"],
  "exclusions": ["terms to EXCLUDE from results — empty array if none"],
  "is_negation": true or false,
  "confidence": "high | medium | low"
}}

Classification rules:
- 'non-laser', 'laser alternative', 'without laser', 'no botox' → intent: exclusion, is_negation: true
- Device brand names (Morpheus8, Fraxel, HydraFacial) → intent: device
- Treatment names (Botox, filler, microneedling) → intent: treatment
- Skin concerns (pigmentation, acne, wrinkles, dark spots) → intent: concern
- City names → intent: location
- Clinic names → intent: clinic

Examples:
- "non-laser"        → intent: exclusion, exclusions: ["laser"], concepts: [], is_negation: true
- "pigment spots"    → intent: concern, concepts: ["pigmentation","melasma","dark spots","sun damage"]
- "morpheus8"        → intent: device, concepts: ["morpheus8","RF microneedling","radiofrequency"]
- "botox"            → intent: treatment, concepts: ["botox","neurotoxin","wrinkle injection","botulinum"]
- "laser behandling" → intent: treatment, concepts: ["laser treatment","laser resurfacing","laser therapy"]
- "håravfall"        → intent: concern, concepts: ["hair loss","hair regrowth","alopecia"]
"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        classification = json.loads(raw)
        log.info(f"  normalized  : {classification.get('normalized')}")
        log.info(f"  intent      : {classification.get('intent')}")
        log.info(f"  concepts    : {classification.get('concepts')}")
        log.info(f"  exclusions  : {classification.get('exclusions')}")
        log.info(f"  is_negation : {classification.get('is_negation')}")
        log.info(f"  confidence  : {classification.get('confidence')}")
        return classification

    except (json.JSONDecodeError, anthropic.APIError, Exception) as e:
        log.error(f"  ✗ Classification failed: {e}")
        # Graceful fallback — treat as generic keyword search
        return {
            "normalized": raw_query.lower(),
            "intent": "treatment",
            "concepts": [raw_query.lower()],
            "exclusions": [],
            "is_negation": False,
            "confidence": "low",
        }


# ── Step 2: Build and Execute DB Query ────────────────────────────────────────
def fetch_candidates(conn, classification: dict, debug: bool = False) -> list:
    log.info("─" * 50)
    log.info("STEP 2 — FETCH CANDIDATES FROM DB")

    concepts   = classification.get("concepts", [])
    exclusions = classification.get("exclusions", [])
    intent     = classification.get("intent", "treatment")

    if not concepts and not classification.get("normalized"):
        log.warning("  No concepts to search — returning empty")
        return []

    # Use normalized term as fallback if concepts empty (negation queries)
    search_terms = concepts if concepts else [classification.get("normalized", "")]

    log.info(f"  Search terms  : {search_terms}")
    log.info(f"  Exclusions    : {exclusions}")
    log.info(f"  Intent        : {intent}")

    cursor = conn.cursor(dictionary=True)

    # ── Build dynamic WHERE clause ─────────────────────────────────────────
    # Concept matching: search across all tag arrays and entity name
    concept_clauses = []
    concept_params  = []

    for term in search_terms:
        t = term.lower()
        concept_clauses.append("""(
            LOWER(t.entity_name)   LIKE %s
            OR JSON_CONTAINS(LOWER(t.primary_tags), JSON_QUOTE(%s))
            OR JSON_CONTAINS(LOWER(t.concerns),     JSON_QUOTE(%s))
            OR JSON_CONTAINS(LOWER(t.synonyms),     JSON_QUOTE(%s))
            OR JSON_CONTAINS(LOWER(t.benefits),     JSON_QUOTE(%s))
            OR LOWER(t.modality)  LIKE %s
            OR LOWER(t.family)    LIKE %s
        )""")
        concept_params.extend([f"%{t}%", t, t, t, t, f"%{t}%", f"%{t}%"])

    # Exclusion filtering — applied BEFORE ranking, not after
    exclusion_clauses = []
    exclusion_params  = []
    for excl in exclusions:
        e = excl.lower()
        exclusion_clauses.append("""(
            NOT JSON_CONTAINS(LOWER(t.primary_tags), JSON_QUOTE(%s))
            AND NOT JSON_CONTAINS(LOWER(t.concerns),  JSON_QUOTE(%s))
            AND NOT JSON_CONTAINS(LOWER(t.synonyms),  JSON_QUOTE(%s))
            AND NOT (LOWER(t.modality) LIKE %s)
            AND NOT (LOWER(t.family)   LIKE %s)
            AND NOT JSON_CONTAINS(LOWER(t.excludes),  JSON_QUOTE(%s))
        )""")
        exclusion_params.extend([e, e, e, f"%{e}%", f"%{e}%", e])

    where_parts = []
    if concept_clauses:
        where_parts.append("(" + " OR ".join(concept_clauses) + ")")
    if exclusion_clauses:
        where_parts.append("(" + " AND ".join(exclusion_clauses) + ")")

    where_sql = " AND ".join(where_parts) if where_parts else "1=1"

    # ── Relevance scoring per candidate ───────────────────────────────────
    # Scores are weighted — exact name match is highest priority
    first_term = search_terms[0].lower() if search_terms else ""

    score_sql = f"""
        (
            /* Exact entity name match — highest weight */
            IF(LOWER(t.entity_name) = %s, 20, 0) +
            /* Partial name match */
            IF(LOWER(t.entity_name) LIKE %s, 10, 0) +
            /* Primary tag match */
            IF(JSON_CONTAINS(LOWER(t.primary_tags), JSON_QUOTE(%s)), 8, 0) +
            /* Concern match */
            IF(JSON_CONTAINS(LOWER(t.concerns), JSON_QUOTE(%s)), 7, 0) +
            /* Synonym match */
            IF(JSON_CONTAINS(LOWER(t.synonyms), JSON_QUOTE(%s)), 6, 0) +
            /* Benefit match */
            IF(JSON_CONTAINS(LOWER(t.benefits), JSON_QUOTE(%s)), 5, 0) +
            /* Modality match */
            IF(LOWER(t.modality) LIKE %s, 4, 0) +
            /* Family match */
            IF(LOWER(t.family) LIKE %s, 4, 0)
        ) AS relevance_score
    """
    score_params = [
        first_term, f"%{first_term}%",
        first_term, first_term, first_term, first_term,
        f"%{first_term}%", f"%{first_term}%",
    ]

    # ── Treatments query ───────────────────────────────────────────────────
    treatment_sql = f"""
        SELECT
            t.entity_id,
            t.entity_name        AS name,
            t.swedish_name       AS swedish_name,
            'treatment'          AS entity_type,
            tr.description_en    AS description,
            tr.concern_en        AS concern,
            tr.benefits_en       AS benefits,
            tr.classification_type AS classification,
            t.modality,
            t.family,
            t.primary_tags,
            t.concerns           AS tag_concerns,
            t.synonyms,
            t.excludes,
            {score_sql}
        FROM entity_search_tags t
        INNER JOIN tbl_treatments tr
            ON tr.treatment_id = t.entity_id
            AND tr.is_deleted = 0
            AND tr.approval_status = 'APPROVED'
        WHERE t.entity_type = 'treatment'
          AND {where_sql}
    """

    # ── Devices query ──────────────────────────────────────────────────────
    device_sql = f"""
        SELECT
            t.entity_id,
            t.entity_name        AS name,
            t.swedish_name       AS swedish_name,
            'device'             AS entity_type,
            NULL                 AS description,
            NULL                 AS concern,
            NULL                 AS benefits,
            t.classification     AS classification,
            t.modality,
            t.family,
            t.primary_tags,
            t.concerns           AS tag_concerns,
            t.synonyms,
            t.excludes,
            {score_sql}
        FROM entity_search_tags t
        INNER JOIN tbl_devices d
            ON d.device_id = t.entity_id
            AND d.is_deleted = 0
            AND d.approval_status = 'APPROVED'
        WHERE t.entity_type = 'device'
          AND {where_sql}
    """

    union_sql = f"""
        SELECT * FROM (
            {treatment_sql}
            UNION ALL
            {device_sql}
        ) combined
        ORDER BY relevance_score DESC
        LIMIT {MAX_RESULTS}
    """

    # Assemble all params: score + where for treatments, score + where for devices
    all_params = (
        score_params + concept_params + exclusion_params +  # treatments
        score_params + concept_params + exclusion_params    # devices
    )

    log.debug(f"  Executing UNION query with {len(search_terms)} search term(s) "
              f"and {len(exclusions)} exclusion(s)")

    try:
        cursor.execute(union_sql, all_params)
        rows = cursor.fetchall()
        cursor.close()
        log.info(f"  DB returned {len(rows)} raw candidates before score filtering")
        return rows
    except mysql.connector.Error as e:
        log.error(f"  ✗ DB query failed: {e}")
        cursor.close()
        return []



# ── Lexical fallback query ────────────────────────────────────────────────────
def lexical_fallback_query(conn, raw_query: str) -> list:
    """
    Called when semantic search returns zero results.
    Queries entity_search_tags directly matching:
      - primary_tags contains the raw query term
      - OR synonyms contains the raw query term
      - OR entity_name LIKE the raw query term
    No LLM involved — pure lexical match on stored tag data.
    """
    log.info("─" * 50)
    log.info(f"LEXICAL FALLBACK — querying primary_tags / synonyms / name for: '{raw_query}'")

    term = raw_query.lower().strip()
    cursor = conn.cursor(dictionary=True)

    sql = """
        SELECT
            t.entity_id,
            t.entity_name        AS name,
            t.swedish_name       AS swedish_name,
            tr.description_en    AS description,
            tr.concern_en        AS concern,
            t.modality,
            t.family,
            t.primary_tags,
            t.concerns           AS tag_concerns,
            t.synonyms,
            t.excludes,
            'treatment'          AS entity_type,
            10                   AS relevance_score
        FROM entity_search_tags t
        INNER JOIN tbl_treatments tr
            ON tr.treatment_id = t.entity_id
            AND tr.is_deleted = 0
            AND tr.approval_status = 'APPROVED'
        WHERE t.entity_type = 'treatment'
          AND (
            JSON_CONTAINS(LOWER(t.primary_tags), JSON_QUOTE(%s))
            OR JSON_CONTAINS(LOWER(t.synonyms),  JSON_QUOTE(%s))
            OR LOWER(t.entity_name) LIKE %s
          )

        UNION ALL

        SELECT
            t.entity_id,
            t.entity_name        AS name,
            t.swedish_name       AS swedish_name,
            NULL                 AS description,
            NULL                 AS concern,
            t.modality,
            t.family,
            t.primary_tags,
            t.concerns           AS tag_concerns,
            t.synonyms,
            t.excludes,
            'device'             AS entity_type,
            10                   AS relevance_score
        FROM entity_search_tags t
        INNER JOIN tbl_devices d
            ON d.device_id = t.entity_id
            AND d.is_deleted = 0
            AND d.approval_status = 'APPROVED'
        WHERE t.entity_type = 'device'
          AND (
            JSON_CONTAINS(LOWER(t.primary_tags), JSON_QUOTE(%s))
            OR JSON_CONTAINS(LOWER(t.synonyms),  JSON_QUOTE(%s))
            OR LOWER(t.entity_name) LIKE %s
          )

        LIMIT 20
    """

    like_term = f"%{term}%"
    params = [term, term, like_term, term, term, like_term]

    try:
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        cursor.close()
        log.info(f"  Lexical fallback returned {len(rows)} result(s)")
        for r in rows:
            log.info(f"    • {r.get('entity_type'):<10} | {r.get('name')}")
        return rows
    except mysql.connector.Error as e:
        log.error(f"  ✗ Lexical fallback DB query failed: {e}")
        cursor.close()
        return []


# ── Step 3: Filter zero-score and log intermediate results ────────────────────
def filter_and_log_candidates(candidates: list, debug: bool = False) -> list:
    log.info("─" * 50)
    log.info("STEP 3 — FILTER & SCORE CANDIDATES")
    log.info(f"  Total before filter : {len(candidates)}")

    filtered = [c for c in candidates if (c.get("relevance_score") or 0) > 0]
    log.info(f"  After zero-score filter : {len(filtered)}")

    if debug:
        log.debug("  INTERMEDIATE CANDIDATES (all, sorted by score):")
        for i, c in enumerate(filtered, 1):
            log.debug(
                f"    [{i:02d}] score={c.get('relevance_score'):>4} | "
                f"type={c.get('entity_type'):<10} | "
                f"modality={str(c.get('modality')):<15} | "
                f"family={str(c.get('family')):<15} | "
                f"name={c.get('name')}"
            )
    else:
        for i, c in enumerate(filtered[:10], 1):
            log.info(
                f"  [{i:02d}] score={c.get('relevance_score'):>4} | "
                f"{c.get('entity_type'):<10} | {c.get('name')}"
            )
        if len(filtered) > 10:
            log.info(f"  ... and {len(filtered)-10} more (run with --debug to see all)")

    return filtered


# ── Step 4: Group by intent ────────────────────────────────────────────────────
def group_results(candidates: list, intent: str, limit: int = 5) -> dict:
    log.info("─" * 50)
    log.info(f"STEP 4 — GROUP RESULTS by intent: '{intent}'")

    treatments = [c for c in candidates if c.get("entity_type") == "treatment"]
    devices    = [c for c in candidates if c.get("entity_type") == "device"]

    log.info(f"  Treatments in pool : {len(treatments)}")
    log.info(f"  Devices in pool    : {len(devices)}")

    # Intent-based grouping order
    if intent == "concern":
        grouped = {
            "treatments":        treatments[:limit],
            "devices":           devices[:limit],
        }
    elif intent == "device":
        grouped = {
            "devices":           devices[:limit],
            "related_treatments": treatments[:limit],
        }
    elif intent == "treatment":
        grouped = {
            "treatments":        treatments[:limit],
            "devices":           devices[:limit],
        }
    elif intent == "exclusion":
        # User wants alternatives — show what they CAN use
        grouped = {
            "alternatives":          treatments[:limit],
            "alternative_devices":   devices[:limit],
        }
    else:
        # Fallback grouping
        grouped = {
            "treatments":        treatments[:limit],
            "devices":           devices[:limit],
        }

    for group_name, items in grouped.items():
        log.info(f"  Group '{group_name}' — {len(items)} result(s):")
        for item in items:
            log.info(f"    • [{item.get('relevance_score')}] {item.get('name')} ({item.get('modality')})")

    return grouped


# ── Step 5: Format output ─────────────────────────────────────────────────────
def format_output(raw_query: str, classification: dict, grouped: dict) -> dict:
    """Clean up result for API or console output."""
    def clean_item(item):
        return {
            "id":           item.get("entity_id"),
            "name":         item.get("name"),
            "swedish_name": item.get("swedish_name"),
            "type":         item.get("entity_type"),
            "modality":     item.get("modality"),
            "family":       item.get("family"),
            "concern":      item.get("concern"),
            "description":  (item.get("description") or "")[:200] if item.get("description") else None,
            "score":        item.get("relevance_score"),
        }

    output = {
        "query":           raw_query,
        "interpreted_as": {
            "normalized":  classification.get("normalized"),
            "intent":      classification.get("intent"),
            "concepts":    classification.get("concepts"),
            "exclusions":  classification.get("exclusions"),
            "is_negation": classification.get("is_negation"),
            "confidence":  classification.get("confidence"),
        },
        "results": {
            group: [clean_item(i) for i in items]
            for group, items in grouped.items()
        },
    }
    return output


# ── Main search pipeline ──────────────────────────────────────────────────────
def search(client, conn, raw_query: str, debug: bool = False, limit: int = 5) -> dict:
    log.info("\n" + "=" * 60)
    log.info(f"SEARCH QUERY: '{raw_query}'")
    log.info(f"Timestamp: {datetime.now().isoformat()}")
    log.info("=" * 60)

    # Step 1 — Classify
    classification = classify_query(client, raw_query)

    # Early exit for pure negation with no concepts
    if classification.get("is_negation") and not classification.get("concepts"):
        log.info("  Pure negation query with no positive concepts — fetching all non-excluded")

    # Step 2 — Fetch from DB
    candidates = fetch_candidates(conn, classification, debug=debug)

    # Step 3 — Filter and log
    filtered = filter_and_log_candidates(candidates, debug=debug)

    # Lexical fallback — if semantic search returned nothing, query by raw term directly
    if not filtered:
        log.info("  Semantic search returned zero results — activating lexical fallback")
        filtered = lexical_fallback_query(conn, raw_query)
        if filtered:
            log.info(f"  Lexical fallback found {len(filtered)} result(s) — using these")
        else:
            log.info("  Lexical fallback also returned zero results — response will be empty")

    # Step 4 — Group
    grouped = group_results(filtered, classification.get("intent", "treatment"), limit=limit)

    # Step 5 — Format
    output = format_output(raw_query, classification, grouped)

    log.info("─" * 50)
    log.info("SEARCH COMPLETE")
    log.info(f"Total results returned: {sum(len(v) for v in grouped.values())}")

    return output



# ── Final output sorter ───────────────────────────────────────────────────────
def sort_final_output(output: dict) -> dict:
    """
    Normalises and re-orders the FINAL OUTPUT so that:
      1. All treatment-side groups are merged into a single "treatments" key — always first
      2. All device-side groups are merged into a single "devices" key — always second
         and sorted internally by modality then family
      3. Any other groups follow after
    Internal item ranking within each merged group is preserved (sorted by score desc).
    Group key names are always "treatments" and "devices" regardless of intent.
    """
    results = output.get("results", {})

    TREATMENT_KEYS = {"treatments", "alternatives", "related_treatments"}
    DEVICE_KEYS    = {"devices", "alternative_devices"}

    # Collect all treatment-side items into one flat list
    all_treatments = []
    for k, v in results.items():
        if k in TREATMENT_KEYS:
            all_treatments.extend(v)

    # Collect all device-side items into one flat list
    all_devices = []
    for k, v in results.items():
        if k in DEVICE_KEYS:
            all_devices.extend(v)

    # Other groups (future: clinics, experts etc.)
    other_groups = {k: v for k, v in results.items() if k not in TREATMENT_KEYS | DEVICE_KEYS}

    # Re-sort treatments by score descending (preserves ranking after merge)
    all_treatments = sorted(all_treatments, key=lambda x: x.get("score", 0), reverse=True)

    # Sort devices by modality then family (device_type), score as tiebreaker
    def device_sort_key(item):
        modality = (item.get("modality") or "zzz").lower()
        family   = (item.get("family")   or "zzz").lower()
        score    = -(item.get("score", 0))  # negative so higher score wins on tie
        return (modality, family, score)

    all_devices = sorted(all_devices, key=device_sort_key)

    log.info(f"  [sort_final_output] treatments ({len(all_treatments)} items):")
    for item in all_treatments:
        log.info(f"    • [{item.get('score')}] {item.get('name')}")
    log.info(f"  [sort_final_output] devices ({len(all_devices)} items) sorted by modality/family:")
    for item in all_devices:
        log.info(f"    • {item.get('name')} | modality={item.get('modality')} | family={item.get('family')}")

    # Always output as "treatments" first, "devices" second
    reordered_results = {}
    if all_treatments:
        reordered_results["treatments"] = all_treatments
    if all_devices:
        reordered_results["devices"] = all_devices
    reordered_results.update(other_groups)

    log.info(f"  [sort_final_output] Final group order: {list(reordered_results.keys())}")

    output["results"] = reordered_results
    return output


# ── Entry point ───────────────────────────────────────────────────────────────
def search_main(
    query: str = None,
    debug: bool = False,
    limit: int = 10,
):
    """
    Main entry point for the search engine.
    Called directly by FastAPI or run standalone via __main__.

    Args:
        query: Search query string. If None, runs in interactive console mode.
        debug: Log all candidate scores for debugging.
        limit: Max results per category in the final output (default: 5).

    Returns:
        dict with query, interpreted_as, and results (treatments + devices).
    Raises:
        RuntimeError: if ANTHROPIC_API_KEY not configured.
    """
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

    try:
        if query:
            # Single query mode — used by FastAPI
            result = search(client, conn, query, debug=debug, limit=limit)
            result = sort_final_output(result)
            print("\n" + "=" * 60)
            print("FINAL OUTPUT:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return result

        else:
            # Interactive console mode
            print("\nZynq Search — Interactive Mode")
            print("Type a query and press Enter. Type 'quit' to exit.\n")
            while True:
                try:
                    raw_query = input("Search: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting.")
                    break
                if not raw_query:
                    continue
                if raw_query.lower() in ("quit", "exit", "q"):
                    print("Exiting.")
                    break
                result = search(client, conn, raw_query, debug=debug, limit=limit)
                result = sort_final_output(result)
                print("\n" + "=" * 60)
                print("FINAL OUTPUT:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                print()

    finally:
        conn.close()


# if __name__ == "__main__":
#     import sys
#     _query = None
#     _debug = "--debug" in sys.argv
#     _limit = 5
#     for i, arg in enumerate(sys.argv[1:], 1):
#         if arg == "--query" and i + 1 < len(sys.argv):
#             _query = sys.argv[i + 1]
#         if arg == "--limit" and i + 1 < len(sys.argv):
#             _limit = int(sys.argv[i + 1])
#     search_main(query=_query, debug=_debug, limit=_limit)
