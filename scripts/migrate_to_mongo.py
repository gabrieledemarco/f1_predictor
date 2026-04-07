"""
scripts/migrate_to_mongo.py
============================
Script ONE-SHOT: migra i dati esistenti su disco verso MongoDB.

Operazioni:
  1. data/cache/jolpica/*.json  → collezione jolpica_cache
  2. data/pinnacle_odds/*.jsonl → collezione odds_records

Esecuzione:
    MONGODB_URI="mongodb+srv://..." python scripts/migrate_to_mongo.py
    python scripts/migrate_to_mongo.py --dry-run   # verifica senza scrivere

Dopo la migrazione rimuovere dal tracking Git:
    git rm --cached -r data/cache/jolpica/
    git commit -m "chore: migrate Jolpica cache to MongoDB"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Jolpica cache migration
# ---------------------------------------------------------------------------

def _parse_jolpica_filename(filename: str):
    """
    Converte filename → (year, round, data_type) o None se non riconoscibile.
    Formati attesi:
      2024_01_results.json    → (2024, 1, "results")
      2024_01_qualifying.json → (2024, 1, "qualifying")
      2024_rounds.json        → (2024, 0, "rounds")
    """
    m = re.match(r"^(\d{4})_(\d{2})_(results|qualifying)\.json$", filename)
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3)
    m2 = re.match(r"^(\d{4})_rounds\.json$", filename)
    if m2:
        return int(m2.group(1)), 0, "rounds"
    return None


def migrate_jolpica_cache(db, cache_dir: Path, dry_run: bool) -> dict:
    """Carica tutti i JSON da cache_dir e li upserta su jolpica_cache."""
    from core.db import jolpica_cache_collection

    json_files = sorted(cache_dir.glob("*.json"))
    if not json_files:
        log.info(f"  Nessun file JSON trovato in {cache_dir}")
        return {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0}

    log.info(f"  Trovati {len(json_files)} file JSON in {cache_dir}")
    coll = jolpica_cache_collection(db) if not dry_run else None

    inserted = updated = skipped = errors = 0

    for path in json_files:
        parsed = _parse_jolpica_filename(path.name)
        if not parsed:
            log.debug(f"  Skip (filename non riconosciuto): {path.name}")
            skipped += 1
            continue

        year, round_num, data_type = parsed

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning(f"  Errore lettura {path.name}: {exc}")
            errors += 1
            continue

        if dry_run:
            log.debug(f"  [DRY-RUN] {path.name} → ({year}, {round_num}, {data_type})")
            inserted += 1
            continue

        result = coll.update_one(
            {"year": year, "round": round_num, "data_type": data_type},
            {"$set": {"payload": payload, "updated_at": datetime.utcnow()}},
            upsert=True,
        )
        if result.upserted_id:
            inserted += 1
        else:
            updated += 1

    return {"inserted": inserted, "updated": updated,
            "skipped": skipped, "errors": errors}


# ---------------------------------------------------------------------------
# Odds records migration
# ---------------------------------------------------------------------------

def migrate_odds_records(db, odds_dir: Path, dry_run: bool) -> dict:
    """Carica tutti i JSONL da odds_dir e li upserta su odds_records."""
    from core.db import odds_records_collection

    jsonl_files = sorted(odds_dir.glob("*.jsonl"))
    if not jsonl_files:
        log.info(f"  Nessun file JSONL trovato in {odds_dir}")
        return {"inserted": 0, "skipped": 0, "errors": 0}

    log.info(f"  Trovati {len(jsonl_files)} file JSONL in {odds_dir}")
    coll = odds_records_collection(db) if not dry_run else None

    inserted = skipped = errors = 0

    for path in jsonl_files:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    log.warning(f"  {path.name}:{line_num} JSON non valido: {exc}")
                    errors += 1
                    continue

                if dry_run:
                    inserted += 1
                    continue

                filter_key = {
                    "race_id":     rec.get("race_id"),
                    "driver_code": rec.get("driver_code"),
                    "market":      rec.get("market"),
                    "timestamp":   rec.get("timestamp"),
                }
                result = coll.update_one(
                    filter_key, {"$setOnInsert": rec}, upsert=True
                )
                if result.upserted_id:
                    inserted += 1
                else:
                    skipped += 1

    return {"inserted": inserted, "skipped": skipped, "errors": errors}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migra dati disco → MongoDB (one-shot)"
    )
    parser.add_argument("--jolpica-cache", default="data/cache/jolpica",
                        help="Directory cache Jolpica (default: data/cache/jolpica)")
    parser.add_argument("--odds-dir", default="data/pinnacle_odds",
                        help="Directory JSONL odds (default: data/pinnacle_odds)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verifica senza scrivere su MongoDB")
    parser.add_argument("--skip-jolpica", action="store_true",
                        help="Salta migrazione jolpica_cache")
    parser.add_argument("--skip-odds", action="store_true",
                        help="Salta migrazione odds_records")
    args = parser.parse_args()

    # ── Connessione ──────────────────────────────────────────────────────
    db = None
    if not args.dry_run:
        try:
            from core.db import get_db_direct
            db = get_db_direct()
            if db is None:
                log.error("MongoDB non raggiungibile. Verifica MONGODB_URI nel .env.")
                sys.exit(1)
            log.info(f"MongoDB connesso: {db.name}")
        except ImportError as e:
            log.error(f"Impossibile importare core.db: {e}")
            sys.exit(1)
    else:
        log.info("[DRY-RUN] Nessun dato verrà scritto su MongoDB")

    results = {}

    # ── Migrazione Jolpica ───────────────────────────────────────────────
    if not args.skip_jolpica:
        jolpica_path = Path(args.jolpica_cache)
        log.info(f"\n[1/2] Migrazione jolpica_cache  ({jolpica_path})")
        if jolpica_path.exists():
            results["jolpica"] = migrate_jolpica_cache(db, jolpica_path, args.dry_run)
        else:
            log.info(f"  Directory non trovata: {jolpica_path} — skip")
            results["jolpica"] = {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0}

    # ── Migrazione Odds ──────────────────────────────────────────────────
    if not args.skip_odds:
        odds_path = Path(args.odds_dir)
        log.info(f"\n[2/2] Migrazione odds_records  ({odds_path})")
        if odds_path.exists():
            results["odds"] = migrate_odds_records(db, odds_path, args.dry_run)
        else:
            log.info(f"  Directory non trovata: {odds_path} — skip")
            results["odds"] = {"inserted": 0, "skipped": 0, "errors": 0}

    # ── Report finale ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Migrazione completata {'(DRY-RUN)' if args.dry_run else ''}")
    print()

    if "jolpica" in results:
        j = results["jolpica"]
        print(f"  jolpica_cache : {j['inserted']:4d} inseriti  "
              f"{j.get('updated', 0):4d} aggiornati  "
              f"{j['skipped']:4d} skippati  "
              f"{j['errors']:4d} errori")

    if "odds" in results:
        o = results["odds"]
        print(f"  odds_records  : {o['inserted']:4d} inseriti  "
              f"{o['skipped']:4d} già presenti  "
              f"{o['errors']:4d} errori")

    if not args.dry_run and db is not None:
        print()
        try:
            from core.db import jolpica_cache_collection, odds_records_collection
            print(f"  Totale jolpica_cache : "
                  f"{jolpica_cache_collection(db).count_documents({})} documenti")
            print(f"  Totale odds_records  : "
                  f"{odds_records_collection(db).count_documents({})} documenti")
        except Exception:
            pass

    print()
    print("  Prossimi passi (dopo verifica):")
    print("    git rm --cached -r data/cache/jolpica/")
    print("    git commit -m 'chore: migrate Jolpica cache to MongoDB'")
    print("=" * 60 + "\n")

    total_errors = sum(
        r.get("errors", 0) for r in results.values()
    )
    if total_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
