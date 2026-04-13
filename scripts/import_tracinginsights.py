#!/usr/bin/env python3
"""
Import TracingInsights Lap Data to MongoDB
==========================================
Parses TracingInsights CSV files and imports lap-by-lap telemetry
into MongoDB f1_lap_times collection.

TracingInsights RaceData repo layout (clona con):
    git clone https://github.com/TracingInsights/RaceData.git data/racedata

Struttura directory effettiva:
    data/racedata/
        data/              ← subdirectory aggiuntiva nel repo
            2024/
                Bahrain/
                    laps.csv
                Saudi_Arabia/
                    laps.csv
                ...
            2023/
                ...

Usage:
    python scripts/import_tracinginsights.py
    python scripts/import_tracinginsights.py --min-year 2022 --max-year 2025
    python scripts/import_tracinginsights.py --year 2024 --force
"""

import os
import sys
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError

load_dotenv()

# ---------------------------------------------------------------------------
# Circuit name mapping: folder TracingInsights → circuit_ref Jolpica/Ergast
# Necessario perché la repo usa nomi inglesi dei GP, non i codici circuito.
# ---------------------------------------------------------------------------
FOLDER_TO_CIRCUIT_REF: dict[str, str] = {
    "Bahrain":          "bahrain",
    "Saudi_Arabia":     "jeddah",
    "Australia":        "albert_park",
    "Japan":            "suzuka",
    "China":            "shanghai",
    "Miami":            "miami",
    "Emilia_Romagna":   "imola",
    "Monaco":           "monaco",
    "Canada":           "villeneuve",
    "Spain":            "catalunya",
    "Austria":          "red_bull_ring",
    "Great_Britain":    "silverstone",
    "Hungary":          "hungaroring",
    "Belgium":          "spa",
    "Netherlands":      "zandvoort",
    "Italy":            "monza",
    "Azerbaijan":       "baku",
    "Singapore":        "marina_bay",
    "United_States":    "americas",
    "Mexico":           "rodriguez",
    "Brazil":           "interlagos",
    "Las_Vegas":        "vegas",
    "Qatar":            "losail",
    "Abu_Dhabi":        "yas_marina",
    # Alias comuni
    "Saudi Arabia":     "jeddah",
    "Great Britain":    "silverstone",
    "United States":    "americas",
    "Las Vegas":        "vegas",
    "Abu Dhabi":        "yas_marina",
    "Emilia Romagna":   "imola",
}


def get_mongo_client():
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is required")
    mongo_db = os.environ.get("MONGO_DB", "betbreaker")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10_000)
    return client[mongo_db]


def parse_lap_time(lap_time_str: str) -> Optional[float]:
    """Parse lap time string to milliseconds. Supports m:ss.mmm, ss.mmm, float."""
    if not lap_time_str or lap_time_str in ("", "nan", "NaT", "None"):
        return None
    try:
        s = str(lap_time_str).strip()
        parts = s.split(":")
        if len(parts) == 3:
            # H:MM:SS.mmm — unlikely but safe
            return int(parts[0]) * 3600000 + int(parts[1]) * 60000 + float(parts[2]) * 1000
        elif len(parts) == 2:
            # M:SS.mmm
            return int(parts[0]) * 60000 + float(parts[1]) * 1000
        else:
            val = float(s)
            # If stored as seconds (< 500), convert; if ms, use directly
            return val * 1000 if val < 500 else val
    except (ValueError, IndexError):
        return None


def resolve_racedata_root(racedata_path: Path) -> Path:
    """
    Trova la root effettiva del repo TracingInsights.
    La repo ha una subdirectory 'data/' che contiene le cartelle per anno.
    Ritorna il path alla directory che contiene direttamente le cartelle {year}/.
    """
    if not racedata_path.exists():
        return racedata_path

    # Controlla se ci sono directory numeriche (anni) direttamente qui
    year_dirs = [d for d in racedata_path.iterdir()
                 if d.is_dir() and d.name.isdigit() and 2018 <= int(d.name) <= 2030]
    if year_dirs:
        return racedata_path

    # Cerca una subdirectory unica che contenga le directory anni
    sub_dirs = [d for d in racedata_path.iterdir() if d.is_dir()]
    for sub in sub_dirs:
        year_dirs = [d for d in sub.iterdir()
                     if d.is_dir() and d.name.isdigit() and 2018 <= int(d.name) <= 2030]
        if year_dirs:
            print(f"  [INFO] TracingInsights data root rilevata: {sub}")
            return sub

    return racedata_path


def folder_to_circuit_ref(folder_name: str, db=None, year: int = None) -> Optional[str]:
    """
    Converte il nome cartella TracingInsights in circuit_ref Jolpica.
    1. Prova il mapping statico FOLDER_TO_CIRCUIT_REF
    2. Fallback: cerca in f1_races per nome simile
    """
    # Mapping diretto
    if folder_name in FOLDER_TO_CIRCUIT_REF:
        return FOLDER_TO_CIRCUIT_REF[folder_name]

    # Prova varianti: underscore→space, tutto minuscolo
    normalized = folder_name.replace("_", " ").strip()
    if normalized in FOLDER_TO_CIRCUIT_REF:
        return FOLDER_TO_CIRCUIT_REF[normalized]

    # Fallback: search MongoDB f1_races
    if db is not None and year is not None:
        slug = folder_name.lower().replace(" ", "_").replace("-", "_")
        race = db.f1_races.find_one({"year": year, "circuit_ref": slug})
        if race:
            return slug
        # Partial match by circuit name
        race = db.f1_races.find_one(
            {"year": year, "circuit_ref": {"$regex": slug[:5], "$options": "i"}}
        )
        if race:
            return race.get("circuit_ref")

    return None


def get_round_number(db, year: int, circuit_ref: str) -> int:
    """Get the round number for a circuit in a given year."""
    race = db.f1_races.find_one({"year": year, "circuit_ref": circuit_ref})
    if race:
        return race.get("round", 0)
    return 0


def import_laps_csv(db, csv_path: Path, year: int, force: bool = False) -> int:
    """Import laps from a single CSV file. Returns number of laps imported."""
    folder_name = csv_path.parent.name
    circuit_ref = folder_to_circuit_ref(folder_name, db=db, year=year)

    if not circuit_ref:
        print(f"    [SKIP] No circuit_ref mapping for folder '{folder_name}'")
        return 0

    round_num = get_round_number(db, year, circuit_ref)
    if round_num == 0:
        print(f"    [SKIP] No race found for {year} circuit_ref='{circuit_ref}' (folder: {folder_name})")
        return 0

    # Check se già importato (skip unless force)
    if not force:
        existing = db.f1_import_log.find_one({
            "source": "tracinginsights", "year": year, "circuit_ref": circuit_ref, "type": "laps"
        })
        if existing and existing.get("laps_count", 0) > 0:
            print(f"    [SKIP] {year} {circuit_ref} già importato ({existing['laps_count']} laps) — usa --force per reimportare")
            return 0

    imported_count = 0
    skipped = 0
    operations = []

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"    [ERROR] Impossibile leggere {csv_path}: {e}")
        return 0

    # Normalizza nomi colonne (strip + title case variants)
    col_map = {c.strip().lower(): c for c in df.columns}

    def get_col(variants: list[str], default=None):
        for v in variants:
            if v.lower() in col_map:
                return col_map[v.lower()]
        return default

    col_driver      = get_col(["driver", "driverabbreviation", "driver_code"])
    col_lap         = get_col(["lapnumber", "lap_number", "lap"])
    col_laptime     = get_col(["laptime", "lap_time", "time"])
    col_fuel        = get_col(["fuelcorrectedlaptime", "fuel_corrected"])
    col_compound    = get_col(["compound", "tyre", "tire"])
    col_tyrelife    = get_col(["tyrelife", "tyre_life", "tirelife"])
    col_pb          = get_col(["ispersonalbest", "is_personal_best", "personalbest"])
    col_accurate    = get_col(["isaccurate", "is_accurate", "accurate"])
    col_trackstatus = get_col(["trackstatus", "track_status"])
    col_team        = get_col(["team", "constructor", "constructorname"])

    if not col_driver or not col_laptime:
        print(f"    [ERROR] Colonne Driver o LapTime non trovate in {csv_path.name}")
        print(f"    Colonne disponibili: {list(df.columns)}")
        return 0

    for _, row in df.iterrows():
        driver_code = str(row.get(col_driver, "")).strip()
        if not driver_code or driver_code == "nan":
            skipped += 1
            continue

        lap_number = int(row.get(col_lap, 0) or 0)
        lap_time_ms = parse_lap_time(str(row.get(col_laptime, "")))
        if lap_time_ms is None or lap_time_ms <= 0 or lap_time_ms > 300_000:  # >5min = outlier
            skipped += 1
            continue

        # Fuel corrected (opzionale)
        fuel_raw = str(row.get(col_fuel, "")) if col_fuel else ""
        fuel_corrected_ms = parse_lap_time(fuel_raw) or lap_time_ms

        # Compound normalizzato
        compound_raw = str(row.get(col_compound, "UNKNOWN")).strip().upper() if col_compound else "UNKNOWN"
        # Map varianti
        compound_map = {
            "SOFT": "SOFT", "S": "SOFT", "C5": "SOFT", "C4": "SOFT",
            "MEDIUM": "MEDIUM", "M": "MEDIUM", "C3": "MEDIUM",
            "HARD": "HARD", "H": "HARD", "C2": "HARD", "C1": "HARD",
            "INTERMEDIATE": "INTERMEDIATE", "INTER": "INTERMEDIATE", "I": "INTERMEDIATE",
            "WET": "WET", "W": "WET", "FULL_WET": "WET",
        }
        compound = compound_map.get(compound_raw, "UNKNOWN")

        tyre_life = int(row.get(col_tyrelife, 0) or 0) if col_tyrelife else 0

        # is_personal_best
        pb_val = str(row.get(col_pb, "False")).strip().lower() if col_pb else "false"
        is_personal_best = pb_val in ("true", "1", "yes")

        # is_valid: basato su IsAccurate; TrackStatus è opzionale
        if col_accurate:
            acc_val = str(row.get(col_accurate, "True")).strip().lower()
            is_accurate = acc_val in ("true", "1", "yes")
        else:
            is_accurate = True

        if col_trackstatus:
            ts_val = str(row.get(col_trackstatus, "1")).strip()
            # TrackStatus "1" = green flag; altri = safety car/red flag/vsc
            # Lo usiamo come flag AGGIUNTIVO ma non blocchiamo l'import
            track_ok = ts_val == "1"
        else:
            track_ok = True

        # is_valid: IsAccurate è il flag principale; TrackStatus filtra SC/VSC laps
        is_valid = is_accurate and track_ok

        team_raw = str(row.get(col_team, "")).strip() if col_team else ""
        team = team_raw.lower().replace(" ", "_").replace("-", "_") if team_raw else ""

        doc = {
            "_id": f"{year}_{circuit_ref}_{driver_code}_{lap_number}",
            "year": year,
            "round": round_num,
            "circuit_ref": circuit_ref,
            "driver_code": driver_code,
            "team": team,
            "lap_number": lap_number,
            "lap_time_ms": lap_time_ms,
            "fuel_corrected_ms": fuel_corrected_ms,
            "compound": compound,
            "tyre_life": tyre_life,
            "is_personal_best": is_personal_best,
            "is_valid": is_valid,
            "imported_at": datetime.utcnow().isoformat() + "Z",
            "source": "tracinginsights",
        }

        operations.append(UpdateOne(
            {"_id": doc["_id"]},
            {"$set": doc},
            upsert=True
        ))

        if len(operations) >= 1000:
            db.f1_lap_times.bulk_write(operations, ordered=False)
            imported_count += len(operations)
            operations = []

    if operations:
        db.f1_lap_times.bulk_write(operations, ordered=False)
        imported_count += len(operations)

    # Log importazione
    db.f1_import_log.update_one(
        {"source": "tracinginsights", "year": year, "circuit_ref": circuit_ref, "type": "laps"},
        {"$set": {
            "imported_at": datetime.utcnow().isoformat() + "Z",
            "laps_count": imported_count,
            "skipped_count": skipped,
        }},
        upsert=True,
    )

    return imported_count


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Import TracingInsights lap data to MongoDB")
    parser.add_argument("--year",     type=int, help="Anno singolo da importare")
    parser.add_argument("--min-year", type=int, default=int(os.environ.get("MIN_YEAR", "2019")),
                        help="Anno minimo (default: 2019)")
    parser.add_argument("--max-year", type=int, default=int(os.environ.get("MAX_YEAR", str(datetime.now().year))),
                        help="Anno massimo (default: anno corrente)")
    parser.add_argument("--force", action="store_true",
                        default=os.environ.get("FORCE", "false").lower() == "true",
                        help="Reimporta anche se già importato")
    args = parser.parse_args()

    # Se --year specificato, sovrascrive il range
    if args.year:
        args.min_year = args.year
        args.max_year = args.year
    # Compatibilità con env var YEAR (vecchio comportamento)
    elif os.environ.get("YEAR"):
        try:
            y = int(os.environ["YEAR"])
            args.min_year = y
            args.max_year = y
        except ValueError:
            pass

    print("=" * 65)
    print("TRACINGINSIGHTS DATA IMPORTER")
    print("=" * 65)
    print(f"Anni: {args.min_year} – {args.max_year}  |  force={args.force}")
    print()

    racedata_path = Path("data/racedata")
    if not racedata_path.exists():
        print("[ERROR] data/racedata directory not found")
        print("Clone TracingInsights repository first:")
        print("  git clone https://github.com/TracingInsights/RaceData.git data/racedata")
        sys.exit(1)

    # Auto-detect root (potrebbe avere una sub 'data/')
    data_root = resolve_racedata_root(racedata_path)

    # Anni disponibili nel repo
    available_years = sorted([
        int(d.name) for d in data_root.iterdir()
        if d.is_dir() and d.name.isdigit() and 2018 <= int(d.name) <= 2030
    ])
    if not available_years:
        print(f"[ERROR] Nessuna directory anno trovata in {data_root}")
        print(f"Contenuto: {[d.name for d in data_root.iterdir() if d.is_dir()]}")
        sys.exit(1)

    print(f"Anni disponibili nel repo: {available_years}")
    years_to_import = [y for y in available_years if args.min_year <= y <= args.max_year]
    if not years_to_import:
        print(f"[WARNING] Nessun anno nel range {args.min_year}–{args.max_year} trovato nel repo")
        print(f"  Range disponibile: {available_years[0]}–{available_years[-1]}")
        sys.exit(0)

    print(f"Anni da importare: {years_to_import}")
    print()

    try:
        db = get_mongo_client()
        print("Connesso a MongoDB:", db.name)

        # Crea indici
        db.f1_lap_times.create_index([("year", 1), ("round", 1), ("driver_code", 1)])
        db.f1_lap_times.create_index([("circuit_ref", 1), ("compound", 1)])
        db.f1_lap_times.create_index([("year", 1), ("is_valid", 1)])
        print("Indici creati/verificati")
        print()

        grand_total = 0

        for year in years_to_import:
            year_path = data_root / str(year)
            circuits = sorted([d for d in year_path.iterdir() if d.is_dir()])
            print(f"── {year}: {len(circuits)} cartelle circuito")

            year_total = 0
            for circuit_dir in circuits:
                laps_file = circuit_dir / "laps.csv"
                if not laps_file.exists():
                    # Cerca anche varianti del nome
                    alternatives = list(circuit_dir.glob("*lap*.csv"))
                    if alternatives:
                        laps_file = alternatives[0]
                    else:
                        continue

                count = import_laps_csv(db, laps_file, year, force=args.force)
                if count > 0:
                    print(f"    {circuit_dir.name}: {count} laps importati")
                year_total += count
                time.sleep(0.05)

            print(f"  → Anno {year}: {year_total} laps totali")
            grand_total += year_total

        print()
        print("=" * 65)
        print(f"COMPLETATO: {grand_total} laps importati su {len(years_to_import)} anni")
        print("=" * 65)

    except PyMongoError as e:
        print(f"\n[MongoDB Error] {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n[Error] {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
