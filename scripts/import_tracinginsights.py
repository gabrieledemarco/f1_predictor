#!/usr/bin/env python3
"""
Import TracingInsights Lap Data to MongoDB
==========================================
Reads the TracingInsights/RaceData flat Ergast-style CSVs and imports
lap-by-lap data into MongoDB f1_lap_times collection.

<<<<<<< HEAD
TracingInsights/RaceData actual repo layout:
    data/racedata/
        data/
            lap_times.csv    ← all laps, all years (raceId, driverId, lap, ...)
            races.csv        ← raceId → year, round, circuitId
            drivers.csv      ← driverId → driverRef, code
            circuits.csv     ← circuitId → circuitRef

Usage:
    python scripts/import_tracinginsights.py
    python scripts/import_tracinginsights.py --min-year 2022 --max-year 2025
    python scripts/import_tracinginsights.py --force
=======
Parses TracingInsights CSV files (Ergast format) and imports lap-by-lap data
into MongoDB f1_lap_times collection.

Usage:
    python scripts/import_tracinginsights.py [--year YEAR] [--force]
>>>>>>> origin/feat/data-unification
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError

load_dotenv()

# ---------------------------------------------------------------------------
# Ergast circuitRef → Jolpica circuit_ref overrides
# TracingInsights uses Ergast circuitRef which mostly matches Jolpica,
# but a few circuits have different refs.
# ---------------------------------------------------------------------------
ERGAST_TO_JOLPICA: dict[str, str] = {
    # Ergast ref          : Jolpica circuit_ref
    "villeneuve":          "villeneuve",
    "americas":            "americas",
    "rodriguez":           "rodriguez",
    "interlagos":          "interlagos",
    "bahrain":             "bahrain",
    "albert_park":         "albert_park",
    "suzuka":              "suzuka",
    "shanghai":            "shanghai",
    "miami":               "miami",
    "imola":               "imola",
    "monaco":              "monaco",
    "catalunya":           "catalunya",
    "red_bull_ring":       "red_bull_ring",
    "silverstone":         "silverstone",
    "hungaroring":         "hungaroring",
    "spa":                 "spa",
    "zandvoort":           "zandvoort",
    "monza":               "monza",
    "marina_bay":          "marina_bay",
    "baku":                "baku",
    "losail":              "losail",
    "yas_marina":          "yas_marina",
    "jeddah":              "jeddah",
    "vegas":               "vegas",
}


def get_mongo_client():
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is required")
    mongo_db = os.environ.get("MONGO_DB", "betbreaker")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10_000)
    return client[mongo_db]


<<<<<<< HEAD
def find_data_dir(racedata_path: Path) -> Optional[Path]:
    """
    Locate the directory containing the flat CSVs (lap_times.csv, races.csv, etc.)
    within the cloned TracingInsights/RaceData repo.
    """
    # Common locations to check
    candidates = [
        racedata_path / "data",
        racedata_path,
    ]
    for candidate in candidates:
        if (candidate / "lap_times.csv").exists():
            return candidate
        if (candidate / "races.csv").exists():
            return candidate

    # Recursive fallback
    for p in racedata_path.rglob("lap_times.csv"):
        return p.parent
    for p in racedata_path.rglob("races.csv"):
        return p.parent

    return None


def load_lookup_tables(data_dir: Path) -> tuple[dict, dict, dict]:
    """
    Load races, drivers, circuits lookup tables.

    Returns:
        race_info   : {raceId: {"year": int, "round": int, "circuit_ref": str}}
        driver_code : {driverId: str}  — 3-letter driver code
        circuit_ref : {circuitId: str} — circuit_ref
    """
    # ── races.csv ───────────────────────────────────────────────────────────
    races_path = data_dir / "races.csv"
    if not races_path.exists():
        raise FileNotFoundError(f"races.csv not found at {races_path}")

    races_df = pd.read_csv(races_path, low_memory=False)
    races_df.columns = [c.strip().lower() for c in races_df.columns]

    # ── circuits.csv ────────────────────────────────────────────────────────
    circuits_path = data_dir / "circuits.csv"
    circuit_ref_map: dict = {}
    if circuits_path.exists():
        circ_df = pd.read_csv(circuits_path, low_memory=False)
        circ_df.columns = [c.strip().lower() for c in circ_df.columns]
        ref_col = next((c for c in ["circuitref", "circuit_ref"] if c in circ_df.columns), None)
        id_col  = next((c for c in ["circuitid", "circuit_id"]  if c in circ_df.columns), None)
        if ref_col and id_col:
            for _, row in circ_df.iterrows():
                cid = str(row[id_col]).strip()
                ref = str(row[ref_col]).strip().lower()
                # Apply Ergast→Jolpica override if present, else keep as-is
                circuit_ref_map[cid] = ERGAST_TO_JOLPICA.get(ref, ref)

    # Build race_info: {raceId → {year, round, circuit_ref}}
    race_info: dict = {}
    raceid_col  = next((c for c in ["raceid", "race_id"]   if c in races_df.columns), None)
    year_col    = next((c for c in ["year"]               if c in races_df.columns), None)
    round_col   = next((c for c in ["round"]              if c in races_df.columns), None)
    circid_col  = next((c for c in ["circuitid", "circuit_id"] if c in races_df.columns), None)

    if not raceid_col:
        raise ValueError(f"raceId column not found in races.csv. Columns: {list(races_df.columns)}")

    for _, row in races_df.iterrows():
        rid = str(row[raceid_col]).strip()
        year  = int(row[year_col])  if year_col  else 0
        round_num = int(row[round_col]) if round_col else 0
        cid   = str(row[circid_col]).strip() if circid_col else ""
        circ_ref = circuit_ref_map.get(cid, cid.lower() if cid else "")
        race_info[rid] = {"year": year, "round": round_num, "circuit_ref": circ_ref}

    # ── drivers.csv ─────────────────────────────────────────────────────────
    drivers_path = data_dir / "drivers.csv"
    driver_code_map: dict = {}
    if drivers_path.exists():
        drv_df = pd.read_csv(drivers_path, low_memory=False)
        drv_df.columns = [c.strip().lower() for c in drv_df.columns]
        did_col  = next((c for c in ["driverid", "driver_id"] if c in drv_df.columns), None)
        code_col = next((c for c in ["code", "driverref", "driver_ref", "abbreviation"] if c in drv_df.columns), None)
        if did_col and code_col:
            for _, row in drv_df.iterrows():
                did  = str(row[did_col]).strip()
                code = str(row[code_col]).strip().upper()
                if code and code != "NAN":
                    driver_code_map[did] = code

    print(f"  Lookup tables loaded: {len(race_info)} races, {len(driver_code_map)} drivers, {len(circuit_ref_map)} circuits")
    return race_info, driver_code_map, circuit_ref_map


def parse_lap_time_ms(val) -> Optional[float]:
    """
    Convert lap time to milliseconds.
    Accepts: integer ms (e.g. 82456), 'M:SS.mmm', 'SS.mmm', float seconds.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s or s in ("nan", "NaT", "None", "\\N"):
        return None
    try:
        parts = s.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60_000 + float(parts[1]) * 1000
        elif len(parts) == 3:
            return int(parts[0]) * 3_600_000 + int(parts[1]) * 60_000 + float(parts[2]) * 1000
        else:
            v = float(s)
            # If already in milliseconds (> 30_000 ms = 30 s), use directly
            if v > 30_000:
                return v
            # Probably in seconds
            return v * 1000
    except (ValueError, IndexError):
        return None


def import_flat_lap_times(
    db,
    data_dir: Path,
    race_info: dict,
    driver_code_map: dict,
    min_year: int,
    max_year: int,
    force: bool = False,
) -> int:
    """
    Import from flat lap_times.csv. Returns total laps imported.
    """
    lap_path = data_dir / "lap_times.csv"
    if not lap_path.exists():
        print(f"[ERROR] lap_times.csv not found at {lap_path}")
        return 0

    print(f"Reading {lap_path} ...")
    df = pd.read_csv(lap_path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"  {len(df):,} rows loaded")

    # Column detection
    col_map = {c: c for c in df.columns}

    def get_col(*variants):
        for v in variants:
            if v.lower() in col_map:
                return v.lower()
        return None

    raceid_col  = get_col("raceid", "race_id")
    driverid_col = get_col("driverid", "driver_id")
    lap_col     = get_col("lap", "lapnumber", "lap_number")
    pos_col     = get_col("position")
    time_col    = get_col("time", "laptime", "lap_time")
    ms_col      = get_col("milliseconds", "lap_time_ms", "ms")

    if not raceid_col:
        print(f"[ERROR] raceId column not found. Columns: {list(df.columns)}")
        return 0
    if not (time_col or ms_col):
        print(f"[ERROR] No lap time column found. Columns: {list(df.columns)}")
        return 0

    # Filter to year range using race_info
    valid_race_ids = {
        rid for rid, info in race_info.items()
        if min_year <= info["year"] <= max_year
    }
    df[raceid_col] = df[raceid_col].astype(str).str.strip()
    df = df[df[raceid_col].isin(valid_race_ids)].copy()
    print(f"  {len(df):,} rows after year filter ({min_year}–{max_year})")

    if df.empty:
        print(f"[WARNING] No lap data found for years {min_year}–{max_year}")
        print(f"  Race IDs in range: {len(valid_race_ids)}")
        if valid_race_ids:
            sample = list(valid_race_ids)[:5]
            race_ids_in_csv = set(df[raceid_col].unique()) if not df.empty else set()
            print(f"  Sample expected race IDs: {sample}")
        return 0

    # Check already imported (skip whole year if not force)
    if not force:
        imported_years = set()
        for log in db.f1_import_log.find(
            {"source": "tracinginsights", "type": "lap_times_flat", "laps_count": {"$gt": 0}},
            {"year": 1}
        ):
            imported_years.add(log.get("year"))

        years_needed = set(range(min_year, max_year + 1)) - imported_years
        if not years_needed:
            print(f"[SKIP] All years {min_year}–{max_year} already imported — usa --force per reimportare")
            return 0
        if len(years_needed) < (max_year - min_year + 1):
            skipped_years = sorted(set(range(min_year, max_year + 1)) - years_needed)
            print(f"[INFO] Skipping already-imported years: {skipped_years}")
            valid_race_ids = {
                rid for rid, info in race_info.items()
                if info["year"] in years_needed
            }
            df = df[df[raceid_col].isin(valid_race_ids)].copy()
            print(f"  {len(df):,} rows after skip-imported filter")

    if driverid_col:
        df[driverid_col] = df[driverid_col].astype(str).str.strip()

    total_imported = 0
    total_skipped = 0
    operations = []

    # Process per race for progress reporting
    grouped = df.groupby(raceid_col)
    races_processed = 0

    for race_id, race_df in grouped:
        info = race_info.get(str(race_id))
        if not info:
            total_skipped += len(race_df)
            continue

        year = info["year"]
        round_num = info["round"]
        circuit_ref = info["circuit_ref"]
        race_imported = 0
        race_skipped = 0

        for _, row in race_df.iterrows():
            # Driver code
            if driverid_col:
                did = str(row[driverid_col]).strip()
                driver_code = driver_code_map.get(did, did[:3].upper() if len(did) >= 3 else did)
            else:
                driver_code = "UNK"

            if not driver_code or driver_code in ("NAN", ""):
                race_skipped += 1
                continue

            # Lap number
            lap_number = 0
            if lap_col:
                try:
                    lap_number = int(float(row[lap_col]))
                except (ValueError, TypeError):
                    pass

            # Lap time in ms
            lap_time_ms = None
            if ms_col:
                try:
                    v = float(row[ms_col])
                    if not pd.isna(v):
                        lap_time_ms = v
                except (ValueError, TypeError):
                    pass
            if lap_time_ms is None and time_col:
                lap_time_ms = parse_lap_time_ms(row[time_col])

            if lap_time_ms is None or lap_time_ms <= 0 or lap_time_ms > 300_000:
                race_skipped += 1
                continue

=======
def load_circuit_mapping(circuits_csv_path: Path) -> Dict[int, str]:
    """Load circuits.csv and return mapping from circuitId to circuitRef."""
    mapping = {}
    with open(circuits_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            circuit_id = int(row.get("circuitId", 0))
            circuit_ref = row.get("circuitRef", "").strip().lower()
            if circuit_ref and circuit_ref != "\\N":
                mapping[circuit_id] = circuit_ref
    return mapping


def load_race_mapping(races_csv_path: Path, target_year: int, circuit_mapping: Dict[int, str]) -> Dict[int, dict]:
    """Load races.csv and return mapping from raceId to {year, round, circuit_ref}."""
    mapping = {}
    with open(races_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = int(row.get("year", 0))
            if year != target_year:
                continue
            race_id = int(row.get("raceId", 0))
            circuit_id = int(row.get("circuitId", 0))
            circuit_ref = circuit_mapping.get(circuit_id, f"circuit_{circuit_id}")
            mapping[race_id] = {
                "year": year,
                "round": int(row.get("round", 0)),
                "circuit_ref": circuit_ref,
                "name": row.get("name", ""),
            }
    return mapping


def load_driver_mapping(drivers_csv_path: Path) -> Dict[int, str]:
    """Load drivers.csv and return mapping from driverId to driver code."""
    mapping = {}
    with open(drivers_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            driver_id = int(row.get("driverId", 0))
            code = row.get("code", "").strip()
            if code and code != "\\N":
                mapping[driver_id] = code
            else:
                driver_ref = row.get("driverRef", "").strip()
                if driver_ref:
                    mapping[driver_id] = driver_ref[:3].upper()
    return mapping


def import_laps_csv(db, laps_csv_path: Path, race_mapping: Dict[int, dict], 
                    driver_mapping: Dict[int, str], target_year: int, force: bool) -> int:
    """Import laps from lap_times.csv file."""
    imported_count = 0
    skipped_count = 0
    operations = []
    
    with open(laps_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            race_id = int(row.get("raceId", 0))
            
            if race_id not in race_mapping:
                skipped_count += 1
                continue
            
            race_info = race_mapping[race_id]
            if race_info["year"] != target_year:
                continue
            
            driver_id = int(row.get("driverId", 0))
            driver_code = driver_mapping.get(driver_id, f"D{driver_id}")
            
            lap_number = int(row.get("lap", 0))
            position = int(row.get("position", 0)) if row.get("position") else 0
            
            milliseconds_raw = row.get("milliseconds", "")
            lap_time_ms = None
            if milliseconds_raw and milliseconds_raw != "\\N":
                try:
                    lap_time_ms = float(milliseconds_raw)
                except ValueError:
                    pass
            
            if lap_time_ms is None:
                skipped_count += 1
                continue
            
            circuit_ref = race_info["circuit_ref"]
            year = race_info["year"]
            round_num = race_info["round"]
            
            doc_id = f"{year}_R{round_num}_{circuit_ref}_{driver_code}_L{lap_number}"
            
>>>>>>> origin/feat/data-unification
            doc = {
                "_id": doc_id,
                "year": year,
                "round": round_num,
                "circuit_ref": circuit_ref,
                "driver_code": driver_code,
<<<<<<< HEAD
                "team": "",        # lap_times.csv doesn't include team — joined from results later
=======
>>>>>>> origin/feat/data-unification
                "lap_number": lap_number,
                "position": position,
                "lap_time_ms": lap_time_ms,
<<<<<<< HEAD
                "fuel_corrected_ms": lap_time_ms,  # no fuel correction in flat CSV
                "compound": "UNKNOWN",   # lap_times.csv doesn't include tyre data
                "tyre_life": 0,
                "is_personal_best": False,
                "is_valid": True,
=======
>>>>>>> origin/feat/data-unification
                "imported_at": datetime.utcnow().isoformat() + "Z",
                "source": "tracinginsights",
            }

            operations.append(UpdateOne(
                {"_id": doc["_id"]},
                {"$set": doc},
                upsert=True
            ))
<<<<<<< HEAD
            race_imported += 1

            if len(operations) >= 2000:
                db.f1_lap_times.bulk_write(operations, ordered=False)
                total_imported += len(operations)
                operations = []

        total_skipped += race_skipped
        races_processed += 1

        if races_processed % 10 == 0 or race_imported > 0:
            print(f"    {year} R{round_num:02d} {circuit_ref}: {race_imported} laps")

        # Log per race
        db.f1_import_log.update_one(
            {"source": "tracinginsights", "year": year, "circuit_ref": circuit_ref, "type": "lap_times_flat"},
            {"$set": {
                "imported_at": datetime.utcnow().isoformat() + "Z",
                "laps_count": race_imported,
                "skipped_count": race_skipped,
            }},
            upsert=True,
        )

    if operations:
        db.f1_lap_times.bulk_write(operations, ordered=False)
        total_imported += len(operations)

    print(f"\n  Total: {total_imported:,} laps imported, {total_skipped:,} skipped")
    return total_imported


def enrich_teams_from_results(db, data_dir: Path, race_info: dict, driver_code_map: dict) -> int:
    """
    Enrich f1_lap_times.team field using results.csv (raceId, driverId, constructorId).
    """
    results_path = data_dir / "results.csv"
    constructors_path = data_dir / "constructors.csv"
    if not results_path.exists():
        print("[INFO] results.csv not found — skipping team enrichment")
        return 0

    print("Enriching team info from results.csv ...")
    res_df = pd.read_csv(results_path, low_memory=False)
    res_df.columns = [c.strip().lower() for c in res_df.columns]

    # Load constructors
    constructor_map = {}
    if constructors_path.exists():
        con_df = pd.read_csv(constructors_path, low_memory=False)
        con_df.columns = [c.strip().lower() for c in con_df.columns]
        cid_col = next((c for c in ["constructorid", "constructor_id"] if c in con_df.columns), None)
        ref_col = next((c for c in ["constructorref", "constructor_ref"] if c in con_df.columns), None)
        if cid_col and ref_col:
            for _, row in con_df.iterrows():
                constructor_map[str(row[cid_col]).strip()] = str(row[ref_col]).strip().lower()

    # Build driver→team per race
    raceid_col   = next((c for c in ["raceid",   "race_id"]        if c in res_df.columns), None)
    driverid_col = next((c for c in ["driverid", "driver_id"]      if c in res_df.columns), None)
    conid_col    = next((c for c in ["constructorid", "constructor_id"] if c in res_df.columns), None)

    if not (raceid_col and driverid_col):
        print("[INFO] Required columns missing in results.csv — skipping team enrichment")
        return 0

    # {(race_id, driver_code) → constructor_ref}
    team_lookup = {}
    for _, row in res_df.iterrows():
        rid = str(row[raceid_col]).strip()
        did = str(row[driverid_col]).strip()
        driver_code = driver_code_map.get(did, "")
        if not driver_code:
            continue
        if conid_col:
            cid = str(row[conid_col]).strip()
            team = constructor_map.get(cid, cid.lower())
        else:
            team = ""
        info = race_info.get(rid)
        if info and team:
            team_lookup[(info["circuit_ref"], driver_code)] = team

    if not team_lookup:
        print("[INFO] No team data found in results.csv")
        return 0

    # Bulk update f1_lap_times with team info
    updated = 0
    ops = []
    for (circuit_ref, driver_code), team in team_lookup.items():
        ops.append(UpdateOne(
            {"circuit_ref": circuit_ref, "driver_code": driver_code, "team": ""},
            {"$set": {"team": team}},
        ))
        if len(ops) >= 2000:
            result = db.f1_lap_times.bulk_write(ops, ordered=False)
            updated += result.modified_count
            ops = []
    if ops:
        result = db.f1_lap_times.bulk_write(ops, ordered=False)
        updated += result.modified_count

    print(f"  Team enrichment: {updated:,} lap records updated")
    return updated
=======
            
            if len(operations) >= 1000:
                if operations:
                    result = db.f1_lap_times.bulk_write(operations, ordered=False)
                    imported_count += result.upserted_count + result.modified_count
                operations = []
        
        if operations:
            result = db.f1_lap_times.bulk_write(operations, ordered=False)
            imported_count += result.upserted_count + result.modified_count
    
    if skipped_count > 0:
        print(f"    (skipped {skipped_count} rows without valid lap times)")
    
    return imported_count
>>>>>>> origin/feat/data-unification


def main():
    import argparse
<<<<<<< HEAD
    parser = argparse.ArgumentParser(description="Import TracingInsights flat lap data to MongoDB")
    parser.add_argument("--year",     type=int, help="Anno singolo (override min/max)")
    parser.add_argument("--min-year", type=int, default=int(os.environ.get("MIN_YEAR", "2019")),
                        help="Anno minimo (default: 2019)")
    parser.add_argument("--max-year", type=int, default=int(os.environ.get("MAX_YEAR", str(datetime.now().year))),
                        help="Anno massimo (default: anno corrente)")
    parser.add_argument("--force", action="store_true",
                        default=os.environ.get("FORCE", "false").lower() == "true",
                        help="Reimporta anche se già importato")
    parser.add_argument("--skip-enrich", action="store_true",
                        help="Salta l'arricchimento team da results.csv")
=======
    parser = argparse.ArgumentParser(description="Import TracingInsights lap data to MongoDB")
    year_env = os.environ.get("YEAR", "").strip()
    default_year = int(year_env) if year_env else datetime.now().year
    parser.add_argument("--year", type=int, default=default_year)
    parser.add_argument("--force", action="store_true", default=os.environ.get("FORCE", "false").lower() == "true")
>>>>>>> origin/feat/data-unification
    args = parser.parse_args()

    if args.year:
        args.min_year = args.year
        args.max_year = args.year

    print("=" * 65)
    print("TRACINGINSIGHTS DATA IMPORTER (flat CSV mode)")
    print("=" * 65)
    print(f"Anni: {args.min_year} – {args.max_year}  |  force={args.force}")
    print()

    racedata_path = Path("data/racedata")
    if not racedata_path.exists():
        print("[ERROR] data/racedata directory not found")
        print("Clone TracingInsights repository first:")
        print("  git clone https://github.com/TracingInsights/RaceData.git data/racedata")
        sys.exit(1)
<<<<<<< HEAD

    # Find where the flat CSVs live
    data_dir = find_data_dir(racedata_path)
    if data_dir is None:
        print("[ERROR] Could not locate lap_times.csv or races.csv in the repo")
        print(f"Contenuto di {racedata_path}:")
        for item in sorted(racedata_path.rglob("*.csv"))[:20]:
            print(f"  {item.relative_to(racedata_path)}")
        sys.exit(1)

    print(f"Data directory: {data_dir}")
    print(f"CSV files found:")
    for f in sorted(data_dir.glob("*.csv"))[:15]:
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name:<40} {size_kb:>7} KB")
    print()

    try:
        race_info, driver_code_map, circuit_ref_map = load_lookup_tables(data_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    try:
        db = get_mongo_client()
        print(f"\nConnesso a MongoDB: {db.name}")

        # Create indexes
        db.f1_lap_times.create_index([("year", 1), ("round", 1), ("driver_code", 1)])
        db.f1_lap_times.create_index([("circuit_ref", 1), ("compound", 1)])
        db.f1_lap_times.create_index([("year", 1), ("is_valid", 1)])

        print()
        total = import_flat_lap_times(
            db, data_dir, race_info, driver_code_map,
            args.min_year, args.max_year, args.force
        )

        if total > 0 and not args.skip_enrich:
            print()
            enrich_teams_from_results(db, data_dir, race_info, driver_code_map)

        count = db.f1_lap_times.count_documents({})
        print()
        print("=" * 65)
        print(f"COMPLETATO: {total:,} laps importati in questa run")
        print(f"f1_lap_times collection totale: {count:,} documenti")
        print("=" * 65)

=======
    
    data_path = racedata_path / "data"
    if not data_path.exists():
        print(f"[ERROR] {data_path} directory not found")
        sys.exit(1)
    
    laps_csv = data_path / "lap_times.csv"
    races_csv = data_path / "races.csv"
    drivers_csv = data_path / "drivers.csv"
    
    if not laps_csv.exists():
        print(f"[ERROR] {laps_csv} not found")
        sys.exit(1)
    if not races_csv.exists():
        print(f"[ERROR] {races_csv} not found")
        sys.exit(1)
    if not drivers_csv.exists():
        print(f"[ERROR] {drivers_csv} not found")
        sys.exit(1)
    
    try:
        db = get_mongo_client()
        print("Connected to MongoDB")
        
        db.f1_lap_times.create_index([
            ("year", 1), ("round", 1), ("driver_code", 1)
        ])
        db.f1_lap_times.create_index([("year", 1), ("lap_number", 1)])
        db.f1_lap_times.create_index([("circuit_ref", 1)])
        
        print("Loading circuit mapping...")
        circuits_csv = data_path / "circuits.csv"
        if not circuits_csv.exists():
            print(f"[WARNING] {circuits_csv} not found - circuit_ref will use circuit ID")
            circuit_mapping = {}
        else:
            circuit_mapping = load_circuit_mapping(circuits_csv)
            print(f"  Found {len(circuit_mapping)} circuits")

        print("\nLoading race mapping...")
        race_mapping = load_race_mapping(races_csv, args.year, circuit_mapping)
        print(f"  Found {len(race_mapping)} races for {args.year}")
        
        if not race_mapping:
            print(f"[ERROR] No races found for year {args.year}")
            sys.exit(1)
        
        print("Loading driver mapping...")
        driver_mapping = load_driver_mapping(drivers_csv)
        print(f"  Found {len(driver_mapping)} drivers")
        
        print(f"\nImporting lap times from {args.year}...")
        total_laps = import_laps_csv(db, laps_csv, race_mapping, driver_mapping, args.year, args.force)
        
        print("\n" + "=" * 60)
        print(f"COMPLETED: {total_laps} laps imported")
        print("=" * 60)
        
        db.f1_import_log.update_one(
            {
                "source": "tracinginsights",
                "year": args.year,
                "type": "laps",
            },
            {
                "$set": {
                    "imported_at": datetime.utcnow().isoformat() + "Z",
                    "laps_count": total_laps,
                }
            },
            upsert=True
        )
        
>>>>>>> origin/feat/data-unification
    except PyMongoError as e:
        print(f"\n[MongoDB Error] {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n[Error] {e}")
<<<<<<< HEAD
=======
        import traceback
>>>>>>> origin/feat/data-unification
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
