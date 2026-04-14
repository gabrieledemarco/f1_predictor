#!/usr/bin/env python3
"""
Import Qualifying Data to MongoDB
=================================
Reads qualifying.csv from TracingInsights RaceData and imports
qualifying session results into MongoDB session_stats_YYYY collections.

Schema:
{
    "_id": "2024_01_VER",
    "year": 2024,
    "round": 1,
    "circuit_ref": "bahrain",
    "driver_code": "VER",
    "position": 1,
    "q1_ms": 92547,
    "q2_ms": 89612,
    "q3_ms": 88291,
    "fastest_lap_ms": 88291,
    "fastest_lap_lap": 21,
    "grid_position": 1,
    "imported_at": "2026-04-13T...",
    "source": "tracinginsights_qualifying"
}

Usage:
    python scripts/import_qualifying.py
    python scripts/import_qualifying.py --min-year 2023 --max-year 2025
    python scripts/import_qualifying.py --force
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError

load_dotenv()

# Mapping from Ergast circuitRef to Jolpica circuit_ref
ERGAST_TO_JOLPICA: dict[str, str] = {
    "villeneuve": "villeneuve",
    "americas": "americas",
    "rodriguez": "rodriguez",
    "interlagos": "interlagos",
    "bahrain": "bahrain",
    "albert_park": "albert_park",
    "suzuka": "suzuka",
    "shanghai": "shanghai",
    "miami": "miami",
    "imola": "imola",
    "monaco": "monaco",
    "catalunya": "catalunya",
    "red_bull_ring": "red_bull_ring",
    "silverstone": "silverstone",
    "hungaroring": "hungaroring",
    "spa": "spa",
    "zandvoort": "zandvoort",
    "monza": "monza",
    "marina_bay": "marina_bay",
    "baku": "baku",
    "losail": "losail",
    "yas_marina": "yas_marina",
    "jeddah": "jeddah",
    "vegas": "vegas",
}


def get_mongo_client():
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is required")
    mongo_db = os.environ.get("MONGO_DB", "betbreaker")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10_000)
    return client[mongo_db]


def find_data_dir(racedata_path: Path) -> Optional[Path]:
    """Locate the directory containing flat CSVs."""
    candidates = [racedata_path / "data", racedata_path]
    for candidate in candidates:
        if (candidate / "qualifying.csv").exists():
            return candidate
        if (candidate / "races.csv").exists():
            return candidate

    for p in racedata_path.rglob("qualifying.csv"):
        return p.parent
    for p in racedata_path.rglob("races.csv"):
        return p.parent

    return None


def load_lookup_tables(data_dir: Path) -> tuple[dict, dict]:
    """Load races and drivers lookup tables."""
    # races.csv
    races_path = data_dir / "races.csv"
    if not races_path.exists():
        raise FileNotFoundError(f"races.csv not found at {races_path}")

    races_df = pd.read_csv(races_path, low_memory=False)
    races_df.columns = [c.strip().lower() for c in races_df.columns]

    # circuits.csv
    circuits_path = data_dir / "circuits.csv"
    circuit_ref_map: dict = {}
    if circuits_path.exists():
        circ_df = pd.read_csv(circuits_path, low_memory=False)
        circ_df.columns = [c.strip().lower() for c in circ_df.columns]
        ref_col = next((c for c in ["circuitref", "circuit_ref"] if c in circ_df.columns), None)
        id_col = next((c for c in ["circuitid", "circuit_id"] if c in circ_df.columns), None)
        if ref_col and id_col:
            for _, row in circ_df.iterrows():
                cid = str(row[id_col]).strip()
                ref = str(row[ref_col]).strip().lower()
                circuit_ref_map[cid] = ERGAST_TO_JOLPICA.get(ref, ref)

    # Build race_info: {raceId -> {year, round, circuit_ref}}
    race_info: dict = {}
    raceid_col = next((c for c in ["raceid", "race_id"] if c in races_df.columns), None)
    year_col = next((c for c in ["year"] if c in races_df.columns), None)
    round_col = next((c for c in ["round"] if c in races_df.columns), None)
    circid_col = next((c for c in ["circuitid", "circuit_id"] if c in races_df.columns), None)

    if not raceid_col:
        raise ValueError(f"raceId column not found in races.csv. Columns: {list(races_df.columns)}")

    for _, row in races_df.iterrows():
        rid = str(row[raceid_col]).strip()
        year = int(row[year_col]) if year_col else 0
        round_num = int(row[round_col]) if round_col else 0
        cid = str(row[circid_col]).strip() if circid_col else ""
        circ_ref = circuit_ref_map.get(cid, cid.lower() if cid else "")
        race_info[rid] = {"year": year, "round": round_num, "circuit_ref": circ_ref}

    # drivers.csv
    drivers_path = data_dir / "drivers.csv"
    driver_code_map: dict = {}
    if drivers_path.exists():
        drv_df = pd.read_csv(drivers_path, low_memory=False)
        drv_df.columns = [c.strip().lower() for c in drv_df.columns]
        did_col = next((c for c in ["driverid", "driver_id"] if c in drv_df.columns), None)
        code_col = next((c for c in ["code", "driverref", "driver_ref", "abbreviation"] if c in drv_df.columns), None)
        if did_col and code_col:
            for _, row in drv_df.iterrows():
                did = str(row[did_col]).strip()
                code = str(row[code_col]).strip().upper()
                if code and code not in ("nan", "NAN", "\\N", "None"):
                    driver_code_map[did] = code

    print(f"  Lookup tables: {len(race_info)} races, {len(driver_code_map)} drivers")
    return race_info, driver_code_map


def parse_time_to_ms(val) -> Optional[float]:
    """Convert time string to milliseconds. Accepts 'M:SS.mmm' or integer ms."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "nat", "none", "\\n"):
        return None
    try:
        parts = s.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60_000 + float(parts[1]) * 1000
        elif len(parts) == 3:
            return int(parts[0]) * 3_600_000 + int(parts[1]) * 60_000 + float(parts[2]) * 1000
        else:
            v = float(s)
            return v if v > 30_000 else v * 1000
    except (ValueError, IndexError):
        return None


def import_qualifying(
    db,
    data_dir: Path,
    race_info: dict,
    driver_code_map: dict,
    min_year: int,
    max_year: int,
    force: bool = False,
) -> int:
    """Import qualifying data from CSV to MongoDB session_stats collections."""

    qualifying_path = data_dir / "qualifying.csv"
    if not qualifying_path.exists():
        print(f"  [WARN] qualifying.csv not found at {qualifying_path}")
        return 0

    df = pd.read_csv(qualifying_path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    # Detect columns
    raceid_col = next((c for c in ["raceid", "race_id"] if c in df.columns), None)
    driverid_col = next((c for c in ["driverid", "driver_id"] if c in df.columns), None)
    position_col = next((c for c in ["position", "grid"] if c in df.columns), None)
    q1_col = next((c for c in ["q1", "q1_time", "q1_time_ms"] if c in df.columns), None)
    q2_col = next((c for c in ["q2", "q2_time", "q2_time_ms"] if c in df.columns), None)
    q3_col = next((c for c in ["q3", "q3_time", "q3_time_ms"] if c in df.columns), None)
    number_col = next((c for c in ["number", "driver_number", "driver_number"] if c in df.columns), None)

    if not raceid_col or not driverid_col:
        print(f"  [ERROR] Required columns not found. Available: {list(df.columns)}")
        return 0

    imported = 0
    ops_by_year: dict[int, list] = {}
    imported_at = datetime.utcnow().isoformat() + "Z"

    for _, row in df.iterrows():
        race_id = str(row[raceid_col]).strip()
        race_data = race_info.get(race_id)
        if not race_data:
            continue

        year = race_data["year"]
        if not (min_year <= year <= max_year):
            continue

        # Skip if already imported (unless force)
        coll_name = f"session_stats_{year}"
        if not force:
            driver_id = str(row[driverid_col]).strip()
            existing = db[coll_name].find_one({
                "year": year,
                "round": race_data["round"],
                "driver_code": driver_code_map.get(driver_id, "")
            })
            if existing:
                continue

        driver_id = str(row[driverid_col]).strip()
        driver_code = driver_code_map.get(driver_id, "")

        if not driver_code:
            continue

        # Parse times
        q1_ms = parse_time_to_ms(row[q1_col]) if q1_col else None
        q2_ms = parse_time_to_ms(row[q2_col]) if q2_col else None
        q3_ms = parse_time_to_ms(row[q3_col]) if q3_col else None

        # Determine fastest lap
        fastest_ms = q3_ms or q2_ms or q1_ms

        # Get position
        position = int(row[position_col]) if position_col else None
        grid_position = position

        # Build document
        doc = {
            "_id": f"{year}_{race_data['round']}_{driver_code}",
            "year": year,
            "round": race_data["round"],
            "circuit_ref": race_data["circuit_ref"],
            "driver_code": driver_code,
            "position": position,
            "grid_position": grid_position,
            "q1_ms": q1_ms,
            "q2_ms": q2_ms,
            "q3_ms": q3_ms,
            "fastest_lap_ms": fastest_ms,
            "session_type": "qualifying",
            "imported_at": imported_at,
            "source": "tracinginsights_qualifying"
        }

        year_key = year
        if year_key not in ops_by_year:
            ops_by_year[year_key] = []
        ops_by_year[year_key].append(UpdateOne(
            {"_id": doc["_id"]},
            {"$set": doc},
            upsert=True
        ))

    # Execute bulk operations per year
    for year, ops in ops_by_year.items():
        coll_name = f"session_stats_{year}"
        if ops:
            result = db[coll_name].bulk_write(ops, ordered=False)
            imported += result.upserted_count
            print(f"  {year}: {result.upserted_count} qualifying records imported to {coll_name}")

    return imported


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Import qualifying data to MongoDB")
    parser.add_argument("--min-year", type=int, default=2019, help="Minimum year")
    parser.add_argument("--max-year", type=int, default=int(datetime.utcnow().year), help="Maximum year")
    parser.add_argument("--force", action="store_true", help="Force re-import")
    parser.add_argument("--racedata-path", type=str, default="data/racedata", help="Path to RaceData directory")
    args = parser.parse_args()

    racedata_path = Path(args.racedata_path)
    if not racedata_path.exists():
        racedata_path = Path(".") / args.racedata_path

    data_dir = find_data_dir(racedata_path)
    if not data_dir:
        print(f"Error: Could not find RaceData directory at {racedata_path}")
        sys.exit(1)

    print(f"Using data directory: {data_dir}")

    db = get_mongo_client()
    print(f"Connected to MongoDB: {db.name}")

    print("Loading lookup tables...")
    race_info, driver_code_map = load_lookup_tables(data_dir)

    print(f"Importing qualifying data ({args.min_year}-{args.max_year})...")
    imported = import_qualifying(
        db, data_dir, race_info, driver_code_map,
        args.min_year, args.max_year, args.force
    )

    print(f"\nTotal qualifying records imported: {imported}")


if __name__ == "__main__":
    main()
