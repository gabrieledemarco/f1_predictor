#!/usr/bin/env python3
"""
Import TracingInsights Lap Data to MongoDB

Parses TracingInsights CSV files and imports lap-by-lap telemetry
into MongoDB f1_lap_times collection.

Usage:
    python scripts/import_tracinginsights.py
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
from pymongo import MongoClient, UpdateOne, InsertOne
from pymongo.errors import PyMongoError

load_dotenv()


def get_mongo_client():
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is required")
    
    mongo_db = os.environ.get("MONGO_DB", "betbreaker")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    return client[mongo_db]


def parse_lap_time(lap_time_str: str) -> Optional[float]:
    """Parse lap time string to milliseconds."""
    if not lap_time_str or lap_time_str == "":
        return None
    try:
        parts = lap_time_str.strip().split(":")
        if len(parts) == 3:
            minutes, seconds, ms = parts
            return int(minutes) * 60000 + float(seconds) * 1000 + float(ms)
        elif len(parts) == 2:
            seconds, ms = parts
            return float(seconds) * 1000 + float(ms)
        return float(lap_time_str)
    except (ValueError, IndexError):
        return None


def get_year_from_path(circuit_path: Path) -> Optional[int]:
    """Extract year from path like data/racedata/2024/Bahrain"""
    parts = circuit_path.parts
    for i, part in enumerate(parts):
        if part in ["racedata", "racedata_tmp"] and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                continue
    return None


def get_round_number(db, year: int, circuit_ref: str) -> int:
    """Get the round number for a circuit in a given year."""
    race = db.f1_races.find_one({"year": year, "circuit_ref": circuit_ref})
    if race:
        return race.get("round", 0)
    return 0


def import_laps_csv(db, csv_path: Path, year: int) -> int:
    """Import laps from a single CSV file."""
    circuit_ref = csv_path.parent.name.lower().replace(" ", "_").replace("-", "_")
    round_num = get_round_number(db, year, circuit_ref)
    
    if round_num == 0:
        print(f"  Warning: No race found for {year} {circuit_ref}, skipping")
        return 0
    
    imported_count = 0
    operations = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            driver_code = row.get("Driver", "").strip()
            if not driver_code:
                continue
            
            lap_number = int(row.get("LapNumber", 0))
            lap_time_raw = row.get("LapTime", "")
            lap_time_ms = parse_lap_time(lap_time_raw)
            
            if lap_time_ms is None:
                continue
            
            fuel_corrected_ms = float(row.get("FuelCorrectedLapTime", lap_time_raw)) if row.get("FuelCorrectedLapTime") else lap_time_ms
            
            compound = row.get("Compound", "UNKNOWN")
            tyre_life = int(row.get("TyreLife", 0))
            is_personal_best = row.get("IsPersonalBest", "False").lower() == "true"
            is_valid = row.get("IsAccurate", "True").lower() == "true" and row.get("TrackStatus", "1") == "1"
            team = row.get("Team", "").strip()
            
            doc = {
                "_id": f"{year}_{circuit_ref}_{driver_code}_{lap_number}",
                "year": year,
                "round": round_num,
                "circuit_ref": circuit_ref,
                "driver_code": driver_code,
                "team": team.lower().replace(" ", "_") if team else "",
                "lap_number": lap_number,
                "lap_time_ms": lap_time_ms,
                "fuel_corrected_ms": fuel_corrected_ms,
                "compound": compound.upper(),
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
    
    db.f1_import_log.update_one(
        {
            "source": "tracinginsights",
            "year": year,
            "circuit_ref": circuit_ref,
            "type": "laps",
        },
        {
            "$set": {
                "imported_at": datetime.utcnow().isoformat() + "Z",
                "laps_count": imported_count,
            }
        },
        upsert=True
    )
    
    return imported_count


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Import TracingInsights lap data to MongoDB")
    year_env = os.environ.get("YEAR", "").strip()
    default_year = int(year_env) if year_env else datetime.now().year
    parser.add_argument("--year", type=int, default=default_year)
    parser.add_argument("--force", action="store_true", default=os.environ.get("FORCE", "false").lower() == "true")
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRACINGINSIGHTS DATA IMPORTER")
    print("=" * 60)
    print(f"Year: {args.year}")
    print()
    
    racedata_path = Path("data/racedata")
    if not racedata_path.exists():
        print("[ERROR] data/racedata directory not found")
        print("Clone TracingInsights repository first:")
        print("  git clone https://github.com/TracingInsights/RaceData.git data/racedata")
        sys.exit(1)
    
    try:
        db = get_mongo_client()
        print("Connected to MongoDB")
        
        db.f1_lap_times.create_index([
            ("year", 1), ("round", 1), ("driver_code", 1)
        ])
        db.f1_lap_times.create_index([("circuit_ref", 1), ("compound", 1)])
        db.f1_lap_times.create_index([("year", 1), ("is_valid", 1)])
        
        total_laps = 0
        
        year_path = racedata_path / str(args.year)
        if not year_path.exists():
            print(f"[ERROR] Year {args.year} not found in racedata")
            sys.exit(1)
        
        circuits = sorted([d for d in year_path.iterdir() if d.is_dir()])
        print(f"\nFound {len(circuits)} circuits for {args.year}")
        
        for circuit_dir in circuits:
            laps_file = circuit_dir / "laps.csv"
            if not laps_file.exists():
                print(f"  {circuit_dir.name}: No laps.csv found, skipping")
                continue
            
            print(f"  Importing {circuit_dir.name}...")
            count = import_laps_csv(db, laps_file, args.year)
            if count > 0:
                print(f"    -> {count} laps imported")
            total_laps += count
            time.sleep(0.1)
        
        print("\n" + "=" * 60)
        print(f"COMPLETED: {total_laps} laps imported")
        print("=" * 60)
        
    except PyMongoError as e:
        print(f"\n[MongoDB Error] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
