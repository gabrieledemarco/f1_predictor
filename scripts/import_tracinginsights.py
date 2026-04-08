#!/usr/bin/env python3
"""
Import TracingInsights Lap Data to MongoDB

Parses TracingInsights CSV files (Ergast format) and imports lap-by-lap data
into MongoDB f1_lap_times collection.

Usage:
    python scripts/import_tracinginsights.py [--year YEAR] [--force]
"""

import os
import sys
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError

load_dotenv()


def get_mongo_client():
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is required")
    
    mongo_db = os.environ.get("MONGO_DB", "betbreaker")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    return client[mongo_db]


def load_race_mapping(races_csv_path: Path, target_year: int) -> Dict[int, dict]:
    """Load races.csv and return mapping from raceId to {year, round, circuitId}."""
    mapping = {}
    with open(races_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = int(row.get("year", 0))
            if year != target_year:
                continue
            race_id = int(row.get("raceId", 0))
            mapping[race_id] = {
                "year": year,
                "round": int(row.get("round", 0)),
                "circuit_id": int(row.get("circuitId", 0)),
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


def get_circuit_ref(db, circuit_id: int) -> str:
    """Get circuit_ref from MongoDB f1_races by circuitId."""
    race = db.f1_races.find_one({"circuit_id": circuit_id})
    if race:
        return race.get("circuit_ref", f"circuit_{circuit_id}")
    return f"circuit_{circuit_id}"


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
            
            circuit_ref = get_circuit_ref(db, race_info["circuit_id"])
            year = race_info["year"]
            round_num = race_info["round"]
            
            doc_id = f"{year}_R{round_num}_{circuit_ref}_{driver_code}_L{lap_number}"
            
            doc = {
                "_id": doc_id,
                "year": year,
                "round": round_num,
                "circuit_ref": circuit_ref,
                "driver_code": driver_code,
                "lap_number": lap_number,
                "position": position,
                "lap_time_ms": lap_time_ms,
                "imported_at": datetime.utcnow().isoformat() + "Z",
                "source": "tracinginsights",
            }
            
            operations.append(UpdateOne(
                {"_id": doc["_id"]},
                {"$set": doc},
                upsert=True
            ))
            
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
        
        print("\nLoading race mapping...")
        race_mapping = load_race_mapping(races_csv, args.year)
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
        
    except PyMongoError as e:
        print(f"\n[MongoDB Error] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
