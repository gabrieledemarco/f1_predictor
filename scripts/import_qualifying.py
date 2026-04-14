#!/usr/bin/env python3
"""
Import TracingInsights Qualifying Data to MongoDB

Parses qualifying.csv from RaceData and imports into MongoDB f1_qualifying collection.
Includes Q1, Q2, Q3 times and positions.

Usage:
    python scripts/import_qualifying.py [--years YEARS] [--force]
"""

import os
import sys
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError

load_dotenv()


def parse_time_to_milliseconds(time_str: str) -> Optional[int]:
    """Convert '1:23.456' or '1:23.456789' to milliseconds."""
    if not time_str or time_str == "\\N":
        return None
    try:
        if ":" in time_str:
            parts = time_str.split(":")
            minutes = int(parts[0])
            seconds = float(parts[1])
            return int((minutes * 60 + seconds) * 1000)
        else:
            return int(float(time_str) * 1000)
    except (ValueError, IndexError):
        return None


def get_mongo_client():
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is required")
    
    mongo_db = os.environ.get("MONGO_DB", "betbreaker")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    return client[mongo_db]


def load_mappings(data_path: Path, target_years: List[int]) -> tuple:
    """Load all necessary mappings from CSV files."""
    
    # Circuit mapping
    circuit_mapping = {}
    circuits_csv = data_path / "circuits.csv"
    if circuits_csv.exists():
        with open(circuits_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                circuit_id = int(row.get("circuitId", 0))
                circuit_ref = row.get("circuitRef", "").strip().lower()
                if circuit_ref and circuit_ref != "\\N":
                    circuit_mapping[circuit_id] = circuit_ref
    
    # Race mapping: raceId -> {year, round, circuit_ref, raceId}
    race_mapping = {}
    races_csv = data_path / "races.csv"
    with open(races_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = int(row.get("year", 0))
            if year not in target_years:
                continue
            race_id = int(row.get("raceId", 0))
            circuit_id = int(row.get("circuitId", 0))
            circuit_ref = circuit_mapping.get(circuit_id, f"circuit_{circuit_id}")
            race_mapping[race_id] = {
                "year": year,
                "round": int(row.get("round", 0)),
                "circuit_ref": circuit_ref,
                "raceId": race_id,
            }
    
    # Driver mapping: driverId -> driver_code
    driver_mapping = {}
    drivers_csv = data_path / "drivers.csv"
    with open(drivers_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            driver_id = int(row.get("driverId", 0))
            code = row.get("code", "").strip()
            if code and code != "\\N":
                driver_mapping[driver_id] = code
            else:
                driver_ref = row.get("driverRef", "").strip()
                if driver_ref:
                    driver_mapping[driver_id] = driver_ref[:3].upper()
    
    # Constructor mapping: constructorId -> constructor_ref
    constructor_mapping = {}
    constructors_csv = data_path / "constructors.csv"
    if constructors_csv.exists():
        with open(constructors_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                constructor_id = int(row.get("constructorId", 0))
                constructor_ref = row.get("constructorRef", "").strip().lower()
                if constructor_ref and constructor_ref != "\\N":
                    constructor_mapping[constructor_id] = constructor_ref
    
    return race_mapping, driver_mapping, constructor_mapping


def import_qualifying(db, qualifying_csv: Path, race_mapping: Dict, 
                      driver_mapping: Dict, constructor_mapping: Dict,
                      target_years: List[int], force: bool) -> int:
    """Import qualifying data from CSV."""
    imported_count = 0
    operations = []
    
    with open(qualifying_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            race_id = int(row.get("raceId", 0))
            
            if race_id not in race_mapping:
                continue
            
            race_info = race_mapping[race_id]
            if race_info["year"] not in target_years:
                continue
            
            driver_id = int(row.get("driverId", 0))
            constructor_id = int(row.get("constructorId", 0))
            
            driver_code = driver_mapping.get(driver_id, f"D{driver_id}")
            constructor_ref = constructor_mapping.get(constructor_id, f"c{constructor_id}")
            
            position = int(row.get("position", 0)) if row.get("position") else None
            
            # Parse Q times
            q1_ms = parse_time_to_milliseconds(row.get("q1", ""))
            q2_ms = parse_time_to_milliseconds(row.get("q2", ""))
            q3_ms = parse_time_to_milliseconds(row.get("q3", ""))
            
            # Calculate total qualifying time (sum of completed sessions)
            total_q_ms = None
            if q1_ms:
                total_q_ms = q1_ms
                if q2_ms:
                    total_q_ms += q2_ms
                    if q3_ms:
                        total_q_ms += q3_ms
            
            doc_id = f"{race_info['year']}_{race_info['round']:02d}_{driver_code}_qual"
            
            doc = {
                "_id": doc_id,
                "year": race_info["year"],
                "round": race_info["round"],
                "circuit_ref": race_info["circuit_ref"],
                "race_id": race_id,
                "driver_code": driver_code,
                "driver_id": driver_id,
                "constructor_ref": constructor_ref,
                "constructor_id": constructor_id,
                "position": position,
                "q1_ms": q1_ms,
                "q2_ms": q2_ms,
                "q3_ms": q3_ms,
                "total_q_ms": total_q_ms,
                "imported_at": datetime.utcnow().isoformat() + "Z",
                "source": "tracinginsights",
            }
            
            operations.append(UpdateOne(
                {"_id": doc["_id"]},
                {"$set": doc},
                upsert=True
            ))
            
            if len(operations) >= 500:
                result = db.f1_qualifying.bulk_write(operations, ordered=False)
                imported_count += result.upserted_count + result.modified_count
                operations = []
    
    if operations:
        result = db.f1_qualifying.bulk_write(operations, ordered=False)
        imported_count += result.upserted_count + result.modified_count
    
    return imported_count


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Import qualifying data to MongoDB")
    years_env = os.environ.get("YEARS", "").strip()
    parser.add_argument("--years", type=str, default=years_env, 
                        help="Comma-separated years (e.g., '2023,2024')")
    parser.add_argument("--force", action="store_true", 
                        default=os.environ.get("FORCE", "false").lower() == "true")
    args = parser.parse_args()
    
    # Parse years
    if args.years:
        try:
            target_years = [int(y.strip()) for y in args.years.split(",")]
        except ValueError:
            print(f"[ERROR] Invalid years format: {args.years}")
            sys.exit(1)
    else:
        target_years = list(range(2018, datetime.now().year + 1))
    
    print("=" * 60)
    print("TRACINGINSIGHTS QUALIFYING IMPORTER")
    print("=" * 60)
    print(f"Years: {target_years}")
    print()
    
    racedata_path = Path("data/racedata")
    if not racedata_path.exists():
        print("[ERROR] data/racedata directory not found")
        print("Clone TracingInsights repository first:")
        print("  git clone https://github.com/TracingInsights/RaceData.git data/racedata")
        sys.exit(1)
    
    data_path = racedata_path / "data"
    qualifying_csv = data_path / "qualifying.csv"
    
    if not qualifying_csv.exists():
        print(f"[ERROR] {qualifying_csv} not found")
        sys.exit(1)
    
    try:
        db = get_mongo_client()
        print("Connected to MongoDB")
        
        # Create indexes
        db.f1_qualifying.create_index([("year", 1), ("round", 1), ("driver_code", 1)])
        db.f1_qualifying.create_index([("year", 1), ("position", 1)])
        db.f1_qualifying.create_index([("circuit_ref", 1)])
        
        # Load mappings
        print("Loading mappings...")
        race_mapping, driver_mapping, constructor_mapping = load_mappings(data_path, target_years)
        print(f"  Races: {len(race_mapping)}")
        print(f"  Drivers: {len(driver_mapping)}")
        print(f"  Constructors: {len(constructor_mapping)}")
        
        # Import
        print(f"\nImporting qualifying data...")
        imported = import_qualifying(db, qualifying_csv, race_mapping, 
                                      driver_mapping, constructor_mapping,
                                      target_years, args.force)
        
        print("\n" + "=" * 60)
        print(f"COMPLETED: {imported} qualifying records imported")
        print("=" * 60)
        
        # Log import
        db.f1_import_log.update_one(
            {
                "source": "tracinginsights",
                "type": "qualifying",
            },
            {
                "$set": {
                    "imported_at": datetime.utcnow().isoformat() + "Z",
                    "years": target_years,
                    "records_count": imported,
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