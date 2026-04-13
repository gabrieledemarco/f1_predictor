#!/usr/bin/env python3
"""
Import TracingInsights Pit Stops Data to MongoDB

Parses pit_stops.csv from RaceData and imports into MongoDB f1_pit_stops collection.
Includes pit stop duration, lap number, and stop count.

Usage:
    python scripts/import_pit_stops.py [--years YEARS] [--force]
"""

import os
import sys
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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
    
    # Race mapping: raceId -> {year, round, circuit_ref}
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
    
    # Results for constructor mapping per driver per race
    results_mapping = {}  # (raceId, driverId) -> constructorId
    results_csv = data_path / "results.csv"
    if results_csv.exists():
        with open(results_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                race_id = int(row.get("raceId", 0))
                if race_id not in race_mapping:
                    continue
                driver_id = int(row.get("driverId", 0))
                constructor_id = int(row.get("constructorId", 0))
                results_mapping[(race_id, driver_id)] = constructor_id
    
    return race_mapping, driver_mapping, constructor_mapping, results_mapping


def import_pit_stops(db, pit_stops_csv: Path, race_mapping: Dict, 
                    driver_mapping: Dict, constructor_mapping: Dict,
                    results_mapping: Dict, target_years: List[int], 
                    force: bool) -> int:
    """Import pit stops data from CSV."""
    imported_count = 0
    operations = []
    
    with open(pit_stops_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            race_id = int(row.get("raceId", 0))
            
            if race_id not in race_mapping:
                continue
            
            race_info = race_mapping[race_id]
            if race_info["year"] not in target_years:
                continue
            
            driver_id = int(row.get("driverId", 0))
            driver_code = driver_mapping.get(driver_id, f"D{driver_id}")
            
            # Get constructor from results
            constructor_id = results_mapping.get((race_id, driver_id), 0)
            constructor_ref = constructor_mapping.get(constructor_id, f"c{constructor_id}")
            
            stop_num = int(row.get("stop", 0))
            lap = int(row.get("lap", 0))
            
            milliseconds_raw = row.get("milliseconds", "")
            duration_ms = None
            duration_sec = None
            if milliseconds_raw and milliseconds_raw != "\\N":
                try:
                    duration_ms = int(milliseconds_raw)
                    duration_sec = duration_ms / 1000.0
                except ValueError:
                    pass
            
            time_of_day = row.get("time", "")
            
            doc_id = f"{race_info['year']}_{race_info['round']:02d}_{driver_code}_stop{stop_num}"
            
            doc = {
                "_id": doc_id,
                "year": race_info["year"],
                "round": race_info["round"],
                "circuit_ref": race_info["circuit_ref"],
                "race_id": race_id,
                "driver_code": driver_code,
                "driver_id": driver_id,
                "constructor_ref": constructor_ref,
                "stop_number": stop_num,
                "lap": lap,
                "time_of_day": time_of_day,
                "duration_ms": duration_ms,
                "duration_sec": duration_sec,
                "imported_at": datetime.utcnow().isoformat() + "Z",
                "source": "tracinginsights",
            }
            
            operations.append(UpdateOne(
                {"_id": doc["_id"]},
                {"$set": doc},
                upsert=True
            ))
            
            if len(operations) >= 500:
                result = db.f1_pit_stops.bulk_write(operations, ordered=False)
                imported_count += result.upserted_count + result.modified_count
                operations = []
    
    if operations:
        result = db.f1_pit_stops.bulk_write(operations, ordered=False)
        imported_count += result.upserted_count + result.modified_count
    
    return imported_count


def compute_team_pit_stop_stats(db, target_years: List[int]) -> None:
    """Compute aggregated pit stop statistics per team per race."""
    print("\nComputing team pit stop statistics...")
    
    pipeline = [
        {"$match": {"year": {"$in": target_years}}},
        {"$group": {
            "_id": {"year": "$year", "round": "$round", "constructor_ref": "$constructor_ref"},
            "avg_pit_stop_ms": {"$avg": "$duration_ms"},
            "min_pit_stop_ms": {"$min": "$duration_ms"},
            "max_pit_stop_ms": {"$max": "$duration_ms"},
            "total_stops": {"$sum": 1},
        }},
        {"$sort": {"_id.year": 1, "_id.round": 1}},
    ]
    
    results = list(db.f1_pit_stops.aggregate(pipeline))
    
    # Store in separate collection
    for r in results:
        doc = {
            "_id": f"{r['_id']['year']}_{r['_id']['round']:02d}_{r['_id']['constructor_ref']}",
            "year": r["_id"]["year"],
            "round": r["_id"]["round"],
            "constructor_ref": r["_id"]["constructor_ref"],
            "avg_pit_stop_ms": r["avg_pit_stop_ms"],
            "min_pit_stop_ms": r["min_pit_stop_ms"],
            "max_pit_stop_ms": r["max_pit_stop_ms"],
            "total_stops": r["total_stops"],
            "computed_at": datetime.utcnow().isoformat() + "Z",
        }
        db.f1_team_pit_stop_stats.update_one(
            {"_id": doc["_id"]},
            {"$set": doc},
            upsert=True
        )
    
    print(f"  Computed stats for {len(results)} team-race combinations")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Import pit stops data to MongoDB")
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
    print("TRACINGINSIGHTS PIT STOPS IMPORTER")
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
    pit_stops_csv = data_path / "pit_stops.csv"
    
    if not pit_stops_csv.exists():
        print(f"[ERROR] {pit_stops_csv} not found")
        sys.exit(1)
    
    try:
        db = get_mongo_client()
        print("Connected to MongoDB")
        
        # Create indexes
        db.f1_pit_stops.create_index([("year", 1), ("round", 1), ("driver_code", 1)])
        db.f1_pit_stops.create_index([("race_id", 1), ("driver_id", 1)])
        db.f1_pit_stops.create_index([("constructor_ref", 1)])
        
        db.f1_team_pit_stop_stats.create_index([
            ("year", 1), ("round", 1), ("constructor_ref", 1)
        ])
        
        # Load mappings
        print("Loading mappings...")
        (race_mapping, driver_mapping, constructor_mapping, 
         results_mapping) = load_mappings(data_path, target_years)
        print(f"  Races: {len(race_mapping)}")
        print(f"  Drivers: {len(driver_mapping)}")
        print(f"  Constructors: {len(constructor_mapping)}")
        
        # Import
        print(f"\nImporting pit stops data...")
        imported = import_pit_stops(db, pit_stops_csv, race_mapping,
                                     driver_mapping, constructor_mapping,
                                     results_mapping, target_years, args.force)
        
        print("\n" + "=" * 60)
        print(f"COMPLETED: {imported} pit stop records imported")
        print("=" * 60)
        
        # Compute team stats
        compute_team_pit_stop_stats(db, target_years)
        
        # Log import
        db.f1_import_log.update_one(
            {
                "source": "tracinginsights",
                "type": "pit_stops",
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