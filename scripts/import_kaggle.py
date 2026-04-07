#!/usr/bin/env python3
"""
Import Kaggle Historical Data to MongoDB

Downloads and imports historical F1 data from Kaggle
into MongoDB f1_lap_times collection.

Usage:
    python scripts/import_kaggle.py
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


def get_mongo_client():
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is required")
    
    mongo_db = os.environ.get("MONGO_DB", "betbreaker")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    return client[mongo_db]


def get_round_from_race_id(db, year: int, kaggle_race_id: int) -> int:
    """Get round number from Kaggle race ID."""
    race = db.f1_races.find_one({"year": year})
    return race.get("round", 0) if race else 0


def get_driver_mapping(db, kaggle_driver_id: int) -> str:
    """Map Kaggle driver ID to driver code."""
    result = db.f1_races.find_one(
        {"results.driver_id": str(kaggle_driver_id)},
        {"results.$": 1}
    )
    if result and result.get("results"):
        return result["results"][0].get("driver_code", "")
    return f"DRV{kaggle_driver_id}"


def get_constructor_mapping(db, kaggle_constructor_id: int) -> str:
    """Map Kaggle constructor ID to constructor ref."""
    result = db.f1_races.find_one(
        {"results.constructor_id": str(kaggle_constructor_id)},
        {"results.$": 1}
    )
    if result and result.get("results"):
        return result["results"][0].get("constructor_ref", "")
    return f"team{kaggle_constructor_id}"


def import_lap_times(db, kaggle_path: Path, years: List[int]) -> int:
    """Import lap times from Kaggle dataset."""
    lap_times_file = kaggle_path / "lap_times.csv"
    if not lap_times_file.exists():
        return 0
    
    races_file = kaggle_path / "races.csv"
    if not races_file.exists():
        return 0
    
    races_df = pd.read_csv(races_file)
    races_df = races_df[races_df["year"].isin(years)]
    
    race_id_to_info = {}
    for _, row in races_df.iterrows():
        year = int(row["year"])
        kaggle_race_id = int(row["raceId"])
        
        existing_race = db.f1_races.find_one({
            "year": year,
            "round": int(row["round"])
        })
        if existing_race:
            race_id_to_info[kaggle_race_id] = {
                "year": year,
                "round": int(row["round"]),
                "circuit_ref": existing_race.get("circuit_ref", "")
            }
    
    if not race_id_to_info:
        return 0
    
    operations = []
    imported_count = 0
    
    chunksize = 10000
    for chunk in pd.read_csv(lap_times_file, chunksize=chunksize):
        chunk = chunk[chunk["raceId"].isin(race_id_to_info.keys())]
        
        for _, row in chunk.iterrows():
            kaggle_race_id = int(row["raceId"])
            if kaggle_race_id not in race_id_to_info:
                continue
            
            race_info = race_id_to_info[kaggle_race_id]
            lap_time_ms = float(row["milliseconds"]) if pd.notna(row["milliseconds"]) else None
            
            if lap_time_ms is None:
                continue
            
            driver_code = get_driver_mapping(db, int(row["driverId"]))
            
            doc = {
                "_id": f"{race_info['year']}_{race_info['circuit_ref']}_{driver_code}_{int(row['lap'])}",
                "year": race_info["year"],
                "round": race_info["round"],
                "circuit_ref": race_info["circuit_ref"],
                "driver_code": driver_code,
                "lap_number": int(row["lap"]),
                "lap_time_ms": lap_time_ms,
                "position": int(row["position"]) if pd.notna(row["position"]) else None,
                "imported_at": datetime.utcnow().isoformat() + "Z",
                "source": "kaggle",
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
            operations = []
    
    return imported_count


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Import Kaggle historical data to MongoDB")
    parser.add_argument("--years", type=str, default=os.environ.get("YEARS", "2018,2019,2020,2021"))
    parser.add_argument("--force", action="store_true", default=os.environ.get("FORCE", "false").lower() == "true")
    args = parser.parse_args()
    
    years = [int(y.strip()) for y in args.years.split(",")]
    
    print("=" * 60)
    print("KAGGLE DATA IMPORTER")
    print("=" * 60)
    print(f"Years: {years}")
    print()
    
    kaggle_path = Path("data/kaggle_raw")
    if not kaggle_path.exists():
        print("[ERROR] data/kaggle_raw directory not found")
        print("Run import-kaggle.yml workflow first to download Kaggle data")
        sys.exit(1)
    
    try:
        db = get_mongo_client()
        print("Connected to MongoDB")
        
        db.f1_lap_times.create_index([
            ("year", 1), ("round", 1), ("driver_code", 1)
        ])
        
        total_laps = import_lap_times(db, kaggle_path, years)
        
        print("\n" + "=" * 60)
        print(f"COMPLETED: {total_laps} laps imported from Kaggle")
        print("=" * 60)
        
    except PyMongoError as e:
        print(f"\n[MongoDB Error] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
