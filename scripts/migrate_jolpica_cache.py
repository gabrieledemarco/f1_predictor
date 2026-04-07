#!/usr/bin/env python3
"""
Migration Script: Import Jolpica Cache to MongoDB

Imports existing Jolpica JSON cache files into MongoDB.
Run this once to migrate historical data from disk to MongoDB.

Usage:
    python scripts/migrate_jolpica_cache.py [--from-dir data/cache/jolpica]
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

load_dotenv()


def get_mongo_client():
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is required")
    
    mongo_db = os.environ.get("MONGO_DB", "betbreaker")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    return client[mongo_db]


def parse_cache_filename(filename: str) -> Optional[tuple]:
    """Parse filename like '2024_01_results.json' to (year, round)."""
    name = filename.replace("_results.json", "").replace("_qualifying.json", "")
    parts = name.split("_")
    if len(parts) == 2:
        try:
            year = int(parts[0])
            round_num = int(parts[1])
            return year, round_num
        except ValueError:
            return None
    return None


def import_cache_file(db, cache_path: Path) -> int:
    """Import a single Jolpica cache file."""
    year, round_num = parse_cache_filename(cache_path.name)
    if year is None:
        return 0
    
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if "MRData" not in data:
        return 0
    
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        return 0
    
    race_data = races[0]
    
    circuit_ref = race_data["Circuit"]["circuitId"].lower().replace(" ", "_")
    
    results = []
    if "Results" in race_data:
        for r in race_data["Results"]:
            results.append({
                "driver_code": r["Driver"]["code"],
                "driver_id": r["Driver"]["driverId"],
                "constructor_ref": r["Constructor"]["constructorId"].lower().replace(" ", "_"),
                "grid_position": int(r.get("gridPosition", 0)),
                "finish_position": int(r.get("position", 0)) if r.get("position") else None,
                "points": float(r.get("points", 0)),
                "laps_completed": int(r.get("laps", 0)),
                "status": r.get("status", "Finished"),
            })
    
    qualifying = []
    if "QualifyingResults" in race_data:
        for q in race_data["QualifyingResults"]:
            qualifying.append({
                "driver_code": q["Driver"]["code"],
                "grid_position": int(q.get("position", 0)),
                "q1": q.get("Q1"),
                "q2": q.get("Q2"),
                "q3": q.get("Q3"),
            })
    
    date_str = race_data.get("date", "")
    is_season_end = int(date_str.split("-")[1] if date_str else "01") in [11, 12]
    
    doc = {
        "_id": f"{year}_{round_num:02d}",
        "year": year,
        "round": round_num,
        "circuit_ref": circuit_ref,
        "circuit_name": race_data["Circuit"]["circuitName"],
        "circuit_type": "mixed",
        "race_name": race_data["raceName"],
        "date": date_str,
        "time": race_data.get("time", "14:00:00Z"),
        "is_sprint_weekend": "Sprint" in race_data,
        "is_season_end": is_season_end,
        "is_major_regulation_change": year in [2022, 2026],
        "results": results,
        "qualifying": qualifying,
        "imported_at": datetime.utcnow().isoformat() + "Z",
        "source": "jolpica",
        "migrated_from_cache": True,
    }
    
    db.f1_races.update_one(
        {"_id": doc["_id"]},
        {"$set": doc},
        upsert=True
    )
    
    return 1


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Migrate Jolpica cache to MongoDB")
    parser.add_argument("--from-dir", type=str, default="data/cache/jolpica")
    args = parser.parse_args()
    
    cache_dir = Path(args.from_dir)
    
    if not cache_dir.exists():
        print(f"[ERROR] Cache directory not found: {cache_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("JOLPICA CACHE MIGRATION")
    print("=" * 60)
    print(f"Source: {cache_dir}")
    print()
    
    try:
        db = get_mongo_client()
        print("Connected to MongoDB")
        
        db.f1_races.create_index([("year", 1), ("round", 1)], unique=True)
        
        cache_files = sorted(cache_dir.glob("*_results.json"))
        print(f"Found {len(cache_files)} cache files")
        
        imported = 0
        skipped = 0
        
        for cache_file in cache_files:
            existing = db.f1_races.find_one({"_id": cache_file.stem.replace("_results", "")})
            if existing and existing.get("migrated_from_cache"):
                skipped += 1
                continue
            
            count = import_cache_file(db, cache_file)
            if count > 0:
                imported += 1
                print(f"  {cache_file.name} -> imported")
            else:
                print(f"  {cache_file.name} -> skipped (invalid)")
        
        print("\n" + "=" * 60)
        print(f"COMPLETED: {imported} imported, {skipped} skipped")
        print("=" * 60)
        
    except PyMongoError as e:
        print(f"\n[MongoDB Error] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
