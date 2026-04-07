#!/usr/bin/env python3
"""
Import Jolpica F1 Data to MongoDB

Fetches race results and qualifying data from Jolpica/Ergast API
and imports them into MongoDB f1_races collection.

Usage:
    python scripts/import_jolpica.py
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError

load_dotenv()

JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1"
CIRCUIT_TYPES = {
    "bahrain": "desert",
    "jeddah": "desert",
    "albert_park": "street",
    "baku": "street",
    "catalunya": "mixed",
    "monaco": "street",
    "spa": "high_speed",
    "monza": "high_speed",
    "singapore": "street",
    "austin": "mixed",
    "mexico_city": "high_altitude",
    "brazil": "mixed",
    "las_vegas": "street",
    "qatar": "desert",
    "abu_dhabi": "desert",
    "suzuka": "high_speed",
    "imola": "mixed",
    "miami": "street",
    "shanghai": "desert",
    "zandvoort": "high_speed",
    "hungaroring": "mixed",
    "silverstone": "high_speed",
    "zhangjiang": "desert",
}


def get_mongo_client():
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is required")
    
    mongo_db = os.environ.get("MONGO_DB", "betbreaker")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    return client[mongo_db]


def fetch_json(url: str, params: dict = None) -> Optional[dict]:
    """Fetch JSON from Jolpica API with rate limiting."""
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            time.sleep(0.25)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return None
    return None


def parse_race_data(race_data: dict, year: int, round_num: int) -> dict:
    """Parse Jolpica API response into MongoDB document format."""
    race_name = race_data["raceName"]
    circuit_ref = race_data["Circuit"]["circuitId"].lower().replace(" ", "_")
    circuit_name = race_data["Circuit"]["circuitName"]
    location = race_data["Circuit"]["Location"]
    
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
                "fastest_lap_rank": int(r.get("FastestLap", {}).get("rank", 0)) if "FastestLap" in r else None,
                "fastest_lap_time": r.get("FastestLap", {}).get("Time", {}).get("time") if "FastestLap" in r else None,
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
    season_end_months = [11, 12]
    is_season_end = int(date_str.split("-")[1] if date_str else "01") in season_end_months
    
    is_major_reg_change = year in [2022, 2026]
    
    return {
        "_id": f"{year}_{round_num:02d}",
        "year": year,
        "round": round_num,
        "circuit_ref": circuit_ref,
        "circuit_name": circuit_name,
        "circuit_type": CIRCUIT_TYPES.get(circuit_ref, "mixed"),
        "location": {
            "country": location.get("country"),
            "locality": location.get("locality"),
            "lat": float(location.get("lat", 0)),
            "lng": float(location.get("long", 0)),
        },
        "race_name": race_name,
        "date": date_str,
        "time": race_data.get("time", "14:00:00Z"),
        "is_sprint_weekend": race_data.get("Sprint", {}).get("date") is not None if "Sprint" in race_data else False,
        "is_season_end": is_season_end,
        "is_major_regulation_change": is_major_reg_change,
        "results": results,
        "qualifying": qualifying,
        "imported_at": datetime.utcnow().isoformat() + "Z",
        "source": "jolpica",
    }


def import_race(db, year: int, round_num: int) -> bool:
    """Import a single race from Jolpica API."""
    url = f"{JOLPICA_BASE_URL}/{year}/{round_num}/results.json"
    data = fetch_json(url)
    
    if not data or "MRData" not in data:
        print(f"  Failed to fetch race {year} R{round_num:02d}")
        return False
    
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        print(f"  No race data for {year} R{round_num:02d}")
        return False
    
    race_doc = parse_race_data(races[0], year, round_num)
    
    db.f1_races.update_one(
        {"_id": race_doc["_id"]},
        {"$set": race_doc},
        upsert=True
    )
    
    db.f1_import_log.update_one(
        {
            "source": "jolpica",
            "year": year,
            "round": round_num,
            "type": "race",
        },
        {
            "$set": {
                "imported_at": datetime.utcnow().isoformat() + "Z",
                "document_id": race_doc["_id"],
            }
        },
        upsert=True
    )
    
    print(f"  Imported {year} R{round_num:02d}: {race_doc['race_name']}")
    return True


def get_season_rounds(year: int) -> int:
    """Get the number of rounds in a season."""
    url = f"{JOLPICA_BASE_URL}/{year}.json"
    data = fetch_json(url)
    
    if not data or "MRData" not in data:
        return 0
    
    races = data["MRData"]["RaceTable"]["Races"]
    return len(races)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Import Jolpica F1 data to MongoDB")
    parser.add_argument("--from-year", type=int, default=int(os.environ.get("FROM_YEAR", "2018")))
    parser.add_argument("--to-year", type=int, default=int(os.environ.get("TO_YEAR", datetime.now().year)))
    parser.add_argument("--force", action="store_true", default=os.environ.get("FORCE", "false").lower() == "true")
    args = parser.parse_args()
    
    print("=" * 60)
    print("JOLPICA DATA IMPORTER")
    print("=" * 60)
    print(f"Years: {args.from_year} - {args.to_year}")
    print(f"Force re-import: {args.force}")
    print()
    
    try:
        db = get_mongo_client()
        print("Connected to MongoDB")
        
        db.f1_races.create_index([("year", 1), ("round", 1)], unique=True)
        db.f1_races.create_index([("year", 1), ("circuit_ref", 1)])
        db.f1_import_log.create_index([("source", 1), ("year", 1), ("imported_at", 1)])
        
        total_imported = 0
        
        for year in range(args.from_year, args.to_year + 1):
            rounds = get_season_rounds(year)
            if rounds == 0:
                print(f"\n[WARNING] Could not get rounds for {year}, skipping...")
                continue
            
            print(f"\n{year} Season ({rounds} rounds):")
            
            for round_num in range(1, rounds + 1):
                if not args.force:
                    existing = db.f1_races.find_one({"_id": f"{year}_{round_num:02d}"})
                    if existing and existing.get("source") == "jolpica":
                        print(f"  {year} R{round_num:02d}: Already imported, skipping")
                        continue
                
                import_race(db, year, round_num)
                total_imported += 1
        
        print("\n" + "=" * 60)
        print(f"COMPLETED: {total_imported} races imported")
        print("=" * 60)
        
    except PyMongoError as e:
        print(f"\n[MongoDB Error] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
