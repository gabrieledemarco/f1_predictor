#!/usr/bin/env python3
"""
Extract Circuit Profiles from FastF1

Uses FastF1 library to extract circuit speed profiles
and stores them in MongoDB f1_circuit_profiles collection.

Usage:
    python scripts/extract_circuit_profiles.py
"""

import os
import sys
from datetime import datetime
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


CIRCUIT_TYPE_MAPPING = {
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
}


def extract_circuit_profile(year: int, circuit_ref: str) -> Optional[Dict]:
    """Extract speed profile for a circuit using FastF1."""
    try:
        import fastf1
        import numpy as np
        
        fastf1.Cache.enable_cache("data/cache/fastf1")
        
        session = fastf1.get_session(year, circuit_ref, "R")
        session.load(laps=True, telemetry=True, weather=False, messages=False)
        
        lap = session.laps.pick_fastest()
        if lap is None:
            return None
        
        telemetry = lap.get_telemetry()
        
        speed = telemetry["Speed"].dropna()
        throttle = telemetry["Throttle"].dropna()
        
        speed = speed[speed > 0]
        
        top_speed = float(speed.max()) if len(speed) > 0 else 0
        min_speed = float(speed.min()) if len(speed) > 0 else 0
        avg_speed = float(speed.mean()) if len(speed) > 0 else 0
        
        full_throttle = (throttle >= 95).mean() * 100
        
        return {
            "circuit_ref": circuit_ref,
            "circuit_type": CIRCUIT_TYPE_MAPPING.get(circuit_ref, "mixed"),
            "top_speed_kmh": top_speed,
            "min_speed_kmh": min_speed,
            "avg_speed_kmh": avg_speed,
            "full_throttle_pct": float(full_throttle),
            "extracted_year": year,
            "extracted_at": datetime.utcnow().isoformat() + "Z",
        }
        
    except Exception as e:
        print(f"  Warning: Failed to extract {circuit_ref}: {e}")
        return None


def get_circuits_from_db(db, year: int) -> list:
    """Get circuit refs from database for a given year."""
    races = db.f1_races.find({"year": year}, {"circuit_ref": 1})
    return list(set(r["circuit_ref"] for r in races))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract circuit profiles from FastF1")
    parser.add_argument("--year", type=int, default=int(os.environ.get("YEAR", str(datetime.now().year))))
    args = parser.parse_args()
    
    print("=" * 60)
    print("CIRCUIT PROFILE EXTRACTION")
    print("=" * 60)
    print(f"Year: {args.year}")
    print()
    
    try:
        db = get_mongo_client()
        print("Connected to MongoDB")
        
        db.f1_circuit_profiles.create_index([("circuit_ref", 1)], unique=True)
        
        circuits = get_circuits_from_db(db, args.year)
        print(f"Found {len(circuits)} circuits for {args.year}")
        
        if not circuits:
            print("[ERROR] No circuits found in database")
            print("Run import-jolpica.yml first to import race data")
            sys.exit(1)
        
        extracted = 0
        for circuit_ref in sorted(circuits):
            print(f"  Extracting {circuit_ref}...")
            profile = extract_circuit_profile(args.year, circuit_ref)
            
            if profile:
                db.f1_circuit_profiles.update_one(
                    {"_id": circuit_ref},
                    {"$set": profile},
                    upsert=True
                )
                extracted += 1
                print(f"    -> Extracted (top_speed={profile['top_speed_kmh']:.1f} km/h)")
        
        db.f1_import_log.update_one(
            {"source": "fastf1", "type": "circuit_profiles"},
            {
                "$set": {
                    "imported_at": datetime.utcnow().isoformat() + "Z",
                    "year": args.year,
                    "circuits_extracted": extracted,
                }
            },
            upsert=True
        )
        
        print("\n" + "=" * 60)
        print(f"COMPLETED: {extracted}/{len(circuits)} circuits extracted")
        print("=" * 60)
        
    except PyMongoError as e:
        print(f"\n[MongoDB Error] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
