#!/usr/bin/env python3
"""
Verification Script: Verify Circuit Profiles in MongoDB

Verifies that circuit profiles have been correctly imported
and checks data quality.

Usage:
    python scripts/verify_circuit_profiles.py
"""

import os
import sys
from pathlib import Path

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


def verify_profiles(db):
    """Verify circuit profiles in MongoDB."""
    profiles = list(db.f1_circuit_profiles.find({}))
    
    print("=" * 60)
    print("CIRCUIT PROFILES VERIFICATION")
    print("=" * 60)
    print(f"\nTotal profiles: {len(profiles)}")
    
    if not profiles:
        print("\n[WARNING] No profiles found in database")
        return False
    
    print("\n| Circuit | Type | Top Speed | Full Throttle |")
    print("|---------|------|-----------|---------------|")
    
    for profile in sorted(profiles, key=lambda x: x["_id"]):
        print(f"| {profile['_id']:<7} | {profile.get('circuit_type', 'N/A'):<4} | "
              f"{profile.get('top_speed_kmh', 0):>9.1f} | "
              f"{profile.get('full_throttle_pct', 0):>12.1f} |")
    
    circuits_from_races = db.f1_races.distinct("circuit_ref")
    profiles_stored = {p["_id"] for p in profiles}
    
    missing = set(circuits_from_races) - profiles_stored
    
    if missing:
        print(f"\n[WARNING] Circuits without profiles: {', '.join(sorted(missing))}")
        return False
    
    print(f"\n[OK] All {len(profiles)} circuits verified")
    return True


def verify_data_counts(db):
    """Verify data counts in all collections."""
    print("\n" + "=" * 60)
    print("DATA COUNTS")
    print("=" * 60)
    
    collections = [
        ("f1_races", "Races"),
        ("f1_lap_times", "Lap Times"),
        ("f1_pace_observations", "Pace Observations"),
        ("f1_pinnacle_odds", "Odds"),
        ("f1_circuit_profiles", "Circuit Profiles"),
        ("f1_import_log", "Import Logs"),
    ]
    
    print("\n| Collection | Count |")
    print("|------------|-------|")
    
    for coll_name, label in collections:
        count = db[coll_name].count_documents({})
        print(f"| {coll_name:<20} | {count:>6} |")
    
    years = list(range(2018, 2026))
    races_per_year = list(db.f1_races.aggregate([
        {"$group": {"_id": "$year", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]))
    
    print("\n| Year | Races |")
    print("|------|-------|")
    for r in races_per_year:
        print(f"| {r['_id']} | {r['count']:>6} |")


def main():
    try:
        db = get_mongo_client()
        print("Connected to MongoDB")
        print()
        
        profiles_ok = verify_profiles(db)
        verify_data_counts(db)
        
        print("\n" + "=" * 60)
        if profiles_ok:
            print("[SUCCESS] Verification complete")
            sys.exit(0)
        else:
            print("[WARNING] Verification completed with warnings")
            sys.exit(1)
        
    except PyMongoError as e:
        print(f"\n[MongoDB Error] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
