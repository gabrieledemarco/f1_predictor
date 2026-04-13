#!/usr/bin/env python3
"""
Import Circuit Profiles to MongoDB

Imports predefined circuit profiles from f1_predictor/data/circuit_profiles.py
into MongoDB f1_circuit_profiles collection.

Usage:
    python scripts/import_circuit_profiles.py
"""

import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from f1_predictor.data.circuit_profiles import CIRCUIT_PROFILES


def get_mongo_client():
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is required")
    
    mongo_db = os.environ.get("MONGO_DB", "betbreaker")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    return client[mongo_db]


def main():
    print("=" * 60)
    print("CIRCUIT PROFILES IMPORT")
    print("=" * 60)
    print(f"Found {len(CIRCUIT_PROFILES)} predefined profiles")
    print()
    
    try:
        db = get_mongo_client()
        print("Connected to MongoDB")
        
        db.f1_circuit_profiles.create_index([("circuit_ref", 1)], unique=True)
        
        operations = []
        for circuit_ref, profile in CIRCUIT_PROFILES.items():
            doc = {
                "_id": circuit_ref,
                "circuit_ref": circuit_ref,
                "circuit_type": profile.circuit_type.value if hasattr(profile.circuit_type, 'value') else str(profile.circuit_type),
                "top_speed_kmh": profile.top_speed_kmh,
                "min_speed_kmh": profile.min_speed_kmh,
                "avg_speed_kmh": profile.avg_speed_kmh,
                "avg_slow_corner_kmh": getattr(profile, 'avg_slow_corner_kmh', 0),
                "avg_fast_corner_kmh": getattr(profile, 'avg_fast_corner_kmh', 0),
                "avg_medium_corner_kmh": getattr(profile, 'avg_medium_corner_kmh', 0),
                "full_throttle_pct": profile.full_throttle_pct,
                "source": getattr(profile, 'source', 'predefined'),
                "imported_at": datetime.utcnow().isoformat() + "Z",
            }
            operations.append(UpdateOne(
                {"_id": circuit_ref},
                {"$set": doc},
                upsert=True
            ))
        
        if operations:
            result = db.f1_circuit_profiles.bulk_write(operations, ordered=False)
            print(f"Imported {result.upserted_count} new profiles")
            print(f"Updated {result.modified_count} existing profiles")
        
        total = db.f1_circuit_profiles.count_documents({})
        print(f"Total profiles in database: {total}")
        
        print("\n" + "=" * 60)
        print(f"COMPLETED: {len(operations)} circuit profiles")
        print("=" * 60)
        
        db.f1_import_log.update_one(
            {"source": "predefined", "type": "circuit_profiles"},
            {
                "$set": {
                    "imported_at": datetime.utcnow().isoformat() + "Z",
                    "profiles_count": len(operations),
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
