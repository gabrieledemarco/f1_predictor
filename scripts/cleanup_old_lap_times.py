#!/usr/bin/env python3
"""
Cleanup old lap times with invalid circuit_ref

Removes lap times where circuit_ref matches pattern 'circuit_*'
(legacy data from old import that didn't resolve circuit names).

Usage:
    python scripts/cleanup_old_lap_times.py
"""

import os
import sys
from datetime import datetime

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


def cleanup_old_lap_times(db) -> int:
    """Remove lap times with invalid circuit_ref (circuit_*)"""
    result = db.f1_lap_times.delete_many({
        "circuit_ref": {"$regex": r"^circuit_"}
    })
    return result.deleted_count


def main():
    print("=" * 60)
    print("CLEANUP OLD LAP TIMES")
    print("=" * 60)
    print()
    
    try:
        db = get_mongo_client()
        print("Connected to MongoDB")
        
        count_before = db.f1_lap_times.count_documents({})
        print(f"Documents before cleanup: {count_before}")
        
        deleted = cleanup_old_lap_times(db)
        print(f"Deleted {deleted} old lap times")
        
        count_after = db.f1_lap_times.count_documents({})
        print(f"Documents after cleanup: {count_after}")
        
        print("\n" + "=" * 60)
        print(f"COMPLETED: {deleted} documents removed")
        print("=" * 60)
        
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
