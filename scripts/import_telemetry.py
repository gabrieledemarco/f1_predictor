#!/usr/bin/env python3
"""
Import Sector Times from TracingInsights using GitHub API

Fetches telemetry data directly from GitHub API - no local files.
Rate limited to avoid GitHub API limits.

Usage:
    python scripts/import_telemetry.py --year 2024
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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


def fetch_json(url: str, retries: int = 3) -> Optional[dict]:
    """Fetch JSON content from GitHub API with retry logic."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/vnd.github.v3+json")
            
            with urllib.request.urlopen(req, timeout=15) as response:
                return json.loads(response.read().decode())
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                return None
    return None


def fetch_gp_directories(year: int) -> List[str]:
    """Fetch list of GP directories from TracingInsights repo."""
    url = f"https://api.github.com/repos/TracingInsights/{year}/contents"
    data = fetch_json(url)
    
    if not data:
        return []
    
    return [d["name"] for d in data if d.get("type") == "dir"]


def fetch_driver_directories(year: int, gp_name: str, session_type: str) -> List[str]:
    """Fetch list of driver directories for a specific GP and session."""
    url = f"https://api.github.com/repos/TracingInsights/{year}/contents/{urllib.parse.quote(gp_name)}/{session_type}"
    data = fetch_json(url)
    
    if not data:
        return []
    
    return [d["name"] for d in data if d.get("type") == "dir"]


def parse_sector_time(value) -> Optional[float]:
    """Parse sector time from string to milliseconds."""
    if value == "None" or value is None:
        return None
    try:
        return float(value) * 1000
    except (ValueError, TypeError):
        return None


def fetch_and_parse_laptimes(year: int, gp_name: str, session_type: str, driver_code: str) -> Optional[dict]:
    """Fetch and parse laps times for a specific driver."""
    url = f"https://raw.githubusercontent.com/TracingInsights/{year}/main/{urllib.parse.quote(gp_name)}/{session_type}/{driver_code}/laptimes.json"
    
    data = fetch_json(url)
    if not data:
        return None
    
    s1_times = data.get("s1", [])
    s2_times = data.get("s2", [])
    s3_times = data.get("s3", [])
    
    s1_best = s2_best = s3_best = None
    
    for i in range(len(s1_times)):
        s1 = parse_sector_time(s1_times[i]) if i < len(s1_times) else None
        s2 = parse_sector_time(s2_times[i]) if i < len(s2_times) else None
        s3 = parse_sector_time(s3_times[i]) if i < len(s3_times) else None
        
        if s1 is not None and (s1_best is None or s1 < s1_best):
            s1_best = s1
        if s2 is not None and (s2_best is None or s2 < s2_best):
            s2_best = s2
        if s3 is not None and (s3_best is None or s3 < s3_best):
            s3_best = s3
    
    return {
        "s1_best_ms": s1_best,
        "s2_best_ms": s2_best,
        "s3_best_ms": s3_best,
    }


def import_session_data(db, year: int) -> int:
    """Import sector times from TracingInsights API."""
    print(f"Fetching GP list for {year}...")
    gp_list = fetch_gp_directories(year)
    
    if not gp_list:
        print(f"[ERROR] No GPs found for year {year}")
        return 0
    
    print(f"  Found {len(gp_list)} Grand Prix")
    
    operations = []
    imported_count = 0
    
    for gp_name in gp_list:
        print(f"  Processing: {gp_name}", end=" ", flush=True)
        
        driver_dirs = fetch_driver_directories(year, gp_name, "Qualifying")
        
        if not driver_dirs:
            print("(no qualifying data)")
            continue
        
        print(f"({len(driver_dirs)} drivers)", end=" ", flush=True)
        
        for i, driver_code in enumerate(driver_dirs):
            if i > 0 and i % 5 == 0:
                time.sleep(0.5)
            
            lap_data = fetch_and_parse_laptimes(year, gp_name, "Qualifying", driver_code)
            
            if not lap_data or not any([lap_data.get("s1_best_ms"), lap_data.get("s2_best_ms"), lap_data.get("s3_best_ms")]):
                continue
            
            gp_ref = gp_name.replace(" Grand Prix", "").replace(" ", "_").lower()
            
            doc_id = f"{year}_{gp_ref}_{driver_code}_qualifying"
            
            doc = {
                "_id": doc_id,
                "year": year,
                "circuit_ref": gp_ref,
                "driver_code": driver_code,
                "session_type": "qualifying",
                "s1_best_ms": lap_data.get("s1_best_ms"),
                "s2_best_ms": lap_data.get("s2_best_ms"),
                "s3_best_ms": lap_data.get("s3_best_ms"),
                "imported_at": datetime.utcnow().isoformat() + "Z",
                "source": "tracinginsights",
            }
            
            s1 = lap_data.get("s1_best_ms")
            s2 = lap_data.get("s2_best_ms")
            s3 = lap_data.get("s3_best_ms")
            if s1 is not None and s2 is not None and s3 is not None:
                doc["total_best_ms"] = float(s1 + s2 + s3)
            
            operations.append(UpdateOne(
                {"_id": doc["_id"]},
                {"$set": doc},
                upsert=True
            ))
        
        if operations:
            result = db.f1_session_stats.bulk_write(operations, ordered=False)
            imported_count += result.upserted_count + result.modified_count
            operations = []
        
        print(f"-> {imported_count} imported")
        
        time.sleep(0.3)
    
    return imported_count


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Import telemetry data from TracingInsights")
    parser.add_argument("--year", type=int, default=datetime.now().year)
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRACINGINSIGHTS TELEMETRY IMPORTER (GitHub API)")
    print("=" * 60)
    print(f"Year: {args.year}")
    print()
    
    try:
        db = get_mongo_client()
        print("Connected to MongoDB")
        
        db.f1_session_stats.create_index([
            ("year", 1), ("circuit_ref", 1), ("driver_code", 1), ("session_type", 1)
        ])
        
        print(f"\nImporting sector times from {args.year}...")
        imported = import_session_data(db, args.year)
        
        print("\n" + "=" * 60)
        print(f"COMPLETED: {imported} session records imported")
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
