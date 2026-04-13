#!/usr/bin/env python3
"""
Import Driver Standings to MongoDB

Fetches driver standings data from Jolpica API and imports them
into MongoDB f1_driver_standings collection.

Usage:
    python scripts/import_standings.py
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Optional

import requests
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError

from rate_limiter import RateLimiter

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1"
jolpica_limiter = RateLimiter(requests_per_second=3.0, max_retries=5, backoff_factor=2.0)


def get_mongo_client():
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is required")
    
    mongo_db = os.environ.get("MONGO_DB", "betbreaker")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    return client[mongo_db]


def fetch_json(url: str, params: dict = None) -> Optional[dict]:
    """Fetch JSON from Jolpica API with rate limiting."""
    try:
        response = jolpica_limiter.request_with_retry("GET", url, params=params, timeout=30)
        
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            log.warning(f"Rate limited. Waiting {retry_after}s...")
            time.sleep(retry_after)
            return fetch_json(url, params)
        
        if response.status_code >= 400:
            log.error(f"API error {response.status_code} for {url}")
            return None
        
        return response.json()
        
    except Exception as e:
        log.error(f"Failed to fetch {url}: {e}")
        return None


def get_season_rounds(year: int) -> int:
    """Get the number of rounds in a season."""
    url = f"{JOLPICA_BASE_URL}/{year}.json"
    data = fetch_json(url)
    
    if not data or "MRData" not in data:
        return 0
    
    races = data["MRData"]["RaceTable"]["Races"]
    return len(races)


def import_standings(db, year: int, round_num: int) -> bool:
    """Import driver standings for a specific race."""
    url = f"{JOLPICA_BASE_URL}/{year}/{round_num}/driverStandings.json"
    data = fetch_json(url)
    
    if not data or "MRData" not in data:
        log.warning(f"  No standings data for {year} R{round_num:02d}")
        return False
    
    standings_table = data["MRData"]["StandingsTable"]
    if not standings_table or "StandingsLists" not in standings_table:
        log.warning(f"  No standings lists for {year} R{round_num:02d}")
        return False
    
    standings_lists = standings_table["StandingsLists"]
    if not standings_lists:
        log.warning(f"  Empty standings list for {year} R{round_num:02d}")
        return False
    
    standings_list = standings_lists[0]
    
    driver_standings = []
    for standing in standings_list.get("DriverStandings", []):
        driver_standings.append({
            "position": int(standing.get("position", 0)),
            "driver_code": standing["Driver"]["code"],
            "driver_id": standing["Driver"]["driverId"],
            "constructor_refs": [
                c["constructorId"].lower().replace(" ", "_")
                for c in standing.get("Constructors", [])
            ],
            "points": float(standing.get("points", 0)),
            "wins": int(standing.get("wins", 0)),
            "position_text": standing.get("positionText", ""),
        })
    
    race = db.f1_races.find_one({"_id": f"{year}_{round_num:02d}"})
    if not race:
        log.warning(f"  Race {year} R{round_num:02d} not found in database")
        return False
    
    circuit_ref = race.get("circuit_ref", "")
    
    doc = {
        "_id": f"{year}_{round_num:02d}_standings",
        "year": year,
        "round": round_num,
        "circuit_ref": circuit_ref,
        "race_name": race.get("race_name", ""),
        "date": standings_list.get("date", ""),
        "standings": driver_standings,
        "imported_at": datetime.utcnow().isoformat() + "Z",
        "source": "jolpica",
    }
    
    db.f1_driver_standings.update_one(
        {"_id": doc["_id"]},
        {"$set": doc},
        upsert=True
    )
    
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Import driver standings to MongoDB")
    parser.add_argument("--from-year", type=int, default=2018)
    parser.add_argument("--to-year", type=int, default=datetime.now().year)
    args = parser.parse_args()
    
    log.info("=" * 60)
    log.info("DRIVER STANDINGS IMPORTER")
    log.info("=" * 60)
    log.info(f"Years: {args.from_year} - {args.to_year}")
    
    try:
        db = get_mongo_client()
        log.info("Connected to MongoDB")
        
        db.f1_driver_standings.create_index([("year", 1), ("round", 1)], unique=True)
        db.f1_driver_standings.create_index([("standings.driver_code", 1)])
        
        total_imported = 0
        
        for year in range(args.from_year, args.to_year + 1):
            rounds = get_season_rounds(year)
            if rounds == 0:
                log.warning(f"Could not get rounds for {year}, skipping...")
                continue
            
            log.info(f"\n{year} Season ({rounds} rounds):")
            
            for round_num in range(1, rounds + 1):
                existing = db.f1_driver_standings.find_one(
                    {"_id": f"{year}_{round_num:02d}_standings"}
                )
                if existing:
                    log.debug(f"  {year} R{round_num:02d}: Already imported, skipping")
                    continue
                
                if import_standings(db, year, round_num):
                    total_imported += 1
                    log.info(f"  Imported {year} R{round_num:02d} standings")
        
        log.info("\n" + "=" * 60)
        log.info(f"COMPLETED: {total_imported} standings imported")
        log.info("=" * 60)
        
    except PyMongoError as e:
        log.error(f"MongoDB Error: {e}")
        sys.exit(1)
    except Exception as e:
        log.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
