#!/usr/bin/env python3
"""
Import Pinnacle Odds to MongoDB

Fetches betting odds from The Odds API and imports them into MongoDB
f1_pinnacle_odds collection.

Usage:
    python scripts/import_pinnacle_odds.py
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import requests
from dotenv import load_dotenv
from pymongo import MongoClient, InsertOne
from pymongo.errors import PyMongoError

load_dotenv()

ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports/motorsport_formula_one/odds"
DRIVER_NAME_MAPPING = {
    "max_verstappen": "VER",
    "leclerc": "LEC",
    "hamilton": "HAM",
    "norris": "NOR",
    "sainz": "SAI",
    "perez": "PER",
    "russell": "RUS",
    "alonso": "ALO",
    "piastri": "PIA",
    "lawson": "LAW",
    "tsunoda": "TSU",
    "ocon": "OCO",
    "gasly": "GAS",
    "albon": "ALB",
    "stroll": "STR",
    "magnussen": "MAG",
    "bottas": "BOT",
    "zhou": "ZHO",
    "hulkenberg": "HUL",
    "ricciardo": "RIC",
    "colapinto": "COL",
    "doohan": "DOO",
    "iyama": "IYR",
}


def get_mongo_client():
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is required")
    
    mongo_db = os.environ.get("MONGO_DB", "betbreaker")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    return client[mongo_db]


def remove_vig(probabilities: List[float]) -> List[float]:
    """Remove bookmaker vig using additive method (Power Method)."""
    probs = np.array(probabilities)
    total = probs.sum()
    if total <= 0:
        return probs.tolist()
    return (probs / total).tolist()


def parse_driver_code(participant_name: str) -> str:
    """Parse driver code from participant name."""
    name_lower = participant_name.lower().replace(" ", "_").replace("-", "_")
    
    for full_name, code in DRIVER_NAME_MAPPING.items():
        if full_name in name_lower or name_lower in full_name:
            return code
    
    words = participant_name.split()
    if len(words) >= 2:
        initials = "".join(w[0].upper() for w in words[:2])
        if len(initials) == 2:
            return initials
    
    return participant_name[:3].upper()


def extract_race_id(event_name: str) -> str:
    """Extract race ID from event name like 'Formula 1 Bahrain Grand Prix 2024'."""
    parts = event_name.lower().split()
    year = None
    circuit = []
    
    for part in parts:
        if part.isdigit() and len(part) == 4:
            year = part
        elif part not in ["formula", "grand", "prix", "sprint", "race"]:
            circuit.append(part)
    
    if not year:
        year = str(datetime.now().year)
    
    circuit_name = "_".join(circuit[:2])
    return f"{year}_{circuit_name[:10]}"


def fetch_odds(api_key: str, regions: List[str], markets: List[str]) -> List[dict]:
    """Fetch odds from The Odds API."""
    url = ODDS_API_BASE
    params = {
        "apiKey": api_key,
        "regions": ",".join(regions),
        "markets": ",".join(markets),
        "oddsFormat": "decimal",
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch odds: {e}")
        return []


def process_event(event: dict, markets: List[str]) -> List[dict]:
    """Process a single event and extract odds records."""
    records = []
    event_name = event.get("sport_title", event.get("home_team", "unknown"))
    event_id = event.get("id", "")
    commence_time = event.get("commence_time", "")
    
    hours_to_race = None
    if commence_time:
        try:
            from datetime import datetime, timedelta, timezone
            commence = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            hours_to_race = (commence - now).total_seconds() / 3600
        except:
            pass
    
    race_id = extract_race_id(event_name)
    
    for market_name, market_data in event.get("bookmakers", {}).items():
        if market_name.lower() != "pinnacle":
            continue
        
        for bookmaker_market in market_data.get("markets", []):
            market_key = bookmaker_market.get("key", "")
            if market_key not in markets:
                continue
            
            outcomes = bookmaker_market.get("outcomes", [])
            
            probabilities = []
            for outcome in outcomes:
                p = 1.0 / outcome.get("price", 0)
                if p > 0:
                    probabilities.append(p)
            
            probs_novig = remove_vig(probabilities)
            
            for i, outcome in enumerate(outcomes):
                driver_name = outcome.get("description", "")
                driver_code = parse_driver_code(driver_name)
                price = outcome.get("price", 0)
                
                if price <= 0:
                    continue
                
                record = {
                    "event_id": event_id,
                    "race_id": race_id,
                    "event_name": event_name,
                    "market": market_key,
                    "driver_code": driver_code,
                    "driver_name": driver_name,
                    "odd_decimal": price,
                    "p_implied_raw": 1.0 / price if price > 0 else 0,
                    "p_novig": probs_novig[i] if i < len(probs_novig) else 0,
                    "bookmaker": "pinnacle",
                    "commence_time": commence_time,
                    "hours_to_race": hours_to_race,
                    "fetched_at": datetime.utcnow().isoformat() + "Z",
                    "source": "the_odds_api",
                }
                records.append(record)
    
    return records


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Import Pinnacle odds to MongoDB")
    parser.add_argument("--markets", type=str, default=os.environ.get("MARKETS", "h2h,outrights"))
    parser.add_argument("--regions", type=str, default=os.environ.get("REGIONS", "eu"))
    args = parser.parse_args()
    
    markets = [m.strip() for m in args.markets.split(",")]
    regions = [r.strip() for r in args.regions.split(",")]
    
    print("=" * 60)
    print("PINNACLE ODDS IMPORTER")
    print("=" * 60)
    print(f"Markets: {markets}")
    print(f"Regions: {regions}")
    print()
    
    api_key = os.environ.get("THE_ODDS_API_KEY")
    if not api_key:
        print("[ERROR] THE_ODDS_API_KEY environment variable is required")
        sys.exit(1)
    
    try:
        db = get_mongo_client()
        print("Connected to MongoDB")
        
        db.f1_pinnacle_odds.create_index([("race_id", 1), ("driver_code", 1), ("market", 1)])
        db.f1_pinnacle_odds.create_index([("fetched_at", 1)])
        db.f1_pinnacle_odds.create_index(
            [("fetched_at", 1)],
            expireAfterSeconds=7776000
        )
        
        events = fetch_odds(api_key, regions, markets)
        print(f"Fetched {len(events)} events from Odds API")
        
        if not events:
            print("[WARNING] No events returned. This might be due to:")
            print("  - No active F1 markets")
            print("  - API rate limit exceeded (500 req/month free tier)")
            print("  - Invalid API key")
            sys.exit(0)
        
        operations = []
        total_records = 0
        
        for event in events:
            records = process_event(event, markets)
            for record in records:
                operations.append(InsertOne(record))
                total_records += 1
            
            if len(operations) >= 1000:
                db.f1_pinnacle_odds.insert_many(operations)
                operations = []
        
        if operations:
            db.f1_pinnacle_odds.insert_many(operations)
        
        db.f1_import_log.update_one(
            {
                "source": "the_odds_api",
                "type": "odds",
            },
            {
                "$set": {
                    "imported_at": datetime.utcnow().isoformat() + "Z",
                    "events_count": len(events),
                    "records_count": total_records,
                }
            },
            upsert=True
        )
        
        print(f"\n[SUCCESS] Imported {total_records} odds records from {len(events)} events")
        print("=" * 60)
        
    except PyMongoError as e:
        print(f"\n[MongoDB Error] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
