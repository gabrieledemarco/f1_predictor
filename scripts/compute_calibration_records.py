#!/usr/bin/env python3
"""
Compute Calibration Records

Creates calibration records by joining Pinnacle odds with actual race results.
These records are used by Layer 4 (Isotonic Calibration) to calibrate model probabilities.

Usage:
    python scripts/compute_calibration_records.py
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_mongo_client():
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is required")
    
    mongo_db = os.environ.get("MONGO_DB", "betbreaker")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    return client[mongo_db]


def get_race_result(db, race_id: str) -> Dict[str, int]:
    """
    Get actual race results as driver_code -> position mapping.
    Only includes drivers who finished (position <= 20).
    """
    race = db.f1_races.find_one({"_id": race_id})
    if not race:
        return {}
    
    results = {}
    for result in race.get("results", []):
        position = result.get("finish_position")
        if position and position <= 20:
            results[result["driver_code"]] = position
    
    return results


def get_race_date(db, race_id: str) -> Optional[str]:
    """Get the date of a race."""
    race = db.f1_races.find_one({"_id": race_id}, {"date": 1})
    return race.get("date") if race else None


def is_race_completed(race_date: str) -> bool:
    """Check if a race has been completed (date is in the past)."""
    if not race_date:
        return False
    
    try:
        race_datetime = datetime.strptime(race_date, "%Y-%m-%d")
        return race_datetime < datetime.now()
    except ValueError:
        return False


def compute_calibration_records(
    db,
    min_hours_to_race: float = 2.0,
    max_hours_to_race: float = 72.0,
) -> int:
    """
    Compute calibration records from odds and race results.
    
    A calibration record is created when:
    1. We have Pinnacle odds for a driver in a race
    2. The race has been completed
    3. The odds were captured between min and max hours before race
    
    Returns:
        Number of calibration records created
    """
    hours_cutoff = (datetime.now() - timedelta(hours=max_hours_to_race)).isoformat() + "Z"
    
    pipeline = [
        {
            "$match": {
                "hours_to_race": {
                    "$gte": min_hours_to_race,
                    "$lte": max_hours_to_race
                },
                "fetched_at": {"$gte": hours_cutoff},
            }
        },
        {
            "$group": {
                "_id": {
                    "race_id": "$race_id",
                    "driver_code": "$driver_code",
                },
                "p_novig": {"$first": "$p_novig"},
                "fetched_at": {"$first": "$fetched_at"},
                "hours_to_race": {"$first": "$hours_to_race"},
            }
        }
    ]
    
    odds_records = list(db.f1_pinnacle_odds.aggregate(pipeline))
    
    if not odds_records:
        log.warning("No odds records found in the specified time range")
        return 0
    
    log.info(f"Found {len(odds_records)} odds records to process")
    
    operations = []
    created = 0
    
    for record in odds_records:
        race_id = record["_id"]["race_id"]
        driver_code = record["_id"]["driver_code"]
        
        race_date = get_race_date(db, race_id)
        if not race_date:
            continue
        
        if not is_race_completed(race_date):
            continue
        
        race_results = get_race_result(db, race_id)
        if driver_code not in race_results:
            continue
        
        actual_position = race_results[driver_code]
        outcome = 1 if actual_position <= 1 else 0
        
        calib_doc = {
            "_id": f"{race_id}_{driver_code}_h2h",
            "race_id": race_id,
            "driver_code": driver_code,
            "market": "h2h",
            "p_model_raw": record["p_novig"],
            "p_pinnacle_novig": record["p_novig"],
            "outcome": outcome,
            "actual_position": actual_position,
            "odds_fetched_at": record["fetched_at"],
            "race_date": race_date,
            "hours_to_race": record.get("hours_to_race", 0),
            "computed_at": datetime.utcnow().isoformat() + "Z",
        }
        
        operations.append(UpdateOne(
            {"_id": calib_doc["_id"]},
            {"$set": calib_doc},
            upsert=True
        ))
        
        if len(operations) >= 1000:
            result = db.f1_calibration_records.bulk_write(operations, ordered=False)
            created += result.upserted_count
            operations = []
    
    if operations:
        result = db.f1_calibration_records.bulk_write(operations, ordered=False)
        created += result.upserted_count
    
    return created


def compute_pod_odds_calibration(db) -> int:
    """
    Compute calibration records for podium (P(1-3)) market.
    """
    hours_cutoff = (datetime.now() - timedelta(hours=72)).isoformat() + "Z"
    
    pipeline = [
        {
            "$match": {
                "market": "outrights",
                "hours_to_race": {"$gte": 2.0, "$lte": 72.0},
                "fetched_at": {"$gte": hours_cutoff},
            }
        },
        {
            "$group": {
                "_id": {
                    "race_id": "$race_id",
                    "driver_code": "$driver_code",
                },
                "p_novig": {"$first": "$p_novig"},
            }
        }
    ]
    
    odds_records = list(db.f1_pinnacle_odds.aggregate(pipeline))
    
    operations = []
    created = 0
    
    for record in odds_records:
        race_id = record["_id"]["race_id"]
        driver_code = record["_id"]["driver_code"]
        
        race_date = get_race_date(db, race_id)
        if not race_date or not is_race_completed(race_date):
            continue
        
        race_results = get_race_result(db, race_id)
        if driver_code not in race_results:
            continue
        
        actual_position = race_results[driver_code]
        outcome = 1 if actual_position <= 3 else 0
        
        calib_doc = {
            "_id": f"{race_id}_{driver_code}_podium",
            "race_id": race_id,
            "driver_code": driver_code,
            "market": "podium",
            "p_model_raw": record["p_novig"],
            "p_pinnacle_novig": record["p_novig"],
            "outcome": outcome,
            "actual_position": actual_position,
            "computed_at": datetime.utcnow().isoformat() + "Z",
        }
        
        operations.append(UpdateOne(
            {"_id": calib_doc["_id"]},
            {"$set": calib_doc},
            upsert=True
        ))
    
    if operations:
        result = db.f1_calibration_records.bulk_write(operations, ordered=False)
        created += result.upserted_count
    
    return created


def main():
    log.info("=" * 60)
    log.info("CALIBRATION RECORDS COMPUTATION")
    log.info("=" * 60)
    
    try:
        db = get_mongo_client()
        log.info("Connected to MongoDB")
        
        db.f1_calibration_records.create_index([("race_id", 1), ("driver_code", 1)])
        db.f1_calibration_records.create_index([("market", 1)])
        db.f1_calibration_records.create_index([("computed_at", 1)])
        
        h2h_created = compute_calibration_records(db)
        log.info(f"H2H calibration records created: {h2h_created}")
        
        podium_created = compute_pod_odds_calibration(db)
        log.info(f"Podium calibration records created: {podium_created}")
        
        total = db.f1_calibration_records.count_documents({})
        log.info(f"\nTotal calibration records: {total}")
        
        log.info("=" * 60)
        log.info("COMPLETED")
        log.info("=" * 60)
        
    except PyMongoError as e:
        log.error(f"MongoDB Error: {e}")
        sys.exit(1)
    except Exception as e:
        log.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
