#!/usr/bin/env python3
"""
Compute Constructor Pace Observations

Aggregates lap times from f1_lap_times collection and computes
constructor pace observations for the ML pipeline.

Usage:
    python scripts/compute_pace_observations.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
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


def get_driver_teams(db) -> Dict[str, str]:
    """Get driver to constructor mapping from race results."""
    mapping = {}
    cursor = db.f1_races.find({}, {"results.driver_code": 1, "results.constructor_ref": 1, "year": 1})
    for doc in cursor:
        for result in doc.get("results", []):
            key = f"{doc['year']}_{result['driver_code']}"
            mapping[key] = result.get("constructor_ref", "")
    return mapping


def compute_pace_observations(db, years: List[int], source: str = None) -> int:
    """Compute pace observations for each constructor per race."""
    match_stage = {"year": {"$in": years}, "is_valid": True}
    
    pipeline = [
        {"$match": match_stage},
        {"$group": {
            "_id": {
                "year": "$year",
                "round": "$round",
                "circuit_ref": "$circuit_ref",
                "team": "$team",
            },
            "avg_pace": {"$avg": "$fuel_corrected_ms"},
            "min_pace": {"$min": "$fuel_corrected_ms"},
            "sample_size": {"$sum": 1},
        }},
        {"$sort": {"_id.year": 1, "_id.round": 1}}
    ]
    
    if source:
        match_stage["source"] = source
        pipeline[0]["$match"] = match_stage
    
    results = list(db.f1_lap_times.aggregate(pipeline))

    if not results:
        print("[WARNING] No valid lap data found")
        return 0

    # json_normalize flattens nested _id dict → columns: _id.year, _id.round, _id.circuit_ref, _id.team
    df = pd.json_normalize(results)

    constructor_medians = df.groupby(["_id.year", "_id.team"])["avg_pace"].median()

    operations = []
    computed = 0

    for _, row in df.iterrows():
        year = int(row["_id.year"])
        team = row["_id.team"]

        team_median = constructor_medians.get((year, team), row["avg_pace"])

        if team_median > 0:
            pace_delta = (row["avg_pace"] - team_median) / 1000
        else:
            pace_delta = 0.0

        doc = {
            "_id": f"{year}_{int(row['_id.round']):02d}_{team}",
            "year": year,
            "round": int(row["_id.round"]),
            "circuit_ref": row["_id.circuit_ref"],
            "constructor_ref": team,
            "pace_delta_ms": pace_delta,
            "avg_pace_ms": row["avg_pace"],
            "min_pace_ms": row["min_pace"],
            "sample_size": int(row["sample_size"]),
            "computed_at": datetime.utcnow().isoformat() + "Z",
            "source": source or "mixed",
        }
        
        operations.append(UpdateOne(
            {"_id": doc["_id"]},
            {"$set": doc},
            upsert=True
        ))
        
        if len(operations) >= 1000:
            db.f1_pace_observations.bulk_write(operations, ordered=False)
            computed += len(operations)
            operations = []
    
    if operations:
        db.f1_pace_observations.bulk_write(operations, ordered=False)
        computed += len(operations)
    
    return computed


def main():
    import argparse
    current_year = datetime.now().year
    parser = argparse.ArgumentParser(description="Compute constructor pace observations")
    parser.add_argument("--year",     type=int, default=None, help="Anno singolo (deprecated: usa --min-year/--max-year)")
    parser.add_argument("--min-year", type=int, default=int(os.environ.get("MIN_YEAR", os.environ.get("YEAR", "2019"))),
                        help="Anno minimo (default: 2019)")
    parser.add_argument("--max-year", type=int, default=int(os.environ.get("MAX_YEAR", str(current_year))),
                        help="Anno massimo (default: anno corrente)")
    parser.add_argument("--source", type=str, default=None, help="Source: tracinginsights, kaggle, or None for all")
    args = parser.parse_args()

    if args.year:
        years = [args.year]
    else:
        years = list(range(args.min_year, args.max_year + 1))
    
    print("=" * 60)
    print("CONSTRUCTOR PACE COMPUTATION")
    print("=" * 60)
    print(f"Years: {years}")
    print(f"Source: {args.source or 'all'}")
    print()
    
    try:
        db = get_mongo_client()
        print("Connected to MongoDB")
        
        db.f1_pace_observations.create_index([
            ("year", 1), ("round", 1), ("constructor_ref", 1)
        ], unique=True)
        db.f1_pace_observations.create_index([("circuit_ref", 1)])
        
        computed = compute_pace_observations(db, years, args.source)
        
        print("\n" + "=" * 60)
        print(f"COMPLETED: {computed} pace observations computed")
        print("=" * 60)
        
    except PyMongoError as e:
        print(f"\n[MongoDB Error] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
