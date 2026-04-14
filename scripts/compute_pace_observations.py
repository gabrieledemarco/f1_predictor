#!/usr/bin/env python3
"""
Compute Constructor Pace Observations

Aggregates lap times from f1_lap_times collection and computes
constructor pace observations for the ML pipeline.

Usage:
    python scripts/compute_pace_observations.py [--year YEAR] [--source SOURCE]
    python scripts/compute_pace_observations.py --min-year 2019 --max-year 2025
"""

import os
import sys
from datetime import datetime
from typing import Dict, List

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


def get_driver_constructor_mapping(db) -> Dict[str, str]:
    """Get driver_code -> constructor_ref mapping from f1_races results."""
    mapping = {}
    cursor = db.f1_races.find(
        {},
        {"year": 1, "results.driver_code": 1, "results.constructor_ref": 1}
    )
    for doc in cursor:
        year = doc.get("year", 0)
        for result in doc.get("results", []):
            driver_code = result.get("driver_code", "")
            constructor_ref = result.get("constructor_ref", "")
            if driver_code and constructor_ref:
                key = f"{year}_{driver_code}"
                mapping[key] = constructor_ref
    return mapping


def compute_pace_observations(db, years: List[int], source: str = "") -> int:
    """Compute pace observations for each constructor per race."""
    print("Building driver->constructor mapping...")
    driver_constructor = get_driver_constructor_mapping(db)
    print(f"  Loaded {len(driver_constructor)} driver-constructor mappings")

    match_query: dict = {"year": {"$in": years}}
    if source:
        match_query["source"] = source

    print(f"Fetching lap times for years {years}...")
    cursor = db.f1_lap_times.find(match_query, {
        "year": 1, "round": 1, "circuit_ref": 1,
        "driver_code": 1, "lap_time_ms": 1
    })

    lap_data = []
    skipped = 0
    for doc in cursor:
        year = doc.get("year", 0)
        driver_code = doc.get("driver_code", "")
        key = f"{year}_{driver_code}"
        constructor_ref = driver_constructor.get(key)

        if not constructor_ref:
            skipped += 1
            continue

        lap_data.append({
            "year": year,
            "round": doc.get("round", 0),
            "circuit_ref": doc.get("circuit_ref", ""),
            "constructor_ref": constructor_ref,
            "lap_time_ms": doc.get("lap_time_ms", 0),
        })

    print(f"  Fetched {len(lap_data)} laps ({skipped} skipped - no constructor mapping)")

    if not lap_data:
        print("[WARNING] No valid lap data found")
        return 0

    df = pd.DataFrame(lap_data)

    agg = df.groupby(["year", "round", "circuit_ref", "constructor_ref"]).agg(
        avg_pace=("lap_time_ms", "mean"),
        min_pace=("lap_time_ms", "min"),
        sample_size=("lap_time_ms", "count"),
    ).reset_index()

    print(f"  Aggregated to {len(agg)} constructor-race combinations")

    constructor_medians = agg.groupby(["year", "constructor_ref"])["avg_pace"].median()

    operations = []
    computed = 0

    for _, row in agg.iterrows():
        year = row["year"]
        team = row["constructor_ref"]

        team_median = float(constructor_medians.get((year, team), row["avg_pace"]) or row["avg_pace"])

        if team_median and team_median > 0:
            pace_delta = (row["avg_pace"] - team_median) / 1000
        else:
            pace_delta = 0.0

        doc = {
            "_id": f"{year}_{int(row['round']):02d}_{team}",
            "year": int(year),
            "round": int(row["round"]),
            "circuit_ref": row["circuit_ref"],
            "constructor_ref": team,
            "pace_delta_ms": float(pace_delta),
            "avg_pace_ms": float(row["avg_pace"]),
            "min_pace_ms": float(row["min_pace"]),
            "sample_size": int(row["sample_size"]),
            "computed_at": datetime.utcnow().isoformat() + "Z",
            "source": source or "mixed",
        }

        operations.append(UpdateOne(
            {"_id": doc["_id"]},
            {"$set": doc},
            upsert=True
        ))

        if len(operations) >= 500:
            result = db.f1_pace_observations.bulk_write(operations, ordered=False)
            computed += result.upserted_count + result.modified_count
            operations = []

    if operations:
        result = db.f1_pace_observations.bulk_write(operations, ordered=False)
        computed += result.upserted_count + result.modified_count

    return computed


def main():
    import argparse
    current_year = datetime.now().year
    parser = argparse.ArgumentParser(description="Compute constructor pace observations")
    parser.add_argument("--year",     type=int, default=None,
                        help="Anno singolo (override min/max)")
    parser.add_argument("--min-year", type=int,
                        default=int(os.environ.get("MIN_YEAR", os.environ.get("YEAR", "2019"))),
                        help="Anno minimo (default: 2019)")
    parser.add_argument("--max-year", type=int,
                        default=int(os.environ.get("MAX_YEAR", str(current_year))),
                        help="Anno massimo (default: anno corrente)")
    parser.add_argument("--source", type=str, default=None,
                        help="Source: tracinginsights, kaggle, or None for all")
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
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
