#!/usr/bin/env python3
"""
Diagnostic script to investigate poor Kendall tau performance.
Run after training to understand what's happening in the model.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from pymongo import MongoClient

# Connect to MongoDB
mongo_uri = os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URI")
if not mongo_uri:
    print("ERROR: MONGODB_URI not set")
    sys.exit(1)

db = MongoClient(mongo_uri)["betbreaker"]

print("=" * 70)
print("DIAGNOSTIC: Investigating Poor Kendall Tau Performance")
print("=" * 70)

# 1. Check race data
print("\n[1] Race Data Overview")
race_count = db.f1_races.count_documents({})
print(f"  Total races in f1_races: {race_count}")

# 2. Check pace observations
print("\n[2] Pace Observations Overview")
pace_count = db.f1_pace_observations.count_documents({})
print(f"  Total pace observations: {pace_count}")

# 3. Check constructor distribution
print("\n[3] Constructor Pace Distribution (2023)")
p2023 = list(db.f1_pace_observations.find({"year": 2023}))
constructors = defaultdict(list)
for doc in p2023:
    constructors[doc["constructor_ref"]].append(doc["pace_delta_ms"])

print("  Constructor | Avg Pace Delta (s/lap) | Count")
print("  " + "-" * 50)
for c in sorted(constructors.keys()):
    vals = constructors[c]
    avg = np.mean(vals)
    print(f"  {c:15} | {avg:+.3f} | {len(vals)} races")

# 4. Check driver results in 2023
print("\n[4] Driver Finishing Positions in 2023")
driver_positions = defaultdict(list)
for race in db.f1_races.find({"year": 2023}):
    for result in race.get("results", []):
        pos = result.get("finish_position")
        if pos:
            driver_positions[result["driver_code"]].append(pos)

print("  Driver | Avg Finish | Races | Wins")
print("  " + "-" * 45)
for d in sorted(driver_positions.keys(), key=lambda x: np.mean(driver_positions[x])):
    positions = driver_positions[d]
    wins = sum(1 for p in positions if p == 1)
    print(f"  {d:6} | {np.mean(positions):.1f} | {len(positions)} | {wins}")

# 5. Check if driver skill model has differentiation
print("\n[5] Sample Race - Check Model Inputs")
sample_race = db.f1_races.find_one({"year": 2023, "round": 1})
if sample_race:
    print(f"  Race: {sample_race.get('race_name', 'N/A')}")
    print(f"  Circuit: {sample_race.get('circuit_ref', 'N/A')}")
    
    # Check if pace data exists
    pace_obs = list(db.f1_pace_observations.find({"year": 2023, "round": 1}))
    print(f"  Pace observations: {len(pace_obs)}")
    if pace_obs:
        print("  Constructor pace deltas (s/lap):")
        for p in pace_obs:
            print(f"    {p['constructor_ref']:15} {p['pace_delta_ms']:+.3f}")

# 6. Check lap times quality
print("\n[6] Lap Times Data Quality")
for year in [2022, 2023]:
    count = db.f1_lap_times.count_documents({"year": year})
    print(f"  {year}: {count} lap records")

# Check if there are lap times for 2023 R1
lap_r1 = list(db.f1_lap_times.find({"year": 2023, "round": 1}).limit(5))
print(f"  Sample lap times 2023 R1: {len(lap_r1)}")
if lap_r1:
    print(f"    First lap: driver={lap_r1[0].get('driver_code')}, lap_time={lap_r1[0].get('lap_time_ms')}ms")

print("\n" + "=" * 70)
print("Diagnostic complete. Key issues to investigate:")
print("  1. Are driver skill ratings differentiated?")
print("  2. Is constructor pace adding signal?")
print("  3. Is the Monte Carlo simulation using features correctly?")
print("=" * 70)