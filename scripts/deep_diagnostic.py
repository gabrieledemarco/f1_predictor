#!/usr/bin/env python3
"""
Deep diagnostic: Analyze model features and data effectiveness.
Investigates why Kendall tau is low despite data being available.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from pymongo import MongoClient

mongo_uri = os.environ.get("MONGODB_URI")
if not mongo_uri:
    print("ERROR: MONGODB_URI not set")
    sys.exit(1)

db = MongoClient(mongo_uri)["betbreaker"]

print("=" * 70)
print("DEEP DIAGNOSTIC: Model Feature Effectiveness")
print("=" * 70)

# ============================================================================
# 1. Analyze pace data quality
# ============================================================================
print("\n[1] PACE DATA QUALITY")
print("-" * 50)

# Get pace observations for 2023
pace_2023 = list(db.f1_pace_observations.find({"year": 2023}))

# Group by constructor
pace_by_team = defaultdict(list)
for doc in pace_2023:
    pace_by_team[doc["constructor_ref"]].append(doc["pace_delta_ms"])

print("Constructor pace deltas (s/lap) - 2023 season:")
print(f"{'Team':<18} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Count':>6}")
print("-" * 60)
for team in sorted(pace_by_team.keys()):
    vals = pace_by_team[team]
    if len(vals) > 0:
        print(f"{team:<18} {np.mean(vals):+.3f} {np.std(vals):.3f} {np.min(vals):+.3f} {np.max(vals):+.3f} {len(vals):>6}")

# Check range of pace differences (should be informative)
all_pace_deltas = [v for vs in pace_by_team.values() for v in vs]
print(f"\nOverall pace delta range: {min(all_pace_deltas):.3f} to {max(all_pace_deltas):.3f} s/lap")
print(f"Std of pace deltas: {np.std(all_pace_deltas):.3f} s/lap")

# ============================================================================
# 2. Analyze driver skill differentiation
# ============================================================================
print("\n[2] DRIVER SKILL DIFFERENTIATION")
print("-" * 50)

# Get race results for 2023
results_2023 = list(db.f1_races.find({"year": 2023}))

# Analyze finishing positions by driver
driver_positions = defaultdict(list)
driver_wins = defaultdict(int)
driver_points = defaultdict(float)

for race in results_2023:
    for result in race.get("results", []):
        driver = result["driver_code"]
        pos = result.get("finish_position")
        pts = result.get("points", 0)
        
        if pos:
            driver_positions[driver].append(pos)
        driver_points[driver] += pts
        if pos == 1:
            driver_wins[driver] += 1

print("Driver performance 2023:")
print(f"{'Driver':<8} {'Avg Pos':>8} {'Wins':>6} {'Points':>8} {'Races':>6}")
print("-" * 40)
for driver in sorted(driver_positions.keys(), key=lambda x: driver_points[x], reverse=True)[:10]:
    positions = driver_positions[driver]
    print(f"{driver:<8} {np.mean(positions):.1f} {driver_wins[driver]:>6} {driver_points[driver]:>8.0f} {len(positions):>6}")

# ============================================================================
# 3. Check if grid position is dominating predictions
# ============================================================================
print("\n[3] GRID POSITION IMPACT")
print("-" * 50)

# Compare grid vs finish for 2023
grid_vs_finish = []
for race in results_2023:
    for result in race.get("results", []):
        grid = result.get("grid_position", 0)
        finish = result.get("finish_position")
        if grid and finish:
            grid_vs_finish.append({
                "driver": result["driver_code"],
                "grid": grid,
                "finish": finish,
                "delta": grid - finish  # positive = improved
            })

if grid_vs_finish:
    # Group by grid position
    grid_groups = defaultdict(list)
    for g in grid_vs_finish:
        grid_groups[g["grid"]].append(g["delta"])
    
    print("Average position gain/loss by starting grid:")
    print(f"{'Grid':>6} {'Avg Delta':>10} {'Count':>6}")
    print("-" * 25)
    for grid in sorted(grid_groups.keys())[:10]:
        deltas = grid_groups[grid]
        print(f"{grid:>6} {np.mean(deltas):>+10.1f} {len(deltas):>6}")

# Correlation between grid and finish
grids = [g["grid"] for g in grid_vs_finish]
finishes = [g["finish"] for g in grid_vs_finish]
if len(grids) > 5:
    corr = np.corrcoef(grids, finishes)[0, 1]
    print(f"\nGrid-Finish correlation: {corr:.3f}")
    print("(If high, model might be over-relying on grid position)")

# ============================================================================
# 4. Check race results vs predictions - who actually wins?
# ============================================================================
print("\n[4] ACTUAL RACE OUTCOMES 2023")
print("-" * 50)

# Count how often each team wins
wins_by_team = defaultdict(int)
for race in results_2023:
    for result in race.get("results", []):
        if result.get("finish_position") == 1:
            wins_by_team[result["constructor_ref"]] += 1

print("Race wins by constructor:")
for team, wins in sorted(wins_by_team.items(), key=lambda x: -x[1]):
    print(f"  {team:<18} {wins:>2} wins")

# ============================================================================
# 5. Check lap times data quality
# ============================================================================
print("\n[5] LAP TIMES DATA QUALITY")
print("-" * 50)

# Check lap times per race in 2023
lap_counts = []
for race in db.f1_races.find({"year": 2023}):
    round_num = race.get("round")
    lap_count = db.f1_lap_times.count_documents({"year": 2023, "round": round_num})
    lap_counts.append((round_num, lap_count))

print("Lap times per race 2023:")
print(f"{'Round':>6} {'Laps':>8}")
print("-" * 15)
for r, c in lap_counts:
    print(f"{r:>6} {c:>8}")

# Check if there are lap times for each race
avg_laps = np.mean([c for _, c in lap_counts])
print(f"\nAverage lap times per race: {avg_laps:.0f}")
print(f"(Should be ~20 drivers * ~50 laps = 1000+)")

# ============================================================================
# 6. Recommendations
# ============================================================================
print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

print("""
Based on analysis:

1. PACE DATA: If pace deltas are too small (e.g., < 0.1s range), 
   the Kalman filter won't differentiate constructors well.
   → Fix: Rerun compute_pace_observations.py after pace formula fix

2. GRID POSITION: If grid-finish correlation is very high (>0.7),
   the model is essentially predicting "who starts front, finishes front"
   → Fix: Add grid penalty feature, weight skill higher

3. TRUE SKILL: If driver ratings have low variance, tau will be low
   → Fix: Increase tau (process noise) to allow faster skill adaptation

4. KALMAN FILTER: If Q (process noise) is too high, observations 
   are ignored; if too low, model doesn't adapt to new data
   → Fix: Tune KalmanConfig parameters

5. FEATURES: The h2h_win_rate, elo_delta, dnf_rate features 
   are defined but may not be computed
   → Fix: Verify they're being populated in ensemble.py
""")

print("=" * 70)