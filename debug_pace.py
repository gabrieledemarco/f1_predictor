#!/usr/bin/env python3
"""
Debug script to inspect constructor_pace_observations loaded from Kaggle data.
"""

import sys
sys.path.insert(0, '.')

from f1_predictor.data import load_training_data

# Load a small subset of years
races = load_training_data(
    years=range(2023, 2024),
    through_round=None,
    jolpica_cache="data/cache/jolpica",
    tracinginsights_dir="data/racedata",
    force_refresh=False,
    use_synthetic_fallback=False,
)

print(f"Loaded {len(races)} races")
for i, race in enumerate(races[:5]):  # first 5 races
    print(f"\nRace {i}: {race.get('year')} {race.get('circuit_ref')} Round {race.get('round')}")
    pace = race.get("constructor_pace_observations", {})
    if pace:
        print(f"  Pace observations ({len(pace)} constructors):")
        for const, delta in sorted(pace.items(), key=lambda x: x[1]):
            print(f"    {const}: {delta:.4f}")
    else:
        print("  NO pace data")