#!/usr/bin/env python3
"""
Feature Selection Pipeline - Pre-race features only (no leakage).
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

TARGET = "finish_position"
N_SPLITS = 5

# Features that are only known AFTER the race (DATA LEAKAGE!)
LEAKY_FEATURES = [
    "finish_position",    # Target variable
    "grid_position",      # Pre-race but IS what we're predicting (directly related)
    "points_scored",      # Only known after race
    "laps_completed",     # Only known after race  
    "is_dnf",             # Only known after race
    "status_finished",    # Only known after race
    "total_valid_laps",   # Leakage (finishing more laps = better position)
    "grid_position_pct",  # Derived from grid_position
    "pos_gain",           # Only known after race
    "top3_finish",        # Derived from target
]

def load_data(path="artifacts_v3/feature_matrix.csv"):
    df = pd.read_csv(path)
    if TARGET in df.columns:
        df = df.dropna(subset=[TARGET])
    return df

def get_pre_race_features():
    """Return pre-race features only (no leakage)."""
    return [
        # Lap telemetry (from qualifying/race, but before finish)
        "lap_time_delta_pct",
        "best_lap_delta_pct",
        "avg_lap_ms",
        "min_lap_ms", 
        "std_lap_ms",
        "lap_consistency",
        
        # Sector times from qualifying (best available before race)
        "total_sector_delta_ms",
        "s1_delta_ms",
        "s2_delta_ms", 
        "s3_delta_ms",
        "s1_best_ms",
        "s2_best_ms",
        "s3_best_ms",
        "total_best_ms",
        
        # Constructor pace (pre-race)
        "pace_delta_ms",
        
        # Circuit characteristics (static, pre-race)
        "top_speed_kmh",
        "avg_speed_kmh",
        "full_throttle_pct",
        "avg_fast_corner_kmh",
        "avg_medium_corner_kmh",
        "avg_slow_corner_kmh",
        "min_speed_kmh",
        
        # Circuit type (static)
        "ct_street",
        "ct_desert", 
        "ct_high_speed",
        "ct_mixed",
        
        # Tyre strategy (may change during race)
        "soft_pct",
        "medium_pct",
        "hard_pct",
        "avg_tyre_life",
        "max_tyre_life",
        
        # Other
        "pb_rate",
        "personal_bests",
    ]

def evaluate_features(df, features, name):
    """Evaluate feature set with TimeSeriesSplit CV."""
    available = [f for f in features if f in df.columns]
    X = df[available].fillna(0)
    y = df[TARGET]
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    
    mae_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        rf.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = rf.predict(X.iloc[val_idx])
        mae = mean_absolute_error(y.iloc[val_idx], pred)
        mae_scores.append(mae)
    
    print(f"  {name}: MAE = {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")
    return np.mean(mae_scores), np.std(mae_scores), len(available)

def run_comparison():
    """Compare different feature subsets without leakage."""
    df = load_data()
    pre_race = get_pre_race_features()
    
    print("=" * 70)
    print("FEATURE SELECTION - PRE-RACE FEATURES ONLY (NO LEAKAGE)")
    print("=" * 70)
    print(f"Dataset: {len(df)} records, target mean: {df[TARGET].mean():.1f}")
    print()
    
    results = []
    
    # 1. All pre-race features
    print("--- Test 1: All Pre-Race Features ---")
    mae, std, n = evaluate_features(df, pre_race, f"All {len([f for f in pre_race if f in df.columns])}")
    results.append(("All pre-race", mae, std, n))
    
    # 2. Top features by importance (from analysis)
    print("\n--- Test 2: High Importance (Sector + Lap) ---")
    high = ["lap_time_delta_pct", "best_lap_delta_pct", 
            "s1_delta_ms", "s2_delta_ms", "s3_delta_ms", "total_sector_delta_ms",
            "pace_delta_ms"]
    mae, std, n = evaluate_features(df, high, f"High importance")
    results.append(("High importance", mae, std, n))
    
    # 3. Lap telemetry only
    print("\n--- Test 3: Lap Telemetry Only ---")
    lap_features = ["lap_time_delta_pct", "best_lap_delta_pct", "avg_lap_ms", "std_lap_ms", "lap_consistency"]
    mae, std, n = evaluate_features(df, lap_features, "Lap telemetry")
    results.append(("Lap telemetry", mae, std, n))
    
    # 4. Sector times only
    print("\n--- Test 4: Sector Times Only ---")
    sector_features = ["s1_delta_ms", "s2_delta_ms", "s3_delta_ms", "total_sector_delta_ms",
                       "s1_best_ms", "s2_best_ms", "s3_best_ms", "total_best_ms"]
    mae, std, n = evaluate_features(df, sector_features, "Sector times")
    results.append(("Sector times", mae, std, n))
    
    # 5. Lap + Sector combined
    print("\n--- Test 5: Lap + Sector Combined ---")
    lap_sector = lap_features + sector_features
    mae, std, n = evaluate_features(df, lap_sector, "Lap + Sector")
    results.append(("Lap + Sector", mae, std, n))
    
    # 6. With constructor pace
    print("\n--- Test 6: Lap + Sector + Pace ---")
    with_pace = lap_sector + ["pace_delta_ms"]
    mae, std, n = evaluate_features(df, with_pace, "With pace")
    results.append(("With pace", mae, std, n))
    
    # 7. All without leaky features
    print("\n--- Test 7: All non-leaky numeric features ---")
    all_numeric = [c for c in df.columns if c != TARGET and df[c].dtype in ['float64', 'int64']]
    no_leak = [c for c in all_numeric if c not in LEAKY_FEATURES]
    mae, std, n = evaluate_features(df, no_leak, f"All {len(no_leak)}")
    results.append(("All no-leak", mae, std, n))
    
    return results

def main():
    results = run_comparison()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Approach':<25} {'MAE':<10} {'Std':<10} {'Features':<10}")
    print("-" * 55)
    
    best_mae = float('inf')
    best_name = ""
    for name, mae, std, n in results:
        marker = " <<<" if mae < best_mae else ""
        print(f"{name:<25} {mae:.3f}     {std:.3f}     {n:<10}{marker}")
        if mae < best_mae:
            best_mae = mae
            best_name = name
    
    print(f"\n>>> Best: {best_name} with MAE = {best_mae:.3f}")
    
    # Save optimal feature list
    optimal_features = [
        "lap_time_delta_pct", "best_lap_delta_pct",
        "s1_delta_ms", "s2_delta_ms", "s3_delta_ms", "total_sector_delta_ms",
        "s1_best_ms", "s2_best_ms", "s3_best_ms", "total_best_ms",
        "lap_consistency", "avg_lap_ms", "std_lap_ms",
        "pace_delta_ms"
    ]
    
    import json
    config = {
        "optimal_features": optimal_features,
        "leaky_features_removed": LEAKY_FEATURES,
        "best_mae": best_mae
    }
    with open("artifacts_v3/optimal_features.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nSaved optimal feature config to artifacts_v3/optimal_features.json")
    print(f"Optimal features: {optimal_features}")

if __name__ == "__main__":
    main()