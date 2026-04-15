#!/usr/bin/env python3
"""
Feature Selection Pipeline - Optimize for predictive accuracy.
Uses proper TimeSeriesSplit with fold-by-fold analysis.
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

def load_data(path="artifacts_v3/feature_matrix.csv"):
    df = pd.read_csv(path)
    if TARGET in df.columns:
        df = df.dropna(subset=[TARGET])
    return df

def get_feature_categories():
    """Define feature categories based on importance analysis."""
    return {
        "keep_high": [
            "lap_time_delta_pct", "best_lap_delta_pct",
            "total_sector_delta_ms", "s2_delta_ms", "s3_delta_ms", "s1_delta_ms",
            "pace_delta_ms", "total_valid_laps"
        ],
        "keep_medium": [
            "lap_consistency", "avg_lap_ms", "min_lap_ms", "std_lap_ms",
            "total_best_ms", "s2_best_ms", "s3_best_ms", "s1_best_ms",
            "grid_position", "medium_pct"
        ],
        "remove": [
            "max_tyre_life", "pb_rate", "personal_bests", "soft_pct",
            "hard_pct", "avg_tyre_life", "year", "round"
        ],
        "circuit_profile": [
            "top_speed_kmh", "avg_speed_kmh", "full_throttle_pct",
            "avg_fast_corner_kmh", "avg_medium_corner_kmh", "avg_slow_corner_kmh",
            "min_speed_kmh"
        ]
    }

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
    """Compare different feature subsets."""
    df = load_data()
    categories = get_feature_categories()
    
    print("=" * 60)
    print("FEATURE SELECTION - TIME SERIES CV")
    print("=" * 60)
    print(f"Dataset: {len(df)} records, target mean: {df[TARGET].mean():.1f}")
    print()
    
    results = []
    
    # 1. High importance only (8 features)
    print("--- Test 1: High Importance Features ---")
    high = categories["keep_high"]
    mae, std, n = evaluate_features(df, high, f"Top {len([f for f in high if f in df.columns])}")
    results.append(("High importance", mae, std, n))
    
    # 2. High + Medium (18 features)  
    print("\n--- Test 2: High + Medium Importance ---")
    medium = categories["keep_medium"]
    combined = list(set(high + medium))
    mae, std, n = evaluate_features(df, combined, f"High+Medium")
    results.append(("High + Medium", mae, std, n))
    
    # 3. All numeric (baseline)
    print("\n--- Test 3: All Numeric Features ---")
    all_numeric = [c for c in df.columns if c != TARGET and df[c].dtype in ['float64', 'int64']]
    mae, std, n = evaluate_features(df, all_numeric, f"All {len(all_numeric)}")
    results.append(("All features", mae, std, n))
    
    # 4. Remove zero-importance
    print("\n--- Test 4: Remove Zero-Importance ---")
    to_remove = ["max_tyre_life", "pb_rate", "personal_bests", "soft_pct", "hard_pct", 
                  "avg_tyre_life", "year", "round", "full_throttle_pct", "ct_street",
                  "avg_speed_kmh", "avg_medium_corner_kmh", "ct_desert", "ct_high_speed",
                  "min_speed_kmh", "avg_slow_corner_kmh"]
    filtered = [c for c in all_numeric if c not in to_remove]
    mae, std, n = evaluate_features(df, filtered, f"Filtered {len(filtered)}")
    results.append(("Filtered", mae, std, n))
    
    # 5. Top 10 by importance
    print("\n--- Test 5: Top 10 Features ---")
    top10 = ["lap_time_delta_pct", "best_lap_delta_pct", "total_sector_delta_ms", 
             "s2_delta_ms", "s3_delta_ms", "s1_delta_ms", "pace_delta_ms", 
             "total_valid_laps", "lap_consistency", "avg_lap_ms"]
    mae, std, n = evaluate_features(df, top10, "Top 10")
    results.append(("Top 10", mae, std, n))
    
    return results

def identify_optimal():
    """Find optimal feature set."""
    print("\n" + "=" * 60)
    print("OPTIMAL FEATURE SET")
    print("=" * 60)
    
    results = run_comparison()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
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
    
    return best_name

if __name__ == "__main__":
    identify_optimal()