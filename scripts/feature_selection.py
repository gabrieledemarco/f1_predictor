#!/usr/bin/env python3
"""
Feature Selection Pipeline - Optimize for predictive accuracy.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import mutual_info_regression, RFE
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

TARGET = "finish_position"
N_SPLITS = 5

def load_data(path="artifacts_v3/feature_matrix.csv"):
    df = pd.read_csv(path)
    # Drop rows where target is NaN (if any)
    if TARGET in df.columns:
        df = df.dropna(subset=[TARGET])
    return df

def get_feature_categories():
    """Define feature categories and selection strategy."""
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

def remove_zero_importance_features(df):
    """Remove features with zero composite score from rankings."""
    rankings = pd.read_csv("artifacts_v3/feature_rankings.csv")
    zero_features = rankings[rankings["composite_score"] < 0.001]["feature"].tolist()
    
    print(f"Removing {len(zero_features)} zero-importance features:")
    for f in zero_features:
        print(f"  - {f}")
        if f in df.columns:
            df = df.drop(columns=[f])
    return df

def remove_correlated_features(df, threshold=0.9):
    """Remove highly correlated features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != TARGET]
    
    corr_matrix = df[numeric_cols].corr().abs()
    
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"\nRemoving {len(to_drop)} highly correlated features (r>{threshold}):")
    for f in to_drop:
        print(f"  - {f}")
    
    return df.drop(columns=to_drop)

def compare_feature_subsets():
    """Compare model performance with different feature subsets."""
    df = load_data()
    categories = get_feature_categories()
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    
    results = []
    y = df[TARGET]
    
    # Test 1: High importance only
    high_features = [f for f in categories["keep_high"] if f in df.columns]
    X = df[high_features].fillna(0)
    
    print(f"Testing {len(high_features)} high-importance features...")
    mae_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        rf.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = rf.predict(X.iloc[val_idx])
        mae = mean_absolute_error(y.iloc[val_idx], pred)
        mae_scores.append(mae)
        print(f"  Fold {fold+1}: MAE = {mae:.3f}")
    
    results.append(("High importance (8)", np.mean(mae_scores), np.std(mae_scores)))
    print(f"\n1. High importance only: MAE = {np.mean(mae_scores):.3f}")
    
    # Test 2: High + Medium
    medium_features = [f for f in categories["keep_medium"] if f in df.columns]
    combined = high_features + medium_features
    X = df[combined].fillna(0)
    
    mae_scores = []
    for train_idx, val_idx in tscv.split(X):
        rf.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = rf.predict(X.iloc[val_idx])
        mae_scores.append(mean_absolute_error(y.iloc[val_idx], pred))
    
    results.append(("High + Medium (18)", np.mean(mae_scores), np.std(mae_scores)))
    print(f"2. High + Medium ({len(combined)} features): MAE = {np.mean(mae_scores):.3f}")
    
    # Test 3: All except zero-importance
    df_clean = remove_zero_importance_features(df.copy())
    numeric_cols = [c for c in df_clean.columns if c != TARGET and c in df_clean.select_dtypes(include=[np.number]).columns]
    X = df_clean[numeric_cols].fillna(0)
    
    mae_scores = []
    for train_idx, val_idx in tscv.split(X):
        rf.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = rf.predict(X.iloc[val_idx])
        mae_scores.append(mean_absolute_error(y.iloc[val_idx], pred))
    
    results.append(("No zero-importance (26)", np.mean(mae_scores), np.std(mae_scores)))
    print(f"3. No zero-importance ({X.shape[1]} features): MAE = {np.mean(mae_scores):.3f}")
    
    # Test 4: After removing correlated
    df_corr = remove_correlated_features(df_clean.copy())
    numeric_cols = [c for c in df_corr.columns if c != TARGET and c in df_corr.select_dtypes(include=[np.number]).columns]
    X = df_corr[numeric_cols].fillna(0)
    
    mae_scores = []
    for train_idx, val_idx in tscv.split(X):
        rf.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = rf.predict(X.iloc[val_idx])
        mae_scores.append(mean_absolute_error(y.iloc[val_idx], pred))
    
    results.append(("No correlated (22)", np.mean(mae_scores), np.std(mae_scores)))
    print(f"4. No correlated ({X.shape[1]} features): MAE = {np.mean(mae_scores):.3f}")
    
    # Test 5: All features (baseline)
    all_numeric = [c for c in df.columns if c != TARGET and df[c].dtype in ['float64', 'int64']]
    X = df[all_numeric].fillna(0)
    
    mae_scores = []
    for train_idx, val_idx in tscv.split(X):
        rf.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = rf.predict(X.iloc[val_idx])
        mae_scores.append(mean_absolute_error(y.iloc[val_idx], pred))
    
    results.append(("All features (38)", np.mean(mae_scores), np.std(mae_scores)))
    print(f"5. All features ({X.shape[1]} features): MAE = {np.mean(mae_scores):.3f}")
    
    return results

def run_rfe_selection():
    """Run Recursive Feature Elimination."""
    df = load_data()
    categories = get_feature_categories()
    
    high_features = [f for f in categories["keep_high"] if f in df.columns]
    X = df[high_features].fillna(0)
    y = df[TARGET]
    
    rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    
    print("\n--- RFE Selection ---")
    for n_features in range(3, len(high_features) + 1):
        rfe = RFE(rf, n_features_to_select=n_features)
        rfe.fit(X, y)
        
        selected = [f for f, s in zip(high_features, rfe.support_) if s]
        
        mae_scores = []
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        for train_idx, val_idx in tscv.split(X):
            rf.fit(X.iloc[train_idx], y.iloc[train_idx])
            X_sel = X[selected]
            pred = rf.predict(X_sel.iloc[val_idx])
            mae_scores.append(mean_absolute_error(y.iloc[val_idx], pred))
        
        print(f"  {n_features} features: MAE = {np.mean(mae_scores):.3f}")
    
    return selected

def save_optimal_config():
    """Save the optimal feature configuration."""
    categories = get_feature_categories()
    
    config = {
        "recommended_features": {
            "high_importance": categories["keep_high"],
            "medium_importance": categories["keep_medium"],
            "total_count": len(categories["keep_high"]) + len(categories["keep_medium"])
        },
        "removed_features": {
            "zero_importance": categories["remove"],
            "circuit_profiles": categories["circuit_profile"]
        },
        "selection_criteria": {
            "composite_score_threshold": 0.01,
            "correlation_threshold": 0.9
        }
    }
    
    import json
    with open("artifacts_v3/feature_selection_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nSaved feature selection config to artifacts_v3/feature_selection_config.json")
    return config

if __name__ == "__main__":
    print("=" * 60)
    print("FEATURE SELECTION PIPELINE")
    print("=" * 60)
    
    results = compare_feature_subsets()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, mae, std in results:
        print(f"{name:25} MAE = {mae:.3f} ± {std:.3f}")
    
    # Best approach based on results
    best = min(results, key=lambda x: x[1])
    print(f"\n>>> Best approach: {best[0]} with MAE = {best[1]:.3f}")
    
    save_optimal_config()