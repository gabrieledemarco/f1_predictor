#!/usr/bin/env python3
"""
Cross-validation comparison: with vs without sector time features.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

TARGET = "finish_position"
N_SPLITS = 5

SECTOR_FEATURES = [
    "s1_best_ms", "s2_best_ms", "s3_best_ms", "total_best_ms",
    "s1_delta_ms", "s2_delta_ms", "s3_delta_ms", "total_sector_delta_ms"
]

CORE_FEATURES = [
    "pace_delta_ms", "grid_position", "lap_time_delta_pct", "best_lap_delta_pct",
    "avg_lap_ms", "min_lap_ms", "std_lap_ms", "total_valid_laps",
    "avg_tyre_life", "lap_consistency", "personal_bests",
    "ct_street", "ct_desert", "ct_high_speed", "ct_mixed",
    "soft_pct", "medium_pct", "hard_pct"
]

def load_data(path="artifacts_v3/feature_matrix.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=[TARGET])
    return df

def prepare_features(df, feature_set):
    available = [f for f in feature_set if f in df.columns]
    X = df[available].fillna(0)
    y = df[TARGET]
    mask = ~y.isna()
    return X[mask], y[mask]

def run_cv_comparison():
    print("=" * 60)
    print("CROSS-VALIDATION: WITH vs WITHOUT SECTOR TIME FEATURES")
    print("=" * 60)
    
    df = load_data()
    print(f"\nDataset: {len(df)} records")
    
    core_X, core_y = prepare_features(df, CORE_FEATURES)
    full_X, full_y = prepare_features(df, CORE_FEATURES + SECTOR_FEATURES)
    
    print(f"Core features: {core_X.shape[1]}")
    print(f"Full features: {full_X.shape[1]}")
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    
    print(f"\n--- Time Series CV ({N_SPLITS} splits) ---")
    
    core_maes = []
    full_maes = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(core_X)):
        X_train, X_val = core_X.iloc[train_idx], core_X.iloc[val_idx]
        y_train, y_val = core_y.iloc[train_idx], core_y.iloc[val_idx]
        
        rf.fit(X_train, y_train)
        pred = rf.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        core_maes.append(mae)
        
        X_train_full = full_X.iloc[train_idx]
        X_val_full = full_X.iloc[val_idx]
        
        rf.fit(X_train_full, y_train)
        pred_full = rf.predict(X_val_full)
        mae_full = mean_absolute_error(y_val, pred_full)
        full_maes.append(mae_full)
        
        print(f"  Fold {fold+1}: Core MAE={mae:.3f}, Full MAE={mae_full:.3f}, Delta={mae_full-mae:+.3f}")
    
    print(f"\n--- RESULTS ---")
    print(f"Core features only:  MAE = {np.mean(core_maes):.3f} ± {np.std(core_maes):.3f}")
    print(f"Full features:       MAE = {np.mean(full_maes):.3f} ± {np.std(full_maes):.3f}")
    print(f"Improvement:         {np.mean(full_maes) - np.mean(core_maes):+.3f} positions")
    
    improvement_pct = (np.mean(core_maes) - np.mean(full_maes)) / np.mean(core_maes) * 100
    print(f"Relative improvement: {improvement_pct:+.1f}%")
    
    if np.mean(full_maes) < np.mean(core_maes):
        print("\n==> Sector features IMPROVE model performance!")
    else:
        print("\n==> Sector features do NOT improve model (or have limited impact)")
    
    print("\n--- Feature Importance (Full Model) ---")
    rf.fit(full_X, full_y)
    importances = pd.Series(rf.feature_importances_, index=full_X.columns)
    for feat, imp in importances.nlargest(10).items():
        is_sector = feat in SECTOR_FEATURES
        marker = "*" if is_sector else " "
        print(f"  {marker} {feat}: {imp:.4f}")

if __name__ == "__main__":
    run_cv_comparison()