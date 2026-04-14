#!/usr/bin/env python3
"""
Fix coverage issues in feature matrix.
"""
import pandas as pd
import numpy as np

def fix_circuit_profiles():
    """Add missing circuit profiles for bahrain, jeddah, imola, ricard, vegas."""
    
    missing_profiles = {
        "bahrain": {
            "circuit_type": "desert",
            "top_speed_kmh": 330,
            "full_throttle_pct": 60,
            "avg_speed_kmh": 215,
            "avg_fast_corner_kmh": 255,
            "avg_medium_corner_kmh": 155,
            "avg_slow_corner_kmh": 75,
            "min_speed_kmh": 60,
        },
        "jeddah": {
            "circuit_type": "street",
            "top_speed_kmh": 350,
            "full_throttle_pct": 72,
            "avg_speed_kmh": 235,
            "avg_fast_corner_kmh": 290,
            "avg_medium_corner_kmh": 165,
            "avg_slow_corner_kmh": 70,
            "min_speed_kmh": 55,
        },
        "imola": {
            "circuit_type": "mixed",
            "top_speed_kmh": 330,
            "full_throttle_pct": 62,
            "avg_speed_kmh": 215,
            "avg_fast_corner_kmh": 265,
            "avg_medium_corner_kmh": 155,
            "avg_slow_corner_kmh": 78,
            "min_speed_kmh": 58,
        },
        "ricard": {
            "circuit_type": "high_speed",
            "top_speed_kmh": 340,
            "full_throttle_pct": 68,
            "avg_speed_kmh": 230,
            "avg_fast_corner_kmh": 285,
            "avg_medium_corner_kmh": 165,
            "avg_slow_corner_kmh": 82,
            "min_speed_kmh": 65,
        },
        "vegas": {
            "circuit_type": "street",
            "top_speed_kmh": 342,
            "full_throttle_pct": 72,
            "avg_speed_kmh": 225,
            "avg_fast_corner_kmh": 278,
            "avg_medium_corner_kmh": 162,
            "avg_slow_corner_kmh": 65,
            "min_speed_kmh": 52,
        }
    }
    return missing_profiles


def apply_fixes():
    """Load and fix feature matrix coverage."""
    df = pd.read_csv("artifacts/feature_matrix.csv")
    print(f"Original shape: {df.shape}")
    
    # Fix 1: Circuit profiles
    profiles = fix_circuit_profiles()
    
    for circuit, profile in profiles.items():
        mask = df["circuit_ref"] == circuit
        if mask.any():
            for col, val in profile.items():
                if col in df.columns:
                    df.loc[mask, col] = df.loc[mask, col].fillna(val)
            print(f"  Fixed {circuit} profile: {mask.sum()} rows")
    
    # Fix 2: Drop pos_gain (not available)
    if "pos_gain" in df.columns:
        df = df.drop(columns=["pos_gain"])
        print("  Dropped pos_gain (0% coverage)")
    
    # Fix 3: Fill remaining gaps with median for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ["year", "round", "finish_position", "is_dnf"]:
            null_pct = df[col].isna().sum() / len(df) * 100
            if null_pct > 0 and null_pct < 50:
                df[col] = df[col].fillna(df[col].median())
    
    # Save
    df.to_csv("artifacts/feature_matrix_fixed.csv", index=False)
    print(f"\nFixed shape: {df.shape}")
    
    # Report coverage
    print("\n=== POST-FIX COVERAGE ===")
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            non_null = df[col].notna().sum()
            pct = non_null / len(df) * 100
            if pct < 97:
                print(f'{col}: {pct:.1f}%')
    
    return df


if __name__ == "__main__":
    apply_fixes()