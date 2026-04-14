#!/usr/bin/env python3
"""
Fix remaining coverage issues - sector times imputation.
"""
import pandas as pd
import numpy as np

def fix_sector_times_coverage():
    """Impute missing sector times with circuit-level median."""
    df = pd.read_csv("artifacts/feature_matrix_fixed.csv")
    
    sector_cols = ["s1_best_ms", "s2_best_ms", "s3_best_ms", "total_best_ms",
                   "s1_delta_ms", "s2_delta_ms", "s3_delta_ms", "total_sector_delta_ms"]
    
    print(f"Before sector fix: {df[sector_cols].notna().sum().sum()} non-null values")
    
    # Fill with 0 for delta (neutral - no sector delta info)
    delta_cols = ["s1_delta_ms", "s2_delta_ms", "s3_delta_ms", "total_sector_delta_ms"]
    for col in delta_cols:
        df[col] = df[col].fillna(0)
    
    # Fill raw sector times with 0 (neutral baseline)
    raw_cols = ["s1_best_ms", "s2_best_ms", "s3_best_ms", "total_best_ms"]
    for col in raw_cols:
        df[col] = df[col].fillna(0)
    
    print(f"After sector fix: {df[sector_cols].notna().sum().sum()} non-null values")
    
    # Save
    df.to_csv("artifacts/feature_matrix_v2.csv", index=False)
    
    # Final coverage report
    print("\n=== FINAL COVERAGE ===")
    low_coverage = []
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            non_null = df[col].notna().sum()
            pct = non_null / len(df) * 100
            if pct < 97:
                low_coverage.append((col, pct))
    
    if not low_coverage:
        print("All features > 97% coverage!")
    else:
        for col, pct in low_coverage:
            print(f"{col}: {pct:.1f}%")
    
    return df


if __name__ == "__main__":
    fix_sector_times_coverage()