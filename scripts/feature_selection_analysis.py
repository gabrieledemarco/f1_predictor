#!/usr/bin/env python3
"""
Feature Analysis from MongoDB Data

Analyzes features from the F1 predictor data pipeline to identify
important features for the ML model.

Usage:
    python scripts/feature_analysis.py
"""

import os
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

load_dotenv()


def get_mongo_client():
    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable is required")
    
    mongo_db = os.environ.get("MONGO_DB", "betbreaker")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    return client[mongo_db]


def extract_lap_features(db, years: List[int]) -> pd.DataFrame:
    """Extract lap time features from f1_lap_times collection."""
    
    print(f"Extracting lap features for years {years}...")
    
    cursor = db.f1_lap_times.find(
        {"year": {"$in": years}},
        {
            "year": 1, "round": 1, "circuit_ref": 1,
            "driver_code": 1, "lap_number": 1, 
            "lap_time_ms": 1, "position": 1
        }
    )
    
    lap_data = list(cursor)
    print(f"  Found {len(lap_data)} lap records")
    
    if not lap_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(lap_data)
    
    agg = df.groupby(["year", "round", "circuit_ref", "driver_code"]).agg(
        avg_lap_time=("lap_time_ms", "mean"),
        min_lap_time=("lap_time_ms", "min"),
        max_lap_time=("lap_time_ms", "max"),
        lap_count=("lap_number", "count"),
        avg_position=("position", "mean"),
    ).reset_index()
    
    return agg


def extract_qualifying_features(db, years: List[int]) -> pd.DataFrame:
    """Extract qualifying features from f1_qualifying collection."""
    
    print(f"Extracting qualifying features for years {years}...")
    
    cursor = db.f1_qualifying.find(
        {"year": {"$in": years}},
        {
            "year": 1, "round": 1, "circuit_ref": 1,
            "driver_code": 1, "constructor_ref": 1,
            "position": 1, "q1_ms": 1, "q2_ms": 1, "q3_ms": 1,
            "total_q_ms": 1
        }
    )
    
    quali_data = list(cursor)
    print(f"  Found {len(quali_data)} qualifying records")
    
    if not quali_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(quali_data)
    
    return df


def extract_pace_features(db, years: List[int]) -> pd.DataFrame:
    """Extract pace features from f1_pace_observations collection."""
    
    print(f"Extracting pace features for years {years}...")
    
    cursor = db.f1_pace_observations.find(
        {"year": {"$in": years}}
    )
    
    pace_data = list(cursor)
    print(f"  Found {len(pace_data)} pace observations")
    
    if not pace_data:
        return pd.DataFrame()
    
    return pd.DataFrame(pace_data)


def extract_sector_features(db, years: List[int]) -> pd.DataFrame:
    """Extract sector time features from f1_session_stats collection."""
    
    print(f"Extracting sector features for years {years}...")
    
    cursor = db.f1_session_stats.find(
        {"year": {"$in": years}, "session_type": "qualifying"},
        {
            "year": 1, "circuit_ref": 1, "driver_code": 1,
            "s1_best_ms": 1, "s2_best_ms": 1, "s3_best_ms": 1,
            "total_best_ms": 1
        }
    )
    
    sector_data = list(cursor)
    print(f"  Found {len(sector_data)} sector time records")
    
    if not sector_data:
        return pd.DataFrame()
    
    return pd.DataFrame(sector_data)


def compute_feature_importance(df: pd.DataFrame, target_col: str = "position") -> pd.DataFrame:
    """Compute feature importance using Random Forest."""
    
    exclude_cols = [target_col, "_id", "driver_code", "circuit_ref", "constructor_ref", "lookup_key", "source", "imported_at"]
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.endswith("_ref")]
    
    X = df[feature_cols].fillna(0)
    y = df[target_col]
    
    if len(y.unique()) < 2:
        return pd.DataFrame()
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    
    return importance


def main():
    print("=" * 60)
    print("FEATURE ANALYSIS FROM MONGODB")
    print("=" * 60)
    print()
    
    try:
        db = get_mongo_client()
        print("Connected to MongoDB")
        
        years = [2023, 2024, 2025]
        
        lap_features = extract_lap_features(db, years)
        quali_features = extract_qualifying_features(db, years)
        pace_features = extract_pace_features(db, years)
        sector_features = extract_sector_features(db, years)
        
        print("\n" + "=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)
        print(f"Lap feature records: {len(lap_features)}")
        print(f"Qualifying feature records: {len(quali_features)}")
        print(f"Pace feature records: {len(pace_features)}")
        print(f"Sector feature records: {len(sector_features)}")
        
        if not lap_features.empty:
            print("\n" + "=" * 60)
            print("LAP FEATURE SAMPLE")
            print("=" * 60)
            print(lap_features.head())
            
            print("\nLap feature statistics:")
            print(lap_features.describe())
        
        if not quali_features.empty:
            print("\n" + "=" * 60)
            print("QUALIFYING FEATURE SAMPLE")
            print("=" * 60)
            print(quali_features.head())
            
            print("\nQualifying feature statistics:")
            print(quali_features.describe())
        
        if not pace_features.empty:
            print("\n" + "=" * 60)
            print("PACE FEATURE SAMPLE")
            print("=" * 60)
            print(pace_features.head())
            
            print("\nPace feature statistics:")
            print(pace_features.describe())
        
        if not sector_features.empty:
            print("\n" + "=" * 60)
            print("SECTOR FEATURE SAMPLE")
            print("=" * 60)
            print(sector_features.head())
            
            print("\nSector feature statistics:")
            print(sector_features.describe())
        
        all_importance = []
        
        if not quali_features.empty:
            print("\n" + "=" * 60)
            print("FEATURE IMPORTANCE (Qualifying -> Position)")
            print("=" * 60)
            
            df = quali_features.copy()
            df = df[df["position"].notna() & (df["position"] > 0)]
            
            if len(df) > 10:
                importance = compute_feature_importance(df)
                importance["source"] = "qualifying"
                print(importance.to_string(index=False))
                all_importance.append(importance)
        
        if not sector_features.empty:
            print("\n" + "=" * 60)
            print("FEATURE IMPORTANCE (Sector Times -> Position)")
            print("=" * 60)
            
            df = sector_features.copy()
            
            if "position" not in df.columns:
                quali_cursor = db.f1_qualifying.find(
                    {"year": {"$in": years}},
                    {"year": 1, "circuit_ref": 1, "driver_code": 1, "position": 1}
                )
                quali_lookup = {f"{q['year']}_{q['circuit_ref']}_{q['driver_code']}": q["position"] 
                              for q in quali_cursor}
                
                df["lookup_key"] = df["year"].astype(str) + "_" + df["circuit_ref"] + "_" + df["driver_code"]
                df["position"] = df["lookup_key"].map(quali_lookup)
            
            df = df[df["position"].notna() & (df["position"] > 0)]
            
            if len(df) > 10:
                importance = compute_feature_importance(df)
                importance["source"] = "sector"
                print(importance.to_string(index=False))
                all_importance.append(importance)
        
        if not pace_features.empty:
            print("\n" + "=" * 60)
            print("FEATURE IMPORTANCE (Pace -> Position)")
            print("=" * 60)
            
            df = pace_features.copy()
            
            if "position" not in df.columns:
                quali_cursor = db.f1_qualifying.find(
                    {"year": {"$in": years}},
                    {"year": 1, "circuit_ref": 1, "driver_code": 1, "position": 1}
                )
                quali_lookup = {f"{q['year']}_{q['circuit_ref']}_{q['driver_code']}": q["position"] 
                              for q in quali_cursor}
                
                df["lookup_key"] = df["year"].astype(str) + "_" + df["circuit_ref"] + "_" + df["constructor_ref"]
                df["position"] = df["lookup_key"].map(quali_lookup)
                df = df.drop(columns=["lookup_key"], errors="ignore")
            
            df = df[df["position"].notna() & (df["position"] > 0)]
            
            if len(df) > 10:
                importance = compute_feature_importance(df)
                importance["source"] = "pace"
                print(importance.to_string(index=False))
                all_importance.append(importance)
        
        if all_importance:
            combined = pd.concat(all_importance, ignore_index=True)
            combined.to_csv("artifacts/feature_importance.csv", index=False)
            print("\n" + "=" * 60)
            print("COMBINED FEATURE IMPORTANCE")
            print("=" * 60)
            print(combined.to_string(index=False))
            print("\nSaved to artifacts/feature_importance.csv")
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
