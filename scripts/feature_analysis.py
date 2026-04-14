#!/usr/bin/env python3
"""
scripts/feature_analysis.py
============================
F1 Feature Significance Study — MongoDB Data Science Analysis

Estrae tutte le feature disponibili da MongoDB, le unisce in una
feature matrix per (year, round, driver_code) e calcola:

  1. Spearman rank correlation con la posizione finale (+ p-value)
  2. Mutual Information score (discretized, regression)
  3. Random Forest feature importance (MDI)
  4. Permutation importance su Random Forest
  5. Gradient Boosting feature importance (MDI)

Output (in --output-dir, default "reports/feature_analysis/"):
  - feature_importance_report.json   — report completo strutturato
  - feature_matrix.csv               — dataset raw (per analisi esterne)
  - feature_rankings.csv             — ranking finale aggregato

Usage:
    python scripts/feature_analysis.py
    python scripts/feature_analysis.py --min-year 2020 --max-year 2025
    python scripts/feature_analysis.py --output-dir /tmp/reports --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("feature_analysis")


# ─────────────────────────────────────────────────────────────────────────────
# MongoDB connection
# ─────────────────────────────────────────────────────────────────────────────

def get_db():
    uri = os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URI")
    if not uri:
        raise ValueError("MONGODB_URI environment variable is required")
    db_name = os.environ.get("MONGO_DB", "betbreaker")
    from pymongo import MongoClient
    client = MongoClient(uri, serverSelectionTimeoutMS=10_000)
    client.admin.command("ping")
    return client[db_name]


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_race_results(db, min_year: int, max_year: int) -> pd.DataFrame:
    """
    Estrae la base della feature matrix da f1_races.
    Una riga per (year, round, driver_code).
    """
    log.info("Extracting race results from f1_races …")
    rows = []
    coll = db["f1_races"]

    cursor = coll.find(
        {"year": {"$gte": min_year, "$lte": max_year}},
        {"year": 1, "round": 1, "circuit_ref": 1, "circuit_type": 1,
         "results": 1, "qualifying": 1, "_id": 0}
    )

    for race in cursor:
        year = race["year"]
        rnd = race["round"]
        circuit_ref = race.get("circuit_ref", "unknown")
        circuit_type = race.get("circuit_type", "unknown")

        # Build qualifying position lookup
        qual_lookup: dict[str, int] = {}
        for q in race.get("qualifying", []):
            dc = q.get("driver_code", "")
            gp = q.get("grid_position") or q.get("position")
            if dc and gp is not None:
                try:
                    qual_lookup[dc] = int(gp)
                except (ValueError, TypeError):
                    pass

        for res in race.get("results", []):
            driver_code = res.get("driver_code", "")
            if not driver_code:
                continue
            finish_pos = res.get("finish_position")
            grid_pos = res.get("grid_position", 0) or 0
            points = res.get("points", 0.0) or 0.0
            laps = res.get("laps_completed", 0) or 0
            status = res.get("status", "Unknown")
            constructor_ref = res.get("constructor_ref", "unknown")

            # Fallback: use qualifying position if grid_position missing from race result
            if not grid_pos and driver_code in qual_lookup:
                grid_pos = qual_lookup[driver_code]

            is_dnf = 1 if finish_pos is None else 0
            finish_pos_filled = finish_pos if finish_pos is not None else 21
            pos_gain = grid_pos - finish_pos_filled if grid_pos > 0 else None
            points_scored = float(points)

            rows.append({
                "year": year,
                "round": rnd,
                "circuit_ref": circuit_ref,
                "circuit_type": str(circuit_type).replace("CircuitType.", ""),
                "driver_code": driver_code,
                "constructor_ref": constructor_ref,
                "grid_position": grid_pos,
                "finish_position": finish_pos_filled,
                "is_dnf": is_dnf,
                "pos_gain": pos_gain,
                "points_scored": points_scored,
                "laps_completed": laps,
                "status_finished": 1 if "Finished" in status or (finish_pos is not None and finish_pos <= 20) else 0,
            })

    df = pd.DataFrame(rows)
    log.info(f"  → {len(df)} driver-race records from {df['year'].nunique() if len(df) else 0} years")
    return df


def extract_pace_observations(db, min_year: int, max_year: int) -> pd.DataFrame:
    """
    Estrae pace_delta_ms da f1_pace_observations.
    Chiave join: (year, round, constructor_ref).
    """
    log.info("Extracting constructor pace from f1_pace_observations …")
    coll = db["f1_pace_observations"]
    docs = list(coll.find(
        {"year": {"$gte": min_year, "$lte": max_year}},
        {"year": 1, "round": 1, "circuit_ref": 1, "constructor_ref": 1,
         "pace_delta_ms": 1, "_id": 0}
    ))
    if not docs:
        log.warning("  f1_pace_observations is empty — pace features will be missing")
        return pd.DataFrame(columns=["year", "round", "constructor_ref", "pace_delta_ms"])

    df = pd.DataFrame(docs)
    df["pace_delta_ms"] = pd.to_numeric(df["pace_delta_ms"], errors="coerce")
    log.info(f"  → {len(df)} pace observation records")
    return df[["year", "round", "constructor_ref", "pace_delta_ms"]]


def extract_circuit_profiles(db) -> pd.DataFrame:
    """
    Estrae feature da f1_circuit_profiles.
    Chiave join: circuit_ref.
    """
    log.info("Extracting circuit profiles from f1_circuit_profiles …")
    coll = db["f1_circuit_profiles"]
    docs = list(coll.find({}, {"_id": 0}))
    if not docs:
        log.warning("  f1_circuit_profiles is empty — circuit features will be missing")
        return pd.DataFrame(columns=["circuit_ref"])

    df = pd.DataFrame(docs)
    numeric_cols = [c for c in df.columns
                    if c not in ("circuit_ref", "circuit_type", "_id")
                    and df[c].dtype in (float, int) or pd.api.types.is_numeric_dtype(df.get(c, pd.Series()))]
    keep = ["circuit_ref"] + [c for c in numeric_cols if c in df.columns]
    df = df[keep].copy()
    log.info(f"  → {len(df)} circuit profiles, features: {keep[1:]}")
    return df


def extract_lap_features(db, min_year: int, max_year: int) -> pd.DataFrame:
    """
    Aggrega lap times TracingInsights per (year, round, driver_code).
    Feature: avg/min lap time, consistency (std), tyre deg rate, compound dist.
    """
    log.info("Extracting lap features from f1_lap_times …")
    coll = db["f1_lap_times"]

    # Aggregation pipeline: group by (year, round, driver_code)
    pipeline = [
        {"$match": {
            "year": {"$gte": min_year, "$lte": max_year},
            "is_valid": True
        }},
        {"$group": {
            "_id": {"year": "$year", "round": "$round", "driver_code": "$driver_code"},
            "avg_lap_ms":      {"$avg": "$lap_time_ms"},
            "min_lap_ms":      {"$min": "$lap_time_ms"},
            "std_lap_ms":      {"$stdDevSamp": "$lap_time_ms"},
            "total_valid_laps": {"$sum": 1},
            "soft_laps":       {"$sum": {"$cond": [{"$eq": ["$compound", "SOFT"]}, 1, 0]}},
            "medium_laps":     {"$sum": {"$cond": [{"$eq": ["$compound", "MEDIUM"]}, 1, 0]}},
            "hard_laps":       {"$sum": {"$cond": [{"$eq": ["$compound", "HARD"]}, 1, 0]}},
            "avg_tyre_life":   {"$avg": "$tyre_life"},
            "max_tyre_life":   {"$max": "$tyre_life"},
            "personal_bests":  {"$sum": {"$cond": ["$is_personal_best", 1, 0]}},
        }},
        {"$project": {
            "year":            "$_id.year",
            "round":           "$_id.round",
            "driver_code":     "$_id.driver_code",
            "avg_lap_ms":      1,
            "min_lap_ms":      1,
            "std_lap_ms":      1,
            "total_valid_laps": 1,
            "soft_pct":        {"$divide": ["$soft_laps",   {"$max": ["$total_valid_laps", 1]}]},
            "medium_pct":      {"$divide": ["$medium_laps", {"$max": ["$total_valid_laps", 1]}]},
            "hard_pct":        {"$divide": ["$hard_laps",   {"$max": ["$total_valid_laps", 1]}]},
            "avg_tyre_life":   1,
            "max_tyre_life":   1,
            "personal_bests":  1,
            "_id":             0,
        }}
    ]

    docs = list(coll.aggregate(pipeline, allowDiskUse=True))
    if not docs:
        log.warning("  f1_lap_times is empty — lap telemetry features will be missing")
        return pd.DataFrame(columns=["year", "round", "driver_code"])

    df = pd.DataFrame(docs)
    # Lap consistency: lower std = more consistent
    df["lap_consistency"] = df["std_lap_ms"] / df["avg_lap_ms"].clip(lower=1)
    # Personal best rate
    df["pb_rate"] = df["personal_bests"] / df["total_valid_laps"].clip(lower=1)
    log.info(f"  → {len(df)} driver-race lap aggregations")
    return df


def extract_odds_features(db, min_year: int, max_year: int) -> pd.DataFrame:
    """
    Estrae probabilità bookmaker più recenti per (year, round, driver_code).
    Usa f1_pinnacle_odds (primary) o odds_records (fallback).
    """
    log.info("Extracting odds features …")
    rows = []

    # Try f1_pinnacle_odds first
    coll_names = db.list_collection_names()

    if "f1_pinnacle_odds" in coll_names:
        coll = db["f1_pinnacle_odds"]
        docs = list(coll.find(
            {"year": {"$gte": min_year, "$lte": max_year}},
            {"year": 1, "round": 1, "driver_code": 1, "p_novig": 1,
             "p_implied_raw": 1, "market": 1, "_id": 0}
        ))
        if docs:
            log.info(f"  → {len(docs)} odds records from f1_pinnacle_odds")
            df = pd.DataFrame(docs)
            # Keep only outrights/winner market, take most recent per driver per race
            if "market" in df.columns:
                win_df = df[df["market"].isin(["outrights", "h2h", "winner"])].copy()
            else:
                win_df = df.copy()
            if not win_df.empty:
                # Aggregate: mean p_novig per (year, round, driver_code)
                agg = win_df.groupby(["year", "round", "driver_code"]).agg(
                    odds_p_novig=("p_novig", "mean"),
                    odds_p_implied=("p_implied_raw", "mean"),
                ).reset_index()
                return agg

    # Fallback: odds_records
    if "odds_records" in coll_names:
        coll = db["odds_records"]
        docs = list(coll.find(
            {},
            {"race_id": 1, "driver_code": 1, "p_novig": 1, "p_implied_raw": 1, "_id": 0}
        ))
        if docs:
            log.info(f"  → {len(docs)} odds records from odds_records (fallback)")
            df = pd.DataFrame(docs)
            # race_id might be "2024_01" format
            if "race_id" in df.columns:
                df[["year", "round"]] = df["race_id"].str.extract(r"(\d{4})_(\d+)").astype(float)
                df = df.dropna(subset=["year", "round"])
                df["year"] = df["year"].astype(int)
                df["round"] = df["round"].astype(int)
            agg = df.groupby(["year", "round", "driver_code"]).agg(
                odds_p_novig=("p_novig", "mean"),
                odds_p_implied=("p_implied_raw", "mean"),
            ).reset_index()
            return agg

    log.warning("  No odds data available — odds features will be missing")
    return pd.DataFrame(columns=["year", "round", "driver_code"])


def extract_session_stats(db, min_year: int, max_year: int) -> pd.DataFrame:
    """
    Estrae feature da f1_session_stats (sector times from TracingInsights).
    Feature: best_sector_times (s1, s2, s3), total_best_ms.
    """
    log.info("Extracting session stats from f1_session_stats …")
    
    # Mapping from TracingInsights circuit names to f1_races circuit_ref
    TRACING_TO_RACES = {
        "abu_dhabi": "yas_marina",
        "australian": "albert_park",
        "austrian": "red_bull_ring",
        "azerbaijan": "baku",
        "british": "silverstone",
        "canadian": "villeneuve",
        "chinese": "shanghai",
        "dutch": "zandvoort",
        "emilia_romagna": "imola",
        "french": "catalunya",
        "hungarian": "hungaroring",
        "italian": "monza",
        "japanese": "suzuka",
        "las_vegas": "vegas",
        "mexico_city": "rodriguez",
        "qatar": "losail",
        "saudi_arabian": "jeddah",
        "singapore": "marina_bay",
        "spanish": "catalunya",
        "são_paulo": "interlagos",
        "united_states": "americas",
    }
    
    coll = db["f1_session_stats"]
    docs = list(coll.find(
        {"year": {"$gte": min_year, "$lte": max_year}},
        {"year": 1, "circuit_ref": 1, "driver_code": 1,
         "s1_best_ms": 1, "s2_best_ms": 1, "s3_best_ms": 1, "total_best_ms": 1,
         "session_type": 1, "_id": 0}
    ))
    
    if not docs:
        log.warning("  f1_session_stats is empty — sector time features will be missing")
        return pd.DataFrame(columns=["year", "round", "driver_code"])
    
    df = pd.DataFrame(docs)
    
    # Filter to qualifying sessions only
    df = df[df.get("session_type", pd.Series([""] * len(df))) == "qualifying"]
    
    # Map TracingInsights circuit names to f1_races circuit_ref
    df["circuit_ref"] = df["circuit_ref"].map(TRACING_TO_RACES).fillna(df["circuit_ref"])
    
    # Get year -> round mapping from f1_races
    race_coll = db["f1_races"]
    race_docs = list(race_coll.find(
        {"year": {"$gte": min_year, "$lte": max_year}},
        {"year": 1, "round": 1, "circuit_ref": 1}
    ))
    year_round_map = {(r["year"], r["circuit_ref"]): r["round"] for r in race_docs}
    
    # Map circuit_ref -> round
    df["round"] = df.apply(lambda row: year_round_map.get((row.get("year", 0), row.get("circuit_ref", "")), 0), axis=1)
    
    # Convert numeric columns
    for col in ["s1_best_ms", "s2_best_ms", "s3_best_ms", "total_best_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Keep only relevant columns
    cols_to_keep = ["year", "round", "driver_code"]
    for col in ["s1_best_ms", "s2_best_ms", "s3_best_ms", "total_best_ms"]:
        if col in df.columns:
            cols_to_keep.append(col)
    
    df = df[cols_to_keep].dropna(subset=["round"]).drop_duplicates(subset=["year", "round", "driver_code"])
    log.info(f"  → {len(df)} qualifying session records with sector times")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature matrix assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    df_races: pd.DataFrame,
    df_pace: pd.DataFrame,
    df_circuits: pd.DataFrame,
    df_laps: pd.DataFrame,
    df_odds: pd.DataFrame,
    df_session: pd.DataFrame,
) -> pd.DataFrame:
    """
    Unisce tutte le feature in un unico DataFrame.
    """
    log.info("Building feature matrix …")
    df = df_races.copy()

    # 1. Join pace observations (by year, round, constructor_ref)
    if len(df_pace) > 0:
        df = df.merge(df_pace, on=["year", "round", "constructor_ref"], how="left")
    else:
        df["pace_delta_ms"] = np.nan

    # 2. Join circuit profiles (by circuit_ref)
    if len(df_circuits) > 0:
        df = df.merge(df_circuits, on="circuit_ref", how="left")

    # 3. Join lap telemetry (by year, round, driver_code)
    if len(df_laps) > 0:
        df = df.merge(df_laps, on=["year", "round", "driver_code"], how="left")

    # 4. Join odds (by year, round, driver_code)
    if len(df_odds) > 0:
        df = df.merge(df_odds, on=["year", "round", "driver_code"], how="left")

    # 5. Join session stats (by year, round, driver_code)
    if len(df_session) > 0:
        df = df.merge(df_session, on=["year", "round", "driver_code"], how="left")
        
        # Create sector time delta features (vs race median)
        for sector in ["s1", "s2", "s3"]:
            col = f"{sector}_best_ms"
            if col in df.columns:
                race_median = df.groupby(["year", "round"])[col].transform("median")
                df[f"{sector}_delta_ms"] = df[col] - race_median
        
        # Total best delta
        if "total_best_ms" in df.columns:
            race_median = df.groupby(["year", "round"])["total_best_ms"].transform("median")
            df["total_sector_delta_ms"] = df["total_best_ms"] - race_median

    # ── Encode categoricals ──────────────────────────────────────────────
    # Circuit type → one-hot
    if "circuit_type" in df.columns:
        ct_dummies = pd.get_dummies(df["circuit_type"], prefix="ct").astype(int)
        df = pd.concat([df, ct_dummies], axis=1)

    # Driver relative pace within race (normalise to field)
    if "avg_lap_ms" in df.columns:
        race_avg = df.groupby(["year", "round"])["avg_lap_ms"].transform("median")
        df["lap_time_delta_pct"] = (df["avg_lap_ms"] - race_avg) / race_avg.clip(lower=1) * 100

    if "min_lap_ms" in df.columns:
        race_min_avg = df.groupby(["year", "round"])["min_lap_ms"].transform("median")
        df["best_lap_delta_pct"] = (df["min_lap_ms"] - race_min_avg) / race_min_avg.clip(lower=1) * 100

    # Grid position relative: normalise [1..n] → [0..1]
    n_drivers = df.groupby(["year", "round"])["driver_code"].transform("count")
    df["grid_position_pct"] = (df["grid_position"] - 1) / (n_drivers - 1).clip(lower=1)

    log.info(f"  Feature matrix: {len(df)} rows × {len(df.columns)} columns")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Analysis functions
# ─────────────────────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    # Race context
    "grid_position",
    "grid_position_pct",
    "year",
    "round",
    # Constructor pace
    "pace_delta_ms",
    # Lap telemetry (TracingInsights)
    "avg_lap_ms",
    "min_lap_ms",
    "std_lap_ms",
    "lap_time_delta_pct",
    "best_lap_delta_pct",
    "lap_consistency",
    "total_valid_laps",
    "soft_pct",
    "medium_pct",
    "hard_pct",
    "avg_tyre_life",
    "max_tyre_life",
    "pb_rate",
    # Odds
    "odds_p_novig",
    "odds_p_implied",
    # Session stats (qualifying) - sector times from TracingInsights
    "s1_best_ms",
    "s2_best_ms",
    "s3_best_ms",
    "total_best_ms",
    # Sector time deltas (vs race median)
    "s1_delta_ms",
    "s2_delta_ms",
    "s3_delta_ms",
    "total_sector_delta_ms",
    "lap_count",
    # Circuit profiles
    "top_speed_kmh",
    "full_throttle_pct",
    "avg_speed_kmh",
    "drs_zones",
    "corner_count",
    "slow_corners",
    "medium_corners",
    "fast_corners",
    # Circuit type one-hot
    "ct_street",
    "ct_high_df",
    "ct_high_speed",
    "ct_mixed",
    "ct_unknown",
]

TARGET = "finish_position"
TARGET_BINARY = "top3_finish"


def run_spearman_correlation(df: pd.DataFrame, features: list[str]) -> list[dict]:
    """Spearman correlation con finish_position (+ p-value)."""
    from scipy.stats import spearmanr
    results = []
    for feat in features:
        if feat not in df.columns:
            continue
        mask = df[[feat, TARGET]].dropna()
        if len(mask) < 30:
            continue
        rho, pval = spearmanr(mask[feat], mask[TARGET])
        if not np.isfinite(rho) or not np.isfinite(pval):
            continue
        results.append({
            "feature": feat,
            "spearman_rho": round(float(rho), 4),
            "p_value": round(float(pval), 6),
            "n_obs": int(len(mask)),
            "significant": bool(pval < 0.05),
        })
    results.sort(key=lambda x: abs(x["spearman_rho"]), reverse=True)
    return results


def run_mutual_information(df: pd.DataFrame, features: list[str]) -> list[dict]:
    """Mutual Information score con finish_position."""
    from sklearn.feature_selection import mutual_info_regression
    valid_feats = [f for f in features if f in df.columns]
    sub = df[valid_feats + [TARGET]].dropna()
    if len(sub) < 50:
        log.warning("Too few complete observations for MI analysis")
        return []

    X = sub[valid_feats].values
    y = sub[TARGET].values
    mi = mutual_info_regression(X, y, random_state=42)
    results = [
        {"feature": feat, "mi_score": round(float(score), 5)}
        for feat, score in zip(valid_feats, mi)
    ]
    results.sort(key=lambda x: x["mi_score"], reverse=True)
    return results


def run_random_forest(df: pd.DataFrame, features: list[str]) -> tuple[list[dict], list[dict]]:
    """Random Forest MDI + Permutation importance."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import cross_val_score

    valid_feats = [f for f in features if f in df.columns]
    sub = df[valid_feats + [TARGET]].dropna()
    if len(sub) < 100:
        log.warning("Too few complete observations for RF analysis")
        return [], []

    X = sub[valid_feats].values
    y = sub[TARGET].values

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X, y)

    # CV score
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
    mae = -cv_scores.mean()
    log.info(f"  RF cross-val MAE: {mae:.3f} positions")

    # MDI importance
    mdi_results = [
        {"feature": feat, "rf_mdi_importance": round(float(imp), 6)}
        for feat, imp in zip(valid_feats, rf.feature_importances_)
    ]
    mdi_results.sort(key=lambda x: x["rf_mdi_importance"], reverse=True)

    # Permutation importance (subsample for speed)
    n_perm = min(len(sub), 2000)
    idx = np.random.default_rng(42).choice(len(sub), size=n_perm, replace=False)
    perm = permutation_importance(
        rf, X[idx], y[idx],
        n_repeats=10, random_state=42, n_jobs=-1
    )
    perm_results = [
        {
            "feature": feat,
            "perm_importance_mean": round(float(m), 6),
            "perm_importance_std":  round(float(s), 6),
        }
        for feat, m, s in zip(valid_feats, perm.importances_mean, perm.importances_std)
    ]
    perm_results.sort(key=lambda x: x["perm_importance_mean"], reverse=True)

    return mdi_results, perm_results, {"cv_mae": round(float(mae), 4), "n_train": int(len(sub))}


def run_gradient_boosting(df: pd.DataFrame, features: list[str]) -> list[dict]:
    """Gradient Boosting MDI feature importance."""
    from sklearn.ensemble import GradientBoostingRegressor

    valid_feats = [f for f in features if f in df.columns]
    sub = df[valid_feats + [TARGET]].dropna()
    if len(sub) < 100:
        return []

    X = sub[valid_feats].values
    y = sub[TARGET].values

    gb = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    gb.fit(X, y)

    results = [
        {"feature": feat, "gb_importance": round(float(imp), 6)}
        for feat, imp in zip(valid_feats, gb.feature_importances_)
    ]
    results.sort(key=lambda x: x["gb_importance"], reverse=True)
    return results


def compute_coverage(df: pd.DataFrame, features: list[str]) -> list[dict]:
    """Calcola la copertura (% righe non-null) di ogni feature."""
    n_total = len(df)
    results = []
    for feat in features:
        if feat not in df.columns:
            continue
        n_valid = df[feat].notna().sum()
        results.append({
            "feature": feat,
            "coverage_pct": round(float(n_valid / n_total * 100), 1),
            "n_valid": int(n_valid),
        })
    results.sort(key=lambda x: x["coverage_pct"], reverse=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Ranking aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_rankings(
    spearman: list[dict],
    mi: list[dict],
    rf_mdi: list[dict],
    perm: list[dict],
    gb: list[dict],
    coverage: list[dict],
) -> pd.DataFrame:
    """
    Combina i ranking da tutti i metodi in un unico score normalizzato [0..1].
    """
    def rank_dict(results: list[dict], key: str) -> dict[str, float]:
        if not results:
            return {}
        vals = {}
        for r in results:
            v = abs(r[key])
            vals[r["feature"]] = 0.0 if (v != v or not np.isfinite(v)) else v
        max_v = max(vals.values()) if vals else 1.0
        if not max_v or not np.isfinite(max_v):
            max_v = 1.0
        return {f: v / max_v for f, v in vals.items()}

    all_feats = set()
    for lst in [spearman, mi, rf_mdi, perm, gb]:
        for r in lst:
            all_feats.add(r["feature"])

    sp_dict   = rank_dict(spearman, "spearman_rho")
    mi_dict   = rank_dict(mi, "mi_score")
    rf_dict   = rank_dict(rf_mdi, "rf_mdi_importance")
    perm_dict = rank_dict(perm, "perm_importance_mean")
    gb_dict   = rank_dict(gb, "gb_importance")
    cov_dict  = {r["feature"]: r["coverage_pct"] / 100.0 for r in coverage}

    rows = []
    for feat in sorted(all_feats):
        sp   = sp_dict.get(feat, 0.0)
        mi_s = mi_dict.get(feat, 0.0)
        rf_s = rf_dict.get(feat, 0.0)
        pm_s = perm_dict.get(feat, 0.0)
        gb_s = gb_dict.get(feat, 0.0)
        cov  = cov_dict.get(feat, 0.0)

        # Weighted composite: MI+RF+PERM are stronger signals
        composite = (
            0.15 * sp +
            0.20 * mi_s +
            0.25 * rf_s +
            0.25 * pm_s +
            0.15 * gb_s
        ) * cov  # penalise low-coverage features
        if not np.isfinite(composite):
            composite = 0.0

        rows.append({
            "feature":           feat,
            "composite_score":   round(composite, 4),
            "spearman_norm":     round(sp, 4),
            "mi_norm":           round(mi_s, 4),
            "rf_mdi_norm":       round(rf_s, 4),
            "perm_norm":         round(pm_s, 4),
            "gb_norm":           round(gb_s, 4),
            "coverage_pct":      round(cov * 100, 1),
        })

    df_rank = pd.DataFrame(rows).sort_values("composite_score", ascending=False)
    df_rank.insert(0, "rank", range(1, len(df_rank) + 1))
    return df_rank


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="F1 MongoDB Feature Significance Analysis")
    p.add_argument("--min-year", type=int, default=2019, help="Primo anno (default: 2019)")
    p.add_argument("--max-year", type=int, default=datetime.now().year, help="Ultimo anno (default: anno corrente)")
    p.add_argument("--output-dir", default="reports/feature_analysis", help="Directory output")
    p.add_argument("--verbose", action="store_true", help="Log dettagliato")
    return p.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("F1 FEATURE SIGNIFICANCE ANALYSIS")
    print("=" * 70)
    print(f"  Range: {args.min_year} – {args.max_year}")
    print(f"  Output: {output_dir.resolve()}")
    print()

    # ── Connect ────────────────────────────────────────────────────────
    try:
        db = get_db()
        print(f"Connected to MongoDB: {db.name}")
    except Exception as e:
        print(f"[ERROR] MongoDB connection failed: {e}")
        sys.exit(1)

    # ── Data inventory ─────────────────────────────────────────────────
    available_collections = set(db.list_collection_names())
    print(f"\nCollections found: {sorted(available_collections)}")
    print()

    # ── Extract ────────────────────────────────────────────────────────
    df_races   = extract_race_results(db, args.min_year, args.max_year)
    df_pace    = extract_pace_observations(db, args.min_year, args.max_year)
    df_circuit = extract_circuit_profiles(db)
    df_laps    = extract_lap_features(db, args.min_year, args.max_year)
    df_odds    = extract_odds_features(db, args.min_year, args.max_year)
    df_session = extract_session_stats(db, args.min_year, args.max_year)

    if df_races.empty:
        print("[ERROR] No race results found. Run import_jolpica.py first.")
        sys.exit(1)

    # ── Build feature matrix ───────────────────────────────────────────
    df = build_feature_matrix(df_races, df_pace, df_circuit, df_laps, df_odds, df_session)

    # Determine which numeric features are actually present
    available_features = [f for f in NUMERIC_FEATURES if f in df.columns]
    # Also add any circuit profile columns not in the static list
    for col in df.columns:
        if col not in available_features and col not in (
            "year", "round", "circuit_ref", "circuit_type",
            "driver_code", "constructor_ref", "finish_position",
            "is_dnf", "pos_gain", "points_scored", "laps_completed",
            "status_finished", TARGET_BINARY
        ):
            if pd.api.types.is_numeric_dtype(df[col]):
                available_features.append(col)

    # Add binary target
    df[TARGET_BINARY] = (df["finish_position"] <= 3).astype(int)

    print(f"\nFeature matrix: {len(df)} rows × {len(df.columns)} columns")
    print(f"Available features for analysis: {len(available_features)}")
    print(f"Target: {TARGET} (missing_as=21 for DNF)")
    print()

    # ── Save raw matrix ────────────────────────────────────────────────
    matrix_path = output_dir / "feature_matrix.csv"
    df.to_csv(matrix_path, index=False)
    log.info(f"Feature matrix saved → {matrix_path}")

    # ── Run analyses ───────────────────────────────────────────────────
    print("Running analyses …")

    print("  [1/5] Spearman correlation …")
    spearman_results = run_spearman_correlation(df, available_features)

    print("  [2/5] Mutual Information …")
    mi_results = run_mutual_information(df, available_features)

    print("  [3/5] Random Forest (MDI + Permutation) …")
    rf_result = run_random_forest(df, available_features)
    rf_mdi_results, perm_results, rf_meta = rf_result if len(rf_result) == 3 else ([], [], {})

    print("  [4/5] Gradient Boosting …")
    gb_results = run_gradient_boosting(df, available_features)

    print("  [5/5] Coverage analysis …")
    coverage_results = compute_coverage(df, available_features)

    # ── Aggregate rankings ─────────────────────────────────────────────
    print("\nAggregating rankings …")
    df_rankings = aggregate_rankings(
        spearman_results, mi_results, rf_mdi_results,
        perm_results, gb_results, coverage_results
    )

    rankings_path = output_dir / "feature_rankings.csv"
    df_rankings.to_csv(rankings_path, index=False)

    # ── Full JSON report ───────────────────────────────────────────────
    report = {
        "generated_at":   datetime.utcnow().isoformat() + "Z",
        "config": {
            "min_year":     args.min_year,
            "max_year":     args.max_year,
            "n_observations": int(len(df)),
            "n_features":   len(available_features),
            "target":       TARGET,
        },
        "data_sources": {
            "f1_races":             int(len(df_races)),
            "f1_pace_observations": int(len(df_pace)),
            "f1_circuit_profiles":  int(len(df_circuit)),
            "f1_lap_times":         int(len(df_laps)),
            "odds":                 int(len(df_odds)),
            "session_stats":        int(len(df_session)),
        },
        "rf_meta":           rf_meta,
        "top_features":      df_rankings.head(20).to_dict(orient="records"),
        "spearman":          spearman_results,
        "mutual_information": mi_results,
        "rf_mdi":            rf_mdi_results,
        "permutation":       perm_results,
        "gradient_boosting": gb_results,
        "coverage":          coverage_results,
        "full_rankings":     df_rankings.to_dict(orient="records"),
    }

    report_path = output_dir / "feature_importance_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Summary ────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("TOP 15 FEATURES BY COMPOSITE SCORE")
    print("=" * 70)
    print(f"{'Rank':<5} {'Feature':<30} {'Score':>7} {'Spear.':>8} {'MI':>7} {'RF':>7} {'Perm':>7} {'Cov%':>6}")
    print("-" * 70)
    for _, row in df_rankings.head(15).iterrows():
        print(
            f"{int(row['rank']):<5} {row['feature']:<30} "
            f"{row['composite_score']:>7.4f} "
            f"{row['spearman_norm']:>8.4f} "
            f"{row['mi_norm']:>7.4f} "
            f"{row['rf_mdi_norm']:>7.4f} "
            f"{row['perm_norm']:>7.4f} "
            f"{row['coverage_pct']:>5.0f}%"
        )

    print()
    print(f"RF cross-val MAE: {rf_meta.get('cv_mae', 'N/A')} positions  "
          f"(n={rf_meta.get('n_train', 'N/A')})")
    
    # ── Generate visualizations ────────────────────────────────────────
    print("\nGenerating visualizations …")
    viz_paths = generate_visualizations(df_rankings, output_dir, args.min_year, args.max_year)
    
    # Add visualization paths to report
    if viz_paths:
        report["visualizations"] = {
            "barchart": viz_paths.get('barchart', ''),
            "radar_chart": viz_paths.get('radar', ''),
            "heatmap": viz_paths.get('heatmap', ''),
            "scatter": viz_paths.get('scatter', ''),
            "method_comparison": viz_paths.get('method_comparison', ''),
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
    
    print()
    print("Output files:")
    print(f"  {report_path}")
    print(f"  {rankings_path}")
    print(f"  {matrix_path}")
    for name, path in viz_paths.items():
        print(f"  {name}: {path}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization functions
# ─────────────────────────────────────────────────────────────────────────────

def generate_visualizations(df_rankings: pd.DataFrame, output_dir: Path, 
                            min_year: int, max_year: int) -> dict:
    """
    Generate graphical visualizations for the feature analysis report.
    Returns dict with paths to generated files.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        log.warning(f"Matplotlib not available: {e}")
        return {}
    
    output_paths = {}
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Composite Score Bar Chart (Top 20)
    fig, ax = plt.subplots(figsize=(12, 8))
    top20 = df_rankings.head(20)
    colors = plt.cm.RdYlGn(top20['composite_score'].values / top20['composite_score'].max())
    bars = ax.barh(range(len(top20)), top20['composite_score'].values, color=colors)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Composite Score', fontsize=12)
    ax.set_title(f'F1 Feature Significance — Top 20 (Composite Score)\n{min_year}-{max_year}', 
                fontsize=14, fontweight='bold')
    for i, (score, cov) in enumerate(zip(top20['composite_score'], top20['coverage_pct'])):
        ax.text(score + 0.01, i, f' {score:.3f} ({cov:.0f}%)', va='center', fontsize=9)
    plt.tight_layout()
    path = output_dir / 'feature_importance_barchart.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    output_paths['barchart'] = str(path)
    
    # 2. Radar/Spider Chart for Top 5 Features
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    top5 = df_rankings.head(5)
    categories = ['Spearman', 'MI', 'RF-MDI', 'Permutation', 'GB', 'Coverage']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    colors_radar = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    for idx, (_, row) in enumerate(top5.iterrows()):
        values = [row['spearman_norm'], row['mi_norm'], row['rf_mdi_norm'],
                 row['perm_norm'], row['gb_norm'], row['coverage_pct'] / 100.0]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=row['feature'], 
                color=colors_radar[idx % len(colors_radar)])
        ax.fill(angles, values, alpha=0.1, color=colors_radar[idx % len(colors_radar)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title(f'Feature Comparison — Top 5\n{min_year}-{max_year}', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    path = output_dir / 'feature_radar_chart.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    output_paths['radar'] = str(path)
    
    # 3. Correlation Heatmap (method scores)
    fig, ax = plt.subplots(figsize=(10, 8))
    score_cols = ['spearman_norm', 'mi_norm', 'rf_mdi_norm', 'perm_norm', 'gb_norm', 'coverage_pct']
    methods = ['Spearman', 'MI', 'RF-MDI', 'Permutation', 'GB', 'Coverage%']
    top15 = df_rankings.head(15)
    heatmap_data = top15[score_cols].copy()
    heatmap_data.columns = methods
    
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', 
                ax=ax, cbar_kws={'label': 'Normalized Score'})
    ax.set_title(f'Feature Scores by Method — Top 15\n{min_year}-{max_year}', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    plt.tight_layout()
    path = output_dir / 'feature_heatmap.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    output_paths['heatmap'] = str(path)
    
    # 4. Coverage vs Score Scatter Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(df_rankings['coverage_pct'], df_rankings['composite_score'],
                       c=df_rankings['composite_score'], cmap='RdYlGn', 
                       s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    # Annotate top 10
    for _, row in df_rankings.head(10).iterrows():
        ax.annotate(row['feature'], (row['coverage_pct'], row['composite_score']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set_xlabel('Coverage %', fontsize=12)
    ax.set_ylabel('Composite Score', fontsize=12)
    ax.set_title(f'Coverage vs Significance\n{min_year}-{max_year}', 
                fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Composite Score')
    plt.tight_layout()
    path = output_dir / 'feature_coverage_scatter.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    output_paths['scatter'] = str(path)
    
    # 5. Method Agreement Stacked Bar
    fig, ax = plt.subplots(figsize=(12, 6))
    top20 = df_rankings.head(20)
    x = np.arange(len(top20))
    width = 0.15
    ax.bar(x - 2*width, top20['spearman_norm'], width, label='Spearman', color='#e74c3c')
    ax.bar(x - width, top20['mi_norm'], width, label='MI', color='#3498db')
    ax.bar(x, top20['rf_mdi_norm'], width, label='RF-MDI', color='#2ecc71')
    ax.bar(x + width, top20['perm_norm'], width, label='Permutation', color='#9b59b6')
    ax.bar(x + 2*width, top20['gb_norm'], width, label='Gradient Boosting', color='#f39c12')
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('Normalized Score', fontsize=12)
    ax.set_title(f'Method Scores by Feature — Top 20\n{min_year}-{max_year}', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top20['feature'], rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    path = output_dir / 'feature_method_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    output_paths['method_comparison'] = str(path)
    
    return output_paths


if __name__ == "__main__":
    main()
