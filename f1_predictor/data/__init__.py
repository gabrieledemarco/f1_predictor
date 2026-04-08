"""
Data Module - Unified Data Loading

This module provides unified data loading from MongoDB.
All data is imported via GitHub Actions workflows and stored in MongoDB.

Usage in train_pipeline.py:
    from data import MongoRaceLoader, MongoPaceLoader, MongoOddsLoader
    
    # Get MongoDB connection
    db = get_db()
    
    # Load races
    race_loader = MongoRaceLoader(db)
    races = race_loader.load_seasons(years=range(2019, 2027), through_round=5)
    
    # Load pace observations
    pace_loader = MongoPaceLoader(db)
    pace_obs = pace_loader.load_pace_observations(years=range(2019, 2027))
    
    # Load odds
    odds_loader = MongoOddsLoader(db)
    odds = odds_loader.load_historical_odds(years=range(2022, 2027))
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)

# MongoDB Loaders
from f1_predictor.data.mongo_loader import (
    MongoRaceLoader,
    Race,
    RaceResult,
    QualifyingResult,
)

from f1_predictor.data.mongo_pace_loader import (
    MongoPaceLoader,
    PaceObservation,
)

from f1_predictor.data.mongo_odds_loader import (
    MongoOddsLoader,
    OddsRecord,
    CalibrationRecord,
)

from f1_predictor.data.mongo_circuit_loader import (
    MongoCircuitProfileLoader,
    CircuitSpeedProfile,
    CircuitType,
    DEFAULT_PROFILES,
)


def get_data_loaders(db):
    """
    Returns a tuple of MongoDB data loaders.
    
    Usage:
        db = get_db()
        race_loader, pace_loader, odds_loader, circuit_loader = get_data_loaders(db)
    """
    return (
        MongoRaceLoader(db),
        MongoPaceLoader(db),
        MongoOddsLoader(db),
        MongoCircuitProfileLoader(db),
    )


def load_training_data_from_mongo(
    db,
    years: range | list[int] = range(2019, 2027),
    through_round: Optional[int] = None,
    jolpica_cache: Optional[str] = None,
    tracinginsights_dir: Optional[str] = None,
    use_synthetic_fallback: bool = True,
    force_refresh: bool = False,
) -> list[dict]:
    """
    Load training data from MongoDB.
    
    Args:
        db: MongoDB database connection
        years: Range of seasons to load
        through_round: Truncate last season to this round
        jolpica_cache: Ignored (for compatibility with train_pipeline.py)
        tracinginsights_dir: Ignored (for compatibility with train_pipeline.py)
        use_synthetic_fallback: If True and no data found, return empty list
        force_refresh: Ignored (for compatibility)
        
    Returns:
        List of race dicts ready for F1PredictionPipeline.fit()
    """
    race_loader = MongoRaceLoader(db)
    pace_loader = MongoPaceLoader(db)
    
    years_list = list(years)
    
    log.info(f"[DataLoader] Loading {years_list} from MongoDB...")
    
    races = race_loader.load_seasons(years_list, through_round)
    
    if not races:
        log.warning("[DataLoader] No races found in MongoDB")
        return []
    
    pace_obs = pace_loader.load_pace_observations(years_list)
    
    race_dicts = []
    for race in races:
        race_dict = race.to_dict()
        
        race_pace = pace_loader.load_race_pace(race.year, race.round)
        if race_pace:
            race_dict["constructor_pace_observations"] = race_pace
        
        race_dicts.append(race_dict)
    
    enriched_count = sum(1 for r in race_dicts if r.get("constructor_pace_observations"))
    log.info(
        f"[DataLoader] {len(race_dicts)} races loaded "
        f"({enriched_count} with pace data)"
    )
    
    return race_dicts


def load_calibration_records_from_mongo(
    db,
    years: range | list[int] = range(2022, 2027),
    market: Optional[str] = None,
    min_hours_to_race: float = 24.0,
) -> list[CalibrationRecord]:
    """
    Load calibration records from MongoDB.
    
    Args:
        db: MongoDB database connection
        years: Range of seasons
        market: Filter by market (h2h, outrights)
        min_hours_to_race: Minimum hours before race
        
    Returns:
        List of OddsRecord for calibration
    """
    odds_loader = MongoOddsLoader(db)
    years_list = list(years)
    
    log.info(f"[DataLoader] Loading odds for {years_list}...")
    
    records = odds_loader.load_historical_odds(
        years=years_list,
        market=market,
        min_hours_to_race=min_hours_to_race,
    )
    
    log.info(f"[DataLoader] {len(records)} odds records loaded")
    
    return records


load_training_data = load_training_data_from_mongo

__all__ = [
    "MongoRaceLoader",
    "MongoPaceLoader",
    "MongoOddsLoader",
    "MongoCircuitProfileLoader",
    "Race",
    "RaceResult",
    "QualifyingResult",
    "PaceObservation",
    "OddsRecord",
    "CalibrationRecord",
    "CircuitSpeedProfile",
    "CircuitType",
    "DEFAULT_PROFILES",
    "get_data_loaders",
    "load_training_data_from_mongo",
    "load_calibration_records_from_mongo",
]
