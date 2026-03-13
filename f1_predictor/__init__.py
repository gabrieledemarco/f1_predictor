"""
f1_predictor - Sistema di predizione multistrato per Formula 1

Questo package implementa un sistema di machine learning a 4 layer per
predire i risultati delle gare di Formula 1 e identificare valore nelle quote.

Layer:
    1a: Driver Skill Rating (TrueSkill Through Time)
    1b: Machine Pace Estimation (Kalman Filter)
    2:  Bayesian Race Simulation (Monte Carlo)
    3:  Ensemble adjustment (Ridge meta-learner)
    4:  Calibration against market odds (Isotonic)

Uso principale:
    from f1_predictor.pipeline import F1PredictionPipeline
    
    pipeline = F1PredictionPipeline()
    pipeline.fit(historical_races)
    result = pipeline.predict_race(race, driver_grid, pinnacle_odds)
"""

__version__ = "2.0.0"
__author__ = "F1 Predictor Team"