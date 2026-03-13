"""
validation - Validazione walk‑forward, backtesting e metriche

Esporta:
    - WalkForwardValidator: Validazione temporale con expanding window
    - BettingBacktester: Simulazione scommesse con Kelly frazionario
    - compute_metrics: Metriche di performance (Brier, AUC, ROI, ECE)
"""

from f1_predictor.validation.walk_forward import WalkForwardValidator
from f1_predictor.validation.backtesting import BettingBacktester, BacktestConfig
from f1_predictor.validation.metrics import compute_metrics

__all__ = [
    "WalkForwardValidator",
    "BettingBacktester",
    "BacktestConfig",
    "compute_metrics",
]