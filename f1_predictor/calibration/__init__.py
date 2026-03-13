"""
calibration - Layer 4: Calibrazione contro quote di mercato

Esporta:
    - devig_power: Correzione margine bookmaker (de‑vigorish)
    - PinnacleCalibrationLayer: Calibratore isotonico vs quote Pinnacle
    - BetaBinomialEdgeTracker: Tracciamento edge e ROI
"""

from f1_predictor.calibration.devig import devig_power
from f1_predictor.calibration.isotonic import PinnacleCalibrationLayer
from f1_predictor.calibration.edge_tracker import BetaBinomialEdgeTracker

__all__ = [
    "devig_power",
    "PinnacleCalibrationLayer",
    "BetaBinomialEdgeTracker",
]