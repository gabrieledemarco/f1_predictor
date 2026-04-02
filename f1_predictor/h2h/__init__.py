"""
f1_predictor/h2h/__init__.py
Modulo H2H: calcolo probabilità e edge per mercati Head-to-Head.
"""
from .h2h_calculator import H2HCalculator, DriverH2H, ConstructorH2H
from .h2h_edge import H2HEdgeCalculator, H2HEdgeBet, H2HEdgeReport, devig_power

__all__ = [
    "H2HCalculator",
    "DriverH2H",
    "ConstructorH2H",
    "H2HEdgeCalculator",
    "H2HEdgeBet",
    "H2HEdgeReport",
    "devig_power",
]
