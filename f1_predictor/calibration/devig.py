"""
calibration/devig.py
====================
Devigging: Removing Bookmaker Margin from Quoted Odds

Academic Reference:
    Strumbelj, E. (2014). On Determining Probability Forecasts from
    Betting Odds. International Journal of Forecasting, 30(4), 934-943.
    https://doi.org/10.1016/j.ijforecast.2014.02.008

    "The Power method produces the most accurate fair probability
    estimates, and should be preferred when the goal is maximising
    predictive accuracy." — Strumbelj (2014), §4.3

Why devigging matters:
    A bookmaker's market for "Who wins?" over 20 drivers sums to ~103-105%
    in implied probability (not 100%). This 3-5% excess is the vig.
    If you compare your model's 100%-summing probabilities directly to
    the raw implied probs, the comparison is systematically biased.

    Example (Bahrain GP winner market):
        Raw odds → sum of 1/odd = 1.037 (3.7% vig)
        Simple normalisation: divide each by 1.037
        Power method: find k s.t. sum((1/odd)^k) = 1.0

Methods implemented:
    1. devig_power    — Strumbelj's recommended method
    2. devig_shin     — Shin (1993) model (accounts for insider trading)
    3. devig_basic    — Simple proportional normalisation (baseline)
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import brentq


def devig_power(odds: list[float]) -> list[float]:
    """
    Power devigging method (Strumbelj, 2014 — Eq. 3).

    Finds exponent k such that sum((1/odd_i)^k) = 1.0.
    This k is the vig-removal exponent: applying it uniformly
    to all implied probabilities gives a normalised distribution
    that preserves the relative probability ordering while removing
    the systematic bookmaker margin.

    Properties:
        - Preserves rank ordering of probabilities
        - Does not systematically bias favourites vs outsiders
        - Asymptotically equivalent to proportional normalisation
          when vig → 0

    Args:
        odds: List of decimal odds (e.g. [1.85, 4.50, 6.00, ...]).
              All odds must be > 1.0.

    Returns:
        List of fair probabilities summing to 1.0.

    Raises:
        ValueError: If odds list is empty or contains odds <= 1.0.
    """
    if not odds:
        raise ValueError("odds list cannot be empty")
    if any(o <= 1.0 for o in odds):
        raise ValueError("All odds must be > 1.0")

    raw_probs = [1.0 / o for o in odds]
    overround = sum(raw_probs)

    if abs(overround - 1.0) < 1e-6:
        return raw_probs  # no vig to remove

    def objective(k: float) -> float:
        return sum(p ** k for p in raw_probs) - 1.0

    try:
        k = brentq(objective, 0.3, 3.0, xtol=1e-8, maxiter=200)
    except ValueError:
        # Fallback to basic normalisation if brentq fails
        return devig_basic(odds)

    fair_probs = [p ** k for p in raw_probs]

    # Normalise to correct floating-point errors
    total = sum(fair_probs)
    return [p / total for p in fair_probs]


def devig_shin(odds: list[float]) -> list[float]:
    """
    Shin devigging method (Shin, 1993 — adapted by Strumbelj, 2014).

    The Shin model assumes the bookmaker faces a fraction z of informed
    bettors (insiders) and sets prices accordingly. The Shin-adjusted
    probabilities account for this insider information premium, which
    disproportionately affects high-probability outcomes.

    Reference:
        Shin, H.S. (1993). Measuring the Incidence of Insider Trading
        in a Market for State-Contingent Claims. Economic Journal.

    Note: Slightly more complex to compute than Power, and Strumbelj (2014)
    found it performs similarly. Included for comparison/diagnostics.

    Args:
        odds: List of decimal odds (> 1.0).

    Returns:
        List of Shin-adjusted fair probabilities summing to 1.0.
    """
    if not odds:
        raise ValueError("odds list cannot be empty")

    raw_probs = [1.0 / o for o in odds]
    overround = sum(raw_probs)
    n = len(odds)

    def objective(z: float) -> float:
        """Find z (insider fraction) s.t. Shin probs sum to 1."""
        shin_probs = []
        for p in raw_probs:
            numerator = np.sqrt(z**2 + 4 * (1 - z) * (p / overround) * p)
            shin_p = (numerator - z) / (2 * (1 - z))
            shin_probs.append(shin_p)
        return sum(shin_probs) - 1.0

    try:
        z = brentq(objective, 0.0, 0.5, xtol=1e-8)
    except ValueError:
        return devig_basic(odds)

    shin_probs = []
    for p in raw_probs:
        numerator = np.sqrt(z**2 + 4 * (1 - z) * (p / overround) * p)
        shin_p = (numerator - z) / (2 * (1 - z))
        shin_probs.append(shin_p)

    total = sum(shin_probs)
    return [p / total for p in shin_probs]


def devig_basic(odds: list[float]) -> list[float]:
    """
    Simple proportional normalisation (baseline/fallback).

    Divides each implied probability by the overround.
    Pros: Simple, always works. Cons: Systematically underestimates
    outsider probabilities and overestimates favourites.

    Reference: Discussed in Strumbelj (2014) §3.1 as the naive baseline.

    Args:
        odds: List of decimal odds.

    Returns:
        Normalised implied probabilities summing to 1.0.
    """
    raw_probs = [1.0 / o for o in odds]
    total = sum(raw_probs)
    return [p / total for p in raw_probs]


def compute_overround(odds: list[float]) -> float:
    """
    Compute the bookmaker's overround (vig) for a market.

    Overround = sum(1/odd_i) - 1.0
    Expressed as a percentage: 0.03 = 3% vig.

    Args:
        odds: List of decimal odds for a complete market.

    Returns:
        Overround as a decimal (e.g. 0.035 = 3.5% vig).
    """
    return sum(1.0 / o for o in odds) - 1.0


def implied_prob_novig(odd: float, all_odds: list[float],
                       method: str = "power") -> float:
    """
    Convenience function: compute fair probability for a single outcome
    within a market, using the specified devigging method.

    Args:
        odd: The decimal odd for the outcome of interest.
        all_odds: Complete list of odds for the market.
        method: "power" (default), "shin", or "basic".

    Returns:
        Fair (devigged) probability for the specified outcome.
    """
    fn = {"power": devig_power, "shin": devig_shin, "basic": devig_basic}
    if method not in fn:
        raise ValueError(f"method must be one of {list(fn.keys())}")

    fair_probs = fn[method](all_odds)
    idx = all_odds.index(odd)
    return fair_probs[idx]


def compare_methods(odds: list[float]) -> dict[str, list[float]]:
    """
    Diagnostic: compare all three devigging methods side-by-side.

    Useful for understanding the sensitivity of edge estimates to
    the devigging method chosen.

    Args:
        odds: List of decimal odds.

    Returns:
        Dict with keys 'power', 'shin', 'basic', each mapping to
        a list of fair probabilities.
    """
    return {
        "power": devig_power(odds),
        "shin": devig_shin(odds),
        "basic": devig_basic(odds),
        "raw": [1.0/o for o in odds],
        "overround": compute_overround(odds),
    }
