"""
models/dnf_estimator.py
========================
Beta-Binomial estimator for driver/constructor DNF rates (TASK 5.3).

Replaces the hardcoded DNF_RATE_PER_LAP dictionary with a probabilistic
model that updates after each race.

Model:
    DNF_rate ~ Beta(α + failures, β + successes)
    
Prior: Beta(α=2, β=18) corresponds to 10% DNF rate per race (conservative).
Conversion from race-level DNF rate to per-lap rate:
    p_lap = 1 - (1 - p_race)^(1/57)  # 57 laps typical race

The estimator maintains separate counts for:
    1. Driver-specific DNF rates
    2. Constructor-specific DNF rates (for rookie drivers without history)
    3. Global baseline

Usage:
    estimator = BetaBinomialDNFRateEstimator()
    estimator.update(race_results)  # call after each race
    p_lap = estimator.get_dnf_rate_per_lap("verstappen", "red_bull", is_rookie=False)
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from collections import defaultdict
import math


class BetaBinomialDNFRateEstimator:
    """
    Bayesian estimator for DNF rates using Beta-Binomial conjugate prior.
    """
    
    def __init__(self, alpha_prior: float = 2.0, beta_prior: float = 18.0):
        """
        Args:
            alpha_prior: Prior successes (DNF events).
            beta_prior: Prior failures (finishes).
            Prior mean = alpha / (alpha + beta) = 0.1 (10% DNF per race).
        """
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        
        # Driver-specific counts: (failures, total_races)
        self.driver_counts: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
        
        # Constructor-specific counts
        self.constructor_counts: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
        
        # Global counts (for fallback)
        self.global_counts: tuple[int, int] = (0, 0)
        
        # Cache for per-lap rates
        self._cache: dict[str, float] = {}
    
    def update(self, race_results: list[dict]):
        """
        Update counts after a race.
        
        Args:
            race_results: List of dicts with keys:
                - driver_code
                - constructor_ref
                - finish_position (None if DNF)
                - status (optional)
        """
        for result in race_results:
            driver = result.get("driver_code", "").lower()
            constructor = result.get("constructor_ref", "").lower()
            is_dnf = result.get("finish_position") is None
            
            # Update driver counts
            d_fail, d_total = self.driver_counts[driver]
            d_fail += 1 if is_dnf else 0
            d_total += 1
            self.driver_counts[driver] = (d_fail, d_total)
            
            # Update constructor counts
            c_fail, c_total = self.constructor_counts[constructor]
            c_fail += 1 if is_dnf else 0
            c_total += 1
            self.constructor_counts[constructor] = (c_fail, c_total)
            
            # Update global counts
            g_fail, g_total = self.global_counts
            g_fail += 1 if is_dnf else 0
            g_total += 1
            self.global_counts = (g_fail, g_total)
        
        # Clear cache after update
        self._cache.clear()
    
    def get_dnf_rate_per_race(self, driver_code: str, constructor_ref: str,
                               is_rookie: bool = False) -> float:
        """
        Get estimated DNF rate per race (0-1).
        
        For rookie drivers with no history, use constructor rate with
        a rookie multiplier (1.5x).
        
        Args:
            driver_code: Driver code (lowercase).
            constructor_ref: Constructor reference (lowercase).
            is_rookie: Whether driver is a rookie (amplifies uncertainty).
            
        Returns:
            Estimated probability of DNF in a race.
        """
        driver = driver_code.lower()
        constructor = constructor_ref.lower()
        
        cache_key = f"{driver}:{constructor}:{is_rookie}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try driver-specific estimate
        d_fail, d_total = self.driver_counts[driver]
        if d_total >= 5:  # Enough data for driver-specific estimate
            alpha_post = self.alpha_prior + d_fail
            beta_post = self.beta_prior + (d_total - d_fail)
            rate = alpha_post / (alpha_post + beta_post)
        else:
            # Not enough driver data, use constructor rate
            c_fail, c_total = self.constructor_counts[constructor]
            if c_total >= 3:
                alpha_post = self.alpha_prior + c_fail
                beta_post = self.beta_prior + (c_total - c_fail)
                rate = alpha_post / (alpha_post + beta_post)
            else:
                # Use global rate
                g_fail, g_total = self.global_counts
                if g_total == 0:
                    rate = self.alpha_prior / (self.alpha_prior + self.beta_prior)
                else:
                    alpha_post = self.alpha_prior + g_fail
                    beta_post = self.beta_prior + (g_total - g_fail)
                    rate = alpha_post / (alpha_post + beta_post)
        
        # Rookie penalty: increase DNF rate by 50% (max 25%)
        if is_rookie:
            rate = min(rate * 1.5, 0.25)
        
        # Clip to reasonable range
        rate = np.clip(rate, 0.01, 0.35)
        
        self._cache[cache_key] = rate
        return rate
    
    def get_dnf_rate_per_lap(self, driver_code: str, constructor_ref: str,
                              is_rookie: bool = False, laps_per_race: int = 57) -> float:
        """
        Convert race-level DNF rate to per-lap rate.
        
        Assumption: DNF events are independent across laps.
            p_lap = 1 - (1 - p_race)^(1/laps_per_race)
        
        Args:
            driver_code: Driver code.
            constructor_ref: Constructor reference.
            is_rookie: Whether driver is a rookie.
            laps_per_race: Typical race length (default 57).
            
        Returns:
            Probability of DNF per lap.
        """
        p_race = self.get_dnf_rate_per_race(driver_code, constructor_ref, is_rookie)
        if p_race <= 0:
            return 0.0
        if p_race >= 1:
            return 1.0
        
        # Solve: 1 - (1 - p_lap)^laps = p_race
        # => p_lap = 1 - (1 - p_race)^(1/laps)
        p_lap = 1 - (1 - p_race) ** (1.0 / laps_per_race)
        return float(p_lap)
    
    def get_all_driver_rates(self) -> dict[str, float]:
        """Get DNF rate per race for all drivers with at least 1 race."""
        rates = {}
        for driver, (fail, total) in self.driver_counts.items():
            if total > 0:
                alpha_post = self.alpha_prior + fail
                beta_post = self.beta_prior + (total - fail)
                rates[driver] = alpha_post / (alpha_post + beta_post)
        return rates
    
    def load_from_historical_races(self, historical_races: list[dict]):
        """
        Initialize counts from historical race data.
        
        Args:
            historical_races: List of race dicts with 'results' key.
        """
        for race in historical_races:
            results = race.get("results", [])
            if results:
                self.update(results)


# Default global instance for backward compatibility
_default_estimator = BetaBinomialDNFRateEstimator()


def get_dnf_rate_per_lap(driver_code: str, constructor_ref: str = "unknown",
                         is_rookie: bool = False) -> float:
    """
    Convenience function for backward compatibility with DNF_RATE_PER_LAP.
    
    Returns per-lap DNF rate using the default estimator.
    """
    return _default_estimator.get_dnf_rate_per_lap(
        driver_code, constructor_ref, is_rookie
    )