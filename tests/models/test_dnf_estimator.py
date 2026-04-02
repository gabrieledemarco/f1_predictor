"""
Test suite for dnf_estimator module (TASK 5.3).
"""

from __future__ import annotations
import pytest
import numpy as np
from typing import Optional

from f1_predictor.domain.entities import RaceResult
from f1_predictor.models.dnf_estimator import (
    BetaBinomialDNFRateEstimator,
    get_dnf_rate_per_lap,
)


def convert_race_dnf_rate_to_per_lap(race_rate: float, laps_per_race: int = 57) -> float:
    """Convert race-level DNF rate to per-lap rate."""
    if race_rate <= 0:
        return 0.0
    if race_rate >= 1:
        return 1.0
    return 1 - (1 - race_rate) ** (1 / laps_per_race)


def create_race_dict(
    race_id: int,
    driver_code: str,
    constructor_ref: str = "mercedes",
    finish_position: Optional[int] = 1,
    status: str = "Finished",
) -> dict:
    """Helper to create a race result dict for testing."""
    if finish_position is None:
        status = "Accident"
    return {
        "race_id": race_id,
        "driver_code": driver_code,
        "constructor_ref": constructor_ref,
        "finish_position": finish_position,
        "points": 25.0,
        "laps_completed": 58,
        "status": status,
        "fastest_lap_rank": None,
    }


class TestBetaBinomialDNFRateEstimator:
    """Test BetaBinomialDNFRateEstimator."""
    
    def test_initialization(self):
        """Test default priors."""
        estimator = BetaBinomialDNFRateEstimator()
        assert estimator.alpha_prior == 2.0
        assert estimator.beta_prior == 18.0
        assert estimator.driver_counts == {}
        assert estimator.constructor_counts == {}
    
    def test_update_with_finish(self):
        """Update with a finished race."""
        estimator = BetaBinomialDNFRateEstimator()
        race = create_race_dict(race_id=1, driver_code="ham", finish_position=1)
        estimator.update([race])
        # Driver should have 0 failures, 1 total
        assert estimator.driver_counts["ham"] == (0, 1)
        # Constructor should also be updated
        assert estimator.constructor_counts["mercedes"] == (0, 1)
    
    def test_update_with_dnf(self):
        """Update with a DNF."""
        estimator = BetaBinomialDNFRateEstimator()
        race = create_race_dict(race_id=1, driver_code="ham", finish_position=None)
        estimator.update([race])
        # Driver should have 1 failure, 1 total
        assert estimator.driver_counts["ham"] == (1, 1)
        assert estimator.constructor_counts["mercedes"] == (1, 1)
    
    def test_get_dnf_rate_per_lap_experienced_driver(self):
        """Test DNF rate for experienced driver with history."""
        estimator = BetaBinomialDNFRateEstimator()
        # Add 5 finishes, 1 DNF for driver
        for i in range(5):
            race = create_race_dict(race_id=i, driver_code="ham", finish_position=1)
            estimator.update([race])
        dnf_race = create_race_dict(race_id=5, driver_code="ham", finish_position=None)
        estimator.update([dnf_race])
        
        rate = estimator.get_dnf_rate_per_lap("ham", "mercedes", is_rookie=False)
        # Expected race-level DNF rate: (alpha_prior + 1) / (alpha_prior + beta_prior + 6) = (2+1)/(2+18+6) = 3/26 ≈ 0.115
        # Convert to per-lap rate
        expected_race_rate = (2.0 + 1) / (2.0 + 18.0 + 6)
        expected_per_lap = 1 - (1 - expected_race_rate) ** (1/57)
        assert rate == pytest.approx(expected_per_lap, rel=1e-3)
    
    def test_get_dnf_rate_per_lap_rookie(self):
        """Test rookie driver uses constructor rate with rookie penalty."""
        estimator = BetaBinomialDNFRateEstimator()
        # Add constructor history: 2 DNFs out of 10 races
        for i in range(8):
            race = create_race_dict(race_id=i, driver_code="some_driver", constructor_ref="ferrari", finish_position=1)
            estimator.update([race])
        for i in range(2):
            race = create_race_dict(race_id=i+8, driver_code="some_driver", constructor_ref="ferrari", finish_position=None)
            estimator.update([race])
        
        # Rookie driver with no personal history should use constructor rate
        rate = estimator.get_dnf_rate_per_lap("rookie", "ferrari", is_rookie=True)
        # Constructor rate: (alpha_prior + 2) / (alpha_prior + beta_prior + 10) = (2+2)/(2+18+10) = 4/30 ≈ 0.133
        base_race_rate = (2.0 + 2) / (2.0 + 18.0 + 10)
        # Rookie penalty: multiply by 1.5, cap at 0.25
        rookie_race_rate = min(base_race_rate * 1.5, 0.25)
        expected_per_lap = 1 - (1 - rookie_race_rate) ** (1/57)
        assert rate == pytest.approx(expected_per_lap, rel=0.1)
    
    def test_get_dnf_rate_per_lap_new_constructor(self):
        """Test new constructor falls back to global prior."""
        estimator = BetaBinomialDNFRateEstimator()
        rate = estimator.get_dnf_rate_per_lap("driver", "unknown_constructor", is_rookie=False)
        # Global prior race rate: alpha_prior / (alpha_prior + beta_prior) = 2/20 = 0.1
        expected_race_rate = 2.0 / 20.0
        expected_per_lap = 1 - (1 - expected_race_rate) ** (1/57)
        assert rate == pytest.approx(expected_per_lap, rel=1e-3)
    
    def test_rolling_window(self):
        """TODO: Test rolling window expiration."""
        pass


def test_convert_race_dnf_rate_to_per_lap():
    """Test conversion from race-level to per-lap rate."""
    # 10% race DNF rate, typical 57 laps
    per_lap = convert_race_dnf_rate_to_per_lap(0.1, laps_per_race=57)
    # Should be lower than race rate
    assert per_lap < 0.1
    # Should satisfy: (1 - per_lap)^57 = 1 - 0.1
    assert pytest.approx((1 - per_lap) ** 57, rel=1e-3) == 0.9
    
    # Edge case: 0% race DNF -> 0% per lap
    assert convert_race_dnf_rate_to_per_lap(0.0) == 0.0
    # Edge case: 100% race DNF -> 100% per lap
    assert convert_race_dnf_rate_to_per_lap(1.0) == 1.0


def test_get_dnf_rate_per_lap():
    """Test backward compatibility function."""
    # Should return same as estimator for known driver
    estimator = BetaBinomialDNFRateEstimator()
    race = create_race_dict(race_id=1, driver_code="ham", finish_position=1)
    estimator.update([race])
    
    rate_estimator = estimator.get_dnf_rate_per_lap("ham", "mercedes", is_rookie=False)
    rate_compat = get_dnf_rate_per_lap("ham", "mercedes", is_rookie=False)
    
    assert rate_compat == pytest.approx(rate_estimator, rel=0.1)
    
    # For unknown driver, should fall back to global prior per-lap rate
    default_rate = get_dnf_rate_per_lap("unknown", "unknown", is_rookie=False)
    expected_default = convert_race_dnf_rate_to_per_lap(2.0 / 20.0)  # alpha/(alpha+beta)
    assert default_rate == pytest.approx(expected_default, rel=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])