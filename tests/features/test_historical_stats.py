"""
Test suite for historical_stats module (TASK 5.1).
"""

from __future__ import annotations
import pytest
import numpy as np
from dataclasses import dataclass
from typing import Optional

from f1_predictor.domain.entities import RaceResult
from f1_predictor.features.historical_stats import (
    DriverHistoricalFeatures,
    EloRatingSystem,
    DNFRateCalculator,
    H2HWinRateCalculator,
    compute_driver_historical_features,
)


def create_race_dict(
    race_id: int,
    driver_code: str,
    finish_position: Optional[int] = 1,
    status: str = "Finished",
    constructor_ref: str = "mercedes",
    grid_position: int = 1,
    points: float = 25.0,
    laps_completed: int = 58,
    fastest_lap_rank: Optional[int] = None,
) -> dict:
    """Helper to create a race result dict for testing."""
    return {
        "race_id": race_id,
        "driver_code": driver_code,
        "finish_position": finish_position,
        "status": status,
        "constructor_ref": constructor_ref,
        "grid_position": grid_position,
        "points": points,
        "laps_completed": laps_completed,
        "fastest_lap_rank": fastest_lap_rank,
    }


class TestEloRatingSystem:
    """Test EloRatingSystem."""
    
    def test_initial_ratings(self):
        """New driver should start at 1500."""
        elo = EloRatingSystem()
        assert elo.get_rating("ham") == 1500.0
        assert elo.get_rating("ver") == 1500.0
    
    def test_update_single_race(self):
        """Update with a single race result."""
        elo = EloRatingSystem()
        races = [
            create_race_dict(race_id=1, driver_code="ham", finish_position=1),
            create_race_dict(race_id=1, driver_code="ver", finish_position=2),
        ]
        elo.update(races, 1)
        # Winner should gain Elo points, loser may still gain but less than winner
        assert elo.get_rating("ham") > 1500.0
        assert elo.get_rating("ver") > 1500.0  # second place still positive performance
        assert elo.get_rating("ham") > elo.get_rating("ver")
    
    def test_update_multiple_drivers(self):
        """Update with multiple drivers."""
        elo = EloRatingSystem()
        races = [
            create_race_dict(race_id=1, driver_code="ham", finish_position=1),
            create_race_dict(race_id=1, driver_code="ver", finish_position=2),
            create_race_dict(race_id=1, driver_code="bot", finish_position=3),
        ]
        elo.update(races, 1)
        assert elo.get_rating("ham") > elo.get_rating("ver") > elo.get_rating("bot")
    
    def test_rolling_window(self):
        """TODO: Test rolling window with multiple drivers."""
        pass


class TestDNFRateCalculator:
    """Test DNFRateCalculator."""
    
    def test_initial_rates(self):
        """New driver should have prior DNF rate."""
        dnf = DNFRateCalculator()
        # Default prior should be around global average (0.1)
        assert dnf.get_dnf_rate("ham") == pytest.approx(0.1, abs=0.01)
    
    def test_update_with_dnf(self):
        """Update with a DNF (finish_position=None)."""
        dnf = DNFRateCalculator()
        race = create_race_dict(race_id=1, driver_code="ham", finish_position=None, status="Accident")
        dnf.update([race], 1, 2024)
        # DNF rate should increase
        assert dnf.get_dnf_rate("ham") > 0.05
    
    def test_update_with_finish(self):
        """Update with a finish (finish_position not None)."""
        dnf = DNFRateCalculator()
        race = create_race_dict(race_id=1, driver_code="ham", finish_position=1)
        dnf.update([race], 1, 2024)
        # DNF rate should decrease slightly from prior 0.1
        assert dnf.get_dnf_rate("ham") < 0.1
    
    def test_rolling_window(self):
        """TODO: Test rolling window expiration."""
        pass


class TestH2HWinRateCalculator:
    """Test H2HWinRateCalculator."""
    
    def test_initial_rate(self):
        """New driver should have default 0.5 win rate."""
        h2h = H2HWinRateCalculator()
        assert h2h.get_h2h_win_rate("ham") == 0.5
    
    def test_update_single_race(self):
        """Update with a single race where driver wins."""
        h2h = H2HWinRateCalculator()
        races = [
            create_race_dict(race_id=1, driver_code="ham", finish_position=1),
            create_race_dict(race_id=1, driver_code="ver", finish_position=2),
            create_race_dict(race_id=1, driver_code="bot", finish_position=3),
        ]
        h2h.update(races, 1, 2024)
        # ham above median (position 1 < median 2) → win
        assert h2h.get_h2h_win_rate("ham") > 0.5
        # ver at median (position 2 == median) → no win, win rate 0
        assert h2h.get_h2h_win_rate("ver") == 0.0
        # bot below median (position 3 > median) → no win, win rate 0
        assert h2h.get_h2h_win_rate("bot") == 0.0
    
    def test_update_with_dnf(self):
        """DNF counts as loss to all finishers."""
        h2h = H2HWinRateCalculator()
        races = [
            create_race_dict(race_id=1, driver_code="ham", finish_position=None),
            create_race_dict(race_id=1, driver_code="ver", finish_position=1),
        ]
        h2h.update(races, 1, 2024)
        # ham DNF: race counted but no win
        assert h2h.get_h2h_win_rate("ham") == 0.0
        # ver finished, but median = 1, position not less than median → no win
        assert h2h.get_h2h_win_rate("ver") == 0.0
    
    def test_season_window(self):
        """TODO: Test that races beyond 3 seasons are dropped."""
        pass


def test_compute_driver_historical_features():
    """TODO: Integration test for compute_driver_historical_features."""
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])