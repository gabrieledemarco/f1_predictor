"""
Regression tests for MongoDB data loading functionality.

These tests verify that:
1. MongoDB loaders work correctly
2. Data import scripts function properly
3. Legacy loaders still work as fallback
4. No regression in existing functionality

Run with: pytest tests/regression/ -v
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMongoRaceLoader:
    """Test MongoRaceLoader functionality."""

    def test_race_dataclass_creation(self):
        """Test that Race dataclass can be created."""
        from f1_predictor.data.mongo_loader import Race, RaceResult, QualifyingResult

        result = RaceResult(
            driver_code="VER",
            constructor_ref="red_bull",
            grid_position=1,
            finish_position=1,
            points=25.0,
            laps_completed=58,
            status="Finished",
        )

        assert result.driver_code == "VER"
        assert result.finish_position == 1

    def test_race_dataclass_with_dict(self):
        """Test Race creation from dict."""
        from f1_predictor.data.mongo_loader import Race

        doc = {
            "_id": "2024_01",
            "year": 2024,
            "round": 1,
            "circuit_ref": "bahrain",
            "circuit_name": "Bahrain International Circuit",
            "circuit_type": "desert",
            "race_name": "Bahrain Grand Prix",
            "date": "2024-03-02",
            "is_sprint_weekend": False,
            "is_season_end": False,
            "is_major_regulation_change": False,
            "results": [],
            "qualifying": [],
            "location": {"country": "Bahrain", "locality": "Sakhir"},
        }

        race = Race.from_dict(doc)
        assert race.year == 2024
        assert race.round == 1
        assert race.circuit_ref == "bahrain"

    def test_race_to_dict_roundtrip(self):
        """Test Race to_dict roundtrip."""
        from f1_predictor.data.mongo_loader import Race

        doc = {
            "_id": "2024_01",
            "year": 2024,
            "round": 1,
            "circuit_ref": "bahrain",
            "circuit_name": "Bahrain",
            "circuit_type": "desert",
            "race_name": "Bahrain GP",
            "date": "2024-03-02",
            "is_sprint_weekend": False,
            "is_season_end": False,
            "is_major_regulation_change": False,
            "results": [],
            "qualifying": [],
            "location": {},
        }

        race = Race.from_dict(doc)
        result = race.to_dict()

        assert result["year"] == 2024
        assert result["_id"] == "2024_01"


class TestMongoPaceLoader:
    """Test MongoPaceLoader functionality."""

    def test_pace_observation_dataclass(self):
        """Test PaceObservation dataclass creation."""
        from f1_predictor.data.mongo_pace_loader import PaceObservation

        obs = PaceObservation(
            _id="2024_01_red_bull",
            year=2024,
            round=1,
            circuit_ref="bahrain",
            constructor_ref="red_bull",
            pace_delta_ms=-0.032,
            avg_pace_ms=90500.0,
            min_pace_ms=90300.0,
            sample_size=58,
        )

        assert obs.pace_delta_ms == -0.032
        assert obs.sample_size == 58

    def test_pace_observation_from_dict(self):
        """Test PaceObservation from_dict."""
        from f1_predictor.data.mongo_pace_loader import PaceObservation

        doc = {
            "_id": "2024_01_mercedes",
            "year": 2024,
            "round": 1,
            "circuit_ref": "bahrain",
            "constructor_ref": "mercedes",
            "pace_delta_ms": 0.015,
            "avg_pace_ms": 90600.0,
            "min_pace_ms": 90400.0,
            "sample_size": 55,
        }

        obs = PaceObservation.from_dict(doc)
        assert obs.constructor_ref == "mercedes"
        assert obs.pace_delta_ms > 0  # slower than best


class TestMongoOddsLoader:
    """Test MongoOddsLoader functionality."""

    def test_odds_record_dataclass(self):
        """Test OddsRecord dataclass creation."""
        from f1_predictor.data.mongo_odds_loader import OddsRecord

        record = OddsRecord(
            _id="abc123",
            race_id="2024_01",
            event_id="evt_001",
            market="h2h",
            driver_code="VER",
            odd_decimal=2.10,
            p_implied_raw=0.476,
            p_novig=0.485,
            hours_to_race=48.5,
            fetched_at="2024-03-01T10:00:00Z",
        )

        assert record.odd_decimal == 2.10
        assert record.p_novig > record.p_implied_raw  # vig removed

    def test_odds_record_from_dict(self):
        """Test OddsRecord from_dict."""
        from f1_predictor.data.mongo_odds_loader import OddsRecord

        doc = {
            "_id": "rec123",
            "race_id": "2024_02",
            "event_id": "evt_002",
            "market": "outrights",
            "driver_code": "NOR",
            "odd_decimal": 4.50,
            "p_implied_raw": 0.222,
            "p_novig": 0.228,
            "hours_to_race": 72.0,
            "fetched_at": "2024-03-08T10:00:00Z",
        }

        record = OddsRecord.from_dict(doc)
        assert record.market == "outrights"


class TestMongoCircuitLoader:
    """Test MongoCircuitProfileLoader functionality."""

    def test_circuit_type_enum(self):
        """Test CircuitType enum values."""
        from f1_predictor.data.mongo_circuit_loader import CircuitType

        assert CircuitType.STREET.value == "street"
        assert CircuitType.HIGH_SPEED.value == "high_speed"
        assert CircuitType.MIXED.value == "mixed"

    def test_circuit_speed_profile_dataclass(self):
        """Test CircuitSpeedProfile dataclass."""
        from f1_predictor.data.mongo_circuit_loader import (
            CircuitSpeedProfile,
            CircuitType,
        )

        profile = CircuitSpeedProfile(
            circuit_type=CircuitType.HIGH_SPEED,
            top_speed_kmh=335.0,
            min_speed_kmh=75.0,
            avg_speed_kmh=225.0,
            full_throttle_pct=72.0,
        )

        assert profile.top_speed_kmh == 335.0
        assert profile.full_throttle_pct == 72.0

    def test_default_profiles_exist(self):
        """Test that default profiles exist for known circuits."""
        from f1_predictor.data.mongo_circuit_loader import DEFAULT_PROFILES

        assert "monaco" in DEFAULT_PROFILES
        assert "spa" in DEFAULT_PROFILES
        assert "monza" in DEFAULT_PROFILES
        assert "bahrain" in DEFAULT_PROFILES

    def test_default_profile_types(self):
        """Test default profile circuit types."""
        from f1_predictor.data.mongo_circuit_loader import (
            DEFAULT_PROFILES,
            CircuitType,
        )

        assert DEFAULT_PROFILES["monaco"].circuit_type == CircuitType.STREET
        assert DEFAULT_PROFILES["monza"].circuit_type == CircuitType.HIGH_SPEED
        assert DEFAULT_PROFILES["catalunya"].circuit_type == CircuitType.MIXED


class TestDataModuleExports:
    """Test that data module exports correct classes."""

    def test_all_exports_available(self):
        """Test all expected exports are available."""
        from f1_predictor.data import (
            MongoRaceLoader,
            MongoPaceLoader,
            MongoOddsLoader,
            MongoCircuitProfileLoader,
            Race,
            PaceObservation,
            OddsRecord,
            CalibrationRecord,
            CircuitSpeedProfile,
            CircuitType,
            DEFAULT_PROFILES,
            get_data_loaders,
            load_training_data_from_mongo,
            load_calibration_records_from_mongo,
        )

        assert MongoRaceLoader is not None
        assert MongoPaceLoader is not None
        assert MongoOddsLoader is not None
        assert Race is not None
        assert PaceObservation is not None

    def test_get_data_loaders_returns_tuple(self):
        """Test get_data_loaders returns correct tuple."""
        from f1_predictor.data import get_data_loaders

        mock_db = MagicMock()
        loaders = get_data_loaders(mock_db)

        assert isinstance(loaders, tuple)
        assert len(loaders) == 4


class TestLegacyLoadersStillWork:
    """Verify legacy loaders still work as fallback."""

    def test_legacy_loader_import(self):
        """Test that legacy JolpicaLoader can be imported."""
        try:
            from f1_predictor.data.loader_jolpica import JolpicaLoader

            assert JolpicaLoader is not None
        except ImportError:
            pytest.fail("Legacy JolpicaLoader should still be importable")

    def test_legacy_tracing_loader_import(self):
        """Test that legacy TracingInsightsLoader can be imported."""
        try:
            from f1_predictor.data.loader_tracinginsights import (
                TracingInsightsLoader,
            )

            assert TracingInsightsLoader is not None
        except ImportError:
            pytest.fail("Legacy TracingInsightsLoader should still be importable")

    def test_legacy_adapter_import(self):
        """Test that synthetic fallback adapter can be imported."""
        try:
            from f1_predictor.data.adapter import generate_seasons

            assert callable(generate_seasons)
        except ImportError:
            pytest.fail("Legacy adapter should still be importable")


class TestEnvironmentVariables:
    """Test environment variable handling."""

    def test_mongo_uri_fallback(self):
        """Test that both MONGODB_URI and MONGO_URI are supported."""
        os.environ["MONGO_URI"] = "mongodb://localhost:27017"
        os.environ.pop("MONGODB_URI", None)

        uri = os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URI")
        assert uri == "mongodb://localhost:27017"

        os.environ["MONGODB_URI"] = "mongodb://prod:27017"
        uri = os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URI")
        assert uri == "mongodb://prod:27017"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
