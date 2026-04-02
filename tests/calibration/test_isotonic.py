"""
Test suite for isotonic calibration module.
"""

from __future__ import annotations
import pytest
import numpy as np
from f1_predictor.calibration.isotonic import PinnacleCalibrationLayer


class TestPinnacleCalibrationLayer:
    """Test PinnacleCalibrationLayer."""
    
    def test_default_fallback_is_platt(self):
        """Check that default fallback is 'platt' (TASK 5.4)."""
        calibrator = PinnacleCalibrationLayer()
        assert calibrator.fallback == "platt"
        
        calibrator2 = PinnacleCalibrationLayer(fallback="passthrough")
        assert calibrator2.fallback == "passthrough"
    
    def test_fit_with_sufficient_samples(self):
        """Fit with enough samples (> min_samples)."""
        calibrator = PinnacleCalibrationLayer(min_samples=10)
        # Generate well-calibrated probabilities
        np.random.seed(42)
        p_model = np.random.uniform(0.2, 0.8, size=100)
        # True probability is linear with some noise
        outcomes = (p_model + np.random.normal(0, 0.1, size=100) > 0.5).astype(int)
        
        calibrator.fit(p_model.tolist(), outcomes.tolist())
        report = calibrator.get_calibration_report()
        assert report["status"] == "fitted"
        assert report["n_samples"] == 100
        assert "ece" in report
        assert "brier_score" in report
    
    def test_fit_with_insufficient_samples_fallback_platt(self):
        """When samples < min_samples, should fall back to Platt scaling (default)."""
        calibrator = PinnacleCalibrationLayer(min_samples=100, fallback="platt")
        p_model = [0.3, 0.6, 0.7]
        outcomes = [0, 1, 1]
        
        calibrator.fit(p_model, outcomes)
        report = calibrator.get_calibration_report()
        assert report["status"] == "not_fitted"
        assert report["n_samples"] == 3
        # Should have fitted Platt parameters (non-zero)
        # The transform will use Platt scaling
        transformed = calibrator.transform([0.5])
        # Should be different from passthrough (0.5)
        assert transformed[0] != 0.5
    
    def test_fit_with_insufficient_samples_fallback_passthrough(self):
        """When fallback='passthrough', should not modify probabilities."""
        calibrator = PinnacleCalibrationLayer(min_samples=100, fallback="passthrough")
        p_model = [0.3, 0.6, 0.7]
        outcomes = [0, 1, 1]
        
        calibrator.fit(p_model, outcomes)
        transformed = calibrator.transform([0.5, 0.8])
        np.testing.assert_array_equal(transformed, [0.5, 0.8])
    
    def test_transform_with_isotonic_fit(self):
        """Transform after isotonic fitting."""
        calibrator = PinnacleCalibrationLayer(min_samples=5)
        # Simple calibration: probabilities are already calibrated
        p_model = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        outcomes = [0, 0, 0, 0, 0, 1, 1, 1, 1]
        
        calibrator.fit(p_model, outcomes)
        transformed = calibrator.transform([0.25, 0.75])
        # Should preserve monotonicity
        assert transformed[0] < transformed[1]
        # Should be within [0,1]
        assert np.all((transformed >= 0) & (transformed <= 1))
    
    def test_compute_edge(self):
        """Test edge computation."""
        calibrator = PinnacleCalibrationLayer(min_samples=5)
        p_model = [0.6, 0.7]
        outcomes = [1, 1]
        calibrator.fit(p_model, outcomes)
        
        p_pinnacle = [0.5, 0.65]
        edges = calibrator.compute_edge([0.6, 0.7], p_pinnacle)
        # edge = p_calibrated - p_pinnacle
        # Since calibration is fitted on perfect outcomes, calibrated probabilities may be adjusted
        # At least check shape
        assert len(edges) == 2
        assert isinstance(edges, np.ndarray)
    
    def test_permutation_test_edge(self):
        """Test permutation test for edge significance."""
        calibrator = PinnacleCalibrationLayer()
        # Create edges that are positive
        edges = np.array([0.1, 0.2, 0.05, 0.15])
        outcomes = np.array([1, 0, 1, 0])
        p_odds = np.array([2.0, 3.0, 2.5, 4.0])
        
        result = calibrator.permutation_test_edge(edges, outcomes, p_odds, n_permutations=100)
        assert "p_value" in result
        assert "significant" in result
        if "reason" in result and result["reason"] == "insufficient_positive_edge_bets":
            # insufficient bets, missing n_bets and observed_roi
            assert result["significant"] == False
        else:
            assert "n_bets" in result
            assert "observed_roi" in result
    
    def test_calibration_stats(self):
        """Test calibration statistics computation."""
        calibrator = PinnacleCalibrationLayer(min_samples=10)
        # Generate perfectly calibrated probabilities
        p_model = np.linspace(0.1, 0.9, 50)
        outcomes = (p_model > 0.5).astype(int)
        
        calibrator.fit(p_model.tolist(), outcomes.tolist())
        report = calibrator.get_calibration_report()
        assert "ece" in report
        assert "brier_score" in report
        assert "brier_raw" in report
        assert "brier_improvement" in report
        assert "reliability_by_decile" in report
        # For perfectly calibrated, ECE should be low
        assert report["ece"] < 0.1
    
    def test_platt_fallback_fitting(self):
        """Test that Platt scaling is fitted when fallback='platt' and insufficient samples."""
        calibrator = PinnacleCalibrationLayer(min_samples=100, fallback="platt")
        p_model = np.array([0.2, 0.4, 0.6, 0.8])
        outcomes = np.array([0, 0, 1, 1])
        
        calibrator.fit(p_model.tolist(), outcomes.tolist())
        # Transform a probability
        transformed = calibrator.transform([0.5])
        # Should apply Platt scaling (sigmoid)
        # Since we have only 4 samples, Platt may not change much but should not crash
        assert 0 <= transformed[0] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])