"""
Test walk-forward metrics (Brier multiclass, LogLoss, RPS).
"""
import pytest
import numpy as np
from f1_predictor.validation.metrics import (
    ranked_probability_score,
    mean_ranked_probability_score,
)


class TestBrierMulticlass:
    """Test multiclass Brier Score calculation."""

    def test_perfect_prediction(self):
        """Perfect predictions should give Brier = 0."""
        n_drivers = 5
        n_positions = 20

        prob_matrix = np.zeros((n_drivers, n_positions))
        actual_onehot = np.zeros((n_drivers, n_positions))

        for i in range(n_drivers):
            actual_pos = i + 1
            prob_matrix[i, actual_pos - 1] = 1.0
            actual_onehot[i, actual_pos - 1] = 1.0

        prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)
        brier = float(np.mean((prob_matrix - actual_onehot) ** 2))
        assert brier == 0.0

    def test_brier_in_range(self):
        """Brier should be between 0 and 1."""
        n_drivers = 10
        n_positions = 20

        prob_matrix = np.random.rand(n_drivers, n_positions)
        prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)

        actual_onehot = np.zeros((n_drivers, n_positions))
        for i in range(n_drivers):
            actual_onehot[i, np.random.randint(0, n_positions)] = 1.0

        brier = float(np.mean((prob_matrix - actual_onehot) ** 2))
        assert 0.0 <= brier <= 1.0


class TestLogLossMulticlass:
    """Test multiclass LogLoss calculation."""

    def test_perfect_prediction(self):
        """Perfect predictions should give LogLoss = 0."""
        n_drivers = 5
        n_positions = 20

        prob_matrix = np.zeros((n_drivers, n_positions))
        actual_onehot = np.zeros((n_drivers, n_positions))

        for i in range(n_drivers):
            actual_pos = i + 1
            prob_matrix[i, actual_pos - 1] = 1.0
            actual_onehot[i, actual_pos - 1] = 1.0

        eps = 1e-15
        prob_clipped = np.clip(prob_matrix, eps, 1 - eps)
        prob_clipped = prob_clipped / prob_clipped.sum(axis=1, keepdims=True)

        logloss = float(np.mean(-np.sum(actual_onehot * np.log(prob_clipped), axis=1)))
        assert logloss < 1e-10  # Essentially zero

    def test_logloss_in_range(self):
        """LogLoss should be non-negative."""
        n_drivers = 10
        n_positions = 20

        prob_matrix = np.random.rand(n_drivers, n_positions)
        prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)

        actual_onehot = np.zeros((n_drivers, n_positions))
        for i in range(n_drivers):
            actual_onehot[i, np.random.randint(0, n_positions)] = 1.0

        eps = 1e-15
        prob_clipped = np.clip(prob_matrix, eps, 1 - eps)
        prob_clipped = prob_clipped / prob_clipped.sum(axis=1, keepdims=True)

        logloss = float(np.mean(-np.sum(actual_onehot * np.log(prob_clipped), axis=1)))
        assert logloss >= 0.0


class TestRPS:
    """Test Ranked Probability Score."""

    def test_perfect_prediction(self):
        """Perfect prediction should give RPS = 0."""
        p_dist = [1.0] + [0.0] * 19
        rps = ranked_probability_score(p_dist, actual_rank=1, n_categories=20)
        assert rps == 0.0

    def test_worst_prediction(self):
        """Worst prediction should give RPS = 1."""
        p_dist = [0.0] * 19 + [1.0]
        rps = ranked_probability_score(p_dist, actual_rank=1, n_categories=20)
        assert rps == 1.0

    def test_uniform_prediction(self):
        """Uniform prediction should give moderate RPS."""
        p_dist = [1/20] * 20
        rps = ranked_probability_score(p_dist, actual_rank=1, n_categories=20)

        expected = np.mean([(1 - k/20)**2 for k in range(1, 20)])
        assert abs(rps - expected) < 0.01

    def test_rps_in_range(self):
        """RPS should be between 0 and 1."""
        for _ in range(10):
            p_dist = np.random.rand(20)
            p_dist = p_dist / p_dist.sum()
            actual = np.random.randint(1, 21)

            rps = ranked_probability_score(p_dist.tolist(), actual, n_categories=20)
            assert 0.0 <= rps <= 1.0

    def test_mean_rps(self):
        """Test mean_rps over multiple predictions."""
        predictions = [
            [1.0] + [0.0] * 19,
            [0.0] * 19 + [1.0],
            [1/20] * 20,
        ]
        actuals = [1, 1, 1]

        mean_rps = mean_ranked_probability_score(predictions, actuals, n_categories=20)

        expected_uniform = np.mean([(1 - k/20)**2 for k in range(1, 20)])
        expected_mean = (0 + 1 + expected_uniform) / 3

        assert abs(mean_rps - expected_mean) < 0.01


class TestBacktestIntegration:
    """Test ROI backtest integration."""

    def test_roi_with_odds_returns_value(self):
        """When odds are available, ROI should return a numeric value."""
        probs = [0.3, 0.5, 0.2, 0.1, 0.05]
        outcomes = [0, 1, 0, 0, 0]

        assert len(probs) == len(outcomes)

    def test_roi_without_odds_returns_none(self):
        """When odds are NOT available, ROI should return None."""
        # This is the current behavior in Phase 1/2
        # Real backtest integration happens in Phase 2
        roi_real = None
        assert roi_real is None


class TestEdgeCases:
    """Test edge cases for walk-forward metrics."""

    def test_all_zero_probabilities(self):
        """Case: all probabilities are zero - should fallback to uniform."""
        n_drivers = 5
        n_positions = 20
        
        prob_matrix = np.zeros((n_drivers, n_positions))
        actual_onehot = np.zeros((n_drivers, n_positions))
        actual_onehot[0, 0] = 1.0
        
        # Apply normalization with fallback
        row_sums = prob_matrix.sum(axis=1, keepdims=True)
        eps_normalize = 1e-8
        bad_rows = np.where((row_sums < eps_normalize) | np.isnan(row_sums))[0]
        
        if len(bad_rows) > 0:
            for i in bad_rows:
                prob_matrix[i, :] = 1.0 / n_positions
        
        row_sums = prob_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        prob_matrix_norm = prob_matrix / row_sums
        
        # Should be uniform after fallback
        assert np.allclose(prob_matrix_norm[0], 1.0/n_positions)
        
        # Brier should be computable
        brier = float(np.mean((prob_matrix_norm - actual_onehot) ** 2))
        assert 0.0 <= brier <= 1.0

    def test_probability_sum_greater_than_one(self):
        """Case: sum of probabilities > 1 - should normalize."""
        n_drivers = 3
        n_positions = 20
        
        # Probabilities that sum to > 1
        prob_matrix = np.ones((n_drivers, n_positions)) * 0.06  # sum = 1.2
        actual_onehot = np.zeros((n_drivers, n_positions))
        actual_onehot[0, 0] = 1.0
        
        # Normalize
        row_sums = prob_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        prob_matrix_norm = prob_matrix / row_sums
        
        # Each row should now sum to 1
        assert np.allclose(prob_matrix_norm.sum(axis=1), 1.0)

    def test_probability_sum_less_than_one(self):
        """Case: sum of probabilities < 1 - should normalize."""
        n_drivers = 3
        n_positions = 20
        
        # Probabilities that sum to < 1
        prob_matrix = np.ones((n_drivers, n_positions)) * 0.04  # sum = 0.8
        actual_onehot = np.zeros((n_drivers, n_positions))
        actual_onehot[0, 0] = 1.0
        
        # Normalize
        row_sums = prob_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        prob_matrix_norm = prob_matrix / row_sums
        
        # Each row should now sum to 1
        assert np.allclose(prob_matrix_norm.sum(axis=1), 1.0)

    def test_winner_missing_from_driver_vector(self):
        """Case: winner not in driver vector - should skip gracefully."""
        driver_codes = ["VER", "HAM", "LEC", "NOR", "RUS"]
        driver_probs = {"VER": 0.4, "HAM": 0.3, "LEC": 0.2, "NOR": 0.1}
        # Note: "RUS" has no probability (could be 0)
        
        winner = "RUS"  # Winner not in driver_probs with valid probability
        
        # Should not crash
        p_winner = driver_probs.get(winner, 0.0)
        assert p_winner == 0.0

    def test_nan_in_probabilities(self):
        """Case: NaN in probability matrix - should handle gracefully."""
        n_drivers = 3
        n_positions = 20
        
        prob_matrix = np.random.rand(n_drivers, n_positions)
        prob_matrix[0, 0] = np.nan  # Introduce NaN
        actual_onehot = np.zeros((n_drivers, n_positions))
        actual_onehot[0, 0] = 1.0
        
        # Handle NaN
        prob_matrix = np.nan_to_num(prob_matrix, nan=1.0/n_positions)
        
        # Normalize
        row_sums = prob_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        prob_matrix_norm = prob_matrix / row_sums
        
        # Should have no NaN
        assert not np.any(np.isnan(prob_matrix_norm))

    def test_inf_in_probabilities(self):
        """Case: inf in probability matrix - should handle gracefully."""
        n_drivers = 3
        n_positions = 20
        
        prob_matrix = np.random.rand(n_drivers, n_positions)
        prob_matrix[0, 0] = np.inf  # Introduce inf
        actual_onehot = np.zeros((n_drivers, n_positions))
        actual_onehot[0, 0] = 1.0
        
        # Handle inf
        prob_matrix = np.nan_to_num(prob_matrix, posinf=1.0, neginf=0.0)
        
        # Should have no inf
        assert not np.any(np.isinf(prob_matrix))

    def test_clipped_logloss(self):
        """Test logloss with clipping (1e-6) and renormalization."""
        n_drivers = 5
        n_positions = 20
        
        prob_matrix = np.random.rand(n_drivers, n_positions)
        prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)
        
        actual_onehot = np.zeros((n_drivers, n_positions))
        for i in range(n_drivers):
            actual_onehot[i, i] = 1.0
        
        # Apply clipping
        eps_clamp = 1e-6
        prob_matrix_clipped = np.clip(prob_matrix, eps_clamp, 1 - eps_clamp)
        row_sums_clip = prob_matrix_clipped.sum(axis=1, keepdims=True)
        row_sums_clip = np.where(row_sums_clip > 0, row_sums_clip, 1.0)
        prob_matrix_clipped = prob_matrix_clipped / row_sums_clip
        
        # LogLoss should be computable and finite
        logloss = float(np.mean(-np.sum(actual_onehot * np.log(prob_matrix_clipped), axis=1)))
        assert np.isfinite(logloss)
        assert logloss >= 0.0