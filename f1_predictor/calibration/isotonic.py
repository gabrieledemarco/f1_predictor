"""
calibration/isotonic.py
=======================
Layer 4: Pinnacle Calibration Layer (Isotonic Regression)

Maps raw model probabilities to calibrated probabilities that
are consistent with observed outcome frequencies — trained on
the gap between model predictions and Pinnacle's efficient market.

Academic References:
    Walsh, B. & Joshi, A. (2023). Machine learning for sports betting:
    should model selection be based on accuracy or calibration?
    arXiv:2303.06021.
    "Models optimised for calibration achieve average ROI of +34.69%
    vs −35.17% for accuracy-optimised models." — Table 2

    Platt, J. (1999). Probabilistic outputs for support vector machines.
    (Platt scaling — alternative calibration method included here)

Why Isotonic Regression:
    - Non-parametric: no assumption about the functional form of
      the miscalibration
    - Monotone: preserves rank ordering of probabilities
    - Piecewise constant: naturally handles the step-like structure
      of sports probabilities (clustering near round-number priors)
    - Computationally cheap: fits in O(n log n)

    vs Platt scaling (logistic): Platt assumes sigmoidal miscalibration
    which may not hold. Isotonic is more flexible but needs ~500+ samples
    for stable estimates — achievable across multiple seasons.

Calibration procedure:
    1. Collect (p_model, outcome) pairs from historical races
    2. Fit isotonic regression: f: [0,1] → [0,1]
    3. Apply: p_calibrated = f(p_model_raw)
    4. Compute edge: p_calibrated - p_pinnacle_novig
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from dataclasses import dataclass

from f1_predictor.domain.entities import CalibrationRecord


class PinnacleCalibrationLayer:
    """
    Layer 4: Isotonic calibration against Pinnacle market.

    Wraps sklearn's IsotonicRegression with additional diagnostics
    and fallback behaviour for insufficient data.

    Usage:
        calibrator = PinnacleCalibrationLayer()
        calibrator.fit(p_model_list, outcomes_list)
        p_cal = calibrator.transform(p_model_raw)
        edge = calibrator.compute_edge(p_cal, p_pinnacle)
    """

    def __init__(self, min_samples: int = 100, fallback: str = "passthrough"):
        """
        Args:
            min_samples: Minimum samples required for fitting.
                         If fewer samples, fall back to passthrough.
            fallback: 'passthrough' (return p_model unchanged) or
                     'platt' (fit logistic as alternative).
        """
        self.min_samples = min_samples
        self.fallback = fallback
        self._isotonic = None
        self._platt_a = 0.0
        self._platt_b = 0.0
        self._is_fitted = False
        self._n_samples = 0
        self._calibration_stats: Optional[dict] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, p_model: list[float],
            outcomes: list[int]) -> "PinnacleCalibrationLayer":
        """
        Fit the calibration layer.

        Args:
            p_model: Raw model probabilities (from Layer 3 ensemble).
            outcomes: Binary outcomes (1 = event occurred, 0 = did not).

        Returns:
            self (fluent interface).
        """
        from sklearn.isotonic import IsotonicRegression

        p = np.array(p_model, dtype=float)
        y = np.array(outcomes, dtype=float)
        self._n_samples = len(p)

        if self._n_samples < self.min_samples:
            if self.fallback == "platt":
                self._fit_platt(p, y)
            # else: passthrough (no fitting)
            self._is_fitted = False
            return self

        # Fit isotonic regression
        self._isotonic = IsotonicRegression(out_of_bounds="clip")
        self._isotonic.fit(p, y)
        self._is_fitted = True

        # Compute calibration statistics
        p_cal = self._isotonic.predict(p)
        self._calibration_stats = self._compute_calibration_stats(p, p_cal, y)

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def transform(self, p_model: list[float]) -> np.ndarray:
        """
        Calibrate raw model probabilities.

        Args:
            p_model: Raw probabilities from Layer 3.

        Returns:
            Calibrated probabilities in [0, 1].
        """
        p = np.array(p_model, dtype=float)

        if self._is_fitted and self._isotonic is not None:
            return self._isotonic.predict(p)
        elif self.fallback == "platt" and self._platt_a != 0:
            return self._apply_platt(p)
        else:
            return p  # passthrough

    def compute_edge(self, p_model: list[float],
                     p_pinnacle: list[float]) -> np.ndarray:
        """
        Compute edge for a list of driver × market combinations.

        Edge = p_calibrated - p_pinnacle_novig

        Positive edge → model sees higher probability than market.
        Negative edge → model sees lower probability (don't bet).

        Args:
            p_model: Raw model probabilities (will be calibrated internally).
            p_pinnacle: Devigged Pinnacle probabilities.

        Returns:
            Array of edge values (can be positive or negative).
        """
        p_cal = self.transform(p_model)
        p_pin = np.array(p_pinnacle, dtype=float)
        return p_cal - p_pin

    def get_calibration_report(self) -> dict:
        """
        Return calibration statistics for model evaluation.

        Includes:
            - Expected Calibration Error (ECE)
            - Brier Score
            - Mean edge (should be ~0 if well-calibrated)
            - Reliability by decile
        """
        if not self._calibration_stats:
            return {"status": "not_fitted", "n_samples": self._n_samples}
        return {
            "status": "fitted",
            "n_samples": self._n_samples,
            **self._calibration_stats
        }

    # ------------------------------------------------------------------
    # Significance testing
    # ------------------------------------------------------------------

    def permutation_test_edge(self, edges: np.ndarray,
                               outcomes: np.ndarray,
                               p_odds: np.ndarray,
                               n_permutations: int = 10_000) -> dict:
        """
        Test whether observed betting edge is statistically significant
        using a permutation test.

        Reference: Walsh & Joshi (2023) §3.2 — permutation test for
        betting strategy evaluation.

        H0: Edge is due to chance (ROI = 0 under flat staking).
        H1: Edge is real (ROI > 0 under flat staking).

        Args:
            edges: Array of edge values (positive = bet on this).
            outcomes: Binary outcomes (1 = won).
            p_odds: Decimal odds at time of bet.
            n_permutations: Bootstrap iterations.

        Returns:
            Dict with p-value, observed ROI, and significance flag.
        """
        # Only include positive-edge bets
        mask = edges > 0
        if mask.sum() < 5:
            return {"p_value": 1.0, "significant": False,
                    "reason": "insufficient_positive_edge_bets"}

        bet_outcomes = outcomes[mask]
        bet_odds = p_odds[mask]

        # Observed P&L under flat unit stake
        pnl = np.where(bet_outcomes == 1, bet_odds - 1, -1.0)
        observed_roi = pnl.mean()

        # Permutation distribution
        permuted_rois = []
        for _ in range(n_permutations):
            shuffled = np.random.permutation(bet_outcomes)
            perm_pnl = np.where(shuffled == 1, bet_odds - 1, -1.0)
            permuted_rois.append(perm_pnl.mean())

        permuted_rois = np.array(permuted_rois)
        p_value = float((permuted_rois >= observed_roi).mean())

        return {
            "observed_roi": float(observed_roi),
            "p_value": p_value,
            "significant": p_value < 0.05,
            "n_bets": int(mask.sum()),
            "permutation_mean_roi": float(permuted_rois.mean()),
            "permutation_95pct": float(np.percentile(permuted_rois, 95)),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_calibration_stats(self, p_raw: np.ndarray,
                                    p_cal: np.ndarray,
                                    outcomes: np.ndarray) -> dict:
        """Compute ECE, Brier score, and reliability by decile."""
        # Expected Calibration Error (ECE) — 10 buckets
        n_buckets = 10
        bucket_edges = np.linspace(0, 1, n_buckets + 1)
        ece_components = []

        reliability = []
        for i in range(n_buckets):
            lo, hi = bucket_edges[i], bucket_edges[i + 1]
            mask = (p_cal >= lo) & (p_cal < hi)
            if mask.sum() == 0:
                continue
            bucket_mean_prob = p_cal[mask].mean()
            bucket_freq = outcomes[mask].mean()
            bucket_n = mask.sum()
            ece_components.append(
                (bucket_n / len(p_cal)) * abs(bucket_mean_prob - bucket_freq)
            )
            reliability.append({
                "bucket": f"{lo:.1f}-{hi:.1f}",
                "mean_predicted": float(bucket_mean_prob),
                "observed_freq": float(bucket_freq),
                "n": int(bucket_n)
            })

        ece = float(sum(ece_components))

        # Brier score
        brier = float(np.mean((p_cal - outcomes) ** 2))

        # Mean absolute calibration error (pre vs post)
        brier_raw = float(np.mean((p_raw - outcomes) ** 2))

        return {
            "ece": ece,
            "brier_score": brier,
            "brier_raw": brier_raw,
            "brier_improvement": float(brier_raw - brier),
            "reliability_by_decile": reliability,
        }

    def _fit_platt(self, p: np.ndarray, y: np.ndarray):
        """Platt scaling as fallback when isotonic has insufficient data."""
        from scipy.optimize import minimize
        from scipy.special import expit

        def neg_log_likelihood(params):
            a, b = params
            logit_p = a * p + b
            probs = expit(logit_p)
            probs = np.clip(probs, 1e-8, 1 - 1e-8)
            return -np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))

        result = minimize(neg_log_likelihood, [1.0, 0.0], method="Nelder-Mead")
        self._platt_a, self._platt_b = result.x

    def _apply_platt(self, p: np.ndarray) -> np.ndarray:
        from scipy.special import expit
        return expit(self._platt_a * p + self._platt_b)
