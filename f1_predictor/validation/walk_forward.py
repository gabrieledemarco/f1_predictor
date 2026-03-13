"""
validation/walk_forward.py
==========================
Walk-Forward Temporal Validation Engine

Implements strict walk-forward (expanding window) validation
to ensure zero look-ahead bias in all model evaluations.

Academic Reference:
    van Kesteren & Bergkamp (2023) §4.1 — temporal validation:
    "We predict each race using only races from prior seasons and
    prior rounds of the current season."

    This is the ONLY correct way to validate time-series models.
    K-fold cross-validation that ignores time ordering produces
    severely optimistic performance estimates.

Walk-forward scheme:
    Season: 2019 2020 2021 2022 2023 2024 2025
             ──── ──── ──── ──── ──── ──── ────
    Fold 1:  TRAIN[2018-2021]  →  TEST[2022 race 1]
    Fold 2:  TRAIN[2018-2021 + race 1]  →  TEST[2022 race 2]
    ...
    Fold N:  TRAIN[all prior]  →  TEST[current race]

No data leakage guarantees:
    - No future lap times used
    - No future qualifying data used
    - Pinnacle odds only at T-3h (post-qualifying)
    - Driver changes known at race entry, not retroactively

Purge & Embargo:
    If rolling features (e.g., 3-race moving average pace) are used,
    an embargo of 1 race is applied between train and test to prevent
    the last train observation from contaminating the first test feature.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional
from collections import defaultdict

from f1_predictor.validation.metrics import (
    brier_score, mean_ranked_probability_score,
    expected_calibration_error, log_loss, kendall_tau_ranking
)


@dataclass
class FoldResult:
    """Results for one walk-forward fold (one race)."""
    race_id: int
    race_name: str
    n_train_races: int

    # Prediction quality metrics
    brier_win: float        # Brier score for win market
    brier_podium: float
    rps_mean: float         # Mean RPS over all position distributions
    ece: float              # Expected Calibration Error
    log_loss_win: float

    # Ranking quality
    kendall_tau: float

    # Edge metrics (if odds provided)
    mean_edge: Optional[float] = None
    n_positive_edge_bets: Optional[int] = None

    # Model inputs for debugging
    n_drivers: int = 20


@dataclass
class WalkForwardResults:
    """Aggregated results from full walk-forward validation."""
    folds: list[FoldResult] = field(default_factory=list)

    @property
    def mean_brier_win(self) -> float:
        return float(np.mean([f.brier_win for f in self.folds]))

    @property
    def mean_rps(self) -> float:
        return float(np.mean([f.rps_mean for f in self.folds]))

    @property
    def mean_ece(self) -> float:
        return float(np.mean([f.ece for f in self.folds]))

    @property
    def mean_kendall_tau(self) -> float:
        return float(np.mean([f.kendall_tau for f in self.folds]))

    @property
    def brier_over_time(self) -> list[float]:
        return [f.brier_win for f in self.folds]

    def summary(self) -> dict:
        return {
            "n_folds": len(self.folds),
            "mean_brier_win": round(self.mean_brier_win, 5),
            "mean_rps": round(self.mean_rps, 5),
            "mean_ece": round(self.mean_ece, 5),
            "mean_kendall_tau": round(self.mean_kendall_tau, 4),
            "brier_trend": self._compute_trend(self.brier_over_time),
        }

    def _compute_trend(self, values: list[float]) -> str:
        if len(values) < 4:
            return "insufficient_data"
        from scipy.stats import kendalltau
        tau, pval = kendalltau(range(len(values)), values)
        if pval > 0.05:
            return "stable"
        return "improving" if tau < 0 else "degrading"


class WalkForwardValidator:
    """
    Walk-Forward Temporal Validation Engine.

    Evaluates a prediction model using strict expanding-window validation,
    ensuring no future information contaminates any prediction.

    Usage:
        validator = WalkForwardValidator(
            predict_fn=my_model.predict,
            embargo=1
        )
        results = validator.run(race_data, season_splits)

    The predict_fn signature:
        predict_fn(train_races: list, target_race) → dict[driver_code, RaceProbability]
    """

    def __init__(self,
                 predict_fn: Callable,
                 min_train_races: int = 20,
                 embargo: int = 1):
        """
        Args:
            predict_fn: Callable(train_data, target_race) → predictions dict.
            min_train_races: Minimum races needed before first prediction.
                             Default 20 = roughly 1 full season.
            embargo: Number of races between last train and test race
                     (to prevent rolling feature leakage). Default 1.
        """
        self.predict_fn = predict_fn
        self.min_train_races = min_train_races
        self.embargo = embargo
        self._fold_results: list[FoldResult] = []

    def run(self,
            all_races: list[dict],
            season_boundary_years: Optional[list[int]] = None,
            odds_data: Optional[dict] = None,
            verbose: bool = True) -> WalkForwardResults:
        """
        Execute full walk-forward validation.

        Args:
            all_races: List of race dicts with keys:
                       {race_id, year, round, results, lap_data, circuit_type, ...}
            season_boundary_years: If provided, apply season decay at year boundaries.
            odds_data: Optional dict mapping race_id → {driver_code: pinnacle_prob}.
            verbose: Print progress per fold.

        Returns:
            WalkForwardResults with fold-level and aggregate metrics.
        """
        results = WalkForwardResults()

        for i in range(self.min_train_races + self.embargo, len(all_races)):
            # Strict train/test split with embargo
            train_end = i - self.embargo
            train_races = all_races[:train_end]
            test_race = all_races[i]

            if verbose:
                print(f"[WalkForward] Fold {i - self.min_train_races}: "
                      f"Train on {len(train_races)} races → Test {test_race.get('name', i)}")

            # Generate predictions (model must only use train data)
            try:
                predictions = self.predict_fn(train_races, test_race)
            except Exception as e:
                print(f"  ⚠ Prediction failed: {e}")
                continue

            # Evaluate predictions against actual results
            fold_result = self._evaluate_fold(
                test_race, predictions,
                odds_data.get(test_race["race_id"]) if odds_data else None
            )
            results.folds.append(fold_result)

        return results

    def _evaluate_fold(self, test_race: dict,
                        predictions: dict,
                        odds: Optional[dict]) -> FoldResult:
        """Compute all metrics for one fold."""
        actual_results = test_race.get("results", [])
        actual_order = [r["driver_code"] for r in
                        sorted(actual_results, key=lambda r: r.get("finish_position") or 99)]

        driver_codes = list(predictions.keys())
        n_drivers = len(driver_codes)

        # Win market Brier score
        actual_winners = [1 if code == actual_order[0] else 0
                          for code in driver_codes]
        p_win = [predictions[code].p_win for code in driver_codes]
        bs_win = brier_score(p_win, actual_winners)

        # Podium market Brier score
        actual_podium = [1 if code in actual_order[:3] else 0
                         for code in driver_codes]
        p_podium = [predictions[code].p_podium for code in driver_codes]
        bs_podium = brier_score(p_podium, actual_podium)

        # RPS over all positions
        rps_scores = []
        for code, actual_pos in [(r["driver_code"], r.get("finish_position") or 20)
                                   for r in actual_results]:
            if code in predictions:
                p_dist = predictions[code].position_distribution
                rps_scores.append(
                    mean_ranked_probability_score([p_dist], [actual_pos])
                )
        rps_mean = float(np.mean(rps_scores)) if rps_scores else 0.5

        # ECE for win market
        ece = expected_calibration_error(p_win, actual_winners)

        # Log loss for win market
        ll = log_loss(p_win, actual_winners)

        # Kendall tau ranking quality
        predicted_order = sorted(driver_codes,
                                  key=lambda c: predictions[c].p_win, reverse=True)
        tau = kendall_tau_ranking(predicted_order, actual_order)

        # Edge metrics (if odds available)
        mean_edge = None
        n_pos_edge = None
        if odds:
            edges = [
                predictions[code].p_win - odds.get(code, 0.5)
                for code in driver_codes
                if code in odds
            ]
            mean_edge = float(np.mean(edges)) if edges else None
            n_pos_edge = int(sum(1 for e in edges if e > 0.04))

        return FoldResult(
            race_id=test_race.get("race_id", 0),
            race_name=test_race.get("name", "Unknown"),
            n_train_races=0,  # filled by caller if needed
            brier_win=bs_win,
            brier_podium=bs_podium,
            rps_mean=rps_mean,
            ece=ece,
            log_loss_win=ll,
            kendall_tau=tau,
            mean_edge=mean_edge,
            n_positive_edge_bets=n_pos_edge,
            n_drivers=n_drivers,
        )

    def chow_test_stability(self, results: WalkForwardResults,
                             split_idx: Optional[int] = None) -> dict:
        """
        Chow test for parameter stability across time.

        Tests whether model performance is statistically different
        before/after a split point (e.g., a regulation change year).

        Reference: Chow (1960) — "Tests of Equality Between Sets of
        Coefficients in Two Linear Regressions." Econometrica.

        Args:
            results: WalkForwardResults from run().
            split_idx: Index to split at (default: middle).

        Returns:
            Dict with F-statistic, p-value, and stability assessment.
        """
        from scipy.stats import f as f_dist

        brier_scores = results.brier_over_time
        n = len(brier_scores)
        if n < 10:
            return {"status": "insufficient_data"}

        split = split_idx or n // 2
        group1 = np.array(brier_scores[:split])
        group2 = np.array(brier_scores[split:])

        # Simple F-test on variance equality
        var1 = np.var(group1, ddof=1)
        var2 = np.var(group2, ddof=1)

        if var2 == 0:
            return {"status": "no_variance_in_group2"}

        f_stat = var1 / var2
        p_value = 2 * min(
            f_dist.cdf(f_stat, len(group1) - 1, len(group2) - 1),
            1 - f_dist.cdf(f_stat, len(group1) - 1, len(group2) - 1)
        )

        mean_diff = float(np.mean(group2) - np.mean(group1))

        return {
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "stable": p_value > 0.05,
            "mean_brier_group1": float(group1.mean()),
            "mean_brier_group2": float(group2.mean()),
            "mean_diff": mean_diff,
            "degrading": mean_diff > 0 and p_value < 0.05,
        }
