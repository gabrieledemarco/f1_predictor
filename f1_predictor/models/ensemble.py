"""
models/ensemble.py
==================
Layer 3: Ensemble Meta-Learner (Ridge Regression Stacking)

Combines the outputs of Layers 1a, 1b, and 2 into a final prediction,
using a meta-learner trained on walk-forward out-of-sample predictions.

Academic Reference:
    Wilkens, S. (2021). Sports prediction and betting models in the
    machine learning age: The case of tennis. Journal of Sports Analytics.
    "Ensemble models combining individual signals are the most
    promising contenders." — §5.3

    Walsh & Joshi (2023). Machine learning for sports betting: should
    model selection be based on accuracy or calibration?
    "Optimising for calibration yields significantly better returns." — §4

Design:
    - Ridge Regression (L2 regularisation) as meta-learner
    - Input features: outputs from L1a, L1b, L2 + contextual features
    - Target: actual finishing position (converted to probability rank)
    - Trained walk-forward: each race uses only past data

Why Ridge and not XGBoost:
    With 20-24 races/season and 20 drivers = 400-480 training samples/season,
    deep models would overfit. Ridge provides stable extrapolation
    and interpretable coefficients.
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from dataclasses import dataclass
from collections import defaultdict

from f1_predictor.domain.entities import RaceProbability


@dataclass
class EnsembleFeatures:
    """
    Feature vector for the ensemble meta-learner.
    Each represents one driver × race observation.
    """
    # Layer 1a outputs
    driver_skill_mu: float
    driver_skill_sigma: float
    driver_skill_conservative: float    # mu - 3*sigma

    # Layer 1b outputs
    machine_pace_mu: float
    machine_pace_sigma: float

    # Layer 2 outputs
    p_win_mc: float
    p_podium_mc: float
    p_dnf_mc: float
    expected_position_mc: float

    # Contextual features
    grid_position: float
    grid_vs_quali_delta: float          # qualifying vs predicted grid delta
    circuit_type_encoded: int           # 0-4 (CircuitType ordinal)
    has_grid_penalty: bool

    # TASK 5.1 — Nuove feature storiche ad alto impatto predittivo
    h2h_win_rate_3season: float = 0.5
    """Win rate H2H del pilota vs il resto del campo nelle ultime 3 stagioni.
    Cattura la performance storica relativa indipendentemente dalla macchina.
    Range [0,1]. 0.5 = neutro (default per nuovi piloti senza storico)."""

    elo_delta_vs_field: float = 0.0
    """Delta ELO rolling (finestra 20 gare) del pilota rispetto alla media del
    campo. Positivo = pilota sopra la media. Scala ~[-200, +200]."""

    dnf_rate_relative: float = 0.0
    """Tasso DNF relativo del pilota vs media campo (rolling 2 stagioni).
    Positivo = pilota piu' affidabile della media. Range [-0.1, +0.1]."""

    # Market signal (informative feature, not target)
    p_pinnacle_novig: Optional[float] = None
    log_odds_pinnacle: Optional[float] = None

    def to_array(self, include_market: bool = False) -> np.ndarray:
        """Serialise to numpy array for model input."""
        base = [
            self.driver_skill_mu,
            self.driver_skill_sigma,
            self.driver_skill_conservative,
            self.machine_pace_mu,
            self.machine_pace_sigma,
            self.p_win_mc,
            self.p_podium_mc,
            self.p_dnf_mc,
            self.expected_position_mc,
            self.grid_position,
            self.grid_vs_quali_delta,
            float(self.circuit_type_encoded),
            float(self.has_grid_penalty),
            # TASK 5.1 — nuove feature storiche
            float(self.h2h_win_rate_3season),
            float(self.elo_delta_vs_field),
            float(self.dnf_rate_relative),
        ]
        if include_market and self.p_pinnacle_novig is not None:
            base += [
                self.p_pinnacle_novig,
                self.log_odds_pinnacle or np.log(self.p_pinnacle_novig + 1e-8)
            ]
        return np.array(base, dtype=np.float64)

    @classmethod
    def feature_names(cls, include_market: bool = False) -> list[str]:
        base = [
            "skill_mu", "skill_sigma", "skill_conservative",
            "pace_mu", "pace_sigma",
            "p_win_mc", "p_podium_mc", "p_dnf_mc", "exp_pos_mc",
            "grid_pos", "grid_quali_delta", "circuit_type", "has_penalty",
            # TASK 5.1
            "h2h_win_rate_3season", "elo_delta_vs_field", "dnf_rate_relative",
        ]
        if include_market:
            base += ["p_pinnacle_novig", "log_odds_pinnacle"]
        return base


class EnsembleModel:
    """
    Layer 3: Ridge Regression meta-learner that stacks L1a + L1b + L2 outputs.

    The model is retrained walk-forward: for race k, it trains on
    observations from races 1..k-1 and predicts race k.

    Usage:
        model = EnsembleModel()
        model.fit(X_train, y_train)          # X: EnsembleFeatures, y: actual_position
        p_adj = model.predict_proba(X_test)  # adjusted win/podium probabilities

    The output is an ADJUSTMENT to the Layer 2 Monte Carlo probabilities,
    not a replacement. If the ensemble learns zero coefficients, it falls
    back to the Monte Carlo output (robust default).
    """

    def __init__(self, alpha: float = 10.0):
        """
        Args:
            alpha: Ridge regularisation strength.
                   Higher = more conservative adjustment from L2 output.
                   Default 10.0 is intentionally high to prevent L3 from
                   overriding the well-calibrated Monte Carlo probabilities.
        """
        self.alpha = alpha
        self._coef: Optional[np.ndarray] = None
        self._intercept: float = 0.0
        self._feature_importance: Optional[dict] = None
        self._is_fitted: bool = False

    def fit(self, features: list[EnsembleFeatures],
            actual_positions: list[int]) -> "EnsembleModel":
        """
        Train the meta-learner.

        Target: actual finishing position normalised to [0, 1]
        (position 1 = 0.05, position 20 = 1.0, with smoothing).

        Args:
            features: List of EnsembleFeatures for each driver × race.
            actual_positions: Corresponding finishing positions (1-20, None=DNF→20).

        Returns:
            self (fluent interface).
        """
        if len(features) < 20:
            # Not enough data — use identity (pass L2 through)
            return self

        X = np.array([f.to_array() for f in features])
        y = np.array([p / 20.0 for p in actual_positions], dtype=float)

        # Ridge regression: (X^T X + alpha * I)^{-1} X^T y
        XtX = X.T @ X + self.alpha * np.eye(X.shape[1])
        Xty = X.T @ y
        self._coef = np.linalg.solve(XtX, Xty)
        self._intercept = 0.0

        # Feature importance (absolute coefficient magnitude)
        names = EnsembleFeatures.feature_names()
        self._feature_importance = {
            name: abs(coef)
            for name, coef in zip(names, self._coef)
        }

        self._is_fitted = True
        return self

    def predict_position_score(self, features: list[EnsembleFeatures]) -> np.ndarray:
        """
        Predict normalised position score (lower = better predicted finish).

        Args:
            features: List of EnsembleFeatures (one per driver).

        Returns:
            Array of position scores in [0, 1].
        """
        if not self._is_fitted:
            # Fallback: use expected position from MC
            return np.array([f.expected_position_mc / 20.0 for f in features])

        X = np.array([f.to_array() for f in features])
        return X @ self._coef + self._intercept

    def adjust_probabilities(self,
                              mc_probs: dict[str, RaceProbability],
                              features: dict[str, EnsembleFeatures],
                              blend_factor: float = 0.3) -> dict[str, RaceProbability]:
        """
        Blend Monte Carlo probabilities with ensemble adjustments.

        The final probability is a weighted blend:
            P_final = (1 - blend) * P_mc + blend * P_ensemble_adjusted

        where P_ensemble_adjusted redistributes probability mass according
        to the ensemble's predicted position ranking.

        Args:
            mc_probs: Layer 2 Monte Carlo probabilities.
            features: EnsembleFeatures per driver.
            blend_factor: Weight given to ensemble adjustment (0 = pure MC).

        Returns:
            Adjusted RaceProbability dict.
        """
        driver_codes = list(mc_probs.keys())
        feat_list = [features[code] for code in driver_codes]
        scores = self.predict_position_score(feat_list)

        # Convert scores to probability weights via softmax (lower score = higher win prob)
        ensemble_win_probs = self._scores_to_probs(-scores)

        adjusted = {}
        for code, p_ensemble_win in zip(driver_codes, ensemble_win_probs):
            mc = mc_probs[code]

            # Blend win probability
            p_win_blend = (1 - blend_factor) * mc.p_win + blend_factor * p_ensemble_win

            # Blend other markets proportionally
            scale = p_win_blend / (mc.p_win + 1e-8)
            p_podium_blend = min(1.0, mc.p_podium * scale)
            p_top6_blend = min(1.0, mc.p_top6 * scale)
            p_points_blend = min(1.0, mc.p_points * scale)

            adjusted[code] = RaceProbability(
                race_id=mc.race_id,
                driver_code=code,
                p_win=float(p_win_blend),
                p_podium=float(p_podium_blend),
                p_top6=float(p_top6_blend),
                p_points=float(p_points_blend),
                p_dnf=mc.p_dnf,
                position_distribution=mc.position_distribution,
                model_version="ensemble_v1"
            )

        return adjusted

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance dict (sorted by magnitude)."""
        if not self._feature_importance:
            return {}
        return dict(sorted(self._feature_importance.items(),
                           key=lambda x: x[1], reverse=True))

    @staticmethod
    def _scores_to_probs(scores: np.ndarray) -> np.ndarray:
        """Convert raw scores to probabilities via softmax."""
        exp_scores = np.exp(scores - scores.max())
        return exp_scores / exp_scores.sum()
