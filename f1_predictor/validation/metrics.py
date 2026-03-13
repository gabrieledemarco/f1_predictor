"""
validation/metrics.py
=====================
Probabilistic Scoring Rules and Calibration Metrics

All metrics are chosen for evaluating PROBABILISTIC predictions,
not just binary accuracy — critical for sports betting applications.

Academic References:
    Gneiting, T. & Raftery, A.E. (2007). Strictly Proper Scoring Rules,
    Prediction, and Estimation. JASA, 102(477), 359-378.
    (Defines proper scoring rules for probabilistic forecasts)

    Brier, G.W. (1950). Verification of forecasts expressed in terms
    of probability. Monthly Weather Review, 78(1), 1-3.
    (Original Brier Score paper)

    Ranked Probability Score (RPS): Extended Brier Score for ordinal
    outcomes, appropriate for position prediction in F1 (1st, 2nd, ..., 20th).

    CRPS: Generalisation of MAE to probability distributions.

    Walsh & Joshi (2023) — use CRPS and calibration error as primary
    metrics for sports betting model evaluation.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Brier Score
# ---------------------------------------------------------------------------

def brier_score(p_pred: list[float], outcomes: list[int]) -> float:
    """
    Brier Score for binary events.

    BS = (1/N) * sum((p_i - o_i)^2)

    Range: [0, 1]. Lower is better. Perfect = 0.
    A model predicting all 0.5 scores 0.25 (reference).

    Args:
        p_pred: List of predicted probabilities.
        outcomes: List of binary outcomes (0 or 1).

    Returns:
        Brier Score (scalar).
    """
    p = np.array(p_pred, dtype=float)
    o = np.array(outcomes, dtype=float)
    return float(np.mean((p - o) ** 2))


def brier_score_decomposition(p_pred: list[float],
                               outcomes: list[int],
                               n_bins: int = 10) -> dict:
    """
    Decompose Brier Score into Reliability + Resolution - Uncertainty.

    Reference: Murphy (1973) decomposition:
        BS = REL - RES + UNC

    REL (Reliability): How well calibrated are the probabilities?
        Lower = better calibrated.
    RES (Resolution): How much do predictions deviate from climatology?
        Higher = more informative.
    UNC (Uncertainty): Inherent unpredictability of outcomes.
        Constant for a given dataset — not under model control.

    Args:
        p_pred: Predicted probabilities.
        outcomes: Binary outcomes.
        n_bins: Number of probability bins for calibration curve.

    Returns:
        Dict with BS, REL, RES, UNC, and bin-level reliability data.
    """
    p = np.array(p_pred, dtype=float)
    o = np.array(outcomes, dtype=float)

    climatology = o.mean()  # base rate
    bs = float(np.mean((p - o) ** 2))
    unc = climatology * (1 - climatology)

    # Bin probabilities
    bins = np.linspace(0, 1 + 1e-8, n_bins + 1)
    bin_indices = np.digitize(p, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    rel = 0.0
    res = 0.0
    bin_data = []

    for k in range(n_bins):
        mask = bin_indices == k
        n_k = mask.sum()
        if n_k == 0:
            continue
        p_k = p[mask].mean()      # mean forecast probability in bin
        o_k = o[mask].mean()      # observed frequency in bin
        rel += (n_k / len(p)) * (p_k - o_k) ** 2
        res += (n_k / len(p)) * (o_k - climatology) ** 2
        bin_data.append({"bin_mean_prob": float(p_k), "obs_freq": float(o_k), "n": int(n_k)})

    return {
        "brier_score": bs,
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": float(unc),
        "check": abs(bs - (rel - res + unc)) < 1e-6,
        "bins": bin_data,
    }


# ---------------------------------------------------------------------------
# Ranked Probability Score (RPS)
# ---------------------------------------------------------------------------

def ranked_probability_score(p_dist: list[float],
                               actual_rank: int,
                               n_categories: int = 20) -> float:
    """
    Ranked Probability Score for ordinal outcomes.

    Appropriate for F1 finishing positions (ordinal, not nominal).
    The RPS is a proper scoring rule that penalises predictions
    proportionally to how far off the predicted distribution is
    from the actual outcome.

    RPS = (1/(K-1)) * sum_{k=1}^{K-1} (CDF_pred(k) - CDF_actual(k))^2

    Range: [0, 1]. Lower is better. Perfect = 0.

    Reference: Epstein (1969) — "A scoring system for probability
    forecasts of ranked categories". J. Applied Meteorology.

    Args:
        p_dist: Predicted probability distribution over K categories.
                p_dist[i] = P(finishing position = i+1).
        actual_rank: Actual finishing position (1-indexed).
        n_categories: Total number of categories (20 for F1).

    Returns:
        RPS value for this single prediction.
    """
    if len(p_dist) != n_categories:
        p_dist = (list(p_dist) + [0.0] * n_categories)[:n_categories]

    # Cumulative distributions
    cdf_pred = np.cumsum(p_dist)
    cdf_actual = np.zeros(n_categories)
    cdf_actual[actual_rank - 1:] = 1.0  # step function at actual rank

    rps = np.mean((cdf_pred[:-1] - cdf_actual[:-1]) ** 2)
    return float(rps)


def mean_ranked_probability_score(predictions: list[list[float]],
                                   actuals: list[int],
                                   n_categories: int = 20) -> float:
    """
    Mean RPS over multiple predictions.

    Args:
        predictions: List of position distributions.
        actuals: List of actual finishing positions (1-indexed).
        n_categories: Number of positions (default 20).

    Returns:
        Mean RPS.
    """
    scores = [
        ranked_probability_score(p, a, n_categories)
        for p, a in zip(predictions, actuals)
    ]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Continuous Ranked Probability Score (CRPS)
# ---------------------------------------------------------------------------

def continuous_ranked_probability_score(p_mean: float, p_std: float,
                                         actual: float) -> float:
    """
    CRPS for a Gaussian predictive distribution.

    CRPS = E|X - y| - 0.5 * E|X - X'|
    For Gaussian N(mu, sigma^2):
    CRPS = sigma * (z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi))
    where z = (y - mu) / sigma

    Reference: Gneiting & Raftery (2007), Eq. 21.

    Used for evaluating continuous predictions (lap times, gaps).

    Args:
        p_mean: Predicted mean.
        p_std: Predicted standard deviation.
        actual: Observed value.

    Returns:
        CRPS value (lower = better).
    """
    from scipy.stats import norm

    if p_std <= 0:
        return abs(actual - p_mean)

    z = (actual - p_mean) / p_std
    crps = p_std * (z * (2 * norm.cdf(z) - 1) +
                    2 * norm.pdf(z) -
                    1.0 / np.sqrt(np.pi))
    return float(crps)


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------

def expected_calibration_error(p_pred: list[float],
                                 outcomes: list[int],
                                 n_bins: int = 10) -> float:
    """
    Expected Calibration Error.

    ECE = sum_b (n_b / N) * |mean_predicted_b - observed_freq_b|

    Measures the average gap between predicted probability and
    observed frequency within bins of similar predictions.

    Reference: Naeini et al. (2015) — "Obtaining Well Calibrated
    Probabilities Using Bayesian Binning into Quantiles".

    Target: ECE < 0.05 for a well-calibrated model.

    Args:
        p_pred: Predicted probabilities.
        outcomes: Binary outcomes.
        n_bins: Number of bins (default 10).

    Returns:
        ECE value in [0, 1].
    """
    p = np.array(p_pred, dtype=float)
    o = np.array(outcomes, dtype=float)
    n = len(p)

    bins = np.linspace(0, 1 + 1e-8, n_bins + 1)
    bin_indices = np.digitize(p, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for k in range(n_bins):
        mask = bin_indices == k
        n_k = mask.sum()
        if n_k == 0:
            continue
        p_k = p[mask].mean()
        o_k = o[mask].mean()
        ece += (n_k / n) * abs(p_k - o_k)

    return float(ece)


def calibration_curve(p_pred: list[float],
                       outcomes: list[int],
                       n_bins: int = 10) -> dict:
    """
    Full calibration curve data for reliability diagrams.

    Returns:
        Dict with mean_predicted and observed_freq arrays per bin.
    """
    p = np.array(p_pred, dtype=float)
    o = np.array(outcomes, dtype=float)

    bins = np.linspace(0, 1 + 1e-8, n_bins + 1)
    bin_indices = np.digitize(p, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    mean_predicted, observed_freq, counts = [], [], []

    for k in range(n_bins):
        mask = bin_indices == k
        if mask.sum() == 0:
            continue
        mean_predicted.append(float(p[mask].mean()))
        observed_freq.append(float(o[mask].mean()))
        counts.append(int(mask.sum()))

    ece = expected_calibration_error(p_pred, outcomes, n_bins)

    return {
        "mean_predicted": mean_predicted,
        "observed_freq": observed_freq,
        "counts": counts,
        "ece": ece,
        "n_total": len(p),
    }


# ---------------------------------------------------------------------------
# Log Loss
# ---------------------------------------------------------------------------

def log_loss(p_pred: list[float], outcomes: list[int],
             eps: float = 1e-8) -> float:
    """
    Binary cross-entropy (Log Loss).

    LL = -(1/N) * sum(o_i * log(p_i) + (1-o_i) * log(1-p_i))

    Args:
        p_pred: Predicted probabilities.
        outcomes: Binary outcomes.
        eps: Clipping value to avoid log(0).

    Returns:
        Log loss (lower = better).
    """
    p = np.clip(np.array(p_pred, dtype=float), eps, 1 - eps)
    o = np.array(outcomes, dtype=float)
    return float(-np.mean(o * np.log(p) + (1 - o) * np.log(1 - p)))


def kendall_tau_ranking(predicted_order: list[str],
                         actual_order: list[str]) -> float:
    """
    Kendall's tau for ranking correlation.

    Used for validating the Driver Skill Rating (Layer 1a).
    A perfect ranking = tau of +1.0.
    Random ranking = tau near 0.

    Reference: van Kesteren & Bergkamp (2023) use Kendall's tau to
    validate the model's ability to reproduce historical driver rankings.

    Target: tau > 0.45 for the skill model to add value.

    Args:
        predicted_order: Driver codes in predicted order (best first).
        actual_order: Driver codes in actual finishing order (best first).

    Returns:
        Kendall's tau in [-1, 1].
    """
    from scipy.stats import kendalltau

    n = min(len(predicted_order), len(actual_order))
    pred_ranks = {code: i for i, code in enumerate(predicted_order[:n])}
    actual_ranks = {code: i for i, code in enumerate(actual_order[:n])}

    common = set(pred_ranks) & set(actual_ranks)
    if len(common) < 2:
        return 0.0

    pred_r = [pred_ranks[c] for c in common]
    actual_r = [actual_ranks[c] for c in common]

    tau, _ = kendalltau(pred_r, actual_r)
    return float(tau)


def compute_metrics(p_pred: list[float], outcomes: list[int], predicted_order: list[str] = None, actual_order: list[str] = None) -> dict:
    """
    Compute multiple performance metrics: Brier, AUC, ROI, ECE.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score
    
    brier = brier_score(p_pred, outcomes)
    try:
        auc = roc_auc_score(outcomes, p_pred)
    except:
        auc = 0.5
    ece = expected_calibration_error(p_pred, outcomes)
    roi = 0.0  # TODO: implement ROI calculation
    
    return {
        "brier": brier,
        "auc": auc,
        "roi": roi,
        "ece": ece,
    }
