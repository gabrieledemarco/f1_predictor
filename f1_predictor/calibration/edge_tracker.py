"""
calibration/edge_tracker.py
============================
Beta-Binomial Edge Tracker

Maintains a running Bayesian estimate of the model's true edge,
updating with each observed bet outcome.

Motivation:
    With only 24 F1 races/year, frequentist p-values require 3+ seasons
    to achieve significance. The Bayesian approach gives an actionable
    edge estimate with credible intervals from day one, narrowing
    progressively as evidence accumulates.

Academic Reference:
    Baker, R.D. & McHale, I.G. (2013). Optimal Betting Under Parameter
    Uncertainty: Improving the Kelly Criterion. Decision Analysis.
    "Betting stakes should be reduced proportionally to the uncertainty
    in the estimated win probability." — Theorem 2

    This motivates using the posterior mean edge for Kelly sizing,
    rather than the point estimate from a single season.

Model:
    Each bet is a Bernoulli trial with unknown win probability p.
    Prior: p ~ Beta(alpha_0, beta_0)  (weakly informative)
    After n_win wins and n_loss losses:
    Posterior: p ~ Beta(alpha_0 + n_win, beta_0 + n_loss)

    The posterior mean is E[p] = alpha / (alpha + beta)
    The posterior 95% credible interval is the Beta quantiles.

    For edge tracking: edge = p_model - p_implied
    We track whether p_model systematically exceeds p_implied
    by comparing posterior win rate to average implied probability.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.stats import beta as beta_dist


@dataclass
class EdgePosterior:
    """
    Current state of the edge posterior estimate.

    alpha, beta: Beta distribution parameters.
    mean_implied: Average implied probability of bets placed.
    """
    alpha: float          # alpha = prior_alpha + wins
    beta_param: float     # beta = prior_beta + losses
    n_bets: int
    mean_implied_prob: float

    @property
    def posterior_mean_win_rate(self) -> float:
        """E[p] = alpha / (alpha + beta) — Bayesian estimate of win rate."""
        return self.alpha / (self.alpha + self.beta_param)

    @property
    def posterior_edge(self) -> float:
        """Edge = posterior win rate - average implied probability."""
        return self.posterior_mean_win_rate - self.mean_implied_prob

    @property
    def credible_interval_95(self) -> tuple[float, float]:
        """95% posterior credible interval for the win rate."""
        lo = beta_dist.ppf(0.025, self.alpha, self.beta_param)
        hi = beta_dist.ppf(0.975, self.alpha, self.beta_param)
        return float(lo), float(hi)

    @property
    def edge_credible_interval_95(self) -> tuple[float, float]:
        """95% CI for the edge (win rate - implied prob)."""
        lo, hi = self.credible_interval_95
        return lo - self.mean_implied_prob, hi - self.mean_implied_prob

    @property
    def is_positive_edge(self) -> bool:
        """
        Returns True if the 95% CI lower bound for edge > 0.
        Conservative threshold: only declare positive edge when
        the lower end of uncertainty is still positive.
        """
        return self.edge_credible_interval_95[0] > 0

    @property
    def effective_kelly_fraction(self) -> float:
        """
        Kelly-adjusted stake fraction accounting for parameter uncertainty.

        From Baker & McHale (2013), Theorem 2:
            f_adjusted = f_kelly * (1 - uncertainty_penalty)
        where uncertainty_penalty decreases as evidence accumulates.

        Returns fraction of bankroll to stake on each bet.
        """
        if self.mean_implied_prob <= 0 or self.mean_implied_prob >= 1:
            return 0.0

        # Kelly fraction based on posterior mean
        b = 1.0 / self.mean_implied_prob - 1  # net odds
        p = self.posterior_mean_win_rate
        q = 1 - p
        f_kelly = (b * p - q) / b if b > 0 else 0.0

        # Uncertainty shrinkage: multiply by effective sample fraction
        # More data → less shrinkage. Shrinks to 0 with very few bets.
        shrinkage = self.n_bets / (self.n_bets + 50)  # 50 = prior strength
        f_adjusted = max(0.0, f_kelly * shrinkage)

        # Apply 1/4 Kelly cap (Baker & McHale safety recommendation)
        return min(f_adjusted, f_kelly * 0.25)


class BetaBinomialEdgeTracker:
    """
    Bayesian edge tracker using Beta-Binomial model.

    Maintains separate trackers per market type (winner, podium, H2H)
    and per circuit type, allowing granular edge detection.

    Usage:
        tracker = BetaBinomialEdgeTracker()
        tracker.update("winner", outcome=1, implied_prob=0.48)
        tracker.update("winner", outcome=0, implied_prob=0.22)
        posterior = tracker.get_posterior("winner")
        print(f"Edge: {posterior.posterior_edge:.3f}")
        print(f"95% CI: {posterior.edge_credible_interval_95}")
    """

    def __init__(self, prior_alpha: float = 2.0, prior_beta: float = 2.0):
        """
        Args:
            prior_alpha: Prior belief in wins (weak prior = 2.0).
            prior_beta: Prior belief in losses (weak prior = 2.0).
                        Beta(2, 2) is a weak symmetric prior centred at 0.5.
                        This shrinks towards 50% win rate with little data,
                        becoming more data-driven as observations accumulate.
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

        # State per market: {market_key: {alpha, beta, n, implied_sum}}
        self._state: dict[str, dict] = {}
        self._bet_log: list[dict] = []

    def update(self, market: str, outcome: int,
               implied_prob: float,
               edge_at_time_of_bet: float = 0.0):
        """
        Update edge estimate with a new bet outcome.

        Args:
            market: Market identifier (e.g. 'winner', 'podium', 'h2h').
            outcome: 1 = won, 0 = lost.
            implied_prob: Pinnacle devigged probability at time of bet.
            edge_at_time_of_bet: Edge estimate when bet was placed.
        """
        if market not in self._state:
            self._state[market] = {
                "alpha": self.prior_alpha,
                "beta": self.prior_beta,
                "n": 0,
                "implied_sum": 0.0,
            }

        s = self._state[market]
        s["alpha"] += outcome
        s["beta"] += (1 - outcome)
        s["n"] += 1
        s["implied_sum"] += implied_prob

        self._bet_log.append({
            "market": market,
            "outcome": outcome,
            "implied_prob": implied_prob,
            "edge_at_bet": edge_at_time_of_bet,
        })

    def get_posterior(self, market: str) -> Optional[EdgePosterior]:
        """
        Get current edge posterior for a market.

        Args:
            market: Market identifier.

        Returns:
            EdgePosterior or None if no bets yet.
        """
        if market not in self._state or self._state[market]["n"] == 0:
            return None

        s = self._state[market]
        n = s["n"]
        mean_implied = s["implied_sum"] / n

        return EdgePosterior(
            alpha=s["alpha"],
            beta_param=s["beta"],
            n_bets=n,
            mean_implied_prob=mean_implied
        )

    def get_all_posteriors(self) -> dict[str, EdgePosterior]:
        """Return posteriors for all tracked markets."""
        return {
            market: self.get_posterior(market)
            for market in self._state
            if self._state[market]["n"] > 0
        }

    def summary_report(self) -> list[dict]:
        """
        Generate summary report of edge estimates across all markets.

        Returns:
            List of dicts, one per market, sorted by posterior edge (desc).
        """
        report = []
        for market, posterior in self.get_all_posteriors().items():
            if posterior is None:
                continue
            lo, hi = posterior.edge_credible_interval_95
            report.append({
                "market": market,
                "n_bets": posterior.n_bets,
                "posterior_win_rate": round(posterior.posterior_mean_win_rate, 4),
                "mean_implied_prob": round(posterior.mean_implied_prob, 4),
                "posterior_edge": round(posterior.posterior_edge, 4),
                "ci_95_lo": round(lo, 4),
                "ci_95_hi": round(hi, 4),
                "is_positive_edge": posterior.is_positive_edge,
                "effective_kelly": round(posterior.effective_kelly_fraction, 4),
            })

        return sorted(report, key=lambda x: x["posterior_edge"], reverse=True)

    def market_efficiency_drift(self, market: str,
                                 window: int = 10) -> dict:
        """
        Detect if edge is eroding over time (market becoming more efficient).

        Computes rolling mean edge over last `window` bets and tests for
        negative trend using Kendall's tau.

        Args:
            market: Market to analyse.
            window: Rolling window size for trend detection.

        Returns:
            Dict with trend coefficient, p-value, and alert flag.
        """
        from scipy.stats import kendalltau

        market_bets = [b for b in self._bet_log if b["market"] == market]
        if len(market_bets) < window * 2:
            return {"status": "insufficient_data", "n_bets": len(market_bets)}

        # Compute rolling mean edge
        edges = [b["edge_at_bet"] for b in market_bets]
        rolling_means = []
        for i in range(window, len(edges) + 1):
            rolling_means.append(np.mean(edges[i-window:i]))

        tau, p_value = kendalltau(range(len(rolling_means)), rolling_means)

        return {
            "market": market,
            "trend_tau": float(tau),
            "p_value": float(p_value),
            "eroding": tau < -0.3 and p_value < 0.10,
            "alert": tau < -0.5 and p_value < 0.05,
            "n_bets": len(market_bets),
        }
