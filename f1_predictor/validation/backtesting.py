"""
validation/backtesting.py
=========================
Betting Backtest Engine with Kelly Criterion

Simulates betting operations over historical races to evaluate
the edge-detection and bankroll management components.

Academic References:
    Thorp, E.O. (2006). The Kelly Criterion in Blackjack, Sports Betting,
    and the Stock Market. Handbook of Asset and Liability Management.
    (Foundational Kelly criterion derivation)

    Baker, R.D. & McHale, I.G. (2013). Optimal Betting Under Parameter
    Uncertainty: Improving the Kelly Criterion. Decision Analysis.
    "Using fractional Kelly proportional to posterior confidence
    significantly improves out-of-sample returns." — §4

    Chu, C., Wu, T.Y. & Swartz, T. (2018). Modified Kelly Criteria.
    Journal of Quantitative Analysis in Sports, SFU Technical Report.

    Walsh & Joshi (2023) §3 — backtest methodology for sports betting models.

Kelly Criterion:
    f* = (b*p - q) / b
    where:
        b = net decimal odds (odd - 1)
        p = model's estimated win probability
        q = 1 - p

    We apply Baker & McHale (2013) shrinkage:
        f_adjusted = f* * (N / (N + lambda))
    where N = number of observations, lambda = shrinkage constant (50)

    And hard cap at 1/4 Kelly for risk management.

Staking strategies implemented:
    1. flat_stake    — fixed unit per bet (baseline)
    2. kelly_quarter — 1/4 Kelly (recommended)
    3. kelly_full    — Full Kelly (aggressive, for comparison only)
    4. kelly_shrunk  — Baker & McHale (2013) shrinkage Kelly

Performance metrics:
    ROI, Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from f1_predictor.domain.entities import BetRecord, CalibrationRecord


@dataclass
class BacktestConfig:
    """
    Backtest configuration.

    Args:
        initial_bankroll: Starting bankroll in units.
        min_edge: Minimum edge threshold to place a bet.
                  Bets below this are skipped.
        max_stake_fraction: Maximum fraction of bankroll per bet.
        staking_strategy: 'flat', 'kelly_quarter', 'kelly_full', 'kelly_shrunk'.
        flat_stake_units: Fixed stake for flat staking.
        kelly_shrinkage_lambda: Lambda for Baker & McHale shrinkage (default 50).
        stop_loss_fraction: Stop betting if bankroll drops below this fraction.
    """
    initial_bankroll: float = 1000.0
    min_edge: float = 0.04
    max_stake_fraction: float = 0.10
    staking_strategy: str = "kelly_quarter"
    flat_stake_units: float = 10.0
    kelly_shrinkage_lambda: float = 50.0
    stop_loss_fraction: float = 0.50


@dataclass
class BacktestResults:
    """Full backtest output."""
    bet_records: list[BetRecord] = field(default_factory=list)
    bankroll_history: list[float] = field(default_factory=list)

    @property
    def total_bets(self) -> int:
        return len(self.bet_records)

    @property
    def wins(self) -> int:
        return sum(1 for b in self.bet_records if b.outcome == 1)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_bets if self.total_bets > 0 else 0.0

    @property
    def total_pnl(self) -> float:
        return sum(b.pnl for b in self.bet_records)

    @property
    def roi(self) -> float:
        total_staked = sum(b.actual_stake for b in self.bet_records)
        return self.total_pnl / total_staked if total_staked > 0 else 0.0

    @property
    def final_bankroll(self) -> float:
        return self.bankroll_history[-1] if self.bankroll_history else 0.0

    @property
    def bankroll_growth(self) -> float:
        return (self.final_bankroll / self.bankroll_history[0]) - 1 if self.bankroll_history else 0.0

    @property
    def sharpe_ratio(self) -> float:
        """
        Annualised Sharpe Ratio of bet returns.

        Reference: Thorp (2006) — Sharpe Ratio as risk-adjusted
        performance measure for betting strategies.

        SR = (mean_return - rf) / std_return * sqrt(N_bets_per_year)
        Using rf = 0 (cash return as benchmark).
        """
        if self.total_bets < 2:
            return 0.0
        returns = [b.roi for b in self.bet_records]
        mean_r = np.mean(returns)
        std_r = np.std(returns, ddof=1)
        if std_r == 0:
            return 0.0
        # Annualise: approximately 100 bets/year across all markets
        n_per_year = 100
        return float(mean_r / std_r * np.sqrt(n_per_year))

    @property
    def max_drawdown(self) -> float:
        """
        Maximum peak-to-trough drawdown as fraction of peak bankroll.

        Range: [0, 1]. Lower is better.
        Alarm threshold: > 0.20 (20% drawdown).
        """
        if not self.bankroll_history:
            return 0.0
        bh = np.array(self.bankroll_history)
        running_max = np.maximum.accumulate(bh)
        drawdowns = (running_max - bh) / running_max
        return float(drawdowns.max())

    @property
    def profit_factor(self) -> float:
        """
        Profit Factor = Total Winnings / Total Losses.
        > 1.0 = profitable. Target: > 1.20.
        """
        winnings = sum(b.pnl for b in self.bet_records if b.pnl > 0)
        losses = abs(sum(b.pnl for b in self.bet_records if b.pnl < 0))
        return float(winnings / losses) if losses > 0 else float("inf")

    def summary(self) -> dict:
        return {
            "total_bets": self.total_bets,
            "wins": self.wins,
            "win_rate": round(self.win_rate, 4),
            "total_pnl": round(self.total_pnl, 2),
            "roi": round(self.roi, 4),
            "bankroll_growth": round(self.bankroll_growth, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "max_drawdown": round(self.max_drawdown, 4),
            "profit_factor": round(self.profit_factor, 3),
            "final_bankroll": round(self.final_bankroll, 2),
        }


class BettingBacktester:
    """
    Full betting simulation engine.

    Simulates placing bets based on model edge, tracking bankroll
    evolution and performance metrics over historical races.

    Usage:
        backtester = BettingBacktester(config=BacktestConfig())
        results = backtester.run(calibration_records)
        print(results.summary())
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(self, records: list[CalibrationRecord],
            n_observations_for_kelly: Optional[dict] = None) -> BacktestResults:
        """
        Run full backtest simulation.

        Args:
            records: CalibrationRecord list sorted by race date.
            n_observations_for_kelly: Optional dict mapping market → n_obs
                                       (for Kelly shrinkage computation).

        Returns:
            BacktestResults with full bet log and performance metrics.
        """
        results = BacktestResults()
        bankroll = self.config.initial_bankroll
        results.bankroll_history.append(bankroll)

        # Stop-loss threshold
        stop_loss_bankroll = bankroll * self.config.stop_loss_fraction
        stopped = False

        for record in sorted(records, key=lambda r: r.timestamp):
            if stopped:
                break

            if bankroll <= stop_loss_bankroll:
                stopped = True
                print(f"⛔ Stop-loss triggered at bankroll {bankroll:.2f}")
                break

            # Only bet when edge exceeds threshold
            if record.edge < self.config.min_edge:
                continue

            # Compute stake
            n_obs = 0
            if n_observations_for_kelly:
                n_obs = n_observations_for_kelly.get(record.market, 0)

            stake = self._compute_stake(
                edge=record.edge,
                implied_prob=record.p_pinnacle_novig,
                bankroll=bankroll,
                n_observations=n_obs
            )

            if stake <= 0:
                continue

            # Simulate bet outcome
            pnl = self._compute_pnl(stake, record)
            bankroll_after = bankroll + pnl

            kelly_f = self._kelly_fraction(record.edge, record.p_pinnacle_novig)

            bet = BetRecord(
                race_id=record.race_id,
                driver_code=record.driver_code,
                market=record.market,
                edge=record.edge,
                kelly_fraction=kelly_f,
                actual_stake=stake,
                odd_decimal=1.0 / record.p_pinnacle_novig if record.p_pinnacle_novig > 0 else 0,
                outcome=record.outcome,
                pnl=pnl,
                bankroll_before=bankroll,
                bankroll_after=bankroll_after,
                timestamp=record.timestamp,
            )

            results.bet_records.append(bet)
            bankroll = bankroll_after
            results.bankroll_history.append(bankroll)

        return results

    def _compute_stake(self, edge: float, implied_prob: float,
                        bankroll: float, n_observations: int) -> float:
        """
        Compute stake based on configured staking strategy.

        Reference: Baker & McHale (2013), Eq. 15 — shrinkage formula.
        """
        strategy = self.config.staking_strategy

        if strategy == "flat":
            return min(self.config.flat_stake_units, bankroll * 0.05)

        kelly_f = self._kelly_fraction(edge, implied_prob)

        if strategy == "kelly_full":
            fraction = kelly_f

        elif strategy == "kelly_quarter":
            fraction = kelly_f * 0.25

        elif strategy == "kelly_shrunk":
            # Baker & McHale (2013) shrinkage
            lambda_ = self.config.kelly_shrinkage_lambda
            shrinkage = n_observations / (n_observations + lambda_)
            fraction = kelly_f * shrinkage * 0.25  # also apply quarter cap

        else:
            fraction = kelly_f * 0.25  # default to quarter Kelly

        fraction = min(fraction, self.config.max_stake_fraction)
        return max(0.0, fraction * bankroll)

    def _kelly_fraction(self, edge: float, implied_prob: float) -> float:
        """
        Raw Kelly fraction.
        f* = (b*p - q) / b
        """
        if implied_prob <= 0 or implied_prob >= 1:
            return 0.0
        b = 1.0 / implied_prob - 1  # net odds
        p_model = implied_prob + edge  # our estimated win probability
        q = 1 - p_model
        f = (b * p_model - q) / b if b > 0 else 0.0
        return max(0.0, f)

    def _compute_pnl(self, stake: float, record: CalibrationRecord) -> float:
        """
        Compute P&L for a bet.

        If won: return stake * (odd - 1) = stake * (1/p_pinnacle - 1)
        If lost: return -stake
        """
        if record.p_pinnacle_novig <= 0:
            return -stake
        odd = 1.0 / record.p_pinnacle_novig
        if record.outcome == 1:
            return stake * (odd - 1.0)
        else:
            return -stake

    def bootstrap_confidence_intervals(self,
                                        results: BacktestResults,
                                        n_bootstrap: int = 10_000) -> dict:
        """
        Bootstrap confidence intervals for ROI and Sharpe.

        Re-samples bets with replacement to estimate the distribution
        of performance metrics, accounting for small sample size.

        Reference: Efron & Tibshirani (1993) — bootstrap CI methodology.

        Args:
            results: BacktestResults from run().
            n_bootstrap: Number of bootstrap iterations.

        Returns:
            Dict with 95% CIs for ROI, Sharpe, and Win Rate.
        """
        bets = results.bet_records
        if len(bets) < 10:
            return {"status": "insufficient_data"}

        roi_samples = []
        sharpe_samples = []
        wr_samples = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(len(bets), size=len(bets), replace=True)
            sampled = [bets[i] for i in sample]
            total_staked = sum(b.actual_stake for b in sampled)
            total_pnl = sum(b.pnl for b in sampled)
            roi_s = total_pnl / total_staked if total_staked > 0 else 0.0
            returns = [b.roi for b in sampled]
            std_r = np.std(returns, ddof=1)
            sharpe_s = np.mean(returns) / std_r * np.sqrt(100) if std_r > 0 else 0.0
            wr_s = sum(1 for b in sampled if b.outcome == 1) / len(sampled)

            roi_samples.append(roi_s)
            sharpe_samples.append(sharpe_s)
            wr_samples.append(wr_s)

        def ci(arr):
            return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

        return {
            "roi_ci_95": ci(roi_samples),
            "roi_mean": float(np.mean(roi_samples)),
            "sharpe_ci_95": ci(sharpe_samples),
            "sharpe_mean": float(np.mean(sharpe_samples)),
            "win_rate_ci_95": ci(wr_samples),
            "n_bets": len(bets),
            "n_bootstrap": n_bootstrap,
        }
