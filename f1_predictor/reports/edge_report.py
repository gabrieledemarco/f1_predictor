"""
reports/edge_report.py
======================
Pre-Race Edge Report Generator

Produces a structured report comparing model probabilities against
Pinnacle market probabilities, identifying actionable betting edges.

This is the operational output of the system — generated each
Saturday post-qualifying before every race.
"""

from __future__ import annotations
import json
from datetime import datetime
from dataclasses import asdict
from typing import Optional

from f1_predictor.domain.entities import RaceProbability, OddsRecord, CalibrationRecord
from f1_predictor.calibration.devig import devig_power, compute_overround


class EdgeReportGenerator:
    """
    Generates pre-race edge reports for betting decisions.

    Usage:
        generator = EdgeReportGenerator(min_edge=0.04)
        report = generator.generate(
            race_name="Bahrain GP 2025",
            model_probs=my_predictions,
            pinnacle_odds=odds_dict,
            calibrator=my_calibrator
        )
        print(report.to_text())
    """

    def __init__(self, min_edge: float = 0.04, min_edge_medium: float = 0.07):
        self.min_edge = min_edge
        self.min_edge_medium = min_edge_medium

    def generate(self,
                  race_name: str,
                  model_probs: dict[str, RaceProbability],
                  pinnacle_odds: dict[str, float],  # driver_code → decimal odd
                  market: str = "winner",
                  calibrator=None,
                  hours_to_race: float = 3.0) -> "EdgeReport":
        """
        Generate edge report for a specific market.

        Args:
            race_name: Human-readable race name.
            model_probs: Model output probabilities per driver.
            pinnacle_odds: Raw Pinnacle decimal odds per driver.
            market: Market name ('winner', 'podium', 'h2h').
            calibrator: Optional PinnacleCalibrationLayer.
            hours_to_race: For report metadata.

        Returns:
            EdgeReport object with all edge calculations.
        """
        # Devig the market
        driver_codes = list(pinnacle_odds.keys())
        odds_list = [pinnacle_odds[c] for c in driver_codes]
        fair_probs = devig_power(odds_list)
        overround = compute_overround(odds_list)

        pinnacle_fair = dict(zip(driver_codes, fair_probs))

        # Compute edges
        entries = []
        for code, p_fair in pinnacle_fair.items():
            if code not in model_probs:
                continue

            mp = model_probs[code]
            p_model_raw = getattr(mp, f"p_{market.split('_')[0]}", mp.p_win)

            # Apply calibration if available
            if calibrator is not None:
                p_model_cal = float(calibrator.transform([p_model_raw])[0])
            else:
                p_model_cal = p_model_raw

            edge = p_model_cal - p_fair

            signal = "✗ SKIP"
            if edge >= self.min_edge_medium:
                signal = "✓✓ MEDIUM"
            elif edge >= self.min_edge:
                signal = "✓ SMALL"

            entries.append({
                "driver": code,
                "p_model": round(p_model_cal, 4),
                "p_pinnacle": round(p_fair, 4),
                "edge": round(edge, 4),
                "odd_decimal": round(pinnacle_odds[code], 2),
                "signal": signal,
            })

        entries.sort(key=lambda x: x["edge"], reverse=True)

        return EdgeReport(
            race_name=race_name,
            market=market,
            generated_at=datetime.utcnow(),
            hours_to_race=hours_to_race,
            overround=overround,
            entries=entries,
            min_edge_used=self.min_edge,
        )


class EdgeReport:
    """Container and formatter for a pre-race edge report."""

    def __init__(self, race_name: str, market: str,
                  generated_at: datetime, hours_to_race: float,
                  overround: float, entries: list[dict],
                  min_edge_used: float):
        self.race_name = race_name
        self.market = market
        self.generated_at = generated_at
        self.hours_to_race = hours_to_race
        self.overround = overround
        self.entries = entries
        self.min_edge_used = min_edge_used

    def to_text(self) -> str:
        """Render report as formatted text."""
        bets = [e for e in self.entries if "✓" in e["signal"]]
        lines = [
            "═" * 60,
            f"EDGE REPORT — {self.race_name}",
            f"Market: {self.market.upper()} | T-{self.hours_to_race:.0f}h",
            f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            f"Market overround: {self.overround*100:.1f}%",
            "─" * 60,
            f"{'Driver':<8} {'P_model':>8} {'P_pinnacle':>10} {'Edge':>8} {'Odd':>6}  Signal",
            "─" * 60,
        ]
        for e in self.entries:
            lines.append(
                f"{e['driver']:<8} {e['p_model']:>8.4f} {e['p_pinnacle']:>10.4f} "
                f"{e['edge']:>+8.4f} {e['odd_decimal']:>6.2f}  {e['signal']}"
            )
        lines += [
            "─" * 60,
            f"ACTIONABLE BETS ({len(bets)} signals above {self.min_edge_used*100:.0f}% edge):",
        ]
        for b in bets:
            lines.append(f"  → {b['driver']}: edge {b['edge']:+.2%} @ {b['odd_decimal']:.2f}  {b['signal']}")
        lines.append("═" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "race_name": self.race_name,
            "market": self.market,
            "generated_at": self.generated_at.isoformat(),
            "hours_to_race": self.hours_to_race,
            "overround": self.overround,
            "entries": self.entries,
            "min_edge_used": self.min_edge_used,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
