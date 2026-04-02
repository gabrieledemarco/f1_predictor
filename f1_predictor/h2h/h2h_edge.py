"""
f1_predictor/h2h/h2h_edge.py
==============================
Calcola edge, EV e stake Kelly per mercati H2H driver e constructor
confrontando le probabilità del modello con le quote bookmaker.

Workflow:
  1. Devi le quote raw del bookmaker (rimuovi il vig con power method)
  2. Calcola P_model dalla H2HCalculator
  3. Edge = P_model - P_fair
  4. EV = Edge × odd_decimal
  5. Kelly stake = (P_model × odd - 1) / (odd - 1) × kelly_fraction

Il report finale include solo le scommesse con edge > min_edge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .h2h_calculator import DriverH2H, ConstructorH2H


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class H2HEdgeBet:
    """Una singola scommessa H2H con edge positivo."""

    # Identificatori
    market_type: str        # "driver_h2h" | "constructor_h2h"
    selection: str          # es. "VER vs NOR → VER"
    driver_a: str
    driver_b: str
    bet_on: str             # il pilota/team su cui scommettere

    # Probabilità
    p_model: float          # probabilità del modello (calibrata)
    p_implied: float        # probabilità implicita dopo devig
    p_raw_bookie: float     # probabilità implicita grezza (con vig)

    # Quote e mercato
    odd_decimal: float      # quota decimale offerta
    odd_fair: float         # quota fair (1/p_implied)
    bookmaker: str          # nome bookmaker

    # Edge e sizing
    edge: float             # p_model - p_implied
    ev: float               # Expected Value = edge × odd_decimal
    kelly_full: float       # Kelly criterio (% bankroll)
    kelly_quarter: float    # Kelly frazionario 0.25×
    kelly_fifth: float      # Kelly frazionario 0.20× (più conservativo)

    # Contesto
    race_name: str = ""
    notes: str = ""

    @property
    def is_value(self) -> bool:
        return self.edge > 0

    @property
    def implied_odd(self) -> float:
        return round(1.0 / self.p_model, 3) if self.p_model > 0 else 999.0

    def to_dict(self) -> dict:
        return {
            "market": self.market_type,
            "selection": self.selection,
            "bet_on": self.bet_on,
            "p_model": round(self.p_model, 4),
            "p_implied": round(self.p_implied, 4),
            "odd_offered": round(self.odd_decimal, 3),
            "odd_fair": round(self.odd_fair, 3),
            "edge": round(self.edge, 4),
            "ev_pct": round(self.ev * 100, 2),
            "kelly_full_pct": round(self.kelly_full * 100, 2),
            "kelly_quarter_pct": round(self.kelly_quarter * 100, 2),
            "kelly_fifth_pct": round(self.kelly_fifth * 100, 2),
            "bookmaker": self.bookmaker,
            "race": self.race_name,
            "notes": self.notes,
        }

    def to_text(self) -> str:
        arrow = "✅" if self.edge >= 0.04 else "⚠️"
        return (
            f"{arrow} {self.selection:<30} "
            f"P_mod={self.p_model:.3f}  P_fair={self.p_implied:.3f}  "
            f"Odd={self.odd_decimal:.2f}  Edge={self.edge:+.3f}  "
            f"EV={self.ev*100:+.1f}%  Kelly¼={self.kelly_quarter*100:.1f}%"
        )


@dataclass
class H2HEdgeReport:
    """Report completo di edge su tutti i mercati H2H."""

    race_name: str
    bets: list[H2HEdgeBet] = field(default_factory=list)
    min_edge: float = 0.04
    n_driver_pairs_evaluated: int = 0
    n_constructor_pairs_evaluated: int = 0

    @property
    def value_bets(self) -> list[H2HEdgeBet]:
        return [b for b in self.bets if b.edge >= self.min_edge]

    @property
    def best_bet(self) -> Optional[H2HEdgeBet]:
        v = self.value_bets
        return max(v, key=lambda b: b.ev) if v else None

    def to_text(self) -> str:
        lines = [
            f"\n{'='*70}",
            f"  H2H EDGE REPORT — {self.race_name}",
            f"{'='*70}",
            f"  Coppie driver valutate:      {self.n_driver_pairs_evaluated}",
            f"  Coppie constructor valutate: {self.n_constructor_pairs_evaluated}",
            f"  Value bets (edge ≥ {self.min_edge:.0%}):    {len(self.value_bets)}",
            "",
        ]

        driver_bets = [b for b in self.value_bets if b.market_type == "driver_h2h"]
        constructor_bets = [b for b in self.value_bets if b.market_type == "constructor_h2h"]

        if driver_bets:
            lines.append("  ── DRIVER H2H ──────────────────────────────────────────────")
            for b in sorted(driver_bets, key=lambda x: -x.edge):
                lines.append("  " + b.to_text())

        if constructor_bets:
            lines.append("")
            lines.append("  ── CONSTRUCTOR H2H ─────────────────────────────────────────")
            for b in sorted(constructor_bets, key=lambda x: -x.edge):
                lines.append("  " + b.to_text())

        if not driver_bets and not constructor_bets:
            lines.append("  Nessun value bet trovato con edge ≥ {:.0%}".format(self.min_edge))

        lines.append(f"{'='*70}\n")
        return "\n".join(lines)

    def to_list(self) -> list[dict]:
        return [b.to_dict() for b in self.value_bets]


# ---------------------------------------------------------------------------
# Devig utilities
# ---------------------------------------------------------------------------

def devig_power(odds: list[float], iterations: int = 100) -> list[float]:
    """
    Rimuove il vig usando il Power Method (Shin/power devig).
    Converge alla distribuzione fair in cui le probabilità sommano a 1.

    Args:
        odds: lista di quote decimali [odd_A, odd_B]
    Returns:
        Lista di probabilità fair (somma = 1.0)
    """
    if len(odds) != 2:
        # Per H2H binario è sempre 2 outcome
        raise ValueError("devig_power richiede esattamente 2 quote per H2H")

    implied = np.array([1.0 / o for o in odds])
    overround = implied.sum()

    if overround <= 1.0:
        # No vig (o odds errate) — restituisce implied normalizate
        return list(implied / implied.sum())

    # Power method: trova k tale che sum(p_i^k) = 1, p_i = implied_i^k / sum(implied_j^k)
    # Approssimazione iterativa
    k = 1.0
    for _ in range(iterations):
        powered = implied ** k
        total = powered.sum()
        # Gradient step per avvicinare total → 1.0
        if abs(total - 1.0) < 1e-9:
            break
        k *= (np.log(1.0) / np.log(total))  # ajustment logaritmico
        k = max(0.5, min(k, 3.0))           # clamp per stabilità

    powered = implied ** k
    fair_probs = powered / powered.sum()
    return list(fair_probs)


def devig_additive(odds: list[float]) -> list[float]:
    """
    Devig additivo semplice: sottrae il vig proporzionalmente.
    Meno accurato del power method ma deterministico.
    """
    implied = np.array([1.0 / o for o in odds])
    overround = implied.sum()
    fair = implied / overround
    return list(fair)


# ---------------------------------------------------------------------------
# H2H Edge Calculator
# ---------------------------------------------------------------------------

class H2HEdgeCalculator:
    """
    Calcola edge e Kelly stake per mercati H2H confrontando
    le probabilità del modello con le quote bookmaker.

    Usage:
        calc = H2HEdgeCalculator(min_edge=0.04, kelly_fraction=0.25)

        # Da una lista di H2H già calcolati + odds
        driver_odds = {
            ("VER", "NOR"): {"VER": 1.65, "NOR": 2.20},
            ("LEC", "HAM"): {"LEC": 1.90, "HAM": 1.90},
        }
        bets = calc.evaluate_driver_bets(driver_h2h_list, driver_odds, "GP Miami")
    """

    def __init__(
        self,
        min_edge: float = 0.04,
        kelly_fraction: float = 0.25,
        devig_method: str = "power",   # "power" | "additive"
        bookmaker: str = "bookmaker",
    ):
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
        self.devig_method = devig_method
        self.bookmaker = bookmaker

    def _devig(self, odds: list[float]) -> list[float]:
        if self.devig_method == "power":
            try:
                return devig_power(odds)
            except Exception:
                return devig_additive(odds)
        return devig_additive(odds)

    def _kelly(self, p_model: float, odd_decimal: float) -> float:
        """Kelly criterio standard. Restituisce frazione di bankroll."""
        if odd_decimal <= 1.0 or p_model <= 0:
            return 0.0
        b = odd_decimal - 1.0           # profitto netto per unità
        kelly = (b * p_model - (1 - p_model)) / b
        return float(max(0.0, kelly))

    def _build_bet(
        self,
        market_type: str,
        driver_a: str,
        driver_b: str,
        bet_on: str,
        p_model: float,
        odd_decimal: float,
        p_implied: float,
        p_raw_bookie: float,
        race_name: str,
    ) -> H2HEdgeBet:
        """Costruisce un H2HEdgeBet con tutti i campi calcolati."""
        edge = p_model - p_implied
        ev = edge * odd_decimal
        kf = self._kelly(p_model, odd_decimal)
        odd_fair = 1.0 / p_implied if p_implied > 0 else 999.0

        opponent = driver_b if bet_on == driver_a else driver_a
        selection = f"{driver_a} vs {driver_b} → {bet_on}"

        return H2HEdgeBet(
            market_type=market_type,
            selection=selection,
            driver_a=driver_a,
            driver_b=driver_b,
            bet_on=bet_on,
            p_model=float(np.clip(p_model, 0.0, 1.0)),
            p_implied=float(np.clip(p_implied, 0.0, 1.0)),
            p_raw_bookie=float(p_raw_bookie),
            odd_decimal=float(odd_decimal),
            odd_fair=float(odd_fair),
            bookmaker=self.bookmaker,
            edge=float(edge),
            ev=float(ev),
            kelly_full=float(np.clip(kf, 0.0, 1.0)),
            kelly_quarter=float(np.clip(kf * 0.25, 0.0, 0.20)),
            kelly_fifth=float(np.clip(kf * 0.20, 0.0, 0.20)),
            race_name=race_name,
        )

    # ---------------------------------------------------------------------------
    # Driver H2H evaluation
    # ---------------------------------------------------------------------------

    def evaluate_driver_bets(
        self,
        h2h_list: list[DriverH2H],
        odds_map: dict,
        race_name: str = "",
    ) -> list[H2HEdgeBet]:
        """
        Valuta una lista di DriverH2H contro le quote bookmaker.

        Args:
            h2h_list: output di H2HCalculator.all_driver_h2h()
            odds_map: {(driver_a, driver_b): {driver_a: odd_A, driver_b: odd_B}}
                      Accetta anche chiave inversa (driver_b, driver_a).
            race_name: nome del GP

        Returns:
            Lista di H2HEdgeBet con edge > 0 (tutti, non solo quelli sopra min_edge).
            Usa .value_bets sulla H2HEdgeReport per filtrare.
        """
        bets = []

        for h2h in h2h_list:
            a, b = h2h.driver_a, h2h.driver_b

            # Cerca la chiave nell'odds_map (accetta entrambe le direzioni)
            odds_entry = odds_map.get((a, b)) or odds_map.get((b, a))
            if odds_entry is None:
                continue

            # Estrai quote (supporta sia stringhe che indici)
            odd_a = odds_entry.get(a) or odds_entry.get(0)
            odd_b = odds_entry.get(b) or odds_entry.get(1)

            if odd_a is None or odd_b is None:
                continue

            # Devig
            p_fair_a, p_fair_b = self._devig([float(odd_a), float(odd_b)])
            p_raw_a = 1.0 / float(odd_a)
            p_raw_b = 1.0 / float(odd_b)

            # Bet su A
            bet_a = self._build_bet(
                "driver_h2h", a, b, bet_on=a,
                p_model=h2h.p_a_beats_b,
                odd_decimal=float(odd_a),
                p_implied=p_fair_a,
                p_raw_bookie=p_raw_a,
                race_name=race_name,
            )
            bets.append(bet_a)

            # Bet su B
            bet_b = self._build_bet(
                "driver_h2h", a, b, bet_on=b,
                p_model=h2h.p_b_beats_a,
                odd_decimal=float(odd_b),
                p_implied=p_fair_b,
                p_raw_bookie=p_raw_b,
                race_name=race_name,
            )
            bets.append(bet_b)

        return bets

    # ---------------------------------------------------------------------------
    # Constructor H2H evaluation
    # ---------------------------------------------------------------------------

    def evaluate_constructor_bets(
        self,
        h2h_list: list[ConstructorH2H],
        odds_map: dict,
        race_name: str = "",
    ) -> list[H2HEdgeBet]:
        """
        Valuta una lista di ConstructorH2H contro le quote bookmaker.

        Args:
            odds_map: {(team_a, team_b): {team_a: odd_A, team_b: odd_B}}
        """
        bets = []

        for h2h in h2h_list:
            a, b = h2h.team_a, h2h.team_b

            odds_entry = odds_map.get((a, b)) or odds_map.get((b, a))
            if odds_entry is None:
                continue

            odd_a = odds_entry.get(a) or odds_entry.get(0)
            odd_b = odds_entry.get(b) or odds_entry.get(1)

            if odd_a is None or odd_b is None:
                continue

            p_fair_a, p_fair_b = self._devig([float(odd_a), float(odd_b)])

            bet_a = self._build_bet(
                "constructor_h2h", a, b, bet_on=a,
                p_model=h2h.p_a_beats_b,
                odd_decimal=float(odd_a),
                p_implied=p_fair_a,
                p_raw_bookie=1.0 / float(odd_a),
                race_name=race_name,
            )
            bet_b = self._build_bet(
                "constructor_h2h", a, b, bet_on=b,
                p_model=h2h.p_b_beats_a,
                odd_decimal=float(odd_b),
                p_implied=p_fair_b,
                p_raw_bookie=1.0 / float(odd_b),
                race_name=race_name,
            )
            bets.append(bet_a)
            bets.append(bet_b)

        return bets

    # ---------------------------------------------------------------------------
    # Full report
    # ---------------------------------------------------------------------------

    def build_report(
        self,
        race_name: str,
        driver_h2h_list: list[DriverH2H],
        constructor_h2h_list: list[ConstructorH2H],
        driver_odds: dict,
        constructor_odds: dict,
    ) -> H2HEdgeReport:
        """
        Genera il report completo H2H edge.

        Args:
            race_name:            nome del GP (es. "Miami GP 2026")
            driver_h2h_list:      output H2HCalculator.all_driver_h2h()
            constructor_h2h_list: output H2HCalculator.all_constructor_h2h()
            driver_odds:          {(da, db): {da: odd, db: odd}}
            constructor_odds:     {(ta, tb): {ta: odd, tb: odd}}

        Returns:
            H2HEdgeReport con tutti i value bets.
        """
        all_bets: list[H2HEdgeBet] = []

        driver_bets = self.evaluate_driver_bets(
            driver_h2h_list, driver_odds, race_name
        )
        all_bets.extend(driver_bets)

        constructor_bets = self.evaluate_constructor_bets(
            constructor_h2h_list, constructor_odds, race_name
        )
        all_bets.extend(constructor_bets)

        # Ordina per edge decrescente, poi per EV
        all_bets.sort(key=lambda b: (-b.edge, -b.ev))

        report = H2HEdgeReport(
            race_name=race_name,
            bets=all_bets,
            min_edge=self.min_edge,
            n_driver_pairs_evaluated=len(driver_h2h_list),
            n_constructor_pairs_evaluated=len(constructor_h2h_list),
        )

        return report
