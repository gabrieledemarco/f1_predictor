"""
f1_predictor/h2h/h2h_calculator.py
====================================
Calcola probabilità Head-to-Head driver e constructor
a partire dalle distribuzioni di posizione prodotte dal Monte Carlo.

Due approcci disponibili:
  - Analitico (default): usa position_distribution per calcolo esatto O(n²).
    Preferito quando il MC è già stato eseguito e le distribuzioni sono disponibili.
  - Da simulazione raw: se viene passata la matrice di simulazioni grezza
    (n_sims × n_drivers), il calcolo è esatto e veloce con numpy.

Formula analitica H2H:
  P(A batte B) = Σ_p  P(A finisce p) * Σ_{q > p} P(B finisce q)
               = Σ_p  P(A finisce p) * P(B finisce peggio di p)

Per i costruttori (best-of-two):
  P(team_X batte team_Y) = P(min_pos(X1,X2) < min_pos(Y1,Y2))
  dove X1,X2 sono i due piloti del team X.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Dataclasses output
# ---------------------------------------------------------------------------

@dataclass
class DriverH2H:
    """Probabilità H2H tra due piloti."""
    driver_a: str
    driver_b: str
    p_a_beats_b: float          # P(A finisce davanti a B), esclude DNF condivisi
    p_b_beats_a: float          # = 1 - p_a_beats_b (no pareggi in F1)
    n_positions: int            # numero posizioni coperte dalla distribuzione
    method: str = "analytic"    # "analytic" | "simulation"

    @property
    def favourite(self) -> str:
        return self.driver_a if self.p_a_beats_b >= 0.5 else self.driver_b

    @property
    def favourite_prob(self) -> float:
        return max(self.p_a_beats_b, self.p_b_beats_a)

    def to_dict(self) -> dict:
        return {
            "driver_a": self.driver_a,
            "driver_b": self.driver_b,
            "p_a_beats_b": round(self.p_a_beats_b, 4),
            "p_b_beats_a": round(self.p_b_beats_a, 4),
            "favourite": self.favourite,
            "favourite_prob": round(self.favourite_prob, 4),
            "method": self.method,
        }


@dataclass
class ConstructorH2H:
    """Probabilità H2H tra due costruttori (best-of-two piloti)."""
    team_a: str
    team_b: str
    drivers_a: list[str]        # 1 o 2 piloti del team A
    drivers_b: list[str]        # 1 o 2 piloti del team B
    p_a_beats_b: float
    p_b_beats_a: float
    method: str = "analytic"

    @property
    def favourite(self) -> str:
        return self.team_a if self.p_a_beats_b >= 0.5 else self.team_b

    @property
    def favourite_prob(self) -> float:
        return max(self.p_a_beats_b, self.p_b_beats_a)

    def to_dict(self) -> dict:
        return {
            "team_a": self.team_a,
            "team_b": self.team_b,
            "drivers_a": self.drivers_a,
            "drivers_b": self.drivers_b,
            "p_a_beats_b": round(self.p_a_beats_b, 4),
            "p_b_beats_a": round(self.p_b_beats_a, 4),
            "favourite": self.favourite,
            "favourite_prob": round(self.favourite_prob, 4),
            "method": self.method,
        }


# ---------------------------------------------------------------------------
# Core calculator
# ---------------------------------------------------------------------------

class H2HCalculator:
    """
    Calcola H2H driver e constructor dalle distribuzioni di posizione MC.

    Usage:
        calc = H2HCalculator()

        # Singolo H2H
        h2h = calc.driver_h2h(probs, "VER", "NOR")

        # Tutti i teammates (utile per H2H intra-team)
        pairs = calc.teammate_pairs(probs, driver_to_constructor)

        # Tutti i constructor H2H
        cons = calc.all_constructor_h2h(probs, driver_to_constructor)
    """

    # ---------------------------------------------------------------------------
    # Driver H2H
    # ---------------------------------------------------------------------------

    def driver_h2h(
        self,
        race_probs: dict,          # {driver_code: RaceProbability}
        driver_a: str,
        driver_b: str,
        sim_matrix: Optional[np.ndarray] = None,   # shape (n_sims, n_drivers)
        driver_index: Optional[dict] = None,        # {driver_code: col_index}
    ) -> DriverH2H:
        """
        Calcola P(A batte B) usando la position_distribution o la sim_matrix.

        Args:
            race_probs: output di pipeline.predict_race()["probabilities"]
            driver_a:   codice pilota A (es. "VER")
            driver_b:   codice pilota B (es. "NOR")
            sim_matrix: opzionale, matrice raw MC (n_sims × n_drivers)
            driver_index: mappa driver_code → colonna in sim_matrix

        Returns:
            DriverH2H con p_a_beats_b calcolata.
        """
        if driver_a not in race_probs:
            raise ValueError(f"Driver {driver_a!r} non trovato in race_probs")
        if driver_b not in race_probs:
            raise ValueError(f"Driver {driver_b!r} non trovato in race_probs")

        # Preferenza: usa simulation matrix se disponibile (più accurata)
        if sim_matrix is not None and driver_index is not None:
            return self._h2h_from_sim_matrix(
                sim_matrix, driver_index, driver_a, driver_b
            )

        return self._h2h_analytic(race_probs, driver_a, driver_b)

    def _h2h_analytic(
        self,
        race_probs: dict,
        driver_a: str,
        driver_b: str,
    ) -> DriverH2H:
        """
        Calcolo analitico da position_distribution.

        P(A batte B) = Σ_p  dist_A[p] * CDF_B(posizioni peggiori di p)

        Le posizioni sono 1-indexed (1 = 1° posto).
        I DNF vengono normalizzati a posizione 20+ e trattati come "sconfitte".
        """
        dist_a = race_probs[driver_a].position_distribution  # {pos: prob}
        dist_b = race_probs[driver_b].position_distribution

        # Normalizza e costruisci array 1..N (posizione 0 non esiste)
        n = max(
            max(dist_a.keys(), default=20),
            max(dist_b.keys(), default=20),
        )

        vec_a = np.zeros(n + 1)
        vec_b = np.zeros(n + 1)

        for pos, p in dist_a.items():
            if 1 <= pos <= n:
                vec_a[pos] += p
        for pos, p in dist_b.items():
            if 1 <= pos <= n:
                vec_b[pos] += p

        # Normalizza residui (massa mancante = DNF non mappato → ultima posizione)
        residual_a = max(0.0, 1.0 - vec_a[1:].sum())
        residual_b = max(0.0, 1.0 - vec_b[1:].sum())
        vec_a[n] += residual_a
        vec_b[n] += residual_b

        # CDF di B (probabilità che B finisca in posizione > p)
        # cdf_b_worse[p] = P(B finisce in posizione > p) = sum(vec_b[p+1:])
        # Calcolato come complemento della CDF cumulativa
        cdf_b = np.cumsum(vec_b)   # P(B finisce in posizione <= p)

        p_a_beats_b = 0.0
        for pos in range(1, n + 1):
            if vec_a[pos] < 1e-10:
                continue
            # P(B finisce peggio di pos) = 1 - P(B finisce <= pos)
            p_b_worse = 1.0 - cdf_b[pos]
            p_a_beats_b += vec_a[pos] * p_b_worse

        p_a_beats_b = float(np.clip(p_a_beats_b, 0.0, 1.0))
        p_b_beats_a = 1.0 - p_a_beats_b

        return DriverH2H(
            driver_a=driver_a,
            driver_b=driver_b,
            p_a_beats_b=p_a_beats_b,
            p_b_beats_a=p_b_beats_a,
            n_positions=n,
            method="analytic",
        )

    def _h2h_from_sim_matrix(
        self,
        sim_matrix: np.ndarray,
        driver_index: dict,
        driver_a: str,
        driver_b: str,
    ) -> DriverH2H:
        """
        Calcolo esatto da matrice MC raw.
        sim_matrix[s, i] = posizione finale del pilota i nella simulazione s.
        Posizione più bassa = migliore (1 = vittoria).
        """
        idx_a = driver_index[driver_a]
        idx_b = driver_index[driver_b]

        pos_a = sim_matrix[:, idx_a]
        pos_b = sim_matrix[:, idx_b]

        n_sims = sim_matrix.shape[0]
        p_a_beats_b = float((pos_a < pos_b).sum() / n_sims)
        p_b_beats_a = 1.0 - p_a_beats_b

        return DriverH2H(
            driver_a=driver_a,
            driver_b=driver_b,
            p_a_beats_b=p_a_beats_b,
            p_b_beats_a=p_b_beats_a,
            n_positions=int(sim_matrix.max()),
            method="simulation",
        )

    # ---------------------------------------------------------------------------
    # Batch H2H helpers
    # ---------------------------------------------------------------------------

    def all_driver_h2h(
        self,
        race_probs: dict,
        pairs: Optional[list[tuple[str, str]]] = None,
        sim_matrix: Optional[np.ndarray] = None,
        driver_index: Optional[dict] = None,
    ) -> list[DriverH2H]:
        """
        Calcola H2H per tutte le coppie specificate, oppure per tutte le combinazioni.

        Args:
            race_probs: output pipeline
            pairs: lista di (driver_a, driver_b); se None calcola tutte le combinazioni
            sim_matrix / driver_index: opzionali per metodo simulation

        Returns:
            Lista di DriverH2H, ordinata per favourite_prob decrescente.
        """
        from itertools import combinations

        drivers = list(race_probs.keys())
        if pairs is None:
            pairs = list(combinations(drivers, 2))

        results = []
        for a, b in pairs:
            try:
                h2h = self.driver_h2h(race_probs, a, b, sim_matrix, driver_index)
                results.append(h2h)
            except ValueError:
                continue

        return sorted(results, key=lambda x: x.favourite_prob, reverse=True)

    def teammate_pairs(
        self,
        race_probs: dict,
        driver_to_constructor: dict,  # {driver_code: constructor_ref}
        sim_matrix: Optional[np.ndarray] = None,
        driver_index: Optional[dict] = None,
    ) -> list[DriverH2H]:
        """Calcola H2H solo tra compagni di squadra."""
        from itertools import combinations
        from collections import defaultdict

        teams: dict[str, list[str]] = defaultdict(list)
        for driver, team in driver_to_constructor.items():
            if driver in race_probs:
                teams[team].append(driver)

        pairs = []
        for team_drivers in teams.values():
            if len(team_drivers) >= 2:
                pairs.extend(list(combinations(sorted(team_drivers), 2)))

        return self.all_driver_h2h(
            race_probs, pairs=pairs,
            sim_matrix=sim_matrix, driver_index=driver_index
        )

    # ---------------------------------------------------------------------------
    # Constructor H2H
    # ---------------------------------------------------------------------------

    def constructor_h2h(
        self,
        race_probs: dict,
        driver_to_constructor: dict,
        team_a: str,
        team_b: str,
        sim_matrix: Optional[np.ndarray] = None,
        driver_index: Optional[dict] = None,
    ) -> ConstructorH2H:
        """
        P(team_A batte team_B) = P(best_finish(A) < best_finish(B)).

        "Batte" significa che il miglior pilota classificato di A
        ha finito davanti al miglior pilota classificato di B.
        """
        drivers_a = [d for d, t in driver_to_constructor.items()
                     if t == team_a and d in race_probs]
        drivers_b = [d for d, t in driver_to_constructor.items()
                     if t == team_b and d in race_probs]

        if not drivers_a:
            raise ValueError(f"Nessun pilota trovato per {team_a!r}")
        if not drivers_b:
            raise ValueError(f"Nessun pilota trovato per {team_b!r}")

        if sim_matrix is not None and driver_index is not None:
            return self._constructor_h2h_sim(
                sim_matrix, driver_index,
                team_a, drivers_a, team_b, drivers_b
            )

        return self._constructor_h2h_analytic(
            race_probs, team_a, drivers_a, team_b, drivers_b
        )

    def _constructor_h2h_analytic(
        self,
        race_probs: dict,
        team_a: str, drivers_a: list[str],
        team_b: str, drivers_b: list[str],
    ) -> ConstructorH2H:
        """
        Approssimazione analitica per constructor H2H.

        Strategia: per ogni coppia (pilota_A, pilota_B), calcola P(A batte B).
        Poi la prob del constructor è la media delle migliori probabilità
        ponderate per i p_win di ciascun pilota (i.e., il pilota più forte
        del team conta di più).
        """
        # Pesi proporzionali a p_win (il pilota più forte domina)
        def weighted_p(team_drivers: list[str]) -> list[float]:
            wins = [race_probs[d].p_win + 1e-6 for d in team_drivers]
            total = sum(wins)
            return [w / total for w in wins]

        weights_a = weighted_p(drivers_a)
        weights_b = weighted_p(drivers_b)

        p_a_beats_b = 0.0
        for da, wa in zip(drivers_a, weights_a):
            for db, wb in zip(drivers_b, weights_b):
                h2h = self._h2h_analytic(race_probs, da, db)
                p_a_beats_b += wa * wb * h2h.p_a_beats_b

        p_a_beats_b = float(np.clip(p_a_beats_b, 0.0, 1.0))

        return ConstructorH2H(
            team_a=team_a,
            team_b=team_b,
            drivers_a=drivers_a,
            drivers_b=drivers_b,
            p_a_beats_b=p_a_beats_b,
            p_b_beats_a=1.0 - p_a_beats_b,
            method="analytic",
        )

    def _constructor_h2h_sim(
        self,
        sim_matrix: np.ndarray,
        driver_index: dict,
        team_a: str, drivers_a: list[str],
        team_b: str, drivers_b: list[str],
    ) -> ConstructorH2H:
        """
        Calcolo esatto da simulazione: best finish di ciascun team.
        """
        n_sims = sim_matrix.shape[0]

        def best_finish(drivers: list[str]) -> np.ndarray:
            cols = [driver_index[d] for d in drivers if d in driver_index]
            if not cols:
                return np.full(n_sims, 99)
            return sim_matrix[:, cols].min(axis=1)

        best_a = best_finish(drivers_a)
        best_b = best_finish(drivers_b)

        p_a_beats_b = float((best_a < best_b).sum() / n_sims)

        return ConstructorH2H(
            team_a=team_a,
            team_b=team_b,
            drivers_a=drivers_a,
            drivers_b=drivers_b,
            p_a_beats_b=p_a_beats_b,
            p_b_beats_a=1.0 - p_a_beats_b,
            method="simulation",
        )

    def all_constructor_h2h(
        self,
        race_probs: dict,
        driver_to_constructor: dict,
        pairs: Optional[list[tuple[str, str]]] = None,
        sim_matrix: Optional[np.ndarray] = None,
        driver_index: Optional[dict] = None,
    ) -> list[ConstructorH2H]:
        """
        Calcola H2H per tutte le coppie di costruttori.

        Args:
            pairs: lista di (team_a, team_b); se None usa tutte le combinazioni
        """
        from itertools import combinations
        from collections import defaultdict

        teams: dict[str, list[str]] = defaultdict(list)
        for driver, team in driver_to_constructor.items():
            if driver in race_probs:
                teams[team].append(driver)

        all_teams = list(teams.keys())
        if pairs is None:
            pairs = list(combinations(sorted(all_teams), 2))

        results = []
        for ta, tb in pairs:
            try:
                h2h = self.constructor_h2h(
                    race_probs, driver_to_constructor,
                    ta, tb, sim_matrix, driver_index
                )
                results.append(h2h)
            except ValueError:
                continue

        return sorted(results, key=lambda x: x.favourite_prob, reverse=True)
