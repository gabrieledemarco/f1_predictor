"""
models/driver_skill.py
======================
Layer 1a: Driver Skill Rating via TrueSkill Through Time (TTT)

Academic Reference:
    Dangauthier, P., Herbrich, R., Minka, T., & Graepel, T. (2007).
    TrueSkill Through Time: Revisiting the History of Chess.
    Advances in Neural Information Processing Systems (NeurIPS).
    https://www.herbrich.me/papers/ttt.pdf

    Extended by: Minka et al. (2018) TrueSkill 2 — incorporates
    per-circuit-type conditioning for heterogeneous performance contexts.

Key design decisions:
    1. Skill is modelled as a Gaussian random walk: between races,
       sigma grows by tau (process noise), reflecting skill evolution.
    2. Ratings are conditioned on circuit_type to capture that Alonso
       on street circuits has a different skill distribution than Alonso
       on high-speed circuits (van Kesteren & Bergkamp, 2023, §3.2).
    3. Conservative rating (mu - 3*sigma) is used for display/comparison
       to avoid overconfident orderings when sigma is large (rookies).

Implementation notes:
    We implement a simplified Gaussian belief propagation version of TTT
    (schedule-based message passing). For full MCMC, use the
    'trueskill-through-time' pip package (requires network access).
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict

from f1_predictor.domain.entities import (
    Driver, RaceResult, DriverSkillRating, CircuitType
)


# ---------------------------------------------------------------------------
# TTT hyperparameters (see Dangauthier et al., Table 1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TTTConfig:
    """
    Configuration for TrueSkill Through Time.

    mu_0 (float):
        Prior mean for driver skill. Set to 25.0 (TrueSkill convention).
    sigma_0 (float):
        Prior uncertainty. Large for new drivers, shrinks as we observe races.
    beta (float):
        Performance noise — spread of a single race performance around
        latent skill. Typically sigma_0 / 2.
    tau (float):
        Process noise — how much skill can change between races.
        Captures learning, physical form, team changes.
        Dangauthier et al. recommend tau ~ 0.1 * sigma_0.
    draw_margin (float):
        Not used in F1 (no draws), set to 0.
    decay_factor (float):
        Between-season decay: sigma increases by this fraction at
        season end to model regulation changes / car resets.
        Set higher (0.30) when regulations change drastically (e.g., 2022).
    """
    mu_0: float = 25.0
    sigma_0: float = 8.333
    beta: float = 4.167       # performance noise per race
    tau: float = 0.833        # process noise (skill drift between races)
    draw_margin: float = 0.0
    decay_factor: float = 0.15

    @classmethod
    def for_2026(cls) -> "TTTConfig":
        """
        TASK 3.3 — Preset calibrato per il regolamento 2026.

        Il regolamento 2026 introduce:
        - Motore ibrido 50/50 (nuovi PU, ranking incerti)
        - Active aero in sostituzione del DRS (nuova curva di apprendimento)
        - Nuovi entrant PU (Audi/sauber, Honda standalone)

        Conseguenze per il modello:
        - beta aumentato: piu' varianza nelle singole gare (novita' tecnica)
        - tau aumentato: skill puo' cambiare piu' velocemente (adattamento)
        - decay_factor aumentato: reset stagionale piu' aggressivo
        """
        return cls(
            mu_0=25.0,
            sigma_0=8.333,
            beta=5.5,          # > 4.167 default: piu' upset possibili con nuovi sistemi
            tau=1.0,           # > 0.833 default: adattamento piu' rapido al nuovo regolamento
            draw_margin=0.0,
            decay_factor=0.25, # > 0.15 default: reset stagionale amplificato (2026 e' una soglia)
        )


# ---------------------------------------------------------------------------
# Gaussian utilities (message passing building blocks)
# ---------------------------------------------------------------------------

def _gaussian_prod(mu1: float, sigma1: float,
                   mu2: float, sigma2: float) -> tuple[float, float]:
    """Product of two Gaussian PDFs (unnormalized), returns (mu, sigma)."""
    var1, var2 = sigma1**2, sigma2**2
    var_new = 1.0 / (1.0/var1 + 1.0/var2)
    mu_new = var_new * (mu1/var1 + mu2/var2)
    return mu_new, np.sqrt(var_new)


def _update_from_comparison(winner_mu: float, winner_sigma: float,
                             loser_mu: float, loser_sigma: float,
                             beta: float) -> tuple[tuple, tuple]:
    """
    Single pairwise comparison update (Herbrich et al., 2007 §4).
    Returns updated (mu, sigma) for winner and loser.

    The update is derived from the factor graph marginalisation:
        P(winner beats loser) = Phi((winner_mu - loser_mu) / c)
        where c = sqrt(2*beta^2 + winner_sigma^2 + loser_sigma^2)
    """
    c = np.sqrt(2 * beta**2 + winner_sigma**2 + loser_sigma**2)
    delta = (winner_mu - loser_mu) / c

    # Truncated Gaussian factors (v and w functions)
    from scipy.stats import norm
    pdf_val = norm.pdf(delta)
    cdf_val = norm.cdf(delta)

    if cdf_val < 1e-10:
        cdf_val = 1e-10

    v = pdf_val / cdf_val
    w = v * (v + delta)

    # Winner update
    w_mu_new = winner_mu + (winner_sigma**2 / c) * v
    w_sigma_new = winner_sigma * np.sqrt(1 - (winner_sigma**2 / c**2) * w)

    # Loser update
    l_mu_new = loser_mu - (loser_sigma**2 / c) * v
    l_sigma_new = loser_sigma * np.sqrt(1 - (loser_sigma**2 / c**2) * w)

    return (w_mu_new, w_sigma_new), (l_mu_new, l_sigma_new)


# ---------------------------------------------------------------------------
# Main DriverSkillModel
# ---------------------------------------------------------------------------

class DriverSkillModel:
    """
    Layer 1a: Driver Skill Rating via TrueSkill Through Time.

    Usage:
        model = DriverSkillModel()
        model.fit(race_results_list)  # list of RaceResult, ordered by date
        rating = model.get_rating("VER", circuit_type=CircuitType.STREET)

    The model maintains:
        - Global ratings: per-driver, ignoring circuit type
        - Conditional ratings: per-driver × per-circuit-type

    Both are updated simultaneously on each race.
    """

    def __init__(self, config: Optional[TTTConfig] = None):
        self.config = config or TTTConfig()
        self._ratings: dict[str, dict] = defaultdict(self._default_rating)
        self._ratings_by_circuit: dict[tuple, dict] = defaultdict(self._default_rating)
        # TASK 5.2 — conta le gare osservate per (driver, circuit_type)
        # necessario per il blending ponderato in get_rating()
        self._n_races_by_circuit: dict[tuple, int] = defaultdict(int)
        self._history: list[dict] = []

    def _default_rating(self) -> dict:
        return {"mu": self.config.mu_0, "sigma": self.config.sigma_0}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, results: list[RaceResult],
            race_metadata: Optional[dict] = None) -> "DriverSkillModel":
        """
        Train the model by processing race results in chronological order.

        Args:
            results: List of RaceResult objects sorted by date.
            race_metadata: Optional dict mapping race_id → {circuit_type, year}.

        Returns:
            self (fluent interface for chaining).
        """
        # Group results by race
        races: dict[int, list[RaceResult]] = defaultdict(list)
        for r in results:
            races[r.race_id].append(r)

        for race_id, race_results in sorted(races.items()):
            circuit_type = None
            year = None
            if race_metadata and race_id in race_metadata:
                circuit_type = race_metadata[race_id].get("circuit_type")
                year = race_metadata[race_id].get("year")

            self._process_race(race_id, race_results, circuit_type, year)

        return self

    def get_rating(self, driver_code: str,
                   circuit_type: Optional[CircuitType] = None) -> DriverSkillRating:
        """
        Retrieve current rating for a driver.

        TASK 5.2 — Circuit-type blending:
        Se il pilota ha >= MIN_CIRCUIT_RACES gare su questo tipo di circuito,
        usa un blend pesato fra rating globale e rating circuit-specifico.
        Sotto soglia, il rating circuit-specifico ha troppa incertezza
        (sigma alta) per essere affidabile da solo — si usa solo il globale.

        Blend:
            w_circuit = min(n_races / MIN_CIRCUIT_RACES, 1.0)
            mu_blend = (1-w) * mu_global + w * mu_circuit
            sigma_blend = min(sigma_global, sigma_circuit) * 1.05

        Questo cattura il fatto che Alonso su circuiti stradali e' piu'
        forte del suo rating globale, ma evita noise per rookies con pochi dati.

        Args:
            driver_code: 3-letter FIA code (e.g., 'VER').
            circuit_type: If provided, returns blended circuit-conditional rating.

        Returns:
            DriverSkillRating with mu, sigma, and conservative_rating.
        """
        MIN_CIRCUIT_RACES = 5  # soglia minima per dare peso al rating circuit-specifico

        global_r = self._ratings[driver_code]

        if circuit_type is None:
            return DriverSkillRating(
                driver_code=driver_code,
                circuit_type=None,
                mu=global_r["mu"],
                sigma=global_r["sigma"],
            )

        key = (driver_code, circuit_type)
        circuit_r = self._ratings_by_circuit[key]
        n_circuit = self._n_races_by_circuit[key]

        if n_circuit < MIN_CIRCUIT_RACES:
            # Troppo pochi dati sul circuit type: usa rating globale puro
            return DriverSkillRating(
                driver_code=driver_code,
                circuit_type=circuit_type,
                mu=global_r["mu"],
                sigma=global_r["sigma"],
            )

        # Blend pesato: piu' gare su quel tipo → piu' peso al rating specifico
        w = min(n_circuit / MIN_CIRCUIT_RACES, 1.0)  # satura a 1.0 dopo MIN_CIRCUIT_RACES gare
        mu_blend    = (1 - w) * global_r["mu"]    + w * circuit_r["mu"]
        sigma_blend = min(global_r["sigma"], circuit_r["sigma"]) * 1.05  # leggero aumento incertezza

        return DriverSkillRating(
            driver_code=driver_code,
            circuit_type=circuit_type,
            mu=float(mu_blend),
            sigma=float(sigma_blend),
        )

    def get_all_ratings(self, circuit_type: Optional[CircuitType] = None
                        ) -> list[DriverSkillRating]:
        """Return all ratings sorted by conservative skill (descending)."""
        if circuit_type is None:
            ratings = [
                self.get_rating(code) for code in self._ratings
            ]
        else:
            drivers = {key[0] for key in self._ratings_by_circuit
                       if key[1] == circuit_type}
            ratings = [self.get_rating(d, circuit_type) for d in drivers]

        return sorted(ratings, key=lambda r: r.conservative_rating, reverse=True)

    def apply_season_decay(self, is_major_regulation_change: bool = False):
        """
        Apply between-season decay (increase sigma to model uncertainty reset).

        Call this at the end of each season before fitting the next season.

        Reference: TTT §5 — between-event drift is modelled as additional
        process noise proportional to time elapsed.

        Args:
            is_major_regulation_change: If True (e.g., 2022), applies 2× decay
                since car performance resets drastically.
        """
        factor = self.config.decay_factor
        if is_major_regulation_change:
            factor *= 2.0

        for code in self._ratings:
            self._ratings[code]["sigma"] = np.sqrt(
                self._ratings[code]["sigma"]**2 + (self.config.mu_0 * factor)**2
            )
        for key in self._ratings_by_circuit:
            self._ratings_by_circuit[key]["sigma"] = np.sqrt(
                self._ratings_by_circuit[key]["sigma"]**2 + (self.config.mu_0 * factor)**2
            )

    def predict_win_probabilities(self, driver_codes: list[str],
                                   circuit_type: Optional[CircuitType] = None,
                                   n_simulations: int = 10_000) -> dict[str, float]:
        """
        Estimate win probability for each driver via Monte Carlo sampling.

        Each simulation draws a performance sample for each driver from
        N(mu, sigma^2 + beta^2) and the driver with the highest sample wins.

        Reference: Ingram (2021) — Bradley-Terry-Luce sampling for F1.

        Args:
            driver_codes: List of drivers to include.
            circuit_type: Circuit type for conditional ratings.
            n_simulations: Number of Monte Carlo draws.

        Returns:
            Dict mapping driver_code → estimated win probability.
        """
        ratings = {
            code: self.get_rating(code, circuit_type)
            for code in driver_codes
        }

        wins = defaultdict(int)
        for _ in range(n_simulations):
            performances = {
                code: np.random.normal(r.mu, np.sqrt(r.sigma**2 + self.config.beta**2))
                for code, r in ratings.items()
            }
            winner = max(performances, key=performances.get)
            wins[winner] += 1

        return {code: wins[code] / n_simulations for code in driver_codes}

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _process_race(self, race_id: int, results: list[RaceResult],
                      circuit_type: Optional[CircuitType], year: Optional[int]):
        """
        Update ratings after one race using pairwise comparison updates.

        Strategy (from Dangauthier et al., §4):
            For each ordered pair (better, worse) of finishers,
            apply one Gaussian update. This is the 'round-robin'
            approximation — computationally efficient and empirically
            similar to full belief propagation.
        """
        # Apply process noise (skill drift since last race)
        self._apply_process_noise()

        # Sort by finish position (DNFs go last, ordered by laps completed)
        finished = [r for r in results if r.finish_position is not None]
        dnfs = [r for r in results if r.finish_position is None]
        dnfs_sorted = sorted(dnfs, key=lambda r: -r.laps_completed)
        ordered = sorted(finished, key=lambda r: r.finish_position) + dnfs_sorted

        # Pairwise updates
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                winner_code = ordered[i].driver_code
                loser_code = ordered[j].driver_code

                # Global rating update
                w_r, l_r = self._ratings[winner_code], self._ratings[loser_code]
                (w_mu, w_sig), (l_mu, l_sig) = _update_from_comparison(
                    w_r["mu"], w_r["sigma"],
                    l_r["mu"], l_r["sigma"],
                    self.config.beta
                )
                self._ratings[winner_code] = {"mu": w_mu, "sigma": w_sig}
                self._ratings[loser_code]  = {"mu": l_mu, "sigma": l_sig}

                # Circuit-conditional update (if circuit_type known)
                if circuit_type is not None:
                    wk = (winner_code, circuit_type)
                    lk = (loser_code, circuit_type)
                    cw_r = self._ratings_by_circuit[wk]
                    cl_r = self._ratings_by_circuit[lk]
                    (cw_mu, cw_sig), (cl_mu, cl_sig) = _update_from_comparison(
                        cw_r["mu"], cw_r["sigma"],
                        cl_r["mu"], cl_r["sigma"],
                        self.config.beta
                    )
                    self._ratings_by_circuit[wk] = {"mu": cw_mu, "sigma": cw_sig}
                    self._ratings_by_circuit[lk] = {"mu": cl_mu, "sigma": cl_sig}

        # TASK 5.2 — aggiorna contatore gare per circuit type (per blending in get_rating)
        if circuit_type is not None:
            for res in ordered:
                key = (res.driver_code, circuit_type)
                self._n_races_by_circuit[key] += 1

        # Log snapshot for history
        snapshot = {
            "race_id": race_id,
            "ratings": {
                code: {"mu": r["mu"], "sigma": r["sigma"]}
                for code, r in self._ratings.items()
            }
        }
        self._history.append(snapshot)

    def _apply_process_noise(self):
        """Increase sigma by tau (random walk component between races)."""
        tau = self.config.tau
        for code in self._ratings:
            old_sigma = self._ratings[code]["sigma"]
            self._ratings[code]["sigma"] = np.sqrt(old_sigma**2 + tau**2)
        for key in self._ratings_by_circuit:
            old_sigma = self._ratings_by_circuit[key]["sigma"]
            self._ratings_by_circuit[key]["sigma"] = np.sqrt(old_sigma**2 + tau**2)

    def get_rating_history(self, driver_code: str) -> list[dict]:
        """Return time series of ratings for a driver (for visualization)."""
        return [
            {
                "race_id": snap["race_id"],
                "mu": snap["ratings"].get(driver_code, {}).get("mu", self.config.mu_0),
                "sigma": snap["ratings"].get(driver_code, {}).get("sigma", self.config.sigma_0),
            }
            for snap in self._history
            if driver_code in snap["ratings"]
        ]
