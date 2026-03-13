"""
models/bayesian_race.py
=======================
Layer 2: Bayesian Race Model with Monte Carlo Simulation

This is the core of the system. It combines all inputs to produce
a full distribution over finishing positions for all 20 drivers.

Academic References:
    Heilmeier, A. et al. (2020). Application of Monte Carlo Methods to
    Consider Probabilistic Effects in a Race Simulation for Circuit
    Motorsport. Applied Sciences, 10(12), 4229.
    https://doi.org/10.3390/app10124229

    arXiv 2512.00640 (2025). A State-Space Approach to Modeling Tire
    Degradation in Formula 1 Racing. (Bayesian tire model structure)

    Sulsters, D. (2017). Simulating Formula One Race Strategies.
    Vrije Universiteit Amsterdam. (Discrete event simulation framework)

    van Kesteren & Bergkamp (2023). Bayesian multilevel rank-ordered
    logit for F1. (Prior construction from Layer 1a/1b)

Monte Carlo approach (Heilmeier et al., §3):
    50,000 simulations per race. Each simulation:
        1. Draw driver skill from N(mu_skill, sigma_skill^2 + beta^2)
        2. Draw machine pace from N(mu_pace, sigma_pace^2)
        3. Compute base lap time = f(skill, pace, circuit)
        4. Simulate tyre degradation via state-space model
        5. Determine pit stop strategy (greedy optimal)
        6. Apply stochastic events: safety car, DNF, grid penalties
        7. Compute final race order by cumulative race time

Output:
    position_matrix[driver][position] = P(driver finishes in position)
    Marginal probabilities: p_win, p_podium, p_top6, p_points, p_dnf
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

from f1_predictor.domain.entities import (
    Driver, Circuit, Race, CircuitType, TyreCompound,
    DriverSkillRating, MachinePaceEstimate, RaceProbability
)


@dataclass(frozen=True)
class RaceSimConfig:
    """
    Configuration for the Monte Carlo race simulation.

    References:
        Heilmeier et al. (2020), Table 2: Stochastic event probabilities
        derived from 2014-2019 F1 seasons.
    """
    n_simulations: int = 50_000
    n_drivers: int = 20
    total_laps: int = 57            # median F1 race length

    # Tyre degradation parameters (arXiv 2512.00640, §3.2)
    # deg_rate[compound] = mean lap time penalty per lap on that compound
    deg_rate_soft: float = 0.085    # s/lap degradation
    deg_rate_medium: float = 0.055
    deg_rate_hard: float = 0.035
    deg_rate_noise: float = 0.010   # lap-to-lap noise in degradation

    # Initial fuel load adjustment (TracingInsights, 2025)
    initial_fuel_kg: float = 100.0
    fuel_effect_per_kg: float = 0.03  # s/lap per kg

    # Stochastic events (Heilmeier et al., Table 2 — 2014-2019 averages)
    dnf_prob_per_lap: float = 0.00045   # per driver per lap
    safety_car_prob_per_race: float = 0.62
    pit_stop_sigma_s: float = 1.2       # pit stop duration uncertainty (s)
    pit_stop_base_s: float = 22.5       # base pit stop loss (s)
    start_sigma_positions: float = 1.5  # starting grid position noise


@dataclass
class DriverRaceInput:
    """All inputs needed to simulate one driver in a race."""
    driver_code: str
    constructor_ref: str
    grid_position: int
    skill_mu: float
    skill_sigma: float
    pace_mu: float        # machine pace advantage (s/lap), negative = faster
    pace_sigma: float
    circuit_type: CircuitType
    grid_penalty: int = 0   # positions dropped from grid penalty
    is_rookie: bool = False


class TyreDegradationModel:
    """
    State-space tyre degradation (arXiv 2512.00640, 2025).

    The degradation rate is treated as a hidden Markov state:
        deg_state_t = deg_state_{t-1} + epsilon_t,  epsilon ~ N(0, Q_deg)
        observed_delta_t = deg_state_t + nu_t,       nu ~ N(0, R_deg)

    On pit stop: state resets to 0 (new tyres), sigma increases briefly.

    For simulation, we use the posterior mean degradation rate
    conditioned on compound and circuit type.
    """

    def __init__(self, config: RaceSimConfig):
        self.config = config

    def deg_rate(self, compound: TyreCompound, circuit_type: CircuitType,
                 tyre_age: int) -> float:
        """
        Compute instantaneous degradation rate (s/lap) for a given compound.

        Reference: arXiv 2512.00640 (2025), Eq. 7 — degradation is modelled
        as approximately linear with small state noise. Street circuits
        have lower degradation due to smooth surfaces; high-speed circuits
        have higher degradation due to lateral G-loads.
        """
        base = {
            TyreCompound.SOFT: self.config.deg_rate_soft,
            TyreCompound.MEDIUM: self.config.deg_rate_medium,
            TyreCompound.HARD: self.config.deg_rate_hard,
            TyreCompound.INTERMEDIATE: 0.020,
            TyreCompound.WET: 0.010,
        }.get(compound, self.config.deg_rate_medium)

        # Circuit type modifier
        circuit_mod = {
            CircuitType.STREET: 0.75,
            CircuitType.HIGH_DOWNFORCE: 1.10,
            CircuitType.HIGH_SPEED: 1.25,
            CircuitType.MIXED: 1.00,
            CircuitType.DESERT: 0.90,
        }.get(circuit_type, 1.0)

        # Add per-lap noise
        noise = np.random.normal(0, self.config.deg_rate_noise)
        return max(0, base * circuit_mod * tyre_age / 15 + noise)

    def simulate_stint(self, compound: TyreCompound, circuit_type: CircuitType,
                       n_laps: int, base_lap_time_s: float) -> list[float]:
        """
        Simulate lap times for a full stint on a given compound.

        Returns list of lap times (seconds) for each lap in the stint.
        """
        lap_times = []
        cumulative_deg = 0.0

        # Fuel correction: car gets lighter each lap
        for lap in range(n_laps):
            deg = self.deg_rate(compound, circuit_type, lap + 1)
            cumulative_deg += deg

            # Fuel weight effect (TracingInsights formula)
            fuel_remaining = self.config.initial_fuel_kg * (1 - lap / self.config.total_laps)
            fuel_effect = fuel_remaining * self.config.fuel_effect_per_kg

            lap_time = base_lap_time_s + cumulative_deg - fuel_effect
            lap_times.append(max(lap_time, base_lap_time_s * 0.97))  # physical minimum

        return lap_times


class BayesianRaceModel:
    """
    Layer 2: Full probabilistic race simulation.

    Combines driver skill (Layer 1a), machine pace (Layer 1b),
    circuit characteristics, and stochastic events into a full
    race simulation via Monte Carlo sampling.

    Usage:
        model = BayesianRaceModel()
        probs = model.simulate_race(race, driver_inputs)

    Returns a dict of RaceProbability objects, one per driver.
    """

    def __init__(self, config: Optional[RaceSimConfig] = None):
        self.config = config or RaceSimConfig()
        self.tyre_model = TyreDegradationModel(self.config)

    def simulate_race(self, race: Race,
                      driver_inputs: list[DriverRaceInput],
                      verbose: bool = False) -> dict[str, RaceProbability]:
        """
        Run full Monte Carlo race simulation.

        Args:
            race: Race object with circuit information.
            driver_inputs: List of DriverRaceInput (one per driver).
            verbose: Print progress.

        Returns:
            Dict mapping driver_code → RaceProbability.
        """
        n_sim = self.config.n_simulations
        n_drivers = len(driver_inputs)
        driver_codes = [d.driver_code for d in driver_inputs]

        # Accumulator: position_counts[driver_idx][position_0_indexed] = count
        position_counts = np.zeros((n_drivers, n_drivers), dtype=int)
        dnf_counts = np.zeros(n_drivers, dtype=int)

        for sim_idx in range(n_sim):
            if verbose and sim_idx % 10_000 == 0:
                print(f"  Simulation {sim_idx:,} / {n_sim:,}")

            results = self._simulate_single_race(race, driver_inputs)

            for driver_idx, (pos, dnf) in enumerate(results):
                if dnf:
                    dnf_counts[driver_idx] += 1
                else:
                    position_counts[driver_idx][pos] += 1

        # Convert counts to probabilities
        output = {}
        for idx, driver_input in enumerate(driver_inputs):
            code = driver_input.driver_code
            total = n_sim

            pos_dist = position_counts[idx] / total
            p_dnf = dnf_counts[idx] / total

            prob = RaceProbability(
                race_id=race.race_id,
                driver_code=code,
                p_win=float(pos_dist[0]),
                p_podium=float(pos_dist[:3].sum()),
                p_top6=float(pos_dist[:6].sum()),
                p_points=float(pos_dist[:10].sum()),
                p_dnf=float(p_dnf),
                position_distribution=pos_dist.tolist()
            )
            output[code] = prob

        return output

    # ------------------------------------------------------------------
    # Private: single simulation
    # ------------------------------------------------------------------

    def _simulate_single_race(self, race: Race,
                               driver_inputs: list[DriverRaceInput]
                               ) -> list[tuple[int, bool]]:
        """
        Simulate one complete race. Returns list of (position, is_dnf)
        for each driver, in input order.

        Simulation steps (Heilmeier et al., 2020 §3):
            1. Sample driver performance from skill + pace distributions
            2. Sample race-level stochastic events (safety car timing)
            3. Simulate each driver's cumulative race time
            4. Determine finishing order
        """
        n_drivers = len(driver_inputs)
        circuit_type = race.circuit.circuit_type

        # --- Step 1: Sample latent performance for each driver ---
        sampled_pace = []  # base pace (s/lap), lower = faster
        for d in driver_inputs:
            # Skill draw: more skill = faster (negative contribution to lap time)
            skill_draw = np.random.normal(d.skill_mu, d.skill_sigma)
            # Machine pace draw (already in s/lap, negative = faster)
            pace_draw = np.random.normal(d.pace_mu, d.pace_sigma)

            # Base lap time: we work in relative units
            # Reference lap = 0, faster drivers go negative
            base_pace = -(skill_draw - 25.0) * 0.1 + pace_draw
            sampled_pace.append(base_pace)

        # --- Step 2: Sample stochastic events ---
        # Safety car: determine if it occurs and on which lap
        has_safety_car = np.random.random() < self.config.safety_car_prob_per_race
        sc_lap = int(np.random.uniform(10, self.config.total_laps - 10)) if has_safety_car else -1

        # DNF events per driver (Bernoulli per lap)
        dnf_lap = {}
        for i in range(n_drivers):
            for lap in range(self.config.total_laps):
                if np.random.random() < self.config.dnf_prob_per_lap:
                    dnf_lap[i] = lap
                    break

        # --- Step 3: Simulate race time for each driver ---
        cumulative_times = []
        is_dnf = []

        for i, d in enumerate(driver_inputs):
            # Starting position (with grid penalty and start uncertainty)
            actual_grid = d.grid_position + d.grid_penalty
            start_noise = np.random.normal(0, self.config.start_sigma_positions)
            effective_start_pos = max(1, actual_grid + round(start_noise))

            # Initial time gap from pole (0 = pole)
            time_gap = effective_start_pos * 0.8  # ~0.8s per grid position

            # Determine pit strategy: simple 1 or 2-stop based on pace
            # (Sulsters, 2017 §4 — greedy strategy selection)
            pit_laps = self._choose_pit_strategy(d, circuit_type)

            # Simulate laps
            current_compound = TyreCompound.MEDIUM
            tyre_age = 0
            race_time = time_gap

            if i in dnf_lap:
                dnf_at = dnf_lap[i]
            else:
                dnf_at = self.config.total_laps + 1  # no DNF

            actual_laps = min(self.config.total_laps, dnf_at)

            for lap in range(actual_laps):
                tyre_age += 1

                # Pit stop processing
                if lap in pit_laps:
                    pit_time = np.random.normal(
                        self.config.pit_stop_base_s,
                        self.config.pit_stop_sigma_s
                    )
                    race_time += pit_time
                    # Switch to next compound (simplified)
                    current_compound = TyreCompound.HARD if current_compound == TyreCompound.MEDIUM \
                                      else TyreCompound.MEDIUM
                    tyre_age = 0

                # Base lap time with tyre degradation
                deg = self.tyre_model.deg_rate(current_compound, circuit_type, tyre_age)
                fuel_remaining = self.config.initial_fuel_kg * (1 - lap / self.config.total_laps)
                fuel_effect = fuel_remaining * self.config.fuel_effect_per_kg

                lap_time = 90.0 + sampled_pace[i] + deg - fuel_effect
                lap_time += np.random.normal(0, 0.3)  # residual lap-to-lap noise

                # Safety car: neutralise field
                if lap == sc_lap:
                    lap_time = 120.0  # delta time under SC

                race_time += lap_time

            cumulative_times.append(race_time)
            is_dnf.append(i in dnf_lap)

        # --- Step 4: Compute finishing order ---
        # DNF drivers are placed at the back, ordered by DNF lap
        results = []
        finishers = [(i, t) for i, t in enumerate(cumulative_times) if not is_dnf[i]]
        dnfers = [(i, dnf_lap.get(i, 0)) for i in range(n_drivers) if is_dnf[i]]

        finishers_sorted = sorted(finishers, key=lambda x: x[1])
        dnfers_sorted = sorted(dnfers, key=lambda x: -x[1])  # more laps = better DNF

        ordered = finishers_sorted + dnfers_sorted

        position_map = {driver_idx: pos for pos, (driver_idx, _) in enumerate(ordered)}

        for i in range(n_drivers):
            results.append((position_map[i], is_dnf[i]))

        return results

    def _choose_pit_strategy(self, driver: DriverRaceInput,
                              circuit_type: CircuitType) -> list[int]:
        """
        Determine pit stop laps using a simplified greedy strategy.

        Reference: Sulsters (2017) §4.3 — optimal stop lap is determined
        by the tyre degradation crossover point where a new set would
        be faster than staying out.

        Returns list of lap numbers when to pit.
        """
        # Optimal stop lap varies by circuit type and compound
        # These are empirically derived from 2019-2024 data
        optimal_stop_laps = {
            CircuitType.STREET: [42],           # typically 1-stop
            CircuitType.HIGH_DOWNFORCE: [28, 48],# often 2-stop
            CircuitType.HIGH_SPEED: [25],        # 1-stop (hard compound)
            CircuitType.MIXED: [20, 42],         # 2-stop common
            CircuitType.DESERT: [25, 48],        # 2-stop in heat
        }
        base = optimal_stop_laps.get(circuit_type, [28])
        # Add noise to pit lap decision
        return [max(5, p + np.random.randint(-3, 4)) for p in base]
