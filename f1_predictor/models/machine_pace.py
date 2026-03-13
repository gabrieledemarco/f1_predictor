"""
models/machine_pace.py
======================
Layer 1b: Machine Pace Model via Kalman Filter

Tracks the latent competitive performance of each constructor over time,
treating technical upgrade packages as observable "shocks" to the state.

Academic Reference:
    arXiv 2512.00640 (2025) — "A State-Space Approach to Modeling Tire
    Degradation in Formula 1 Racing". We adapt their state-space framework
    from tire degradation to overall machine pace tracking.

    Standard Kalman Filter formulation follows:
    Welch & Bishop (2006) — "An Introduction to the Kalman Filter",
    UNC-Chapel Hill TR 95-041.

State-space model:
    State:   x_t = mu_pace_t  (latent pace advantage in seconds/lap vs field)
    Obs:     z_t = relative_pace_observed + noise
    Transition: x_t = x_{t-1} + upgrade_effect_t + w_t,  w_t ~ N(0, Q)
    Observation: z_t = x_t + v_t,                          v_t ~ N(0, R)

Key insight (Heilmeier et al., 2020 §2.3):
    Machine pace is measured as:
        pace_relative = (constructor_median_lap - field_median_lap) / field_median_lap
    This normalizes out circuit-specific effects, leaving pure competitive delta.

Note on upgrade shocks:
    When a constructor introduces a major upgrade, we add a known
    observation to the filter: z_upgrade = estimated_gain ± uncertainty.
    This is an informative prior injection rather than a pure measurement.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

from f1_predictor.domain.entities import MachinePaceEstimate, LapData


@dataclass(frozen=True)
class KalmanConfig:
    """
    Kalman Filter hyperparameters for machine pace tracking.

    Q (float): Process noise variance.
        How much pace can change between races (development rate).
        Higher Q = faster adaptation. Typical: 0.001 (s/lap)^2.
    R (float): Observation noise variance.
        How noisy is the per-race pace measurement.
        Typical: 0.004 (s/lap)^2 (2× sqrt(R) ≈ 0.13s per lap).
    x0 (float): Initial pace estimate (0 = at field average).
    P0 (float): Initial estimate uncertainty.
    upgrade_uncertainty (float): Default sigma for upgrade shocks.
    """
    Q: float = 0.001        # process noise
    R: float = 0.004        # observation noise
    x0: float = 0.0
    P0: float = 0.1
    upgrade_uncertainty: float = 0.05


class MachinePaceModel:
    """
    Layer 1b: Constructor pace tracking via Kalman Filter.

    Maintains one independent Kalman Filter per constructor.
    The filter state is the pace advantage (negative = faster than field).

    Usage:
        model = MachinePaceModel()
        model.update("red_bull", race_id=1, observed_pace=-0.42)
        model.inject_upgrade("mercedes", race_id=5, estimated_gain=-0.15)
        estimate = model.get_estimate("red_bull", race_id=6)

    Pace sign convention:
        Negative = faster (fewer seconds per lap than field median).
        This is consistent with the F1 timing convention.
    """

    def __init__(self, config: Optional[KalmanConfig] = None):
        self.config = config or KalmanConfig()
        self._states: dict[str, dict] = defaultdict(self._init_state)
        self._history: dict[str, list] = defaultdict(list)

    def _init_state(self) -> dict:
        return {"x": self.config.x0, "P": self.config.P0}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, constructor_ref: str, race_id: int,
               observed_pace: float) -> MachinePaceEstimate:
        """
        Update pace estimate with a new observation.

        Args:
            constructor_ref: e.g. 'red_bull', 'mercedes'.
            race_id: Numeric race ID for history tracking.
            observed_pace: Observed pace relative to field (s/lap).
                           Negative = faster than field.

        Returns:
            MachinePaceEstimate after update.
        """
        state = self._states[constructor_ref]

        # --- Predict step ---
        x_pred = state["x"]                    # no drift (zero mean model)
        P_pred = state["P"] + self.config.Q    # state uncertainty grows

        # --- Update step ---
        K = P_pred / (P_pred + self.config.R)          # Kalman gain
        x_upd = x_pred + K * (observed_pace - x_pred)  # posterior mean
        P_upd = (1 - K) * P_pred                        # posterior variance

        self._states[constructor_ref] = {"x": x_upd, "P": P_upd}

        estimate = MachinePaceEstimate(
            constructor_ref=constructor_ref,
            race_id=race_id,
            mu_pace=x_upd,
            sigma_pace=np.sqrt(P_upd),
            kalman_gain=K
        )
        self._history[constructor_ref].append({
            "race_id": race_id, "x": x_upd, "P": P_upd,
            "K": K, "observed": observed_pace
        })
        return estimate

    def inject_upgrade(self, constructor_ref: str, race_id: int,
                       estimated_gain: float,
                       uncertainty: Optional[float] = None):
        """
        Inject a technical upgrade as an informative observation.

        This implements a 'measurement update with known physics' —
        we treat the upgrade as an observation z = current_state + gain,
        with uncertainty = upgrade_uncertainty from config.

        Reference: Adapts the 'known input' formulation from
        Welch & Bishop (2006) §2.2 (control input model).

        Args:
            constructor_ref: e.g. 'mercedes'.
            race_id: Race at which upgrade is first available.
            estimated_gain: Expected lap time improvement (negative = faster).
            uncertainty: Sigma of upgrade effectiveness (default from config).
        """
        unc = uncertainty or self.config.upgrade_uncertainty
        state = self._states[constructor_ref]

        # The upgrade shifts our prior mean estimate
        x_new = state["x"] + estimated_gain
        # Uncertainty temporarily increases (upgrade effect not fully known)
        P_new = state["P"] + unc**2

        self._states[constructor_ref] = {"x": x_new, "P": P_new}
        self._history[constructor_ref].append({
            "race_id": race_id, "event": "upgrade",
            "gain": estimated_gain, "uncertainty": unc,
            "x_after": x_new, "P_after": P_new
        })

    def get_estimate(self, constructor_ref: str,
                     race_id: Optional[int] = None) -> MachinePaceEstimate:
        """
        Retrieve current pace estimate for a constructor.

        Args:
            constructor_ref: Constructor reference string.
            race_id: For record-keeping only (doesn't filter history).

        Returns:
            MachinePaceEstimate with current mu and sigma.
        """
        state = self._states[constructor_ref]
        return MachinePaceEstimate(
            constructor_ref=constructor_ref,
            race_id=race_id or -1,
            mu_pace=state["x"],
            sigma_pace=np.sqrt(state["P"]),
            kalman_gain=0.0
        )

    def compute_observed_pace(self, lap_data: list[LapData],
                               constructor_drivers: list[str]) -> Optional[float]:
        """
        Compute observed pace for a constructor from lap telemetry.

        Method: Take the median fuel-corrected lap time for the constructor's
        drivers (clean air laps only), normalize against field median.

        Args:
            lap_data: All lap data for the race.
            constructor_drivers: Driver codes belonging to this constructor.

        Returns:
            Relative pace in seconds/lap (negative = faster), or None if
            insufficient data.
        """
        # Filter to valid laps, excluding pit laps
        valid_laps = [
            lap for lap in lap_data
            if lap.is_valid and not lap.pit_in_lap and not lap.pit_out_lap
            and lap.tyre_life > 2  # exclude outlaps
        ]

        if len(valid_laps) < 10:
            return None

        # Compute fuel-corrected times
        def corrected(lap: LapData) -> float:
            return lap.fuel_correction() if lap.fuel_corrected_ms == 0 else lap.fuel_corrected_ms

        field_times = [corrected(l) for l in valid_laps]
        field_median = np.median(field_times)

        constructor_laps = [l for l in valid_laps if l.driver_code in constructor_drivers]
        if len(constructor_laps) < 5:
            return None

        constructor_times = [corrected(l) for l in constructor_laps]
        constructor_median = np.median(constructor_times)

        # Pace relative to field, in seconds
        return (constructor_median - field_median) / 1000.0  # ms → s

    def get_pace_history(self, constructor_ref: str) -> list[dict]:
        """Return full state history for a constructor (for visualization)."""
        return self._history[constructor_ref].copy()

    def get_all_estimates(self) -> dict[str, MachinePaceEstimate]:
        """Return current estimates for all tracked constructors."""
        return {
            ref: self.get_estimate(ref)
            for ref in self._states
        }
