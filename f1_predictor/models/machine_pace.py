"""
models/machine_pace.py  (v2 — Multivariate Kalman Filter)
===========================================================
Layer 1b: Machine Pace Model via Multivariate Kalman Filter

Versione estesa rispetto a v1: il filtro scalare è sostituito da un
filtro con **vettore di stato 12-dimensionale** che apprende online
le sensibilità di ciascun costruttore alle caratteristiche del circuito.

Modello stato-spazio:

    State x_t in R^12  (vettore di coefficienti latenti)
    x[0]  = pace base (intercetta — pace media su tutti i circuiti)
    x[1]  = beta_top_speed          (top speed km/h, normalizzato)
    x[2]  = beta_min_speed          (min speed km/h, normalizzato)
    x[3]  = beta_avg_speed          (avg speed km/h, normalizzato)
    x[4]  = beta_slow_corners       (avg speed curve lente)
    x[5]  = beta_fast_corners       (avg speed curve veloci)
    x[6]  = beta_medium_corners     (avg speed curve medie)
    x[7]  = beta_full_throttle_pct  (% tempo a gas pieno)
    x[8]  = beta_street             (dummy: circuito cittadino)
    x[9]  = beta_high_df            (dummy: alto carico aero)
    x[10] = beta_high_speed         (dummy: basso carico / veloce)
    x[11] = beta_desert             (dummy: circuiti deserto)
    Baseline circuit type = MIXED (tutti i dummy = 0)

    Observation model:
        z_t = H_t @ x_t + v_t,   v_t ~ N(0, R)
        H_t = [1, feat_1_norm, ..., feat_7_norm, street, high_df, high_speed, desert]

    Transition model (random walk con process noise differenziato):
        x_t = x_t-1 + w_t,   w_t ~ N(0, Q)
        Q = diag([Q_base, Q_feat×7, Q_circuit_type×4])

    Kalman update:
        K_t = P @ H_t^T / (H_t @ P @ H_t^T + R)
        x_upd = x_pred + K_t * innovation
        P_upd = (I - K_t @ H_t^T) @ P_pred

Normalizzazione feature:
    Z-score online (Welford's algorithm) senza lookahead.
    Le statistiche vengono aggiornate DOPO la trasformazione,
    quindi ogni osservazione viene normalizzata con le statistiche
    delle sole osservazioni precedenti.

Riferimenti:
    - Welch & Bishop (2006) — "An Introduction to the Kalman Filter"
    - Heilmeier et al. (2020) — "Application of a Race Simulation for
      the Optimization of the Pit Stop Strategy"
    - arXiv 2512.00640 (2025) — State-space approach F1 tyre degradation
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

from f1_predictor.domain.entities import (
    MachinePaceEstimate, LapData, CircuitType
)

# ---------------------------------------------------------------------------
# Indici nel vettore di stato
# ---------------------------------------------------------------------------
IDX_BASE           = 0
IDX_TOP_SPEED      = 1
IDX_MIN_SPEED      = 2
IDX_AVG_SPEED      = 3
IDX_SLOW_CORNERS   = 4
IDX_FAST_CORNERS   = 5
IDX_MEDIUM_CORNERS = 6
IDX_THROTTLE_PCT   = 7
IDX_STREET         = 8
IDX_HIGH_DF        = 9
IDX_HIGH_SPEED     = 10
IDX_DESERT         = 11

STATE_DIM = 12

CONTINUOUS_FEATURE_NAMES = [
    "top_speed_kmh",
    "min_speed_kmh",
    "avg_speed_kmh",
    "slow_corners_kmh",
    "fast_corners_kmh",
    "medium_corners_kmh",
    "full_throttle_pct",
]
N_CONTINUOUS = len(CONTINUOUS_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# CircuitSpeedProfile
# ---------------------------------------------------------------------------

@dataclass
class CircuitSpeedProfile:
    """
    Profilo cinematico di un circuito per il Kalman Filter multivariato.

    Classificazione curve (Heilmeier et al. 2020 par.3.1):
        Lente  : v_corner < 120 km/h
        Medie  : 120 <= v_corner < 200 km/h
        Veloci : v_corner >= 200 km/h

    Tutti i valori di velocita sono in km/h.
    full_throttle_pct e la percentuale del giro trascorsa a gas pieno (0-100).
    """
    circuit_type: CircuitType
    top_speed_kmh: float
    min_speed_kmh: float
    avg_speed_kmh: float
    avg_slow_corner_kmh: float
    avg_fast_corner_kmh: float
    avg_medium_corner_kmh: float
    full_throttle_pct: float
    circuit_ref: str = ""
    source: str = "manual"

    def to_feature_vector(self) -> np.ndarray:
        return np.array([
            self.top_speed_kmh,
            self.min_speed_kmh,
            self.avg_speed_kmh,
            self.avg_slow_corner_kmh,
            self.avg_fast_corner_kmh,
            self.avg_medium_corner_kmh,
            self.full_throttle_pct,
        ], dtype=float)

    def circuit_type_dummies(self) -> np.ndarray:
        ct = self.circuit_type
        return np.array([
            1.0 if ct == CircuitType.STREET        else 0.0,
            1.0 if ct == CircuitType.HIGH_DOWNFORCE else 0.0,
            1.0 if ct == CircuitType.HIGH_SPEED     else 0.0,
            1.0 if ct == CircuitType.DESERT         else 0.0,
        ], dtype=float)


# ---------------------------------------------------------------------------
# Online standardizer (Welford, no lookahead)
# ---------------------------------------------------------------------------

class OnlineStandardizer:
    def __init__(self, n_features: int):
        self.n    = 0
        self.mean = np.zeros(n_features)
        self.M2   = np.ones(n_features)

    def update(self, x: np.ndarray):
        self.n += 1
        delta       = x - self.mean
        self.mean  += delta / self.n
        self.M2    += delta * (x - self.mean)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.n < 2:
            return np.zeros_like(x)
        std = np.sqrt(self.M2 / (self.n - 1))
        std = np.where(std < 1e-6, 1.0, std)
        return (x - self.mean) / std

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Trasforma con statistiche PRIMA di x, poi aggiorna (no lookahead)."""
        z = self.transform(x)
        self.update(x)
        return z


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KalmanConfig:
    """
    Iperparametri del KF multivariato.

    Q_base:         process noise intercetta (quanto cambia il pace base tra gare)
    Q_continuous:   process noise coefficienti feature continue (evoluzione lenta)
    Q_circuit_type: process noise dummy circuit_type (molto stabile)
    R:              observation noise variance (rumorosita della misura per-gara)
    P0_base:        incertezza iniziale intercetta
    P0_coef:        incertezza iniziale coefficienti
    upgrade_uncertainty: sigma default per inject_upgrade
    """
    Q_base:           float = 0.001
    Q_continuous:     float = 0.0003
    Q_circuit_type:   float = 0.0001
    R:                float = 0.004
    P0_base:          float = 0.10
    P0_coef:          float = 0.05
    upgrade_uncertainty: float = 0.05

    def Q_matrix(self) -> np.ndarray:
        q = np.zeros(STATE_DIM)
        q[IDX_BASE] = self.Q_base
        q[IDX_TOP_SPEED:IDX_THROTTLE_PCT + 1] = self.Q_continuous
        q[IDX_STREET:IDX_DESERT + 1] = self.Q_circuit_type
        return np.diag(q)

    def P0_matrix(self) -> np.ndarray:
        p = np.full(STATE_DIM, self.P0_coef)
        p[IDX_BASE] = self.P0_base
        return np.diag(p)


# ---------------------------------------------------------------------------
# MachinePaceModel
# ---------------------------------------------------------------------------

class MachinePaceModel:
    """
    Layer 1b: Constructor pace tracking via Multivariate Kalman Filter.

    Mantiene uno stato 12-dim per costruttore. A ogni gara aggiorna i
    coefficienti usando il vettore feature del circuito come H_t.

    Il modello apprende online, per ogni costruttore, DOVE e' forte/debole:
      - beta_top_speed < 0  => il costruttore guadagna sui rettilini
      - beta_slow_corners < 0 => vantaggioso nelle chicane/curve strette
      - beta_street < 0    => preferisce circuiti cittadini
      ecc.

    Usage:
        profile_monza = CircuitSpeedProfile(
            circuit_type=CircuitType.HIGH_SPEED,
            top_speed_kmh=352.0,
            min_speed_kmh=62.0,
            avg_speed_kmh=248.0,
            avg_slow_corner_kmh=88.0,
            avg_fast_corner_kmh=280.0,
            avg_medium_corner_kmh=165.0,
            full_throttle_pct=76.0,
            circuit_ref="monza",
        )
        model = MachinePaceModel()
        model.update("red_bull", race_id=1, observed_pace=-0.42, circuit=profile_monza)
        est = model.get_estimate("red_bull", circuit=profile_monza)
    """

    def __init__(self, config: Optional[KalmanConfig] = None):
        self.config = config or KalmanConfig()
        self._Q = self.config.Q_matrix()
        self._R = self.config.R
        self._states: dict[str, dict] = {}
        self._normalizers: dict[str, OnlineStandardizer] = {}
        self._history: dict[str, list] = defaultdict(list)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_init_state(self, ref: str) -> dict:
        if ref not in self._states:
            self._states[ref] = {
                "x": np.zeros(STATE_DIM),
                "P": self.config.P0_matrix().copy(),
            }
        return self._states[ref]

    def _get_or_init_normalizer(self, ref: str) -> OnlineStandardizer:
        if ref not in self._normalizers:
            self._normalizers[ref] = OnlineStandardizer(N_CONTINUOUS)
        return self._normalizers[ref]

    def _build_H(self, circuit: CircuitSpeedProfile, ref: str,
                 update: bool = False) -> np.ndarray:
        norm     = self._get_or_init_normalizer(ref)
        raw_feat = circuit.to_feature_vector()
        feat_norm = norm.fit_transform(raw_feat) if update else norm.transform(raw_feat)
        dummies  = circuit.circuit_type_dummies()

        H = np.zeros(STATE_DIM)
        H[IDX_BASE] = 1.0
        H[IDX_TOP_SPEED:IDX_THROTTLE_PCT + 1] = feat_norm
        H[IDX_STREET:IDX_DESERT + 1] = dummies
        return H

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        constructor_ref: str,
        race_id: int,
        observed_pace: float,
        circuit: Optional[CircuitSpeedProfile] = None,
    ) -> MachinePaceEstimate:
        """
        Aggiorna il filtro con una nuova osservazione di pace.

        Args:
            constructor_ref: es. 'red_bull', 'mercedes'
            race_id:         ID gara (per history)
            observed_pace:   pace relativo al campo (s/lap, negativo = piu veloce)
            circuit:         profilo cinematico del circuito (opzionale)

        Returns:
            MachinePaceEstimate dopo l'update.
        """
        if circuit is None:
            circuit = CircuitSpeedProfile(
                circuit_ref="generic",
                speed_factor=1.0,
                continuous_pct=0.5,
                throttle_pct=0.5,
                source="dummy",
            )

        state = self._get_or_init_state(constructor_ref)
        x, P  = state["x"], state["P"]

        H = self._build_H(circuit, constructor_ref, update=True)

        # Predict
        x_pred = x.copy()
        P_pred = P + self._Q

        # Update
        z_hat      = float(H @ x_pred)
        innovation = observed_pace - z_hat
        S          = float(H @ P_pred @ H) + self._R
        K          = (P_pred @ H) / S

        x_upd = x_pred + K * innovation
        P_upd = (np.eye(STATE_DIM) - np.outer(K, H)) @ P_pred
        P_upd = 0.5 * (P_upd + P_upd.T)   # symmetrize

        self._states[constructor_ref] = {"x": x_upd, "P": P_upd}

        mu_pace    = float(H @ x_upd)
        sigma_pace = float(np.sqrt(max(0.0, float(H @ P_upd @ H))))

        self._history[constructor_ref].append({
            "race_id":     race_id,
            "observed":    observed_pace,
            "predicted":   z_hat,
            "innovation":  innovation,
            "mu_pace":     mu_pace,
            "sigma_pace":  sigma_pace,
            "K_norm":      float(np.linalg.norm(K)),
            "circuit_type": circuit.circuit_type.value,
            "circuit_ref": circuit.circuit_ref,
        })

        return MachinePaceEstimate(
            constructor_ref=constructor_ref,
            race_id=race_id,
            mu_pace=mu_pace,
            sigma_pace=sigma_pace,
            kalman_gain=float(np.linalg.norm(K)),
        )

    def inject_upgrade(
        self,
        constructor_ref: str,
        race_id: int,
        estimated_gain: float,
        uncertainty: Optional[float] = None,
    ):
        """
        Inietta un pacchetto tecnico come shock sull'intercetta (pace base).
        L'upgrade migliora il pace indipendentemente dal circuito.
        """
        unc   = uncertainty or self.config.upgrade_uncertainty
        state = self._get_or_init_state(constructor_ref)
        state["x"][IDX_BASE]           += estimated_gain
        state["P"][IDX_BASE, IDX_BASE] += unc ** 2
        self._history[constructor_ref].append({
            "race_id": race_id, "event": "upgrade",
            "gain": estimated_gain, "unc": unc,
        })

    def get_estimate(
        self,
        constructor_ref: str,
        circuit: Optional[CircuitSpeedProfile] = None,
        race_id: Optional[int] = None,
    ) -> MachinePaceEstimate:
        """
        Stima di pace corrente per un costruttore su un circuito specifico.
        Se circuit e None, restituisce il pace base (intercetta).
        """
        state      = self._get_or_init_state(constructor_ref)
        x, P       = state["x"], state["P"]

        if circuit is not None:
            H          = self._build_H(circuit, constructor_ref, update=False)
            mu_pace    = float(H @ x)
            sigma_pace = float(np.sqrt(max(0.0, float(H @ P @ H))))
        else:
            mu_pace    = float(x[IDX_BASE])
            sigma_pace = float(np.sqrt(max(0.0, P[IDX_BASE, IDX_BASE])))

        return MachinePaceEstimate(
            constructor_ref=constructor_ref,
            race_id=race_id or -1,
            mu_pace=mu_pace,
            sigma_pace=sigma_pace,
            kalman_gain=0.0,
        )

    def get_coefficients(self, constructor_ref: str) -> dict:
        """
        Coefficienti appresi per un costruttore con incertezza.
        Utile per interpretare i punti di forza/debolezza.

        Returns:
            {feature_name: {"coef": float, "sigma": float}}
        """
        state  = self._get_or_init_state(constructor_ref)
        x, P   = state["x"], state["P"]
        sigmas = np.sqrt(np.diag(P))
        names  = (
            ["base_pace"]
            + CONTINUOUS_FEATURE_NAMES
            + ["circuit_street", "circuit_high_df",
               "circuit_high_speed", "circuit_desert"]
        )
        return {
            name: {"coef": float(x[i]), "sigma": float(sigmas[i])}
            for i, name in enumerate(names)
        }

    def compare_constructors(
        self,
        circuit: CircuitSpeedProfile,
        constructor_refs: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Classifica i costruttori per pace stimato su un dato circuito.
        """
        refs = constructor_refs or list(self._states.keys())
        rows = []
        for ref in refs:
            est = self.get_estimate(ref, circuit=circuit)
            rows.append({
                "constructor": ref,
                "mu_pace":    round(est.mu_pace, 4),
                "sigma_pace": round(est.sigma_pace, 4),
                "lower_95":   round(est.mu_pace - 2 * est.sigma_pace, 4),
                "upper_95":   round(est.mu_pace + 2 * est.sigma_pace, 4),
            })
        return sorted(rows, key=lambda r: r["mu_pace"])

    def compute_observed_pace(
        self,
        lap_data: list["LapData"],
        constructor_drivers: list[str],
    ) -> Optional[float]:
        """
        Calcola il pace relativo osservato dai dati lap.
        Normalizzato rispetto alla mediana del campo (s/lap).
        """
        valid_laps = [
            l for l in lap_data
            if l.is_valid and not l.pit_in_lap
            and not l.pit_out_lap and l.tyre_life > 2
        ]
        if len(valid_laps) < 10:
            return None

        def corrected(lap: "LapData") -> float:
            return lap.fuel_correction() if lap.fuel_corrected_ms == 0 else lap.fuel_corrected_ms

        field_median = np.median([corrected(l) for l in valid_laps])
        c_laps = [l for l in valid_laps if l.driver_code in constructor_drivers]
        if len(c_laps) < 5:
            return None

        constructor_median = np.median([corrected(l) for l in c_laps])
        return (constructor_median - field_median) / 1000.0

    def get_pace_history(self, constructor_ref: str) -> list[dict]:
        return self._history[constructor_ref].copy()

    def get_all_estimates(
        self,
        circuit: Optional[CircuitSpeedProfile] = None,
    ) -> dict[str, MachinePaceEstimate]:
        return {
            ref: self.get_estimate(ref, circuit=circuit)
            for ref in self._states
        }
