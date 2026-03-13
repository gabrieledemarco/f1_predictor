"""
domain/entities.py
==================
Pure domain entities — zero external dependencies.
These are the canonical data structures that flow through the entire system.

Design principle (van Kesteren & Bergkamp, 2023):
    Separate domain objects from statistical models to allow swapping
    model implementations without changing the data contract.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TyreCompound(Enum):
    SOFT = "S"
    MEDIUM = "M"
    HARD = "H"
    INTERMEDIATE = "I"
    WET = "W"


class SessionType(Enum):
    FP1 = "FP1"
    FP2 = "FP2"
    FP3 = "FP3"
    QUALIFYING = "Q"
    SPRINT_QUALIFYING = "SQ"
    SPRINT = "SR"
    RACE = "R"


class CircuitType(Enum):
    """
    Circuit clustering from Heilmeier et al. (2020):
    5 clusters based on speed profile, downforce requirement,
    and historical tyre degradation patterns.
    """
    STREET = "street"           # Monaco, Baku, Singapore, Las Vegas
    HIGH_DOWNFORCE = "high_df"  # Hungary, Spain, China
    HIGH_SPEED = "high_speed"   # Monza, Spa (partially)
    MIXED = "mixed"             # Silverstone, Suzuka, Austin
    DESERT = "desert"           # Bahrain, Abu Dhabi, Saudi


# ---------------------------------------------------------------------------
# Core domain entities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Driver:
    """
    Canonical driver identity.
    The 'code' is the 3-letter FIA code (e.g. 'VER', 'HAM').
    """
    driver_id: int          # Ergast/Jolpica numeric ID
    code: str               # 3-letter FIA code
    forename: str
    surname: str
    nationality: str
    dob: Optional[str] = None

    @property
    def full_name(self) -> str:
        return f"{self.forename} {self.surname}"

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True)
class Constructor:
    """Formula 1 constructor (team)."""
    constructor_id: int
    ref: str                # e.g. 'red_bull', 'mercedes'
    name: str
    nationality: str


@dataclass(frozen=True)
class Circuit:
    """
    Circuit with physical characteristics used for clustering
    (Heilmeier et al., 2020 — Table 1: circuit feature vector).
    """
    circuit_id: int
    ref: str                        # e.g. 'bahrain'
    name: str
    location: str
    country: str
    circuit_type: CircuitType
    lap_length_km: float = 0.0
    full_throttle_pct: float = 0.0  # % of lap at full throttle
    avg_speed_kmh: float = 0.0


@dataclass(frozen=True)
class Race:
    """Single Grand Prix event."""
    race_id: int
    year: int
    round: int
    circuit: Circuit
    name: str
    date: str                       # ISO 8601
    is_sprint_weekend: bool = False


@dataclass
class LapData:
    """
    Lap-level telemetry record derived from FastF1.
    Used for Machine Pace Model (Layer 1b) and tire degradation
    modelling (arXiv 2512.00640, 2025).
    """
    race_id: int
    driver_code: str
    lap_number: int
    lap_time_ms: float
    sector1_ms: float
    sector2_ms: float
    sector3_ms: float
    compound: TyreCompound
    tyre_life: int                  # laps on current set
    fuel_corrected_ms: float = 0.0  # see TracingInsights fuel correction formula
    is_valid: bool = True
    pit_in_lap: bool = False
    pit_out_lap: bool = False

    def fuel_correction(self, initial_fuel_kg: float = 100.0,
                        total_laps: int = 57) -> float:
        """
        Apply fuel weight correction to raw lap time.
        Source: TracingInsights (2025) documentation:
            Corrected = Original - (RemainingFuel * 0.03)
            RemainingFuel = InitialFuel * (1 - lap/total_laps)
        Returns corrected lap time in ms.
        """
        remaining_fuel = initial_fuel_kg * (1 - self.lap_number / total_laps)
        correction_s = remaining_fuel * 0.03
        return self.lap_time_ms - (correction_s * 1000)


@dataclass
class RaceResult:
    """Official race result for one driver."""
    race_id: int
    driver_code: str
    constructor_ref: str
    grid_position: int
    finish_position: Optional[int]   # None = DNF
    points: float
    laps_completed: int
    status: str                      # 'Finished', '+1 Lap', 'Engine', etc.
    fastest_lap_rank: Optional[int] = None


@dataclass
class DriverSkillRating:
    """
    Output of Layer 1a (TrueSkill Through Time).
    Represents a driver's latent skill as a Gaussian distribution.

    Reference: Dangauthier et al. (NeurIPS) — TTT model:
        skill(t) ~ N(mu, sigma^2)
        skill evolves as random walk between events
    """
    driver_code: str
    circuit_type: Optional[CircuitType]   # None = global rating
    mu: float                             # skill mean
    sigma: float                          # skill uncertainty
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def conservative_rating(self) -> float:
        """
        Conservative skill estimate: mu - 3*sigma.
        Used by TrueSkill as the display rating to avoid
        overconfident comparisons (Herbrich et al., 2007).
        """
        return self.mu - 3 * self.sigma

    @property
    def variance(self) -> float:
        return self.sigma ** 2


@dataclass
class MachinePaceEstimate:
    """
    Output of Layer 1b (Kalman Filter on machine pace).
    Tracks the team's underlying car performance as a
    latent state that evolves with technical updates.

    Reference: Standard Kalman Filter for time-series
    pace tracking with update shocks at known upgrade points.
    """
    constructor_ref: str
    race_id: int
    mu_pace: float          # mean pace advantage (seconds per lap vs field median)
    sigma_pace: float       # uncertainty on pace estimate
    kalman_gain: float      # last Kalman gain (diagnostic)


@dataclass
class RaceProbability:
    """
    Output of the full 4-layer model for a single driver in a race.
    The probability vector covers positions 1..20.
    """
    race_id: int
    driver_code: str
    p_win: float
    p_podium: float
    p_top6: float
    p_points: float             # top 10
    p_dnf: float
    position_distribution: list  # list of 20 floats, P(pos=k) for k=1..20
    model_version: str = "1.0"
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def expected_position(self) -> float:
        """E[position] = sum(k * P(pos=k))"""
        return sum((k + 1) * p for k, p in enumerate(self.position_distribution))


@dataclass
class OddsRecord:
    """
    A single odds observation from Pinnacle (or other bookmaker).
    Stored with full timestamp for time-series analysis of market movement.
    """
    race_id: int
    driver_code: str
    market: str                 # 'winner', 'podium', 'h2h_VER_PER', etc.
    odd_decimal: float
    bookmaker: str              # 'pinnacle', 'betfair_exchange'
    timestamp: datetime
    hours_to_race: float        # negative = post-race

    @property
    def implied_prob_raw(self) -> float:
        """Raw implied probability (includes vig)."""
        return 1.0 / self.odd_decimal

    def net_return(self, stake: float = 1.0) -> float:
        """Net return on a winning bet with given stake."""
        return stake * (self.odd_decimal - 1)


@dataclass
class CalibrationRecord:
    """
    One calibration observation: model prediction vs market vs outcome.
    These records are the training data for Layer 4 (Isotonic Regression).

    Reference: Walsh & Joshi (2023) — calibration dataset structure.
    """
    race_id: int
    driver_code: str
    market: str
    p_model_raw: float          # raw model output (pre-calibration)
    p_model_calibrated: float   # post-isotonic calibration
    p_pinnacle_novig: float     # Pinnacle devigged probability
    edge: float                 # p_model_calibrated - p_pinnacle_novig
    outcome: int                # 1 = bet won, 0 = bet lost
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def edge_bucket(self) -> str:
        if self.edge > 0.08:  return "large_positive"
        if self.edge > 0.04:  return "small_positive"
        if self.edge > -0.04: return "neutral"
        if self.edge > -0.08: return "small_negative"
        return "large_negative"


@dataclass
class BetRecord:
    """
    One placed bet with full tracking data for bankroll management
    and post-hoc performance analysis.

    Reference: Baker & McHale (2013) — Kelly shrinkage framework.
    """
    race_id: int
    driver_code: str
    market: str
    edge: float
    kelly_fraction: float       # recommended fraction of bankroll
    actual_stake: float         # actual amount staked (Kelly * bankroll * shrinkage)
    odd_decimal: float
    outcome: int                # 1 = win, 0 = loss
    pnl: float                  # profit/loss in stake units
    bankroll_before: float
    bankroll_after: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def roi(self) -> float:
        return self.pnl / self.actual_stake if self.actual_stake > 0 else 0.0
