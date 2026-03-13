"""
pipeline.py
===========
F1 Prediction Pipeline — Full Integration of All 4 Layers

This is the top-level orchestrator that wires all components together
into a single coherent prediction workflow.

Pipeline flow:
    1. Data ingestion (TracingInsights RaceData + telemetry)
    2. Layer 1a: Driver Skill Rating (TTT update)
    3. Layer 1b: Machine Pace (Kalman Filter update)
    4. Layer 2: Bayesian Race Simulation (Monte Carlo)
    5. Layer 3: Ensemble adjustment (Ridge meta-learner)
    6. Layer 4: Calibration against Pinnacle (Isotonic)
    7. Edge computation and report generation

Usage:
    pipeline = F1PredictionPipeline()
    pipeline.fit(historical_data)   # train on past seasons
    report = pipeline.predict_race(race, driver_grid, pinnacle_odds)
    print(report.to_text())
"""

from __future__ import annotations
from typing import Optional

from f1_predictor.domain.entities import (
    Race, RaceProbability, OddsRecord, CalibrationRecord
)
from f1_predictor.models.driver_skill import DriverSkillModel, TTTConfig
from f1_predictor.models.machine_pace import MachinePaceModel, KalmanConfig
from f1_predictor.models.bayesian_race import BayesianRaceModel, RaceSimConfig, DriverRaceInput
from f1_predictor.models.ensemble import EnsembleModel, EnsembleFeatures
from f1_predictor.calibration.devig import devig_power
from f1_predictor.calibration.isotonic import PinnacleCalibrationLayer
from f1_predictor.calibration.edge_tracker import BetaBinomialEdgeTracker
from f1_predictor.validation.backtesting import BettingBacktester, BacktestConfig
from f1_predictor.reports.edge_report import EdgeReportGenerator


class F1PredictionPipeline:
    """
    Top-level orchestrator for the F1 prediction system.

    Manages the lifecycle of all 4 model layers and provides
    a clean interface for training, prediction, and reporting.
    """

    def __init__(self,
                 ttt_config: Optional[TTTConfig] = None,
                 kalman_config: Optional[KalmanConfig] = None,
                 sim_config: Optional[RaceSimConfig] = None,
                 backtest_config: Optional[BacktestConfig] = None,
                 min_edge: float = 0.04):

        # Layer 1a
        self.driver_skill = DriverSkillModel(config=ttt_config)
        # Layer 1b
        self.machine_pace = MachinePaceModel(config=kalman_config)
        # Layer 2
        self.race_sim = BayesianRaceModel(config=sim_config)
        # Layer 3
        self.ensemble = EnsembleModel(alpha=10.0)
        # Layer 4
        self.calibrator = PinnacleCalibrationLayer(min_samples=100)
        # Edge tracking
        self.edge_tracker = BetaBinomialEdgeTracker()
        # Reporting
        self.report_generator = EdgeReportGenerator(min_edge=min_edge)
        # Backtesting
        self.backtester = BettingBacktester(config=backtest_config)

        self._calibration_records: list[CalibrationRecord] = []
        self._is_fitted = False

    def fit(self, historical_races: list[dict],
            historical_odds: Optional[dict] = None,
            verbose: bool = True) -> "F1PredictionPipeline":
        """
        Train all layers on historical data.

        Args:
            historical_races: List of race dicts from TracingInsights/Ergast.
            historical_odds: Optional odds history for calibration layer.
            verbose: Print progress.

        Returns:
            self (fluent interface).
        """
        from f1_predictor.domain.entities import RaceResult

        if verbose:
            print(f"[Pipeline] Fitting on {len(historical_races)} races...")

        all_results = []
        race_meta = {}

        for race_dict in historical_races:
            race_id = race_dict.get("race_id")
            year = race_dict.get("year")
            circuit_type = race_dict.get("circuit_type")

            race_meta[race_id] = {"circuit_type": circuit_type, "year": year}

            # Build RaceResult objects for Layer 1a
            for r in race_dict.get("results", []):
                rr = RaceResult(
                    race_id=race_id,
                    driver_code=r["driver_code"],
                    constructor_ref=r.get("constructor_ref", "unknown"),
                    grid_position=r.get("grid_position", 10),
                    finish_position=r.get("finish_position"),
                    points=r.get("points", 0.0),
                    laps_completed=r.get("laps_completed", 0),
                    status=r.get("status", "Unknown"),
                )
                all_results.append(rr)

            # Update machine pace (Layer 1b)
            pace_obs = race_dict.get("constructor_pace_observations", {})
            for constructor_ref, observed_pace in pace_obs.items():
                self.machine_pace.update(constructor_ref, race_id, observed_pace)

            # Season boundary: apply decay
            if race_dict.get("is_season_end"):
                major_change = race_dict.get("is_major_regulation_change", False)
                self.driver_skill.apply_season_decay(major_change)
                if verbose:
                    print(f"  ↻ Season decay applied (major_change={major_change})")

        # Fit Layer 1a (TTT)
        if verbose:
            print(f"[Pipeline] Fitting TTT Driver Skill on {len(all_results)} results...")
        self.driver_skill.fit(all_results, race_meta)

        # Fit calibration layer (Layer 4) if odds available
        if historical_odds and len(historical_odds) > 0:
            cal_probs = []
            cal_outcomes = []
            for record in self._calibration_records:
                cal_probs.append(record.p_model_raw)
                cal_outcomes.append(record.outcome)
            if len(cal_probs) >= self.calibrator.min_samples:
                if verbose:
                    print(f"[Pipeline] Fitting Isotonic Calibrator on {len(cal_probs)} records...")
                self.calibrator.fit(cal_probs, cal_outcomes)

        self._is_fitted = True
        if verbose:
            print("[Pipeline] ✓ Fitting complete.")
        return self

    def predict_race(self, race: Race,
                      driver_grid: list[dict],
                      pinnacle_odds: Optional[dict] = None,
                      verbose: bool = False) -> dict:
        """
        Generate full predictions for an upcoming race.

        Args:
            race: Race object.
            driver_grid: List of dicts with driver/constructor info and grid positions.
            pinnacle_odds: Optional dict {driver_code: decimal_odd} for edge report.
            verbose: Print simulation progress.

        Returns:
            Dict with 'probabilities', 'edge_report' (if odds provided),
            and 'model_state' diagnostics.
        """
        # Build Layer 2 inputs
        driver_inputs = []
        for entry in driver_grid:
            code = entry["driver_code"]
            constructor = entry.get("constructor_ref", "unknown")

            skill = self.driver_skill.get_rating(code, race.circuit.circuit_type)
            pace = self.machine_pace.get_estimate(constructor, race.race_id)

            di = DriverRaceInput(
                driver_code=code,
                constructor_ref=constructor,
                grid_position=entry.get("grid_position", 10),
                skill_mu=skill.mu,
                skill_sigma=skill.sigma,
                pace_mu=pace.mu_pace,
                pace_sigma=pace.sigma_pace,
                circuit_type=race.circuit.circuit_type,
                grid_penalty=entry.get("grid_penalty", 0),
            )
            driver_inputs.append(di)

        # Layer 2: Monte Carlo simulation
        if verbose:
            print(f"[Pipeline] Running Monte Carlo ({self.race_sim.config.n_simulations:,} sims)...")
        mc_probs = self.race_sim.simulate_race(race, driver_inputs, verbose=verbose)

        # Layer 3: Ensemble adjustment (if fitted)
        features = {}
        for di in driver_inputs:
            code = di.driver_code
            mp = mc_probs[code]
            feat = EnsembleFeatures(
                driver_skill_mu=di.skill_mu,
                driver_skill_sigma=di.skill_sigma,
                driver_skill_conservative=di.skill_mu - 3 * di.skill_sigma,
                machine_pace_mu=di.pace_mu,
                machine_pace_sigma=di.pace_sigma,
                p_win_mc=mp.p_win,
                p_podium_mc=mp.p_podium,
                p_dnf_mc=mp.p_dnf,
                expected_position_mc=mp.expected_position(),
                grid_position=float(di.grid_position),
                grid_vs_quali_delta=0.0,
                circuit_type_encoded=list(race.circuit.circuit_type.__class__).index(
                    race.circuit.circuit_type
                ) if hasattr(race.circuit.circuit_type.__class__, '__iter__') else 0,
                has_grid_penalty=di.grid_penalty > 0,
            )
            features[code] = feat

        final_probs = self.ensemble.adjust_probabilities(mc_probs, features)

        # Layer 4: Calibration
        driver_codes = list(final_probs.keys())
        p_raw = [final_probs[c].p_win for c in driver_codes]
        p_cal = self.calibrator.transform(p_raw)

        for code, cal in zip(driver_codes, p_cal):
            fp = final_probs[code]
            final_probs[code] = RaceProbability(
                race_id=fp.race_id,
                driver_code=code,
                p_win=float(cal),
                p_podium=fp.p_podium,
                p_top6=fp.p_top6,
                p_points=fp.p_points,
                p_dnf=fp.p_dnf,
                position_distribution=fp.position_distribution,
                model_version="pipeline_v1"
            )

        # Edge report (if odds provided)
        edge_report = None
        if pinnacle_odds:
            edge_report = self.report_generator.generate(
                race_name=race.name,
                model_probs=final_probs,
                pinnacle_odds=pinnacle_odds,
                calibrator=self.calibrator,
            )

        return {
            "probabilities": final_probs,
            "edge_report": edge_report,
            "model_state": {
                "is_fitted": self._is_fitted,
                "calibrator_fitted": self.calibrator._is_fitted,
                "n_calibration_records": len(self._calibration_records),
            }
        }

    def update_with_race_result(self, race_id: int, results: list[dict],
                                  odds_used: Optional[dict] = None):
        """
        Update all models after a race completes.

        Call this after each race to keep the pipeline current.

        Args:
            race_id: Completed race ID.
            results: Actual race results [{driver_code, finish_position, ...}].
            odds_used: Pinnacle odds used for that race (for calibration update).
        """
        from f1_predictor.domain.entities import RaceResult

        # Update Layer 1a
        rr_list = [
            RaceResult(
                race_id=race_id,
                driver_code=r["driver_code"],
                constructor_ref=r.get("constructor_ref", "unknown"),
                grid_position=r.get("grid_position", 10),
                finish_position=r.get("finish_position"),
                points=r.get("points", 0.0),
                laps_completed=r.get("laps_completed", 0),
                status=r.get("status", "Unknown"),
            )
            for r in results
        ]
        self.driver_skill._process_race(
            race_id, rr_list, circuit_type=None, year=None
        )

    def get_edge_summary(self) -> list[dict]:
        """Return current edge summary across all markets."""
        return self.edge_tracker.summary_report()
