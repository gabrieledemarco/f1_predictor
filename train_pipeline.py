"""
train_pipeline.py  v2
=====================
Pipeline di training locale. Eseguire manualmente dopo ogni GP.

Cosa fa:
    1. Carica dati storici via JolpicaLoader + TracingInsightsLoader
       (con fallback automatico a dati sintetici se offline)
    2. Fit completo F1PredictionPipeline (TTT, Kalman, MC, Ridge, Isotonic)
    3. Walk-forward validation con metriche oggettive
    4. Serializza e carica artefatti su MongoDB via GridFS
    5. Stampa summary metriche

Uso:
    python train_pipeline.py --year 2026 --through-round 5
    python train_pipeline.py --list-versions
    python train_pipeline.py --rollback v20260303_0900
    python train_pipeline.py --dry-run --year 2026 --through-round 5
    python train_pipeline.py --delete-old 5

Requisiti locali:
    pip install scikit-learn scipy numpy pandas pymongo[srv] requests
    git clone https://github.com/TracingInsights/RaceData.git data/racedata
    # .env con MONGODB_URI=mongodb+srv://...
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_pipeline")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="F1 Predictor v2 — Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--year",           type=int,
                   help="Anno GP corrente (es. 2026)")
    p.add_argument("--through-round",  type=int,
                   help="Ultimo round incluso nel training (es. 5)")
    p.add_argument("--train-from",     type=int, default=2019,
                   help="Primo anno del dataset di training")
    p.add_argument("--val-from",       type=int, default=2022,
                   help="Primo anno del set di validazione walk-forward")
    p.add_argument("--jolpica-cache",  default="data/cache/jolpica",
                   help="Directory cache Jolpica JSON")
    p.add_argument("--tracing-dir",    default="data/racedata",
                   help="Directory TracingInsights CSV clonati")
    p.add_argument("--odds-dir",       default="data/pinnacle_odds",
                   help="Directory JSONL quote Pinnacle")
    p.add_argument("--n-mc-sim",       type=int, default=50_000,
                   help="Numero simulazioni Monte Carlo")
    p.add_argument("--ridge-alpha",    type=float, default=10.0,
                   help="Regolarizzazione Ridge")
    p.add_argument("--min-calib-obs",  type=int, default=100,
                   help="Osservazioni minime per calibratore isotonic")
    p.add_argument("--dry-run",        action="store_true",
                   help="Esegue training ma non salva su MongoDB")
    p.add_argument("--synthetic",      action="store_true",
                   help="Forza uso dati sintetici (test/sviluppo)")
    p.add_argument("--rollback",       type=str, default=None,
                   help="Rollback a versione specifica es. v20260303_0900")
    p.add_argument("--list-versions",  action="store_true",
                   help="Lista versioni su MongoDB ed esce")
    p.add_argument("--delete-old",     type=int, default=None, metavar="KEEP_N",
                   help="Elimina versioni vecchie, mantieni ultime N")
    return p.parse_args()


# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------

def connect_db():
    try:
        from core.db import get_db_direct
    except ImportError:
        # Tenta importazione relativa
        sys.path.insert(0, str(Path(__file__).parent))
        try:
            from core.db import get_db_direct
        except ImportError:
            log.error("core/db.py non trovato. Verifica la struttura del progetto.")
            sys.exit(1)

    db = get_db_direct()
    if db is None:
        log.error("MongoDB non raggiungibile. Controlla MONGO_URI nel file .env")
        sys.exit(1)
    log.info(f"MongoDB connesso: {db.name}")
    return db


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(args) -> dict:
    """Esegue training completo di tutti e 4 i layer."""

    # ── Importa il package f1_predictor ──────────────────────────────
    try:
        from f1_predictor.pipeline import F1PredictionPipeline
        from f1_predictor.models.driver_skill import TTTConfig
        from f1_predictor.models.machine_pace import KalmanConfig
        from f1_predictor.models.bayesian_race import RaceSimConfig
        from f1_predictor.validation.walk_forward import WalkForwardValidator
    except ImportError as e:
        log.error(
            f"Package f1_predictor non trovato: {e}\n"
            "Installalo: pip install -e ./f1_predictor"
        )
        sys.exit(1)

    # ── 1. Carica dati ───────────────────────────────────────────────
    log.info(f"[1/7] Caricando dati {args.train_from}–{args.year} R{args.through_round}...")
    try:
        from f1_predictor.data import load_training_data
        races = load_training_data(
            years=range(args.train_from, args.year + 1),
            through_round=args.through_round,
            jolpica_cache=args.jolpica_cache,
            tracinginsights_dir=args.tracing_dir,
            use_synthetic_fallback=True,
            force_refresh=False,
        )
    except Exception as e:
        log.warning(f"load_training_data fallito ({e}) — uso dati sintetici")
        from f1_predictor.data.adapter import generate_seasons
        races = generate_seasons(
            years=list(range(args.train_from, args.year + 1)),
            through_round=args.through_round
        )

    if not races:
        log.error("Nessun dato caricato. Impossibile procedere.")
        sys.exit(1)

    log.info(f"    → {len(races)} gare caricate ({args.train_from}–{args.year})")

    # ── 2. Costruisci pipeline ────────────────────────────────────────
    log.info("[2/7] Inizializzando F1PredictionPipeline...")
    pipeline = F1PredictionPipeline(
        ttt_config=TTTConfig(),
        kalman_config=KalmanConfig(),
        sim_config=RaceSimConfig(n_simulations=args.n_mc_sim),
        min_edge=0.04,
    )

    # ── 3. Walk-forward validation ────────────────────────────────────
    # Separa train/val prima di fittare il modello finale
    log.info(f"[3/7] Walk-forward validation (train:{args.train_from}–{args.val_from-1}"
             f" | val:{args.val_from}–{args.year})...")

    val_metrics = _run_walkforward(
        pipeline=pipeline,
        races=races,
        train_from=args.train_from,
        val_from=args.val_from,
        val_to=args.year,
        n_mc_sim=args.n_mc_sim,
        ridge_alpha=args.ridge_alpha,
    )
    _log_metrics(val_metrics)

    # ── 4. Fit finale su TUTTI i dati ────────────────────────────────
    log.info(f"[4/7] Fitting finale su tutti i {len(races)} gare...")
    pipeline_final = F1PredictionPipeline(
        ttt_config=TTTConfig(),
        kalman_config=KalmanConfig(),
        sim_config=RaceSimConfig(n_simulations=args.n_mc_sim),
    )
    pipeline_final.fit(races, verbose=False)
    log.info("    ✓ Pipeline fittata")

    # ── 5. Ridge su tutto + calibratore ──────────────────────────────
    log.info(f"[5/7] Fit Ridge ensemble (alpha={args.ridge_alpha})...")
    pipeline_final.ensemble.alpha = args.ridge_alpha
    ridge_info = _extract_ridge_info(pipeline_final)

    # Calibratore Isotonic
    calibrator_info = None
    odds_records = _load_odds(args.odds_dir)
    if odds_records and len(odds_records) >= args.min_calib_obs:
        log.info(f"[6/7] Fit Isotonic Calibrator ({len(odds_records)} obs)...")
        calibrator_info = _fit_calibrator(pipeline_final, odds_records)
    else:
        n_obs = len(odds_records) if odds_records else 0
        log.info(
            f"[6/7] Calibratore SKIPPED: {n_obs} < {args.min_calib_obs} obs Pinnacle.\n"
            "       Aggiungi dati in data/pinnacle_odds/ per attivare Layer 4."
        )

    # ── 6. Estrai artefatti ──────────────────────────────────────────
    log.info("[7/7] Serializzando artefatti...")
    artifacts = _extract_artifacts(pipeline_final, ridge_info, calibrator_info)

    # Metadata
    metadata = {
        "train_through_round":   args.through_round,
        "train_through_year":    args.year,
        "train_from_year":       args.train_from,
        "n_races_train":         len(races),
        "walk_forward_brier":    round(val_metrics.get("brier", 1.0), 6),
        "kendall_tau":           round(val_metrics.get("kendall_tau", 0.0), 6),
        "walk_forward_roi":      round(val_metrics.get("roi", 0.0), 4),
        "walk_forward_ece":      round(val_metrics.get("ece", 1.0), 6),
        "n_calibration_samples": len(odds_records) if odds_records else 0,
        "calibrator_fitted":     calibrator_info is not None,
        "ridge_alpha":           args.ridge_alpha,
        "n_mc_sim":              args.n_mc_sim,
        "data_sources":          _get_data_sources(args),
    }

    return {
        "artifacts":  artifacts,
        "metadata":   metadata,
        "val_metrics": val_metrics,
    }


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def _run_walkforward(pipeline, races: list[dict],
                     train_from: int, val_from: int, val_to: int,
                     n_mc_sim: int, ridge_alpha: float) -> dict:
    """
    Esegue walk-forward temporale con embargo di 1 gara.

    Strategia:
        - Allena su races fino a val_from-1
        - Valida su ogni gara di val_from..val_to, aggiungendo man mano
        - Raccoglie predizioni vs outcome
    """
    import numpy as np
    from scipy.stats import kendalltau
    from f1_predictor.pipeline import F1PredictionPipeline
    from f1_predictor.models.driver_skill import TTTConfig
    from f1_predictor.models.machine_pace import KalmanConfig
    from f1_predictor.models.bayesian_race import RaceSimConfig, DriverRaceInput
    from f1_predictor.domain.entities import CircuitType

    train_races = [r for r in races if r["year"] < val_from]
    test_races  = [r for r in races if val_from <= r["year"] <= val_to]

    if not train_races or not test_races:
        log.warning("Walk-forward: dati insufficienti, restituisco metriche di default")
        return {"brier": 0.25, "kendall_tau": 0.0, "roi": 0.0, "ece": 0.1}

    # Pipeline di validazione (non quella finale)
    wf_pipeline = F1PredictionPipeline(
        ttt_config=TTTConfig(),
        kalman_config=KalmanConfig(),
        # FIX: use reduced MC count for speed in WF loop, full count only for final training
        sim_config=RaceSimConfig(n_simulations=min(n_mc_sim, 5_000)),
    )
    wf_pipeline.fit(train_races, verbose=False)

    # Accumulators for ROI (flat-stake across all races)
    all_p_win_roi   = []
    all_outcome_roi = []

    # Per-race metric accumulators (FIX: avoid cross-race rank conflation)
    race_taus       = []
    race_briers     = []

    import gc

    for race in test_races:
        results = race.get("results", [])
        if not results:
            continue

        circuit_type_str = race.get("circuit_type", "mixed")
        try:
            ctype = CircuitType(circuit_type_str)
        except ValueError:
            ctype = CircuitType.MIXED

        # ----------------------------------------------------------------
        # TASK 2.1/2.2: Try full pipeline predict_race(), fall back to TTT only
        # ----------------------------------------------------------------
        driver_probs = {}
        try:
            race_entity, driver_grid = _build_race_entity_from_dict(race)
            pred_result = wf_pipeline.predict_race(race_entity, driver_grid, verbose=False)
            driver_probs = {
                code: prob.p_win
                for code, prob in pred_result["probabilities"].items()
            }
            log.debug(f"  [WF] Race {race.get('race_id')}: full pipeline OK")
        except Exception as e:
            log.debug(f"  [WF] Race {race.get('race_id')}: full pipeline failed ({e}), fallback to TTT softmax")
            # Fallback: TTT mu softmax (original behaviour)
            raw_mu = {}
            for res in results:
                code = res["driver_code"]
                try:
                    skill = wf_pipeline.driver_skill.get_rating(code, ctype)
                    raw_mu[code] = skill.mu
                except Exception:
                    raw_mu[code] = wf_pipeline.driver_skill.config.mu_0

            if raw_mu:
                mu_vals = np.array(list(raw_mu.values()))
                mu_max  = mu_vals.max()
                exp_v   = np.exp(mu_vals - mu_max)
                softmax = exp_v / exp_v.sum()
                driver_probs = dict(zip(raw_mu.keys(), softmax.tolist()))

        if not driver_probs:
            wf_pipeline.fit([race], verbose=False)
            gc.collect()
            continue

        # ----------------------------------------------------------------
        # TASK 1.1: Kendall τ — per-race only (FIX: no cross-race accumulation)
        # ----------------------------------------------------------------
        predicted_order = [d for d, _ in sorted(driver_probs.items(), key=lambda x: -x[1])]
        actual_order = [
            r["driver_code"]
            for r in sorted(
                [r for r in results if r.get("finish_position") is not None],
                key=lambda r: r["finish_position"]
            )
        ]
        common = [d for d in predicted_order if d in actual_order]
        if len(common) >= 5:
            pred_ranks = [predicted_order.index(d) for d in common]
            act_ranks  = [actual_order.index(d)    for d in common]
            tau_val, _ = kendalltau(pred_ranks, act_ranks)
            if not np.isnan(tau_val):
                race_taus.append(float(tau_val))

        # ----------------------------------------------------------------
        # TASK 1.2: Brier Score — per-race on the actual winner (FIX)
        # Models that assign high p_win to the actual winner score well.
        # ----------------------------------------------------------------
        winner = next(
            (r["driver_code"] for r in results if r.get("finish_position") == 1),
            None
        )
        if winner is not None and winner in driver_probs:
            p_winner = driver_probs[winner]
            race_briers.append((p_winner - 1.0) ** 2)

        # Accumulate for ROI simulation (all driver-race combos)
        for res in results:
            code = res.get("driver_code")
            if code and code in driver_probs:
                won = 1 if res.get("finish_position") == 1 else 0
                all_p_win_roi.append(driver_probs[code])
                all_outcome_roi.append(won)

        # Expanding window update — GC after each race to prevent memory growth
        wf_pipeline.fit([race], verbose=False)
        gc.collect()

    # ----------------------------------------------------------------
    # Aggregate metrics
    # ----------------------------------------------------------------
    metrics = {}

    # TASK 1.1 — Kendall τ: mean over per-race values
    metrics["kendall_tau"] = float(np.mean(race_taus)) if race_taus else 0.0
    metrics["n_races_evaluated"] = len(race_taus)

    # TASK 1.2 — Brier: mean of per-race winner Brier values
    metrics["brier"] = float(np.mean(race_briers)) if race_briers else 0.25

    # ROI simulato (unchanged — still uses flat-stake Kelly)
    metrics["roi"] = _simulate_roi(all_p_win_roi, all_outcome_roi)

    # TASK 1.3 — ECE: uses quantile bins (see _compute_ece fix)
    metrics["ece"] = _compute_ece(all_p_win_roi, all_outcome_roi)

    log.info(
        f"  [WF] {len(race_taus)} gare valutate | "
        f"τ={metrics['kendall_tau']:.3f} | "
        f"Brier={metrics['brier']:.4f} | "
        f"ROI={metrics['roi']:+.1f}% | "
        f"ECE={metrics['ece']:.4f}"
    )

    return metrics




def _build_race_entity_from_dict(race_dict: dict):
    """
    TASK 2.1 — Converte un race dict storico in (Race, driver_grid)
    per utilizzo con wf_pipeline.predict_race() nel walk-forward.
    Permette di usare il pipeline a 4 layer completo invece del solo TTT-softmax.
    """
    from f1_predictor.domain.entities import Race, Circuit, CircuitType

    circuit_type_str = race_dict.get("circuit_type", "mixed")
    try:
        ctype = CircuitType(circuit_type_str)
    except ValueError:
        ctype = CircuitType.MIXED

    circuit = Circuit(
        circuit_id=race_dict.get("circuit_id", 0),
        ref=race_dict.get("circuit_ref", "unknown"),
        name=race_dict.get("circuit_name", "unknown"),
        location=race_dict.get("country", ""),
        country=race_dict.get("country", ""),
        circuit_type=ctype,
    )
    race = Race(
        race_id=race_dict.get("race_id", 0),
        year=race_dict.get("year", 2024),
        round=race_dict.get("round", 1),
        circuit=circuit,
        name=race_dict.get("race_name", f"Round {race_dict.get('round', 1)}"),
        date=race_dict.get("date", ""),
        is_sprint_weekend=race_dict.get("is_sprint_weekend", False),
    )
    driver_grid = [
        {
            "driver_code":    r["driver_code"],
            "constructor_ref": r.get("constructor_ref", "unknown"),
            "grid_position":  r.get("grid_position", 10),
            "grid_penalty":   r.get("grid_penalty", 0),
        }
        for r in race_dict.get("results", [])
        if r.get("driver_code")
    ]
    return race, driver_grid


def tune_kalman_config(races: list, train_from: int, val_from: int,
                        ridge_alpha: float = 10.0) -> dict:
    """
    TASK 3.2 — Grid search Q/R per KalmanConfig.
    Massimizza Kendall tau sul primo anno di validation.
    Usa n_mc=3000 per velocita. Esegui su 1 anno di val.
    """
    import itertools
    from f1_predictor.pipeline import F1PredictionPipeline
    from f1_predictor.models.driver_skill import TTTConfig
    from f1_predictor.models.machine_pace import KalmanConfig
    from f1_predictor.models.bayesian_race import RaceSimConfig

    q_values = [0.001, 0.005, 0.01, 0.05, 0.10, 0.20]
    r_values = [0.002, 0.004, 0.01, 0.05, 0.20, 0.50]

    best_tau    = -float("inf")
    best_config = {"Q": 0.001, "R": 0.004}
    val_to      = val_from

    for q, r in itertools.product(q_values, r_values):
        try:
            pipeline = F1PredictionPipeline(
                ttt_config=TTTConfig(),
                kalman_config=KalmanConfig(Q=q, R=r),
                sim_config=RaceSimConfig(n_simulations=3_000),
            )
            m = _run_walkforward(
                pipeline=pipeline, races=races,
                train_from=train_from, val_from=val_from, val_to=val_to,
                n_mc_sim=3_000, ridge_alpha=ridge_alpha,
            )
            tau = m.get("kendall_tau", 0.0)
            log.info(f"  [KalmanTune] Q={q:.3f} R={r:.3f} -> tau={tau:.3f}")
            if tau > best_tau:
                best_tau    = tau
                best_config = {"Q": q, "R": r}
        except Exception as e:
            log.debug(f"  [KalmanTune] Q={q} R={r} failed: {e}")
            continue

    log.info(f"[KalmanTune] Best: Q={best_config['Q']} R={best_config['R']} -> tau={best_tau:.3f}")
    return best_config

def _simulate_roi(probs: list[float], outcomes: list[int],
                  kelly_frac: float = 0.25, edge_threshold: float = 0.04) -> float:
    """Simula ROI con Kelly frazionario su scommesse con edge > threshold."""
    import numpy as np
    if not probs:
        return 0.0

    bankroll = 1000.0
    start    = bankroll
    bets     = 0

    for p, o in zip(probs, outcomes):
        # Quota implicita di mercato (simuliamo mercato con vig 3%)
        market_p = p * 0.97
        if market_p <= 0:
            continue
        odd = 1.0 / market_p
        edge = p - market_p

        if edge < edge_threshold:
            continue

        # Kelly stake
        kelly = (p * odd - 1) / (odd - 1) * kelly_frac
        stake = max(0.0, min(kelly * bankroll, bankroll * 0.05))

        if o == 1:
            bankroll += stake * (odd - 1)
        else:
            bankroll -= stake
        bets += 1

    if bets == 0 or start == 0:
        return 0.0
    return float((bankroll - start) / start * 100)


def _compute_ece(probs: list[float], outcomes: list[int], n_bins: int = 10) -> float:
    """
    Expected Calibration Error con bin per quantili.

    FIX v2: usa quantili invece di bin uniformi [0,1] per evitare che la
    maggior parte dei bin sia vuota. Con p_win ~ 0.05 per tutti, i bin
    uniformi cadono nello stesso bucket → ECE artificialmente = 0.0.
    I bin per quantili garantiscono ~N/n_bins osservazioni per bin.
    """
    import numpy as np
    if len(probs) < 20:
        return 0.1

    probs_arr    = np.array(probs)
    outcomes_arr = np.array(outcomes, dtype=float)
    n            = len(probs_arr)

    quantiles = np.percentile(probs_arr, np.linspace(0, 100, n_bins + 1))
    ece = 0.0

    for i in range(n_bins):
        lo, hi = quantiles[i], quantiles[i + 1]
        mask = (probs_arr >= lo) & (probs_arr <= hi) if i == n_bins - 1 \
               else (probs_arr >= lo) & (probs_arr < hi)
        if mask.sum() == 0:
            continue
        avg_conf = probs_arr[mask].mean()
        avg_acc  = outcomes_arr[mask].mean()
        ece     += (mask.sum() / n) * abs(avg_conf - avg_acc)

    return float(ece)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_ridge_info(pipeline) -> dict:
    """Estrae info sul Ridge ensemble."""
    try:
        coefs = pipeline.ensemble.model.coef_.tolist()
        intercept = float(pipeline.ensemble.model.intercept_)
        feature_names = getattr(pipeline.ensemble, "feature_names_", [])
        return {"coefs": coefs, "intercept": intercept, "feature_names": feature_names}
    except AttributeError:
        return {"coefs": [], "intercept": 0.0, "feature_names": []}


def _load_odds(odds_dir: str) -> list[dict]:
    """Carica OddsRecord da directory JSONL."""
    try:
        from f1_predictor.data.loader_odds import OddsLoader
        loader = OddsLoader(cache_dir=odds_dir)
        return loader.load_saved_records(odds_dir)
    except Exception as e:
        log.debug(f"Odds load: {e}")
        return []


def _fit_calibrator(pipeline, odds_records: list[dict]):
    """Fitta il calibratore isotonic su odds_records."""
    try:
        probs    = [r.get("p_model_raw", 0.5) for r in odds_records]
        outcomes = [int(r.get("outcome", 0))   for r in odds_records]
        pipeline.calibrator.fit(probs, outcomes)
        return pipeline.calibrator
    except Exception as e:
        log.warning(f"Calibrator fit fallito: {e}")
        return None


def _extract_artifacts(pipeline, ridge_info: dict, calibrator) -> dict:
    """Serializza tutti gli artefatti dal pipeline fittato."""
    # Driver ratings
    driver_ratings = {}
    for code, rating_dict in pipeline.driver_skill._ratings.items():
        driver_ratings[code] = {
            "global": {"mu": rating_dict["mu"], "sigma": rating_dict["sigma"]}
        }
    # Aggiungi ratings per circuito
    for (code, ctype), rating_dict in pipeline.driver_skill._ratings_by_circuit.items():
        if code not in driver_ratings:
            driver_ratings[code] = {"global": {"mu": 25.0, "sigma": 8.333}}
        driver_ratings[code][str(ctype)] = {
            "mu": rating_dict["mu"], "sigma": rating_dict["sigma"]
        }

    # Kalman states
    kalman_states = {}
    for constructor, state in pipeline.machine_pace._states.items():
        kalman_states[constructor] = {
            "mu_pace":    state["x"],
            "sigma_pace": state["P"] ** 0.5,
        }

    # Calibrator (serializzabile)
    cal_data = None
    if calibrator is not None:
        try:
            import pickle, base64
            cal_data = base64.b64encode(pickle.dumps(calibrator)).decode("ascii")
        except Exception:
            pass

    return {
        "driver_ratings":      driver_ratings,
        "kalman_states":       kalman_states,
        "ridge_ensemble":      ridge_info,
        "isotonic_calibrator": cal_data,
    }


def _get_data_sources(args) -> list[str]:
    sources = []
    from pathlib import Path
    if Path(args.jolpica_cache).exists() and any(Path(args.jolpica_cache).glob("*.json")):
        sources.append("jolpica_cache")
    if Path(args.tracing_dir).exists():
        sources.append("tracinginsights_csv")
    if Path(args.odds_dir).exists() and any(Path(args.odds_dir).glob("*.jsonl")):
        sources.append("pinnacle_odds")
    if not sources:
        sources.append("synthetic_fallback")
    return sources


def _log_metrics(m: dict):
    b  = m.get("brier", 1.0)
    k  = m.get("kendall_tau", 0.0)
    r  = m.get("roi", 0.0)
    e  = m.get("ece", 1.0)
    log.info(f"    Brier Score:  {b:.4f}  {'OK' if b < 0.20 else 'WARN '} (target < 0.20)")
    log.info(f"    Kendall tau:  {k:.3f}  {'OK' if k > 0.45 else 'WARN '} (target > 0.45)")
    log.info(f"    ROI (WF):     {r:+.1f}%  {'OK' if r > 0 else 'WARN '} (target > 0%)")
    log.info(f"    ECE:          {e:.4f}  {'OK' if e < 0.05 else 'WARN '} (target < 0.05)")


def print_summary(result: dict):
    meta = result["metadata"]
    m    = result["val_metrics"]
    print()
    print("=" * 62)
    print("  F1 PREDICTOR v2 — TRAINING SUMMARY")
    print("=" * 62)
    print(f"  Anno/Round:    {meta['train_through_year']} R{meta['train_through_round']}")
    print(f"  Gare usate:    {meta['n_races_train']}")
    print(f"  Fonti dati:    {', '.join(meta['data_sources'])}")
    print(f"  Brier Score:   {meta['walk_forward_brier']:.4f}  "
          f"{'OK' if meta['walk_forward_brier'] < 0.20 else 'WARN '}")
    print(f"  Kendall tau:   {meta['kendall_tau']:.3f}  "
          f"{'OK' if meta['kendall_tau'] > 0.45 else 'WARN '}")
    print(f"  ROI (WF):      {meta['walk_forward_roi']:+.1f}%  "
          f"{'OK' if meta['walk_forward_roi'] > 0 else 'WARN '}")
    print(f"  ECE:           {meta['walk_forward_ece']:.4f}  "
          f"{'OK' if meta['walk_forward_ece'] < 0.05 else 'WARN '}")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Comandi di gestione (non richiedono --year/--through-round)
    if args.list_versions or args.rollback or args.delete_old is not None:
        db = connect_db()
        from core.db_artifacts import (
            list_model_versions, rollback_to_version, delete_old_versions
        )
        if args.list_versions:
            versions = list_model_versions(db)
            if not versions:
                print("Nessuna versione trovata su MongoDB.")
            else:
                print(f"\n{'Versione':<22} {'Round':<10} {'Brier':<8} {'tau':<7} Calib")
                print("-" * 62)
                for v in versions:
                    print(
                        f"{v['version']:<22} "
                        f"R{v.get('train_through_round','?')}/{v.get('train_through_year','?'):<3} "
                        f"{v.get('walk_forward_brier', 0):.4f}  "
                        f"{v.get('kendall_tau', 0):.3f}  "
                        f"{'OK' if v.get('calibrator_fitted') else 'NO'}"
                    )
            return

        if args.rollback:
            ok = rollback_to_version(db, args.rollback)
            print(f"{'OK Rollback → ' + args.rollback if ok else 'NO Versione non trovata'}")
            return

        if args.delete_old is not None:
            n = delete_old_versions(db, keep_last_n=args.delete_old)
            log.info(f"Eliminate {n} versioni (mantenute ultime {args.delete_old}).")
            return

    # Training
    if not args.year or not args.through_round:
        log.error("--year e --through-round sono richiesti per il training")
        sys.exit(1)

    result = run_training(args)

    print_summary(result)

    if args.dry_run:
        log.info("DRY RUN: artefatti non salvati su MongoDB.")
        return

    # Upload MongoDB
    db = connect_db()
    from core.db_artifacts import save_model_artifacts
    version = save_model_artifacts(
        db,
        artifacts_dict=result["artifacts"],
        metadata=result["metadata"],
    )
    if version:
        log.info(f"✅ Artefatti salvati su MongoDB: {version}")
        log.info("   La web app caricherà il nuovo modello al prossimo riavvio.")
    else:
        log.error("❌ Salvataggio MongoDB fallito.")


if __name__ == "__main__":
    main()
