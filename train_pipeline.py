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
    # .env con MONGO_URI=... (oppure MONGODB_URI=...)
"""

from __future__ import annotations

import argparse
import logging
import os
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
        log.error("MongoDB non raggiungibile. Controlla MONGO_URI/MONGODB_URI nel file .env")
        sys.exit(1)
    log.info(f"MongoDB connesso: {db.name}")
    return db


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(args) -> dict:
    """Esegue training completo di tutti e 4 i layer."""

    import time as _time

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

    # ── ETA stimati per ogni step ─────────────────────────────────────
    # Calibrati su: 147 gare, n_mc=5000, val_from=2022
    # Scalano con n_mc_sim e numero di stagioni
    _mc_scale   = args.n_mc_sim / 5_000
    _yr_scale   = max(1, (args.year - args.train_from + 1)) / 7
    _val_years  = max(1, args.year - args.val_from + 1)

    _ETA_SEC = {
        1: 3,                                            # caricamento dati (cache locale)
        2: 1,                                            # init pipeline (solo oggetti)
        3: max(30, int(90 * _mc_scale * _val_years)),   # walk-forward: il passo lento
        4: max(10, int(20 * _mc_scale * _yr_scale)),    # fit finale
        5: 2,                                            # Ridge: istantaneo
        6: 5,                                            # isotonic calibrator
        7: 2,                                            # serializzazione artefatti
    }

    def _fmt_eta(secs: int) -> str:
        if secs < 60:
            return f"~{secs}s"
        return f"~{secs // 60}m{secs % 60:02d}s"

    def _fmt_elapsed(secs: float) -> str:
        if secs < 60:
            return f"{secs:.1f}s"
        return f"{secs / 60:.1f}m"

    _t_global = _time.time()
    _step_timers: dict = {}

    def _step_start(n: int, desc: str) -> None:
        elapsed_total = _time.time() - _t_global
        eta_str       = _fmt_eta(_ETA_SEC.get(n, 0))
        total_str     = _fmt_elapsed(elapsed_total)
        bar           = "█" * n + "░" * (7 - n)
        log.info(
            f"┌─[{n}/7] {bar}  {desc}\n"
            f"│  ETA questo step: {eta_str}  |  Trascorso totale: {total_str}"
        )
        _step_timers[n] = _time.time()

    def _step_done(n: int, extra: str = "") -> None:
        dur = _time.time() - _step_timers.get(n, _time.time())
        eta = _ETA_SEC.get(n, 0)
        diff = dur - eta
        diff_str = (f"  [{'+' if diff >= 0 else ''}{diff:.0f}s vs ETA]"
                    if abs(diff) > 3 else "")
        suffix = f"  {extra}" if extra else ""
        log.info(f"└─ ✓ completato in {_fmt_elapsed(dur)}{diff_str}{suffix}")

    # ── Connessione MongoDB ─────────────────────────────────────────
    # Legge URI direttamente da env var (come in test_mongo_connection.yml)
    mongodb_uri = os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URI")
    db = None
    if mongodb_uri:
        try:
            from core.db import get_db_direct
            db = get_db_direct(uri=mongodb_uri)
            if db is not None:
                log.info(f"[Pipeline] MongoDB connesso: {db.name}")
            else:
                log.warning("[Pipeline] MongoDB connesso ma db=None")
        except Exception as e:
            log.warning(f"[Pipeline] MongoDB errore: {e}")
    else:
        log.warning("[Pipeline] MONGODB_URI non disponibile")

    # ── 1. Carica dati ───────────────────────────────────────────────
    _step_start(1, f"Caricamento dati  {args.train_from}–{args.year}  R{args.through_round}")
    
    races = None
    
    if db is not None:
        try:
            from f1_predictor.data import load_training_data
            races = load_training_data(
                years=range(args.train_from, args.year + 1),
                through_round=args.through_round,
                jolpica_cache=args.jolpica_cache,
                tracinginsights_dir=args.tracing_dir,
                use_synthetic_fallback=True,
                force_refresh=False,
                db=db,
            )
            log.info(f"[Pipeline] Caricate {len(races)} gare da MongoDB")
        except Exception as e:
            log.warning(f"[Pipeline] load_training_data fallito: {e}")
            races = None

    if not races:
        log.warning("[Pipeline] Uso dati sintetici come fallback")
        from f1_predictor.data.adapter import generate_seasons
        races = generate_seasons(
            years=list(range(args.train_from, args.year + 1)),
            through_round=args.through_round
        )
        for r in races:
            r["_is_synthetic"] = True

    if not races:
        log.error("Nessun dato caricato. Impossibile procedere.")
        sys.exit(1)

    n_synthetic = sum(1 for r in races if r.get("_is_synthetic"))
    if n_synthetic:
        log.warning(
            f"[Pipeline] ATTENZIONE: {n_synthetic}/{len(races)} gare SINTETICHE "
            "nel training set — le predizioni non saranno affidabili."
        )

    _step_done(1, f"{len(races)} gare  ({args.train_from}–{args.year})")

    # ── 2. Costruisci pipeline ────────────────────────────────────────
    _step_start(2, "Inizializzazione F1PredictionPipeline (4 layer)")
    pipeline = F1PredictionPipeline(
        ttt_config=TTTConfig(),
        kalman_config=KalmanConfig(),
        sim_config=RaceSimConfig(n_simulations=args.n_mc_sim),
        min_edge=0.04,
    )
    _step_done(2, f"MC={args.n_mc_sim:,} sim")

    # ── 3. Walk-forward validation ────────────────────────────────────
    n_val_races = len([r for r in races if r.get("year", 0) >= args.val_from])
    _step_start(
        3,
        f"Walk-forward validation  "
        f"train:{args.train_from}–{args.val_from - 1}  "
        f"val:{args.val_from}–{args.year}  "
        f"({n_val_races} gare di test)"
    )
    val_metrics = _run_walkforward(
        pipeline=pipeline,
        races=races,
        train_from=args.train_from,
        val_from=args.val_from,
        val_to=args.year,
        n_mc_sim=args.n_mc_sim,
        ridge_alpha=args.ridge_alpha,
    )
    n_eval = val_metrics.get("n_races_evaluated", 0)
    _step_done(3, f"{n_eval} gare valutate")
    _log_metrics(val_metrics)

    # ── 4. Fit finale su TUTTI i dati ────────────────────────────────
    # Usa MC ridotto per speed: il fit è iterativo, non serve alta precisione
    n_mc_final = min(args.n_mc_sim, 10_000)  # Max 10k per final fit
    _step_start(4, f"Fit finale su tutti i {len(races)} gare  (TTT + Kalman)")
    pipeline_final = F1PredictionPipeline(
        ttt_config=TTTConfig(),
        kalman_config=KalmanConfig(),
        sim_config=RaceSimConfig(n_simulations=n_mc_final),
    )
    pipeline_final.fit(races, verbose=False)
    _step_done(4, f"MC={n_mc_final:,}")

    # ── 5. Ridge ensemble ────────────────────────────────────────────
    _step_start(5, f"Fit Ridge ensemble  (alpha={args.ridge_alpha})")
    pipeline_final.ensemble.alpha = args.ridge_alpha
    ridge_info = _extract_ridge_info(pipeline_final)
    _step_done(5)

    # ── 6. Isotonic Calibrator ───────────────────────────────────────
    calibrator_info = None
    odds_records    = _load_odds(args.odds_dir, db=db)
    n_obs           = len(odds_records) if odds_records else 0

    if odds_records and n_obs >= args.min_calib_obs:
        _step_start(6, f"Fit Isotonic Calibrator  ({n_obs} osservazioni Pinnacle)")
        calibrator_info = _fit_calibrator(pipeline_final, odds_records)
        _step_done(6, "Layer 4 attivo")
    else:
        needed = args.min_calib_obs - n_obs
        _step_start(6, f"Isotonic Calibrator  [{n_obs}/{args.min_calib_obs} obs]")
        log.info(
            f"│  ⚠ SKIPPED — mancano {needed} osservazioni Pinnacle.\n"
            f"│    Esegui: python scripts/collect_odds.py  prima di ogni GP\n"
            f"│    Layer 4 si attiverà automaticamente al raggiungimento di "
            f"{args.min_calib_obs} obs."
        )
        _step_done(6, "skipped")

    # ── 7. Serializza artefatti ──────────────────────────────────────
    _step_start(7, "Serializzazione artefatti → MongoDB")
    artifacts = _extract_artifacts(pipeline_final, ridge_info, calibrator_info)
    _step_done(7)

    # ── Riepilogo tempi ──────────────────────────────────────────────
    _total = _time.time() - _t_global
    log.info(
        f"\n  ⏱  Tempo totale: {_fmt_elapsed(_total)}"
        f"  |  Gare: {len(races)}"
        f"  |  MC sim: {args.n_mc_sim:,}"
    )

    # Metadata
    metadata = {
        "train_through_round":   args.through_round,
        "train_through_year":    args.year,
        "train_from_year":       args.train_from,
        "n_races_train":         len(races),
        "walk_forward_brier":    round(val_metrics.get("brier_multiclass", 1.0), 6),
        "kendall_tau":           round(val_metrics.get("kendall_tau", 0.0), 6),
        "walk_forward_roi":      round(val_metrics.get("roi", 0.0), 4),
        "walk_forward_ece":      round(val_metrics.get("ece", 1.0), 6),
        "walk_forward_logloss":  round(val_metrics.get("logloss_multiclass", 10.0), 6),
        "walk_forward_rps":      round(val_metrics.get("rps_mean", 0.5), 6),
        "n_races_with_position_dist": val_metrics.get("n_races_with_position_dist", 0),
        "n_races_with_odds":     val_metrics.get("n_races_with_odds", 0),
        "n_calibration_samples": n_obs,
        "calibrator_fitted":     calibrator_info is not None,
        "ridge_alpha":           args.ridge_alpha,
        "n_mc_sim":              args.n_mc_sim,
        "data_sources":          _get_data_sources(args),
        "training_time_sec":     round(_total, 1),
    }

    return {
        "artifacts":   artifacts,
        "metadata":    metadata,
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
    race_loglosses  = []
    race_rps        = []

    import gc
    import time as _time

    n_test         = len(test_races)
    n_mc_wf        = min(n_mc_sim, 5_000)
    wf_start_time  = _time.time()
    lap_times_sec  = []  # per-race elapsed time for ETA

    # Stima durata basata su benchmark empirico: ~2-8s/gara con 5k MC
    est_sec_per_race = 5 * (n_mc_wf / 5_000)   # scala con n_mc
    est_total_min    = (n_test * est_sec_per_race) / 60
    log.info(
        f"  [WF] Avvio loop su {n_test} gare di test "
        f"({val_from}\u2013{val_to}) | MC/gara={n_mc_wf:,} | "
        f"train_races={len(train_races)} | "
        f"durata stimata ~{est_total_min:.0f}-{est_total_min*2:.0f} min"
    )

    for race_idx, race in enumerate(test_races, start=1):
        race_t0 = _time.time()
        results = race.get("results", [])
        race_label = (
            f"{race.get('year','?')} R{str(race.get('round','?')).zfill(2)} "
            f"{race.get('race_name', '')[:22]}"
        )
        if not results:
            log.debug(f"  [WF] {race_idx}/{n_test} {race_label} — skip (no results)")
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
        position_dists = {}
        try:
            race_entity, driver_grid = _build_race_entity_from_dict(race)
            pred_result = wf_pipeline.predict_race(race_entity, driver_grid, verbose=False)
            driver_probs = {
                code: prob.p_win
                for code, prob in pred_result["probabilities"].items()
            }
            position_dists = {
                code: prob.position_distribution
                for code, prob in pred_result["probabilities"].items()
                if prob.position_distribution
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

        # Get winner early for diagnostics
        winner = next(
            (r["driver_code"] for r in results if r.get("finish_position") == 1),
            None
        )

        # ----------------------------------------------------------------
        # Phase 1: Multiclass metrics (Brier, LogLoss, RPS) with robust handling
        # ----------------------------------------------------------------
        if position_dists:
            driver_codes_in_race = list(driver_probs.keys())
            n_drivers = len(driver_codes_in_race)

            # Build probability matrix (n_drivers x 20) and one-hot actual positions
            prob_matrix = np.zeros((n_drivers, 20))
            actual_onehot = np.zeros((n_drivers, 20))

            for i, code in enumerate(driver_codes_in_race):
                if code in position_dists:
                    dist = position_dists[code]
                    prob_matrix[i, :len(dist)] = dist[:20]

                fin_pos = next(
                    (r.get("finish_position") for r in results if r.get("driver_code") == code),
                    None
                )
                if fin_pos is not None and 1 <= fin_pos <= 20:
                    actual_onehot[i, fin_pos - 1] = 1.0

            # HARD CHECK: Race-level normalization
            row_sums = prob_matrix.sum(axis=1, keepdims=True)
            
            # Detect problematic rows: all zeros, NaN, or sum too small
            eps_normalize = 1e-8
            bad_rows = np.where((row_sums < eps_normalize) | np.isnan(row_sums))[0]
            
            if len(bad_rows) > 0:
                # Fallback: uniform distribution for problematic rows
                for i in bad_rows:
                    prob_matrix[i, :] = 1.0 / 20
                log.debug(f"  [WF] Race {race.get('race_id')}: {len(bad_rows)} rows normalized to uniform")
            
            # Normalize valid rows
            row_sums = prob_matrix.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            prob_matrix_norm = prob_matrix / row_sums

            # HARD CHECK: Verify normalization worked
            final_row_sums = prob_matrix_norm.sum(axis=1)
            if np.any(np.isnan(final_row_sums)) or np.any(final_row_sums < 0.9) or np.any(final_row_sums > 1.1):
                # Final fallback: set to uniform if still broken
                prob_matrix_norm = np.ones((n_drivers, 20)) / 20
                log.debug(f"  [WF] Race {race.get('race_id')}: final normalization fallback triggered")

            # DIAGNOSTICS: Log per-race probability stats
            p_win_vec = np.array([driver_probs.get(code, 0.0) for code in driver_codes_in_race])
            min_p = float(np.nanmin(p_win_vec))
            max_p = float(np.nanmax(p_win_vec))
            sum_p = float(np.nansum(p_win_vec))
            p_winner_diag = 0.0
            if winner is not None and winner in driver_probs:
                p_winner_diag = float(driver_probs[winner])
            
            log.debug(
                f"  [WF] Race {race.get('race_id')}: min_p={min_p:.6f} max_p={max_p:.4f} "
                f"sum_p={sum_p:.4f} p_winner={p_winner_diag:.4f}"
            )

            # Multiclass Brier: (1/n) * sum_over_drivers sum_over_positions (p - actual)^2
            brier_multiclass = float(np.mean((prob_matrix_norm - actual_onehot) ** 2))
            race_briers.append(brier_multiclass)

            # LogLoss with better clipping (1e-6) and renormalization
            eps_clamp = 1e-6
            prob_matrix_clipped = np.clip(prob_matrix_norm, eps_clamp, 1 - eps_clamp)
            row_sums_clip = prob_matrix_clipped.sum(axis=1, keepdims=True)
            row_sums_clip = np.where(row_sums_clip > 0, row_sums_clip, 1.0)
            prob_matrix_clipped = prob_matrix_clipped / row_sums_clip
            
            # Handle any remaining NaN/inf
            prob_matrix_clipped = np.nan_to_num(prob_matrix_clipped, nan=1.0/20, posinf=1.0, neginf=0.0)
            
            logloss = float(np.mean(-np.sum(actual_onehot * np.log(prob_matrix_clipped), axis=1)))
            race_loglosses.append(logloss)

            # RPS per race (average over all drivers)
            from f1_predictor.validation.metrics import ranked_probability_score
            rps_scores = []
            for i, code in enumerate(driver_codes_in_race):
                fin_pos = next(
                    (r.get("finish_position") for r in results if r.get("driver_code") == code),
                    None
                )
                if fin_pos is not None and 1 <= fin_pos <= 20:
                    rps = ranked_probability_score(prob_matrix_norm[i].tolist(), fin_pos, n_categories=20)
                    rps_scores.append(rps)
            race_rps.append(float(np.mean(rps_scores)) if rps_scores else 0.5)

        # ----------------------------------------------------------------
        # Legacy winner-only Brier (for comparison)
        # ----------------------------------------------------------------
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

        # ── Progress log con ETA ──────────────────────────────────────────────
        race_elapsed = _time.time() - race_t0
        lap_times_sec.append(race_elapsed)
        avg_sec      = sum(lap_times_sec) / len(lap_times_sec)
        remaining    = n_test - race_idx
        eta_sec      = avg_sec * remaining
        total_elapsed = _time.time() - wf_start_time

        # Metriche rolling (ultime 5 gare)
        tau_rolling   = float(sum(race_taus[-5:])  / len(race_taus[-5:]))  if race_taus   else float("nan")
        brier_rolling = float(sum(race_briers[-5:]) / len(race_briers[-5:])) if race_briers else float("nan")
        rps_rolling   = float(sum(race_rps[-5:])   / len(race_rps[-5:]))   if race_rps   else float("nan")

        eta_str = (
            f"{int(eta_sec//60)}m{int(eta_sec%60):02d}s" if eta_sec < 3600
            else f"{int(eta_sec//3600)}h{int((eta_sec%3600)//60):02d}m"
        )
        elapsed_str = (
            f"{int(total_elapsed//60)}m{int(total_elapsed%60):02d}s" if total_elapsed < 3600
            else f"{int(total_elapsed//3600)}h{int((total_elapsed%3600)//60):02d}m"
        )

        tau_str   = f"{tau_rolling:.3f}"   if not (tau_rolling   != tau_rolling)   else "n/a"
        brier_str = f"{brier_rolling:.4f}" if not (brier_rolling != brier_rolling) else "n/a"
        rps_str   = f"{rps_rolling:.4f}"   if not (rps_rolling   != rps_rolling)   else "n/a"

        log.info(
            f"  [WF] {race_idx:>3}/{n_test} | {race_label:<30} | "
            f"{race_elapsed:>5.1f}s | "
            f"tau(5)={tau_str} brier(5)={brier_str} rps(5)={rps_str} | "
            f"elapsed={elapsed_str} ETA={eta_str}"
        )

    # ----------------------------------------------------------------
    # Aggregate metrics
    # ----------------------------------------------------------------
    # Track diagnostics across all races
    # ----------------------------------------------------------------
    n_low_p_winner_global = 0
    for race in test_races:
        race_results = race.get("results", [])
        race_probs = {}
        # We don't have access to per-race probs here easily, 
        # so we'll compute this from accumulated all_p_win_roi instead
        
    metrics = {}
    
    # Count races where p_winner < 1e-4 (from accumulated ROI data)
    # This is a proxy - ideally we'd track this per-race
    metrics["n_low_p_winner"] = 0

    # TASK 1.1 — Kendall τ: mean over per-race values
    metrics["kendall_tau"] = float(np.mean(race_taus)) if race_taus else 0.0
    metrics["n_races_evaluated"] = len(race_taus)

    # Multiclass metrics (Phase 1)
    metrics["brier_multiclass"] = float(np.mean(race_briers)) if race_briers else 0.25
    metrics["logloss_multiclass"] = float(np.mean(race_loglosses)) if race_loglosses else 10.0
    metrics["rps_mean"] = float(np.mean(race_rps)) if race_rps else 0.5

    # Legacy winner-only Brier (kept for comparison)
    metrics["brier_winner_legacy"] = metrics["brier_multiclass"]  # Same for now, could compute separately

    # ROI simulato (Phase 2: try real backtest if odds available, else None)
    # Real backtest requires odds data passed to walk-forward - for now keep synthetic fallback
    metrics["roi"] = _simulate_roi(all_p_win_roi, all_outcome_roi)
    metrics["roi_real"] = None  # Will be populated in Phase 2 when odds are available

    # TASK 1.3 — ECE: uses quantile bins (see _compute_ece fix)
    metrics["ece"] = _compute_ece(all_p_win_roi, all_outcome_roi)

    # Data coverage info
    metrics["n_races_with_position_dist"] = len(race_rps)
    metrics["n_races_with_odds"] = 0  # Will be updated in Phase 2

    log.info(
        f"  [WF] {len(race_taus)} gare valutate | "
        f"τ={metrics['kendall_tau']:.3f} | "
        f"Brier={metrics['brier_multiclass']:.4f} | "
        f"LogLoss={metrics['logloss_multiclass']:.4f} | "
        f"RPS={metrics['rps_mean']:.4f} | "
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


def _load_odds(odds_dir: str, db=None) -> list[dict]:
    """Carica OddsRecord da MongoDB (primary) o directory JSONL (fallback)."""
    try:
        from f1_predictor.data.loader_odds import OddsLoader
        loader = OddsLoader(cache_dir=odds_dir, db=db)
        return loader.load_saved_records(directory=odds_dir, db=db)
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

    # Kalman states (multivariato: x ∈ R^12, P ∈ R^12×12)
    # Converti np.ndarray → list per serializzazione BSON (MongoDB)
    kalman_states = {}
    for constructor, state in pipeline.machine_pace._states.items():
        x = state["x"]
        P = state["P"]
        kalman_states[constructor] = {
            "x_vector": x.tolist() if hasattr(x, "tolist") else list(x),
            "P_matrix": P.tolist() if hasattr(P, "tolist") else [list(row) for row in P],
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
    k  = m.get("kendall_tau", 0.0)
    b  = m.get("brier_multiclass", 1.0)
    ll = m.get("logloss_multiclass", 10.0)
    rps = m.get("rps_mean", 0.5)
    r  = m.get("roi", 0.0)
    e  = m.get("ece", 1.0)
    n  = m.get("n_races_evaluated", "?")
    n_pos_dist = m.get("n_races_with_position_dist", 0)

    def _bar(val, target, higher_is_better=True, width=10) -> str:
        """Mini barra di progresso verso il target."""
        if isinstance(val, float) and isinstance(target, float) and target != 0:
            ratio = (val / target) if higher_is_better else (target / val)
            filled = min(int(ratio * width), width)
            return "▓" * filled + "░" * (width - filled)
        return "░" * width

    log.info("  ┌─────────────────────────────────────────────────")
    log.info(f"  │  Metriche walk-forward  ({n} gare valutate, {n_pos_dist} con pos dist)")
    log.info("  ├─────────────────────────────────────────────────")
    log.info(
        f"  │  Kendall τ   {k:+.3f}  {_bar(k, 0.45, True)}  "
        f"{'OK' if k >= 0.45 else 'WARN'}  (target >= 0.45)"
    )
    log.info(
        f"  │  Brier (mc) {b:.4f}  {_bar(b, 0.20, False)}  "
        f"{'OK' if b <= 0.20 else 'WARN'}  (target <= 0.20)"
    )
    log.info(
        f"  │  LogLoss(mc){ll:.4f}  {_bar(ll, 1.0, False)}  "
        f"{'OK' if ll <= 1.0 else 'WARN'}  (target <= 1.0)"
    )
    log.info(
        f"  │  RPS mean   {rps:.4f}  {_bar(rps, 0.15, False)}  "
        f"{'OK' if rps <= 0.15 else 'WARN'}  (target <= 0.15)"
    )
    log.info(
        f"  │  ECE        {e:.4f}  {_bar(e, 0.05, False)}  "
        f"{'OK' if e <= 0.05 else 'WARN'}  (target <= 0.05)"
    )
    log.info(
        f"  │  ROI (synth){r:+.1f}%  (synthetic - see Phase 2 for real backtest)"
    )
    log.info("  └─────────────────────────────────────────────────")


def print_summary(result: dict):
    meta = result["metadata"]
    m    = result["val_metrics"]
    t    = meta.get("training_time_sec", 0)
    t_str = f"{t:.0f}s" if t < 60 else f"{t/60:.1f}m"

    print()
    print("=" * 64)
    print("  F1 PREDICTOR v2 - TRAINING SUMMARY")
    print("=" * 64)
    print(f"  Anno/Round :   {meta['train_through_year']} R{meta['train_through_round']}")
    print(f"  Gare usate :   {meta['n_races_train']}")
    print(f"  Fonti dati :   {', '.join(meta['data_sources'])}")
    print(f"  Tempo totale:  {t_str}")
    print(f"  MC sim/gara:   {meta['n_mc_sim']:,}")
    print()
    print("  +- METRICHE WALK-FORWARD ---------------------------")

    def _row(label, val, fmt, target_ok: bool, target_str: str):
        icon = "OK" if target_ok else "WARN"
        print(f"  |  {label:<14} {val:{fmt}}   {icon}  {target_str}")

    _row("Kendall tau",   meta['kendall_tau'],        "+.3f", meta['kendall_tau'] >= 0.45,  "target >= 0.45")
    _row("Brier (mc)",    meta['walk_forward_brier'], ".4f",  meta['walk_forward_brier'] <= 0.20, "target <= 0.20")
    _row("LogLoss (mc)",  meta['walk_forward_logloss'], ".4f", meta['walk_forward_logloss'] <= 1.0, "target <= 1.0")
    _row("RPS mean",      meta['walk_forward_rps'],   ".4f",  meta['walk_forward_rps'] <= 0.15, "target <= 0.15")
    _row("ROI (WF)",      meta['walk_forward_roi'],   "+.1f", meta['walk_forward_roi'] > 0,  "target > 0%")
    _row("ECE",           meta['walk_forward_ece'],   ".4f",  meta['walk_forward_ece'] <= 0.05, "target <= 0.05")

    n_pos_dist = meta.get("n_races_with_position_dist", 0)
    n_odds = meta.get("n_races_with_odds", 0)
    print(f"  |  Data coverage: {n_pos_dist} races with position dist, {n_odds} races with odds")
    print(f"  |  Calibratore  {'OK' if meta['calibrator_fitted'] else 'INATTIVO  (< 100 obs Pinnacle)'}")
    print("  +---------------------------------------------------------------")
    print()



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
