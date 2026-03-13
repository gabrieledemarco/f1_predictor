"""
data/__init__.py
================
Interfaccia unificata per tutti i loader dati.

Utilizzo tipico in train_pipeline.py:
    from data import load_training_data, load_calibration_records

    races = load_training_data(
        years=range(2019, 2027),
        through_round=5,
        jolpica_cache="data/cache/jolpica",
        tracinginsights_dir="data/racedata",
    )

    cal_records = load_calibration_records(
        odds_dir="data/pinnacle_odds",
        model_predictions={...},
        outcomes={...},
    )
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)


def load_training_data(
    years: range | list[int] = range(2019, 2027),
    through_round: Optional[int] = None,
    jolpica_cache: str = "data/cache/jolpica",
    tracinginsights_dir: str = "data/racedata",
    force_refresh: bool = False,
    use_synthetic_fallback: bool = True,
) -> list[dict]:
    """
    Carica i dati di training dalla pipeline completa.

    Tenta:
        1. JolpicaLoader → struttura gara, risultati, qualifiche
        2. TracingInsightsLoader → constructor_pace_observations
        3. Se nessuna fonte disponibile e use_synthetic_fallback=True → dati sintetici

    Args:
        years: Range di stagioni da caricare.
        through_round: Tronca l'ultima stagione a questo round.
        jolpica_cache: Directory cache JSON Jolpica.
        tracinginsights_dir: Directory CSV TracingInsights clonato.
        force_refresh: Ignora cache e ri-fetcha tutto.
        use_synthetic_fallback: Se True, usa dati sintetici quando reali non disponibili.

    Returns:
        Lista di race dict pronti per F1PredictionPipeline.fit().
    """
    from .loader_jolpica import JolpicaLoader
    from .loader_tracinginsights import TracingInsightsLoader
    from pathlib import Path

    jolpica_cache_path = Path(jolpica_cache)
    tracing_path = Path(tracinginsights_dir)

    # Check se c'è qualcosa in cache o sul disco
    has_jolpica_cache = (jolpica_cache_path.exists() and
                         any(jolpica_cache_path.glob("*.json")))
    has_tracinginsights = (tracing_path.exists() and
                           any(tracing_path.iterdir()))

    if not has_jolpica_cache and not has_tracinginsights:
        if use_synthetic_fallback:
            log.warning(
                "[DataLoader] Nessun dato reale trovato. "
                "Uso dati SINTETICI per test/sviluppo. "
                "Per training reale: "
                "(1) avvia in ambiente con rete per fetch Jolpica, oppure "
                "(2) clona TracingInsights: git clone "
                "https://github.com/TracingInsights/RaceData.git data/racedata"
            )
            return _load_synthetic_fallback(years, through_round)
        else:
            raise RuntimeError(
                "Nessun dato disponibile. "
                "Imposta use_synthetic_fallback=True o fornisci i dati reali."
            )

    # Carica da Jolpica (cache o API)
    log.info(f"[DataLoader] Loading {list(years)} via Jolpica...")
    jolpica = JolpicaLoader(
        cache_dir=jolpica_cache,
        force_refresh=force_refresh,
    )

    try:
        races = jolpica.load_seasons(years, through_round=through_round)
    except Exception as exc:
        log.warning(f"[DataLoader] Jolpica fetch fallito: {exc}")
        if use_synthetic_fallback:
            return _load_synthetic_fallback(years, through_round)
        raise

    if not races:
        if use_synthetic_fallback:
            log.warning("[DataLoader] Jolpica ha restituito 0 gare → fallback sintetico")
            return _load_synthetic_fallback(years, through_round)
        return []

    # Arricchisci con dati pace se disponibili (Kaggle o TracingInsights)
    kaggle_loader = None
    if has_tracinginsights:
        # Prima prova Kaggle (CSV flat)
        try:
            from .loader_kaggle import create_kaggle_loader
            kaggle_data_dir = tracing_path / "data"
            kaggle_loader = create_kaggle_loader(data_dir=str(kaggle_data_dir))
        except ImportError:
            kaggle_loader = None
        
        if kaggle_loader:
            log.info(f"[DataLoader] Enriching with Kaggle pace data...")
            races = kaggle_loader.enrich_races(races, years=list(years))
        else:
            # Fallback a TracingInsights (struttura directory)
            log.info(f"[DataLoader] Enriching with TracingInsights pace data...")
            ti = TracingInsightsLoader(data_dir=tracinginsights_dir)
            races = ti.enrich_races(races, years=list(years))
    else:
        log.info("[DataLoader] Nessun dato pace disponibile — pace obs={}. "
                 "Layer 1b userà media di campo come prior.")

    enriched_count = sum(1 for r in races if r.get("constructor_pace_observations"))
    log.info(
        f"[DataLoader] Pronte {len(races)} gare "
        f"({enriched_count} con pace data) per training"
    )
    return races


def load_calibration_records(
    odds_dir: str = "data/pinnacle_odds",
    model_predictions: Optional[dict] = None,
    outcomes: Optional[dict] = None,
) -> list[dict]:
    """
    Carica CalibrationRecord per il Layer 4 Isotonic.

    Args:
        odds_dir: Directory con i file JSONL delle quote Pinnacle salvate.
        model_predictions: {f"{race_id}_{driver_code}": p_raw} — output modello.
        outcomes: {f"{race_id}_{driver_code}": 1|0} — risultati reali.

    Returns:
        Lista CalibrationRecord dict. Vuota se odds non disponibili.
    """
    from .loader_odds import OddsLoader
    from pathlib import Path

    if not Path(odds_dir).exists():
        log.info(f"[DataLoader] Directory quote non trovata: {odds_dir}")
        return []

    loader = OddsLoader(cache_dir=odds_dir)
    odds_records = loader.load_saved_records(odds_dir)

    if not odds_records:
        log.info("[DataLoader] Nessun OddsRecord trovato — Layer 4 non calibrato")
        return []

    if model_predictions and outcomes:
        cal_records = loader.build_calibration_records(
            odds_records, model_predictions, outcomes
        )
        log.info(f"[DataLoader] {len(cal_records)} CalibrationRecord pronti")
        return cal_records

    log.info(f"[DataLoader] {len(odds_records)} OddsRecord trovati (senza match modello)")
    return odds_records


# ---------------------------------------------------------------------------
# Fallback sintetico
# ---------------------------------------------------------------------------

def _load_synthetic_fallback(years, through_round) -> list[dict]:
    """
    Genera dati sintetici come fallback quando i dati reali non sono disponibili.
    Usa data/adapter.py che wrappa il generator esistente.
    """
    try:
        from .adapter import generate_seasons
        years_list = list(years)
        races = generate_seasons(years=years_list, through_round=through_round)
        log.info(f"[DataLoader] Generati {len(races)} race sintetiche per {years_list}")
        return races
    except Exception as exc:
        log.warning(f"[DataLoader] Fallback sintetico fallito: {exc}")
        return []
