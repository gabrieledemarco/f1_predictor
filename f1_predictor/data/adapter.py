"""
data/adapter.py
===============
Converte vari formati sorgente nel formato dict atteso da
F1PredictionPipeline.fit().

Formato target per ogni gara:
    {
      "race_id"    : int,          # anno*1000 + round
      "year"       : int,
      "round"      : int,
      "circuit_ref": str,
      "circuit_type": str,         # "street"|"high_df"|"high_speed"|"mixed"|"desert"
      "race_name"  : str,
      "date"       : str,
      "is_season_end"              : bool,
      "is_major_regulation_change" : bool,
      "results": [
          {
            "driver_code"    : str,    # 3 lettere
            "constructor_ref": str,
            "grid_position"  : int,
            "finish_position": int|None,  # None = DNF
            "points"         : float,
            "laps_completed" : int,
            "status"         : str,
          },
          ...
      ],
      "qualifying": [...],
      "constructor_pace_observations": {constructor_ref: float}  # pace delta
    }
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Mapping circuito sintetico (nome stringa) → circuit_type
SYNTHETIC_CIRCUIT_TYPE: dict[str, str] = {
    "Monaco":       "street",
    "Baku":         "street",
    "Singapore":    "street",
    "Melbourne":    "street",
    "Miami":        "street",
    "Las Vegas":    "street",
    "Jeddah":       "street",
    "Monza":        "high_speed",
    "Spa":          "high_speed",
    "Silverstone":  "high_speed",
    "Suzuka":       "high_speed",
    "Hungaroring":  "high_df",
    "Barcelona":    "high_df",
    "Imola":        "high_df",
    "Abu Dhabi":    "high_df",
    "Zandvoort":    "high_df",
    "Shanghai":     "high_df",
    "Bahrain":      "desert",
    "Lusail":       "desert",
    "Austin":       "mixed",
    "Mexico":       "mixed",
    "Interlagos":   "mixed",
    "Canada":       "mixed",
}

# Punti F1 standard per posizione
F1_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
             6: 8,  7: 6,  8: 4,  9: 2,  10: 1}

# Stagioni con cambio regolamentare maggiore
MAJOR_REG_CHANGE_YEARS = {2009, 2014, 2017, 2022}


def dataframe_to_race_dicts(df) -> list[dict]:
    """
    Converte il DataFrame del generator sintetico (generator.py) nel
    formato dict atteso da F1PredictionPipeline.fit().

    Colonne DataFrame attese:
        season, round, circuit, driver, team, position, retired, grid

    Args:
        df: pandas DataFrame dal generator.

    Returns:
        Lista ordinata di race dict (per season, round).
    """
    import pandas as pd

    races = []
    grouped = df.groupby(["season", "round"])

    for (season, round_num), group in grouped:
        season, round_num = int(season), int(round_num)
        circuit = str(group["circuit"].iloc[0])
        circuit_ref = circuit.lower().replace(" ", "_")
        circuit_type = SYNTHETIC_CIRCUIT_TYPE.get(circuit, "mixed")

        # Determina ultimo round stagione
        max_round = int(df[df["season"] == season]["round"].max())
        is_season_end = (round_num == max_round)

        results = []
        for _, row in group.iterrows():
            retired  = bool(row.get("retired", False))
            pos_raw  = row.get("position", None)
            try:
                finish_pos = None if retired or pos_raw is None else int(pos_raw)
            except (ValueError, TypeError):
                finish_pos = None

            grid = int(row.get("grid", 10))
            grid = max(1, min(grid, 20))

            driver_raw = str(row.get("driver", "UNK"))
            # Normalizza a 3 lettere
            driver_code = driver_raw[:3].upper()

            team_raw = str(row.get("team", "unknown"))
            constructor_ref = team_raw.lower().replace(" ", "_")

            points = F1_POINTS.get(finish_pos, 0.0) if finish_pos else 0.0

            results.append({
                "driver_code":     driver_code,
                "constructor_ref": constructor_ref,
                "grid_position":   grid,
                "finish_position": finish_pos,
                "points":          points,
                "laps_completed":  55 if not retired else np.random.randint(1, 55),
                "status":          "DNF" if retired else "Finished",
            })

        races.append({
            "race_id":   season * 1000 + round_num,
            "year":      season,
            "round":     round_num,
            "circuit_ref": circuit_ref,
            "circuit_type": circuit_type,
            "race_name": circuit,
            "date":      f"{season}-01-01",
            "is_season_end": is_season_end,
            "is_major_regulation_change": (season in MAJOR_REG_CHANGE_YEARS and round_num == 1),
            "results":   results,
            "qualifying": [],
            "constructor_pace_observations": {},
        })

    races.sort(key=lambda r: (r["year"], r["round"]))
    return races


def generate_seasons(years: list[int],
                     through_round: Optional[int] = None) -> list[dict]:
    """
    Genera dati sintetici per le stagioni richieste e li converte
    nel formato pipeline dict.

    Compatibile con data/__init__._load_synthetic_fallback().

    Args:
        years: Lista anni da generare.
        through_round: Se specificato, tronca l'ultima stagione.

    Returns:
        Lista race dict pronti per F1PredictionPipeline.fit().
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    try:
        from f1_predictor.data.generator import generate_season, load_historical_data
        import pandas as pd

        frames = []
        for year in years:
            rng = np.random.default_rng(seed=year * 42)
            df  = generate_season(year, rng)
            if through_round and year == max(years):
                df = df[df["round"] <= through_round]
            frames.append(df)

        full_df = pd.concat(frames, ignore_index=True)
        races   = dataframe_to_race_dicts(full_df)

        log.info(f"[Adapter] Generated {len(races)} synthetic races for {years}")
        return races

    except ImportError:
        # Fallback: genera direttamente senza il package
        log.warning("[Adapter] f1_predictor.data.generator non trovato — generazione minimale")
        return _minimal_synthetic(years, through_round)


def _minimal_synthetic(years: list[int],
                        through_round: Optional[int] = None) -> list[dict]:
    """
    Generazione sintetica minimale senza dipendenze esterne.
    Usato come ultimo fallback quando nemmeno il generator è disponibile.
    """
    DRIVERS = ["VER", "HAM", "LEC", "SAI", "NOR", "PIA",
               "RUS", "ALO", "PER", "GAS", "OCO", "ALB",
               "HUL", "MAG", "BOT", "ZHO", "TSU", "LAW", "STR", "BEA"]
    CONSTRUCTORS = {
        "VER": "red_bull", "PER": "red_bull",
        "HAM": "mercedes", "RUS": "mercedes",
        "LEC": "ferrari",  "SAI": "ferrari",
        "NOR": "mclaren",  "PIA": "mclaren",
        "ALO": "aston_martin", "STR": "aston_martin",
        "GAS": "alpine",   "OCO": "alpine",
        "ALB": "williams", "LAW": "williams",
        "HUL": "haas",     "MAG": "haas",
        "BOT": "sauber",   "ZHO": "sauber",
        "TSU": "alphatauri","BEA": "alphatauri",
    }
    CIRCUITS = [
        ("bahrain", "desert"), ("jeddah", "street"), ("albert_park", "street"),
        ("suzuka", "mixed"), ("shanghai", "high_df"), ("miami", "street"),
        ("imola", "high_df"), ("monaco", "street"), ("villeneuve", "mixed"),
        ("catalunya", "high_df"), ("red_bull_ring", "high_speed"),
        ("silverstone", "high_speed"), ("hungaroring", "high_df"),
        ("spa", "high_speed"), ("zandvoort", "high_df"), ("monza", "high_speed"),
        ("baku", "street"), ("marina_bay", "street"), ("americas", "mixed"),
        ("rodriguez", "mixed"), ("interlagos", "mixed"), ("vegas", "street"),
        ("losail", "desert"), ("yas_marina", "desert"),
    ]

    rng = np.random.default_rng(42)
    races = []

    for year in years:
        n_rounds = len(CIRCUITS)
        last_round = through_round if (through_round and year == max(years)) else n_rounds

        for rnd in range(1, last_round + 1):
            circuit_ref, circuit_type = CIRCUITS[(rnd - 1) % len(CIRCUITS)]

            # Skill-based ordering con rumore
            order = rng.permutation(len(DRIVERS))
            results = []
            pos = 1
            for i, idx in enumerate(order):
                driver = DRIVERS[idx]
                retired = rng.random() < 0.05 and i > 10  # ~5% DNF, mostly backmarkers
                results.append({
                    "driver_code":     driver,
                    "constructor_ref": CONSTRUCTORS[driver],
                    "grid_position":   i + 1,
                    "finish_position": None if retired else pos,
                    "points":          F1_POINTS.get(pos, 0.0) if not retired else 0.0,
                    "laps_completed":  55 if not retired else int(rng.integers(1, 55)),
                    "status":          "DNF" if retired else "Finished",
                })
                if not retired:
                    pos += 1

            races.append({
                "race_id":   year * 1000 + rnd,
                "year":      year,
                "round":     rnd,
                "circuit_ref": circuit_ref,
                "circuit_type": circuit_type,
                "race_name": circuit_ref.replace("_", " ").title(),
                "date":      f"{year}-01-01",
                "is_season_end": (rnd == last_round),
                "is_major_regulation_change": (year in MAJOR_REG_CHANGE_YEARS and rnd == 1),
                "results":   results,
                "qualifying": [],
                "constructor_pace_observations": {},
            })

    log.info(f"[Adapter] Minimal synthetic: {len(races)} races for {years}")
    return races
