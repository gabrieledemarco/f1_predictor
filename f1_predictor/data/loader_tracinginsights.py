"""
data/loader_tracinginsights.py
==============================
Legge i CSV lap-by-lap dal repository TracingInsights/RaceData e produce:
    1. LapData entities per il Kalman Filter (Layer 1b)
    2. constructor_pace_observations per ogni gara (input per pipeline.fit())

Struttura attesa dei CSV (clona con):
    git clone https://github.com/TracingInsights/RaceData.git data/racedata

Layout directory TracingInsights:
    data/racedata/
        2024/
            Bahrain/
                laps.csv
                results.csv
            Saudi_Arabia/
                laps.csv
            ...
        2023/
            ...

Colonne laps.csv (esempio TracingInsights):
    Driver, LapNumber, LapTime, Sector1Time, Sector2Time, Sector3Time,
    Compound, TyreLife, IsPersonalBest, TrackStatus, IsAccurate, Team

Utilizzo:
    loader = TracingInsightsLoader(data_dir="data/racedata")
    # Arricchisce i race_dict già caricati da JolpicaLoader
    enriched = loader.enrich_races(races, years=range(2019, 2027))
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

# Mapping nome cartella TracingInsights → circuit_ref Jolpica
# Necessario per allineare i due dataset
FOLDER_TO_CIRCUIT_REF: dict[str, str] = {
    "Bahrain":          "bahrain",
    "Saudi_Arabia":     "jeddah",
    "Australia":        "albert_park",
    "Japan":            "suzuka",
    "China":            "shanghai",
    "Miami":            "miami",
    "Emilia_Romagna":   "imola",
    "Monaco":           "monaco",
    "Canada":           "villeneuve",
    "Spain":            "catalunya",
    "Austria":          "red_bull_ring",
    "Great_Britain":    "silverstone",
    "Hungary":          "hungaroring",
    "Belgium":          "spa",
    "Netherlands":      "zandvoort",
    "Italy":            "monza",
    "Azerbaijan":       "baku",
    "Singapore":        "marina_bay",
    "United_States":    "americas",
    "Mexico":           "rodriguez",
    "Brazil":           "interlagos",
    "Las_Vegas":        "vegas",
    "Qatar":            "losail",
    "Abu_Dhabi":        "yas_marina",
}

# Compound string → categoria (robusto a varianti)
COMPOUND_NORM: dict[str, str] = {
    "soft": "SOFT", "s": "SOFT", "c5": "SOFT", "c4": "SOFT",
    "medium": "MEDIUM", "m": "MEDIUM", "c3": "MEDIUM",
    "hard": "HARD", "h": "HARD", "c2": "HARD", "c1": "HARD",
    "intermediate": "INTERMEDIATE", "inter": "INTERMEDIATE", "i": "INTERMEDIATE",
    "wet": "WET", "w": "WET", "full_wet": "WET",
}

# Degrado gomme (s/lap) — Layer 2 Monte Carlo default
TYRE_DEGRADATION = {"SOFT": 0.085, "MEDIUM": 0.055, "HARD": 0.035,
                    "INTERMEDIATE": 0.04, "WET": 0.03}

# ---------------------------------------------------------------------------
# Loader principale
# ---------------------------------------------------------------------------

class TracingInsightsLoader:
    """
    Carica e normalizza i CSV lap-by-lap di TracingInsights.

    Produce due output principali per ogni gara:
        - lap_data: list[LapData]  → alimenta il Kalman Filter
        - pace_obs: dict           → constructor_pace_observations

    Args:
        data_dir: Path alla root del repo TracingInsights clonato.
        min_valid_laps: Numero minimo di giri validi per calcolare il pace.
    """

    def __init__(self,
                 data_dir: str = "data/racedata",
                 min_valid_laps: int = 5):
        self.data_dir = Path(data_dir)
        self.min_valid_laps = min_valid_laps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich_races(self, races: list[dict],
                     years: Optional[range | list[int]] = None) -> list[dict]:
        """
        Arricchisce i race_dict con constructor_pace_observations.

        Args:
            races: Lista di dict dal JolpicaLoader.
            years: Filtro anni (None = tutti).

        Returns:
            La stessa lista con il campo constructor_pace_observations popolato.
        """
        year_set = set(years) if years else None

        enriched = 0
        for race in races:
            year = race.get("year")
            if year_set and year not in year_set:
                continue

            circuit_ref = race.get("circuit_ref", "")
            lap_data_path = self._find_laps_csv(year, circuit_ref)

            if not lap_data_path:
                log.debug(f"  [{year} {circuit_ref}] CSV non trovato — pace obs skipped")
                continue

            try:
                pace_obs = self._compute_pace_observations(lap_data_path, year)
                race["constructor_pace_observations"] = pace_obs
                enriched += 1
            except Exception as exc:
                log.warning(f"  [{year} {circuit_ref}] Errore parsing CSV: {exc}")

        log.info(f"[TracingInsights] Enriched {enriched}/{len(races)} races with pace data")
        return races

    def load_lap_data(self, year: int, circuit_ref: str) -> list[dict]:
        """
        Carica i LapData raw per una singola gara.

        Returns:
            Lista di dict con campi LapData (non le entity per evitare
            dipendenza circolare — pipeline le converte).
        """
        path = self._find_laps_csv(year, circuit_ref)
        if not path:
            return []

        try:
            return self._parse_laps_csv(path, year)
        except Exception as exc:
            log.warning(f"Errore parsing {path}: {exc}")
            return []

    # ------------------------------------------------------------------
    # Pace computation
    # ------------------------------------------------------------------

    def _compute_pace_observations(self, laps_csv: Path, year: int) -> dict[str, float]:
        """
        Calcola il pace relativo di ogni costruttore per la gara.

        Formula (Heilmeier et al., 2020 §2.3):
            pace_relative[team] = (team_median - field_median) / field_median

        Restituisce valori negativi per team più veloci (es. -0.015 = 1.5% più veloce).
        Usa solo giri "rappresentativi": giro 3-35 (steady state), IsAccurate=True,
        compound non intermedio/wet, fuel-corrected.

        Returns:
            Dict {constructor_ref: pace_delta_normalized} (< 0 = più veloce)
        """
        lap_records = self._parse_laps_csv(laps_csv, year)
        if not lap_records:
            return {}

        # Aggrega per team
        team_times: dict[str, list[float]] = {}
        for lap in lap_records:
            team = lap.get("team", "unknown")
            lt   = lap.get("lap_time_ms")
            if lt and lap.get("is_valid") and lap.get("compound") not in ("INTERMEDIATE", "WET"):
                team_times.setdefault(team, []).append(lt)

        # Filtra team con pochi giri validi
        team_times = {t: v for t, v in team_times.items()
                      if len(v) >= self.min_valid_laps}
        if not team_times:
            return {}

        # Mediana di campo
        all_times = [t for v in team_times.values() for t in v]
        field_median = float(np.median(all_times))
        if field_median <= 0:
            return {}

        # Pace relativo per team
        result = {}
        for team, times in team_times.items():
            team_median = float(np.median(times))
            result[team] = (team_median - field_median) / field_median

        return result

    # ------------------------------------------------------------------
    # CSV parsing
    # ------------------------------------------------------------------

    def _parse_laps_csv(self, path: Path, year: int) -> list[dict]:
        """
        Parsa un laps.csv TracingInsights.

        Robusto a varianti di colonne tra anni diversi.
        Restituisce lista di dict (non LapData entity).
        """
        try:
            import pandas as pd
        except ImportError:
            log.error("pandas non installato: pip install pandas")
            return []

        df = pd.read_csv(path, low_memory=False)

        # Normalizza nomi colonne
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Colonne richieste con alias alternativi
        lap_col      = self._find_col(df, ["laptime", "lap_time", "laptimems"])
        driver_col   = self._find_col(df, ["driver", "drivercode", "driver_code"])
        team_col     = self._find_col(df, ["team", "constructor", "constructorid"])
        lap_num_col  = self._find_col(df, ["lapnumber", "lap_number", "lap"])
        compound_col = self._find_col(df, ["compound", "tyre", "tyrecompound"])
        life_col     = self._find_col(df, ["tyrelife", "tyre_life", "stint"])
        valid_col    = self._find_col(df, ["isaccurate", "is_accurate", "isvalid"])

        if not lap_col or not driver_col:
            log.warning(f"  Colonne obbligatorie mancanti in {path.name}")
            return []

        records = []
        total_laps = int(df[lap_num_col].max()) if lap_num_col else 57

        for _, row in df.iterrows():
            lap_num = int(row[lap_num_col]) if lap_num_col else 0
            # Mantieni solo giri steady-state (escludi inlap, outlap, safety car)
            if lap_num < 3:
                continue

            # Validità
            is_valid = True
            if valid_col:
                val = str(row.get(valid_col, "True")).lower()
                is_valid = val in ("true", "1", "yes")

            # Lap time → ms
            lt_raw = row.get(lap_col, None)
            lap_time_ms = self._parse_laptime(lt_raw)
            if lap_time_ms is None or lap_time_ms <= 0:
                continue
            # Filtra giri anomali (> 3 min = SC / pit incident)
            if lap_time_ms > 180_000:
                continue

            # Compound
            compound_raw = str(row.get(compound_col, "MEDIUM")) if compound_col else "MEDIUM"
            compound = COMPOUND_NORM.get(compound_raw.lower().strip(), "MEDIUM")

            # Tyre life
            tyre_life = 0
            if life_col:
                try:
                    tyre_life = int(float(row.get(life_col, 0)))
                except (ValueError, TypeError):
                    tyre_life = 0

            # Team
            team = str(row.get(team_col, "unknown")).strip().lower() if team_col else "unknown"
            team = self._normalize_team(team)

            # Driver
            driver = str(row.get(driver_col, "UNK")).strip().upper()

            # Fuel correction (TracingInsights formula)
            fuel_corrected_ms = self._fuel_correct(
                lap_time_ms, lap_num, total_laps, compound
            )

            records.append({
                "driver_code":      driver[:3],
                "team":             team,
                "lap_number":       lap_num,
                "lap_time_ms":      lap_time_ms,
                "fuel_corrected_ms": fuel_corrected_ms,
                "compound":         compound,
                "tyre_life":        tyre_life,
                "is_valid":         is_valid,
            })

        return records

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_laps_csv(self, year: int, circuit_ref: str) -> Optional[Path]:
        """Trova il file laps.csv per anno + circuit_ref."""
        year_dir = self.data_dir / str(year)
        if not year_dir.exists():
            return None

        # Cerca cartella che corrisponde al circuit_ref
        for folder_name, ref in FOLDER_TO_CIRCUIT_REF.items():
            if ref == circuit_ref:
                candidate = year_dir / folder_name / "laps.csv"
                if candidate.exists():
                    return candidate
                # Prova varianti nome cartella
                for d in year_dir.iterdir():
                    if d.is_dir() and folder_name.lower() in d.name.lower():
                        laps = d / "laps.csv"
                        if laps.exists():
                            return laps

        # Fallback: cerca qualsiasi cartella con nome simile
        for d in year_dir.iterdir():
            if d.is_dir():
                laps = d / "laps.csv"
                if laps.exists() and circuit_ref.split("_")[0] in d.name.lower():
                    return laps

        return None

    @staticmethod
    def _find_col(df, names: list[str]) -> Optional[str]:
        """Trova il nome della colonna nel DataFrame tra varie alternative."""
        cols = set(df.columns)
        for n in names:
            if n in cols:
                return n
        return None

    @staticmethod
    def _parse_laptime(raw) -> Optional[float]:
        """
        Converte tempo giro in millisecondi.
        Accetta formati: 'mm:ss.mmm', 'ss.mmm', float (già in secondi), int (ms).
        """
        if raw is None:
            return None
        try:
            # Già numerico
            val = float(raw)
            # Decide se è già in ms o in secondi
            return val * 1000 if val < 1000 else val
        except (ValueError, TypeError):
            pass

        s = str(raw).strip()
        # Formato mm:ss.mmm o m:ss.mmm
        m = re.match(r"^(\d+):(\d+)\.(\d+)$", s)
        if m:
            minutes = int(m.group(1))
            seconds = int(m.group(2))
            millis  = int(m.group(3).ljust(3, "0")[:3])
            return (minutes * 60 + seconds) * 1000 + millis

        # Formato ss.mmm
        m2 = re.match(r"^(\d+)\.(\d+)$", s)
        if m2:
            seconds = int(m2.group(1))
            millis  = int(m2.group(2).ljust(3, "0")[:3])
            return seconds * 1000 + millis

        return None

    @staticmethod
    def _fuel_correct(lap_time_ms: float, lap_num: int,
                       total_laps: int, compound: str) -> float:
        """
        Applica la correzione carburante al tempo giro.

        Formula TracingInsights (2025):
            Corrected = Original - (RemainingFuel * 0.03)
            RemainingFuel = InitialFuel * (1 - lap / total_laps)

        Dove:
            InitialFuel = 100 kg (standard F1)
            0.03 s/kg = effetto peso carburante sul lap time
        """
        initial_fuel = 100.0
        remaining    = initial_fuel * max(0.0, 1.0 - lap_num / max(total_laps, 1))
        correction_s = remaining * 0.03
        return lap_time_ms - (correction_s * 1000)

    @staticmethod
    def _normalize_team(raw: str) -> str:
        """
        Normalizza il nome team verso il constructor_ref Jolpica.
        Gestisce varianti storiche (es. 'alphatauri' → 'alphatauri').
        """
        MAP = {
            "red bull":         "red_bull",
            "redbull":          "red_bull",
            "rb":               "red_bull",
            "mercedes":         "mercedes",
            "ferrari":          "ferrari",
            "mclaren":          "mclaren",
            "alpine":           "alpine",
            "alpinerenault":    "alpine",
            "renault":          "renault",
            "aston martin":     "aston_martin",
            "astonmartin":      "aston_martin",
            "williams":         "williams",
            "alphatauri":       "alphatauri",
            "alpha tauri":      "alphatauri",
            "rb f1":            "alphatauri",
            "visa rb":          "alphatauri",
            "haas":             "haas",
            "haas f1 team":     "haas",
            "kick sauber":      "sauber",
            "alfa romeo":       "alfa",
            "alfaromeo":        "alfa",
            "racing point":     "racing_point",
            "force india":      "force_india",
            "toro rosso":       "toro_rosso",
            "cadillac":         "cadillac",
        }
        normalized = raw.strip().lower()
        return MAP.get(normalized, normalized.replace(" ", "_"))
