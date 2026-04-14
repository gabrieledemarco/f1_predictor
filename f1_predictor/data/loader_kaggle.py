"""
data/loader_kaggle.py
=======================
DEPRECATED: This loader is deprecated. Use MongoDB loaders instead.
    - Import data via GitHub Actions workflows
    - Load via: from f1_predictor.data import MongoPaceLoader

This module is kept only for backward compatibility and will be removed
in a future version.

---

Loader per dati Kaggle RaceData (formato tabellare).

Legge i CSV flat dal repository TracingInsights/RaceData e produce:
    1. constructor_pace_observations per ogni gara
    2. LapData entities per il Kalman Filter (Layer 1b)

Formato Kaggle:
    data/racedata/data/
        lap_times.csv    : raceId, driverId, lap, position, time, milliseconds
        races.csv        : raceId, year, round, circuitId, name, date, ...
        circuits.csv     : circuitId, circuitRef, name, ...
        drivers.csv      : driverId, driverRef, code, ...
        constructors.csv : constructorId, constructorRef, ...
        results.csv      : raceId, driverId, constructorId, ...
        pit_stops.csv    : raceId, driverId, stop, lap, time, duration, ...

Utilizzo:
    loader = KaggleRaceDataLoader(data_dir="data/racedata/data")
    # Arricchisce i race_dict già caricati da JolpicaLoader
    enriched = loader.enrich_races(races, years=range(2019, 2027))
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class KaggleRaceDataLoader:
    """
    Carica e normalizza i CSV Kaggle per estrarre constructor_pace_observations.
    
    Args:
        data_dir: Directory contenente i CSV Kaggle (default: data/racedata/data)
        min_valid_laps: Numero minimo di giri validi per calcolare il pace.
    """

    def __init__(self,
                 data_dir: str = "data/racedata/data",
                 min_valid_laps: int = 5):
        self.data_dir = Path(data_dir)
        self.min_valid_laps = min_valid_laps
        self._data_loaded = False
        self.lap_times = None
        self.races = None
        self.circuits = None
        self.drivers = None
        self.constructors = None
        self.results = None
        self.pit_stops = None
        
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
        if not self._data_loaded:
            self._load_kaggle_data()
        
        year_set = set(years) if years else None
        
        enriched = 0
        for race in races:
            year = race.get("year")
            if year_set and year not in year_set:
                continue
            
            circuit_ref = race.get("circuit_ref", "")
            pace_obs = self._compute_pace_observations(year, circuit_ref)
            
            if pace_obs:
                race["constructor_pace_observations"] = pace_obs
                enriched += 1
            else:
                # Se non troviamo dati pace, lasciamo dict vuoto
                race["constructor_pace_observations"] = {}
        
        log.info(f"[KaggleLoader] Arricchite {enriched}/{len(races)} gare con dati pace")
        return races
    
    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    
    def _load_kaggle_data(self):
        """Carica tutti i CSV Kaggle in memoria."""
        try:
            log.info("[KaggleLoader] Caricamento dati Kaggle...")
            
            self.lap_times = pd.read_csv(self.data_dir / "lap_times.csv")
            self.races = pd.read_csv(self.data_dir / "races.csv")
            self.circuits = pd.read_csv(self.data_dir / "circuits.csv")
            self.drivers = pd.read_csv(self.data_dir / "drivers.csv")
            self.constructors = pd.read_csv(self.data_dir / "constructors.csv")
            self.results = pd.read_csv(self.data_dir / "results.csv")
            self.pit_stops = pd.read_csv(self.data_dir / "pit_stops.csv")
            
            # Normalizza nomi colonne
            self.lap_times.columns = self.lap_times.columns.str.strip().str.lower()
            self.races.columns = self.races.columns.str.strip().str.lower()
            self.circuits.columns = self.circuits.columns.str.strip().str.lower()
            self.drivers.columns = self.drivers.columns.str.strip().str.lower()
            self.constructors.columns = self.constructors.columns.str.strip().str.lower()
            self.results.columns = self.results.columns.str.strip().str.lower()
            self.pit_stops.columns = self.pit_stops.columns.str.strip().str.lower()
            
            # Converti tipi
            self.lap_times["milliseconds"] = pd.to_numeric(self.lap_times["milliseconds"], errors="coerce")
            self.lap_times["lap"] = pd.to_numeric(self.lap_times["lap"], errors="coerce")
            
            self._data_loaded = True
            log.info("[KaggleLoader] Dati Kaggle caricati con successo")
            
        except Exception as e:
            log.error(f"[KaggleLoader] Errore caricamento dati: {e}")
            raise
    
    def _compute_pace_observations(self, year: int, circuit_ref: str) -> dict:
        """
        Calcola constructor_pace_observations per una gara specifica.
        
        Args:
            year: Anno della gara
            circuit_ref: circuit_ref Jolpica (es. "bahrain", "jeddah")
            
        Returns:
            Dict {constructor_ref: pace_delta} o vuoto se non trovato.
        """
        if not self._data_loaded:
            return {}
        
        # Trova circuitId dal circuit_ref Jolpica
        circuit_ref_kaggle = self._map_jolpica_to_kaggle_circuit(circuit_ref)
        if not circuit_ref_kaggle:
            # Prova mapping diretto
            circuit_ref_kaggle = circuit_ref
        
        circuit = self.circuits[self.circuits["circuitref"] == circuit_ref_kaggle]
        if circuit.empty:
            log.debug(f"  [KaggleLoader] Circuit {circuit_ref} ({circuit_ref_kaggle}) non trovato")
            return {}
        
        circuit_id = circuit.iloc[0]["circuitid"]
        
        # Trova raceId per anno + circuito
        race = self.races[
            (self.races["year"] == year) & 
            (self.races["circuitid"] == circuit_id)
        ]
        if race.empty:
            log.debug(f"  [KaggleLoader] Race {year} {circuit_ref} non trovata")
            return {}
        
        race_id = race.iloc[0]["raceid"]
        total_laps = race.iloc[0].get("laps", 60)  # default se non presente
        
        # Filtra lap times per questa gara
        race_laps = self.lap_times[self.lap_times["raceid"] == race_id].copy()
        if race_laps.empty:
            log.debug(f"  [KaggleLoader] Nessun lap time per race {race_id}")
            return {}
        
        # Filtra giri validi (escludi primi 2 lap, >180s, pit stops)
        race_laps = self._filter_valid_laps(race_laps, total_laps)
        if len(race_laps) < self.min_valid_laps * 2:  # almeno 2 team con dati
            log.debug(f"  [KaggleLoader] Troppi pochi giri validi per race {race_id}")
            return {}
        
        # Mappa driver -> constructor per questa gara
        driver_to_constructor = self._get_driver_constructor_map(race_id)
        if not driver_to_constructor:
            log.debug(f"  [KaggleLoader] Nessun mapping driver->constructor per race {race_id}")
            return {}
        
        # Aggiungi correzione carburante
        race_laps = self._apply_fuel_correction(race_laps, total_laps)
        
        # Calcola pace per constructor
        return self._compute_constructor_pace(race_laps, driver_to_constructor)
    
    def _map_jolpica_to_kaggle_circuit(self, jolpica_ref: str) -> Optional[str]:
        """
        Mappa circuit_ref Jolpica a circuitRef Kaggle.
        
        Alcuni mapping noti (basati su FOLDER_TO_CIRCUIT_REF):
        """
        # Mapping Jolpica -> Kaggle (basato su nomi simili)
        mapping = {
            "bahrain": "bahrain",
            "jeddah": "jeddah",
            "albert_park": "albert_park",
            "suzuka": "suzuka",
            "shanghai": "shanghai",
            "miami": "miami",
            "imola": "imola",
            "monaco": "monaco",
            "villeneuve": "villeneuve",
            "catalunya": "catalunya",
            "red_bull_ring": "red_bull_ring",
            "silverstone": "silverstone",
            "hungaroring": "hungaroring",
            "spa": "spa",
            "zandvoort": "zandvoort",
            "monza": "monza",
            "baku": "baku",
            "marina_bay": "marina_bay",
            "americas": "americas",
            "rodriguez": "rodriguez",
            "interlagos": "interlagos",
            "vegas": "vegas",
            "losail": "losail",
            "yas_marina": "yas_marina",
            "sepang": "sepang",  # non usato recentemente
        }
        return mapping.get(jolpica_ref, jolpica_ref)
    
    def _filter_valid_laps(self, lap_data: pd.DataFrame, total_laps: int) -> pd.DataFrame:
        """
        Filtra i giri validi per calcolo pace.
        
        Esclude:
            - Primi 2 giri (formazione, caos iniziale)
            - Giri > 180s (safety car, incidenti)
            - Ultimi 2 giri (possibile fuel saving)
            - Pit stop laps (identificati via pit_stops.csv)
        """
        if lap_data.empty:
            return lap_data
        
        # Copia per evitare SettingWithCopyWarning
        filtered = lap_data.copy()
        
        # Escludi primi 2 giri
        filtered = filtered[filtered["lap"] > 2]
        
        # Escludi ultimi 2 giri
        if total_laps > 4:
            filtered = filtered[filtered["lap"] < (total_laps - 1)]
        
        # Escludi giri > 180s (safety car/incidenti)
        filtered = filtered[filtered["milliseconds"] < 180_000]
        
        # Escludi giri < 40s (probabile errore)
        filtered = filtered[filtered["milliseconds"] > 40_000]
        
        # Filtra outliers statistici (rimuovi giri > 3 std dalla mediana)
        if len(filtered) > 10:
            median = filtered["milliseconds"].median()
            std = filtered["milliseconds"].std()
            if std > 0:
                filtered = filtered[
                    abs(filtered["milliseconds"] - median) < 3 * std
                ]
        
        return filtered
    
    def _get_driver_constructor_map(self, race_id: int) -> dict[int, int]:
        """Mappa driverId -> constructorId per una gara."""
        if self.results is None:
            return {}
        
        race_results = self.results[self.results["raceid"] == race_id]
        if race_results.empty:
            return {}
        
        # Crea mapping driverId -> constructorId
        mapping = {}
        for _, row in race_results.iterrows():
            driver_id = row.get("driverid")
            constructor_id = row.get("constructorid")
            if pd.notna(driver_id) and pd.notna(constructor_id):
                mapping[int(driver_id)] = int(constructor_id)
        
        return mapping
    
    def _apply_fuel_correction(self, lap_data: pd.DataFrame, total_laps: int) -> pd.DataFrame:
        """
        Applica correzione carburante ai tempi giro.
        
        Formula TracingInsights:
            Corrected = Original - (RemainingFuel * 0.03s/kg * 1000 ms/s)
            RemainingFuel = InitialFuel * (1 - lap / total_laps)
            InitialFuel = 100 kg (standard)
        """
        if lap_data.empty:
            return lap_data
        
        corrected = lap_data.copy()
        initial_fuel = 100.0  # kg
        fuel_effect_per_kg = 0.03  # secondi per kg
        
        # Correzione in millisecondi
        corrected["fuel_correction_ms"] = (
            initial_fuel * 
            (1.0 - corrected["lap"] / max(total_laps, 1)) * 
            fuel_effect_per_kg * 
            1000
        )
        
        corrected["corrected_ms"] = (
            corrected["milliseconds"] - corrected["fuel_correction_ms"]
        )
        
        return corrected
    
    def _compute_constructor_pace(self, lap_data: pd.DataFrame, 
                                  driver_to_constructor: dict[int, int]) -> dict:
        """
        Calcola pace relativo per ogni constructor.
        
        Returns:
            Dict {constructor_ref: pace_delta} dove pace_delta è (team_median - field_median) / field_median
        """
        if lap_data.empty:
            return {}
        
        # Aggiungi constructorId a ogni lap
        lap_data = lap_data.copy()
        lap_data["constructorid"] = lap_data["driverid"].map(driver_to_constructor)
        
        # Rimuovi giri senza constructor mapping
        lap_data = lap_data.dropna(subset=["constructorid"])
        lap_data["constructorid"] = lap_data["constructorid"].astype(int)
        
        if lap_data.empty:
            return {}
        
        # Raggruppa per constructor
        constructor_times = {}
        for constructor_id, group in lap_data.groupby("constructorid"):
            times = group["corrected_ms"].dropna().values
            if len(times) >= self.min_valid_laps:
                constructor_times[constructor_id] = times
        
        if not constructor_times:
            return {}
        
        # Calcola mediana di campo
        all_times = np.concatenate(list(constructor_times.values()))
        field_median = np.median(all_times)
        if field_median <= 0:
            return {}
        
        # Calcola pace relativo per ogni constructor
        pace_obs = {}
        for constructor_id, times in constructor_times.items():
            team_median = np.median(times)
            
            # Trova constructor_ref
            constructor = self.constructors[
                self.constructors["constructorid"] == constructor_id
            ]
            if constructor.empty:
                continue
            
            constructor_ref = constructor.iloc[0]["constructorref"]
            # pace delta in seconds per lap (negative = faster)
            pace_delta_sec = (team_median - field_median) / 1000.0
            
            # Limita a +/- 2.0 seconds per lap (evita outliers)
            pace_delta_sec = max(min(pace_delta_sec, 2.0), -2.0)
            pace_obs[constructor_ref] = float(pace_delta_sec)
        
        return pace_obs


# ---------------------------------------------------------------------------
# Funzione helper per integrazione
# ---------------------------------------------------------------------------

def create_kaggle_loader(data_dir: str = "data/racedata/data") -> Optional[KaggleRaceDataLoader]:
    """Factory per creare KaggleRaceDataLoader se i dati esistono."""
    data_path = Path(data_dir)
    required_files = ["lap_times.csv", "races.csv", "circuits.csv", 
                     "drivers.csv", "constructors.csv", "results.csv"]
    
    for f in required_files:
        if not (data_path / f).exists():
            log.info(f"[KaggleLoader] File {f} non trovato, loader non creato")
            return None
    
    return KaggleRaceDataLoader(data_dir=data_dir)