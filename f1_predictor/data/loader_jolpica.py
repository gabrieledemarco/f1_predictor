"""
data/loader_jolpica.py
======================
Carica dati storici F1 via Jolpica API (fork Ergast mantenuto).
Endpoint base: https://api.jolpi.ca/ergast/f1/

Flusso:
    1. Controlla cache locale (data/cache/jolpica/)
    2. Se mancante, fetch dall'API con rate-limiting (4 req/sec)
    3. Serializza su disco per run successive (zero fetch ripetuti)
    4. Converte nel formato dict atteso da F1PredictionPipeline.fit()

Formato output per ogni gara:
    {
      "race_id"   : int,          # anno*1000 + round (es. 2024001)
      "year"      : int,
      "round"     : int,
      "circuit_ref": str,
      "circuit_type": CircuitType,
      "race_name" : str,
      "date"      : str,          # ISO 8601
      "is_season_end": bool,
      "is_major_regulation_change": bool,
      "results"   : list[dict],   # vedi _parse_result()
      "qualifying": list[dict],   # grid positions
      "constructor_pace_observations": dict  # populate da TracingInsights
    }

Utilizzo:
    loader = JolpicaLoader(cache_dir="data/cache/jolpica")
    races = loader.load_seasons(range(2019, 2027))
    # oppure singola stagione:
    races_2024 = loader.load_season(2024, through_round=24)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

BASE_URL = "https://api.jolpi.ca/ergast/f1"
REQUEST_DELAY = 0.26          # sec tra chiamate (< 4 req/sec)
MAX_RETRIES   = 3
RETRY_DELAY   = 5.0           # sec prima di ritentare dopo errore

# Stagioni con cambi regolamentari maggiori (impattano il decay TTT)
MAJOR_REG_CHANGES = {2009, 2014, 2017, 2022}

# Mapping circuitId Jolpica → CircuitType enum
# Fonte: Heilmeier et al. (2020) — 5 cluster
# Base mapping - can be extended via JolpicaLoader.add_circuit_type()
_BASE_CIRCUIT_TYPE_MAP: dict[str, str] = {
    # STREET
    "monaco":          "street",
    "baku":            "street",
    "marina_bay":      "street",
    "vegas":           "street",
    "jeddah":          "street",
    "miami":           "street",
    # HIGH_DOWNFORCE
    "hungaroring":     "high_df",
    "catalunya":       "high_df",
    "shanghai":        "high_df",
    "imola":           "high_df",
    "zandvoort":       "high_df",
    # HIGH_SPEED
    "monza":           "high_speed",
    "spa":             "high_speed",
    "red_bull_ring":   "high_speed",
    # MIXED
    "silverstone":     "mixed",
    "suzuka":          "mixed",
    "americas":        "mixed",    # Austin COTA
    "interlagos":      "mixed",
    "rodriguez":       "mixed",    # Città del Messico
    "villeneuve":      "mixed",    # Montreal
    # DESERT
    "bahrain":         "desert",
    "yas_marina":      "desert",
    "losail":          "desert",
    "riyadh":          "desert",
}

# Ultimi round di ogni stagione (per is_season_end)
SEASON_LAST_ROUND: dict[int, int] = {
    2019: 21, 2020: 17, 2021: 22, 2022: 22,
    2023: 22, 2024: 24, 2025: 24, 2026: 24,
}

# ---------------------------------------------------------------------------
# Loader principale
# ---------------------------------------------------------------------------

class JolpicaLoader:
    """
    Loader per dati F1 storici via Jolpica API con cache locale.

    Il cache previene chiamate ripetute all'API e permette di lavorare
    offline dopo il primo fetch. I file JSON vengono salvati come:
        cache_dir/{year}_{round:02d}_results.json
        cache_dir/{year}_{round:02d}_qualifying.json
        cache_dir/{year}_rounds.json  (elenco round disponibili)

    Args:
        cache_dir: Directory locale per la cache JSON.
        force_refresh: Se True, ignora la cache e re-fetcha sempre.
    """
    
    # Circuit type mapping - can be extended at runtime
    circuit_type_map: dict[str, str] = _BASE_CIRCUIT_TYPE_MAP.copy()
    
    @classmethod
    def add_circuit_type(cls, circuit_id: str, circuit_type: str):
        """
        Add or update a circuit type mapping.
        
        Args:
            circuit_id: Jolpica circuit identifier (e.g., 'monaco').
            circuit_type: One of 'street', 'high_df', 'high_speed', 'mixed', 'desert'.
        """
        if circuit_type not in {'street', 'high_df', 'high_speed', 'mixed', 'desert'}:
            raise ValueError(f"Invalid circuit_type: {circuit_type}")
        cls.circuit_type_map[circuit_id] = circuit_type
    
    @classmethod
    def load_circuit_types_from_json(cls, json_path: str):
        """
        Load circuit type mappings from a JSON file.
        
        JSON format: {"circuit_id": "circuit_type", ...}
        """
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        for circuit_id, circuit_type in mappings.items():
            cls.add_circuit_type(circuit_id, circuit_type)

    def __init__(self,
                 cache_dir: str = "data/cache/jolpica",
                 force_refresh: bool = False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.force_refresh = force_refresh
        self._session = None   # lazy init

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_seasons(self,
                     years: range | list[int],
                     through_round: Optional[int] = None) -> list[dict]:
        """
        Carica più stagioni in sequenza.

        Args:
            years: Range o lista di anni (es. range(2019, 2027)).
            through_round: Se specificato, l'ultima stagione viene troncata
                           a questo round (per training incrementale).

        Returns:
            Lista ordinata di race dict pronti per pipeline.fit().
        """
        races = []
        years_list = list(years)
        for i, year in enumerate(years_list):
            tr = through_round if i == len(years_list) - 1 else None
            year_races = self.load_season(year, through_round=tr)
            races.extend(year_races)
            log.info(f"[Jolpica] Season {year}: {len(year_races)} races loaded")
        return races

    def load_season(self,
                    year: int,
                    through_round: Optional[int] = None) -> list[dict]:
        """
        Carica una singola stagione.

        Args:
            year: Anno F1.
            through_round: Tronca ai primi N round (None = tutti).

        Returns:
            Lista ordinata (per round) di race dict.
        """
        rounds = self._get_available_rounds(year)
        if through_round:
            rounds = [r for r in rounds if r <= through_round]

        races = []
        total_rounds = len(rounds)
        for idx, round_num in enumerate(rounds):
            race = self._load_race(year, round_num)
            if race:
                races.append(race)
            # Piccolo progresso
            if (idx + 1) % 5 == 0 or idx + 1 == total_rounds:
                log.info(f"  [{year}] {idx+1}/{total_rounds} rounds processed")

        return races

    # ------------------------------------------------------------------
    # Core race loading
    # ------------------------------------------------------------------

    def _load_race(self, year: int, round_num: int) -> Optional[dict]:
        """Carica una singola gara con risultati e qualifiche."""
        results_raw  = self._fetch_results(year, round_num)
        quali_raw    = self._fetch_qualifying(year, round_num)

        if not results_raw:
            log.warning(f"  [{year} R{round_num}] No results data — skipping")
            return None

        return self._build_race_dict(year, round_num, results_raw, quali_raw)

    def _build_race_dict(self, year: int, round_num: int,
                         results_raw: dict, quali_raw: Optional[dict]) -> dict:
        """Converte i raw JSON Jolpica nel formato pipeline."""

        race_table = results_raw.get("MRData", {}).get("RaceTable", {})
        races_list = race_table.get("Races", [])
        if not races_list:
            return None

        race_info = races_list[0]
        circuit   = race_info.get("Circuit", {})
        circuit_id = circuit.get("circuitId", "unknown")
        circuit_type = self.circuit_type_map.get(circuit_id, "mixed")

        last_round = SEASON_LAST_ROUND.get(year, 24)
        is_season_end = (round_num == last_round)

        race_dict = {
            "race_id":    year * 1000 + round_num,
            "year":       year,
            "round":      round_num,
            "circuit_ref": circuit_id,
            "circuit_type": circuit_type,
            "race_name":  race_info.get("raceName", ""),
            "date":       race_info.get("date", ""),
            "is_season_end": is_season_end,
            "is_major_regulation_change": (year in MAJOR_REG_CHANGES and round_num == 1),
            "results":    [],
            "qualifying": [],
            "constructor_pace_observations": {},   # riempito da loader_tracinginsights
        }

        # Parse risultati gara
        for res in race_info.get("Results", []):
            race_dict["results"].append(self._parse_result(res))

        # Parse qualifiche
        if quali_raw:
            qt = quali_raw.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if qt:
                for qr in qt[0].get("QualifyingResults", []):
                    race_dict["qualifying"].append({
                        "driver_code":    self._extract_code(qr.get("Driver", {})),
                        "grid_position":  int(qr.get("position", 20)),
                        "q1": qr.get("Q1", ""),
                        "q2": qr.get("Q2", ""),
                        "q3": qr.get("Q3", ""),
                    })

        return race_dict

    @staticmethod
    def _parse_result(res: dict) -> dict:
        """
        Converte un singolo risultato Jolpica nel formato pipeline.

        DNF viene rappresentato con finish_position=None.
        """
        driver    = res.get("Driver", {})
        construct = res.get("Constructor", {})
        status    = res.get("status", "Unknown")

        # Finish position: None se ritiro
        pos_str = res.get("position", "")
        try:
            finish_pos = int(pos_str)
        except (ValueError, TypeError):
            finish_pos = None

        # Punti (Jolpica restituisce stringa)
        try:
            points = float(res.get("points", "0"))
        except (ValueError, TypeError):
            points = 0.0

        return {
            "driver_code":     JolpicaLoader._extract_code(driver),
            "driver_forename": driver.get("givenName", ""),
            "driver_surname":  driver.get("familyName", ""),
            "constructor_ref": construct.get("constructorId", "unknown"),
            "constructor_name": construct.get("name", ""),
            "grid_position":   int(res.get("grid", "0") or 0),
            "finish_position": finish_pos,
            "points":          points,
            "laps_completed":  int(res.get("laps", "0") or 0),
            "status":          status,
            "fastest_lap_rank": None,
        }

    @staticmethod
    def _extract_code(driver: dict) -> str:
        """
        Estrae il codice pilota a 3 lettere.
        Jolpica usa 'code' (es. 'VER'); fallback su cognome[:3].upper().
        """
        code = driver.get("code", "")
        if code:
            return code.upper()
        surname = driver.get("familyName", "UNK")
        return surname[:3].upper()

    # ------------------------------------------------------------------
    # API fetch con cache
    # ------------------------------------------------------------------

    def _fetch_results(self, year: int, round_num: int) -> Optional[dict]:
        cache_key = f"{year}_{round_num:02d}_results"
        return self._fetch_with_cache(
            cache_key,
            f"{BASE_URL}/{year}/{round_num}/results.json?limit=30"
        )

    def _fetch_qualifying(self, year: int, round_num: int) -> Optional[dict]:
        cache_key = f"{year}_{round_num:02d}_qualifying"
        return self._fetch_with_cache(
            cache_key,
            f"{BASE_URL}/{year}/{round_num}/qualifying.json?limit=30"
        )

    def _get_available_rounds(self, year: int) -> list[int]:
        """Restituisce la lista di round disponibili per una stagione."""
        cache_key = f"{year}_rounds"
        cached = self._read_cache(cache_key)
        if cached:
            return cached.get("rounds", [])

        data = self._http_get(f"{BASE_URL}/{year}/races.json?limit=30")
        if not data:
            # Fallback: usa i round noti
            last = SEASON_LAST_ROUND.get(year, 24)
            rounds = list(range(1, last + 1))
        else:
            races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            rounds = [int(r["round"]) for r in races]

        self._write_cache(cache_key, {"rounds": rounds})
        return rounds

    def _fetch_with_cache(self, cache_key: str, url: str) -> Optional[dict]:
        if not self.force_refresh:
            cached = self._read_cache(cache_key)
            if cached is not None:
                return cached

        data = self._http_get(url)
        if data:
            self._write_cache(cache_key, data)
        return data

    def _http_get(self, url: str) -> Optional[dict]:
        """HTTP GET con retry e rate limiting."""
        try:
            import requests
        except ImportError:
            log.error("requests non installato: pip install requests")
            return None

        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(REQUEST_DELAY)
                resp = self._get_session().get(url, timeout=15)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    log.warning(f"Rate limited. Attendo {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                else:
                    log.warning(f"HTTP {resp.status_code} per {url}")
                    return None
            except Exception as exc:
                log.warning(f"Tentativo {attempt+1}/{MAX_RETRIES} fallito: {exc}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)

        log.error(f"Tutti i tentativi falliti per: {url}")
        return None

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": "BetBreakerF1/1.0 (research project)",
                "Accept": "application/json",
            })
        return self._session

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> Optional[dict]:
        path = self._cache_path(key)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning(f"Cache corrotta ({key}): {exc}")
        return None

    def _write_cache(self, key: str, data: dict) -> None:
        try:
            self._cache_path(key).write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception as exc:
            log.warning(f"Impossibile scrivere cache ({key}): {exc}")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Elimina tutta la cache locale."""
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
        log.info("Cache Jolpica eliminata.")

    def cache_stats(self) -> dict:
        """Statistiche sui file in cache."""
        files = list(self.cache_dir.glob("*.json"))
        return {
            "total_files": len(files),
            "total_size_kb": sum(f.stat().st_size for f in files) // 1024,
        }
