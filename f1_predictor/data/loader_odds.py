"""
data/loader_odds.py
===================
Carica quote Pinnacle F1 via The Odds API e produce CalibrationRecord
entities per il Layer 4 (Isotonic Calibrator).

The Odds API: https://the-odds-api.com
    - Sport:       motorsport_formula_one
    - Bookmaker:   pinnacle (vig ≤ 3%)
    - Markets:     h2h (race winner), outrights (podium, top6)

Flusso:
    1. Fetch quote pre-gara (-48h, -24h, -2h da start)
    2. Devigging con Power Method (Strumbelj, 2014)
    3. Allineamento con race_id e driver_code (via mapping piloti)
    4. Salvataggio come OddsRecord + CalibrationRecord

Setup:
    Crea .env o imposta env var:
        THE_ODDS_API_KEY=your_key_here

    Oppure passa la chiave direttamente:
        loader = OddsLoader(api_key="your_key")

Utilizzo:
    loader = OddsLoader()

    # Fetch quote storiche 2023-2025 (una tantum)
    records = loader.fetch_historical_season(2024)
    loader.save_records(records, "data/pinnacle_odds/2024.jsonl")

    # Fetch quote prossimo GP (routine pre-gara)
    upcoming = loader.fetch_upcoming()
    loader.save_records(upcoming, "data/pinnacle_odds/upcoming.jsonl")

    # Carica records salvati per il calibratore
    all_records = loader.load_saved_records("data/pinnacle_odds/")
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

API_BASE        = "https://api.the-odds-api.com/v4"
SPORT_KEY       = "motorsport_formula_one"
BOOKMAKER       = "pinnacle"
REQUEST_DELAY   = 0.5   # sec — The Odds API: 500 req/mese free tier
MAX_RETRIES     = 3

# Mapping nome Odds API → driver_code FIA (aggiornare ogni stagione)
# Chiave: nome come appare nell'API, valore: codice 3 lettere
DRIVER_NAME_MAP: dict[str, str] = {
    "Max Verstappen":       "VER",
    "Lewis Hamilton":       "HAM",
    "Charles Leclerc":      "LEC",
    "Carlos Sainz":         "SAI",
    "Lando Norris":         "NOR",
    "Oscar Piastri":        "PIA",
    "George Russell":       "RUS",
    "Fernando Alonso":      "ALO",
    "Lance Stroll":         "STR",
    "Sergio Perez":         "PER",
    "Pierre Gasly":         "GAS",
    "Esteban Ocon":         "OCO",
    "Alexander Albon":      "ALB",
    "Logan Sargeant":       "SAR",
    "Nico Hulkenberg":      "HUL",
    "Kevin Magnussen":      "MAG",
    "Valtteri Bottas":      "BOT",
    "Guanyu Zhou":          "ZHO",
    "Yuki Tsunoda":         "TSU",
    "Daniel Ricciardo":     "RIC",
    "Liam Lawson":          "LAW",
    "Franco Colapinto":     "COL",
    "Oliver Bearman":       "BEA",
    "Jack Doohan":          "DOO",
    "Andrea Kimi Antonelli":"ANT",
    "Isack Hadjar":         "HAD",
    "Gabriel Bortoleto":    "BOR",
}

# ---------------------------------------------------------------------------
# Devigging (Strumbelj, 2014 — Power Method)
# ---------------------------------------------------------------------------

def devig_power(implied_probs: list[float], epsilon: float = 1e-8) -> list[float]:
    """
    Rimuove il vig dalle probabilità implicite usando il Power Method.

    P(A_true) = P(A_raw)^k  dove k risolve: sum(P_i^k) = 1

    Superiore al Basic Method (divisione per totale) e al Shin Method
    per F1 secondo Strumbelj (2014) — vedi anche Walsh & Joshi (2023).

    Args:
        implied_probs: Lista di prob implicite crude (includono vig).
        epsilon: Tolleranza per la ricerca binaria di k.

    Returns:
        Lista di probabilità vere (somma = 1).
    """
    probs = np.array(implied_probs, dtype=float)
    probs = np.clip(probs, 1e-6, 1.0)

    # Cerca k con binary search
    k_lo, k_hi = 0.5, 3.0
    for _ in range(100):
        k_mid = (k_lo + k_hi) / 2
        total = float(np.sum(probs ** k_mid))
        if total > 1.0:
            k_lo = k_mid
        else:
            k_hi = k_mid
        if abs(total - 1.0) < epsilon:
            break

    k = (k_lo + k_hi) / 2
    true_probs = probs ** k
    true_probs /= true_probs.sum()   # renormalizza per floating point
    return true_probs.tolist()


# ---------------------------------------------------------------------------
# OddsLoader
# ---------------------------------------------------------------------------

class OddsLoader:
    """
    Wrapper su The Odds API per quote Pinnacle F1.

    Produce OddsRecord e CalibrationRecord entities salvate come JSONL
    per la fase di training del Layer 4 Isotonic Calibrator.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 cache_dir: str = "data/pinnacle_odds",
                 db=None):
        self.api_key = api_key or os.environ.get("THE_ODDS_API_KEY", "")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db = db           # MongoDB connection (None = JSONL-only)
        self._session = None

        if not self.api_key:
            log.warning("THE_ODDS_API_KEY non impostata. Fetch API non disponibile.")

    # ------------------------------------------------------------------
    # Public API — Fetch
    # ------------------------------------------------------------------

    def fetch_upcoming(self, markets: list[str] = None) -> list[dict]:
        """
        Fetch quote per il prossimo GP (regime live/pre-gara).

        Args:
            markets: Lista mercati (default: ['h2h', 'outrights']).

        Returns:
            Lista OddsRecord dict.
        """
        markets = markets or ["h2h", "outrights"]
        records = []
        for market in markets:
            data = self._api_get("sports", SPORT_KEY, "odds", params={
                "apiKey":     self.api_key,
                "bookmakers": BOOKMAKER,
                "markets":    market,
                "oddsFormat": "decimal",
            })
            if data:
                records.extend(self._parse_odds_response(data, market))

        log.info(f"[OddsLoader] Fetch upcoming: {len(records)} records")
        return records

    def fetch_historical_season(self, year: int) -> list[dict]:
        """
        Fetch quote storiche per una stagione intera.

        Usa l'endpoint /historical/ di The Odds API (richiede piano a pagamento
        per stagioni complete; il free tier copre solo le ultime ~6 settimane).

        Returns:
            Lista OddsRecord dict ordinata per timestamp.
        """
        if not self.api_key:
            log.error("API key necessaria per fetch storici")
            return []

        # Fetch da file cache se già presenti
        cache_file = self.cache_dir / f"{year}_raw.jsonl"
        if cache_file.exists():
            log.info(f"[OddsLoader] Caricando {year} dalla cache: {cache_file}")
            return self._load_jsonl(cache_file)

        # Nota: l'endpoint storico richiede timestamps specifici per ogni evento
        # In pratica questo richiede prima di costruire il calendario con i date/times
        # dei singoli GP, poi fetchare le odds per ogni evento.
        # Implementazione semplificata: fetch per evento singolo.
        log.warning(
            f"Fetch storico {year}: richiede la lista eventi dal calendario F1. "
            f"Usa fetch_event() per ogni GP con il suo eventId."
        )
        return []

    def fetch_event(self, event_id: str, market: str = "h2h") -> list[dict]:
        """
        Fetch quote per un singolo evento (GP) via eventId.

        L'eventId si trova nella risposta di /sports/{sport}/events.

        Returns:
            Lista OddsRecord dict.
        """
        data = self._api_get("sports", SPORT_KEY, "events", event_id, "odds", params={
            "apiKey":     self.api_key,
            "bookmakers": BOOKMAKER,
            "markets":    market,
            "oddsFormat": "decimal",
        })
        if not data:
            return []
        return self._parse_odds_response([data], market)

    def list_events(self) -> list[dict]:
        """Lista eventi F1 disponibili sull'API."""
        data = self._api_get("sports", SPORT_KEY, "events", params={
            "apiKey": self.api_key,
        })
        return data if isinstance(data, list) else []

    # ------------------------------------------------------------------
    # Public API — Storage
    # ------------------------------------------------------------------

    def save_records(self, records: list[dict],
                     filepath: Optional[str] = None,
                     db=None) -> None:
        """
        Salva records su MongoDB (upsert) e/o JSONL su disco.

        Args:
            records:  Lista record da salvare.
            filepath: Percorso JSONL su disco (opzionale).
            db:       Connessione MongoDB (override self.db, opzionale).
        """
        target_db = db or self.db

        # ── MongoDB upsert ──────────────────────────────────────
        if target_db is not None:
            try:
                from core.db import odds_records_collection
                coll = odds_records_collection(target_db)
                inserted = 0
                skipped = 0
                for rec in records:
                    filter_key = {
                        "race_id":     rec.get("race_id"),
                        "driver_code": rec.get("driver_code"),
                        "market":      rec.get("market"),
                        "timestamp":   rec.get("timestamp"),
                    }
                    result = coll.update_one(filter_key, {"$setOnInsert": rec}, upsert=True)
                    if result.upserted_id:
                        inserted += 1
                    else:
                        skipped += 1
                log.info(
                    f"[OddsLoader] MongoDB: {inserted} inseriti, "
                    f"{skipped} già presenti"
                )
            except Exception as exc:
                log.warning(f"[OddsLoader] MongoDB save fallita: {exc}")

        # ── JSONL su disco con dedup ─────────────────────────────
        if filepath is not None:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)

            existing_keys: set[tuple] = set()
            if path.exists():
                for existing in self._load_jsonl(path):
                    existing_keys.add((
                        existing.get("race_id"),
                        existing.get("driver_code"),
                        existing.get("market"),
                        existing.get("timestamp"),
                    ))

            new_records = [
                r for r in records
                if (r.get("race_id"), r.get("driver_code"),
                    r.get("market"),  r.get("timestamp")) not in existing_keys
            ]

            if new_records:
                with open(path, "a", encoding="utf-8") as f:
                    for rec in new_records:
                        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                log.info(
                    f"[OddsLoader] JSONL: {len(new_records)}/{len(records)} "
                    f"nuovi record → {path.name}"
                )
            else:
                log.info(f"[OddsLoader] JSONL: nessun record nuovo (tutti già presenti)")

    def load_saved_records(self, directory: Optional[str] = None,
                           db=None) -> list[dict]:
        """
        Carica tutti gli OddsRecord da MongoDB o da JSONL su disco.

        Args:
            directory: Directory JSONL (fallback se MongoDB non disponibile).
            db:        Connessione MongoDB (override self.db).

        Returns:
            Lista di OddsRecord dict ordinata per timestamp desc.
        """
        target_db = db or self.db

        # ── MongoDB primary ──────────────────────────────────────
        if target_db is not None:
            try:
                from core.db import odds_records_collection
                coll = odds_records_collection(target_db)
                records = list(coll.find({}, {"_id": 0}).sort("timestamp", -1))
                log.info(f"[OddsLoader] MongoDB: caricati {len(records)} records")
                return records
            except Exception as exc:
                log.warning(f"[OddsLoader] MongoDB load fallita: {exc} — fallback JSONL")

        # ── Fallback JSONL ───────────────────────────────────────
        search_dir = Path(directory) if directory else self.cache_dir
        all_records = []
        for path in sorted(search_dir.glob("*.jsonl")):
            all_records.extend(self._load_jsonl(path))
        log.info(f"[OddsLoader] JSONL: caricati {len(all_records)} records totali")
        return all_records

    def build_calibration_records(self,
                                   odds_records: list[dict],
                                   model_predictions: dict[str, float],
                                   outcomes: dict[str, int]) -> list[dict]:
        """
        Costruisce CalibrationRecord dict da OddsRecord + predizioni modello.

        Args:
            odds_records: Lista OddsRecord dict.
            model_predictions: {f"{race_id}_{driver_code}": p_model_raw}
            outcomes: {f"{race_id}_{driver_code}": 1 or 0}

        Returns:
            Lista CalibrationRecord dict per il Layer 4.
        """
        cal_records = []
        for rec in odds_records:
            key = f"{rec['race_id']}_{rec['driver_code']}"
            p_raw = model_predictions.get(key)
            outcome = outcomes.get(key)

            if p_raw is None or outcome is None:
                continue

            cal_records.append({
                "race_id":            rec["race_id"],
                "driver_code":        rec["driver_code"],
                "market":             rec["market"],
                "p_model_raw":        p_raw,
                "p_model_calibrated": p_raw,   # sarà aggiornato dopo fit isotonic
                "p_pinnacle_novig":   rec["p_novig"],
                "edge":               p_raw - rec["p_novig"],
                "outcome":            outcome,
                "timestamp":          rec["timestamp"],
            })

        log.info(f"[OddsLoader] Costruiti {len(cal_records)} CalibrationRecord")
        return cal_records

    # ------------------------------------------------------------------
    # API fetch
    # ------------------------------------------------------------------

    def _api_get(self, *path_parts: str, params: dict = None) -> Optional[dict | list]:
        """Esegue GET su The Odds API con retry."""
        url = "/".join([API_BASE.rstrip("/")] + list(path_parts))
        try:
            import requests
        except ImportError:
            log.error("requests non installato")
            return None

        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(REQUEST_DELAY)
                resp = self._get_session().get(url, params=params, timeout=15)

                if resp.status_code == 200:
                    # Log remaining quota
                    remaining = resp.headers.get("x-requests-remaining", "?")
                    log.debug(f"API quota rimanente: {remaining}")
                    return resp.json()
                elif resp.status_code == 401:
                    log.error("API key non valida o non autorizzata")
                    return None
                elif resp.status_code == 422:
                    log.warning(f"Parametri non validi: {resp.text}")
                    return None
                elif resp.status_code == 429:
                    log.warning(f"Rate limit. Attendo 10s...")
                    time.sleep(10)
                else:
                    log.warning(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    return None

            except Exception as exc:
                log.warning(f"Tentativo {attempt+1}/{MAX_RETRIES}: {exc}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(5)

        return None

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers["User-Agent"] = "BetBreakerF1/1.0"
        return self._session

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_odds_response(self, data: list[dict], market: str) -> list[dict]:
        """
        Converte risposta API in lista OddsRecord dict con devigging.

        Ogni evento contiene una lista di bookmakers; filtra solo Pinnacle.
        """
        records = []
        now = datetime.now(timezone.utc).isoformat()

        for event in (data if isinstance(data, list) else [data]):
            event_id      = event.get("id", "")
            commence_time = event.get("commence_time", now)

            # Calcola hours_to_race
            try:
                race_dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
                now_dt  = datetime.now(timezone.utc)
                hours_to_race = (race_dt - now_dt).total_seconds() / 3600
            except Exception:
                hours_to_race = 0.0

            for bookmaker in event.get("bookmakers", []):
                if bookmaker.get("key") != BOOKMAKER:
                    continue

                for mkt in bookmaker.get("markets", []):
                    if mkt.get("key") != market:
                        continue

                    outcomes = mkt.get("outcomes", [])
                    if not outcomes:
                        continue

                    # Devigging
                    raw_probs = []
                    names = []
                    for o in outcomes:
                        price = float(o.get("price", 2.0))
                        raw_probs.append(1.0 / price)
                        names.append(o.get("name", ""))

                    try:
                        true_probs = devig_power(raw_probs)
                    except Exception:
                        total = sum(raw_probs) or 1.0
                        true_probs = [p / total for p in raw_probs]

                    # Tenta di derivare race_id dall'event_id/nome
                    race_id = self._parse_race_id(event_id, event.get("home_team", ""))

                    for name, raw_p, true_p, outcome_raw in zip(
                            names, raw_probs, true_probs, outcomes):
                        driver_code = self._resolve_driver(name)
                        if not driver_code:
                            continue

                        price = float(outcome_raw.get("price", 2.0))
                        records.append({
                            "race_id":        race_id,
                            "driver_code":    driver_code,
                            "market":         market,
                            "odd_decimal":    price,
                            "p_implied_raw":  raw_p,
                            "p_novig":        float(true_p),
                            "bookmaker":      BOOKMAKER,
                            "timestamp":      now,
                            "hours_to_race":  hours_to_race,
                            "event_id":       event_id,
                        })

        return records

    @staticmethod
    def _resolve_driver(name: str) -> Optional[str]:
        """Risolve nome completo → codice 3 lettere."""
        # Lookup diretto
        code = DRIVER_NAME_MAP.get(name)
        if code:
            return code
        # Cerca per cognome
        surname = name.split()[-1] if name else ""
        for full_name, c in DRIVER_NAME_MAP.items():
            if surname.lower() == full_name.split()[-1].lower():
                return c
        log.debug(f"Driver non risolto: '{name}'")
        return None

    @staticmethod
    def _parse_race_id(event_id: str, event_name: str) -> int:
        """
        Tenta di derivare race_id dal nome evento.
        Fallback: 0 (da aggiornare manualmente dopo il fetch).
        """
        import re
        m = re.search(r"(\d{4})", event_id + event_name)
        return int(m.group(1)) * 1000 if m else 0

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records
