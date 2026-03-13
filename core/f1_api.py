"""
core/f1_api.py
Recupera dati reali F1 da:
  - OpenF1 API   (openf1.org)     — sessioni, tempi sul giro, posizioni
  - Jolpica API  (jolpi.ca/ergast) — qualifiche, gare, sprint strutturate
"""
import requests
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from core.db import get_db, lap_times_collection, driver_info_collection, session_stats_collection

logger = logging.getLogger("f1_api")

# ─── Base URLs ─────────────────────────────────────────────────────────
OPENF1_BASE  = "https://api.openf1.org/v1"
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
TIMEOUT      = 12
RETRY        = 2

# ─── Session type mapping ──────────────────────────────────────────────
SESSION_TYPE_MAP = {
    "Practice 1":  "FP1",
    "Practice 2":  "FP2",
    "Practice 3":  "FP3",
    "Qualifying":  "Quali",
    "Sprint":      "Sprint",
    "Sprint Qualifying": "Sprint Quali",
    "Race":        "Gara",
}

DRIVER_NAME_MAP = {
    # Jolpica surname → display name
    "russell": "Russell", "antonelli": "Antonelli", "leclerc": "Leclerc",
    "hamilton": "Hamilton", "piastri": "Piastri", "norris": "Norris",
    "hadjar": "Hadjar", "verstappen": "Verstappen", "lawson": "Lawson",
    "lindblad": "Lindblad", "bortoleto": "Bortoleto", "hulkenberg": "Hulkenberg",
    "ocon": "Ocon", "bearman": "Bearman", "gasly": "Gasly",
    "colapinto": "Colapinto", "albon": "Albon", "sainz": "Sainz",
    "bottas": "Bottas", "perez": "Perez", "alonso": "Alonso", "stroll": "Stroll",
    # OpenF1 usa numeri/codici
    "RUS": "Russell", "ANT": "Antonelli", "LEC": "Leclerc", "HAM": "Hamilton",
    "PIA": "Piastri", "NOR": "Norris", "HAD": "Hadjar", "VER": "Verstappen",
    "LAW": "Lawson", "LIN": "Lindblad", "BOR": "Bortoleto", "HUL": "Hulkenberg",
    "OCO": "Ocon", "BEA": "Bearman", "GAS": "Gasly", "COL": "Colapinto",
    "ALB": "Albon", "SAI": "Sainz", "BOT": "Bottas", "PER": "Perez",
    "ALO": "Alonso", "STR": "Stroll",
}


def _get(url: str, params: Optional[dict] = None) -> Optional[Union[dict, list]]:
    for attempt in range(RETRY):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"GET {url} attempt {attempt+1} failed: {e}")
            time.sleep(1.5)
    return None


def _normalize_driver(name: str) -> str:
    """Normalizza il nome pilota."""
    name = str(name).strip()
    if name in DRIVER_NAME_MAP:
        return DRIVER_NAME_MAP[name]
    low = name.lower()
    if low in DRIVER_NAME_MAP:
        return DRIVER_NAME_MAP[low]
    # Prova match parziale
    for k, v in DRIVER_NAME_MAP.items():
        if low in k or k in low:
            return v
    return name.capitalize()


# ══════════════════════════════════════════════════════════════
# JOLPICA / ERGAST  —  Struttura risultati qualifiche/gara
# ══════════════════════════════════════════════════════════════

def get_season_schedule(year: int = 2026) -> List[dict]:
    """Lista gare del calendario."""
    data = _get(f"{JOLPICA_BASE}/{year}.json")
    if not data:
        return []
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    return [{"round": r["round"], "name": r["raceName"],
             "circuit": r["Circuit"]["circuitName"],
             "country": r["Circuit"]["Location"]["country"],
             "date": r.get("date", "")} for r in races]


def get_qualifying_results(year: int, round_num: int) -> Dict[str, dict]:
    """
    Qualifiche: ritorna {pilota: {Q1_gap, Q2_gap, Q3_gap, position, q3_time_sec}}
    """
    data = _get(f"{JOLPICA_BASE}/{year}/{round_num}/qualifying.json")
    if not data:
        return {}

    results = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not results:
        return {}

    quali_list = results[0].get("QualifyingResults", [])
    out = {}
    best_q3 = None

    # Prima passata: trova il leader (miglior Q3)
    q3_times = {}
    for r in quali_list:
        q3 = r.get("Q3", "")
        if q3:
            try:
                q3_times[_normalize_driver(r["Driver"]["familyName"])] = _time_to_sec(q3)
            except:
                pass

    if q3_times:
        best_q3 = min(q3_times.values())

    for r in quali_list:
        drv    = _normalize_driver(r["Driver"]["familyName"])
        pos    = int(r.get("position", 99))
        q1     = r.get("Q1", "")
        q2     = r.get("Q2", "")
        q3     = r.get("Q3", "")
        q3_sec = _time_to_sec(q3) if q3 else None
        gap    = (q3_sec - best_q3) if (q3_sec and best_q3) else 5.0

        out[drv] = {
            "position": pos,
            "Q1": q1, "Q2": q2, "Q3": q3,
            "Q3_sec": q3_sec,
            "gap_to_leader": min(gap, 5.0),
        }
    return out


def get_sprint_results(year: int, round_num: int) -> Dict[str, dict]:
    """Sprint race: ritorna {pilota: {position, gap_to_leader}}"""
    data = _get(f"{JOLPICA_BASE}/{year}/{round_num}/sprint.json")
    if not data:
        return {}
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return {}
    sprint_list = races[0].get("SprintResults", [])
    out = {}
    best_time = None

    # trova miglior tempo
    for r in sprint_list:
        t = r.get("Time", {}).get("time", "")
        if t and best_time is None:
            best_time = 0.0  # winner = 0 gap

    for r in sprint_list:
        drv = _normalize_driver(r["Driver"]["familyName"])
        pos = int(r.get("position", 99))
        t   = r.get("Time", {}).get("time", "")
        gap_str = r.get("Time", {}).get("time", "")
        # Gap è spesso "+X.XXXs" per chi non è primo
        gap = _parse_gap_string(gap_str) if pos > 1 else 0.0
        out[drv] = {"position": pos, "gap_to_leader": min(gap, 5.0)}
    return out


# ══════════════════════════════════════════════════════════════
# OPENF1  —  Sessioni prove libere
# ══════════════════════════════════════════════════════════════

def get_openf1_sessions(year: int, country_key: str) -> List[dict]:
    """
    Recupera le sessioni disponibili per un evento.
    country_key es: 'Australia', 'Bahrain', ecc.
    """
    data = _get(f"{OPENF1_BASE}/sessions", params={
        "year": year,
        "country_name": country_key,
    })
    if not data:
        return []
    return [{"session_key": s["session_key"],
             "session_name": SESSION_TYPE_MAP.get(s.get("session_name", ""), s.get("session_name", "")),
             "session_type": s.get("session_type", ""),
             "date": s.get("date_start", "")}
            for s in data]


def get_fp_gaps(session_key: int) -> Dict[str, float]:
    """
    Recupera i gap dal leader in una sessione FP (OpenF1).
    Ritorna {driver_name: gap_sec}
    """
    # Recupera piloti della sessione
    drivers_data = _get(f"{OPENF1_BASE}/drivers", params={"session_key": session_key})
    if not drivers_data:
        return {}

    driver_map = {}
    for d in drivers_data:
        code  = d.get("name_acronym", "")
        full  = d.get("full_name", "")
        last  = full.split()[-1] if full else code
        driver_map[d.get("driver_number", 0)] = _normalize_driver(last) or _normalize_driver(code)

    # Recupera i giri — prendi il miglior giro di ogni pilota
    laps_data = _get(f"{OPENF1_BASE}/laps", params={"session_key": session_key})
    if not laps_data:
        return {}

    best_laps: Dict[int, float] = {}
    for lap in laps_data:
        drv_num  = lap.get("driver_number", 0)
        lap_dur  = lap.get("lap_duration")
        if lap_dur is None:
            continue
        try:
            lap_dur = float(lap_dur)
        except:
            continue
        if lap_dur < 60:  # Filtro tempi assurdi
            continue
        if drv_num not in best_laps or lap_dur < best_laps[drv_num]:
            best_laps[drv_num] = lap_dur

    if not best_laps:
        return {}

    leader_time = min(best_laps.values())
    gaps = {}
    for num, t in best_laps.items():
        name = driver_map.get(num, f"Driver{num}")
        gaps[name] = min(t - leader_time, 5.0)

    return gaps


# ══════════════════════════════════════════════════════════════
# LAP TIMES & DRIVER INFO CACHING
# ══════════════════════════════════════════════════════════════

def fetch_lap_times(session_key: int, year: int, round_num: int, driver_number: Optional[int] = None) -> List[dict]:
    """
    Recupera tutti i lap times di una sessione (con caching MongoDB).
    Se driver_number è specificato, filtra solo i giri di quel pilota.
    Ritorna lista di dict con campi OpenF1 + year, round_num.
    """
    db = get_db()
    laps = []
    
    # 1. Cerca in cache (MongoDB)
    if db:
        coll = lap_times_collection(db, year)
        query = {"session_key": session_key}
        if driver_number is not None:
            query["driver_number"] = driver_number
        cached = list(coll.find(query))
        if cached:
            logger.debug(f"Lap times cache hit: session {session_key}, {len(cached)} records")
            return cached
    
    # 2. Cache miss → scarica da OpenF1
    logger.info(f"Scaricando lap times da OpenF1 per sessione {session_key}")
    params = {"session_key": session_key}
    if driver_number is not None:
        params["driver_number"] = driver_number
    raw = _get(f"{OPENF1_BASE}/laps", params=params)
    if not raw:
        logger.warning(f"Nessun lap time disponibile per sessione {session_key}")
        return []
    if isinstance(raw, dict):
        # Se per qualche motivo l'API ritorna un oggetto singolo, lo convertiamo in lista
        raw = [raw]
    assert isinstance(raw, list), "Expected list from OpenF1 API"
    
    # 3. Aggiungi metadati (year, round_num) e normalizza
    for lap in raw:
        lap["year"] = year
        lap["round_num"] = round_num
        # Converti campi nulli in None
        for f in ("lap_duration", "duration_sector_1", "duration_sector_2", "duration_sector_3"):
            if lap.get(f) is None:
                lap[f] = None
        # Flag is_pit_out_lap è booleano già presente
    
    # 4. Salva in cache (se MongoDB disponibile)
    if db:
        coll = lap_times_collection(db, year)
        try:
            # Upsert per evitare duplicati (unique index su session_key+driver_number+lap_number)
            for lap in raw:
                filter_dict = {
                    "session_key": lap["session_key"],
                    "driver_number": lap["driver_number"],
                    "lap_number": lap["lap_number"]
                }
                coll.update_one(filter_dict, {"$set": lap}, upsert=True)
            logger.info(f"Lap times salvati in cache: {len(raw)} record")
        except Exception as e:
            logger.warning(f"Errore salvataggio lap times in cache: {e}")
    
    return raw


def fetch_driver_info(session_key: int, year: int, round_num: int) -> List[dict]:
    """
    Recupera informazioni pilota per una sessione (con caching MongoDB).
    Ritorna lista di dict con campi OpenF1 + year, round_num.
    """
    db = get_db()
    
    # 1. Cerca in cache
    if db:
        coll = driver_info_collection(db, year)
        cached = list(coll.find({"session_key": session_key}))
        if cached:
            logger.debug(f"Driver info cache hit: session {session_key}, {len(cached)} records")
            return cached
    
    # 2. Cache miss → scarica da OpenF1
    logger.info(f"Scaricando driver info da OpenF1 per sessione {session_key}")
    raw = _get(f"{OPENF1_BASE}/drivers", params={"session_key": session_key})
    if not raw:
        logger.warning(f"Nessun driver info disponibile per sessione {session_key}")
        return []
    if isinstance(raw, dict):
        raw = [raw]
    assert isinstance(raw, list), "Expected list from OpenF1 API"
    
    # 3. Aggiungi metadati
    for drv in raw:
        drv["year"] = year
        drv["round_num"] = round_num
    
    # 4. Salva in cache
    if db:
        coll = driver_info_collection(db, year)
        try:
            for drv in raw:
                filter_dict = {
                    "session_key": drv["session_key"],
                    "driver_number": drv["driver_number"]
                }
                coll.update_one(filter_dict, {"$set": drv}, upsert=True)
            logger.info(f"Driver info salvati in cache: {len(raw)} record")
        except Exception as e:
            logger.warning(f"Errore salvataggio driver info in cache: {e}")
    
    return raw


# ══════════════════════════════════════════════════════════════
# HIGH-LEVEL: carica tutto l'evento
# ══════════════════════════════════════════════════════════════

@dataclass
class RealEventData:
    event_name:    str
    circuit:       str
    year:          int
    round_num:     int
    sessions:      Dict[str, Dict[str, float]] = field(default_factory=dict)
    session_keys:  Dict[str, int]              = field(default_factory=dict)  # mappa nome sessione → session_key OpenF1
    grid:          Dict[str, int]              = field(default_factory=dict)
    quali_data:    Dict[str, dict]             = field(default_factory=dict)
    sprint_data:   Dict[str, dict]             = field(default_factory=dict)
    errors:        List[str]                   = field(default_factory=list)
    data_sources:  Dict[str, str]              = field(default_factory=dict)


def fetch_event_data(year: int, round_num: int,
                     country_name: str = "",
                     event_name: str = "") -> RealEventData:
    """
    Recupera TUTTI i dati di sessione per un evento F1.
    Combina OpenF1 (FP) + Jolpica (Quali/Sprint).
    """
    ev = RealEventData(
        event_name=event_name or f"Round {round_num} {year}",
        circuit="",
        year=year,
        round_num=round_num,
    )

    progress = []

    # ─── 1. QUALIFICHE (Jolpica) ───────────────────────────────
    progress.append("📡 Scaricando qualifiche...")
    quali = get_qualifying_results(year, round_num)
    if quali:
        ev.quali_data = quali
        ev.sessions["Quali"] = {d: v["gap_to_leader"] for d, v in quali.items()}
        ev.grid       = {d: v["position"] for d, v in quali.items()}
        ev.data_sources["Quali"] = "Jolpica/Ergast API"
        progress.append(f"  ✅ Qualifiche: {len(quali)} piloti")
    else:
        ev.errors.append("Qualifiche non disponibili")
        progress.append("  ⚠️ Qualifiche: dati non disponibili")

    # ─── 2. SPRINT (Jolpica) ──────────────────────────────────
    sprint = get_sprint_results(year, round_num)
    if sprint:
        ev.sprint_data = sprint
        ev.sessions["Sprint"] = {d: v["gap_to_leader"] for d, v in sprint.items()}
        ev.data_sources["Sprint"] = "Jolpica/Ergast API"
        progress.append(f"  ✅ Sprint: {len(sprint)} piloti")

    # ─── 3. PROVE LIBERE (OpenF1) ─────────────────────────────
    if country_name:
        progress.append("📡 Scaricando sessioni OpenF1...")
        openf1_sessions = get_openf1_sessions(year, country_name)

        fp_sessions = [s for s in openf1_sessions
                       if s["session_name"] in ("FP1", "FP2", "FP3", "Sprint Quali", "Quali", "Sprint")]

        for s in fp_sessions:
            sname = s["session_name"]
            progress.append(f"  📡 {sname} (key={s['session_key']})...")
            gaps = get_fp_gaps(s["session_key"])
            if gaps:
                ev.sessions[sname] = gaps
                ev.session_keys[sname] = s["session_key"]
                ev.data_sources[sname] = "OpenF1 API"
                progress.append(f"  ✅ {sname}: {len(gaps)} piloti")
            else:
                ev.errors.append(f"{sname}: nessun dato disponibile")
                progress.append(f"  ⚠️ {sname}: vuoto")

    ev._progress_log = progress
    return ev


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _time_to_sec(time_str: str) -> Optional[float]:
    """Converte '1:18.518' in secondi (78.518)."""
    if not time_str:
        return None
    try:
        time_str = time_str.strip()
        if ':' in time_str:
            parts = time_str.split(':')
            return float(parts[0]) * 60 + float(parts[1])
        return float(time_str)
    except:
        return None


def _parse_gap_string(s: str) -> float:
    """Converte '+1.234' o '1 Lap' in float secondi."""
    if not s:
        return 5.0
    s = s.strip().lstrip('+')
    try:
        return float(s)
    except:
        return 5.0


def get_available_rounds(year: int = 2026) -> List[dict]:
    """Lista round disponibili con nome GP."""
    return get_season_schedule(year)


def get_sc_history(circuit_name: str) -> List[int]:
    """
    Storico safety car per circuito (hardcoded + da Ergast parzialmente).
    """
    SC_HISTORY = {
        "albert park": [1,1,1,1,0,1,1,1,0,1,1,1],
        "bahrain":     [1,0,1,0,1,0,1,1,0,1,0,1],
        "jeddah":      [1,1,0,1,1,0,1,1],
        "suzuka":      [0,1,0,0,1,0,1,0,1,0],
        "shanghai":    [1,0,1,0,0,1,1,0],
        "miami":       [1,0,1,1,0],
        "monaco":      [1,0,1,1,1,0,1,1,1,1,0,1],
        "montreal":    [1,1,0,1,1,0,1,1,0,1],
        "barcelona":   [0,0,1,0,0,1,0,0,0,0],
        "silverstone": [1,0,1,1,0,1,1,0,1,1],
        "hungaroring": [0,1,0,1,0,0,0,1,0,0],
        "spa":         [1,1,0,1,1,0,1,0,1,1],
        "monza":       [1,1,1,0,1,0,1,1,0,1],
        "marina bay":  [1,1,1,1,1,0,1,1,1,1],
        "cota":        [1,0,1,1,0,1,1,0],
        "hermanos":    [1,0,1,0,1,0,0,1],
        "interlagos":  [1,1,0,1,1,1,0,1,1,1],
        "las vegas":   [0,1,0],
        "losail":      [1,0,1,1],
        "yas marina":  [0,0,0,0,0,1,0,0,0,0],
    }
    name_low = circuit_name.lower()
    for key, hist in SC_HISTORY.items():
        if key in name_low or name_low in key:
            return hist
    return [1,0,1,0,1,0,1,0,1,0]  # default 50%
