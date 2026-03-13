"""
core/session_stats.py
Calcolo statistiche dettagliate per sessione (piloti, scuderie, giri validi/invalidi).
Integrazione con cache MongoDB per performance.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import defaultdict

from core.f1_api import fetch_lap_times, fetch_driver_info

logger = logging.getLogger("session_stats")

# ─── Costanti per identificazione giri invalidi ──────────────────────────────
MAX_LAP_DURATION = 300.0          # oltre 300s → giro cancellato/safety car
MIN_LAP_DURATION = 60.0           # sotto 60s → errore di misura
NULL_SECTOR_THRESHOLD = 2         # se almeno 2 settori nulli → giro invalido

def identify_invalid_laps(lap_records: List[dict]) -> Tuple[List[dict], List[dict]]:
    """
    Separa giri validi da invalidi basandosi su pattern osservati:
      - lap_duration = null
      - duration_sector_3 = null (spesso indica giro interrotto)
      - lap_duration > MAX_LAP_DURATION (giri under safety car / cancellati)
      - lap_duration < MIN_LAP_DURATION (misura impossibile)
      - is_pit_out_lap = True (giro uscita box → escludi dalle statistiche)
      - settori nulli (se almeno NULL_SECTOR_THRESHOLD settori sono null)
    
    Ritorna (valid_laps, invalid_laps).
    """
    valid = []
    invalid = []
    
    for lap in lap_records:
        lap_dur = lap.get("lap_duration")
        sector1 = lap.get("duration_sector_1")
        sector2 = lap.get("duration_sector_2")
        sector3 = lap.get("duration_sector_3")
        pit_out = lap.get("is_pit_out_lap", False)
        
        # Controllo nulli
        null_sectors = sum(1 for s in (sector1, sector2, sector3) if s is None)
        
        invalid_reason = None
        if lap_dur is None:
            invalid_reason = "lap_duration null"
        elif sector3 is None:
            invalid_reason = "sector3 null"
        elif lap_dur > MAX_LAP_DURATION:
            invalid_reason = f"lap_duration > {MAX_LAP_DURATION}s"
        elif lap_dur < MIN_LAP_DURATION:
            invalid_reason = f"lap_duration < {MIN_LAP_DURATION}s"
        elif pit_out:
            invalid_reason = "pit out lap"
        elif null_sectors >= NULL_SECTOR_THRESHOLD:
            invalid_reason = f"{null_sectors} null sectors"
        
        if invalid_reason:
            lap["_invalid_reason"] = invalid_reason
            invalid.append(lap)
        else:
            valid.append(lap)
    
    logger.debug(f"Giri totali: {len(lap_records)} → validi: {len(valid)}, invalidi: {len(invalid)}")
    return valid, invalid


def aggregate_lap_times_by_driver(lap_records: List[dict]) -> Dict[int, Dict[str, Any]]:
    """
    Aggrega i lap times per driver_number.
    Ritorna dict: driver_number -> statistiche.
    """
    from collections import defaultdict
    import numpy as np
    
    driver_laps = defaultdict(list)
    for lap in lap_records:
        drv = lap["driver_number"]
        driver_laps[drv].append(lap)
    
    stats = {}
    for drv, laps in driver_laps.items():
        durations = [l["lap_duration"] for l in laps if l["lap_duration"] is not None]
        if not durations:
            continue
        
        # Statistiche base
        stats[drv] = {
            "driver_number": drv,
            "total_laps": len(laps),
            "valid_laps": len(durations),
            "min_time": float(np.min(durations)),
            "max_time": float(np.max(durations)),
            "mean_time": float(np.mean(durations)),
            "median_time": float(np.median(durations)),
            "std_time": float(np.std(durations)),
            "q1": float(np.percentile(durations, 25)),
            "q3": float(np.percentile(durations, 75)),
            "iqr": float(np.percentile(durations, 75) - np.percentile(durations, 25)),
            "laps": laps,  # riferimento ai record originali (per grafici)
        }
    return stats


def aggregate_sector_times_by_driver(lap_records: List[dict]) -> Dict[int, Dict[str, Any]]:
    """
    Aggrega i tempi dei settori per driver_number.
    Ritorna dict: driver_number -> statistiche per settore.
    """
    from collections import defaultdict
    import numpy as np
    
    driver_laps = defaultdict(list)
    for lap in lap_records:
        drv = lap["driver_number"]
        driver_laps[drv].append(lap)
    
    stats = {}
    for drv, laps in driver_laps.items():
        # Raccoglie tempi validi per ogni settore
        sector1_times = [l["duration_sector_1"] for l in laps if l.get("duration_sector_1") is not None]
        sector2_times = [l["duration_sector_2"] for l in laps if l.get("duration_sector_2") is not None]
        sector3_times = [l["duration_sector_3"] for l in laps if l.get("duration_sector_3") is not None]
        
        # Calcola statistiche per ogni settore
        sector_stats = {}
        for i, (sector_name, times) in enumerate([
            ("sector1", sector1_times),
            ("sector2", sector2_times),
            ("sector3", sector3_times)
        ], 1):
            if not times:
                sector_stats[sector_name] = None
                continue
                
            sector_stats[sector_name] = {
                "min": float(np.min(times)),
                "max": float(np.max(times)),
                "mean": float(np.mean(times)),
                "median": float(np.median(times)),
                "std": float(np.std(times)),
                "q1": float(np.percentile(times, 25)),
                "q3": float(np.percentile(times, 75)),
                "count": len(times),
            }
        
        # Calcola contributo percentuale medio dei settori al tempo totale
        total_times = []
        s1_contrib = []
        s2_contrib = []
        s3_contrib = []
        
        for lap in laps:
            if (lap.get("duration_sector_1") is not None and 
                lap.get("duration_sector_2") is not None and 
                lap.get("duration_sector_3") is not None and
                lap.get("lap_duration") is not None):
                total = lap["lap_duration"]
                s1 = lap["duration_sector_1"]
                s2 = lap["duration_sector_2"]
                s3 = lap["duration_sector_3"]
                # Verifica che la somma sia vicina al totale (entro 0.5s)
                if abs((s1 + s2 + s3) - total) < 0.5:
                    total_times.append(total)
                    s1_contrib.append(s1 / total * 100)
                    s2_contrib.append(s2 / total * 100)
                    s3_contrib.append(s3 / total * 100)
        
        contribution_stats = {}
        if total_times:
            contribution_stats = {
                "sector1_pct_mean": float(np.mean(s1_contrib)),
                "sector2_pct_mean": float(np.mean(s2_contrib)),
                "sector3_pct_mean": float(np.mean(s3_contrib)),
                "sector1_pct_std": float(np.std(s1_contrib)),
                "sector2_pct_std": float(np.std(s2_contrib)),
                "sector3_pct_std": float(np.std(s3_contrib)),
            }
        
        stats[drv] = {
            "driver_number": drv,
            "total_laps": len(laps),
            "valid_sector1": len(sector1_times),
            "valid_sector2": len(sector2_times),
            "valid_sector3": len(sector3_times),
            "sector_stats": sector_stats,
            "contribution_stats": contribution_stats if contribution_stats else None,
        }
    
    return stats


def aggregate_lap_times_by_team(lap_records: List[dict],
                                driver_info: List[dict]) -> Dict[str, Dict[str, Any]]:
    """
    Aggrega i lap times per scuderia (team_name).
    Richiede driver_info per mappare driver_number -> team_name.
    """
    # Mappa driver -> team
    driver_to_team = {}
    for drv in driver_info:
        driver_to_team[drv["driver_number"]] = drv.get("team_name", "Unknown")
    
    # Raggruppa giri per team
    team_laps = defaultdict(list)
    for lap in lap_records:
        drv = lap["driver_number"]
        team = driver_to_team.get(drv, "Unknown")
        team_laps[team].append(lap)
    
    stats = {}
    for team, laps in team_laps.items():
        durations = [l["lap_duration"] for l in laps if l["lap_duration"] is not None]
        if not durations:
            continue
        
        stats[team] = {
            "team_name": team,
            "total_laps": len(laps),
            "valid_laps": len(durations),
            "min_time": float(np.min(durations)),
            "max_time": float(np.max(durations)),
            "mean_time": float(np.mean(durations)),
            "median_time": float(np.median(durations)),
            "std_time": float(np.std(durations)),
            "drivers": list({l["driver_number"] for l in laps}),
            "laps": laps,
        }
    return stats


def compute_session_stats(session_key: int, year: int, round_num: int) -> Dict[str, Any]:
    """
    Calcola statistiche complete per una sessione.
    Ritorna dict con:
      - session_key, year, round_num
      - driver_stats: dict per pilota (driver_number -> stats)
      - team_stats: dict per scuderia (team_name -> stats)
      - lap_summary: total_laps, valid_laps, invalid_laps, pit_out_laps
      - invalid_breakdown: conteggio per motivo
      - raw_lap_count: numero record grezzi
    """
    logger.info(f"Calcolo statistiche sessione {session_key} ({year} round {round_num})")
    
    # 1. Recupera dati (con cache)
    all_laps = fetch_lap_times(session_key, year, round_num)
    driver_info = fetch_driver_info(session_key, year, round_num)
    
    if not all_laps:
        logger.warning(f"Nessun lap time per sessione {session_key}")
        return {}
    
    # 2. Identifica giri invalidi
    valid_laps, invalid_laps = identify_invalid_laps(all_laps)
    
    # 3. Statistiche per pilota (solo giri validi)
    driver_stats = aggregate_lap_times_by_driver(valid_laps)
    
    # 4. Statistiche per settore per pilota
    sector_stats = aggregate_sector_times_by_driver(valid_laps)
    
    # 5. Statistiche per scuderia (solo giri validi)
    team_stats = aggregate_lap_times_by_team(valid_laps, driver_info)
    
    # 6. Riepilogo invalidi
    invalid_breakdown = defaultdict(int)
    for lap in invalid_laps:
        reason = lap.get("_invalid_reason", "unknown")
        invalid_breakdown[reason] += 1
    
    # 7. Conta pit-out laps (anche se validi)
    pit_out_laps = [lap for lap in all_laps if lap.get("is_pit_out_lap", False)]
    
    result = {
        "session_key": session_key,
        "year": year,
        "round_num": round_num,
        "driver_stats": driver_stats,
        "sector_stats": sector_stats,
        "team_stats": team_stats,
        "lap_summary": {
            "total_laps": len(all_laps),
            "valid_laps": len(valid_laps),
            "invalid_laps": len(invalid_laps),
            "pit_out_laps": len(pit_out_laps),
        },
        "invalid_breakdown": dict(invalid_breakdown),
        "raw_lap_count": len(all_laps),
        "valid_laps": valid_laps,
        "driver_info": driver_info,
    }
    
    logger.info(f"Statistiche calcolate: {len(valid_laps)} giri validi, {len(driver_stats)} piloti, {len(team_stats)} scuderie")
    return result


def get_session_stats_cached(session_key: int, year: int, round_num: int,
                             force_refresh: bool = False) -> Dict[str, Any]:
    """
    Versione con cache MongoDB delle statistiche già calcolate.
    Se force_refresh=True, ricalcola e aggiorna cache.
    """
    from core.db import get_db, session_stats_collection
    
    db = get_db()
    if not db or force_refresh:
        # Nessuna cache disponibile o refresh forzato
        return compute_session_stats(session_key, year, round_num)
    
    coll = session_stats_collection(db, year)
    cached = coll.find_one({
        "session_key": session_key,
        "year": year,
        "round_num": round_num
    })
    
    if cached:
        logger.debug(f"Session stats cache hit per sessione {session_key}")
        # Rimuovi _id per consistenza
        cached.pop("_id", None)
        return cached
    
    # Cache miss → calcola e salva
    stats = compute_session_stats(session_key, year, round_num)
    if stats:
        try:
            coll.update_one(
                {"session_key": session_key, "year": year, "round_num": round_num},
                {"$set": stats},
                upsert=True
            )
            logger.info(f"Statistiche salvate in cache per sessione {session_key}")
        except Exception as e:
            logger.warning(f"Errore salvataggio cache statistiche: {e}")
    
    return stats


def get_driver_display_name(driver_number: int, driver_info: List[dict]) -> str:
    """Mappa driver_number a nome visualizzabile."""
    for drv in driver_info:
        if drv["driver_number"] == driver_number:
            return drv.get("full_name", drv.get("name_acronym", str(driver_number)))
    return str(driver_number)


def get_team_colour(team_name: str, driver_info: List[dict]) -> str:
    """Restituisce il colore della scuderia (esadecimale)."""
    for drv in driver_info:
        if drv.get("team_name") == team_name and drv.get("team_colour"):
            return drv["team_colour"]
    return "#888888"  # grigio default