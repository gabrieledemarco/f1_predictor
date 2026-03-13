"""
core/db.py
Layer di connessione MongoDB per BetBreaker.

Strategia dual-mode:
  - PRODUZIONE (Streamlit Cloud): legge MONGO_URI da st.secrets
  - SVILUPPO LOCALE: legge MONGO_URI da .env oppure usa fallback JSON
  - ADMIN TOOL: usa get_db_direct() senza dipendere da Streamlit

Struttura Atlas:
  Database  : betbreaker          (configurabile via MONGO_DB)
  Collezioni: predictions_2024
              predictions_2025
              predictions_2026    ← una per anno → isolamento completo
              telemetry           ← sessioni aggregate (separato dai record)

Indice univoco su ogni collezione: { year, round_num, label }
"""

from __future__ import annotations
import os
from typing import Optional

# ── Costanti ──────────────────────────────────────────────────────────
DEFAULT_DB_NAME = "betbreaker"
_COLLECTION_PREFIX = "predictions_"
_LAPTIMES_PREFIX = "lap_times_"
_DRIVERINFO_PREFIX = "driver_info_"
_SESSIONSTATS_PREFIX = "session_stats_"


# ══════════════════════════════════════════════════════════════════════
# CONNESSIONE STREAMLIT  (usa @st.cache_resource — singleton per app)
# ══════════════════════════════════════════════════════════════════════

def get_db():
    """
    Ritorna il database MongoDB.
    Chiamata dall'app Streamlit — usa @st.cache_resource internamente.
    Se MONGO_URI non è configurato ritorna None → il layer di persistenza
    cade automaticamente sul fallback JSON locale.
    """
    import streamlit as st

    @st.cache_resource
    def _connect():
        uri = _resolve_uri()
        if not uri:
            return None
        try:
            from pymongo import MongoClient
            from pymongo.server_api import ServerApi
            client = MongoClient(
                uri,
                server_api=ServerApi("1"),
                serverSelectionTimeoutMS=5_000,   # fallisce veloce se Atlas irraggiungibile
                connectTimeoutMS=5_000,
            )
            # Ping per verificare la connessione al primo avvio
            client.admin.command("ping")
            db_name = _resolve_db_name()
            return client[db_name]
        except Exception as e:
            # Non crashare l'app — logga e ritorna None → JSON fallback
            import logging
            logging.warning(f"[BetBreaker] MongoDB non disponibile: {e}. Uso fallback JSON.")
            return None

    return _connect()


# ══════════════════════════════════════════════════════════════════════
# CONNESSIONE DIRETTA  (admin tool, script, test — senza Streamlit)
# ══════════════════════════════════════════════════════════════════════

def get_db_direct(uri: str = None, db_name: str = None):
    """
    Connessione diretta a MongoDB — usata dall'admin tool e dai test.
    Non dipende da Streamlit.

    Priorità URI:
      1. Parametro esplicito
      2. Variabile d'ambiente MONGO_URI
      3. File .env nella root del progetto
      4. None → fallback JSON
    """
    resolved_uri = uri or _resolve_uri()
    if not resolved_uri:
        return None
    try:
        from pymongo import MongoClient
        from pymongo.server_api import ServerApi
        client = MongoClient(
            resolved_uri,
            server_api=ServerApi("1"),
            serverSelectionTimeoutMS=5_000,
        )
        client.admin.command("ping")
        return client[db_name or _resolve_db_name()]
    except Exception as e:
        print(f"  WARNING  MongoDB non raggiungibile: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# OPERAZIONI COLLEZIONE
# ══════════════════════════════════════════════════════════════════════

def collection_for_year(db, year: int):
    """
    Ritorna la collezione MongoDB per l'anno dato.
    La crea automaticamente (MongoDB è schema-less) e assicura l'indice univoco.
    """
    coll_name = f"{_COLLECTION_PREFIX}{year}"
    coll = db[coll_name]
    # Indice univoco — evita duplicati senza logica manuale
    _ensure_index(coll)
    return coll


def _ensure_index(coll):
    """Crea l'indice univoco se non esiste. Idempotente."""
    from pymongo import ASCENDING
    existing = {idx["name"] for idx in coll.list_indexes()}
    if "unique_prediction" not in existing:
        coll.create_index(
            [("year", ASCENDING), ("round_num", ASCENDING), ("label", ASCENDING)],
            unique=True,
            name="unique_prediction",
            background=True,
        )


def lap_times_collection(db, year: int):
    """
    Ritorna la collezione lap_times per l'anno dato.
    Indici: {year, round_num, session_key, driver_number} per query veloci.
    """
    coll_name = f"{_LAPTIMES_PREFIX}{year}"
    coll = db[coll_name]
    _ensure_lap_times_index(coll)
    return coll


def driver_info_collection(db, year: int):
    """
    Ritorna la collezione driver_info per l'anno dato.
    Indici: {year, round_num, session_key, driver_number} per join con lap_times.
    """
    coll_name = f"{_DRIVERINFO_PREFIX}{year}"
    coll = db[coll_name]
    _ensure_driver_info_index(coll)
    return coll


def session_stats_collection(db, year: int):
    """
    Ritorna la collezione session_stats per l'anno dato.
    Indici: {year, round_num, session_name, driver} per statistiche rapide.
    """
    coll_name = f"{_SESSIONSTATS_PREFIX}{year}"
    coll = db[coll_name]
    _ensure_session_stats_index(coll)
    return coll


def _ensure_lap_times_index(coll):
    """Crea indici per query efficienti sui lap times."""
    from pymongo import ASCENDING
    existing = {idx["name"] for idx in coll.list_indexes()}
    
    # Indice univoco per evitare duplicati (session_key + driver_number + lap_number)
    if "lap_times_unique" not in existing:
        coll.create_index(
            [
                ("session_key", ASCENDING),
                ("driver_number", ASCENDING),
                ("lap_number", ASCENDING)
            ],
            unique=True,
            name="lap_times_unique",
            background=True
        )
    
    # Indice composto per query più comuni: anno + round + sessione + pilota
    if "lap_times_main" not in existing:
        coll.create_index(
            [
                ("year", ASCENDING),
                ("round_num", ASCENDING), 
                ("session_key", ASCENDING),
                ("driver_number", ASCENDING),
                ("lap_number", ASCENDING)
            ],
            name="lap_times_main",
            background=True
        )
    
    # Indice per ricerca per session_key (usato spesso)
    if "session_key_idx" not in existing:
        coll.create_index([("session_key", ASCENDING)], name="session_key_idx", background=True)
    
    # Indice per driver_number (analisi per pilota)
    if "driver_number_idx" not in existing:
        coll.create_index([("driver_number", ASCENDING)], name="driver_number_idx", background=True)


def _ensure_driver_info_index(coll):
    """Crea indici per info pilota."""
    from pymongo import ASCENDING
    existing = {idx["name"] for idx in coll.list_indexes()}
    
    # Indice univoco: anno + round + sessione + pilota (un record per pilota per sessione)
    if "driver_info_unique" not in existing:
        coll.create_index(
            [
                ("year", ASCENDING),
                ("round_num", ASCENDING),
                ("session_key", ASCENDING),
                ("driver_number", ASCENDING)
            ],
            unique=True,
            name="driver_info_unique",
            background=True
        )
    
    # Indice per team_name (aggregazioni per scuderia)
    if "team_name_idx" not in existing:
        coll.create_index([("team_name", ASCENDING)], name="team_name_idx", background=True)


def _ensure_session_stats_index(coll):
    """Crea indici per statistiche di sessione."""
    from pymongo import ASCENDING
    existing = {idx["name"] for idx in coll.list_indexes()}
    
    # Indice per query più comuni: anno + round + sessione + pilota
    if "session_stats_main" not in existing:
        coll.create_index(
            [
                ("year", ASCENDING),
                ("round_num", ASCENDING),
                ("session_name", ASCENDING),
                ("driver", ASCENDING)
            ],
            name="session_stats_main",
            background=True
        )
    
    # Indice per session_name (filter rapido)
    if "session_name_idx" not in existing:
        coll.create_index([("session_name", ASCENDING)], name="session_name_idx", background=True)


def list_prediction_collections(db) -> list[str]:
    """Elenca le collezioni predictions_YYYY presenti nel database."""
    return sorted([
        name for name in db.list_collection_names()
        if name.startswith(_COLLECTION_PREFIX)
    ])


def collection_year_from_name(name: str) -> Optional[int]:
    """'predictions_2026' → 2026"""
    try:
        return int(name.replace(_COLLECTION_PREFIX, ""))
    except ValueError:
        return None


# ══════════════════════════════════════════════════════════════════════
# HELPERS PRIVATI
# ══════════════════════════════════════════════════════════════════════

def _resolve_uri() -> Optional[str]:
    """
    Risolve MONGO_URI nell'ordine:
      1. st.secrets["MONGO_URI"]         ← Streamlit Cloud / secrets.toml locale
      2. os.environ["MONGO_URI"]          ← variabile d'ambiente
      3. .env file nella project root     ← sviluppo locale
    """
    # 1. Streamlit secrets (non fallisce se non siamo in Streamlit)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "MONGO_URI" in st.secrets:
            return st.secrets["MONGO_URI"]
    except Exception:
        pass

    # 2. Env var diretta
    uri = os.environ.get("MONGO_URI")
    if uri:
        return uri

    # 3. .env file
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("MONGO_URI="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")

    return None


def _resolve_db_name() -> str:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "MONGO_DB" in st.secrets:
            return st.secrets["MONGO_DB"]
    except Exception:
        pass
    return os.environ.get("MONGO_DB", DEFAULT_DB_NAME)


def is_mongo_available() -> bool:
    """Controlla se MongoDB è configurato (senza aprire connessione)."""
    return bool(_resolve_uri())