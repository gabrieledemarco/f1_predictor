"""
core/session_tracker.py
Raccoglie informazioni di sessione in modo privacy-safe per BetBreaker.

Principi GDPR applicati:
  - IP conservato SOLO come hash SHA-256 irreversibile (non è più dato personale diretto)
  - Nessuna correlazione cross-session senza consenso
  - Retention limit: 90 giorni (applicato dall'admin tool)
  - User-Agent troncato alle info tecnicamente rilevanti
  - Nessuna profilazione individuale
"""
import hashlib
import datetime
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
# DATA MODEL
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SessionInfo:
    """
    Snapshot della sessione utente al momento dell'azione.
    Tutti i dati potenzialmente identificativi sono anonimizzati.
    """
    # Temporalità
    timestamp:       str   = ""       # ISO 8601, UTC
    date_only:       str   = ""       # YYYY-MM-DD — per aggregazioni giornaliere

    # Sessione (non persistente tra riavvii del browser)
    session_id:      str   = ""       # UUID della sessione Streamlit (tab-level)
    session_hash:    str   = ""       # SHA-256(session_id) — per deduplicazione

    # Rete (anonimizzata)
    ip_hash:         str   = ""       # SHA-256(ip) — non reversibile
    ip_country:      str   = ""       # paese approssimativo da GeoIP (es. "IT")
    ip_region:       str   = ""       # regione approssimativa (es. "Emilia-Romagna")

    # Browser / client
    user_agent_raw:  str   = ""       # User-Agent completo (solo in admin)
    browser_family:  str   = ""       # "Chrome", "Firefox", "Safari", "Mobile", ecc.
    os_family:       str   = ""       # "Windows", "macOS", "Linux", "Android", "iOS"
    language:        str   = ""       # Accept-Language principale (es. "it", "en")
    is_mobile:       bool  = False

    # Contesto app
    action:          str   = ""       # "load_session", "run_ml", "evaluate", "export"
    gp_year:         Optional[int] = None
    gp_round:        Optional[int] = None

    # Flag qualità
    is_bot_suspected: bool = False    # User-Agent suggerisce bot/crawler


def _sha256(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha256(value.encode()).hexdigest()[:16]  # primi 16 hex — sufficiente per dedup


def _parse_ua(ua: str) -> dict:
    """Estrae browser e OS dal User-Agent senza librerie esterne."""
    ua_lower = ua.lower()

    # Browser
    if "edg/" in ua_lower or "edge/" in ua_lower:
        browser = "Edge"
    elif "chrome/" in ua_lower and "chromium" not in ua_lower:
        browser = "Chrome"
    elif "firefox/" in ua_lower:
        browser = "Firefox"
    elif "safari/" in ua_lower and "chrome" not in ua_lower:
        browser = "Safari"
    elif "opera/" in ua_lower or "opr/" in ua_lower:
        browser = "Opera"
    elif "curl" in ua_lower or "python" in ua_lower or "bot" in ua_lower or "crawler" in ua_lower:
        browser = "Bot/Script"
    else:
        browser = "Other"

    # OS
    if "windows" in ua_lower:
        os_f = "Windows"
    elif "macintosh" in ua_lower or "mac os" in ua_lower:
        os_f = "macOS"
    elif "android" in ua_lower:
        os_f = "Android"
    elif "iphone" in ua_lower or "ipad" in ua_lower:
        os_f = "iOS"
    elif "linux" in ua_lower:
        os_f = "Linux"
    else:
        os_f = "Other"

    is_mobile = any(x in ua_lower for x in ["android","iphone","ipad","mobile","tablet"])
    is_bot    = any(x in ua_lower for x in ["bot","crawler","spider","scraper","curl","python-requests","wget"])

    return {"browser": browser, "os": os_f, "is_mobile": is_mobile, "is_bot": is_bot}


def _geoip_lookup(ip: str) -> dict:
    """
    Lookup GeoIP leggero tramite ipapi.co (gratuito, 1000 req/giorno).
    Ritorna solo paese e regione — nessun dato preciso di localizzazione.
    Fallisce silenziosamente se offline o rate-limited.
    """
    import urllib.request, json, socket
    if not ip or ip in ("127.0.0.1", "::1", "unknown"):
        return {"country": "local", "region": ""}
    try:
        # Timeout aggressivo — non bloccare l'app
        url = f"https://ipapi.co/{ip}/json/"
        req = urllib.request.Request(url, headers={"User-Agent": "BetBreaker/1.0"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
        return {
            "country": data.get("country_code", ""),
            "region":  data.get("region", ""),
        }
    except Exception:
        return {"country": "", "region": ""}


# ══════════════════════════════════════════════════════════════════════
# MAIN COLLECTOR
# ══════════════════════════════════════════════════════════════════════

def collect_session_info(action: str = "generic",
                          gp_year: int = None,
                          gp_round: int = None,
                          do_geoip: bool = False) -> SessionInfo:
    """
    Raccoglie le informazioni di sessione disponibili nell'ambiente Streamlit.
    Chiamata una volta per azione rilevante (es. "run_ml", "evaluate").

    do_geoip=True esegue il lookup GeoIP — aggiunge ~2s, usare con parsimonia.
    """
    import streamlit as st

    now = datetime.datetime.utcnow()
    info = SessionInfo(
        timestamp  = now.isoformat() + "Z",
        date_only  = now.strftime("%Y-%m-%d"),
        action     = action,
        gp_year    = gp_year,
        gp_round   = gp_round,
    )

    # ── Session ID ────────────────────────────────────────────────────
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        if ctx:
            raw_sid = str(ctx.session_id)
            info.session_id   = raw_sid[:8] + "…"  # troncato — non conservare completo
            info.session_hash = _sha256(raw_sid)
    except Exception:
        info.session_id   = "unavailable"
        info.session_hash = ""

    # ── Headers (Streamlit ≥ 1.37) ────────────────────────────────────
    raw_ip = ""
    raw_ua = ""
    raw_lang = ""
    try:
        headers = st.context.headers          # dict-like
        # IP: Streamlit Cloud passa X-Forwarded-For
        raw_ip = (headers.get("X-Forwarded-For", "") or
                  headers.get("X-Real-Ip", "") or
                  headers.get("Remote-Addr", "")).split(",")[0].strip()
        raw_ua   = headers.get("User-Agent", "")
        raw_lang = headers.get("Accept-Language", "").split(",")[0].split(";")[0].strip()
    except AttributeError:
        # Streamlit < 1.37 — fallback silenzioso
        pass
    except Exception:
        pass

    # ── IP anonimizzato ───────────────────────────────────────────────
    if raw_ip:
        info.ip_hash = _sha256(raw_ip)
        if do_geoip:
            geo = _geoip_lookup(raw_ip)
            info.ip_country = geo.get("country", "")
            info.ip_region  = geo.get("region", "")

    # ── User-Agent parsing ────────────────────────────────────────────
    if raw_ua:
        info.user_agent_raw = raw_ua[:200]   # troncato a 200 char
        parsed = _parse_ua(raw_ua)
        info.browser_family   = parsed["browser"]
        info.os_family        = parsed["os"]
        info.is_mobile        = parsed["is_mobile"]
        info.is_bot_suspected = parsed["is_bot"]

    # ── Lingua ────────────────────────────────────────────────────────
    info.language = raw_lang[:5] if raw_lang else ""

    return info


def session_info_to_dict(info: SessionInfo, admin_view: bool = False) -> dict:
    """
    Serializza SessionInfo.
    admin_view=True include user_agent_raw (default False — non finisce in history_*.json).
    """
    d = asdict(info)
    if not admin_view:
        d.pop("user_agent_raw", None)   # non salvare UA completo nello storico pubblico
    return d
