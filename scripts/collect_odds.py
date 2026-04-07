"""
scripts/collect_odds.py
=======================
TASK 4.3 — Raccolta automatica quote Pinnacle via The Odds API.

Eseguire PRIMA di ogni gara (giovedi/venerdi GP weekend) per accumulare
le osservazioni necessarie all'attivazione del Layer 4 (calibratore
isotonic — richiede >= 100 record in data/pinnacle_odds/).

Uso:
    python scripts/collect_odds.py
    python scripts/collect_odds.py --market outrights
    python scripts/collect_odds.py --market h2h
    python scripts/collect_odds.py --out-dir data/pinnacle_odds
    python scripts/collect_odds.py --all-markets --mongo

Requisiti:
    pip install requests python-dotenv
    THE_ODDS_API_KEY nel file .env o come variabile d'ambiente

Output (senza --mongo):
    data/pinnacle_odds/odds_YYYYMMDD_HHMM.jsonl

Note Layer 4:
    Una volta raggiunte 100+ osservazioni, il calibratore isotonic
    si attiva automaticamente al successivo run di train_pipeline.py.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def load_api_key() -> str:
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("THE_ODDS_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    key = os.environ.get("THE_ODDS_API_KEY", "")
    if key:
        return key
    print("ERROR: THE_ODDS_API_KEY non trovata nel .env o nelle variabili d'ambiente.")
    print("  Aggiungi al .env: THE_ODDS_API_KEY=la_tua_chiave")
    sys.exit(1)


def fetch_odds(api_key: str, market: str = "outrights") -> list[dict]:
    import requests
    url = "https://api.the-odds-api.com/v4/sports/motorsport_formula_one/odds"
    params = {
        "apiKey":     api_key,
        "regions":    "eu",
        "markets":    market,
        "oddsFormat": "decimal",
        "bookmakers": "pinnacle",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 401:
            print("ERROR: API key non valida o scaduta.")
        elif resp.status_code == 422:
            print("INFO: Nessuna gara F1 disponibile al momento (fuori stagione?).")
            return []
        else:
            print(f"ERROR HTTP {resp.status_code}: {e}")
        return []
    except Exception as e:
        print(f"ERROR fetching odds: {e}")
        return []

    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    print(f"  [API] Requests usate: {used} | Rimanenti: {remaining}")

    records = []
    now = datetime.now(timezone.utc).isoformat()
    for event in raw:
        event_name = event.get("sport_title", "F1")
        commence = event.get("commence_time", "")
        for bookmaker in event.get("bookmakers", []):
            if bookmaker.get("key") != "pinnacle":
                continue
            for mkt in bookmaker.get("markets", []):
                mkt_key = mkt.get("key", "")
                for outcome in mkt.get("outcomes", []):
                    records.append({
                        "event":     event_name,
                        "commence":  commence,
                        "market":    mkt_key,
                        "driver":    outcome.get("name", ""),
                        "odds":      outcome.get("price", 0.0),
                        "bookmaker": "pinnacle",
                        "timestamp": now,
                        "p_implied": round(1.0 / outcome.get("price", 1e9), 6)
                                     if outcome.get("price", 0) > 0 else 0.0,
                    })
    return records


def save_records(records: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"odds_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"
    with open(fname, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return fname


def count_existing_records(out_dir: Path, db=None) -> int:
    if db is not None:
        try:
            import sys
            from pathlib import Path as _Path
            sys.path.insert(0, str(_Path(__file__).parent.parent))
            from core.db import odds_records_collection
            return odds_records_collection(db).count_documents({})
        except Exception:
            pass
    total = 0
    for jsonl_file in out_dir.glob("*.jsonl"):
        try:
            with open(jsonl_file) as f:
                total += sum(1 for _ in f)
        except Exception:
            pass
    return total


def save_to_mongo(records: list[dict]) -> int:
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent.parent))
    from core.db import get_db_direct, odds_records_collection
    db = get_db_direct()
    if db is None:
        print("ERROR: MongoDB non raggiungibile. Verifica MONGODB_URI nel .env.")
        return 0
    coll = odds_records_collection(db)
    inserted = 0
    for rec in records:
        filter_key = {
            # "commence" identifica univocamente l'evento (ISO timestamp GP start).
            # Non usiamo race_id perché in questo script vale sempre 0.
            "commence":    rec.get("commence", ""),
            "driver":      rec.get("driver", ""),
            "market":      rec.get("market", ""),
            "timestamp":   rec.get("timestamp", ""),
        }
        result = coll.update_one(filter_key, {"$setOnInsert": rec}, upsert=True)
        if result.upserted_id:
            inserted += 1
    return inserted


def main():
    parser = argparse.ArgumentParser(
        description="Raccolta quote Pinnacle F1 via The Odds API"
    )
    parser.add_argument("--market", default="outrights",
                        choices=["outrights", "h2h"])
    parser.add_argument("--out-dir", default="data/pinnacle_odds")
    parser.add_argument("--all-markets", action="store_true",
                        help="Scarica sia outrights che h2h")
    parser.add_argument("--mongo", action="store_true",
                        help="Salva su MongoDB invece di JSONL su disco")
    args = parser.parse_args()

    api_key = load_api_key()
    out_dir = Path(args.out_dir)
    markets = ["outrights", "h2h"] if args.all_markets else [args.market]
    all_records = []

    for mkt in markets:
        print(f"Scaricando {mkt}...")
        records = fetch_odds(api_key, market=mkt)
        all_records.extend(records)
        if len(markets) > 1:
            time.sleep(1)

    if not all_records:
        print("Nessun record scaricato. Verifica connessione o calendario F1.")
        return

    if args.mongo:
        n_inserted = save_to_mongo(all_records)
        db_obj = None
        try:
            import sys as _sys
            from pathlib import Path as _Path
            _sys.path.insert(0, str(_Path(__file__).parent.parent))
            from core.db import get_db_direct
            db_obj = get_db_direct()
        except Exception:
            pass
        total_existing = count_existing_records(out_dir, db=db_obj)
        print(f"\n{'='*50}")
        print(f"  Salvati su MongoDB: {n_inserted} nuovi record")
        print(f"  Totale odds_records su MongoDB: {total_existing}")
    else:
        fname = save_records(all_records, out_dir)
        total_existing = count_existing_records(out_dir)
        print(f"\n{'='*50}")
        print(f"  Salvati: {len(all_records)} record → {fname.name}")
        print(f"  Totale osservazioni in {out_dir}: {total_existing}")

    print()
    if total_existing < 100:
        print(f"  [Layer 4] {100 - total_existing} osservazioni mancanti al target 100.")
    else:
        print(f"  [Layer 4] Soglia 100+ raggiunta ({total_existing} obs). ")
        print(f"  Esegui: python train_pipeline.py ... per attivare il calibratore.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
