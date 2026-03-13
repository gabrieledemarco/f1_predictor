"""
core/results.py
Persistenza dual-mode: MongoDB (produzione) | JSON locale (sviluppo).
"""
import json, time, requests, glob as _glob
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict

def _safe_int(val, default=99):
    if val is None or val == "": return default
    try: return int(val)
    except: return default

def _safe_float(val, default=0.0):
    if val is None or val == "": return default
    try: return float(val)
    except: return default

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
TIMEOUT = 12

@dataclass
class DriverResult:
    driver: str; position: int; grid: int; points: float; status: str; fastest_lap: bool = False
    @property
    def finished(self): return self.position <= 20 and "Retired" not in self.status
    @property
    def podium(self): return self.position <= 3
    @property
    def top6(self): return self.position <= 6

@dataclass
class RaceResult:
    year: int; round_num: int; event_name: str; circuit: str; date: str
    results: Dict[str, DriverResult] = field(default_factory=dict)
    h2h_outcomes: Dict[str, bool] = field(default_factory=dict)

@dataclass
class PredictionRecord:
    year: int; round_num: int; event_name: str
    market: str; label: str; driver1: str; driver2: Optional[str]
    quota: float; implied_prob: float; ensemble_prob: float; edge: float
    outcome: Optional[bool] = None
    outcome_position_d1: Optional[int] = None
    outcome_position_d2: Optional[int] = None
    source: str = "real"
    timestamp: str = ""; session_hash: str = ""; ip_hash: str = ""
    ip_country: str = ""; browser_family: str = ""; os_family: str = ""
    is_mobile: bool = False; language: str = ""; action: str = ""

# ── Jolpica API ───────────────────────────────────────────────────────

def _get(url):
    for i in range(2):
        try:
            r = requests.get(url, timeout=TIMEOUT); r.raise_for_status(); return r.json()
        except:
            if i == 0: time.sleep(1.5)
    return None

def _norm(name):
    M = {"russell":"Russell","antonelli":"Antonelli","leclerc":"Leclerc","hamilton":"Hamilton",
         "piastri":"Piastri","norris":"Norris","hadjar":"Hadjar","verstappen":"Verstappen",
         "lawson":"Lawson","lindblad":"Lindblad","bortoleto":"Bortoleto","hulkenberg":"Hulkenberg",
         "ocon":"Ocon","bearman":"Bearman","gasly":"Gasly","colapinto":"Colapinto",
         "albon":"Albon","sainz":"Sainz","bottas":"Bottas","perez":"Perez",
         "alonso":"Alonso","stroll":"Stroll"}
    return M.get(name.lower(), name.capitalize())

def fetch_race_result(year, round_num):
    data = _get(f"{JOLPICA_BASE}/{year}/{round_num}/results.json")
    if not data: return None
    races = data.get("MRData",{}).get("RaceTable",{}).get("Races",[])
    if not races: return None
    raw = races[0]
    rr = RaceResult(year=year, round_num=round_num, event_name=raw.get("raceName",""),
                    circuit=raw.get("Circuit",{}).get("circuitName",""), date=raw.get("date",""))
    for r in raw.get("Results",[]):
        d = _norm(r["Driver"]["familyName"])
        rr.results[d] = DriverResult(d, _safe_int(r.get("position"),99),
            _safe_int(r.get("grid"),99), _safe_float(r.get("points"),0.0),
            r.get("status","Unknown") or "Unknown",
            r.get("FastestLap",{}).get("rank","99")=="1")
    items = list(rr.results.items())
    for i in range(len(items)):
        for j in range(i+1,len(items)):
            d1,r1=items[i]; d2,r2=items[j]
            rr.h2h_outcomes[f"{d1} vs {d2}"] = r1.position < r2.position
    return rr

def fetch_sprint_result(year, round_num):
    data = _get(f"{JOLPICA_BASE}/{year}/{round_num}/sprint.json")
    if not data: return None
    races = data.get("MRData",{}).get("RaceTable",{}).get("Races",[])
    if not races: return None
    raw = races[0]
    rr = RaceResult(year=year, round_num=round_num, event_name=raw.get("raceName","")+" Sprint",
                    circuit=raw.get("Circuit",{}).get("circuitName",""), date=raw.get("date",""))
    for r in raw.get("SprintResults",[]):
        d = _norm(r["Driver"]["familyName"])
        rr.results[d] = DriverResult(d, _safe_int(r.get("position"),99),
            _safe_int(r.get("grid"),99), _safe_float(r.get("points"),0.0),
            r.get("status","") or "")
    return rr

# ── Valutazione predizioni ────────────────────────────────────────────

def evaluate_predictions(preds, race):
    updated = []
    for p in preds:
        p2 = PredictionRecord(**asdict(p))
        r1 = race.results.get(p.driver1)
        if p.market == "T/T Gara" and p.driver2:
            r2 = race.results.get(p.driver2)
            if r1 and r2:
                p2.outcome = r1.position < r2.position
                p2.outcome_position_d1 = r1.position; p2.outcome_position_d2 = r2.position
        elif p.market in ("Top 6","Top6"):
            if r1: p2.outcome = r1.top6; p2.outcome_position_d1 = r1.position
        elif p.market == "Podio":
            if r1: p2.outcome = r1.podium; p2.outcome_position_d1 = r1.position
        elif p.market == "Migliore Gruppo":
            if r1: p2.outcome_position_d1 = r1.position
        updated.append(p2)
    return updated

# ── Record conversion ─────────────────────────────────────────────────

def _to_doc(r):
    d = asdict(r); d.pop("_id", None); return d

def _from_doc(doc):
    doc = {k:v for k,v in doc.items() if k != "_id"}
    doc.setdefault("source","real")
    known = set(PredictionRecord.__dataclass_fields__)
    try:
        return PredictionRecord(**{k:v for k,v in doc.items() if k in known})
    except Exception:
        return None

# ── MongoDB backend ───────────────────────────────────────────────────

def _mongo_save(db, records):
    from pymongo import UpdateOne
    from core.db import collection_for_year
    by_year = {}
    for r in records:
        by_year.setdefault(r.year, []).append(r)
    saved = 0
    for yr, recs in by_year.items():
        coll = collection_for_year(db, yr)
        ops = []
        for r in recs:
            if not r.timestamp: r.timestamp = datetime.now(timezone.utc).isoformat()
            ops.append(UpdateOne(
                {"year":r.year,"round_num":r.round_num,"label":r.label},
                {"$setOnInsert": _to_doc(r)}, upsert=True))
        if ops:
            saved += coll.bulk_write(ops, ordered=False).upserted_count
    return saved

def _mongo_load(db, year=None, filters=None):
    from core.db import collection_for_year, list_prediction_collections
    q = filters or {}
    out = []
    if year:
        for doc in collection_for_year(db,year).find(q,{"_id":0}).sort("round_num",1):
            r = _from_doc(doc)
            if r: out.append(r)
    else:
        for cn in list_prediction_collections(db):
            for doc in db[cn].find(q,{"_id":0}).sort([("year",1),("round_num",1)]):
                r = _from_doc(doc)
                if r: out.append(r)
    return out

def _mongo_delete(db, year=None, round_num=None, source=None):
    from core.db import collection_for_year, list_prediction_collections
    q = {}
    if source:    q["source"]    = source
    if round_num: q["round_num"] = round_num
    deleted = 0
    if year:
        deleted += collection_for_year(db,year).delete_many(q).deleted_count
    else:
        for cn in list_prediction_collections(db):
            deleted += db[cn].delete_many(q).deleted_count
    return deleted

# ── JSON backend ──────────────────────────────────────────────────────

def _json_path(year=None):
    return f"data/history_{year}.json" if year else None

def _json_load_path(path):
    if not path or not Path(path).exists(): return []
    try:
        with open(path) as f: raw = json.load(f)
    except: return []
    out = []
    for d in raw:
        d.setdefault("source","real")
        r = _from_doc(d)
        if r: out.append(r)
    return out

def _json_load(year=None):
    if year:
        return _json_load_path(_json_path(year))
    out = []
    for p in sorted(_glob.glob("data/history_*.json")):
        out.extend(_json_load_path(p))
    return out

def _json_save(records, year=None):
    by_year = {}
    for r in records: by_year.setdefault(r.year,[]).append(r)
    saved = 0
    for yr, recs in by_year.items():
        path = _json_path(yr)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        existing = _json_load(yr)
        seen = {(r.year,r.round_num,r.label) for r in existing}
        new = [r for r in recs if (r.year,r.round_num,r.label) not in seen]
        for r in new:
            if not r.timestamp: r.timestamp = datetime.now(timezone.utc).isoformat()
        with open(path,"w") as f: json.dump([asdict(r) for r in existing+new], f, indent=2)
        saved += len(new)
    return saved

def _json_delete(year=None, round_num=None, source=None, path=None):
    paths = [path] if path else ([_json_path(year)] if year else sorted(_glob.glob("data/history_*.json")))
    deleted = 0
    for p in (paths or []):
        if not p or not Path(p).exists(): continue
        recs = _json_load_path(p)
        kept = []
        for r in recs:
            rm = True
            if source    is not None and r.source    != source:    rm = False
            if round_num is not None and r.round_num != round_num: rm = False
            if not rm: kept.append(r)
        deleted += len(recs)-len(kept)
        with open(p,"w") as f: json.dump([asdict(r) for r in kept], f, indent=2)
    return deleted

# ── Public API ────────────────────────────────────────────────────────

def save_records(records, db=None, year=None, path=None):
    if not records: return 0
    if db is not None: return _mongo_save(db, records)
    return _json_save(records, year)

def load_records(db=None, year=None, path=None, filters=None):
    if db is not None: return _mongo_load(db, year=year, filters=filters)
    if path: return _json_load_path(path)
    return _json_load(year)

def delete_records(db=None, year=None, round_num=None, source=None, path=None):
    if db is not None: return _mongo_delete(db, year=year, round_num=round_num, source=source)
    return _json_delete(year=year, round_num=round_num, source=source, path=path)

def get_real_records(db=None, year=None):
    return [r for r in load_records(db=db,year=year) if getattr(r,"source","real")=="real"]

def history_summary(db=None, year=None):
    all_r = load_records(db=db, year=year)
    real  = [r for r in all_r if getattr(r,"source","real")=="real"]
    demo  = [r for r in all_r if getattr(r,"source","real")=="demo"]
    return {"total":len(all_r), "real":len(real),
            "real_evaluated":sum(1 for r in real if r.outcome is not None),
            "demo":len(demo), "demo_evaluated":sum(1 for r in demo if r.outcome is not None)}

def list_history_files():
    return sorted(_glob.glob("data/history_*.json"))

def _history_path(year=None):
    return _json_path(year)

def get_evaluated_records(db=None, year=None):
    return [r for r in load_records(db=db,year=year) if r.outcome is not None]

# ── Retention policy ─────────────────────────────────────────────────

_TELE_FIELDS = ["ip_hash","session_hash","browser_family","os_family"]

def apply_retention_policy(max_days=90, db=None, year=None, path=None):
    cutoff = datetime.now(timezone.utc).timestamp() - max_days*86400
    cleaned = 0
    if db is not None:
        from core.db import list_prediction_collections
        colls = ([f"predictions_{year}"] if year else list_prediction_collections(db))
        for cn in colls:
            coll = db[cn]
            for doc in coll.find({"timestamp":{"$ne":""}},{"_id":1,"timestamp":1}):
                try:
                    ts = datetime.fromisoformat(doc["timestamp"].rstrip("Z")).timestamp()
                    if ts < cutoff:
                        coll.update_one({"_id":doc["_id"]},{"$set":{f:"" for f in _TELE_FIELDS}})
                        cleaned += 1
                except: pass
    else:
        paths = [path] if path else ([_json_path(year)] if year else sorted(_glob.glob("data/history_*.json")))
        for p in (paths or []):
            if not p or not Path(p).exists(): continue
            recs = _json_load_path(p); changed = False
            for r in recs:
                if not r.timestamp: continue
                try:
                    ts = datetime.fromisoformat(r.timestamp.rstrip("Z")).timestamp()
                    if ts < cutoff:
                        for f in _TELE_FIELDS: setattr(r,f,"")
                        cleaned += 1; changed = True
                except: pass
            if changed:
                with open(p,"w") as f: json.dump([asdict(r) for r in recs], f, indent=2)
    return cleaned

# ── Migration JSON → MongoDB ──────────────────────────────────────────

def migrate_json_to_mongo(db, dry_run=False):
    files = list_history_files()
    if not files: return {"files":0,"records":0,"inserted":0}
    total_r = total_i = 0
    detail = []
    for fp in files:
        recs = _json_load_path(fp); total_r += len(recs)
        n = 0
        if not dry_run and recs: n = _mongo_save(db, recs)
        total_i += n
        detail.append({"file":fp,"records":len(recs),"inserted":n})
    return {"files":len(files),"records":total_r,"inserted":total_i,"detail":detail}