"""
core/odds_parser.py — Parse quote da testo libero o immagine (Claude Vision)
"""
import re, json, base64
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class MktType(Enum):
    H2H      = "T/T Gara"
    H2H_GIRO = "T/T 1° Giro"
    GRUPPO   = "Migliore Gruppo"
    PODIO    = "Podio"
    TOP6     = "Top 6"
    VINCITORE= "Vincitore"
    SC       = "Safety Car"


@dataclass
class OddsEntry:
    mkt:       MktType
    label:     str
    driver1:   str
    driver2:   Optional[str]
    quota:     float
    group:     Optional[str] = None


def _clean(name: str) -> str:
    name = str(name).strip().rstrip('.')
    name = re.sub(r'\s+[A-Z]\.$','',name).strip()
    name = re.sub(r'\s+[A-Z]\s*$','',name).strip()
    return ' '.join(p.capitalize() for p in name.split()) if name else name


def parse_text(text: str) -> List[OddsEntry]:
    entries = []
    ctx = MktType.H2H

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            low = line.lower()
            if 'primo giro' in low: ctx = MktType.H2H_GIRO
            elif 't/t' in low or 'h2h' in low or 'testa' in low: ctx = MktType.H2H
            continue

        low = line.lower()
        if 'primo giro' in low and ':' not in low: ctx = MktType.H2H_GIRO; continue
        if any(k in low for k in ['t/t gara','h2h gara','testa a testa']): ctx = MktType.H2H; continue

        # ── GRUPPO ──
        grp = re.match(r'(?:gruppo\s*|g)(\d+)\s*(?:gp[\s\w]*)?\s*[:\-]\s*(.+)', line, re.I)
        if grp:
            gn = f"G{grp.group(1)}"
            pairs = re.findall(r'([A-Za-zÀ-ú][A-Za-zÀ-ú\s\.]+?)\s*[=:]\s*(\d+[\.,]\d+)', grp.group(2))
            if not pairs:
                pairs = re.findall(r'([A-Za-zÀ-ú][A-Za-zÀ-ú]+)\s+(\d+[\.,]\d+)', grp.group(2))
            for d,q in pairs:
                d = _clean(d); q = float(q.replace(',','.'))
                entries.append(OddsEntry(MktType.GRUPPO, f"{d} migliore {gn}", d, None, q, gn))
            continue

        # ── T/T full ──
        tt = re.match(
            r'([A-Za-zÀ-ú][A-Za-zÀ-ú\s\.]*)(?:\s+vs?\.?\s+|\s*[-–]\s*)([A-Za-zÀ-ú][A-Za-zÀ-ú\s\.]*)'
            r'[:\s]+(\d+[\.,]\d+)\s*[/|\\]\s*(\d+[\.,]\d+)', line, re.I)
        if tt:
            d1=_clean(tt.group(1)); d2=_clean(tt.group(2))
            q1=float(tt.group(3).replace(',','.')); q2=float(tt.group(4).replace(',','.'))
            entries.append(OddsEntry(ctx, f"{d1} batte {d2}", d1, d2, q1))
            entries.append(OddsEntry(ctx, f"{d2} batte {d1}", d2, d1, q2))
            continue

        # ── Singolo ──
        sg = re.match(
            r'([A-Za-zÀ-ú][A-Za-zÀ-ú\s\.]+?)\s+(podio|top\s*6|vincitore|safety\s*car)\s*[:\-]?\s*(\d+[\.,]\d+)',
            line, re.I)
        if sg:
            d=_clean(sg.group(1)); mk=sg.group(2).lower(); q=float(sg.group(3).replace(',','.'))
            if 'podio' in mk:     entries.append(OddsEntry(MktType.PODIO,   f"{d} Podio",     d, None, q))
            elif 'top' in mk:     entries.append(OddsEntry(MktType.TOP6,    f"{d} Top6",      d, None, q))
            elif 'safety' in mk:  entries.append(OddsEntry(MktType.SC,      "Safety Car Sì",  "SC", None, q))
            else:                  entries.append(OddsEntry(MktType.VINCITORE,f"{d} Vince",    d, None, q))
    return entries


def parse_image_with_claude(image_bytes: bytes, mime: str = "image/jpeg",
                              api_key: str = None) -> List[OddsEntry]:
    """Usa Claude Vision per estrarre le quote da screenshot."""
    try:
        import anthropic
    except ImportError:
        return []
    if not api_key:
        import os; api_key = os.environ.get("ANTHROPIC_API_KEY","")
    if not api_key:
        return []

    b64 = base64.standard_b64encode(image_bytes).decode()
    client = anthropic.Anthropic(api_key=api_key)
    prompt = """Analizza questo screenshot di un'app di scommesse (Eurobet/Snai/Sisal) ed estrai TUTTE le quote visibili.
Restituisci SOLO JSON, nessun altro testo, nessun markdown:
{"markets":[
  {"type":"H2H_GARA","d1":"Leclerc","d2":"Norris","q1":1.25,"q2":3.50},
  {"type":"MIGLIORE_GRUPPO","group":"G1","entries":[{"driver":"Russell","quota":1.25},{"driver":"Antonelli","quota":4.50}]},
  {"type":"TOP6","driver":"Hamilton","quota":1.55},
  {"type":"PODIO","driver":"Antonelli","quota":1.60},
  {"type":"SAFETY_CAR","quota":1.55}
]}
Types: H2H_GARA, H2H_GIRO, MIGLIORE_GRUPPO, PODIO, TOP6, VINCITORE, SAFETY_CAR"""

    resp = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=2000,
        messages=[{"role":"user","content":[
            {"type":"image","source":{"type":"base64","media_type":mime,"data":b64}},
            {"type":"text","text":prompt}
        ]}]
    )
    raw = resp.content[0].text.strip()
    raw = re.sub(r'^```[a-z]*\s*','',raw); raw = re.sub(r'```\s*$','',raw)
    try:
        parsed = json.loads(raw)
    except:
        return []
    return _json_to_entries(parsed)


def _json_to_entries(parsed: dict) -> List[OddsEntry]:
    TM = {"H2H_GARA":MktType.H2H,"H2H_GIRO":MktType.H2H_GIRO,
          "MIGLIORE_GRUPPO":MktType.GRUPPO,"PODIO":MktType.PODIO,
          "TOP6":MktType.TOP6,"VINCITORE":MktType.VINCITORE,"SAFETY_CAR":MktType.SC}
    out = []
    for m in parsed.get("markets",[]):
        t = TM.get(m.get("type",""), MktType.H2H)
        if t in (MktType.H2H, MktType.H2H_GIRO):
            d1,d2 = m.get("d1",""),m.get("d2","")
            q1,q2 = float(m.get("q1",2)),float(m.get("q2",2))
            out.append(OddsEntry(t,f"{d1} batte {d2}",d1,d2,q1))
            out.append(OddsEntry(t,f"{d2} batte {d1}",d2,d1,q2))
        elif t == MktType.GRUPPO:
            gn = m.get("group","G?")
            for e in m.get("entries",[]):
                d,q = e.get("driver",""),float(e.get("quota",3))
                out.append(OddsEntry(t,f"{d} migliore {gn}",d,None,q,gn))
        elif t in (MktType.PODIO,MktType.TOP6,MktType.VINCITORE):
            d,q = m.get("driver",""),float(m.get("quota",3))
            out.append(OddsEntry(t,f"{d} {t.value}",d,None,q))
        elif t == MktType.SC:
            out.append(OddsEntry(t,"Safety Car Sì","SC",None,float(m.get("quota",1.60))))
    return out
