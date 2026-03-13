"""
core/analysis.py — Motore centrale: combina dati sessione + ML + quote → predizioni
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from core.odds_parser import OddsEntry, MktType
from ml.models import F1MLEngine, build_driver_features, DEFAULT_CS, DEFAULT_EXP


@dataclass
class Pred:
    entry:    OddsEntry
    lr:       float
    rf:       float
    gb:       float
    ens:      float
    impl:     float          # prob implicita corretta per margine
    edge:     float          # ens - impl
    fair_q:   float          # 1/ens

    @property
    def edge_pct(self): return self.edge * 100
    @property
    def label(self): return self.entry.label
    @property
    def quota(self): return self.entry.quota
    @property
    def verdict(self):
        if self.edge >  0.04: return "🟢 SOTTOSTIMATA"
        if self.edge < -0.05: return "🔴 SOVRASTIMATA"
        return "🟡 EQUA"
    @property
    def color(self):
        if self.edge >  0.04: return "#00e676"
        if self.edge < -0.05: return "#ff1744"
        return "#ffd740"


@dataclass
class Multipla:
    label:  str
    picks:  List[Pred]
    note:   str   = ""
    stake:  float = 2.0

    @property
    def q_comb(self): return float(np.prod([p.quota for p in self.picks]))
    @property
    def win(self): return self.q_comb * self.stake


class AnalysisEngine:

    def __init__(self, engine: F1MLEngine,
                 sessions: Dict[str, Dict[str, float]],
                 grid: Dict[str, int],
                 issues: Dict[str, int] = None,
                 sc_history: List[int] = None):
        self.ml       = engine
        self.sessions = sessions
        self.grid     = grid
        self.issues   = issues or {}
        self.sc_hist  = sc_history or []
        self._groups: Dict[str, List[str]] = {}
        self._group_q: Dict[str, Dict[str,float]] = {}

    def _feat(self, driver: str) -> np.ndarray:
        return build_driver_features(driver, self.sessions, self.grid,
                                     self.issues, DEFAULT_CS, DEFAULT_EXP)

    def _implied_corr(self, entry: OddsEntry, counterpart_q: Optional[float]) -> float:
        raw = 1 / entry.quota
        if counterpart_q:
            tot = raw + 1/counterpart_q
            return raw/tot
        return raw

    def set_groups(self, entries: List[OddsEntry]):
        for e in entries:
            if e.mkt == MktType.GRUPPO and e.group:
                self._groups.setdefault(e.group, [])
                if e.driver1 not in self._groups[e.group]:
                    self._groups[e.group].append(e.driver1)
                self._group_q.setdefault(e.group, {})[e.driver1] = e.quota

    def predict(self, entry: OddsEntry,
                counterpart_q: Optional[float] = None) -> Optional[Pred]:
        d1 = entry.driver1

        try:
            if entry.mkt in (MktType.H2H, MktType.H2H_GIRO):
                d2 = entry.driver2
                if not d2: return None
                f1, f2 = self._feat(d1), self._feat(d2)
                lr,rf,gb,ens = self.ml.predict_h2h(f1, f2)
                impl = self._implied_corr(entry, counterpart_q)

            elif entry.mkt == MktType.GRUPPO:
                members = self._groups.get(entry.group, [])
                if len(members) < 2: return None
                scores = []
                for d2 in members:
                    if d2 != d1:
                        f1,f2 = self._feat(d1), self._feat(d2)
                        _,_,_,e = self.ml.predict_h2h(f1,f2)
                        scores.append(e)
                raw_score = float(np.mean(scores)) if scores else 0.25
                # normalizza nel gruppo
                all_scores = {}
                for m in members:
                    sc = []
                    for m2 in members:
                        if m2!=m:
                            _,_,_,e = self.ml.predict_h2h(self._feat(m),self._feat(m2))
                            sc.append(e)
                    all_scores[m] = float(np.mean(sc)) if sc else 0.25
                tot = sum(all_scores.values())
                ens = all_scores.get(d1,0.25)/tot if tot>0 else 0.25
                lr=rf=gb=ens
                # implied corretta per margine del gruppo
                q_dict = self._group_q.get(entry.group, {})
                raw_all = {d:1/q for d,q in q_dict.items()}
                tot_impl = sum(raw_all.values())
                impl = raw_all.get(d1, 1/entry.quota)/tot_impl if tot_impl>0 else 1/entry.quota

            elif entry.mkt in (MktType.PODIO, MktType.TOP6):
                f = self._feat(d1)
                ens = self.ml.predict_top6(f)
                lr=rf=gb=ens
                impl = 1/entry.quota

            elif entry.mkt == MktType.SC:
                ens = sum(self.sc_hist)/len(self.sc_hist) if self.sc_hist else 0.70
                lr=rf=gb=ens; impl=1/entry.quota

            elif entry.mkt == MktType.VINCITORE:
                # usa posizione in griglia + top6 come proxy
                f = self._feat(d1)
                ens = self.ml.predict_top6(f) * 0.35
                lr=rf=gb=ens; impl=1/entry.quota
            else:
                return None

        except Exception:
            return None

        edge   = ens - impl
        fair_q = 1/ens if ens > 0.01 else 99.0
        return Pred(entry, lr, rf, gb, ens, impl, edge, fair_q)

    def predict_all(self, entries: List[OddsEntry]) -> List[Pred]:
        # Pre-build mappa counterpart per H2H
        cp: Dict[Tuple[str,str], float] = {}
        for e in entries:
            if e.mkt in (MktType.H2H, MktType.H2H_GIRO) and e.driver2:
                key = (e.driver1, e.driver2)
                # il counterpart è l'entry con d1/d2 invertiti
                for e2 in entries:
                    if e2.driver1==e.driver2 and e2.driver2==e.driver1 and e2.mkt==e.mkt:
                        cp[key] = e2.quota
                        break
        preds = []
        for e in entries:
            key = (e.driver1, e.driver2) if e.driver2 else None
            cq  = cp.get(key) if key else None
            p   = self.predict(e, cq)
            if p: preds.append(p)
        return preds

    # ── Multipla builder ──────────────────────────────────────────

    def build_multiples(self, preds: List[Pred], stake: float=2.0,
                         target: float=25.0, n: int=5) -> List[Multipla]:
        from itertools import combinations
        pos = sorted([p for p in preds if p.edge >= -0.01], key=lambda x: x.edge, reverse=True)

        # Deduplicazione H2H (tieni solo il lato con edge maggiore per match)
        seen, deduped = set(), []
        for p in pos:
            if p.entry.mkt in (MktType.H2H, MktType.H2H_GIRO):
                key = tuple(sorted([p.entry.driver1, p.entry.driver2 or '']))
                if key in seen: continue
                seen.add(key)
            deduped.append(p)

        mults = []
        if len(deduped) >= 3:
            mults.append(Multipla('M1 🟢 "I Più Certi"', deduped[:3],
                "Top 3 selezioni per edge ML.", stake))
        if len(deduped) >= 3:
            # cerca combo ~target
            best, bw = None, 0
            for combo in combinations(deduped[:10], 3):
                w = np.prod([p.quota for p in combo])*stake
                if w >= target*0.6 and w > bw:
                    best=combo; bw=w
            if best:
                mults.append(Multipla(f'M2 🟢 "Target €{target:.0f}"', list(best),
                    f"Ottimizzata per vincita ≥ €{target:.0f}.", stake))
        if len(deduped) >= 4:
            mults.append(Multipla('M3 🟡 "4 Pick Edge+"', deduped[:4],
                "Quattro pick tutti con edge positivo.", stake))
        if len(deduped) >= 4:
            # cerca combo 4 pick ~target
            best4, bw4 = None, 0
            for combo in combinations(deduped[:8], 4):
                w = np.prod([p.quota for p in combo])*stake
                if w >= target and w > bw4:
                    best4=combo; bw4=w
            if best4:
                mults.append(Multipla('M4 🟡 "4 Pick Valore"', list(best4),
                    f"4 pick con edge ML positivo, vincita stimata €{bw4:.2f}.", stake))
        if len(deduped) >= 5:
            mults.append(Multipla('M5 🔴 "Max Valore ML"', deduped[:5],
                "5 pick con i migliori edge del palinsesto.", stake))

        return mults[:n]
