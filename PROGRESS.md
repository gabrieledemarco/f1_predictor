# Progress Log — Feature Significance Analysis Pipeline

Questo file documenta il lavoro svolto da Claude Code per sistemare e migliorare
la pipeline di analisi della significatività delle feature F1 su MongoDB.

---

## PR #1 — Fix NaN composite_score e grid_position mancante
**Branch:** `claude/goofy-chatelet`  
**Data:** 2026-04-13  
**Stato:** Merged pending

### Obiettivo
Correggere il crash del workflow `feature-analysis.yml` e i problemi di qualità dati
che impedivano una corretta misurazione delle feature.

### Bug risolti

| Bug | File | Fix |
|-----|------|-----|
| `ValueError: could not convert string to float: ''` nel workflow summary | `.github/workflows/feature-analysis.yml:105` | Gestione stringa vuota con `float(x) if x.strip() else 0.0` |
| `scipy.stats.spearmanr()` restituisce NaN per feature costanti (es. circuit one-hot) | `scripts/feature_analysis.py:485` | Guard `if not np.isfinite(rho)` prima di `results.append()` |
| `rank_dict()` propaga NaN quando tutti i valori sono NaN | `scripts/feature_analysis.py:633` | Sanitizza NaN→0 e protegge da `max_v=0/NaN` |
| `composite_score` NaN nei dati finali | `scripts/feature_analysis.py:668` | Clamp `if not np.isfinite(composite): composite = 0.0` |
| `grid_position` costantemente 0 (campo mancante in `f1_races.results[]`) | `scripts/feature_analysis.py:118` | Fallback su `qual_lookup[driver_code]` |

### Anomalie rilevate nel report
- **f1_lap_times empty** → 32 feature su 48 mancanti (da correggere in PR #2)
- **Odds mancanti** → feature più predittiva assente
- **`ct_street` MI anomalo** → MI inflazionata su binary feature rara (non un bug)
- **`round`/`year` in top 5** → temporal leakage (da escludere dal modello)
- **RF MAE = 5.08 posizioni** → atteso ≤ 4.0 con dataset completo

---

## PR #2 — Fix import TracingInsights: path, circuit mapping, multi-year
**Branch:** `claude/goofy-chatelet`  
**Data:** 2026-04-13  
**Stato:** In progress

### Obiettivo
Correggere il workflow `import-tracinginsights.yml` che importava 0 lap per tre bug distinti,
rendendo `f1_lap_times` sempre vuota e invalidando l'analisi delle feature di telemetria.

### Root cause analisi

**1. Path sbagliato** — Bug principale  
Il repository TracingInsights/RaceData ha struttura:
```
RaceData/
    data/           ← subdirectory NON documentata
        2024/
            Bahrain/laps.csv
        2023/
            ...
```
Ma lo script cercava `data/racedata/{year}/` invece di `data/racedata/data/{year}/`.  
→ `"Anni disponibili: ['data']"` — il log confermava il problema.

**2. Circuit name mapping mancante**  
Lo script usava `folder_name.lower().replace(" ", "_")` come `circuit_ref`, ma f1_races
usa i codici Jolpica/Ergast:
- `Saudi_Arabia` → cercava `saudi_arabia` ma il DB ha `jeddah`  
- `Australia` → cercava `australia` ma il DB ha `albert_park`  
- ecc. per tutti i 24 circuiti

→ `"Warning: No race found for {year} {circuit_ref}, skipping"` per ogni circuito.

**3. Import single-year**  
Lo script accettava solo un anno (default: anno corrente = 2026) ma il repo
TracingInsights non ha ancora dati 2026.  
→ `"[WARNING] Year 2026 not found in racedata"` e exit(0) immediato.

### Fix implementati

| File | Modifica |
|------|----------|
| `scripts/import_tracinginsights.py` | Auto-detect path via `resolve_racedata_root()` |
| `scripts/import_tracinginsights.py` | Aggiunto `FOLDER_TO_CIRCUIT_REF` dict con 24+5 circuiti |
| `scripts/import_tracinginsights.py` | Supporto `--min-year`/`--max-year` con loop su anni disponibili |
| `scripts/import_tracinginsights.py` | Colonne CSV rilevate dinamicamente (robusto a varianti di nome) |
| `scripts/import_tracinginsights.py` | Compound normalizzato (SOFT/MEDIUM/HARD/UNKNOWN) |
| `scripts/import_tracinginsights.py` | Filtro outlier lap_time: `0 < ms < 300_000` |
| `scripts/compute_pace_observations.py` | Supporto `--min-year`/`--max-year` via env `MIN_YEAR`/`MAX_YEAR` |
| `.github/workflows/import-tracinginsights.yml` | Input `min_year`/`max_year` al posto di `year` |
| `.github/workflows/import-tracinginsights.yml` | Outputs `min_year`/`max_year` passati al job compute-pace |

### Feature attese dopo import corretto

| Feature | Sorgente | Impatto atteso |
|---------|---------|----------------|
| `avg_lap_ms`, `min_lap_ms`, `std_lap_ms` | f1_lap_times | MAE da 5.08 → ~3.5 posizioni |
| `lap_consistency` (std/avg) | f1_lap_times | Spearman rho atteso ~0.25-0.35 |
| `soft_pct`, `medium_pct`, `hard_pct` | f1_lap_times | Feature strategia gomme |
| `avg_tyre_life`, `max_tyre_life` | f1_lap_times | Indice di gestione gomme |
| `pb_rate` | f1_lap_times | Personal best rate per driver |
| `pace_delta_ms` (aggiornato) | f1_pace_observations | Già rank #1, qualità migliorata |

---

## Prossimi step

- [ ] Verificare run workflow post-fix: `f1_lap_times` deve essere > 0
- [ ] Re-run `feature-analysis.yml` e verificare MAE < 4.0 posizioni
- [ ] Import odds (`import-pinnacle-odds.yml`) — feature `odds_p_novig` attesa rank #1
- [ ] Escludere `year` e `round` dall'analisi (temporal leakage)
- [ ] Aggiungere `qualifying` features da `import-jolpica.yml`
