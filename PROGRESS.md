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

## PR #2 — Fix import TracingInsights: redesign per struttura flat CSV
**Branch:** `claude/goofy-chatelet`  
**Data:** 2026-04-13  
**Stato:** In progress (commit 3/3)

### Obiettivo
Correggere il workflow `import-tracinginsights.yml` che importava 0 lap,
rendendo `f1_lap_times` sempre vuota e invalidando l'analisi delle feature di telemetria.

### Root cause analisi (finale — dopo 3 run diagnostici)

**Assunzione sbagliata sulla struttura del repo**  
Lo script assumeva che TracingInsights/RaceData avesse una struttura per-circuito:
```
RaceData/data/{year}/{Circuit}/laps.csv
```
ma il repo usa invece lo schema **Ergast flat CSV**:
```
RaceData/data/
    lap_times.csv     ← TUTTI i giri di TUTTI gli anni (raceId, driverId, lap, ms)
    races.csv         ← raceId → year, round, circuitId
    drivers.csv       ← driverId → driverRef/code
    circuits.csv      ← circuitId → circuitRef
    results.csv       ← raceId, driverId, constructorId (per enrichment team)
```
→ Il `rglob("laps.csv")` non trovava nulla perché il file si chiama `lap_times.csv`
  ed è uno solo per l'intero dataset storico (2019-2025).

### Fix implementati (riscrittura completa)

| File | Modifica |
|------|----------|
| `scripts/import_tracinginsights.py` | **Riscrittura completa**: lettura flat `lap_times.csv` con join su `races.csv`/`drivers.csv`/`circuits.csv` |
| `scripts/import_tracinginsights.py` | `find_data_dir()`: auto-localizza la directory con i CSV flat |
| `scripts/import_tracinginsights.py` | `load_lookup_tables()`: carica race_info, driver_code_map, circuit_ref_map |
| `scripts/import_tracinginsights.py` | `import_flat_lap_times()`: import con filtro anno range, skip già importati |
| `scripts/import_tracinginsights.py` | `enrich_teams_from_results()`: arricchisce il campo `team` da `results.csv` |
| `scripts/import_tracinginsights.py` | Bulk write a batch di 2000 per performance |
| `scripts/compute_pace_observations.py` | Supporto `--min-year`/`--max-year` via env `MIN_YEAR`/`MAX_YEAR` |
| `.github/workflows/import-tracinginsights.yml` | Input `min_year`/`max_year` al posto di `year` |
| `.github/workflows/import-tracinginsights.yml` | Outputs `min_year`/`max_year` passati al job compute-pace |

### Feature attese dopo import corretto

| Feature | Sorgente | Note |
|---------|---------|------|
| `avg_lap_ms`, `min_lap_ms`, `std_lap_ms` | f1_lap_times | Disponibili in lap_times.csv (milliseconds col) |
| `lap_consistency` (std/avg) | f1_lap_times | Calcolabile da lap_time_ms |
| `pace_delta_ms` (aggiornato) | f1_pace_observations | Già rank #1, qualità migliorata |
| `soft_pct`, `medium_pct`, `hard_pct` | **NON disponibili** | lap_times.csv non ha dati gomme; servirebbero FastF1 o altro |
| `avg_tyre_life`, `max_tyre_life` | **NON disponibili** | Stessa limitazione |
| `pb_rate` | **NON disponibili** | lap_times.csv non ha flag personal best |

---

## Prossimi step

- [ ] Verificare run workflow post-fix: `f1_lap_times` deve essere > 0
- [ ] Re-run `feature-analysis.yml` e verificare MAE < 4.0 posizioni
- [ ] Import odds (`import-pinnacle-odds.yml`) — feature `odds_p_novig` attesa rank #1
- [ ] Escludere `year` e `round` dall'analisi (temporal leakage)
- [ ] Aggiungere `qualifying` features da `import-jolpica.yml`
