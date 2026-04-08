# F1 Predictor - Data Architecture Mapping

**Ultimo aggiornamento:** 2026-04-08  
**Stato:** In progress - TracingInsights import FUNZIONANTE ✅

---

## Obiettivo

Unificare la gestione dati del progetto F1 Predictor su MongoDB Atlas come singola fonte della verità, sostituendo progressivamente i file loader locali.

---

## Panoramica Collezioni MongoDB

### Stato Attuale

| Collection | Count | Status | Fonte Dati |
|------------|-------|--------|------------|
| `f1_races` | 176 | ✅ Pieno | Jolpica API |
| `f1_driver_standings` | 176 | ✅ Pieno | Jolpica API |
| `f1_lap_times` | 38,657 | ⚠️ Parziale | TracingInsights (26,657 correct + 12,000 old) |
| `f1_pace_observations` | **240** | ✅ Calcolato | Computed from lap_times |
| `f1_pinnacle_odds` | **0** | ❌ Vuoto | The Odds API |
| `f1_circuit_profiles` | **0** | ❌ Vuoto | FastF1 extraction |
| `model_versions` | **0** | ❌ Vuoto | Train pipeline |
| `jolpica_cache` | 39 | ✅ | Jolpica API (cache) |
| `f1_import_log` | 178+ | ✅ | Workflows |

### Note Importazione TracingInsights

- Workflow `import-tracinginsights.yml` funzionante ✅
- 26,657 lap times importati per il 2025
- 350 constructor-race combinations aggregate
- **DA FARE:** Pulire 12,000 vecchi lap times con `circuit_ref = circuit_*`

### Collection Schema

#### f1_races
```json
{
  "_id": "2019_01",
  "year": 2019,
  "round": 1,
  "circuit_ref": "albert_park",
  "circuit_name": "Australian Grand Prix",
  "date": "2019-03-17",
  "time": "05:10:00Z",
  "race_name": "Australian Grand Prix",
  "circuit_type": "street",
  "location": {
    "country": "Australia",
    "locality": "Melbourne",
    "lat": -37.8497,
    "lng": 144.968
  },
  "is_sprint_weekend": false,
  "is_major_regulation_change": false,
  "is_season_end": false,
  "qualifying": [],
  "results": [
    {
      "driver_code": "HAM",
      "driver_id": "hamilton",
      "constructor_ref": "mercedes",
      "grid_position": 1,
      "finish_position": 1,
      "points": 26,
      "laps_completed": 58,
      "status": "Finished",
      "fastest_lap_rank": 2,
      "fastest_lap_time": "1:26.057"
    }
  ],
  "source": "jolpica",
  "imported_at": "2026-04-07T15:10:49Z"
}
```

#### f1_driver_standings
```json
{
  "_id": "2019_01",
  "year": 2019,
  "round": 1,
  "date": "2019-03-17",
  "race_name": "Australian Grand Prix",
  "circuit_ref": "albert_park",
  "standings": [
    {
      "position": 1,
      "driver_code": "HAM",
      "driver_id": "hamilton",
      "constructor_refs": ["mercedes"],
      "points": 26,
      "wins": 1,
      "position_text": "1"
    }
  ],
  "source": "jolpica",
  "imported_at": "2026-04-07T15:10:49Z"
}
```

#### f1_lap_times
```json
{
  "_id": "2025_R1_albert_park_NOR_L1",
  "year": 2025,
  "round": 1,
  "circuit_ref": "albert_park",
  "driver_code": "NOR",
  "lap_number": 1,
  "position": 1,
  "lap_time_ms": 117099,
  "imported_at": "2026-04-08T07:54:14Z",
  "source": "tracinginsights"
}
```

#### f1_pace_observations
```json
{
  "_id": "2025_R1_mclaren_albert_park_race_1",
  "year": 2025,
  "round": 1,
  "circuit_ref": "albert_park",
  "constructor_ref": "mclaren",
  "session_type": "race",
  "fuel_corrected_pace_ms": 83250,
  "pace_uncertainty": 120,
  "sample_size": 48,
  "is_reliable": true,
  "imported_at": "2026-04-08T08:00:00Z"
}
```

#### f1_pinnacle_odds
```json
{
  "_id": "race_2025_01_HAM_win",
  "race_id": "2025_01",
  "market": "winner",
  "driver_code": "HAM",
  "odds": 3.45,
  "implied_probability": 0.2899,
  "bookmaker": "pinnacle",
  "fetched_at": "2026-04-07T14:30:00Z"
}
```

#### f1_circuit_profiles
```json
{
  "_id": "albert_park",
  "circuit_type": "street",
  "characteristics": {
    "overtaking_difficulty": 5,
    "braking_severity": 6,
    "downforce_level": "high",
    "top_speed_factor": 0.9
  },
  "segments": [],
  "default_aero_map": "high_downforce",
  "default_compound_strategy": ["medium", "hard"],
  "imported_at": "2026-04-08T00:00:00Z"
}
```

---

## Pipeline Layers - Mappatura Dati

```
Raw Data Sources (Jolpica + TracingInsights + The Odds API + FastF1)
         │
         ▼
┌─────────────────────────────┐
│  Layer 1a — Driver Skill    │  TrueSkill Through Time (TTT)
│  Fonte: f1_races.results     │  Output: driver μ, σ per circuit type
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Layer 1b — Machine Pace    │  Kalman Filter per constructor
│  Fonte: f1_pace_observations│  Output: constructor pace per circuit
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Layer 2 — Race Simulation  │  Bayesian Monte Carlo (50k sims)
│  Fonte: L1a + L1b outputs  │  Output: raw win probabilities
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Layer 3 — Ensemble        │  Ridge meta-learner (α = 10.0)
│  Fonte: L1 + L2 + features │  Output: adjusted probabilities
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Layer 4 — Calibration     │  Isotonic vs Pinnacle odds
│  Fonte: f1_pinnacle_odds   │  Output: calibrated probabilities
└──────────────┬──────────────┘
               │
               ▼
       Edge Report + EV / Kelly
```

### Dipendenze Dettagliate

| Layer | Modulo | File | Input Collection | Output |
|-------|--------|------|-------------------|--------|
| 1a | `driver_skill.py` | `f1_predictor/models/` | `f1_races` | DriverSkillState |
| 1b | `machine_pace.py` | `f1_predictor/models/` | `f1_pace_observations` | ConstructorPaceState |
| 2 | `bayesian_race.py` | `f1_predictor/models/` | L1a + L1b outputs | RaceSimulation |
| 3 | `ensemble.py` | `f1_predictor/models/` | L1 + L2 + features | EnsembleProb |
| 4 | `isotonic.py` | `f1_predictor/calibration/` | `f1_pinnacle_odds` | CalibratedProb |

---

## Fonti Dati Esterne

### 1. Jolpica API (Ergast)
- **URL:** https://api.jolpica-f1.com/
- **Uso:** Race results, standings, qualifying
- **Dati:** Races (1950+), results, driver/constructor standings
- **Workflow:** `import-jolpica.yml`
- **Collection:** `f1_races`, `f1_driver_standings`

### 2. TracingInsights RaceData
- **URL:** https://github.com/TracingInsights/RaceData
- **Formato:** CSV (Ergast-style)
- **Dati:** Lap times storici (1950+), ~17MB
- **URL Raw:** https://raw.githubusercontent.com/TracingInsights/RaceData/main/data/
- **Files:**
  - `lap_times.csv` - raceId, driverId, lap, position, time, milliseconds
  - `races.csv` - raceId, year, round, circuitId, name, date
  - `drivers.csv` - driverId, driverRef, code, forename, surname
  - `circuits.csv` - circuitId, circuitRef, name, country
  - `results.csv` - raceId, driverId, constructorId, position, points
- **Workflow:** `import-tracinginsights.yml`
- **Collection:** `f1_lap_times`

### 3. TracingInsights Anno-Specifico
- **Repo:** https://github.com/TracingInsights/2025, /2026, etc.
- **Dati:** Telemetria avanzata (speed, throttle, brake, DRS)
- **Files per GP:**
  - `telR.py` - Race telemetry
  - `telFP1.py`, `telFP2.py`, `telFP3.py` - Practice telemetry
  - `telQ.py` - Qualifying telemetry
- **Uso:** Circuit profiles, advanced pace analysis

### 4. The Odds API (Pinnacle)
- **URL:** https://the-odds-api.com/
- **API Key:** `THE_ODDS_API_KEY` (GitHub Secret)
- **Uso:** Pre-race win probabilities
- **Mercati:** winner, podium, head-to-head
- **Workflow:** `import-pinnacle-odds.yml`
- **Collection:** `f1_pinnacle_odds`

### 5. FastF1
- **URL:** https://github.com/the-orangeman/FastF1
- **Uso:** Circuit profiles extraction
- **Script:** `extract_circuit_profiles.py`
- **Collection:** `f1_circuit_profiles`

---

## Workflows GitHub Actions

### Workflows Esistenti

| Workflow | Trigger | Jobs | Status |
|----------|---------|------|--------|
| `retrain.yml` | Manual + Schedule (Lun 06:00 UTC) | retrain, quality-gate | ✅ Funziona |
| `import-jolpica.yml` | Schedule (02:00 UTC) | import-races, import-standings | ✅ Funziona |

### Workflows da Implementare/Correggere

| Workflow | Trigger | Jobs | Status |
|----------|---------|------|--------|
| `import-tracinginsights.yml` | Manual + Schedule (Mer 03:00 UTC) | sync-and-import, compute-pace | ⚠️ Da correggere |
| `import-pinnacle-odds.yml` | Manual + Schedule (Ven 18:00 UTC) | fetch-odds | ❌ Da creare |
| `import-circuit-profiles.yml` | Manual + Schedule (Annuale) | extract-profiles | ❌ Da creare |
| `import-kaggle.yml` | Manual + Schedule (Mensile) | import-historical | ❌ Da creare |

---

## Script Import

| Script | Fonte | Output | Dipendenze |
|--------|-------|--------|------------|
| `scripts/import_jolpica.py` | Jolpica API | `f1_races` | requests |
| `scripts/import_standings.py` | Jolpica API | `f1_driver_standings` | requests |
| `scripts/import_tracinginsights.py` | GitHub Raw CSV | `f1_lap_times` | requests, pandas |
| `scripts/import_pinnacle_odds.py` | The Odds API | `f1_pinnacle_odds` | requests |
| `scripts/import_kaggle.py` | TracingInsights CSV | `f1_lap_times` | pandas |
| `scripts/compute_pace_observations.py` | `f1_lap_times` + `f1_races` | `f1_pace_observations` | numpy, scipy |
| `scripts/compute_calibration_records.py` | `f1_pinnacle_odds` + `f1_races` | calibration records | - |
| `scripts/extract_circuit_profiles.py` | FastF1 | `f1_circuit_profiles` | fastf1 |
| `scripts/migrate_jolpica_cache.py` | Disk cache | `f1_races` | - |

---

## Data Loaders (f1_predictor/data/)

| Loader | Collection | Uso |
|--------|------------|-----|
| `mongo_loader.py` (MongoRaceLoader) | `f1_races` | Load races for training |
| `mongo_pace_loader.py` (MongoPaceLoader) | `f1_pace_observations` | Load pace for Layer 1b |
| `mongo_odds_loader.py` (MongoOddsLoader) | `f1_pinnacle_odds` | Load odds for Layer 4 |
| `mongo_circuit_loader.py` (MongoCircuitLoader) | `f1_circuit_profiles` | Load circuit profiles |

---

## Gap Analysis

### Critici (Bloccano training)

| Issue | Impact | Solution |
|-------|--------|----------|
| `f1_pace_observations` = 0 | Layer 1b non può funzionare | Fix `import-tracinginsights.yml` + `compute_pace_observations.py` |
| `f1_pinnacle_odds` = 0 | Layer 4 non può funzionare | Create `import-pinnacle-odds.yml` |
| `f1_circuit_profiles` = 0 | Layer 1b manca circuit context | Create `import-circuit-profiles.yml` |

### Importanti (Degradano performance)

| Issue | Impact | Solution |
|-------|--------|----------|
| `f1_lap_times` = 12,000 (incompleto) | Layer 1b准确性 ridotta | Import completo da TracingInsights |
| `jolpica_cache` = 39 (vuoto) | Refresh dati lento | Implementare caching strategy |

---

## Piano di Implementazione

### Fase 1: Fix Criticali ✅
- [x] Creare `feat/data-unification` branch
- [x] Merge branch consolidati
- [x] Creare workflow base `import-jolpica.yml`
- [x] Verificare `f1_races` e `f1_driver_standings` in MongoDB

### Fase 2: TracingInsights Import ⚠️ IN PROGRESS
- [x] Creare `import-tracinginsights.yml`
- [x] Riscrivere `import_tracinginsights.py` per formato CSV reale
- [ ] Fix job dependency (YEAR non passato)
- [ ] Test import completo 2018+
- [ ] Verificare `f1_lap_times` popolato

### Fase 3: Pace Observations
- [ ] Verificare `compute_pace_observations.py` con nuovi lap_times
- [ ] Test `f1_pace_observations` popolato
- [ ] Validare Layer 1b funziona

### Fase 4: Odds Import
- [ ] Creare `import-pinnacle-odds.yml`
- [ ] Implementare `import_pinnacle_odds.py`
- [ ] Test `f1_pinnacle_odds` popolato

### Fase 5: Circuit Profiles
- [ ] Creare `import-circuit-profiles.yml`
- [ ] Implementare `extract_circuit_profiles.py`
- [ ] Test `f1_circuit_profiles` popolato

### Fase 6: Consolidamento
- [ ] Regression test suite completa
- [ ] Merge `feat/data-unification` → `main`
- [ ] Documentazione finale

---

## Note Tecniche

### Formato CSV TracingInsights

```csv
# lap_times.csv
raceId,driverId,lap,position,time,milliseconds
841,20,1,1,1:38.109,98109

# races.csv
raceId,year,round,circuitId,name,date,time
1,2009,1,1,"Australian Grand Prix","2009-03-29","06:00:00"

# drivers.csv
driverId,driverRef,number,code,forename,surname
1,"hamilton",44,"HAM","Lewis","Hamilton"

# circuits.csv
circuitId,circuitRef,name,location,country,lat,lng
1,"albert_park","Albert Park Grand Prix Circuit","Melbourne","Australia",-37.8497,144.968
```

### GitHub Actions Runner Limits
- File system NON condiviso tra job → Unire sync + import nello stesso job
- Timeout massimo: 60 minuti per job
- Bulk operations richieste per MongoDB

### Rate Limiting
- TracingInsights: Non limitato (CSV statico)
- The Odds API: 500 req/min (tier base)
- Jolpica API: Rate limitado

---

## Riferimenti

- Repo principale: https://github.com/gabrieledemarco/f1_predictor
- Branch: `feat/data-unification`
- TracingInsights RaceData: https://github.com/TracingInsights/RaceData
- TracingInsights 2025: https://github.com/TracingInsights/2025
- Jolpica API: https://jolpica-f1.readthedocs.io/
- The Odds API: https://the-odds-api.com/
