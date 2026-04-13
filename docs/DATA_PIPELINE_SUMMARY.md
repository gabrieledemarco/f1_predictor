# F1 Data Pipeline - Unification Summary

## Overview

This document summarizes the work done to unify the F1 prediction data pipeline using MongoDB as the single source of truth, with data imported directly from third-party sources (TracingInsights) without local clones.

## Completed Work

### 1. Data Architecture Analysis
- Analyzed TracingInsights data sources (RaceData CSV and annual repos)
- Identified two data formats: flat CSV (RaceData) and detailed JSON (annual repos)
- Mapped data flow from external sources to MongoDB collections

### 2. Data Import Pipeline

| Collection | Records | Source | Years |
|------------|---------|--------|-------|
| `f1_lap_times` | 351,701 | TracingInsights RaceData | 2018-2025 |
| `f1_qualifying` | 3,455 | TracingInsights RaceData | 2018-2025 |
| `f1_session_stats` | 1,357 | TracingInsights GitHub API | 2022-2024 |
| `f1_pace_observations` | 2,016 | Computed from lap times | 2023-2025 |
| `f1_races` | 176 | Jolpica F1 | 2018-2025 |
| `f1_pit_stops` | 825 | TracingInsights RaceData | 2024 |

### 3. Scripts Created/Updated

| Script | Purpose |
|--------|---------|
| `import_tracinginsights.py` | Import lap times from RaceData CSV |
| `import_qualifying.py` | Import qualifying data for all years |
| `import_telemetry.py` | Import sector times via GitHub API (no clone) |
| `import_pit_stops.py` | Import pit stop data |
| `compute_pace_observations.py` | Compute constructor pace from lap times |
| `feature_selection_analysis.py` | Analyze feature importance from MongoDB |

### 4. Workflows Updated

- **Import TracingInsights Data** - Main data import workflow
  - Sync & Import (lap times, qualifying, pit stops)
  - Compute Constructor Pace
  - Import Telemetry (2024, 2023, 2022)
  - Feature Selection Analysis

### 5. Feature Importance Analysis

#### From Qualifying Times
| Feature | Importance |
|---------|------------|
| `total_q_ms` | 0.295 |
| `q1_ms` | 0.271 |
| `q2_ms` | 0.197 |
| `q3_ms` | 0.142 |

#### From Sector Times
| Feature | Importance |
|---------|------------|
| `s1_best_ms` | 0.264 |
| `s3_best_ms` | 0.244 |
| `s2_best_ms` | 0.234 |
| `total_best_ms` | 0.226 |

## Key Decisions

1. **No local data clones**: All data imported directly from GitHub APIs
2. **Sparse checkout approach**: Using GitHub API for on-demand data fetching
3. **MongoDB single source**: All collections now populated from TracingInsights

## Next Steps

### High Priority
1. **Integrate sector times into ML model** - Use s1, s2, s3 as features
2. **Add 2025 telemetry** - Import sector times for 2025 season
3. **Enhance feature engineering** - Combine qualifying + sector + pace features

### Medium Priority
1. **Add FastF1 for historical sector times** - Fill gap for 2018-2021
2. **Create feature pipeline** - Automated feature generation for training
3. **Validate model performance** - Test with new sector time features

### Low Priority
1. **Add weather data** - Temperature, humidity from TracingInsights
2. **Pit stop analysis** - Feature importance for pit stops
3. **Tyre strategy features** - Compound, stint length, laps

## Repository Status

- Branch: `feat/data-unification`
- All data imported and validated
- Feature analysis complete
- Ready for merge to main

---

*Generated: 2026-04-13*
