# 🏎️ F1 Predictor

> A 4-layer probabilistic machine learning pipeline for Formula 1 race outcome prediction and betting edge detection.

Built on top of real historical data (Jolpica/Ergast + TracingInsights), the system computes calibrated win/podium probabilities for each driver and exposes them as actionable edges against bookmaker odds.

---

## Architecture

The pipeline is composed of 4 sequential layers:

```
Raw Data (Jolpica + TracingInsights)
         │
         ▼
┌─────────────────────────────┐
│  Layer 1a — Driver Skill    │  TrueSkill Through Time (TTT)
│  Layer 1b — Machine Pace    │  Kalman Filter per constructor
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Layer 2 — Race Simulation  │  Bayesian Monte Carlo (50 000 sims)
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Layer 3 — Ensemble         │  Ridge meta-learner (α = 10.0)
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Layer 4 — Calibration      │  Isotonic Regression vs Pinnacle odds
└──────────────┬──────────────┘
               │
               ▼
      Edge Report + EV / Kelly
```

### Layer details

| Layer | Model | Purpose |
|-------|-------|---------|
| 1a | TrueSkill Through Time | Tracks driver skill (μ, σ) per circuit type, with season decay and regulation-change penalties |
| 1b | Kalman Filter | Tracks constructor pace, updating after each race with process noise |
| 2 | Bayesian Monte Carlo | Simulates 50 000 race outcomes sampling from Layer 1a/1b posteriors |
| 3 | Ridge ensemble | Adjusts raw MC probabilities using engineered features (grid position, pace delta, σ uncertainty) |
| 4 | Isotonic Calibration | Maps model-raw probabilities to well-calibrated ones using Pinnacle market odds as ground truth |

---

## Repository Structure

```
f1_predictor/
├── core/
│   ├── db.py                   # MongoDB connection helper
│   └── db_artifacts.py         # GridFS artifact store (save/load/rollback)
├── data/
│   ├── cache/jolpica/          # Jolpica JSON cache (auto-populated)
│   ├── racedata/               # TracingInsights CSV clone (git submodule)
│   └── pinnacle_odds/          # Historical Pinnacle odds JSONL (optional)
├── f1_predictor/
│   ├── domain/entities.py      # Dataclasses: Race, RaceResult, RaceProbability…
│   ├── models/
│   │   ├── driver_skill.py     # TTT implementation
│   │   ├── machine_pace.py     # Kalman Filter
│   │   ├── bayesian_race.py    # Monte Carlo simulation
│   │   └── ensemble.py         # Ridge meta-learner
│   ├── calibration/
│   │   ├── devig.py            # Power devig (implied probability)
│   │   ├── isotonic.py         # Isotonic calibration layer
│   │   └── edge_tracker.py     # Beta-Binomial edge tracker
│   ├── validation/
│   │   ├── walk_forward.py     # Temporal walk-forward validator
│   │   └── backtesting.py      # Betting backtest engine
│   ├── reports/
│   │   └── edge_report.py      # Edge report generator
│   └── pipeline.py             # F1PredictionPipeline orchestrator
├── pipeline.py                 # Top-level pipeline import/usage example
├── train_pipeline.py           # CLI training script (run locally after each GP)
├── debug_pace.py               # Debug utility for pace model
├── test_mongo.py               # MongoDB connectivity test
└── .env                        # MONGODB_URI (not committed)
```

---

## Deployment Architecture

Training and inference are intentionally separated:

```
Local machine (admin)              Streamlit Cloud (inference only)
───────────────────────            ────────────────────────────────
train_pipeline.py                  BetBreaker app (separate repo)
       │                                    │
       │  saves artifacts                   │  loads artifacts at startup
       ▼                                    ▼
  MongoDB Atlas ◄──────────── GridFS ───────────────────────►
```

The web app never trains — it only loads the latest serialized artifact version at boot.

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/gabrieledemarco/f1_predictor.git
cd f1_predictor

# 2. Install dependencies
pip install scikit-learn scipy numpy pandas pymongo[srv] requests

# 3. Install the f1_predictor package in editable mode
pip install -e ./f1_predictor

# 4. Clone TracingInsights race data (optional but recommended)
git clone https://github.com/TracingInsights/RaceData.git data/racedata

# 5. Configure MongoDB Atlas
cp .env.example .env
# Edit .env and set: MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>
```

---

## Usage

### Training

Run after each Grand Prix to retrain all 4 layers and push the new artifact to MongoDB:

```bash
# Standard training for 2026 through Round 5
python train_pipeline.py --year 2026 --through-round 5

# Dry run (train but don't save to MongoDB)
python train_pipeline.py --year 2026 --through-round 5 --dry-run

# Force synthetic data (for development/testing)
python train_pipeline.py --year 2026 --through-round 5 --synthetic

# Custom hyperparameters
python train_pipeline.py --year 2026 --through-round 5 \
    --n-mc-sim 100000 \
    --ridge-alpha 5.0 \
    --train-from 2018 \
    --val-from 2023
```

### Artifact Management

```bash
# List all saved model versions on MongoDB
python train_pipeline.py --list-versions

# Rollback to a specific version
python train_pipeline.py --rollback v20260303_0900

# Delete old versions (keep last N)
python train_pipeline.py --delete-old 5
```

### Programmatic Usage

```python
from f1_predictor.pipeline import F1PredictionPipeline

pipeline = F1PredictionPipeline()
pipeline.fit(historical_races)

report = pipeline.predict_race(
    race=race,
    driver_grid=driver_grid,
    pinnacle_odds={"VER": 2.10, "NOR": 4.50, "LEC": 6.00, ...}
)

for driver_code, prob in report["probabilities"].items():
    print(f"{driver_code}: P(win)={prob.p_win:.3f} | P(podium)={prob.p_podium:.3f}")

if report["edge_report"]:
    print(report["edge_report"])
```

---

## Validation Metrics

Walk-forward validation is run automatically on each training call. The following thresholds are used:

| Metric | Target | Description |
|--------|--------|-------------|
| **Brier Score** | < 0.20 | Primary model selection criterion. Measures calibration of P(win) |
| **Kendall τ** | > 0.45 | Ranking correlation between predicted and actual finishing order |
| **ECE** | < 0.05 | Expected Calibration Error across 10 probability bins |
| **ROI (WF)** | > 0% | Simulated return with fractional Kelly (0.25×) on edges > 4% |

Brier Score is the primary criterion for accepting or rejecting a new model version, in line with the methodology of Walsh & Joshi (2023).

---

## Data Sources

| Source | Usage | Notes |
|--------|-------|-------|
| [Jolpica API](https://github.com/jolpica/jolpica-f1) | Race results, driver/constructor history | Ergast-compatible fork, auto-cached locally |
| [TracingInsights RaceData](https://github.com/TracingInsights/RaceData) | Constructor pace telemetry | Clone to `data/racedata/` |
| Betfair Exchange Historical | Historical odds for calibration | Recommended for 2022–2024 backfill |
| API-Sports | Forward odds collection | For live seasons |
| Synthetic fallback | Development/testing | Auto-generated if no real data available |

---

## Configuration Reference

Key CLI arguments for `train_pipeline.py`:

| Argument | Default | Description |
|----------|---------|-------------|
| `--year` | required | Current season year |
| `--through-round` | required | Last completed round to include |
| `--train-from` | 2019 | First year of training data |
| `--val-from` | 2022 | First year of walk-forward validation window |
| `--n-mc-sim` | 50 000 | Number of Monte Carlo simulations |
| `--ridge-alpha` | 10.0 | Ridge regularization strength |
| `--min-calib-obs` | 100 | Minimum samples to activate Isotonic calibrator |
| `--dry-run` | false | Train but skip MongoDB upload |
| `--synthetic` | false | Force synthetic data (testing only) |

---

## Environment Variables

```env
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/<dbname>
```

The `.env` file is loaded automatically at runtime. Never commit this file.

---

## License

Private project — all rights reserved.
