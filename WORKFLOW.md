# F1 Predictor — Workflow

## End-to-End Pipeline

```mermaid
flowchart TD
    subgraph SOURCES["Data Sources"]
        A1["Jolpica API\n(race results 2018–2026)"]
        A2["TracingInsights CSV\n(lap-by-lap telemetry)"]
        A3["Pinnacle / Betfair Odds\n(JSONL historical)"]
    end

    subgraph LOADING["Data Loading & Caching"]
        B1["JolpicaLoader\n+ JSON cache"]
        B2["TracingInsightsLoader\nlap times → constructor pace"]
        B3["OddsLoader\ndevig + implied prob"]
    end

    subgraph ADAPTER["Data Adapter"]
        C["adapter.py\n• Normalize race format\n• Enrich with pace obs.\n• Merge odds for calibration"]
    end

    subgraph TRAINING["Training  —  train_pipeline.py"]
        direction TB
        L1A["Layer 1a · Driver Skill\nTrueSkill Through Time (TTT)\nper driver × circuit type"]
        L1B["Layer 1b · Machine Pace\nKalman Filter\nper constructor"]
        L2["Layer 2 · Race Simulation\nMonte Carlo 50K sims\ntyre deg · pit strategy · DNF · SC"]
        L3["Layer 3 · Ensemble\nRidge Regression α=10\nwalk-forward, no look-ahead"]
        L4["Layer 4 · Calibration\nIsotonic Regression\ntrained on ≥100 races vs Pinnacle"]

        L1A --> L2
        L1B --> L2
        L2  --> L3
        L3  --> L4
    end

    subgraph VALIDATION["Walk-Forward Validation"]
        V["Temporal expanding window\n• Brier < 0.20\n• Kendall τ > 0.45\n• ECE < 0.05\n• ROI > 0%"]
    end

    subgraph STORAGE["Artifact Storage"]
        DB["MongoDB Atlas · GridFS\nbetbreaker / ml_artifacts_<year>\nversioned: v<YYYYMMDD_HHMM>"]
    end

    subgraph INFERENCE["Inference  —  pipeline.predict_race()"]
        direction TB
        I1["Load latest model\nfrom MongoDB"]
        I2["Score drivers\nLayer 1a + 1b"]
        I3["Run 50K MC sims\nLayer 2"]
        I4["Ensemble adjust\nLayer 3 Ridge"]
        I5["Isotonic calibrate\nLayer 4"]
        I1 --> I2 --> I3 --> I4 --> I5
    end

    subgraph EDGE["Edge Reporting"]
        E1["Devig Pinnacle odds\n(power devig)"]
        E2["Compute EV\np_model − p_pinnacle_novig"]
        E3["Kelly criterion\nf* = EV / odds − 1"]
        E4["EdgeReport JSON\nranked +EV bets  (min 4% edge)"]
        E5["H2H Markets\ndriver vs driver\nconstructor vs constructor"]
        E1 --> E2 --> E3 --> E4
        I5 --> E5
    end

    %% connections
    A1 --> B1
    A2 --> B2
    A3 --> B3

    B1 & B2 & B3 --> C

    C --> TRAINING

    TRAINING --> VALIDATION
    VALIDATION -->|"metrics OK"| STORAGE
    VALIDATION -->|"metrics KO"| TRAINING

    STORAGE --> INFERENCE

    I5 --> E1
    E4 & E5 -->|"pre-race report"| OUT["Pre-Race Output\n• Win / podium probabilities\n• Ranked betting edges\n• H2H matchup probs"]
```

---

## Execution Contexts

| Context | Command / Entry Point | Trigger |
|---------|----------------------|---------|
| **Training** | `python train_pipeline.py --year 2026 --through-round N` | After each race (local) |
| **Inference** | `pipeline.predict_race(race, grid, odds)` | Pre-race (Saturday post-quali) |
| **H2H extension** | `pipeline_h2h_extension.py` | Same timing as inference |
| **Diagnostics** | `python debug_pace.py` | On demand |
| **DB check** | `python test_mongo.py` | On demand |

## Key CLI Flags — `train_pipeline.py`

| Flag | Default | Purpose |
|------|---------|---------|
| `--year` | required | Season to train |
| `--through-round` | required | Last round included |
| `--train-from` | 2019 | Training window start |
| `--val-from` | 2023 | Validation window start |
| `--n-mc-sim` | 50000 | Monte Carlo simulations |
| `--ridge-alpha` | 10.0 | Ridge regularization |
| `--dry-run` | false | Train without saving |
| `--synthetic` | false | Force synthetic data |
| `--rollback` | — | Restore previous version |

## Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| Brier Score | < 0.20 | Win probability calibration |
| Kendall τ | > 0.45 | Finishing-order ranking |
| ECE | < 0.05 | Expected calibration error |
| ROI | > 0% | Simulated fractional-Kelly return |
```
