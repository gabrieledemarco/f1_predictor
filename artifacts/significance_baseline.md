# Significance Analysis Baseline

## Frozen Configuration

### Dataset Split
- **Training years**: 2022
- **Validation years**: 2023-2024 (first 5 races: R1-R5)
- **Walk-forward**: Expanding window, train on years < val_from

### Evaluation Metrics
All metrics computed per-race and averaged:
- **Kendall τ** (rank correlation) - Target: > 0.45
- **Brier** (multiclass) - Target: < 0.20
- **ECE** (calibration) - Target: < 0.05
- **RPS** (ordinal) - Target: < 0.15
- **LogLoss** (indicative) - Target: < 1.0

### Model Configuration
- Monte Carlo simulations: 1,000 per race
- Ridge alpha: 10.0
- Kalman Q/R: default (Q_base=0.01, R=0.05)
- TTT: default config

---

## Baseline Results (Current Model)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Kendall τ | 0.451 | > 0.45 | PASS (borderline) |
| Brier | 0.164 | < 0.20 | PASS |
| ECE | 0.015 | < 0.05 | PASS |
| RPS | 0.152 | < 0.15 | BORDERLINE |
| LogLoss | 5.39 | < 1.0 | NEEDS WORK |

### Bootstrap 95% CI
- Kendall τ: [0.384, 0.470]
- Brier: [0.125, 0.171]

---

## Feature Groups Analysis Scope

### Layer 3 (Ensemble) Features
- **L1a**: skill_mu, skill_sigma (from DriverSkill)
- **L1b**: pace_mu, pace_sigma (from Kalman)
- **L2**: p_win_mc, p_podium_mc, exp_pos_mc (from Monte Carlo)
- **Context**: circuit_type, grid_position, has_grid_penalty
- **Historical**: h2h_win_rate_3season, elo_delta_vs_field, dnf_rate_relative

### Layer 1b (Kalman) State Dimensions
- Base intercept
- Top speed, continuous_pct, throttle_pct
- Street, desert, hybrid, temp circuit types

### Layer 1a (DriverSkill) Components
- Global rating
- Circuit-type specific rating
- Observation count threshold

---

## Artifacts Generated
- `significance_baseline.md` (this file)
- `l3_feature_importance.csv` / `.md`
- `l1b_significance.csv` / `.md`
- `l1a_ablation_results.csv` / `.md`
- `pruned_model_comparison.csv` / `.md`
- `release_decision_significance.md`

---

*Baseline frozen: May 2026*