# F1 Predictor Improvement Plan

## Overview

This document outlines the improvement plan for the F1 Predictor model based on research analysis conducted in April 2026. The plan identifies gaps, validates existing implementations, and proposes actionable improvements organized by priority.

## Research Sources

| Search Date | Topic | Key Sources |
|------------|-------|-------------|
| 2026-04-15 | TrueSkill Through Time | J. Stat. Soft. v112i06; Dangauthier et al. 2007; Minka et al. 2018 |
| 2026-04-15 | Bayesian Calibration | Pakdaman Naeini et al. 2015; Berta et al. 2024; sklearn docs |
| 2026-04-15 | Monte Carlo Racing | Lienkamp et al. 2020; Fry et al. 2023; SIG Machine Learning |
| 2026-04-15 | Walk-Forward Validation | QuantSport 2026; DataField.Dev; WagerProof |
| 2026-04-15 | Kelly + EV Betting | Kelly 1956; Baker & McHale 2013; AgentBets.ai 2026 |

---

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    F1 PREDICTION PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────┤
│  RAW DATA (MongoDB)                                                    │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐        │
│  │ Jolpica-F1   │TracingInsights│ Pinnacle    │Circuit     │        │
│  │ (race results)│ (lap times)  │ (bet odds)  │Profiles    │        │
│  └──────┬───────┴──────┬───────┴──────┬───────┴──────┘        │
│         │              │              │              │                │
│         ▼              ▼              ▼              ▼                │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │               MONGODB ATLAS COLLECTIONS                      │       │
│  │  f1_races, f1_lap_times, f1_pace_obs, f1_session_stats       │       │
│  └─────────────────────────┬───────────────────────────────────┘       │
│                          │                                             │
│         ┌────────────────┼────────────────┐                          │
│         ▼                ▼                ▼                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                    │
│  │ Layer 1a   │  │ Layer 1b   │  │ Layer 4   │                    │
│  │ TTT        │  │ Kalman    │  │ Isotonic  │                    │
│  │ Driver    │  │ Constructor│  │ Calibr.   │                    │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘                    │
│        │              │              │                             │
│        └──────────────┼──────────────┘                             │
│                     ▼                                               │
│            ┌────────────────┐                                      │
│            │ Layer 2        │                                      │
│            │ Monte Carlo   │  (50,000 sims)                         │
│            │ Race Sim      │                                      │
│            └───────┬────────┘                                      │
│                    ▼                                               │
│            ┌────────────────┐                                      │
│            │ Layer 3        │                                      │
│            │ Ridge Ensemble │                                      │
│            │ Meta-learner   │                                      │
│            └───────┬────────┘                                      │
│                    ▼                                               │
│            ┌────────────────┐                                      │
│            │ Layer 4        │                                      │
│            │ Final        │                                      │
│            │ Calibration │                                      │
│            └───────┬────────┘                                      │
│                    ▼                                               │
│            ┌────────────────┐                                      │
│            │ OUTPUT        │                                      │
│            │ Probabilities│                                      │
│            │ + Edge Report │                                       │
│            └───────────────┘                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Validation Results by Layer

### Layer 1a: TrueSkill Through Time (Driver Skill)

| Aspect | Implementation | Research Match | Status |
|--------|---------------|----------------|--------|
| Gaussian belief propagation | Custom implementation | TTT paper | ✅ Match |
| Per circuit-type skill | `get_rating(code, circuit_type)` | TTT multidimensional | ✅ Match |
| Season decay | `apply_season_decay()` | TTT paper | ✅ Match |
| Regulation changes | `major_change` detection | TTT paper | ✅ Match |
| **Batch inference** | Online only | TTT full-batch | ❌ Gap |
| **Gamma optimization** | tau=0.833 (fixed) | Grid search needed | ⚠️ Gap |

### Layer 1b: Kalman Filter (Constructor Pace)

| Aspect | Implementation | Research Match | Status |
|--------|---------------|----------------|--------|
| Kalman state estimation | Present | State-space models | ✅ Match |
| Process noise | Configurable | Standard KF | ✅ Match |
| Uncertainty tracking | Via sigma | Standard KF | ✅ Match |

### Layer 2: Monte Carlo Race Simulation

| Aspect | Implementation | Research Match | Status |
|--------|---------------|----------------|--------|
| N simulations | 50,000 | 10,000-1M range | ✅ Match |
| Driver skill samples | From Layer 1a | Standard MC | ✅ Match |
| Constructor pace samples | From Layer 1b | Standard MC | ✅ Match |
| **Variance reduction** | Not implemented | Antithetic variates | ❌ Gap |
| **Ghost car approach** | Not implemented | Lienkamp 2020 | ❌ Gap |

### Layer 3: Ridge Meta-learner

| Aspect | Implementation | Research Match | Status |
|--------|---------------|----------------|--------|
| Ridge regression | alpha=10.0 | Standard ridge | ✅ Match |
| Feature engineering | Grid, pace, skill | Domain-specific | ✅ Match |
| Multi-class support | One-vs-rest | Standard | ✅ Match |

### Layer 4: Isotonic Calibration

| Aspect | Implementation | Research Match | Status |
|--------|---------------|----------------|--------|
| Isotonic regression | Present | sklearn | ✅ Match |
| Pinnacle odds alignment | Present | Industry practice | ✅ Match |
| ECE metric | `expected_calibration_error()` | Walsh & Joshi | ✅ Match |
| **ROC-regularized** | Not implemented | Berta et al. 2024 | ❌ Gap |
| **Bayesian isotonic** | Not implemented | Pakdaman Naeini 2015 | ❌ Gap |

### Validation & Backtesting

| Aspect | Implementation | Research Match | Status |
|--------|---------------|----------------|--------|
| Walk-forward | Present | Standard WFV | ✅ Match |
| **Purged CV** | Not implemented | DataField.Dev | ❌ Gap |
| **Expanding vs sliding** | Rolling window | Standard | ✅ Match |
| Kelly criterion | Quarter-Kelly | Baker & McHale | ✅ Match |
| Edge tracking | BetaBinomialEdgeTracker | Advanced | ✅ Match |

---

## Improvement Actions

### Priority 1: Quick Wins (1 week)

| # | Action | Files to Modify | Validation |
|-----|--------|-----------------|------------|
| 1.1 | Gamma (tau) grid search optimization | `models/driver_skill.py` ✅ | tau: 0.833→0.05 |
| 1.2 | Add per-surface multidimensional skill | `models/driver_skill.py` | Per-surface accuracy |
| 1.3 | Benchmark TTT batch vs online | `models/driver_skill.py` | Historical comparability |

### Priority 2: Medium Improvements (2 weeks)

| # | Action | Files to Modify | Validation |
|-----|--------|-----------------|------------|
| 2.1 | Implement purged CV with gap | `validation/walk_forward.py` | No leakage detection |
| 2.2 | ROC-regularized isotonic calibration | `calibration/isotonic.py` | ECE + AUC preserve |
| 2.3 | Monte Carlo variance reduction | `scripts/variance_reduction_test.py` | VR: 34% |

### Priority 3: Advanced Features (4 weeks)

| # | Action | Files to Modify | Validation |
|-----|--------|-----------------|------------|
| 3.1 | Bayesian isotonic calibration | `calibration/isotonic.py` | Smoothness vs ECE |
| 3.2 | Kelly as Bayesian model evaluation | `calibration/edge_tracker.py` | Real-time bankroll |
| 3.3 | RL race strategy (future) | New module | Mercedes F1 paper ref |

---

## Implementation Tracking

### Sprint 1 (Quick Wins)

- [x] **1.1** Gamma (tau) optimization
  - Owner: gabrieledemarco
  - Status: COMPLETE ✅
  - Notes: tau updated from 0.833 to 0.05 (default) and 0.10 (2026 preset)
- [x] **1.2** Per-surface skill
  - Owner: gabrieledemarco
  - Status: COMPLETE
  - Notes: Verified - requires race_metadata with circuit_type in fit()
- [x] **1.3** TTT benchmark
  - Owner: gabrieledemarco
  - Status: COMPLETE
  - Notes: Batch vs Online identici (correlazione=1.0)

- [x] **2.1** Purged CV
  - Owner: gabrieledemarco
  - Status: COMPLETE
  - Notes: Implemented purge_gap parameter in WalkForwardValidator 
- [x] **2.2** ROC-regularized isotonic
  - Owner: gabrieledemarco
  - Status: COMPLETE
  - Notes: New class to preserve AUC (f1_predictor/calibration/)
- [x] **2.3** Monte Carlo variance reduction
  - Owner: gabrieledemarco
  - Status: COMPLETE
  - Notes: Test shows VR benefit; implementation optional

### Sprint 2 (Medium)

- [x] **2.1** Purged CV
  - Owner: gabrieledemarco
  - Status: COMPLETE
  - Notes: Implemented in walk_forward.py with purge_gap=3 
- [ ] **2.2** ROC-regularized isotonic
  - Owner: 
  - Status: 
  - Notes: 
- [ ] **2.3** MC variance reduction
  - Owner: 
  - Status: 
  - Notes: 

### Sprint 3 (Advanced)

- [ ] **3.1** Bayesian isotonic
  - Owner: 
  - Status: 
  - Notes: 
- [ ] **3.2** Kelly evaluation
  - Owner: 
  - Status: 
  - Notes: 

---

## Key References

### TrueSkill Through Time
- Dangauthier, P., Herbrich, R., Minka, T., & Graepel, T. (2007). TrueSkill Through Time: Revisiting the History of Chess. NeurIPS.
- Minka, T., Cleven, R., & Zaykov, Y. (2018). TrueSkill 2: An Improved Bayesian Skill Rating System.
- Landfried, G. & Mocskos, E. (2025). TrueSkillThroughTime. J. Stat. Soft.

### Probability Calibration
- Pakdaman Naeini, M., Cooper, G., & Hauskrecht, M. (2015). Obtaining Well Calibrated Probabilities Using Bayesian Binning. AAAI.
- Berta, E., Bach, F., & Jordan, M. (2024). Classifier Calibration with ROC-Regularized Isotonic Regression. PMLR.

### Monte Carlo Racing
- Lienkamp, A., et al. (2020). Application of Monte Carlo Methods to Consider Probabilistic Effects in Race Simulation. Appl. Sci.
- Fry, M., et al. (2023). Time-rank Duality in Formula 1 Racing. arXiv.

### Walk-Forward Validation
- DataField.Dev (2026). Chapter 30: Model Evaluation and Selection. Sports Betting Textbook.
- QuantSport (2026). Why Static Cross-Validation Fails in Sports Modeling.

### Kelly Betting
- Kelly, J. (1956). A New Interpretation of Information Rate. Bell System Technical Journal.
- Baker, R. & McHale, I. (2013). Optimal Fixed Fractional Betting Strategies. IMA J Management Math.
- AgentBets.ai (2026). The Kelly Criterion: Optimal Bet Sizing for Autonomous Agents.

---

## Document History

| Date | Author | Changes |
|------|--------|---------|
| 2026-04-15 | Research Analysis | Initial document from research findings |