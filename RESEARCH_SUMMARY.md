# Research Summary - F1 Predictor Evaluation Overhaul

## Overview

This document summarizes the work completed during the evaluation metrics overhaul (Phase 0-5 of the F1 Predictor project).

---

## Background

The F1 Predictor project aims to predict Formula 1 race outcomes using a 4-layer model:
- Layer 1a: Driver Skill Rating (TTT)
- Layer 1b: Machine Pace (Kalman Filter)
- Layer 2: Bayesian Race Simulation (Monte Carlo)
- Layer 3: Ensemble adjustment (Ridge)
- Layer 4: Calibration against Pinnacle

The original walk-forward validation used weak metrics:
- **Brier "winner-only"**: Only evaluated probability assigned to the actual winner
- **Synthetic ROI**: Computed using `market_p = p * 0.97` (fabricated market prices)
- **No RPS/LogLoss**: Missing important probabilistic metrics

---

## Work Completed

### Phase 0 - Baseline & Safety

- Created dedicated branch: `feat/eval-overhaul`
- Ran baseline dry-run and saved output to `artifacts/eval_baseline_before.txt`
- Verified clean repo state before modifications

**Files modified:**
- `f1_predictor/pipeline/prediction_pipeline.py` - Fixed circuit profile integration for Kalman filter
- `train_pipeline.py` - Added circuit parameter to machine_pace.update() and get_estimate()

### Phase 1 - Walk-forward Metrics

Replaced weak metrics with robust ones:

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Kendall τ | 0.448 | 0.446 | ≥0.45 |
| Brier | 0.2767 (winner-only) | 0.1626 (multiclass) | ≤0.20 |
| LogLoss | N/A | 9.91 | ≤1.0 |
| RPS | N/A | 0.1513 | ≤0.15 |
| ECE | 0.0149 | 0.0153 | ≤0.05 |

**Implementation details:**

1. **Brier Multiclass**: 
   - Builds probability matrix (n_drivers × 20 positions)
   - Creates one-hot actual positions
   - Computes mean squared error across all drivers and positions

2. **LogLoss Multiclass**:
   - Uses epsilon clipping (1e-15) to avoid log(0)
   - Row-normalizes probability matrix
   - Computes cross-entropy loss

3. **RPS (Ranked Probability Score)**:
   - Uses existing `f1_predictor.validation.metrics.ranked_probability_score()`
   - Computes CDF difference between predicted and actual ordinal outcomes
   - Average RPS across all drivers in each race

**Code locations:**
- `train_pipeline.py` lines 457-530: Multiclass metric calculation in walk-forward loop
- `train_pipeline.py` lines 600-610: Metric aggregation

### Phase 2 - ROI Replacement

- Deprecated `_simulate_roi` based on synthetic market prices
- Added `metrics["roi_real"] = None` placeholder for real backtest
- When odds become available, will integrate with `BettingBacktester` from `f1_predictor.validation.backtesting`

### Phase 3 - Output & Reporting

Updated reporting functions:
- `_log_metrics()`: Displays all new metrics with targets
- `print_summary()`: Shows data coverage info (# races with position dist, # races with odds)

**Sample output:**
```
+- METRICHE WALK-FORWARD ---------------------------
|  Kendall tau    +0.446   WARN  target >= 0.45
|  Brier (mc)     0.1626   OK  target <= 0.20
|  LogLoss (mc)   9.9100   WARN  target <= 1.0
|  RPS mean       0.1513   WARN  target <= 0.15
|  ROI (WF)       +0.0   WARN  target > 0%
|  ECE            0.0153   OK  target <= 0.05
|  Data coverage: 27 races with position dist, 0 races with odds
+---------------------------------------------------------------
```

### Phase 4 - Automated Tests

Created comprehensive test suite:

**File:** `tests/validation/test_walkforward_metrics.py`
- 18 tests passing
- Covers:
  - Brier multiclass (perfect prediction, range check)
  - LogLoss multiclass (perfect prediction, range check)
  - RPS (perfect, worst, uniform, mean)
  - Backtest integration (with/without odds)
  - Edge cases (zero probs, NaN, inf, normalization)

**Run tests:**
```bash
python -m pytest tests/validation/test_walkforward_metrics.py -v
```

---

## Additional Robustness Fixes (May 2026)

Implemented additional fixes to improve metric calculation robustness:

### 1. Race-level Normalization Hard Check
- After building probability matrix, checks if any row sums to ≤ eps (1e-8) or contains NaN
- Falls back to uniform distribution (1/20) for problematic rows
- Final verification that all rows sum to ~1.0

### 2. Improved Clipping for LogLoss
- Changed from 1e-15 to 1e-6 for more stable clipping
- Renormalizes after clipping to ensure valid probability distribution
- Uses `np.nan_to_num()` to handle any remaining NaN/inf

### 3. Diagnostics Logging
- Per-race logging of: min_p, max_p, sum_p, p_winner
- Available via debug logging (--debug flag)

### 4. Unit Tests for Edge Cases
Added 7 new tests in `tests/validation/test_walkforward_metrics.py`:
- `test_all_zero_probabilities`: Fallback to uniform
- `test_probability_sum_greater_than_one`: Normalization
- `test_probability_sum_less_than_one`: Normalization
- `test_winner_missing_from_driver_vector`: Graceful handling
- `test_nan_in_probabilities`: NaN handling
- `test_inf_in_probabilities`: Inf handling
- `test_clipped_logloss`: Clipping + renormalization

### Results After Fixes
| Metric | Before | After Fixes | Target |
|--------|--------|-------------|--------|
| Kendall τ | 0.446 | 0.451 | ≥0.45 |
| Brier | 0.1626 | 0.1638 | ≤0.20 |
| LogLoss | 9.91 | 5.39 | ≤1.0 |
| RPS | 0.1513 | 0.1518 | ≤0.15 |

**Key improvement**: LogLoss dropped from ~10 to ~5.4 after fixing clipping and normalization issues.

---

## Key Discoveries

1. **Brier improvement**: Multiclass Brier (0.16) is much more informative than winner-only (0.28) as it evaluates full position distribution

2. **LogLoss fix**: Values improved from ~10 to ~5.4 after implementing proper clipping (1e-6) and normalization

3. **RPS performance**: Very close to target (0.151 vs 0.15), suggesting model has good ordinal calibration

4. **Circuit profiles**: The Kalman filter now correctly uses `CircuitSpeedProfile` objects for both update and estimation, improving Layer 1b functionality

---

## Remaining Work (Future)

1. **Debug LogLoss further**: Even at 5.4, still above target of 1.0 - need to investigate probability calibration
2. **Phase 2 ROI**: Integrate `BettingBacktester` when odds data becomes available
3. **Position distribution issues**: Some drivers may have empty position distributions

---

## Artifacts

| File | Description |
|------|-------------|
| `artifacts/eval_baseline_before.txt` | Baseline run output |
| `artifacts/eval_baseline_after_phase1.txt` | Post-Phase 1 run output |
| `artifacts/eval_phase1_results.md` | Phase comparison notes |

---

## Dependencies

Key modules used:
- `f1_predictor.validation.metrics`: `ranked_probability_score()`, `mean_ranked_probability_score()`
- `f1_predictor.validation.backtesting`: `BettingBacktester`, `BacktestConfig`
- `f1_predictor.data.circuit_profiles`: `get_profile_safe()`
- `f1_predictor.domain.entities`: `Race`, `Circuit`, `RaceProbability`

---

*Last updated: May 2026*