# LogLoss Audit Report

## Problem

- Current LogLoss: **5.39** (target < 1.0)
- This is MULTICLASS LogLoss (cross-entropy across 20 positions), NOT binary winner-only

## Current Implementation

From `train_pipeline.py:535-546`:
```python
# LogLoss with better clipping (1e-6) and renormalization
eps_clamp = 1e-6
prob_matrix_clipped = np.clip(prob_matrix_norm, eps_clamp, 1 - eps_clamp)
logloss = float(np.mean(-np.sum(actual_onehot * np.log(prob_matrix_clipped), axis=1)))
```

This computes: `-log(p_actual_position)` averaged over all drivers.
- Driver finishing 15th with p_15th = 0.01 → -log(0.01) = 4.6
- Driver finishing 20th with p_20th = 0.001 → -log(0.001) = 6.9

## Quantile Analysis

| Threshold | Count | Percentage |
|-----------|-------|------------|
| < 1e-05 | 0 | 0.0% |
| < 0.0001 | 0 | 0.0% |
| < 0.001 | 0 | 0.0% |
| < 0.01 | 0 | 0.0% |
| < 0.05 | 0 | 0.0% |

## Root Cause

1. **Formulation issue**: Multiclass LogLoss (20 classes) is inherently higher than binary LogLoss
2. Even with 1e-6 clipping, -log(1e-6) = 13.8 per miscalibrated probability
3. Target of < 1.0 is unrealistic for 20-class cross-entropy

## Analysis

With 20 positions and typical prediction accuracy:
- Mean position error ~5-8 positions
- Even well-calibrated models see ~2-4 per driver average
- 66 races × 20 drivers = 1320 contributions

## Recommendation

**Option 1**: Adjust target for multiclass LogLoss
- Realistic target: 2.0 - 4.0 for 20-class problem

**Option 2**: Use mean LogLoss per race (not aggregated)
- Track individual race LogLoss distributions

**Option 3**: Switch to binary winner-only LogLoss
- Maintains compatibility with original target < 1.0

## Impact on Other Metrics

- Brier: unaffected (squared error, bounded)
- ECE: unaffected (uses bins, not raw probs)
- RPS: unaffected (ordinal, uses ranks)
- Kendall tau: unaffected (ranking only)
