# L1 Fix Plan - Diagnosis and Targeted Fixes

## Phase 1: Data Quality Audit Results

**Finding**: Data coverage is excellent (96.7% overall)
- All constructors have >95% coverage
- Pace values have reasonable distribution
- Small outlier percentage (<2%)

**Action**: No fixes needed for data quality

---

## Phase 2: L1a (Driver Skill) Intrinsic Results

**Finding**: Current blend is best variant
- Shows positive signal for ranking
- Circuit-type specificity beneficial
- Rookies harder to predict

**Potential Issues Identified**:
1. Rookie drivers have lower predictive accuracy
2. Season start has less data for circuit-specific ratings

**Recommended Fixes**:

| Fix | Priority | Rationale |
|-----|----------|----------|
| Tune MIN_CIRCUIT_RACES threshold | HIGH | Current threshold of 10 may be too aggressive; try 15-20 |
| Adjust season decay | MEDIUM | Current decay may be too steep for mid-season stability |
| Weight conservative rating | LOW | mu - 3σ conservative rating could use different weight |

---

## Phase 3: L1b (Machine Pace) Intrinsic Results

**Finding**: Model shows positive rank correlation (0.71) but RMSE is moderate
- RMSE: 0.30s/lap
- Some constructors show higher error variance

**Potential Issues Identified**:
1. Circuit type coefficients are noisy (low z-scores)
2. May be slightly over-parameterized (8D vs reduced)
3. Top teams have lower error than backmarker teams

**Recommended Fixes**:

| Fix | Priority | Rationale |
|-----|----------|----------|
| Reduce state dimension | HIGH | Drop circuit type dummies, keep only intercept + main features |
| Increase Q shrinkage | HIGH | More conservative updates for stability |
| Tune R parameter | MEDIUM | Current R may be too low, causing overfitting |
| Add prior shrinkage | LOW | Initialize with stronger prior to prevent drift |

---

## Priority Matrix

### L1a Fixes
| Priority | Fix | Expected Impact |
|----------|-----|------------------|
| HIGH | Adjust MIN_CIRCUIT_RACES from 10 to 15 | +0.02-0.03 tau |
| MEDIUM | Reduce season decay rate | Better mid-season stability |
| LOW | Weight adjustment | Minimal expected change |

### L1b Fixes  
| Priority | Fix | Expected Impact |
|----------|-----|------------------|
| HIGH | Reduce to 4D state (intercept + top_speed + continuous + throttle) | Stabilize predictions |
| HIGH | Increase Q by 50% | More conservative updates |
| MEDIUM | Tune R from 0.05 to 0.08 | Better calibration |

---

## Implementation Order

1. **L1b: Reduce state dimension** - Highest expected ROI
2. **L1b: Increase Q shrinkage** - Quick change with likely benefit  
3. **L1a: Adjust MIN_CIRCUIT_RACES** - Simple config change
4. **L1a: Season decay tuning** - Requires more validation

---

## Verification Plan

After implementing fixes, re-run:
1. L1a intrinsic eval → expect improvement in rookie segment
2. L1b intrinsic eval → expect lower RMSE, better rank correlation
3. End-to-end (Phase 5) → verify τ, Brier, ECE improvement

---

*Generated from Phase 1-3 analysis*