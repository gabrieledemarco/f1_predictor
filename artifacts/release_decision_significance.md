# Release Decision - Significatività Regressori

## Date: May 2026

## Executive Summary

This document provides the go/no-go decision based on the significatività (significance) analysis of model regressors across all layers.

---

## Analysis Summary

### Step 0: Baseline Frozen
- **Training**: 2022
- **Validation**: 2023-2024 (27 races)
- **Metrics**: Kendall τ, Brier, ECE, RPS, LogLoss

### Step 1: L3 (Ensemble) Significance
**Results**:
| Group | Mean Permutation Importance |
|-------|------------------------------|
| L2 (Monte Carlo) | 0.0451 |
| L1a (Driver Skill) | 0.0297 |
| Context | 0.0191 |
| L1b (Machine Pace) | 0.0067 |
| Historical | 0.0050 |

**Finding**: L2 features are most important, Historical features least important.

### Step 2: L1b (Kalman) Significance
**Results**:
| State Variable | Mean z-score | Classification |
|----------------|--------------|----------------|
| base_intercept | 2.54 | strong |
| top_speed_coef | 1.87 | weak |
| continuous_pct_coef | 1.83 | weak |
| throttle_pct_coef | 1.26 | weak |
| circuit_type dummies | ~0.80 | unstable |

**Finding**: Base intercept is strongest, circuit type coefficients are noisy.

### Step 3: L1a (DriverSkill) Ablation
**Results**:
| Variant | Kendall τ | Brier |
|---------|-----------|-------|
| global_only | 0.437 | 0.167 |
| current_blend | 0.451 | 0.164 |
| conservative_blend | 0.445 | 0.166 |

**Finding**: Current blend is best; circuit-type specificity adds value.

### Step 4: Pruning Proposal
**Results**:
| Model | Kendall τ | Delta τ |
|-------|-----------|---------|
| baseline | 0.451 | - |
| pruned_light | 0.449 | -0.002 |
| pruned_strict | 0.435 | -0.016 |

**Finding**: Pruning causes regression - recommend keeping baseline.

---

## Go/No-Go Criteria

### Required Targets
- [x] τ not worse than baseline (≥ 0.45) → **0.451 - PASS**
- [x] Brier < 0.20 → **0.164 - PASS**
- [x] ECE < 0.05 → **0.015 - PASS**

### Significance Findings
- [x] L2 features are significant
- [x] L1a features are significant
- [x] L1b base is significant, circuit types unstable
- [x] Historical features are low importance but don't hurt

### Pruning Result
- [x] No pruning needed - baseline is optimal

---

## Decision: **GO**

### Rationale
1. **All required targets met**: τ, Brier, ECE all within target
2. **Feature importance validated**: Model uses meaningful features from all layers
3. **Pruning not beneficial**: Feature set is already optimized
4. **No regression risk**: Ablation shows current config is best

### Artifacts Delivered
- `artifacts/significance_baseline.md` - Baseline frozen config
- `artifacts/l3_feature_importance.csv` / `.md` - L3 significance
- `artifacts/l1b_significance.csv` / `.md` - L1b significance  
- `artifacts/l1a_ablation_results.csv` / `.md` - L1a ablation
- `artifacts/pruning_plan.md` - Pruning proposal
- `artifacts/pruned_model_comparison.csv` - Comparison results
- `scripts/l3_significance.py` - L3 analysis script
- `scripts/l1b_kalman_significance.py` - L1b analysis script
- `scripts/l1a_ablation.py` - L1a ablation script
- `scripts/pruning_comparison.py` - Pruning script

### Recommendations for Future Work
1. Collect more historical feature data to increase their importance
2. Investigate Kalman circuit-type coefficient instability
3. Consider ensemble weight optimization (beyond Ridge)

---

*Decision: GO - Release with current feature set*