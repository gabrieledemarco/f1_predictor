# Evaluation Contract - F1 Predictor Metrics

## Overview

This document defines the formal specification for all walk-forward evaluation metrics used in the F1 Predictor project. These metrics form the **metric contract** that must be satisfied for model release decisions.

---

## Metric Definitions

### 1. Kendall τ (Rank Correlation)

**Definition**: Per-race Kendall τ rank correlation coefficient, averaged across all validation races.

**Formula**:
```
For each race:
    1. Sort drivers by predicted p_win (descending) → predicted_order
    2. Sort drivers by actual finish_position (ascending) → actual_order
    3. Compute Kendall τ on overlapping drivers (must have ≥5 common drivers)
    4. Take mean of per-race τ values across all validation races
```

**Target**: `τ > 0.45`

**Interpretation**:
- ≥ 0.45: Good predictive ordering
- 0.35 - 0.45: Moderate predictive power
- < 0.35: Weak ordering

---

### 2. Brier Score (Multiclass)

**Definition**: Mean squared error between predicted position probabilities and actual one-hot encodings.

**Formula**:
```
For each race:
    1. Build probability matrix P (n_drivers × 20) where P[i,j] = P(driver_i finishes position j)
    2. Build one-hot matrix A (n_drivers × 20) where A[i,actual_pos-1] = 1
    3. Normalize rows: P_norm = P / row_sums (fallback to uniform if sum ≤ 1e-8)
    4. Brier = mean((P_norm - A)^2)
    
Final Brier = mean of per-race Brier values
```

**Clipping**: None required for Brier (squared error naturally bounded)

**Target**: `Brier < 0.20`

**Interpretation**:
- < 0.15: Excellent calibration
- 0.15 - 0.20: Good calibration
- 0.20 - 0.30: Moderate calibration
- > 0.30: Poor calibration

---

### 3. LogLoss (Multiclass)

**Definition**: Mean cross-entropy loss between predicted position distributions and actual outcomes.

**Formula**:
```
For each race:
    1. Build probability matrix P (n_drivers × 20) as described above
    2. Apply clipping: P_clipped = clip(P, 1e-6, 1 - 1e-6)
    3. Renormalize: P_final = P_clipped / row_sums
    4. Handle NaN/inf: P_final = nan_to_num(P_final, nan=1/20, posinf=1.0, neginf=0.0)
    5. LogLoss = mean(-sum(A * log(P_final), axis=1))
    
Final LogLoss = mean of per-race LogLoss values
```

**Clipping**: `eps = 1e-6` (chosen for numerical stability while preserving probability mass)

**Target**: `LogLoss < 1.0` (indicative, lower is better)

**Interpretation**:
- < 1.0: Good probabilistic predictions
- 1.0 - 2.0: Moderate
- > 2.0: Poor probabilistic calibration

---

### 4. RPS (Ranked Probability Score)

**Definition**: Mean squared error between predicted and actual cumulative distribution functions.

**Formula**:
```
For each driver in each race:
    1. Get predicted distribution P (length 20) for position probabilities
    2. Get actual finishing position pos (1-20)
    3. Compute CDF_pred[k] = sum(P[0:k]) for k = 1..19
    4. Compute CDF_actual[k] = 1 if k >= pos, else 0
    5. RPS = mean((CDF_pred[:-1] - CDF_actual[:-1])^2)
    
Final RPS = mean of per-driver RPS values, averaged across races
```

**Target**: `RPS < 0.15` (indicative)

**Interpretation**:
- < 0.10: Excellent ordinal calibration
- 0.10 - 0.15: Good ordinal calibration
- 0.15 - 0.25: Moderate
- > 0.25: Poor ordinal calibration

---

### 5. ECE (Expected Calibration Error)

**Definition**: Difference between predicted confidence and observed accuracy across probability bins.

**Formula**:
```
For all predictions across validation races:
    1. Bin predictions into n_bins (default 10) based on quantiles
    2. For each bin b:
        - avg_confidence = mean(predicted_probabilities in bin)
        - accuracy = mean(actual_outcomes in bin)
        - weight = count(bin) / total_count
    3. ECE = sum(weight_b * |accuracy_b - avg_confidence_b|)
    
Final ECE = weighted average across bins
```

**Target**: `ECE < 0.05`

**Interpretation**:
- < 0.03: Well calibrated
- 0.03 - 0.05: Acceptable calibration
- > 0.05: Poor calibration

---

## Official Targets Summary

| Metric | Target | Type |
|--------|--------|------|
| Kendall τ | > 0.45 | Higher is better |
| Brier | < 0.20 | Lower is better |
| ECE | < 0.05 | Lower is better |
| LogLoss | < 1.0 | Lower is better (indicative) |
| RPS | < 0.15 | Lower is better (indicative) |

---

## Probability Guardrails

All metrics require the following probability validity checks:

1. **No NaN/inf**: Probability matrix must not contain NaN or infinite values
2. **Row sums ~1**: Each driver's probability distribution must sum to ~1.0
3. **Fallback**: If probability row is invalid (sum ≤ 1e-8 or NaN), fallback to uniform distribution (1/20 for each position)

---

## Implementation Requirements

### In train_pipeline.py (_run_walkforward):

```python
# Normalization guardrail
row_sums = prob_matrix.sum(axis=1, keepdims=True)
eps_normalize = 1e-8
bad_rows = np.where((row_sums < eps_normalize) | np.isnan(row_sums))[0]
for i in bad_rows:
    prob_matrix[i, :] = 1.0 / 20
prob_matrix_norm = prob_matrix / np.maximum(row_sums, 1.0)

# Clipping for LogLoss
eps_clamp = 1e-6
prob_matrix_clipped = np.clip(prob_matrix_norm, eps_clamp, 1 - eps_clamp)
prob_matrix_clipped = prob_matrix_clipped / prob_matrix_clipped.sum(axis=1, keepdims=True)
prob_matrix_clipped = np.nan_to_num(prob_matrix_clipped, nan=1.0/20, posinf=1.0, neginf=0.0)
```

### Diagnostic logging (per race):
- min_p: minimum p_win across drivers
- max_p: maximum p_win across drivers  
- sum_p: sum of p_win (should be ~1.0)
- p_winner: probability assigned to actual winner

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | April 2026 | Initial contract |
| 1.1 | May 2026 | Added LogLoss clipping specification, probability guardrails |

---

*This document defines the authoritative metric contract for F1 Predictor evaluation.*