# L1b State Reduction A/B Test

## Configuration

| Model | State Dimensions | Components |
|-------|------------------|------------|
| baseline_12D | 12 | base + top_speed + continuous_pct + throttle_pct + street + desert + hybrid + temp |
| reduced_8D | 8 | base + top_speed + continuous_pct + throttle_pct (no circuit types) |

## Rationale

- L1b significance analysis showed circuit type coefficients unstable (z~0.8)
- Base intercept significant (z~2.5), continuous features moderately (z~1.8-2.0)
- Removing 4 unstable circuit type features should reduce noise

## Results

| Metric | Baseline (12D) | Reduced (8D) | Delta | 95% CI (Reduced) |
|--------|----------------|-------------|------|------------------|
| Kendall tau | 0.451 | 0.457 | +0.005 | [0.416, 0.498] |
| Brier | 0.1640 | 0.1585 | -0.0051 | - |
| ECE | 0.0150 | 0.0136 | -0.0015 | - |
| RPS | 0.1520 | 0.1501 | -0.0014 | - |
| LogLoss | 5.39 | 5.44 | +0.08 | - |

## Bootstrap Distribution

- Reduced model tau improvement: **+1.1%**
- P(reduced > baseline): **75.0%**

## Final Gates

| Gate | Threshold | Reduced Model | Status |
|------|-----------|---------------|--------|
| Kendall tau | >= 0.45 | 0.457 | PASS |
| Brier | < 0.20 | 0.1585 | PASS |
| ECE | < 0.05 | 0.0136 | PASS |

## Verdict: **GO** - L1b state reduction recommended

## Recommendation

1. Implement reduced 8D state (remove circuit type coefficients)
2. Expected improvement: ~0.5-1.5% in Kendall tau
3. Reduced noise in Kalman estimates
