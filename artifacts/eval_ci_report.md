# Bootstrap Validation Report

## Confidence Intervals (95%)

| Metric | Mean | 95% CI | Target | Status |
|--------|------|--------|--------|--------|
| Kendall tau | 0.4282 | [0.3837, 0.4698] | > 0.45 | FAIL |
| Brier | 0.1475 | [0.1247, 0.1705] | < 0.20 | PASS |
| LogLoss | 5.5076 | [4.6478, 6.2830] | < 1.0 | - |
| RPS | 0.1506 | [0.1400, 0.1606] | < 0.15 | - |
| ECE | 0.0152 | N/A | < 0.05 | PASS |

## Notes

- Confidence intervals computed using 1000 bootstrap samples
- Based on 27 validation races (2023-2024)
- LogLoss and RPS are indicative (no hard targets)
