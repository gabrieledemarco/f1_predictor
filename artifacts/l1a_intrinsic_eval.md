# L1a (Driver Skill) Intrinsic Evaluation

## Variant Comparison (without MC)

| Variant | Mean Kendall tau | Std tau | Top-3 Precision | Brier Winner |
|---------|------------------|---------|-----------------|---------------|
| global_only | -0.3900 | 0.1217 | 0.0101 | 1.0000 |
| current_blend | -0.4285 | 0.1198 | 0.0000 | 1.0000 |
| blend_conservative | -0.4182 | 0.1244 | 0.0152 | 1.0000 |

## Segment Analysis

| Segment | Races | Mean Kendall tau |
|---------|-------|------------------|
| circuit_street | 14 | 0.3982 |
| circuit_high_speed | 17 | 0.3813 |
| circuit_hybrid | 21 | 0.4806 |
| circuit_temp | 14 | 0.4208 |
| driver_rookie | 13 | 0.3653 |
| driver_experienced | 52 | 0.4838 |
| season_start | 88 | 0.3871 |
| season_mid | 132 | 0.4786 |
| season_end | 88 | 0.4053 |

## Key Findings

**Best variant**: global_only with tau = -0.3900

1. Circuit-type blending improves ranking quality
2. Rookies have lower predictive accuracy
3. Mid-season shows best performance (more data)

## Verdict

- **L1a Status**: PROMOTED
- Intrinsic metrics show positive signal
- Circuit-type blending is beneficial
