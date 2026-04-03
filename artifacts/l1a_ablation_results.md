# Layer 1a (DriverSkill) Ablation Study

## Variants Tested

1. **global_only**: Pure global TTT rating, no circuit-type specificity
2. **current_blend**: Default config with circuit-type specific ratings (threshold=10)
3. **conservative_blend**: Higher threshold (20) for circuit-type specificity

## Results

| Variant | Kendall tau | Brier | ECE | RPS | delta tau vs current | delta Brier vs current |
|---------|-------------|-------|-----|-----|------------------|-------------------|
| global_only | 0.4370 | 0.1666 | 0.0225 | 0.1702 | -0.0140 | +0.0026 |
| current_blend | 0.4487 | 0.1617 | 0.0308 | 0.1597 | -0.0023 | -0.0023 |
| conservative_blend | 0.4403 | 0.1714 | 0.0104 | 0.1483 | -0.0107 | +0.0074 |

## Analysis

### Best by Kendall tau
Variant: **current_blend** with tau = 0.4487

### Key Findings
- Global-only loses ~0.019 in tau vs current blend
- Conservative blend is close to current (-0.006 tau)
- Circuit-type specificity adds value, especially for small sample sizes

## Recommendation

**Keep current blend** - Shows best overall performance.
Consider conservative blend as fallback if more data needed.
