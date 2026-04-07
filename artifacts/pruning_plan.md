# Pruning Plan

## Candidates

### light

- **Removed**: ['dnf_rate_relative', 'h2h_win_rate_3season']
- **Rationale**: Historical features have lowest permutation importance
- **Expected impact**: minimal - these features contribute < 1% to ensemble

### strict

- **Removed**: ['dnf_rate_relative', 'h2h_win_rate_3season', 'has_penalty', 'grid_quali_delta']
- **Rationale**: Historical + low-signal context features
- **Expected impact**: small - may improve generalization

## Comparison Results

| Model | Kendall tau | Brier | ECE | RPS | Features Removed | Delta tau | Delta Brier |
|-------|-------------|-------|-----|-----|-------------------|-----------|-------------|
| baseline | 0.4510 | 0.1640 | 0.0150 | 0.1520 | 0 | +0.0000 | +0.0000 |
| pruned_light | 0.4485 | 0.1685 | 0.0159 | 0.1530 | 0 | -0.0025 | +0.0045 |
| pruned_strict | 0.4349 | 0.1563 | 0.0123 | 0.1550 | 0 | -0.0161 | -0.0077 |

## Recommendation

**Recommended: baseline**

- Maintains Kendall tau at 0.4510
- No regression in Brier/ECE
- Similar RPS

## Next Steps

1. Implement pruned feature configuration
2. Run full validation on test set
3. Monitor for degradation over time
