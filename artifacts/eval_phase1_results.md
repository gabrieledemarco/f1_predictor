# Evaluation Overhaul - Phase 1 Results

## Baseline (Before)
- Kendall τ = 0.448
- Brier (winner-only) = 0.2767
- ROI = +0.0% (synthetic)
- ECE = 0.0149
- RPS = N/A (not computed)
- LogLoss = N/A (not computed)

## After Phase 1 (walk-forward metrics overhaul)
- Kendall τ = 0.446
- Brier (multiclass) = 0.1626 ✓ (target <= 0.20)
- LogLoss (multiclass) = 9.91 (target <= 1.0) - NOTE: needs investigation
- RPS mean = 0.1513 (target <= 0.15) - very close to target
- ECE = 0.0153 ✓ (target <= 0.05)
- ROI = +0.0% (still synthetic, Phase 2 pending)
- Data coverage: 27 races with position dist, 0 races with odds

## Key Changes Implemented
1. **Brier Multiclass**: Full position distribution vs actual one-hot
2. **LogLoss**: Added multiclass log loss with clipping (NOTE: high values ~10 need investigation)
3. **RPS**: Added Ranked Probability Score using validation/metrics.py utility
4. **Reporting**: Updated _log_metrics and print_summary for new metrics
5. **Data coverage**: Added tracking for races with position distribution

## Tests Created
- tests/validation/test_walkforward_metrics.py - 11 tests passing
- Tests cover Brier multiclass, LogLoss, RPS, and backtest integration

## Known Issues
- LogLoss ~10 is unexpectedly high (expected ~1-3 for reasonable predictions)
- Need to investigate probability distribution normalization

## Next Steps (for future work)
1. Debug LogLoss calculation - check for zero probability rows
2. Phase 2: Integrate real backtest with BettingBacktester when odds become available