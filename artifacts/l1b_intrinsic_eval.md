# L1b (Machine Pace) Intrinsic Evaluation

## Overall Metrics

- **RMSE**: 0.2960 s/lap
- **MAE**: 0.2342 s/lap
- **Mean Rank Correlation**: 0.7098

## Per Constructor

| Constructor | Bias | Std | RMSE |
|-------------|------|-----|------|
| alfa_romeo | 0.0097 | 0.2454 | 0.2437 |
| alpha_tauri | 0.0391 | 0.3216 | 0.3216 |
| alpine | -0.0019 | 0.2818 | 0.2797 |
| aston_martin | -0.0511 | 0.3573 | 0.3583 |
| ferrari | 0.0751 | 0.3066 | 0.3134 |
| haas | 0.0739 | 0.3102 | 0.3166 |
| mclaren | 0.0110 | 0.2847 | 0.2827 |
| mercedes | -0.0307 | 0.2735 | 0.2731 |
| red_bull | -0.0327 | 0.2899 | 0.2895 |
| williams | 0.0260 | 0.2656 | 0.2649 |

## Per Circuit Type

| Circuit Type | Bias | Std | RMSE |
|-------------|------|-----|------|
| high_speed | 0.0347 | 0.3119 | 0.3129 |
| hybrid | 0.0083 | 0.3081 | 0.3074 |
| street | 0.0017 | 0.2690 | 0.2682 |
| temp | 0.0010 | 0.2933 | 0.2923 |

## Model Complexity Comparison

| Model | RMSE |
|-------|------|
| Full (8D) | 0.2960 |
| Reduced (5D) | 0.3223 |

**Finding**: Full model performs similarly to reduced - may be slightly over-parameterized

## Verdict

- **L1b Status**: NEEDS IMPROVEMENT
- RMSE of 0.30s is acceptable but could be better
- Rank correlation of {mean_rank_corr:.2f} shows predictive power
- Circuit type effects are captured but may be noisy
