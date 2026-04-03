# Layer 1b (Kalman) Significance Analysis

## Methodology

For each constructor × state variable pair:
1. Track z = x / sqrt(P) over time
2. Compute % time |z| > 1.96 (significant)
3. Compute sign stability (% of races without sign change)

## Classification

- **strong**: |mean_z| > 1.96 AND pct_unstable < 20%
- **weak**: |mean_z| > 1.0 OR pct_unstable < 40%
- **unstable**: otherwise

## Results by State Variable

### base_intercept

| Constructor | Mean z | Std z | % Unstable | Sign Stable | Class |
|-------------|--------|-------|------------|--------------|-------|
| mercedes | 2.96 | 1.09 | 76.9% | 100.0% | weak |
| alpha_tauri | 2.89 | 1.19 | 88.1% | 97.6% | weak |
| ferrari | 2.70 | 1.36 | 70.0% | 96.6% | weak |
| aston_martin | 2.55 | 1.02 | 66.7% | 100.0% | weak |
| red_bull | 2.52 | 1.07 | 69.1% | 100.0% | weak |

### top_speed_coef

| Constructor | Mean z | Std z | % Unstable | Sign Stable | Class |
|-------------|--------|-------|------------|--------------|-------|
| alpine | 2.01 | 1.18 | 56.6% | 88.0% | weak |
| aston_martin | 2.01 | 0.99 | 53.1% | 96.8% | weak |
| mercedes | 2.01 | 0.96 | 45.7% | 100.0% | weak |
| williams | 1.99 | 1.13 | 54.1% | 94.4% | weak |
| haas | 1.97 | 1.30 | 51.5% | 90.8% | weak |

### continuous_pct_coef

| Constructor | Mean z | Std z | % Unstable | Sign Stable | Class |
|-------------|--------|-------|------------|--------------|-------|
| haas | 1.98 | 1.22 | 48.1% | 94.7% | weak |
| mercedes | 1.97 | 1.03 | 46.6% | 97.2% | weak |
| ferrari | 1.94 | 1.35 | 50.0% | 81.3% | weak |
| williams | 1.85 | 1.07 | 53.8% | 87.5% | weak |
| alpha_tauri | 1.84 | 1.02 | 45.2% | 93.3% | weak |

### throttle_pct_coef

| Constructor | Mean z | Std z | % Unstable | Sign Stable | Class |
|-------------|--------|-------|------------|--------------|-------|
| red_bull | 1.61 | 1.03 | 40.5% | 94.4% | weak |
| mclaren | 1.51 | 1.20 | 36.2% | 82.5% | weak |
| alpha_tauri | 1.32 | 1.30 | 32.4% | 75.8% | weak |
| haas | 1.27 | 1.24 | 31.2% | 80.3% | weak |
| mercedes | 1.23 | 1.28 | 31.1% | 60.0% | weak |

### street_circuit_coef

| Constructor | Mean z | Std z | % Unstable | Sign Stable | Class |
|-------------|--------|-------|------------|--------------|-------|
| mercedes | 1.12 | 1.19 | 24.5% | 68.8% | weak |
| alpha_tauri | 0.96 | 1.15 | 19.0% | 67.7% | weak |
| ferrari | 0.93 | 1.29 | 22.4% | 68.4% | weak |
| mclaren | 0.91 | 1.19 | 20.3% | 68.3% | weak |
| aston_martin | 0.88 | 1.12 | 18.6% | 70.7% | weak |

### desert_circuit_coef

| Constructor | Mean z | Std z | % Unstable | Sign Stable | Class |
|-------------|--------|-------|------------|--------------|-------|
| red_bull | 1.21 | 1.24 | 26.7% | 82.8% | weak |
| mclaren | 1.03 | 1.36 | 27.8% | 65.7% | weak |
| mercedes | 0.90 | 1.44 | 31.4% | 55.9% | weak |
| alfa_romeo | 0.79 | 1.26 | 15.2% | 66.7% | weak |
| aston_martin | 0.74 | 1.20 | 12.5% | 64.5% | weak |

### hybrid_circuit_coef

| Constructor | Mean z | Std z | % Unstable | Sign Stable | Class |
|-------------|--------|-------|------------|--------------|-------|
| alfa_romeo | 1.12 | 1.12 | 22.7% | 60.5% | weak |
| aston_martin | 1.01 | 1.28 | 24.6% | 65.0% | weak |
| alpha_tauri | 0.94 | 1.20 | 18.8% | 61.7% | weak |
| ferrari | 0.90 | 1.10 | 18.4% | 59.5% | weak |
| red_bull | 0.89 | 0.87 | 9.7% | 71.8% | weak |

### temp_circuit_coef

| Constructor | Mean z | Std z | % Unstable | Sign Stable | Class |
|-------------|--------|-------|------------|--------------|-------|
| haas | 1.35 | 1.21 | 34.6% | 80.4% | weak |
| red_bull | 0.95 | 1.17 | 22.4% | 75.0% | weak |
| mclaren | 0.92 | 1.12 | 19.7% | 66.2% | weak |
| alfa_romeo | 0.81 | 1.29 | 25.9% | 50.9% | weak |
| aston_martin | 0.76 | 1.06 | 12.1% | 63.2% | weak |

## Constructor Summary

| Constructor | Mean z | % Strong | % Unstable |
|-------------|--------|----------|------------|
| alfa_romeo | 1.29 | 0% | 33.5% |
| alpha_tauri | 1.34 | 0% | 35.6% |
| alpine | 1.28 | 0% | 36.7% |
| aston_martin | 1.34 | 0% | 32.4% |
| ferrari | 1.34 | 0% | 30.8% |
| haas | 1.39 | 0% | 32.9% |
| mclaren | 1.34 | 0% | 32.1% |
| mercedes | 1.46 | 0% | 36.3% |
| red_bull | 1.44 | 0% | 34.1% |
| williams | 1.24 | 0% | 34.8% |

## Interpretation

- **base_intercept**: Most stable, captures baseline pace
- **top_speed/continuous**: Moderate signal, constructor-specific
- **circuit_type dummies**: Lower significance, more noise
- Some constructors show high instability (likely due to limited data)

## Pruning Recommendations

Circuit type coefficients are candidates for pruning: []
