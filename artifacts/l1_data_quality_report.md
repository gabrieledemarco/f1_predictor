# L1 Data Quality Report

## Constructor Coverage

| Constructor | Races with Obs | Coverage % |
|-------------|----------------|-------------|
| mercedes | 65 | 98.5% |
| ferrari | 65 | 98.5% |
| mclaren | 64 | 97.0% |
| alpha_tauri | 64 | 97.0% |
| haas | 64 | 97.0% |
| aston_martin | 64 | 97.0% |
| alfa_romeo | 63 | 95.5% |
| alpine | 63 | 95.5% |
| red_bull | 63 | 95.5% |
| williams | 63 | 95.5% |

## Season Coverage

| Year | Coverage % |
|------|------------|
| 2022.0 | 100.0% |
| 2023.0 | 100.0% |
| 2024.0 | 100.0% |

## Pace Value Distribution

- Mean: -0.459 s/lap
- Std: 1.494 s/lap
- Min: -5.721 s/lap
- Max: 5.279 s/lap
- Outliers (>3 IQR): 0 (0.0%)

## Circuit Type Coverage

- street: 91%
- high_speed: 95%
- hybrid: 91%
- temp: 94%

## Key Findings

1. **Overall Coverage**: 95%+ for most constructors
2. **Data Quality**: Pace values have reasonable distribution
3. **Outliers**: Small percentage (<2%) - may need filtering
4. **Circuit Types**: All well covered (>90%)

## Recommendations

- **Low priority**: Current coverage is sufficient
- **Medium**: Consider outlier clipping at 3 IQR
- **Low**: Circuit type coverage is adequate
