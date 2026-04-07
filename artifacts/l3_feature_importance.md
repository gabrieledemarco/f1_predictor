# Layer 3 (Ensemble) Feature Significance Analysis

## Methodology

Analysis based on:
1. Standardized Ridge coefficients
2. Permutation importance (OOS)
3. Drop-column importance

## Results by Feature Group

### L1a

| Feature | Std Coef | Perm Importance | Drop Importance |
|---------|----------|-----------------|----------------|
| skill_mu | 0.3124 | 0.0385 | 0.0160 |
| skill_conservative | 0.2174 | 0.0360 | 0.0140 |
| skill_sigma | 0.3796 | 0.0147 | 0.0073 |

### L1b

| Feature | Std Coef | Perm Importance | Drop Importance |
|---------|----------|-----------------|----------------|
| pace_sigma | 0.2665 | 0.0082 | 0.0035 |
| pace_mu | 0.2416 | 0.0053 | 0.0098 |

### L2

| Feature | Std Coef | Perm Importance | Drop Importance |
|---------|----------|-----------------|----------------|
| exp_pos_mc | 0.5280 | 0.0671 | 0.0160 |
| p_win_mc | 0.3917 | 0.0383 | 0.0257 |
| p_dnf_mc | 0.3697 | 0.0375 | 0.0210 |
| p_podium_mc | 0.5160 | 0.0375 | 0.0284 |

### Context

| Feature | Std Coef | Perm Importance | Drop Importance |
|---------|----------|-----------------|----------------|
| circuit_type | 0.3591 | 0.0293 | 0.0131 |
| grid_pos | 0.0114 | 0.0218 | 0.0055 |
| grid_quali_delta | 0.0860 | 0.0134 | 0.0057 |
| has_penalty | -0.1563 | 0.0120 | 0.0118 |

### Historical

| Feature | Std Coef | Perm Importance | Drop Importance |
|---------|----------|-----------------|----------------|
| elo_delta_vs_field | -0.0897 | 0.0092 | 0.0017 |
| dnf_rate_relative | 0.0988 | 0.0038 | 0.0028 |
| h2h_win_rate_3season | 0.0320 | 0.0021 | 0.0027 |

## Group Aggregates (by Perm Importance)

| Group | Mean Perm Importance |
|-------|----------------------|
| L2 | 0.0451 |
| L1a | 0.0297 |
| Context | 0.0191 |
| L1b | 0.0067 |
| Historical | 0.0050 |

## Interpretation

- **L2 (Monte Carlo)** is the strongest predictor - captures race simulation dynamics
- **L1a (Driver Skill)** provides good signal through TTT ratings
- **L1b (Machine Pace)** adds constructor-level info
- **Context (grid)** matters especially for top positions
- **Historical** features have limited data and low importance

## Pruning Recommendations

Low importance candidates for pruning: ['h2h_win_rate_3season']
