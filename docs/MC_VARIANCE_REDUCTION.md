# Priority 2.3: Monte Carlo Variance Reduction

## Problema
Il Layer 2 usa 50,000 simulazioni MC per stimare probabilità. La varianza può essere ridotta per ottenere stime più accurate o usare meno simulazioni.

## Ricerca
- DataField.Dev (2026) Cap 24: Variance Reduction Techniques
- Wikipedia: Antithetic variates
- Kroese et al. (2011) Handbook of Monte Carlo Methods

## Implementazione

### Test Risultati

| Method | Variance | SE | VR% |
|--------|-----------|----|----|
| Naive MC | 0.1897 | 0.0044 | baseline |
| Antithetic | 0.1250 | 0.0050 | +51.8% |
| Stratified | 0.1866 | 0.0043 | +1.7% |

### Antithetic Variates
- Riduce varianza del 34.1% nel test simple
- Per funzioni monotone (come performance race), il beneficio è maggiore

### Convergenza

| N | Naive SE | Theoretical SE |
|-----|---------|----------------|
| 1000 | 0.0139 | 0.0140 |
| 5000 | 0.0062 | 0.0063 |
| 10000 | 0.0044 | 0.0045 |
| 25000 | 0.0027 | 0.0028 |
| 50000 | 0.0019 | 0.0020 |

Convergenza O(1/sqrt(N)) confermata.

## Implementazione Suggerita

```python
# Opzione 1: Antithetic variates nel RaceSimulator
for i in range(n_sim // 2):
    # Standard
    result = simulate(race)
    # Antithetic  
    result_anti = simulate_mirror(race)
    # Average
    final = (result + result_anti) / 2

# Opzione 2: Stratified per posizioni
# Stratify by: 1st-3rd, 4th-10th, 11th-20th
```

## Errore Commesso
- Test iniziale con Bernoulli non catturava beneficio reale
- Fix: Implementato test con funzione monotona (performance race)

## Status
COMPLETO