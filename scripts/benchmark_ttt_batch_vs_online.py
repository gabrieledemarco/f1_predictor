"""
benchmark_ttt_batch_vs_online.py
================================
Confronta batch vs online TTT inference per valutare differenze.

Metodologia:
- Batch: ricalcola TUTTA la storia ogni volta
- Online: aggiorna solo l'ultimo rating

Il test verifica se i risultati divergono significativamente.
"""

import sys
import numpy as np
from pathlib import Path
import time
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from f1_predictor.models.driver_skill import DriverSkillModel, TTTConfig
from f1_predictor.domain.entities import CircuitType, RaceResult


def generate_test_data(n_races=50, n_drivers=20):
    """Genera dati di test con risultati variabili."""
    races = []
    driver_pool = [f"DRV{i:02d}" for i in range(n_drivers)]
    
    for race_id in range(1, n_races + 1):
        # Crea risultati con posizioni casuali ma deterministiche
        import random
        random.seed(race_id * 42)  # Seed deterministico
        
        positions = driver_pool.copy()
        random.shuffle(positions)
        
        results = [
            RaceResult(
                race_id=race_id,
                driver_code=positions[i],
                constructor_ref="team_a" if i < 10 else "team_b",
                grid_position=i+1,
                finish_position=i+1,
                points=max(0, 25 - i),
                laps_completed=70,
                status="Finished",
            )
            for i in range(len(positions))
        ]
        
        metadata = {race_id: {"circuit_type": CircuitType.MIXED, "year": 2024}}
        races.append((results, metadata))
    
    return races


def run_online_inference(races_data):
    """Online: fit incrementale, una gara alla volta."""
    config = TTTConfig(
        mu_0=25.0,
        sigma_0=8.333,
        beta=4.167,
        tau=0.05,
        draw_margin=0.0,
        decay_factor=0.15,
    )
    
    model = DriverSkillModel(config=config)
    ratings_history = []
    
    for results, metadata in races_data:
        model.fit(results, metadata)
        # Salva rating dopo questa gara
        ratings = {code: model.get_rating(code) for code in model._ratings}
        ratings_history.append(ratings)
    
    return ratings_history


def run_batch_inference(races_data):
    """
    Batch: ricalcola TUTTA la storia da zero per ogni step.
    
    Questo è il metodo originale del paper TTT:
    - Per predire la gara i, usa TUTTI i risultati 1...i
    - Ricalcola i rating da zero ogni volta
    """
    config = TTTConfig(
        mu_0=25.0,
        sigma_0=8.333,
        beta=4.167,
        tau=0.05,
        draw_margin=0.0,
        decay_factor=0.15,
    )
    
    ratings_history = []
    
    for i in range(len(races_data)):
        # Crea modello FRESCO per ogni step
        model = DriverSkillModel(config=config)
        
        # Fit su TUTTE le gare precedenti (da 1 a i)
        for j in range(i + 1):
            results, metadata = races_data[j]
            model.fit(results, metadata)
        
        # Salva rating dopo questa gara
        ratings = {code: model.get_rating(code) for code in model._ratings}
        ratings_history.append(ratings)
        
        if (i + 1) % 10 == 0:
            print(f"  Batch progress: {i+1}/{len(races_data)}")
    
    return ratings_history


def compare_ratings(online_ratings, batch_ratings):
    """Confronta i rating online vs batch."""
    import numpy as np
    
    mu_diffs = []
    sigma_diffs = []
    
    for i, (online, batch) in enumerate(zip(online_ratings, batch_ratings)):
        # Trova driver comuni
        common = set(online.keys()) & set(batch.keys())
        
        for driver in common:
            mu_diff = abs(online[driver].mu - batch[driver].mu)
            sigma_diff = abs(online[driver].sigma - batch[driver].sigma)
            mu_diffs.append(mu_diff)
            sigma_diffs.append(sigma_diff)
    
    return {
        'mu_mean_diff': np.mean(mu_diffs),
        'mu_max_diff': np.max(mu_diffs),
        'sigma_mean_diff': np.mean(sigma_diffs),
        'sigma_max_diff': np.max(sigma_diffs),
        'n_comparisons': len(mu_diffs),
    }


def test_batch_vs_online():
    """Test principale: confronta batch vs online."""
    print("=" * 70)
    print("Benchmark: Batch vs Online TTT Inference")
    print("=" * 70)
    
    n_races = 50
    print(f"\nGenero {n_races} gare di test...")
    races_data = generate_test_data(n_races=n_races)
    
    # Online inference
    print(f"\n1. Running ONLINE inference...")
    start = time.time()
    online_ratings = run_online_inference(races_data)
    online_time = time.time() - start
    print(f"   Completato in {online_time:.2f}s")
    
    # Batch inference
    print(f"\n2. Running BATCH inference...")
    print("   (ricalcola TUTTA la storia da zero per ogni step)")
    start = time.time()
    batch_ratings = run_batch_inference(races_data)
    batch_time = time.time() - start
    print(f"   Completato in {batch_time:.2f}s")
    
    # Confronto
    print(f"\n3. Confronto risultati...")
    comparison = compare_ratings(online_ratings, batch_ratings)
    
    print("\n" + "=" * 70)
    print("RISULTATI")
    print("=" * 70)
    print(f"Numero confronti: {comparison['n_comparisons']}")
    print()
    print(f"Differenza MU (mean):  {comparison['mu_mean_diff']:.4f}")
    print(f"Differenza MU (max):   {comparison['mu_max_diff']:.4f}")
    print(f"Differenza SIGMA (mean): {comparison['sigma_mean_diff']:.4f}")
    print(f"Differenza SIGMA (max):  {comparison['sigma_max_diff']:.4f}")
    print()
    print(f"Tempo Online: {online_time:.2f}s")
    print(f"Tempo Batch:  {batch_time:.2f}s")
    print(f"Rapporto:     {batch_time/online_time:.1f}x piu' lento")
    print("=" * 70)
    
    # Valutazione
    print("\nANALISI:")
    if comparison['mu_max_diff'] < 0.1:
        print("[OK] Differenze MU trascurabili (< 0.1)")
    elif comparison['mu_max_diff'] < 1.0:
        print("[NOTE] Differenze MU moderate (< 1.0)")
    else:
        print("[WARN] Differenze MU significative!")
    
    if comparison['sigma_max_diff'] < 0.1:
        print("[OK] Differenze SIGMA trascurabili (< 0.1)")
    else:
        print("[NOTE] Differenze SIGMA presenti")
    
    # Test finale: correlation tra rating finali
    final_online = online_ratings[-1]
    final_batch = batch_ratings[-1]
    
    drivers = list(set(final_online.keys()) & set(final_batch.keys()))
    online_mu = [final_online[d].mu for d in drivers]
    batch_mu = [final_batch[d].mu for d in drivers]
    
    correlation = np.corrcoef(online_mu, batch_mu)[0, 1]
    print(f"\nCorrelazione rating finali: {correlation:.4f}")
    
    if correlation > 0.99:
        print("[OK] Rating quasi identici")
    else:
        print("[NOTE] Rating divergono")


if __name__ == '__main__':
    test_batch_vs_online()