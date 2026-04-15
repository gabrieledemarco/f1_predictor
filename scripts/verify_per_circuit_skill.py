"""
verify_per_circuit_skill.py
======================
Verifica che il per-circuit-type skill funzioni correttamente.

Test:
1. Crea DriverSkillModel
2. Fit su dati di un driver su diversi circuiti con RISULTATI DIVERSI
3. Verifica rating globale vs circuit-type diverso
"""

import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from f1_predictor.models.driver_skill import DriverSkillModel, TTTConfig
from f1_predictor.domain.entities import CircuitType, RaceResult


def test_per_circuit_skill():
    """Test per-circuit-type skill blending."""
    print("=" * 60)
    print("Test: Per-Circuit-Type Skill Blending")
    print("=" * 60)
    
    # Config con tau ottimizzato
    config = TTTConfig(
        mu_0=25.0,
        sigma_0=8.333,
        beta=4.167,
        tau=0.05,  
        draw_margin=0.0,
        decay_factor=0.15,
    )
    
    model = DriverSkillModel(config=config)
    
    # Driver di test - VER performa diversamente su circuiti diversi!
    # STREET: vince SEMPRE (10 vittorie)
    # HIGH_SPEED: perde spesso (solo 2 vittorie)  
    # MIXED: medio (5 vittorie)
    
    race_id = 1
    
    # STREET circuiti (10 gare - VER vince sempre)
    for i in range(10):
        results = [
            RaceResult(race_id=race_id, driver_code="VER", constructor_ref="red_bull",
                     grid_position=1, finish_position=1, points=25, laps_completed=70, status="Finished"),
            RaceResult(race_id=race_id, driver_code="HAM", constructor_ref="mercedes",
                     grid_position=2, finish_position=2, points=18, laps_completed=70, status="Finished"),
            RaceResult(race_id=race_id, driver_code="LEC", constructor_ref="ferrari",
                     grid_position=3, finish_position=3, points=15, laps_completed=70, status="Finished"),
        ] + [
            RaceResult(race_id=race_id, driver_code=f"DRV{y}", constructor_ref="other",
                     grid_position=i+4, finish_position=i+4, points=max(0, 12-i), 
                     laps_completed=70, status="Finished")
            for y in range(17)
        ]
        # Passa race_metadata con circuit_type!
        # HIGH_SPEED circuiti (10 gare - VER male)
        for i in range(10):
            results = [
                RaceResult(race_id=race_id, driver_code="HAM", constructor_ref="mercedes",
                        grid_position=1, finish_position=1, points=25, laps_completed=70, status="Finished"),
                RaceResult(race_id=race_id, driver_code="VER", constructor_ref="red_bull",
                        grid_position=2, finish_position=5, points=10, laps_completed=70, status="Finished"),
                RaceResult(race_id=race_id, driver_code="LEC", constructor_ref="ferrari",
                        grid_position=3, finish_position=3, points=15, laps_completed=70, status="Finished"),
            ] + [
                RaceResult(race_id=race_id, driver_code=f"DRV{y}", constructor_ref="other",
                        grid_position=i+4, finish_position=i+4, points=max(0, 12-i), 
                        laps_completed=70, status="Finished")
                for y in range(17)
            ]
            metadata = {race_id: {"circuit_type": CircuitType.HIGH_SPEED, "year": 2024}}
            model.fit(results, metadata)
            race_id += 1
        
        # MIXED circuiti (10 gare - VER medio)
        for i in range(10):
            results = [
                RaceResult(race_id=race_id, driver_code="VER", constructor_ref="red_bull",
                        grid_position=1, finish_position=2 if i % 2 == 0 else 1, 
                        points=18 if i % 2 == 0 else 25, laps_completed=70, status="Finished"),
                RaceResult(race_id=race_id, driver_code="HAM", constructor_ref="mercedes",
                        grid_position=2, finish_position=1 if i % 2 == 0 else 3, 
                        points=25 if i % 2 == 0 else 15, laps_completed=70, status="Finished"),
                RaceResult(race_id=race_id, driver_code="LEC", constructor_ref="ferrari",
                        grid_position=3, finish_position=3, points=15, laps_completed=70, status="Finished"),
            ] + [
                RaceResult(race_id=race_id, driver_code=f"DRV{y}", constructor_ref="other",
                        grid_position=i+4, finish_position=i+4, points=max(0, 12-i), 
                        laps_completed=70, status="Finished")
                for y in range(17)
            ]
            metadata = {race_id: {"circuit_type": CircuitType.MIXED, "year": 2024}}
            model.fit(results, metadata)
            race_id += 1
    
    print(f"Fit completati: 30 gare (10 per 3 circuiti)")
    print()
    
    # Verifica rating
    print("Global rating:")
    r_global = model.get_rating("VER")
    print(f"  mu={r_global.mu:.2f}, sigma={r_global.sigma:.2f}")
    print()
    
    for circuit in [CircuitType.STREET, CircuitType.HIGH_SPEED, CircuitType.MIXED]:
        r = model.get_rating("VER", circuit)
        print(f"{circuit.value} rating:")
        print(f"  mu={r.mu:.2f}, sigma={r.sigma:.2f}")
    print()
    
    # Test: STREET dovrebbe essere il rating piu' alto (10vittorie)
    r_street = model.get_rating("VER", CircuitType.STREET)
    r_high = model.get_rating("VER", CircuitType.HIGH_SPEED)
    r_mixed = model.get_rating("VER", CircuitType.MIXED)
    
    print("Test risultati:")
    print(f"  STREET mu={r_street.mu:.2f} (10vittorie)")
    print(f"  HIGH_SPEED mu={r_high.mu:.2f} (perse)")  
    print(f"  MIXED mu={r_mixed.mu:.2f} (metà)")
    print()
    
    # Verifica ordine corretto
    success = True
    
    if r_street.mu > r_mixed.mu:
        print("[OK] STREET > MIXED")
    else:
        print("[ERR] STREET <= MIXED")
        success = False
        
    if r_mixed.mu > r_high.mu:
        print("[OK] MIXED > HIGH_SPEED")
    else:
        print("[ERR] MIXED <= HIGH_SPEED")
        success = False
    
    # Sigma dovrebbe essere piu' basso per circuiti con piu' dati
    if r_street.sigma < r_global.sigma:
        print(f"[OK] sigma STREET ({r_street.sigma:.2f}) < sigma GLOBAL ({r_global.sigma:.2f})")
    else:
        print(f"[NOTE] sigma STREET >= sigma GLOBAL")
    
    print()
    if success:
        print("=== TUTTI I TEST PASSATI ===")
    else:
        print("=== ALCUNI TEST FALLITI ===")


if __name__ == '__main__':
    test_per_circuit_skill()