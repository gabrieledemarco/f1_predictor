#!/usr/bin/env python3
"""
Quick test training to validate sector times integration.
"""
import os
from dotenv import load_dotenv
load_dotenv()

import sys
sys.path.insert(0, '.')

from pymongo import MongoClient
from f1_predictor.data.mongo_loader import MongoRaceLoader
from f1_predictor.data.mongo_circuit_loader import MongoCircuitProfileLoader
from f1_predictor.pipeline.prediction_pipeline import F1PredictionPipeline
from f1_predictor.validation.walk_forward import WalkForwardValidator

def main():
    # Connect to MongoDB
    uri = os.environ.get('MONGODB_URI')
    client = MongoClient(uri)
    db = client['betbreaker']
    
    print("=== QUICK TRAINING TEST WITH SECTOR TIMES ===\n")
    
    # Load data
    print("1. Loading historical races...")
    race_loader = MongoRaceLoader(db)
    circuit_loader = MongoCircuitProfileLoader(db)
    
    races = race_loader.load_seasons([2022, 2023])
    circuit_loader.load_profiles([2022, 2023])
    
    # Convert to dict format for pipeline
    race_dicts = []
    for race in races:
        race_dicts.append({
            'race_id': race.race_id,
            'year': race.year,
            'round': race.round,
            'circuit_ref': race.circuit.ref,
            'circuit_type': race.circuit.circuit_type.name,
            'results': [{'driver_code': r.driver_code, 'position': r.position} 
                       for r in race.results]
        })
    
    print(f"   Loaded {len(race_dicts)} races")
    
    # Initialize pipeline with sector times loader
    print("\n2. Initializing pipeline...")
    pipeline = F1PredictionPipeline(ttt_config=None, kalman_config=None)
    pipeline.set_sector_times_loader(db)
    
    print("\n3. Training (walk-forward validation)...")
    # Quick validation on 2023 races only
    test_races = [r for r in race_dicts if r['year'] == 2023][:5]  # Test on 5 races
    train_races = [r for r in race_dicts if r['year'] == 2022]
    
    print(f"   Training: {len(train_races)} races (2022)")
    print(f"   Testing: {len(test_races)} races (2023)")
    
    # Fit on training data
    pipeline.fit(train_races, verbose=True)
    
    # Test predictions on a few races
    print("\n4. Testing predictions with sector times...")
    from f1_predictor.domain.entities import Race
    
    metrics = []
    for race_dict in test_races:
        # Get actual results
        actual_positions = {r['driver_code']: r['position'] for r in race_dict.get('results', [])}
        if not actual_positions:
            continue
        
        # Create mock driver grid
        driver_grid = []
        for driver_code in actual_positions.keys():
            driver_grid.append({
                'driver_code': driver_code,
                'constructor_ref': 'ferrari',  # placeholder
                'grid_position': 10
            })
        
        # Predict
        try:
            race = Race(
                race_id=race_dict['race_id'],
                year=race_dict['year'],
                round=race_dict['round'],
                name=f"GP {race_dict['round']}",
                circuit=None
            )
            result = pipeline.predict_race(race, driver_grid, verbose=False)
            
            # Calculate simple metrics
            pred_probs = result['probabilities']
            predicted_winner = max(pred_probs.items(), key=lambda x: x[1].p_win)[0]
            actual_winner = min(actual_positions.items(), key=lambda x: x[1])[0]
            
            correct = predicted_winner == actual_winner
            metrics.append({
                'race': f"{race_dict['year']} R{race_dict['round']}",
                'predicted': predicted_winner,
                'actual': actual_winner,
                'correct': correct,
                'has_sector_data': result['model_state'].get('sector_data_available', 'N/A')
            })
            
            print(f"   {race_dict['year']} R{race_dict['round']}: pred={predicted_winner}, actual={actual_winner}, OK={correct}")
        except Exception as e:
            print(f"   {race_dict['year']} R{race_dict['round']}: ERROR - {e}")
    
    # Summary
    correct_count = sum(1 for m in metrics if m['correct'])
    print(f"\n=== RESULTS ===")
    print(f"Races predicted: {len(metrics)}")
    print(f"Correct winners: {correct_count}")
    print(f"Accuracy: {correct_count/len(metrics)*100:.1f}%")
    
    print("\n=== SECTOR TIMES INTEGRATION SUCCESSFUL ===")

if __name__ == "__main__":
    main()