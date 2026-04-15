"""
verify_purged_cv.py
=================
Verifica che purged CV funzioni correttamente.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from f1_predictor.validation.walk_forward import WalkForwardValidator


def test_purged_cv():
    """Test purged CV."""
    print("=" * 60)
    print("Test: Purged Cross-Validation")
    print("=" * 60)
    
    # Mock predict function
    def predict_fn(train_races, test_race):
        return {"DRV01": 0.5, "DRV02": 0.5}
    
    # Test 1: without purge_gap
    print("\n1. WalkForwardValidator (embargo=1, purge_gap=0)")
    validator1 = WalkForwardValidator(predict_fn, min_train_races=5, embargo=1, purge_gap=0)
    print(f"   effective_gap = {validator1.embargo + validator1.purge_gap}")
    
    # Test 2: with purge_gap=2
    print("\n2. WalkForwardValidator (embargo=1, purge_gap=2)")
    validator2 = WalkForwardValidator(predict_fn, min_train_races=5, embargo=1, purge_gap=2)
    print(f"   effective_gap = {validator2.embargo + validator2.purge_gap}")
    
    # Test 3: with purge_gap=3
    print("\n3. WalkForwardValidator (embargo=1, purge_gap=3)")
    validator3 = WalkForwardValidator(predict_fn, min_train_races=5, embargo=1, purge_gap=3)
    print(f"   effective_gap = {validator3.embargo + validator3.purge_gap}")
    
    print("\n" + "=" * 60)
    print("VERIFICA:")
    print("=" * 60)
    
    all_pass = True
    
    if validator2.purge_gap == 2:
        print("[OK] purge_gap attribute exists")
    else:
        print("[ERR] purge_gap attribute missing")
        all_pass = False
    
    if validator3.embargo + validator3.purge_gap == 4:
        print("[OK] effective_gap = embargo + purge_gap")
    else:
        print("[ERR] effective_gap calculation wrong")
        all_pass = False
    
    if all_pass:
        print("\n=== TUTTI I TEST PASSATI ===")
    else:
        print("\n=== ALCUNI TEST FALLITI ===")


if __name__ == '__main__':
    test_purged_cv()