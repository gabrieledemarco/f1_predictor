#!/usr/bin/env python3
"""
Sector Times Loader - Integration with prediction pipeline.
"""
from typing import Optional, Dict
from pymongo.database import Database


class SectorTimesLoader:
    """Loads sector times from f1_session_stats collection."""
    
    TRACING_TO_RACES = {
        "abu_dhabi": "yas_marina",
        "australian": "albert_park",
        "austrian": "red_bull_ring",
        "azerbaijan": "baku",
        "british": "silverstone",
        "canadian": "villeneuve",
        "chinese": "shanghai",
        "dutch": "zandvoort",
        "emilia_romagna": "imola",
        "french": "catalunya",
        "hungarian": "hungaroring",
        "italian": "monza",
        "japanese": "suzuka",
        "las_vegas": "vegas",
        "mexico_city": "rodriguez",
        "qatar": "losail",
        "saudi_arabian": "jeddah",
        "singapore": "marina_bay",
        "spanish": "catalunya",
        "são_paulo": "interlagos",
        "united_states": "americas",
    }
    
    def __init__(self, db: Database):
        self.db = db
    
    def get_sector_times(self, year: int, circuit_ref: str, driver_code: str) -> Optional[Dict]:
        """
        Get sector times for a specific driver at a specific race.
        
        Args:
            year: Race year
            circuit_ref: Circuit reference (from f1_races, e.g., "yas_marina")
            driver_code: Driver code (e.g., "VER", "HAM")
        
        Returns:
            Dict with s1_best_ms, s2_best_ms, s3_best_ms, total_best_ms or None
        """
        # Map from our circuit_ref to TracingInsights format
        reverse_map = {v: k for k, v in self.TRACING_TO_RACES.items()}
        ti_circuit = reverse_map.get(circuit_ref)
        
        if not ti_circuit:
            ti_circuit = circuit_ref  # Try direct
        
        # Try direct lookup with mapped circuit
        doc = self.db.f1_session_stats.find_one({
            "year": year,
            "circuit_ref": ti_circuit,
            "driver_code": driver_code,
            "session_type": "qualifying"
        })
        
        if doc:
            return {
                "s1_best_ms": doc.get("s1_best_ms"),
                "s2_best_ms": doc.get("s2_best_ms"),
                "s3_best_ms": doc.get("s3_best_ms"),
                "total_best_ms": doc.get("total_best_ms")
            }
        
        # Also try direct with original circuit_ref
        doc = self.db.f1_session_stats.find_one({
            "year": year,
            "circuit_ref": circuit_ref,
            "driver_code": driver_code,
            "session_type": "qualifying"
        })
        
        if doc:
            return {
                "s1_best_ms": doc.get("s1_best_ms"),
                "s2_best_ms": doc.get("s2_best_ms"),
                "s3_best_ms": doc.get("s3_best_ms"),
                "total_best_ms": doc.get("total_best_ms")
            }
        
        return None
    
    def get_race_medians(self, year: int, circuit_ref: str) -> Optional[Dict]:
        """Get median sector times for all drivers at a race (for delta calculation)."""
        # Map to TracingInsights format
        reverse_map = {v: k for k, v in self.TRACING_TO_RACES.items()}
        ti_circuit = reverse_map.get(circuit_ref)
        
        query_circuit = ti_circuit if ti_circuit else circuit_ref
        
        docs = list(self.db.f1_session_stats.find({
            "year": year,
            "circuit_ref": query_circuit,
            "session_type": "qualifying"
        }))
        
        if not docs:
            # Try direct
            docs = list(self.db.f1_session_stats.find({
                "year": year,
                "circuit_ref": circuit_ref,
                "session_type": "qualifying"
            }))
        
        if not docs:
            return None
        
        s1_times = [d.get("s1_best_ms") for d in docs if d.get("s1_best_ms")]
        s2_times = [d.get("s2_best_ms") for d in docs if d.get("s2_best_ms")]
        s3_times = [d.get("s3_best_ms") for d in docs if d.get("s3_best_ms")]
        total_times = [d.get("total_best_ms") for d in docs if d.get("total_best_ms")]
        
        return {
            "s1_median": sum(s1_times) / len(s1_times) if s1_times else None,
            "s2_median": sum(s2_times) / len(s2_times) if s2_times else None,
            "s3_median": sum(s3_times) / len(s3_times) if s3_times else None,
            "total_median": sum(total_times) / len(total_times) if total_times else None
        }
    
    def compute_deltas(self, year: int, circuit_ref: str, driver_code: str) -> Optional[Dict]:
        """
        Compute sector time deltas vs race median.
        
        Returns:
            Dict with s1_delta_ms, s2_delta_ms, s3_delta_ms, total_sector_delta_ms
            (negative = faster than median = better)
        """
        sector_times = self.get_sector_times(year, circuit_ref, driver_code)
        if not sector_times:
            return None
        
        medians = self.get_race_medians(year, circuit_ref)
        if not medians:
            return None
        
        deltas = {}
        if sector_times.get("s1_best_ms") and medians.get("s1_median"):
            deltas["s1_delta_ms"] = sector_times["s1_best_ms"] - medians["s1_median"]
        if sector_times.get("s2_best_ms") and medians.get("s2_median"):
            deltas["s2_delta_ms"] = sector_times["s2_best_ms"] - medians["s2_median"]
        if sector_times.get("s3_best_ms") and medians.get("s3_median"):
            deltas["s3_delta_ms"] = sector_times["s3_best_ms"] - medians["s3_median"]
        if sector_times.get("total_best_ms") and medians.get("total_median"):
            deltas["total_sector_delta_ms"] = sector_times["total_best_ms"] - medians["total_median"]
        
        return deltas if deltas else None


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    from pymongo import MongoClient
    
    uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    client = MongoClient(uri)
    db = client["betbreaker"]
    
    loader = SectorTimesLoader(db)
    
    # Test: Get sector times for VER at a known race
    result = loader.compute_deltas(2024, "yas_marina", "VER")
    print(f"VER at Abu Dhabi 2024: {result}")
    
    # Check what we have
    count = db.f1_session_stats.count_documents({})
    print(f"\nTotal session stats: {count}")