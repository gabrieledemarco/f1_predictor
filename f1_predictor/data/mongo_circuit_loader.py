"""
MongoDB Circuit Profiles Loader

Loads circuit profiles from MongoDB f1_circuit_profiles collection.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from pymongo.database import Database


class CircuitType(Enum):
    STREET = "street"
    HIGH_SPEED = "high_speed"
    HIGH_DOWNFORCE = "high_df"
    HIGH_ALTITUDE = "high_altitude"
    MIXED = "mixed"
    DESERT = "desert"


@dataclass
class CircuitSpeedProfile:
    circuit_type: CircuitType
    top_speed_kmh: float
    min_speed_kmh: float
    avg_speed_kmh: float
    avg_slow_corner_kmh: float = 0.0
    avg_medium_corner_kmh: float = 0.0
    avg_fast_corner_kmh: float = 0.0
    full_throttle_pct: float = 0.0
    source: str = ""
    
    @classmethod
    def from_dict(cls, doc: Dict) -> "CircuitSpeedProfile":
        circuit_type_str = doc.get("circuit_type", "mixed")
        try:
            circuit_type = CircuitType(circuit_type_str)
        except ValueError:
            circuit_type = CircuitType.MIXED
        
        return cls(
            circuit_type=circuit_type,
            top_speed_kmh=doc.get("top_speed_kmh", 0.0),
            min_speed_kmh=doc.get("min_speed_kmh", 0.0),
            avg_speed_kmh=doc.get("avg_speed_kmh", 0.0),
            avg_slow_corner_kmh=doc.get("avg_slow_corner_kmh", 0.0),
            avg_medium_corner_kmh=doc.get("avg_medium_corner_kmh", 0.0),
            avg_fast_corner_kmh=doc.get("avg_fast_corner_kmh", 0.0),
            full_throttle_pct=doc.get("full_throttle_pct", 0.0),
            source=doc.get("source", ""),
        )


DEFAULT_PROFILES: Dict[str, CircuitSpeedProfile] = {
    "bahrain": CircuitSpeedProfile(CircuitType.DESERT, 318.5, 85.2, 205.3, full_throttle_pct=62.1),
    "jeddah": CircuitSpeedProfile(CircuitType.STREET, 322.0, 105.0, 235.0, full_throttle_pct=73.0),
    "albert_park": CircuitSpeedProfile(CircuitType.STREET, 315.0, 95.0, 210.0, full_throttle_pct=65.0),
    "suzuka": CircuitSpeedProfile(CircuitType.HIGH_SPEED, 330.0, 80.0, 215.0, full_throttle_pct=68.0),
    "monaco": CircuitSpeedProfile(CircuitType.STREET, 300.0, 50.0, 160.0, full_throttle_pct=45.0),
    "spa": CircuitSpeedProfile(CircuitType.HIGH_SPEED, 335.0, 75.0, 225.0, full_throttle_pct=72.0),
    "monza": CircuitSpeedProfile(CircuitType.HIGH_SPEED, 360.0, 95.0, 245.0, full_throttle_pct=78.0),
    "silverstone": CircuitSpeedProfile(CircuitType.HIGH_SPEED, 330.0, 70.0, 220.0, full_throttle_pct=70.0),
    "catalunya": CircuitSpeedProfile(CircuitType.MIXED, 320.0, 85.0, 210.0, full_throttle_pct=63.0),
    "hungaroring": CircuitSpeedProfile(CircuitType.MIXED, 305.0, 75.0, 195.0, full_throttle_pct=55.0),
    "singapore": CircuitSpeedProfile(CircuitType.STREET, 310.0, 60.0, 175.0, full_throttle_pct=50.0),
    "austin": CircuitSpeedProfile(CircuitType.MIXED, 325.0, 85.0, 215.0, full_throttle_pct=66.0),
    "mexico_city": CircuitSpeedProfile(CircuitType.HIGH_ALTITUDE, 360.0, 90.0, 195.0, full_throttle_pct=55.0),
    "brazil": CircuitSpeedProfile(CircuitType.MIXED, 330.0, 80.0, 210.0, full_throttle_pct=62.0),
    "las_vegas": CircuitSpeedProfile(CircuitType.STREET, 315.0, 75.0, 200.0, full_throttle_pct=58.0),
    "abu_dhabi": CircuitSpeedProfile(CircuitType.DESERT, 325.0, 90.0, 215.0, full_throttle_pct=65.0),
    "imola": CircuitSpeedProfile(CircuitType.MIXED, 320.0, 80.0, 210.0, full_throttle_pct=64.0),
    "miami": CircuitSpeedProfile(CircuitType.STREET, 320.0, 85.0, 205.0, full_throttle_pct=60.0),
    "shanghai": CircuitSpeedProfile(CircuitType.DESERT, 325.0, 80.0, 210.0, full_throttle_pct=62.0),
    "zandvoort": CircuitSpeedProfile(CircuitType.HIGH_SPEED, 320.0, 75.0, 215.0, full_throttle_pct=67.0),
    "baku": CircuitSpeedProfile(CircuitType.STREET, 370.0, 95.0, 215.0, full_throttle_pct=60.0),
    "qatar": CircuitSpeedProfile(CircuitType.DESERT, 330.0, 85.0, 205.0, full_throttle_pct=60.0),
}


class MongoCircuitProfileLoader:
    """Loads circuit profiles from MongoDB."""
    
    def __init__(self, db: Database):
        self.db = db
    
    def load_profile(self, circuit_ref: str) -> Optional[CircuitSpeedProfile]:
        """Load a specific circuit profile."""
        doc = self.db.f1_circuit_profiles.find_one({"_id": circuit_ref})
        
        if doc:
            return CircuitSpeedProfile.from_dict(doc)
        
        return DEFAULT_PROFILES.get(circuit_ref)
    
    def load_all_profiles(self) -> Dict[str, CircuitSpeedProfile]:
        """Load all circuit profiles, falling back to defaults."""
        profiles: Dict[str, CircuitSpeedProfile] = {}
        
        cursor = self.db.f1_circuit_profiles.find({})
        for doc in cursor:
            circuit_ref = doc["_id"]
            profiles[circuit_ref] = CircuitSpeedProfile.from_dict(doc)
        
        for circuit_ref, profile in DEFAULT_PROFILES.items():
            if circuit_ref not in profiles:
                profiles[circuit_ref] = profile
        
        return profiles
    
    def get_circuit_type(self, circuit_ref: str) -> CircuitType:
        """Get circuit type, defaulting to MIXED."""
        profile = self.load_profile(circuit_ref)
        return profile.circuit_type if profile else CircuitType.MIXED
    
    def count_profiles(self) -> int:
        """Count stored circuit profiles."""
        return self.db.f1_circuit_profiles.count_documents({})
