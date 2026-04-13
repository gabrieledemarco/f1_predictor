"""
MongoDB Race Loader

Loads race data from MongoDB f1_races collection.
Replaces file-based JolpicaLoader.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from pymongo.database import Database


@dataclass
class RaceResult:
    driver_code: str
    driver_id: str  # Added to match MongoDB structure
    constructor_ref: str
    grid_position: int
    finish_position: Optional[int]
    points: float
    laps_completed: int
    status: str
    fastest_lap_rank: Optional[int] = None
    fastest_lap_time: Optional[str] = None


@dataclass
class QualifyingResult:
    driver_code: str
    grid_position: int
    q1: Optional[str]
    q2: Optional[str]
    q3: Optional[str]


@dataclass
class Race:
    _id: str
    year: int
    round: int
    circuit_ref: str
    circuit_name: str
    circuit_type: str
    race_name: str
    date: str
    is_sprint_weekend: bool
    is_season_end: bool
    is_major_regulation_change: bool
    results: List[RaceResult]
    qualifying: List[QualifyingResult]
    location: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, doc: Dict) -> "Race":
        results = [
            RaceResult(
                driver_code=r.get("driver_code", ""),
                driver_id=r.get("driver_id", ""),
                constructor_ref=r.get("constructor_ref", ""),
                grid_position=r.get("grid_position", 0),
                finish_position=r.get("finish_position"),
                points=float(r.get("points", 0)),
                laps_completed=r.get("laps_completed", 0),
                status=r.get("status", ""),
                fastest_lap_rank=r.get("fastest_lap_rank"),
                fastest_lap_time=r.get("fastest_lap_time"),
            )
            for r in doc.get("results", [])
        ]
        qualifying = [
            QualifyingResult(
                driver_code=q.get("driver_code", ""),
                grid_position=q.get("grid_position", 0),
                q1=q.get("q1"),
                q2=q.get("q2"),
                q3=q.get("q3"),
            )
            for q in doc.get("qualifying", [])
        ]
        
        return cls(
            _id=doc["_id"],
            year=doc["year"],
            round=doc["round"],
            circuit_ref=doc.get("circuit_ref", ""),
            circuit_name=doc.get("circuit_name", ""),
            circuit_type=doc.get("circuit_type", "mixed"),
            race_name=doc.get("race_name", ""),
            date=doc.get("date", ""),
            is_sprint_weekend=doc.get("is_sprint_weekend", False),
            is_season_end=doc.get("is_season_end", False),
            is_major_regulation_change=doc.get("is_major_regulation_change", False),
            results=results,
            qualifying=qualifying,
            location=doc.get("location", {}),
        )
    
    def to_dict(self) -> Dict:
        return {
            "_id": self._id,
            "year": self.year,
            "round": self.round,
            "circuit_ref": self.circuit_ref,
            "circuit_name": self.circuit_name,
            "circuit_type": self.circuit_type,
            "race_name": self.race_name,
            "date": self.date,
            "is_sprint_weekend": self.is_sprint_weekend,
            "is_season_end": self.is_season_end,
            "is_major_regulation_change": self.is_major_regulation_change,
            "results": [
                {
                    "driver_code": r.driver_code,
                    "constructor_ref": r.constructor_ref,
                    "grid_position": r.grid_position,
                    "finish_position": r.finish_position,
                    "points": r.points,
                    "laps_completed": r.laps_completed,
                    "status": r.status,
                    "fastest_lap_rank": r.fastest_lap_rank,
                    "fastest_lap_time": r.fastest_lap_time,
                }
                for r in self.results
            ],
            "qualifying": [
                {
                    "driver_code": q.driver_code,
                    "grid_position": q.grid_position,
                    "q1": q.q1,
                    "q2": q.q2,
                    "q3": q.q3,
                }
                for q in self.qualifying
            ],
            "location": self.location,
        }


class MongoRaceLoader:
    """Loads race data from MongoDB."""
    
    def __init__(self, db: Database):
        self.db = db
    
    def load_seasons(
        self,
        years: List[int],
        through_round: Optional[int] = None
    ) -> List[Race]:
        """Load multiple seasons of race data."""
        query: Dict[str, Any] = {"year": {"$in": years}}
        
        if through_round is not None:
            query["round"] = {"$lte": through_round}
        
        cursor = self.db.f1_races.find(query).sort([
            ("year", 1),
            ("round", 1)
        ])
        
        return [Race.from_dict(doc) for doc in cursor]
    
    def load_season(self, year: int, through_round: Optional[int] = None) -> List[Race]:
        """Load a single season."""
        return self.load_seasons([year], through_round)
    
    def load_race(self, year: int, round_num: int) -> Optional[Race]:
        """Load a specific race."""
        doc = self.db.f1_races.find_one({"_id": f"{year}_{round_num:02d}"})
        return Race.from_dict(doc) if doc else None
    
    def get_circuit_refs(self, years: List[int]) -> List[str]:
        """Get unique circuit refs for given years."""
        docs = self.db.f1_races.distinct("circuit_ref", {"year": {"$in": years}})
        return docs
    
    def get_driver_codes(self, years: List[int]) -> List[str]:
        """Get unique driver codes from race results."""
        pipeline = [
            {"$match": {"year": {"$in": years}}},
            {"$unwind": "$results"},
            {"$group": {"_id": "$results.driver_code"}},
            {"$sort": {"_id": 1}}
        ]
        result = list(self.db.f1_races.aggregate(pipeline))
        return [r["_id"] for r in result]
    
    def get_constructor_refs(self, years: List[int]) -> List[str]:
        """Get unique constructor refs from race results."""
        pipeline = [
            {"$match": {"year": {"$in": years}}},
            {"$unwind": "$results"},
            {"$group": {"_id": "$results.constructor_ref"}},
            {"$sort": {"_id": 1}}
        ]
        result = list(self.db.f1_races.aggregate(pipeline))
        return [r["_id"] for r in result]
    
    def count_races(self, years: List[int]) -> int:
        """Count total races for given years."""
        return self.db.f1_races.count_documents({"year": {"$in": years}})
