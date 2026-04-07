"""
MongoDB Pace Loader

Loads pace observations from MongoDB f1_pace_observations collection.
Replaces file-based TracingInsights and Kaggle loaders.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from pymongo.database import Database


@dataclass
class PaceObservation:
    _id: str
    year: int
    round: int
    circuit_ref: str
    constructor_ref: str
    pace_delta_ms: float
    avg_pace_ms: float
    min_pace_ms: float
    sample_size: int
    
    @classmethod
    def from_dict(cls, doc: Dict) -> "PaceObservation":
        return cls(
            _id=doc["_id"],
            year=doc["year"],
            round=doc["round"],
            circuit_ref=doc["circuit_ref"],
            constructor_ref=doc["constructor_ref"],
            pace_delta_ms=doc.get("pace_delta_ms", 0.0),
            avg_pace_ms=doc.get("avg_pace_ms", 0.0),
            min_pace_ms=doc.get("min_pace_ms", 0.0),
            sample_size=doc.get("sample_size", 0),
        )


class MongoPaceLoader:
    """Loads pace observations from MongoDB."""
    
    def __init__(self, db: Database):
        self.db = db
    
    def load_pace_observations(
        self,
        years: List[int],
        source: Optional[str] = None
    ) -> Dict[str, List[PaceObservation]]:
        """
        Load pace observations grouped by constructor.
        
        Returns:
            Dict mapping constructor_ref to list of PaceObservation
        """
        query: Dict = {"year": {"$in": years}}
        
        if source:
            query["source"] = source
        
        cursor = self.db.f1_pace_observations.find(query).sort([
            ("year", 1),
            ("round", 1)
        ])
        
        result: Dict[str, List[PaceObservation]] = {}
        for doc in cursor:
            obs = PaceObservation.from_dict(doc)
            if obs.constructor_ref not in result:
                result[obs.constructor_ref] = []
            result[obs.constructor_ref].append(obs)
        
        return result
    
    def load_race_pace(
        self,
        year: int,
        round_num: int
    ) -> Dict[str, float]:
        """
        Load pace deltas for a specific race.
        
        Returns:
            Dict mapping constructor_ref to pace_delta_ms
        """
        docs = self.db.f1_pace_observations.find({
            "year": year,
            "round": round_num,
        })
        
        return {doc["constructor_ref"]: doc.get("pace_delta_ms", 0.0) for doc in docs}
    
    def get_recent_pace(
        self,
        constructor_ref: str,
        circuit_type: str,
        n_races: int = 5
    ) -> float:
        """
        Get recent average pace for a constructor on a circuit type.
        Used as fallback when no direct pace data is available.
        """
        pipeline = [
            {"$match": {"constructor_ref": constructor_ref}},
            {"$lookup": {
                "from": "f1_races",
                "let": {"year": "$year", "round": "$round"},
                "pipeline": [
                    {"$match": {
                        "$expr": {
                            "$and": [
                                {"$eq": ["$year", "$$year"]},
                                {"$eq": ["$round", "$$round"]}
                            ]
                        },
                        "circuit_type": circuit_type,
                    }}
                ],
                "as": "race_info"
            }},
            {"$match": {"race_info": {"$ne": []}}},
            {"$sort": {"year": -1, "round": -1}},
            {"$limit": n_races},
            {"$group": {
                "_id": None,
                "avg_pace_delta": {"$avg": "$pace_delta_ms"},
            }}
        ]
        
        result = list(self.db.f1_pace_observations.aggregate(pipeline))
        return result[0]["avg_pace_delta"] if result else 0.0
    
    def get_constructor_medians(
        self,
        years: List[int]
    ) -> Dict[str, float]:
        """
        Get median pace for each constructor across all races.
        Used for normalizing pace deltas.
        """
        pipeline = [
            {"$match": {"year": {"$in": years}}},
            {"$group": {
                "_id": "$constructor_ref",
                "median_pace": {"$median": "$avg_pace_ms"},
            }}
        ]
        
        result = list(self.db.f1_pace_observations.aggregate(pipeline))
        return {r["_id"]: r["median_pace"] for r in result}
    
    def count_observations(self, years: List[int]) -> int:
        """Count total pace observations for given years."""
        return self.db.f1_pace_observations.count_documents(
            {"year": {"$in": years}}
        )
