"""
MongoDB Odds Loader

Loads Pinnacle odds from MongoDB f1_pinnacle_odds collection.
Replaces file-based OddsLoader.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from pymongo.database import Database


@dataclass
class OddsRecord:
    _id: str
    race_id: str
    event_id: str
    market: str
    driver_code: str
    odd_decimal: float
    p_implied_raw: float
    p_novig: float
    hours_to_race: Optional[float]
    fetched_at: str
    
    @classmethod
    def from_dict(cls, doc: Dict) -> "OddsRecord":
        return cls(
            _id=str(doc["_id"]),
            race_id=doc["race_id"],
            event_id=doc.get("event_id", ""),
            market=doc["market"],
            driver_code=doc["driver_code"],
            odd_decimal=doc["odd_decimal"],
            p_implied_raw=doc.get("p_implied_raw", 0.0),
            p_novig=doc.get("p_novig", 0.0),
            hours_to_race=doc.get("hours_to_race"),
            fetched_at=doc.get("fetched_at", ""),
        )


@dataclass
class CalibrationRecord:
    race_id: str
    driver_code: str
    market: str
    p_model_raw: float
    p_pinnacle_novig: float
    outcome: int
    timestamp: str


class MongoOddsLoader:
    """Loads Pinnacle odds from MongoDB."""
    
    def __init__(self, db: Database):
        self.db = db
    
    def load_race_odds(
        self,
        year: int,
        round_num: int,
        market: Optional[str] = None
    ) -> List[OddsRecord]:
        """Load odds for a specific race."""
        race_id = f"{year}_{round_num:02d}"
        
        query: Dict = {"race_id": race_id}
        if market:
            query["market"] = market
        
        cursor = self.db.f1_pinnacle_odds.find(query)
        return [OddsRecord.from_dict(doc) for doc in cursor]
    
    def load_historical_odds(
        self,
        years: List[int],
        market: Optional[str] = None,
        min_hours_to_race: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[OddsRecord]:
        """
        Load historical odds for multiple years.
        
        Args:
            years: List of years to load
            market: Filter by market (h2h, outrights)
            min_hours_to_race: Minimum hours before race
            limit: Maximum number of records
        """
        pipeline = []
        
        pipeline.append({
            "$addFields": {
                "year": {
                    "$toInt": {"$arrayElemAt": [{"$split": ["$race_id", "_"]}, 0]}
                }
            }
        })
        
        pipeline.append({"$match": {"year": {"$in": years}}})
        
        if market:
            pipeline.append({"$match": {"market": market}})
        
        if min_hours_to_race is not None:
            pipeline.append({
                "$match": {"hours_to_race": {"$gte": min_hours_to_race}}
            })
        
        pipeline.append({"$sort": {"fetched_at": -1}})
        
        if limit:
            pipeline.append({"$limit": limit})
        
        cursor = self.db.f1_pinnacle_odds.aggregate(pipeline)
        return [OddsRecord.from_dict(doc) for doc in cursor]
    
    def load_recent_odds(
        self,
        days: int = 7
    ) -> List[OddsRecord]:
        """Load odds from the last N days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        cursor = self.db.f1_pinnacle_odds.find({
            "fetched_at": {"$gte": cutoff.isoformat() + "Z"}
        }).sort("fetched_at", -1)
        
        return [OddsRecord.from_dict(doc) for doc in cursor]
    
    def get_upcoming_odds(
        self,
        hours_ahead: float = 48.0
    ) -> List[OddsRecord]:
        """Get odds for races happening within N hours."""
        cursor = self.db.f1_pinnacle_odds.find({
            "hours_to_race": {"$lte": hours_ahead, "$gte": 0}
        }).sort("race_id", 1)
        
        return [OddsRecord.from_dict(doc) for doc in cursor]
    
    def count_odds(self, years: List[int]) -> int:
        """Count total odds records for given years."""
        pipeline = [
            {"$addFields": {
                "year": {
                    "$toInt": {"$arrayElemAt": [{"$split": ["$race_id", "_"]}, 0]}
                }
            }},
            {"$match": {"year": {"$in": years}}},
            {"$count": "total"}
        ]
        
        result = list(self.db.f1_pinnacle_odds.aggregate(pipeline))
        return result[0]["total"] if result else 0
    
    def get_available_markets(self, year: int, round_num: int) -> List[str]:
        """Get available market types for a race."""
        race_id = f"{year}_{round_num:02d}"
        
        pipeline = [
            {"$match": {"race_id": race_id}},
            {"$group": {"_id": "$market"}},
            {"$sort": {"_id": 1}}
        ]
        
        result = list(self.db.f1_pinnacle_odds.aggregate(pipeline))
        return [r["_id"] for r in result]
