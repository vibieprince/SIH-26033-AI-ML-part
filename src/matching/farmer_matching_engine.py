from dataclasses import dataclass
from typing import List
import math

@dataclass
class FarmerProfile:
    farmer_id: str
    name: str
    crop: str
    available_quantity_kg: float
    latitude: float
    longitude: float
    harvest_ready: bool

@dataclass
class MatchResult:
    farmer_id: str
    name: str
    match_score: float
    distance_km: float
    available_quantity_kg: float
    reason: str

class FarmerMatchingEngine:
    def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0  # Earth radius in kilometers
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def match_farmers(
        self,
        crop: str,
        demand_lat: float,
        demand_lon: float,
        required_gap_kg: float,
        candidate_farmers: List[FarmerProfile],
        max_radius_km: float = 150.0
    ) -> List[MatchResult]:
        
        matches = []
        for f in candidate_farmers:
            if f.crop.lower() != crop.lower() or not f.harvest_ready:
                continue

            dist = self._haversine(demand_lat, demand_lon, f.latitude, f.longitude)
            if dist > max_radius_km:
                continue

            # Compute Match Score: Proximity (50%) + Quantity Fulfillment Ratio (50%)
            dist_score = max(0.0, 100.0 - (dist / max_radius_km * 100.0))
            qty_score = min(100.0, (f.available_quantity_kg / (required_gap_kg + 1e-5)) * 100.0)
            
            final_match = round(0.6 * dist_score + 0.4 * qty_score, 1)

            matches.append(MatchResult(
                farmer_id=f.farmer_id,
                name=f.name,
                match_score=final_match,
                distance_km=round(dist, 1),
                available_quantity_kg=f.available_quantity_kg,
                reason=f"Ready crop match located {round(dist, 1)}km from demand hub."
            ))

        return sorted(matches, key=lambda x: x.match_score, reverse=True)