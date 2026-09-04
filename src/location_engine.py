# src/location_engine.py
import math
import requests
import logging
from typing import Dict, Tuple, Any

class LocationEngine:
    """Universal Geocoding and Nearest District Spatial Resolver."""

    # Major district centroids present in our 16.5M mandi dataset
    DISTRICT_CENTROIDS = {
        "Sangrur": (30.2458, 75.8421),
        "Ludhiana": (30.9010, 75.8573),
        "Amritsar": (31.6340, 74.8723),
        "Gautam Buddha Nagar": (28.5355, 77.3910), # Noida / Greater Noida
        "Agra": (27.1767, 78.0081),
        "Varanasi": (25.3176, 82.9739),
        "Nashik": (20.0059, 73.7898),
        "Pune": (18.5204, 73.8567),
        "Ahmednagar": (19.0948, 74.7480),
        "Ganganagar": (29.9038, 73.8772),
        "Jaipur": (26.9124, 75.7873),
        "Indore": (22.7196, 75.8577),
        "Kolar": (13.1367, 78.1292)
    }

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculates distance in kilometers between two GPS coordinates."""
        R = 6371.0 # Earth radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    @classmethod
    def auto_detect_ip_location(cls) -> Dict[str, Any]:
        """Auto-detects user location based on public IP address."""
        try:
            res = requests.get("http://ip-api.com/json/", timeout=3.0).json()
            if res.get("status") == "success":
                return {
                    "city": res.get("city", "Delhi"),
                    "region": res.get("regionName", "Delhi"),
                    "lat": float(res.get("lat", 28.6139)),
                    "lon": float(res.get("lon", 77.2090)),
                    "method": "IP_GEOLOCATION"
                }
        except Exception as e:
            logging.warning(f"IP auto-detect failed: {e}")
            
        # Fallback to Delhi default
        return {"city": "Delhi", "region": "Delhi", "lat": 28.6139, "lon": 77.2090, "method": "DEFAULT"}

    @classmethod
    def geocode_query(cls, query: str) -> Dict[str, Any]:
        """Geocodes any free-text Indian place name (e.g. 'Khanna', 'Baramati', 'Noida') using OpenStreetMap."""
        if not query or len(query.strip()) < 2:
            return cls.auto_detect_ip_location()

        url = f"https://nominatim.openstreetmap.org/search?q={query},India&format=json&limit=1"
        headers = {"User-Agent": "KisanGuard_Hackathon_App/1.0"}
        
        try:
            res = requests.get(url, headers=headers, timeout=3.0)
            if res.status_code == 200 and res.json():
                data = res.json()[0]
                return {
                    "city": data.get("display_name", query).split(",")[0],
                    "region": query,
                    "lat": float(data["lat"]),
                    "lon": float(data["lon"]),
                    "method": "GEOCODED_SEARCH"
                }
        except Exception as e:
            logging.warning(f"Geocoding failed for {query}: {e}")

        return cls.auto_detect_ip_location()

    @classmethod
    def resolve_nearest_model_district(cls, lat: float, lon: float) -> Tuple[str, float]:
        """Finds the closest trained agricultural district in our ML dataset by spatial distance."""
        min_dist = float("inf")
        best_district = "Gautam Buddha Nagar"
        
        for district, coords in cls.DISTRICT_CENTROIDS.items():
            dist = cls.haversine_distance(lat, lon, coords[0], coords[1])
            if dist < min_dist:
                min_dist = dist
                best_district = district

        return best_district, round(min_dist, 1)