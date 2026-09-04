# src/weather_api.py
import requests
import logging
from typing import Dict, Any

class WeatherAdapter:
    """Fetches real-time weather signals with zero-failure fallback."""
    
    # Common agricultural district fallback coordinates
    DISTRICT_COORDS = {
        'Noida': {'lat': 28.5355, 'lon': 77.3910},
        'Sangrur': {'lat': 30.2458, 'lon': 75.8421},
        'Ludhiana': {'lat': 30.9010, 'lon': 75.8573},
        'Nashik': {'lat': 20.0059, 'lon': 73.7898},
        'Ganganagar': {'lat': 29.9038, 'lon': 73.8772}
    }

    @classmethod
    def get_district_weather(cls, district: str) -> Dict[str, Any]:
        """Queries Open-Meteo free API (no API key required) for weather data."""
        dist_clean = district.title()
        coords = cls.DISTRICT_COORDS.get(dist_clean, {'lat': 28.6139, 'lon': 77.2090}) # Default Delhi
        
        url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&current_weather=true"
        
        try:
            response = requests.get(url, timeout=3.0)
            if response.status_code == 200:
                data = response.json().get('current_weather', {})
                temp = data.get('temperature', 25.0)
                weather_code = data.get('weathercode', 0)
                
                # WMO Weather Code interpretation
                if weather_code in [51, 53, 55, 61, 63, 65, 80, 81, 82]:
                    condition = "Heavy Rain"
                    factor = -0.10  # Supply transit delay
                elif weather_code in [95, 96, 99]:
                    condition = "Storm"
                    factor = -0.15
                elif temp > 40.0:
                    condition = "Heatwave"
                    factor = 0.05   # Perishable urgency
                else:
                    condition = "Normal"
                    factor = 0.0
                    
                return {
                    "temperature_c": temp,
                    "condition": condition,
                    "weather_adjustment_factor": factor,
                    "is_live_api": True
                }
        except Exception as e:
            logging.warning(f"Weather API unavailable: {str(e)}. Using safe fallback.")
            
        return {
            "temperature_c": 25.0,
            "condition": "Normal",
            "weather_adjustment_factor": 0.0,
            "is_live_api": False
        }