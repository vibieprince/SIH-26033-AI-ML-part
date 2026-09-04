# src/weather_engine.py
"""
Weather Engine – Open-Meteo API integration.

Returns a normalized dict with a `weather_risk_factor` key (negative for
supply-disrupting events like heavy rain, positive for demand-boosting
events like heatwaves) in addition to the legacy `weather_adjustment_factor`
alias.
"""

import logging
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)

# WMO weather interpretation codes
# https://open-meteo.com/en/docs#weathervariables
_RAIN_STORM_CODES  = {51, 53, 55, 61, 63, 65, 67, 80, 81, 82, 85, 86, 95, 96, 99}
_SNOW_CODES        = {71, 73, 75, 77}
_FOG_CODES         = {45, 48}


class WeatherEngine:
    """Fetches real-time weather using exact latitude & longitude from Open-Meteo."""

    @classmethod
    def get_live_weather(cls, lat: float, lon: float) -> Dict[str, Any]:
        """
        Queries Open-Meteo current-weather endpoint.

        Returns
        -------
        dict with keys:
            temperature_c              : float – current temperature in Celsius
            condition                  : str   – human-readable condition label
            weather_adjustment_factor  : float – demand/supply impact coefficient
                                                 negative = supply disruption
                                                 positive = demand urgency boost
            weather_risk_factor        : float – alias of weather_adjustment_factor
            weather_code               : int   – raw WMO code
        """
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current_weather=true"
            f"&hourly=precipitation_probability"
            f"&forecast_days=1"
        )

        try:
            res = requests.get(url, timeout=4.0)
            if res.status_code == 200:
                data = res.json().get("current_weather", {})
                temp         = float(data.get("temperature", 25.0))
                weather_code = int(data.get("weathercode", 0))
                return cls._interpret(temp, weather_code)
        except Exception as exc:
            logger.warning("Open-Meteo query failed (%s). Using neutral fallback.", exc)

        return cls._neutral_fallback()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _interpret(temp: float, code: int) -> Dict[str, Any]:
        """Translate raw WMO code + temperature into an actionable weather dict."""
        if code in _RAIN_STORM_CODES:
            condition = "Heavy Rain / Storm"
            factor    = -0.12  # Transport disruption → supply drops by ~12%
        elif code in _SNOW_CODES:
            condition = "Snowfall / Cold Wave"
            factor    = -0.08  # Harvest access disrupted
        elif code in _FOG_CODES:
            condition = "Dense Fog"
            factor    = -0.05  # Road logistics affected
        elif temp > 40.0:
            condition = "Severe Heatwave (>40°C)"
            factor    = 0.12   # Perishable spoilage urgency lifts demand
        elif temp > 37.0:
            condition = "Extreme Heat"
            factor    = 0.08
        elif code == 0 and 22.0 <= temp <= 35.0:
            condition = "Clear / Favorable"
            factor    = 0.0    # Ideal conditions, no adjustment
        else:
            condition = "Partly Cloudy / Mild"
            factor    = 0.0

        return {
            "temperature_c":             round(temp, 1),
            "condition":                  condition,
            "weather_adjustment_factor":  factor,
            "weather_risk_factor":        factor,   # alias used by llm_synthesizer
            "weather_code":               code,
        }

    @staticmethod
    def _neutral_fallback() -> Dict[str, Any]:
        return {
            "temperature_c":             25.0,
            "condition":                 "Clear / Favorable",
            "weather_adjustment_factor": 0.0,
            "weather_risk_factor":       0.0,
            "weather_code":              0,
        }