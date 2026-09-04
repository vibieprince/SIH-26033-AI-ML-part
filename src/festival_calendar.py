# src/festival_calendar.py
import datetime
from typing import Dict, Tuple

class FestivalEngine:
    """Calculates festival demand surges for Indian commodities."""
    
    # Precise multi-year festival date ranges and commodity sensitivity
    FESTIVAL_EVENTS = [
        {
            "name": "Navratri & Dussehra",
            "start": (9, 15), "end": (10, 25),
            "multipliers": {"Potato": 0.20, "Tomato": 0.15, "Onion": -0.10, "Wheat": 0.05} # Fasting effects
        },
        {
            "name": "Diwali Season",
            "start": (10, 20), "end": (11, 15),
            "multipliers": {"Potato": 0.25, "Tomato": 0.20, "Onion": 0.20, "Wheat": 0.15}
        },
        {
            "name": "Wedding Season Peak",
            "start": (11, 20), "end": (12, 25),
            "multipliers": {"Potato": 0.15, "Tomato": 0.25, "Onion": 0.25, "Wheat": 0.10}
        },
        {
            "name": "Holi Celebration",
            "start": (3, 1), "end": (3, 25),
            "multipliers": {"Wheat": 0.20, "Potato": 0.10, "Tomato": 0.10, "Onion": 0.10}
        }
    ]

    @classmethod
    def evaluate_festival_impact(cls, commodity: str, target_date: datetime.date = None) -> Tuple[str, float]:
        """Determines active festival event and associated commodity multiplier."""
        if target_date is None:
            target_date = datetime.date.today()
            
        comm_clean = commodity.capitalize()
        month, day = target_date.month, target_date.day
        
        for event in cls.FESTIVAL_EVENTS:
            s_m, s_d = event["start"]
            e_m, e_d = event["end"]
            
            # Check if current date falls within festival window
            if (s_m <= month <= e_m) and (s_d <= day or month > s_m) and (day <= e_d or month < e_m):
                multiplier = event["multipliers"].get(comm_clean, 0.05)
                return event["name"], multiplier
                
        return "None (Standard Period)", 0.0