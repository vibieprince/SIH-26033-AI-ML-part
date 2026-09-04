from typing import Dict, Tuple

class DynamicSignalAdjuster:
    """Applies festival multipliers, weather impacts, and platform order overrides."""
    
    FESTIVAL_CALENDAR: Dict[str, Dict] = {
        'Diwali': {'months': [10, 11], 'boost': 0.25},
        'Navratri': {'months': [9, 10], 'boost': 0.15},
        'Holi': {'months': [3], 'boost': 0.10},
        'Eid': {'months': [4, 5], 'boost': 0.15}
    }

    @classmethod
    def calculate_adjustments(
        cls, 
        commodity: str, 
        target_month: int, 
        weather_condition: str
    ) -> Tuple[float, float, float]:
        """Calculates dynamic multiplier factors based on festive and weather signals."""
        fest_factor = 0.0
        for event, meta in cls.FESTIVAL_CALENDAR.items():
            if target_month in meta['months']:
                fest_factor += meta['boost']
                
        weather_clean = weather_condition.lower()
        if weather_clean in ['heavy rain', 'storm', 'flood']:
            weather_factor = -0.10
        elif weather_clean in ['heatwave']:
            weather_factor = 0.05
        else:
            weather_factor = 0.0
            
        total_factor = fest_factor + weather_factor
        return fest_factor, weather_factor, total_factor

    @classmethod
    def apply_hybrid_forecast(
        cls, 
        baseline_kg: float, 
        total_adjustment_factor: float, 
        confirmed_orders_kg: float
    ) -> float:
        """Applies multipliers and enforces confirmed orders floor constraint."""
        adjusted_base = baseline_kg * (1.0 + total_adjustment_factor)
        final_requirement_kg = max(adjusted_base, confirmed_orders_kg)
        return float(final_requirement_kg)