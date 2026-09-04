from dataclasses import dataclass

@dataclass
class MarketGapResult:
    forecast_demand_kg: float
    forecast_supply_kg: float
    gap_kg: float
    gap_percentage: float
    market_condition: str  # HIGH_SHORTAGE, MODERATE_SHORTAGE, BALANCED, SURPLUS

class GapEngine:
    def calculate_gap(self, forecast_demand_kg: float, forecast_supply_kg: float) -> MarketGapResult:
        gap_kg = forecast_demand_kg - forecast_supply_kg
        gap_pct = (gap_kg / (forecast_supply_kg + 1e-5)) * 100.0

        if gap_pct > 20.0:
            condition = "HIGH_SHORTAGE"
        elif 10.0 <= gap_pct <= 20.0:
            condition = "MODERATE_SHORTAGE"
        elif -10.0 <= gap_pct < 10.0:
            condition = "BALANCED"
        else:
            condition = "SURPLUS"

        return MarketGapResult(
            forecast_demand_kg=round(forecast_demand_kg, 2),
            forecast_supply_kg=round(forecast_supply_kg, 2),
            gap_kg=round(gap_kg, 2),
            gap_percentage=round(gap_pct, 2),
            market_condition=condition
        )