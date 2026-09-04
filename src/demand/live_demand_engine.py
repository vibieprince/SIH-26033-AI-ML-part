from dataclasses import dataclass
from typing import List, Dict
import math

@dataclass
class Order:
    order_id: str
    crop: str
    quantity_kg: float
    timestamp_epoch: float
    urgency: str  # STANDARD, PRIORITY, CRITICAL
    buyer_category: str  # HOUSEHOLD, RETAILER, WHOLESALER, INSTITUTION
    state: str
    district: str

@dataclass
class DynamicDemandResult:
    baseline_demand_kg: float
    live_order_addition_kg: float
    contextual_multiplier: float
    final_dynamic_demand_kg: float
    breakdown: Dict[str, float]
    demand_status: str  # NORMAL, ELEVATED, HIGH, SURGE

class LiveDemandEngine:
    def __init__(self, baseline_engine):
        self.baseline_engine = baseline_engine
        self.recency_half_life_hours = 48.0

    def compute_dynamic_demand(
        self,
        crop: str,
        state: str,
        district: str,
        target_date: str,
        active_orders: List[Order],
        festival_multiplier: float = 1.0,
        weather_multiplier: float = 1.0,
        seasonal_multiplier: float = 1.0,
        current_time_epoch: float = 0.0
    ) -> DynamicDemandResult:
        
        # 1. Obtain deterministic baseline
        base_res = self.baseline_engine.calculate_baseline(crop, state, district, target_date)
        
        # 2. Compute Contextual Multiplier with hard bounds [0.60, 2.00]
        raw_multiplier = festival_multiplier * weather_multiplier * seasonal_multiplier
        clamped_multiplier = max(0.60, min(2.00, raw_multiplier))
        
        # 3. Aggregate Live Orders with Exponential Recency & Category Decay
        weighted_live_kg = 0.0
        urgency_weights = {"STANDARD": 1.0, "PRIORITY": 1.25, "CRITICAL": 1.5}
        buyer_weights = {"HOUSEHOLD": 1.0, "RETAILER": 1.1, "WHOLESALER": 1.25, "INSTITUTION": 1.3}

        for ord in active_orders:
            if ord.crop.lower() == crop.lower() and ord.state.lower() == state.lower():
                age_hours = max(0.0, (current_time_epoch - ord.timestamp_epoch) / 3600.0)
                recency_decay = math.exp(-0.693 * age_hours / self.recency_half_life_hours)
                
                u_w = urgency_weights.get(ord.urgency.upper(), 1.0)
                b_w = buyer_weights.get(ord.buyer_category.upper(), 1.0)
                
                weighted_live_kg += ord.quantity_kg * recency_decay * u_w * b_w

        # 4. Synthesize Final Forecast
        base_adjusted = base_res.baseline_demand_kg * clamped_multiplier
        final_demand = base_adjusted + weighted_live_kg
        
        ratio = final_demand / (base_res.baseline_demand_kg + 1e-5)
        status = "NORMAL" if ratio < 1.10 else "ELEVATED" if ratio < 1.25 else "HIGH" if ratio < 1.50 else "SURGE"

        return DynamicDemandResult(
            baseline_demand_kg=base_res.baseline_demand_kg,
            live_order_addition_kg=round(weighted_live_kg, 2),
            contextual_multiplier=round(clamped_multiplier, 3),
            final_dynamic_demand_kg=round(final_demand, 2),
            breakdown={
                "baseline_kg": base_res.baseline_demand_kg,
                "seasonal_adj_kg": round(base_res.baseline_demand_kg * (seasonal_multiplier - 1.0), 2),
                "festival_adj_kg": round(base_res.baseline_demand_kg * (festival_multiplier - 1.0), 2),
                "weather_adj_kg": round(base_res.baseline_demand_kg * (weather_multiplier - 1.0), 2),
                "live_orders_kg": round(weighted_live_kg, 2)
            },
            demand_status=status
        )