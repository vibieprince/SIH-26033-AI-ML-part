from dataclasses import dataclass
from src.demand.demand_config import opp_config

@dataclass
class OpportunityEvaluation:
    score: float  # 0 to 100
    priority: str  # LOW, MEDIUM, HIGH, CRITICAL
    reasons: list

class OpportunityEngine:
    def evaluate(
        self,
        gap_percentage: float,
        price_trend_ratio: float,
        weather_risk_factor: float,
        closest_farmer_distance_km: float
    ) -> OpportunityEvaluation:
        reasons = []

        # 1. Gap Score (0-100)
        gap_score = min(100.0, max(0.0, gap_percentage * 2.0))
        if gap_percentage > 20.0:
            reasons.append(f"Severe supply shortage detected (+{gap_percentage:.1f}%).")

        # 2. Price Score (0-100)
        price_score = min(100.0, max(0.0, (price_trend_ratio - 1.0) * 200.0))
        if price_trend_ratio > 1.1:
            reasons.append("Rising price momentum enhances producer margin.")

        # 3. Weather Safety Score (0-100)
        weather_score = max(0.0, min(100.0, weather_risk_factor * 100.0))
        if weather_risk_factor < 0.8:
            reasons.append("Localized weather risk flagged for route planning.")

        # 4. Distance Decay Score (0-100)
        dist_score = max(0.0, 100.0 - (closest_farmer_distance_km / 2.0))

        # Composite score
        final_score = (
            opp_config.gap_weight * gap_score +
            opp_config.price_weight * price_score +
            opp_config.weather_weight * weather_score +
            opp_config.distance_weight * dist_score
        )
        final_score = round(min(100.0, max(0.0, final_score)), 1)

        priority = "LOW" if final_score < 40 else "MEDIUM" if final_score < 70 else "HIGH" if final_score < 85 else "CRITICAL"

        return OpportunityEvaluation(
            score=final_score,
            priority=priority,
            reasons=reasons
        )