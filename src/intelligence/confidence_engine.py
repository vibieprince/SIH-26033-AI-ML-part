from dataclasses import dataclass

@dataclass
class ConfidenceInterval:
    lower_bound_kg: float
    expected_kg: float
    upper_bound_kg: float
    confidence_score: float  # 0.0 to 1.0

class ConfidenceEngine:
    def __init__(self, historical_smape: float = 18.35):
        self.base_error_margin = historical_smape / 100.0[cite: 2]

    def compute_bounds(self, forecast_value_kg: float, sample_count: int = 100) -> ConfidenceInterval:
        # Scale margin of error based on data availability
        data_density_penalty = 1.0 if sample_count >= 50 else 1.25
        effective_error = self.base_error_margin * data_density_penalty

        lower = max(0.0, forecast_value_kg * (1.0 - effective_error))
        upper = forecast_value_kg * (1.0 + effective_error)
        conf_score = max(0.40, min(0.95, 1.0 - (effective_error / 2.0)))

        return ConfidenceInterval(
            lower_bound_kg=round(lower, 2),
            expected_kg=round(forecast_value_kg, 2),
            upper_bound_kg=round(upper, 2),
            confidence_score=round(conf_score, 2)
        )