from pydantic import BaseModel, Field
from typing import Optional

class GapAnalysisRequest(BaseModel):
    commodity: str = Field(..., example="Tomato")
    location: str = Field(..., example="Noida")
    forecast_days: int = Field(default=7, ge=1, le=30, example=7)
    available_supply_kg: float = Field(..., ge=0.0, example=18000.0)
    confirmed_orders_kg: float = Field(default=0.0, ge=0.0, example=8000.0)
    weather_condition: Optional[str] = Field(default="Normal", example="Normal")

class ForecastDetails(BaseModel):
    predicted_baseline_requirement_kg: float
    adjusted_requirement_kg: float
    trend: str

class RealtimeSignals(BaseModel):
    confirmed_orders_kg: float
    festival_adjustment_factor: float
    weather_adjustment_factor: float
    total_adjustment_factor: float

class SupplyDetails(BaseModel):
    available_kg: float
    gap_kg: float
    severity: str

class RecommendationDetails(BaseModel):
    notify_farmers: bool
    additional_supply_needed_kg: float
    action_message: str

class GapAnalysisResponse(BaseModel):
    status: str
    commodity: str
    location: str
    forecast_days: int
    forecast: ForecastDetails
    realtime_signals: RealtimeSignals
    supply: SupplyDetails
    recommendation: RecommendationDetails