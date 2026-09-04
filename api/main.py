# api/main.py
"""
Kisan Guard – FastAPI Demand Forecasting Engine  v5.0
=====================================================

Endpoints
---------
GET  /health                  – liveness probe
GET  /orders                  – list logged orders (filterable)
POST /orders                  – legacy order logging (compat)
POST /api/v1/log-order        – canonical order logging endpoint
POST /api/v1/predict-demand   – full demand forecast pipeline

Design notes
------------
- Zero hardcoded demand or supply constants.
- Supply is computed dynamically from `src.predictor.compute_predicted_supply_kg`.
- Deficit % and Opportunity Score are computed from live variables.
- DemandSignalStore uses normalized string matching internally.
- Festival signal is injected from `src.festival_calendar.FestivalEngine`.
- The LLM synthesizer receives all pre-computed numeric context to ensure
  the LLM narrative reflects real numbers, not hallucinated ones.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.demand_store import DemandSignalStore
from src.festival_calendar import FestivalEngine
from src.llm_synthesizer import LangChainDemandSynthesizer
from src.location_engine import LocationEngine
from src.predictor import (
    DemandPredictor,
    compute_dynamic_baseline_kg,
    compute_predicted_supply_kg,
    get_supply_factor,
)
from src.weather_engine import WeatherEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Kisan Guard – Demand Forecasting Engine",
    version="5.0.0",
    description="Dynamic agricultural demand forecasting with LightGBM, LangChain, and live signal streams.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH   = "./models/lightgbm_model.pkl"
DATASET_PATH = "./data/processed/clean_mandi_data.parquet"

_predictor: Optional[DemandPredictor] = None


# ------------------------------------------------------------------ #
# Pydantic schemas (single source of truth – shared with dashboard)
# ------------------------------------------------------------------ #

class OrderSignalRequest(BaseModel):
    """POST /api/v1/log-order  (canonical)"""
    commodity:   str   = Field(..., example="Onion")
    location:    str   = Field(..., example="Noida")
    quantity_kg: float = Field(..., gt=0, example=2500.0)
    buyer_type:  str   = Field(default="retailer", example="retailer")
    order_id:    Optional[str] = Field(default=None)


class LegacyOrderRequest(BaseModel):
    """POST /orders  (backward-compat; accepts both crop/commodity, state/district/location)"""
    order_id:       Optional[str]   = None
    crop:           Optional[str]   = None
    commodity:      Optional[str]   = None
    quantity_kg:    float           = Field(..., gt=0)
    urgency:        Optional[str]   = "STANDARD"
    buyer_category: Optional[str]   = "retailer"
    buyer_type:     Optional[str]   = None
    state:          Optional[str]   = ""
    district:       Optional[str]   = ""
    location:       Optional[str]   = ""


class DemandForecastRequest(BaseModel):
    """POST /api/v1/predict-demand"""
    commodity:      str            = Field(..., example="Onion")
    location_query: Optional[str]  = Field(default="", example="Noida")
    latitude:       Optional[float] = Field(default=None)
    longitude:      Optional[float] = Field(default=None)
    forecast_days:  int            = Field(default=7, ge=1, le=30)
    language:       str            = Field(default="en", example="en")


# ------------------------------------------------------------------ #
# Startup
# ------------------------------------------------------------------ #

@app.on_event("startup")
def load_artifacts() -> None:
    global _predictor
    model_ok   = Path(MODEL_PATH).exists()
    dataset_ok = Path(DATASET_PATH).exists()
    if model_ok and dataset_ok:
        try:
            _predictor = DemandPredictor(MODEL_PATH, DATASET_PATH)
            logger.info("LightGBM predictor loaded successfully.")
        except Exception as exc:
            logger.warning("DemandPredictor init failed (%s). Will use dynamic baseline.", exc)
    else:
        missing = []
        if not model_ok:   missing.append(MODEL_PATH)
        if not dataset_ok: missing.append(DATASET_PATH)
        logger.warning("Missing artifacts %s – using dynamic baseline fallback.", missing)


# ------------------------------------------------------------------ #
# Health
# ------------------------------------------------------------------ #

@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
def health() -> Dict[str, Any]:
    return {
        "status": "online",
        "version": "5.0.0",
        "model_loaded": _predictor is not None,
        "active_orders": DemandSignalStore.total_orders(),
    }


# ------------------------------------------------------------------ #
# Order endpoints
# ------------------------------------------------------------------ #

@app.get("/orders", response_model=List[Dict[str, Any]], tags=["Orders"])
@app.get("/orders/", response_model=List[Dict[str, Any]], tags=["Orders"])
def get_active_orders(
    crop:      Optional[str] = Query(None),
    commodity: Optional[str] = Query(None),
    state:     Optional[str] = Query(None),
    location:  Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    """Return filtered list of logged orders (normalized matching)."""
    target_commodity = crop or commodity or ""
    target_location  = state or location or ""
    orders = DemandSignalStore.get_orders(
        commodity=target_commodity,
        location=target_location,
    )
    return orders


@app.post("/orders", tags=["Orders"])
@app.post("/orders/", tags=["Orders"])
def legacy_create_order(req: LegacyOrderRequest) -> Dict[str, Any]:
    """Backward-compatible order logging; resolves crop/commodity and location aliases."""
    comm    = (req.commodity or req.crop or "Unknown").strip()
    loc     = (req.location or req.district or req.state or "Unknown").strip()
    b_type  = (req.buyer_type or req.buyer_category or "retailer").strip()

    if not comm or comm == "Unknown":
        raise HTTPException(status_code=422, detail="commodity or crop field is required.")
    if not loc or loc == "Unknown":
        raise HTTPException(status_code=422, detail="location, district, or state field is required.")

    order = DemandSignalStore.add_order(
        commodity=comm,
        location=loc,
        quantity_kg=req.quantity_kg,
        buyer_type=b_type,
        order_id=req.order_id,
    )
    total_live_kg = DemandSignalStore.get_aggregated_demand_kg(comm, loc)
    return {
        "status": "success",
        "logged_order": order,
        "total_active_platform_orders_kg": total_live_kg,
    }


@app.post("/api/v1/log-order", tags=["Orders"])
def log_customer_order(req: OrderSignalRequest) -> Dict[str, Any]:
    """
    Canonical order logging endpoint used by the Streamlit dashboard and
    mobile app.  Every logged order is immediately available to the next
    /api/v1/predict-demand call via DemandSignalStore.
    """
    order = DemandSignalStore.add_order(
        commodity=req.commodity,
        location=req.location,
        quantity_kg=req.quantity_kg,
        buyer_type=req.buyer_type,
        order_id=req.order_id,
    )
    total_live_kg = DemandSignalStore.get_aggregated_demand_kg(
        req.commodity, req.location
    )
    return {
        "status": "success",
        "logged_order": order,
        "total_active_platform_orders_kg": total_live_kg,
        "order_count": DemandSignalStore.get_order_count(req.commodity, req.location),
    }


# ------------------------------------------------------------------ #
# Demand Forecast Pipeline
# ------------------------------------------------------------------ #

@app.post("/api/v1/predict-demand", tags=["Forecast"])
def predict_demand(req: DemandForecastRequest) -> Dict[str, Any]:
    """
    Full demand forecasting pipeline:

    1. Geocode / resolve location → lat/lon + nearest model district
    2. Fetch live weather (Open-Meteo)
    3. Run LightGBM or dynamic baseline to get baseline_ml_kg
    4. Retrieve live platform order volume from DemandSignalStore
    5. Compute festival and seasonal multipliers
    6. Compute total dynamic demand, predicted supply, gap%, opportunity score
    7. Synthesize LangChain + Gemini advisory
    """

    # 1. Location resolution
    if req.latitude and req.longitude:
        geo_info = {
            "city":   "GPS Location",
            "region": "GPS Location",
            "lat":    req.latitude,
            "lon":    req.longitude,
            "method": "GPS_COORDINATES",
        }
    else:
        geo_info = LocationEngine.geocode_query(req.location_query or "")

    model_district, distance_km = LocationEngine.resolve_nearest_model_district(
        geo_info["lat"], geo_info["lon"]
    )

    # 2. Live weather
    weather = WeatherEngine.get_live_weather(geo_info["lat"], geo_info["lon"])
    weather_risk = float(weather.get("weather_adjustment_factor", 0.0))

    # 3. Baseline ML prediction (with dynamic fallback)
    today_month = date.today().month
    if _predictor:
        baseline_ml_kg = _predictor.predict_baseline_kg(
            commodity=req.commodity,
            district=model_district,
            forecast_days=req.forecast_days,
            month=today_month,
        )
    else:
        baseline_ml_kg = compute_dynamic_baseline_kg(
            commodity=req.commodity,
            district=model_district,
            forecast_days=req.forecast_days,
            month=today_month,
        )

    # 4. Live platform orders
    live_app_orders_kg = DemandSignalStore.get_aggregated_demand_kg(
        req.commodity, model_district
    )
    # Also aggregate against the raw location query string (handles city-vs-district mismatch)
    if req.location_query:
        live_app_orders_kg += DemandSignalStore.get_aggregated_demand_kg(
            req.commodity, req.location_query
        )

    # 5. Festival + seasonal signals
    fest_name, fest_mult = FestivalEngine.evaluate_festival_impact(req.commodity)
    from src.predictor import _get_seasonal_curve  # isolated import to avoid circular
    season_mult = _get_seasonal_curve(today_month)

    festival_info = {"name": fest_name, "multiplier": fest_mult}

    # 6. Compute demand, supply, gap, opportunity score
    demand_multiplier = (1.0 + fest_mult) * season_mult
    final_demand_kg   = round((baseline_ml_kg * demand_multiplier) + live_app_orders_kg, 1)

    predicted_supply_kg = compute_predicted_supply_kg(
        commodity=req.commodity,
        district=model_district,
        baseline_demand_kg=baseline_ml_kg,
        weather_adjustment_factor=weather_risk,
    )

    gap_kg  = final_demand_kg - predicted_supply_kg
    gap_pct = round(
        (gap_kg / predicted_supply_kg) * 100.0 if predicted_supply_kg > 0 else 0.0, 2
    )
    live_order_ratio = min(1.0, live_app_orders_kg / max(1.0, baseline_ml_kg))
    opportunity_score = round(
        min(100.0, max(0.0,
            (gap_pct * 1.5)
            + (live_order_ratio * 25.0)
            + (abs(weather_risk) * 15.0)
        )),
        1,
    )

    # 7. LangChain advisory synthesis
    ai_report = LangChainDemandSynthesizer.synthesize_report(
        commodity=req.commodity,
        location_name=geo_info["city"],
        forecast_days=req.forecast_days,
        baseline_ml_kg=baseline_ml_kg,
        live_app_orders_kg=live_app_orders_kg,
        weather_info=weather,
        festival_info=festival_info,
        season_mult=season_mult,
        predicted_supply_kg=predicted_supply_kg,
        language=req.language,
    )

    return {
        "status":          "success",
        "input_location":  geo_info,
        "mapped_district": {"district": model_district, "distance_km": distance_km},
        "weather":         weather,
        "festival":        festival_info,
        "season_mult":     season_mult,
        # Core metrics – all dynamically computed
        "baseline_ml_kg":       baseline_ml_kg,
        "live_app_orders_kg":   live_app_orders_kg,
        "final_demand_kg":      final_demand_kg,
        "predicted_supply_kg":  predicted_supply_kg,
        "gap_kg":               round(gap_kg, 1),
        "gap_pct":              gap_pct,
        "opportunity_score":    opportunity_score,
        # AI narrative
        "ai_report": ai_report.model_dump() if hasattr(ai_report, "model_dump") else ai_report.dict(),
    }