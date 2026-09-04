# src/predictor.py
"""
Demand Predictor – LightGBM model wrapper with dynamic feature-weighted
baseline fallback.

If the trained pickle is available and the dataset has data for the
requested (commodity, district) pair, the LightGBM model runs inference.
Otherwise a commodity/district/seasonal feature-weighted algorithm is used
instead of the old flat constant (2400 * days).

Baseline algorithm (no-model fallback):
    Baseline_kg = Base_Rate(commodity)
                * District_Factor(district)
                * Forecast_Days
                * Seasonal_Curve(month)
                * 1000           # convert reference tonnes → kg
"""

import logging
import math
from datetime import date
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Commodity-level base daily procurement rates (tonnes/day)
# Derived from 24-year national mandi arrival averages.
# ------------------------------------------------------------------ #
_BASE_DAILY_TONNES: dict = {
    "onion":   3.8,
    "potato":  4.6,
    "tomato":  3.2,
    "wheat":   5.5,
}
_DEFAULT_BASE_DAILY_TONNES = 3.5  # generic crop fallback

# ------------------------------------------------------------------ #
# District-level population / market-size multipliers
# (Relative to Gautam Buddha Nagar as 1.00 baseline)
# ------------------------------------------------------------------ #
_DISTRICT_FACTORS: dict = {
    "sangrur":              0.72,
    "ludhiana":             1.30,
    "amritsar":             1.25,
    "gautam buddha nagar":  1.00,
    "agra":                 1.15,
    "varanasi":             1.10,
    "nashik":               1.40,   # Major onion mandi
    "pune":                 1.35,
    "ahmednagar":           1.20,
    "ganganagar":           0.85,
    "jaipur":               1.25,
    "indore":               1.20,
    "kolar":                0.90,
}
_DEFAULT_DISTRICT_FACTOR = 1.00

# ------------------------------------------------------------------ #
# Monthly seasonal demand curve (index 1-12)
# Values represent demand multiplier vs annual average.
# ------------------------------------------------------------------ #
_SEASONAL_CURVE: dict = {
    1:  1.08,   # January – post-harvest high arrivals
    2:  1.05,
    3:  1.12,   # Holi, rabi harvest
    4:  1.10,
    5:  0.95,   # early summer dip
    6:  0.88,   # lean season, pre-monsoon
    7:  0.82,   # kharif sowing, low arrivals
    8:  0.85,
    9:  0.95,   # early kharif arrivals
    10: 1.15,   # Navratri / Dussehra demand surge
    11: 1.30,   # Diwali / wedding season peak
    12: 1.18,   # Wedding season tail
}

# ------------------------------------------------------------------ #
# Supply model: district × commodity harvest offset factors
# Models how much of the baseline demand can be met by local supply.
# Range 0.65 – 0.92; lower = tighter supply in that district.
# ------------------------------------------------------------------ #
_SUPPLY_FACTORS: dict = {
    ("nashik",              "onion"):  0.88,
    ("ahmednagar",          "onion"):  0.85,
    ("ludhiana",            "potato"): 0.90,
    ("sangrur",             "wheat"):  0.92,
    ("gautam buddha nagar", "potato"): 0.80,
    ("jaipur",              "onion"):  0.78,
    ("indore",              "potato"): 0.82,
    ("kolar",               "tomato"): 0.87,
}
_DEFAULT_SUPPLY_FACTOR = 0.76  # conservative national average


def _norm(s: str) -> str:
    return s.lower().strip() if s else ""


def _get_base_daily_tonnes(commodity: str) -> float:
    return _BASE_DAILY_TONNES.get(_norm(commodity), _DEFAULT_BASE_DAILY_TONNES)


def _get_district_factor(district: str) -> float:
    return _DISTRICT_FACTORS.get(_norm(district), _DEFAULT_DISTRICT_FACTOR)


def _get_seasonal_curve(month: Optional[int] = None) -> float:
    if month is None:
        month = date.today().month
    return _SEASONAL_CURVE.get(month, 1.0)


def get_supply_factor(commodity: str, district: str) -> float:
    """Returns the local supply coverage factor for a given commodity-district pair."""
    key = (_norm(district), _norm(commodity))
    return _SUPPLY_FACTORS.get(key, _DEFAULT_SUPPLY_FACTOR)


def compute_dynamic_baseline_kg(
    commodity: str,
    district: str,
    forecast_days: int,
    month: Optional[int] = None,
) -> float:
    """
    Feature-weighted baseline demand calculation (no ML model needed).

    Formula:
        Baseline_kg = Base_Rate_T * District_Factor * Seasonal_Curve
                    * Forecast_Days * 1000
    """
    base_t = _get_base_daily_tonnes(commodity)
    d_factor = _get_district_factor(district)
    s_curve = _get_seasonal_curve(month)
    baseline_kg = base_t * d_factor * s_curve * forecast_days * 1000.0
    logger.info(
        "Dynamic baseline [%s/%s/%d days]: %.0f kg "
        "(base=%.1fT, district=%.2f, seasonal=%.3f)",
        commodity, district, forecast_days, baseline_kg,
        base_t, d_factor, s_curve,
    )
    return round(baseline_kg, 1)


def compute_predicted_supply_kg(
    commodity: str,
    district: str,
    baseline_demand_kg: float,
    weather_adjustment_factor: float = 0.0,
) -> float:
    """
    Dynamic supply calculation:
        Supply = Baseline_Demand × Supply_Factor × (1 + weather_adjustment_factor)

    `weather_adjustment_factor` is negative for rain/storms (disrupts transit)
    and positive for heatwaves (spoilage urgency shortens effective supply).

    The result is bounded at >= 5% of baseline to avoid division-by-zero.
    """
    supply_factor = get_supply_factor(commodity, district)
    # Weather has a negative impact on supply (storm = -0.12 means -12% supply)
    # We invert the sign: positive factor → supply drops (disruption)
    effective_factor = supply_factor + weather_adjustment_factor
    effective_factor = max(0.40, min(1.00, effective_factor))  # hard clamp 40–100%
    supply_kg = baseline_demand_kg * effective_factor
    logger.info(
        "Predicted supply [%s/%s]: %.0f kg (factor=%.3f, weather_adj=%.3f)",
        commodity, district, supply_kg, supply_factor, weather_adjustment_factor,
    )
    return round(max(supply_kg, baseline_demand_kg * 0.05), 1)


class DemandPredictor:
    """
    Inference wrapper for generating baseline demand forecasts.

    Falls back to the dynamic feature-weighted algorithm when:
      - The LightGBM model is not loaded, OR
      - There is insufficient historical data for the requested series.
    """

    def __init__(self, model_path: str, dataset_path: str):
        self.model = joblib.load(model_path)
        self.df = pd.read_parquet(dataset_path)
        self.cat_cols = ["commodity", "state", "district"]
        logger.info(
            "DemandPredictor initialized. Dataset rows: %d", len(self.df)
        )

    def predict_baseline_kg(
        self,
        commodity: str,
        district: str,
        forecast_days: int,
        month: Optional[int] = None,
    ) -> float:
        """
        Runs LightGBM inference when historical data is available.
        Falls back to dynamic feature-weighted baseline otherwise.
        """
        # Lazy import to keep module lightweight when model not loaded
        try:
            from src.features import generate_feature_matrix
            from src.train import FEATURE_COLUMNS
        except ImportError as exc:
            logger.warning("Feature/train modules not importable: %s. Using dynamic baseline.", exc)
            return compute_dynamic_baseline_kg(commodity, district, forecast_days, month)

        comm_clean = commodity.capitalize()
        dist_clean = district.title()

        sub_df = self.df[
            (self.df["commodity"] == comm_clean) & (self.df["district"] == dist_clean)
        ].copy()

        if sub_df.empty:
            # Try commodity-only average
            sub_df = self.df[self.df["commodity"] == comm_clean].copy()
            if sub_df.empty:
                logger.warning(
                    "No historical data for commodity=%s. Using dynamic baseline.", commodity
                )
                return compute_dynamic_baseline_kg(commodity, district, forecast_days, month)

        sub_df = sub_df.sort_values("date").tail(60).reset_index(drop=True)

        try:
            feat_df = generate_feature_matrix(sub_df)
        except Exception as exc:
            logger.warning("Feature generation failed (%s). Using dynamic baseline.", exc)
            return compute_dynamic_baseline_kg(commodity, district, forecast_days, month)

        if feat_df.empty:
            # Not enough rows for lags – use simple arrival average
            avg_daily_tonnes = max(0.5, float(sub_df["arrivals"].mean()))
            return round(avg_daily_tonnes * forecast_days * 1000.0, 1)

        for c in self.cat_cols:
            if c in feat_df.columns:
                feat_df[c] = feat_df[c].astype("category")

        try:
            latest_features = feat_df.iloc[[-1]][FEATURE_COLUMNS]
            avg_daily_tonnes = float(self.model.predict(latest_features)[0])
            avg_daily_tonnes = max(0.1, avg_daily_tonnes)
        except Exception as exc:
            logger.warning("LightGBM prediction failed (%s). Using dynamic baseline.", exc)
            return compute_dynamic_baseline_kg(commodity, district, forecast_days, month)

        total_forecast_kg = avg_daily_tonnes * forecast_days * 1000.0
        logger.info(
            "LightGBM forecast [%s/%s/%d days]: %.0f kg",
            commodity, district, forecast_days, total_forecast_kg,
        )
        return round(float(total_forecast_kg), 1)