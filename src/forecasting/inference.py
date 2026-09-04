import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass

@dataclass
class ForecastPrediction:
    crop: str
    state: str
    district: str
    predicted_supply_tonnes: float
    predicted_supply_kg: float
    predicted_price_rs_kg: float

class SupplyForecaster:
    def __init__(self, artifact_dir: str = "models"):
        self.supply_model = joblib.load(f"{artifact_dir}/supply/global_supply_model.joblib")
        self.price_model = joblib.load(f"{artifact_dir}/price/global_price_model.joblib")
        
        with open(f"{artifact_dir}/metadata/categorical_mappings.json", "r") as f:
            self.cat_mappings = json.load(f)
            
        with open(f"{artifact_dir}/metadata/feature_schema.json", "r") as f:
            self.features = json.load(f)["features"][cite: 1]

    def predict(
        self, crop: str, state: str, district: str, current_arrivals_tonnes: float, current_price_rs_kg: float
    ) -> ForecastPrediction:
        now = datetime.now()
        dayofyear = now.timetuple().tm_yday

        c_code = self.cat_mappings["crop"].get(crop, 0)[cite: 3]
        s_code = self.cat_mappings["state"].get(state, 0)[cite: 3]
        d_code = self.cat_mappings["district"].get(district, 0)[cite: 3]

        input_row = pd.DataFrame([{
            "crop_cat": c_code,
            "state_cat": s_code,
            "district_cat": d_code,
            "arrivals_tonnes": current_arrivals_tonnes,
            "modal_price_rs_kg": current_price_rs_kg,
            "year": now.year,
            "month": now.month,
            "week": now.isocalendar()[1],
            "sin_day": np.sin(2 * np.pi * dayofyear / 365.25),
            "cos_day": np.cos(2 * np.pi * dayofyear / 365.25)
        }])[self.features]

        pred_supply = max(10.0, float(self.supply_model.predict(input_row)[0]))
        pred_price = max(5.0, float(self.price_model.predict(input_row)[0]))

        return ForecastPrediction(
            crop=crop,
            state=state,
            district=district,
            predicted_supply_tonnes=round(pred_supply, 2),
            predicted_supply_kg=round(pred_supply * 1000.0, 2),
            predicted_price_rs_kg=round(pred_price, 2)
        )