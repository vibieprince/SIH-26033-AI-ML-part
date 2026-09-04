import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

def train_and_export_models(data_path: str, artifact_dir: str = "models"):
    df = pd.read_csv(data_path)
    df['reported_date'] = pd.to_datetime(df['reported_date'])
    
    # Temporal feature engineering
    df['year'] = df['reported_date'].dt.year
    df['month'] = df['reported_date'].dt.month
    df['week'] = df['reported_date'].dt.isocalendar().week.astype(int)
    df['dayofyear'] = df['reported_date'].dt.dayofyear
    df['sin_day'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['cos_day'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)

    df['target_arrivals_h7'] = df.groupby(['crop', 'district'])['arrivals_tonnes'].shift(-1)
    df['target_price_h7'] = df.groupby(['crop', 'district'])['modal_price_rs_kg'].shift(-1)

    crop_cats = sorted(list(df['crop'].unique()))
    state_cats = sorted(list(df['state'].unique()))
    district_cats = sorted(list(df['district'].unique()))

    crop_map = {c: i for i, c in enumerate(crop_cats)}
    state_map = {s: i for i, s in enumerate(state_cats)}
    district_map = {d: i for i, d in enumerate(district_cats)}

    df['crop_cat'] = df['crop'].map(crop_map)
    df['state_cat'] = df['state'].map(state_map)
    df['district_cat'] = df['district'].map(district_map)

    df_clean = df.dropna(subset=['target_arrivals_h7', 'target_price_h7']).copy()

    features = [
        'crop_cat', 'state_cat', 'district_cat', 'arrivals_tonnes', 'modal_price_rs_kg',
        'year', 'month', 'week', 'sin_day', 'cos_day'
    ]

    train_df = df_clean[df_clean['year'] <= 2018]
    test_df = df_clean[df_clean['year'] >= 2022]

    # Supply model training
    X_train, y_train_s = train_df[features], train_df['target_arrivals_h7']
    supply_model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, random_state=42)
    supply_model.fit(X_train, y_train_s)

    # Price model training
    y_train_p = train_df['target_price_h7']
    price_model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, random_state=42)
    price_model.fit(X_train, y_train_p)

    # Save artifacts
    os.makedirs(f"{artifact_dir}/supply", exist_ok=True)
    os.makedirs(f"{artifact_dir}/price", exist_ok=True)
    os.makedirs(f"{artifact_dir}/metadata", exist_ok=True)

    joblib.dump(supply_model, f"{artifact_dir}/supply/global_supply_model.joblib")
    joblib.dump(price_model, f"{artifact_dir}/price/global_price_model.joblib")

    with open(f"{artifact_dir}/metadata/categorical_mappings.json", "w") as f:
        json.dump({"crop": crop_map, "state": state_map, "district": district_map}, f, indent=2)

    with open(f"{artifact_dir}/metadata/feature_schema.json", "w") as f:
        json.dump({"features": features}, f, indent=2)

if __name__ == "__main__":
    train_and_export_models("data/processed/clean_mandi_data.csv")