import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path
import logging
from src.features import generate_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FEATURE_COLUMNS = [
    'commodity', 'state', 'district', 'dayofweek', 'month', 'quarter', 
    'dayofyear', 'is_weekend', 'lag_1', 'lag_7', 'lag_14', 'lag_28',
    'roll_mean_7', 'roll_std_7', 'roll_mean_14', 'roll_mean_28',
    'price_lag_1', 'price_roll_mean_7'
]
TARGET_COLUMN = 'arrivals'

def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Calculates evaluation metrics MAE, RMSE, and WAPE."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) if np.sum(y_true) > 0 else 0.0
    return mae, rmse, wape

def train_baseline_model(processed_data_path: str, model_output_path: str):
    logging.info(f"Loading processed data from {processed_data_path}...")
    df = pd.read_parquet(processed_data_path)
    
    feat_df = generate_feature_matrix(df)
    
    # Set categorical columns for LightGBM
    cat_cols = ['commodity', 'state', 'district']
    for c in cat_cols:
        feat_df[c] = feat_df[c].astype('category')
        
    # Chronological Out-of-Time Split (Last 60 days for validation)
    split_date = feat_df['date'].max() - pd.Timedelta(days=60)
    train_mask = feat_df['date'] < split_date
    val_mask = feat_df['date'] >= split_date
    
    X_train, y_train = feat_df.loc[train_mask, FEATURE_COLUMNS], feat_df.loc[train_mask, TARGET_COLUMN]
    X_val, y_val = feat_df.loc[val_mask, FEATURE_COLUMNS], feat_df.loc[val_mask, TARGET_COLUMN]
    
    logging.info(f"Training set: {len(X_train)} samples | Validation set: {len(X_val)} samples")
    
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
    )
    
    # Validation metrics
    val_preds = model.predict(X_val)
    val_preds = np.clip(val_preds, a_min=0.0, a_max=None)
    mae, rmse, wape = evaluate_metrics(y_val.values, val_preds)
    
    logging.info("=" * 50)
    logging.info(f"VALIDATION METRICS:")
    logging.info(f"  • MAE  : {mae:.4f} Tonnes")
    logging.info(f"  • RMSE : {rmse:.4f} Tonnes")
    logging.info(f"  • WAPE : {wape * 100:.2f}%")
    logging.info("=" * 50)
    
    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)
    logging.info(f"Model saved successfully to {model_output_path}")

if __name__ == "__main__":
    train_baseline_model("./data/processed/clean_mandi_data.parquet", "./models/lightgbm_model.pkl")