import pandas as pd
import numpy as np
import logging

def generate_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Computes calendar, lag, and rolling window features for time-series forecasting."""
    logging.info("Engineering feature matrix...")
    feat_df = df.copy()
    
    # Calendar features
    feat_df['dayofweek'] = feat_df['date'].dt.dayofweek.astype('int8')
    feat_df['month'] = feat_df['date'].dt.month.astype('int8')
    feat_df['quarter'] = feat_df['date'].dt.quarter.astype('int8')
    feat_df['dayofyear'] = feat_df['date'].dt.dayofyear.astype('int16')
    feat_df['is_weekend'] = feat_df['dayofweek'].isin([5, 6]).astype('int8')
    
    # Ensure chronological order for grouped shifts
    feat_df = feat_df.sort_values(by=['commodity', 'district', 'date']).reset_index(drop=True)
    
    # Lag features
    group_arrivals = feat_df.groupby(['commodity', 'district'])['arrivals']
    group_price = feat_df.groupby(['commodity', 'district'])['modal_price']
    
    feat_df['lag_1'] = group_arrivals.shift(1)
    feat_df['lag_7'] = group_arrivals.shift(7)
    feat_df['lag_14'] = group_arrivals.shift(14)
    feat_df['lag_28'] = group_arrivals.shift(28)
    
    # Rolling statistics (shifted by 1 day to prevent target leakage)
    feat_df['roll_mean_7'] = group_arrivals.transform(lambda x: x.shift(1).rolling(7).mean())
    feat_df['roll_std_7'] = group_arrivals.transform(lambda x: x.shift(1).rolling(7).std())
    feat_df['roll_mean_14'] = group_arrivals.transform(lambda x: x.shift(1).rolling(14).mean())
    feat_df['roll_mean_28'] = group_arrivals.transform(lambda x: x.shift(1).rolling(28).mean())
    
    feat_df['price_lag_1'] = group_price.shift(1)
    feat_df['price_roll_mean_7'] = group_price.transform(lambda x: x.shift(1).rolling(7).mean())
    
    # Drop rows containing NaNs from lagging window initializations
    before_drop = len(feat_df)
    feat_df = feat_df.dropna().reset_index(drop=True)
    logging.info(f"Feature matrix generated. Dropped {before_drop - len(feat_df)} initial warm-up rows.")
    
    return feat_df