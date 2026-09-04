import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_raw_datasets(data_dir: Path) -> pd.DataFrame:
    """Loads all commodity CSVs, standardizes schema, and handles raw anomalies."""
    commodities = ['tomato', 'potato', 'onion', 'wheat']
    dfs = []
    
    for comm in commodities:
        file_path = data_dir / f"{comm}.csv"
        if not file_path.exists():
            logging.warning(f"File missing: {file_path}. Skipping.")
            continue
            
        logging.info(f"Loading {comm}.csv...")
        df = pd.read_csv(file_path)
        df['commodity'] = comm.capitalize()
        
        # Standardize column names
        df.rename(columns={
            'State Name': 'state',
            'District Name': 'district',
            'Market Name': 'market',
            'Arrivals (Tonnes)': 'arrivals',
            'Min Price (Rs./Quintal)': 'min_price',
            'Max Price (Rs./Quintal)': 'max_price',
            'Modal Price (Rs./Quintal)': 'modal_price',
            'Reported Date': 'date'
        }, inplace=True)
        
        # Date parsing
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Price anomaly cleaning
        invalid_prices = (df['modal_price'] <= 0) | (df['min_price'] > df['max_price'])
        df.loc[invalid_prices, ['min_price', 'max_price', 'modal_price']] = np.nan
        
        dfs.append(df)
        
    if not dfs:
        raise FileNotFoundError("No raw CSV files were successfully loaded.")
        
    merged = pd.concat(dfs, ignore_index=True)
    return merged

def aggregate_and_clip(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates market records to District level and clips extreme arrival outliers."""
    logging.info("Aggregating records to District level...")
    
    # Aggregate to daily District level
    agg_df = df.groupby(['date', 'commodity', 'state', 'district'], as_index=False).agg({
        'arrivals': 'sum',
        'modal_price': 'mean'
    })
    
    # Clip extreme arrival outliers at 99.5th percentile per commodity-district
    logging.info("Clipping extreme arrival anomalies...")
    def clip_series(group):
        p995 = group['arrivals'].quantile(0.995)
        group['arrivals'] = np.clip(group['arrivals'], a_min=0.0, a_max=p995)
        return group
        
    agg_df = agg_df.groupby(['commodity', 'district'], group_keys=False).apply(clip_series)
    return agg_df

def build_continuous_time_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Reindexes each Commodity-District series to a continuous daily date grid."""
    logging.info("Building continuous daily time grid...")
    
    combos = df[['commodity', 'state', 'district']].drop_duplicates()
    all_dates = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
    
    grid_list = []
    for _, row in combos.iterrows():
        sub_df = pd.DataFrame({
            'date': all_dates,
            'commodity': row['commodity'],
            'state': row['state'],
            'district': row['district']
        })
        grid_list.append(sub_df)
        
    full_grid = pd.concat(grid_list, ignore_index=True)
    merged_grid = pd.merge(full_grid, df, on=['date', 'commodity', 'state', 'district'], how='left')
    
    # Impute missing sequence records
    merged_grid['arrivals'] = merged_grid['arrivals'].fillna(0.0)
    merged_grid['modal_price'] = merged_grid.groupby(['commodity', 'district'])['modal_price'].transform(
        lambda x: x.ffill().bfill()
    )
    # Fill remaining prices with global median if an entire district lacks prices
    merged_grid['modal_price'] = merged_grid['modal_price'].fillna(merged_grid['modal_price'].median())
    
    return merged_grid.sort_values(by=['commodity', 'district', 'date']).reset_index(drop=True)

def run_data_pipeline(raw_dir: str, output_path: str) -> pd.DataFrame:
    """Executes full ETL data processing pipeline and persists clean dataset."""
    raw_df = load_raw_datasets(Path(raw_dir))
    agg_df = aggregate_and_clip(raw_df)
    clean_df = build_continuous_time_grid(agg_df)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(output_path, index=False)
    logging.info(f"Clean continuous dataset saved to {output_path}. Total rows: {len(clean_df)}")
    return clean_df

if __name__ == "__main__":
    run_data_pipeline("./data/raw", "./data/processed/clean_mandi_data.parquet")