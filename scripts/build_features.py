"""
Simple feature builder for exported trades.
Reads `datasets/raw_trades_snapshot.csv` and emits `datasets/engineered_features.parquet`.
Produces minimal features per event: timestamp_epoch, event_type_id, qty, price, rolling_price_change (per symbol, window=3 events).
"""
import os
import pandas as pd
from datetime import datetime

IN_CSV = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'raw_trades_snapshot.csv')
OUT_PARQ = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'engineered_features.parquet')

if not os.path.exists(IN_CSV):
    print('Input CSV not found:', IN_CSV)
    raise SystemExit(1)

df = pd.read_csv(IN_CSV)
# Normalize timestamp
try:
    df['ts_dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['timestamp_epoch'] = df['ts_dt'].astype('int64') // 10**9
except Exception:
    # Fallback: current time
    df['timestamp_epoch'] = int(datetime.now().timestamp())

# map event to id
evt_map = {'ENTRY':1, 'EXIT':2, 'STOP_PLACED':3, 'FILL':4, 'ORDER_RESULT':5, 'FEE_REPORTED':6}
df['event_id'] = df['event'].map(evt_map).fillna(0).astype(int)

# simple numeric features
for col in ['qty','price','fee_usd']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

# rolling price change per symbol (by event order)
df.sort_values(['symbol','timestamp_epoch'], inplace=True)

df['price_lag1'] = df.groupby('symbol')['price'].shift(1)
df['price_change'] = (df['price'] - df['price_lag1']) / df['price_lag1'].replace({0:pd.NA})
df['price_change'] = df['price_change'].fillna(0.0)

# rolling mean over last 3 events per symbol
df['price_roll3'] = df.groupby('symbol')['price_change'].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)

# select features to write
feat_cols = ['session_file','timestamp','timestamp_epoch','symbol','event','event_id','qty','price','fee_usd','price_change','price_roll3','order_id','raw']
feat_df = df[feat_cols]

# write parquet
try:
    feat_df.to_parquet(OUT_PARQ)
    print('Wrote features to', OUT_PARQ)
except Exception as e:
    print('Parquet write failed:', e)
    out_csv = OUT_PARQ.replace('.parquet', '.csv')
    feat_df.to_csv(out_csv, index=False)
    print('Wrote features CSV to', out_csv)
