"""
Build richer features: realized volatility (rolling std of returns), ATR (if price bars present), and volume spike indicator.
Reads `datasets/raw_trades_snapshot.parquet` or `datasets/engineered_features.parquet` and outputs `datasets/rich_features.parquet`.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime

BASE = os.path.join(os.path.dirname(__file__), '..')
IN_PARQ = os.path.join(BASE, 'datasets', 'engineered_features.parquet')
OUT_PARQ = os.path.join(BASE, 'datasets', 'rich_features.parquet')

if not os.path.exists(IN_PARQ):
    print('Input features missing:', IN_PARQ)
    raise SystemExit(1)

df = pd.read_parquet(IN_PARQ)
# Ensure ordered
df = df.sort_values(['symbol','timestamp']).reset_index(drop=True)

# Compute returns at event resolution
df['ret'] = df.groupby('symbol')['price'].pct_change().fillna(0)

# Rolling realized volatility (window 10 events) - sample -> annualize by sqrt(252*24*(60/15)) approximation for 15m bars
# Approx factor for 15m bars to annual: sqrt(252*24*4)=sqrt(24192)=155.57
ANNUALIZE = np.sqrt(252*24*4)

df['rv_10'] = df.groupby('symbol')['ret'].rolling(10, min_periods=2).std().reset_index(level=0, drop=True)
df['rv_10_ann'] = df['rv_10'] * ANNUALIZE

# ATR proxy: use rolling high-low if present; fallback to rv_10 * price
if all(c in df.columns for c in ('price',)):
    # we don't have high/low; approximate ATR as rv * price
    df['atr'] = df['rv_10'] * df['price']
else:
    df['atr'] = 0.0

# Volume spike: compare qty to rolling median qty
df['qty'] = pd.to_numeric(df['qty'].fillna(0.0))
df['qty_med3'] = df.groupby('symbol')['qty'].rolling(5, min_periods=1).median().reset_index(level=0, drop=True)
df['vol_spike'] = (df['qty'] > 3 * df['qty_med3']).astype(int)

# Fill NaNs
for c in ['rv_10','rv_10_ann','atr','vol_spike']:
    df[c] = df[c].fillna(0.0)

# Save subset of columns
cols_out = ['session_file','timestamp','symbol','event','qty','price','price_next','ret','rv_10','rv_10_ann','atr','vol_spike','order_id','fee_usd','raw']
out_df = df[[c for c in cols_out if c in df.columns]]

out_df.to_parquet(OUT_PARQ)
print('Wrote', OUT_PARQ)
