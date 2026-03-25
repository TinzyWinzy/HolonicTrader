"""
Find positive-expectancy symbol pairs using real extracted events (no mock data).
Uses `datasets/raw_trades_snapshot.parquet` and `datasets/engineered_features.parquet`.
Outputs `reports/positive_pairs.csv` and `reports/positive_pairs.json`.
"""
import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure project root on path
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
import config

BASE = PROJECT_DIR
DATA_PARQ = os.path.join(BASE, 'datasets', 'raw_trades_snapshot.parquet')
FEAT_PARQ = os.path.join(BASE, 'datasets', 'engineered_features.parquet')
OUT_CSV = os.path.join(BASE, 'reports', 'positive_pairs.csv')
OUT_JSON = os.path.join(BASE, 'reports', 'positive_pairs.json')
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

MIN_TRADES = int(sys.argv[1]) if len(sys.argv) > 1 else 5
FEE_PCT = getattr(config, 'ESTIMATED_FEE_PCT', 0.001)
SLIP_PCT = getattr(config, 'ESTIMATED_SLIPPAGE_PCT', 0.001)
PRINCIPAL = getattr(config, 'PRINCIPAL', 25.0)

# Prefer engineered features (has price_next). Fallback to raw dataset.
if os.path.exists(FEAT_PARQ):
    df = pd.read_parquet(FEAT_PARQ)
else:
    df = pd.read_parquet(DATA_PARQ)

# Ensure ordering and next price
if 'price_next' not in df.columns:
    df = df.sort_values(['symbol','timestamp']).reset_index(drop=True)
    df['price_next'] = df.groupby('symbol')['price'].shift(-1)

# Drop rows without next price (cannot compute realized next-event return)
df = df.dropna(subset=['price_next'])

# Compute gross return and net return using available fee info where present
# gross_ret = (price_next - price)/price
# fee_usd if present reduces pnl_usd directly; otherwise use fee_pct+slip_pct

df['gross_ret'] = (df['price_next'] - df['price']) / df['price'].replace({0: np.nan})
df['gross_ret'] = df['gross_ret'].fillna(0.0)

# Compute pnl_usd before fees
df['pnl_usd_gross'] = df['gross_ret'] * PRINCIPAL

# Use fee_usd column if present, else estimate fees as principal*(fee+slip)
if 'fee_usd' in df.columns:
    df['fee_usd_est'] = df['fee_usd'].fillna(0.0)
else:
    df['fee_usd_est'] = 0.0

# Add estimated proportional fee when fee_usd not present
mask_no_fee = df['fee_usd_est'] == 0.0
df.loc[mask_no_fee, 'fee_usd_est'] = PRINCIPAL * (FEE_PCT + SLIP_PCT)

# net pnl
df['pnl_usd_net'] = df['pnl_usd_gross'] - df['fee_usd_est']

# per-symbol aggregations
group = df.groupby('symbol')
summary = group['pnl_usd_net'].agg(['count','mean','sum'])
summary = summary.rename(columns={'count':'n_trades','mean':'avg_pnl_usd','sum':'total_pnl_usd'})

# win rate
win_rate = group.apply(lambda g: (g['pnl_usd_net'] > 0).sum() / max(1, len(g)))
summary['win_rate'] = win_rate
summary['expectancy_per_trade_usd'] = summary['avg_pnl_usd']

# filter
candidates = summary[summary['n_trades'] >= MIN_TRADES]
candidates = candidates.sort_values('expectancy_per_trade_usd', ascending=False)

# Save results
candidates.to_csv(OUT_CSV)
with open(OUT_JSON, 'w') as fh:
    fh.write(candidates.reset_index().to_json(orient='records', indent=2))

print('Wrote', OUT_CSV)
print('Top candidates:')
print(candidates.head(20))
