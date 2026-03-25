"""
Sweep sizing-impact by varying target annual volatility and computing ex-post PnL
using `datasets/rich_features.parquet` (requires `rv_10_ann`, `price`, `price_next`).
Outputs `backtests/sizing_impact.csv` and prints a short summary.
"""
import os
import pandas as pd
import numpy as np
import json
from pathlib import Path

BASE = os.path.join(os.path.dirname(__file__), '..')
IN_PARQ = os.path.join(BASE, 'datasets', 'rich_features.parquet')
FALLBACK_PARQ = os.path.join(BASE, 'datasets', 'engineered_features.parquet')
OUT_DIR = os.path.join(BASE, 'backtests')
OUT_CSV = os.path.join(OUT_DIR, 'sizing_impact.csv')

if not os.path.exists(IN_PARQ):
    print('Missing input:', IN_PARQ)
    raise SystemExit(1)

os.makedirs(OUT_DIR, exist_ok=True)

# Prefer rich features (may contain realized vol) but engineered features have sequential prices
if os.path.exists(IN_PARQ):
    try:
        df = pd.read_parquet(IN_PARQ)
    except Exception:
        df = pd.read_parquet(FALLBACK_PARQ)
else:
    df = pd.read_parquet(FALLBACK_PARQ)

# If rich features did not include price_next or rv_10_ann, compute them from engineered/raw data
if 'price_next' not in df.columns or 'rv_10_ann' not in df.columns:
    # load engineered features (sequential price events)
    eng = pd.read_parquet(FALLBACK_PARQ)
    eng = eng.sort_values(['symbol','timestamp']).reset_index(drop=True)
    eng['price_next'] = eng.groupby('symbol')['price'].shift(-1)
    eng['ret'] = eng.groupby('symbol')['price'].pct_change().fillna(0)
    ANNUALIZE = np.sqrt(252*24*4)
    eng['rv_10'] = eng.groupby('symbol')['ret'].rolling(10, min_periods=2).std().reset_index(level=0, drop=True)
    eng['rv_10_ann'] = eng['rv_10'] * ANNUALIZE
    df = eng

# Use only rows with a next-price (realized next-event return) and positive ann vol
df = df.dropna(subset=['price','price_next','rv_10_ann'])
if df.empty:
    print('No usable rows in input')
    raise SystemExit(1)

# Compute realized return
df['real_ret'] = (df['price_next'] - df['price']) / df['price']

# Use baseline capital from config if available else 100
try:
    import config
    BASE_CAP = float(getattr(config, 'INITIAL_CAPITAL', 100.0))
except Exception:
    BASE_CAP = 100.0

# Grid of annual target vols (fractions of account equity used as target dollar vol fraction)
TARGET_VOLS = [0.01, 0.02, 0.05, 0.1, 0.2]

rows = []
for tv in TARGET_VOLS:
    qtys = []
    notionals = []
    pnls = []
    used = 0
    for _, r in df.iterrows():
        vol_ann = float(r.get('rv_10_ann', 0.0))
        price = float(r['price'])
        if vol_ann <= 0 or price <= 0:
            continue
        # target dollar volatility per year
        target_dollar_vol = tv * max(1.0, BASE_CAP)
        notional = target_dollar_vol / vol_ann
        qty = notional / price
        # enforce min order value if present
        # if (qty*price) < getattr(config, 'MIN_ORDER_VALUE', 0): skip
        pnl = float(r['real_ret']) * notional
        qtys.append(qty)
        notionals.append(notional)
        pnls.append(pnl)
        used += 1
    total_pnl = sum(pnls)
    mean_pnl = np.mean(pnls) if pnls else 0.0
    avg_notional = np.mean(notionals) if notionals else 0.0
    rows.append({'target_vol': tv, 'n_trades': used, 'total_pnl': total_pnl, 'mean_pnl_per_trade': mean_pnl, 'avg_notional': avg_notional})

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_CSV, index=False)
print('Wrote', OUT_CSV)
print(out_df.to_string(index=False))
