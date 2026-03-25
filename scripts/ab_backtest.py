"""
A/B backtest: compare baseline model (lgbm_return_v1.pkl) with model+sizing (lgbm_return_rich.pkl + volatility sizing).
Saves results to `backtests/ab_backtest.csv` and prints summary metrics.
Updated to use refined feature sets from feature_engineering.py pipeline.
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np
import warnings

BASE = os.path.join(os.path.dirname(__file__), '..')
ENG = os.path.join(BASE, 'datasets', 'engineered_features.parquet')
ENG_V2 = os.path.join(BASE, 'datasets', 'engineered_features_v2.parquet')
RICH = os.path.join(BASE, 'datasets', 'rich_features.parquet')
MODEL_BASE = os.path.join(BASE, 'models', 'lgbm_return_v1.pkl')
MODEL_RICH = os.path.join(BASE, 'models', 'lgbm_return_rich.pkl')
OUT = os.path.join(BASE, 'backtests', 'ab_backtest.csv')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Parameters
THRESHOLD = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0005
TARGET_VOL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
NOTIONAL_USD = 25.0
FEE_PCT = 0.001
SLIP_PCT = 0.001

# --- Load data (prefer engineered v2) ---
if os.path.exists(ENG_V2):
    df = pd.read_parquet(ENG_V2)
    print(f"Loaded engineered_features_v2.parquet: {df.shape}")
elif os.path.exists(RICH):
    df = pd.read_parquet(RICH)
    print(f"Loaded rich_features.parquet: {df.shape}")
else:
    df = pd.read_parquet(ENG)
    print(f"Loaded engineered_features.parquet: {df.shape}")

# Inspect a few rows for manual check
print("Sample rows from features:")
print(df.head(5))

# Ensure sequential order and price_next
df = df.sort_values(['symbol','timestamp']).reset_index(drop=True)
if 'price_next' not in df.columns:
    df['price_next'] = df.groupby('symbol')['price'].shift(-1)

df = df.dropna(subset=['price_next'])

# Compute target if not present
if 'target' not in df.columns:
    df['target'] = np.log(df['price_next'] / df['price'])

# split time-based: last 20% as test
uniq_ts = sorted(df['timestamp'].unique())
cut_idx = int(len(uniq_ts)*0.8)
cut_ts = uniq_ts[cut_idx] if uniq_ts else None
if cut_ts:
    test_df = df[df['timestamp'] >= cut_ts].copy()
else:
    test_df = df.copy()

print(f"\nTest set: {test_df.shape[0]} rows from {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

# --- Load models ---
if not os.path.exists(MODEL_BASE):
    print('Missing baseline model:', MODEL_BASE)
    print('Run: python scripts/train_quick_model.py')
    sys.exit(1)
model_base = joblib.load(MODEL_BASE)

if not os.path.exists(MODEL_RICH):
    print('Rich model missing; training now...')
    os.system(f'python "{os.path.join(BASE, "scripts", "train_save_rich.py")}"')

if os.path.exists(MODEL_RICH):
    model_rich = joblib.load(MODEL_RICH)
else:
    model_rich = model_base


# --- Prepare features for baseline model ---
# Updated features: dropped event_id, fee_usd (constant/zero variance)
BASE_FEATS = ['qty', 'price', 'price_change', 'price_roll3']
for c in BASE_FEATS:
    if c not in test_df.columns:
        test_df[c] = 0.0
X_base = test_df[BASE_FEATS].replace([np.inf, -np.inf], 0).fillna(0.0)
print("\nBASE FEATS STATS:")
print(X_base.describe())
if (X_base.abs().sum().sum() == 0) or (X_base.var().sum() < 1e-8):
    warnings.warn("All BASE_FEATS are zero or have very low variance!")
pred_base = model_base.predict(X_base)

# --- Rich model features ---
# Full feature set from feature engineering pipeline
RICH_FEATS = [
    'qty', 'price', 'ret', 'rv_10', 'rv_10_ann', 'atr', 'vol_spike',
    'hour', 'day_of_week', 'is_weekend',
    'ret_roll3', 'ret_std3', 'price_ma3', 'price_std3', 'price_ma5',
    'ret_lag1', 'ret_lag2', 'vol_lag1',
    'price_momentum', 'vol_regime'
]
# Filter to available features
RICH_FEATS = [f for f in RICH_FEATS if f in test_df.columns]
print(f"\nUsing {len(RICH_FEATS)} rich features: {RICH_FEATS}")

for c in RICH_FEATS:
    if c not in test_df.columns:
        test_df[c] = 0.0
X_rich = test_df[RICH_FEATS].replace([np.inf, -np.inf], 0).fillna(0.0)
print("\nRICH FEATS STATS:")
print(X_rich.describe())
if (X_rich.abs().sum().sum() == 0) or (X_rich.var().sum() < 1e-8):
    warnings.warn("All RICH_FEATS are zero or have very low variance!")
pred_rich = model_rich.predict(X_rich)

# --- Baseline sim: fixed notional ---
rows = []
for idx_pos, (i, r) in enumerate(test_df.iterrows()):
    pb = float(pred_base[idx_pos])
    pr = float(pred_rich[idx_pos])
    entry = float(r['price'])
    exit_p = float(r['price_next'])
    gross_ret = (exit_p - entry) / entry
    net_ret_base = gross_ret - FEE_PCT - SLIP_PCT
    pnl_base = net_ret_base * NOTIONAL_USD if pb >= THRESHOLD else 0.0

    # Rich+vol sizing
    pnl_rich = 0.0
    if pr >= THRESHOLD:
        vol_ann = float(r.get('rv_10_ann') or 0.0)
        if vol_ann > 0:
            target_dollar_vol = TARGET_VOL * max(1.0, getattr(__import__('config'), 'INITIAL_CAPITAL', 100.0))
            notional = target_dollar_vol / vol_ann
            # enforce min order value
            min_val = getattr(__import__('config'), 'MIN_ORDER_VALUE', 0.0)
            if (notional > 0) and ((notional * entry) >= min_val):
                pnl_rich = gross_ret * notional - (FEE_PCT + SLIP_PCT) * notional
        else:
            # Fallback to fixed notional if no volatility
            pnl_rich = gross_ret * NOTIONAL_USD - (FEE_PCT + SLIP_PCT) * NOTIONAL_USD
    rows.append({
        'timestamp': r['timestamp'],
        'symbol': r['symbol'],
        'entry': entry,
        'exit': exit_p,
        'pred_base': pb,
        'pred_rich': pr,
        'pnl_base': pnl_base,
        'pnl_rich': pnl_rich
    })

out = pd.DataFrame(rows)
summary = {
    'n_base_trades': int((out['pnl_base'] != 0).sum()),
    'total_pnl_base': float(out['pnl_base'].sum()),
    'n_rich_trades': int((out['pnl_rich'] != 0).sum()),
    'total_pnl_rich': float(out['pnl_rich'].sum())
}
print('\nA/B Backtest Summary')
print(summary)
out.to_csv(OUT, index=False)
print('Saved results to', OUT)

# --- JSON summary output for orchestration ---
import json as _json
json_out = OUT.replace('.csv', '.json')
summary['expectancy_base'] = float(out['pnl_base'].sum()) / max(1, summary['n_base_trades'])
summary['expectancy_rich'] = float(out['pnl_rich'].sum()) / max(1, summary['n_rich_trades'])
summary['timestamp'] = pd.Timestamp.now().isoformat()
with open(json_out, 'w') as f:
    _json.dump(summary, f, indent=2)
print('Saved JSON summary to', json_out)
