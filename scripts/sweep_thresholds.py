"""
Sweep prediction thresholds and produce expectancy vs threshold CSV.
Reads `datasets/engineered_features.parquet` and `models/lgbm_return_v1.pkl`.
Saves `backtests/expectancy_vs_threshold.csv`.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), '..')
FEAT_PARQ = os.path.join(BASE, 'datasets', 'engineered_features.parquet')
MODEL_PATH = os.path.join(BASE, 'models', 'lgbm_return_v1.pkl')
OUT_CSV = os.path.join(BASE, 'backtests', 'expectancy_vs_threshold.csv')
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

if not os.path.exists(FEAT_PARQ):
    print('Missing features:', FEAT_PARQ)
    raise SystemExit(1)
if not os.path.exists(MODEL_PATH):
    print('Missing model:', MODEL_PATH)
    raise SystemExit(1)

df = pd.read_parquet(FEAT_PARQ).sort_values(['symbol','timestamp']).reset_index(drop=True)
# ensure next price
if 'price_next' not in df.columns:
    df['price_next'] = df.groupby('symbol')['price'].shift(-1)

df = df.dropna(subset=['price_next']).reset_index(drop=True)

model = joblib.load(MODEL_PATH)
FEATURES = ['event_id','qty','price','fee_usd','price_change','price_roll3']
for c in FEATURES:
    if c not in df.columns:
        df[c] = 0.0
X = df[FEATURES].fillna(0.0)
preds = model.predict(X)
df['pred'] = preds

thresholds = np.concatenate((np.linspace(0.00001, 0.0002, 10), np.linspace(0.0003, 0.001, 8), np.linspace(0.002, 0.01, 10)))
rows = []
PRINCIPAL = 25.0
FEE_PCT = float(os.getenv('ESTIMATED_FEE_PCT', '0.001'))
SLIP_PCT = float(os.getenv('ESTIMATED_SLIPPAGE_PCT', '0.001'))

for thr in thresholds:
    sel = df[df['pred'] >= thr]
    if sel.empty:
        rows.append({'threshold': thr, 'n_trades': 0, 'win_rate': None, 'avg_pnl_usd': None, 'expectancy': None})
        continue
    sel['gross_ret'] = (sel['price_next'] - sel['price']) / sel['price']
    sel['pnl_usd_gross'] = sel['gross_ret'] * PRINCIPAL
    sel['fee_usd_est'] = sel['fee_usd'].fillna(0.0)
    mask = sel['fee_usd_est'] == 0.0
    sel.loc[mask, 'fee_usd_est'] = PRINCIPAL * (FEE_PCT + SLIP_PCT)
    sel['pnl_usd_net'] = sel['pnl_usd_gross'] - sel['fee_usd_est']
    n = len(sel)
    wins = (sel['pnl_usd_net'] > 0).sum()
    avg_pnl = sel['pnl_usd_net'].mean()
    expectancy = avg_pnl
    rows.append({'threshold': thr, 'n_trades': n, 'win_rate': wins / n, 'avg_pnl_usd': avg_pnl, 'expectancy': expectancy})

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)
print('Wrote', OUT_CSV)
