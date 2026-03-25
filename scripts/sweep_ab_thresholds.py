"""
Sweep thresholds for both baseline and A/B (rich model + volatility sizing).
Uses `ab_backtest.py` logic inline for speed and writes `backtests/ab_threshold_sweep.csv`.
"""
import os, sys, joblib
import pandas as pd
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), '..')
ENG = os.path.join(BASE, 'datasets', 'engineered_features.parquet')
RICH = os.path.join(BASE, 'datasets', 'rich_features.parquet')
MODEL_BASE = os.path.join(BASE, 'models', 'lgbm_return_v1.pkl')
MODEL_RICH = os.path.join(BASE, 'models', 'lgbm_return_rich.pkl')
OUT = os.path.join(BASE, 'backtests', 'ab_threshold_sweep.csv')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# params
TARGET_VOL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
NOTIONAL_USD = 25.0
FEE_PCT = 0.001
SLIP_PCT = 0.001

# load data
if os.path.exists(RICH):
    df = pd.read_parquet(RICH)
else:
    df = pd.read_parquet(ENG)

# prepare
df = df.sort_values(['symbol','timestamp']).reset_index(drop=True)
if 'price_next' not in df.columns:
    df['price_next'] = df.groupby('symbol')['price'].shift(-1)

df = df.dropna(subset=['price_next']).reset_index(drop=True)

# load models
if not os.path.exists(MODEL_BASE):
    print('Missing baseline model'); sys.exit(1)
model_base = joblib.load(MODEL_BASE)
if not os.path.exists(MODEL_RICH):
    print('Missing rich model, training...')
    os.system(f'python "{os.path.join(BASE, "scripts", "train_save_rich.py")}"')
model_rich = joblib.load(MODEL_RICH) if os.path.exists(MODEL_RICH) else model_base

# feature matrices
BASE_FEATS = ['event_id','qty','price','fee_usd','price_change','price_roll3']
RICH_FEATS = ['qty','price','ret','rv_10','rv_10_ann','atr','vol_spike']
for c in BASE_FEATS:
    if c not in df.columns: df[c]=0.0
for c in RICH_FEATS:
    if c not in df.columns: df[c]=0.0
X_base = df[BASE_FEATS].fillna(0.0)
X_rich = df[RICH_FEATS].fillna(0.0)
pred_base = model_base.predict(X_base)
pred_rich = model_rich.predict(X_rich)

thresholds = np.concatenate((np.linspace(0.00001, 0.0002, 20), np.linspace(0.0003, 0.001, 10), np.linspace(0.002, 0.01, 10)))
rows = []
for thr in thresholds:
    # baseline
    mask_b = pred_base >= thr
    sel_b = df[mask_b].copy()
    sel_b['gross_ret'] = (sel_b['price_next'] - sel_b['price']) / sel_b['price']
    sel_b['pnl_net'] = sel_b['gross_ret'] * NOTIONAL_USD - (sel_b['fee_usd'].fillna(NOTIONAL_USD*(FEE_PCT+SLIP_PCT)))
    n_b = len(sel_b)
    total_b = sel_b['pnl_net'].sum() if n_b>0 else 0.0

    # rich + vol sizing
    mask_r = pred_rich >= thr
    sel_r = df[mask_r].copy()
    pnls_r = []
    for _, rr in sel_r.iterrows():
        vol_ann = float(rr.get('rv_10_ann') or 0.0)
        entry = float(rr['price'])
        exit_p = float(rr['price_next'])
        gross_ret = (exit_p - entry)/entry
        if vol_ann<=0:
            continue
        target_dollar_vol = TARGET_VOL * max(1.0, getattr(__import__('config'), 'INITIAL_CAPITAL', 100.0))
        notional = target_dollar_vol / vol_ann
        if (notional*entry) < getattr(__import__('config'), 'MIN_ORDER_VALUE', 0.0):
            continue
        pnl = gross_ret * notional - (FEE_PCT+SLIP_PCT)*notional
        pnls_r.append(pnl)
    n_r = len(pnls_r)
    total_r = sum(pnls_r)

    rows.append({'threshold': thr, 'n_base': n_b, 'total_pnl_base': total_b, 'n_rich': n_r, 'total_pnl_rich': total_r})

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print('Wrote', OUT)
print(out.head(10).to_string(index=False))
