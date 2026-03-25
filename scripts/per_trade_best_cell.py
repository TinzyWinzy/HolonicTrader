"""
Extract per-trade details for a chosen walkforward cell (target_vol, top_pct).
Saves `backtests/per_trade_best_cell.csv`.
"""
import os, sys, joblib
import pandas as pd
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), '..')
ENG = os.path.join(BASE, 'datasets', 'engineered_features.parquet')
RICH = os.path.join(BASE, 'datasets', 'rich_features.parquet')
MODEL_BASE = os.path.join(BASE, 'models', 'lgbm_return_v1.pkl')
MODEL_RICH = os.path.join(BASE, 'models', 'lgbm_return_rich.pkl')
OUT = os.path.join(BASE, 'backtests', 'per_trade_best_cell.csv')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# args
TARGET_VOL = float(sys.argv[1]) if len(sys.argv)>1 else 0.02
TOP_PCT = int(sys.argv[2]) if len(sys.argv)>2 else 1

# ensure config import
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
import config

NOTIONAL_USD = getattr(config, 'PRINCIPAL', 25.0)
FEE_PCT = getattr(config, 'ESTIMATED_FEE_PCT', 0.001)
SLIP_PCT = getattr(config, 'ESTIMATED_SLIPPAGE_PCT', 0.001)
MIN_ORDER_VALUE = getattr(config, 'MIN_ORDER_VALUE', 0.0)

# load data
if not os.path.exists(ENG):
    print('Missing engineered features:', ENG); raise SystemExit(1)
df = pd.read_parquet(ENG).sort_values(['timestamp','symbol']).reset_index(drop=True)
if 'price_next' not in df.columns:
    df['price_next'] = df.groupby('symbol')['price'].shift(-1)
df = df.dropna(subset=['price_next']).reset_index(drop=True)

# rich map
rich_map = {}
if os.path.exists(RICH):
    dr = pd.read_parquet(RICH).sort_values(['timestamp','symbol']).reset_index(drop=True)
    if 'rv_10_ann' not in dr.columns:
        dr['ret'] = dr.groupby('symbol')['price'].pct_change().fillna(0)
        ANNUALIZE = np.sqrt(252*24*4)
        dr['rv_10'] = dr.groupby('symbol')['ret'].rolling(10, min_periods=2).std().reset_index(level=0, drop=True)
        dr['rv_10_ann'] = dr['rv_10'] * ANNUALIZE
    for i,row in dr.iterrows():
        rich_map[(row['timestamp'], row['symbol'])] = {'rv_10_ann': float(row.get('rv_10_ann') or 0.0)}

# load models
if not os.path.exists(MODEL_BASE):
    print('Missing baseline model', MODEL_BASE); raise SystemExit(1)
model_base = joblib.load(MODEL_BASE)
if not os.path.exists(MODEL_RICH):
    print('Rich model missing; training...')
    os.system(f'python "{os.path.join(BASE, "scripts", "train_save_rich.py")}"')
model_rich = joblib.load(MODEL_RICH) if os.path.exists(MODEL_RICH) else model_base

# preds
BASE_FEATS = ['event_id','qty','price','fee_usd','price_change','price_roll3']
for c in BASE_FEATS:
    if c not in df.columns: df[c]=0.0
X_base = df[BASE_FEATS].fillna(0.0)
pred_base_all = model_base.predict(X_base)

if os.path.exists(RICH):
    dr = pd.read_parquet(RICH).sort_values(['timestamp','symbol']).reset_index(drop=True)
    RICH_FEATS = ['qty','price','ret','rv_10','rv_10_ann','atr','vol_spike']
    for c in RICH_FEATS:
        if c not in dr.columns: dr[c]=0.0
    X_r = dr[RICH_FEATS].fillna(0.0)
    pred_rich_all = model_rich.predict(X_r)
    if len(pred_rich_all) >= len(df):
        pred_rich_all = pred_rich_all[:len(df)]
    else:
        pad = np.full(len(df)-len(pred_rich_all), np.mean(pred_rich_all))
        pred_rich_all = np.concatenate([pred_rich_all, pad])
else:
    pred_rich_all = pred_base_all

# attach preds
df['pred_base'] = pred_base_all
df['pred_rich'] = pred_rich_all

# build folds
uniq_ts = sorted(df['timestamp'].unique())
TRAIN_FRAC=0.6; TEST_FRAC=0.2; STEP_FRAC=0.2
train_n = max(1, int(len(uniq_ts) * TRAIN_FRAC))
test_n = max(1, int(len(uniq_ts) * TEST_FRAC))
step_n = max(1, int(len(uniq_ts) * STEP_FRAC))
folds = []
for start in range(0, len(uniq_ts) - train_n - test_n + 1, step_n):
    train_ts = uniq_ts[start:start+train_n]
    test_ts = uniq_ts[start+train_n:start+train_n+test_n]
    if not test_ts: continue
    train_df = df[df['timestamp'].isin(train_ts)]
    test_df = df[df['timestamp'].isin(test_ts)]
    folds.append((train_df, test_df))

# collect per-trade rows
rows = []
for train_df, test_df in folds:
    cutoff_r = np.percentile(train_df['pred_rich'], 100-TOP_PCT)
    mask_r = test_df['pred_rich'] >= cutoff_r
    sel_r = test_df[mask_r].copy()
    for _, rr in sel_r.iterrows():
        entry = float(rr['price']); exit_p = float(rr['price_next'])
        gross = (exit_p - entry) / entry
        key = (rr['timestamp'], rr['symbol'])
        vol_ann = rich_map.get(key, {}).get('rv_10_ann', 0.0)
        if vol_ann <= 0:
            notional = NOTIONAL_USD
        else:
            target_dollar_vol = TARGET_VOL * max(1.0, getattr(config, 'INITIAL_CAPITAL', 100.0))
            notional = target_dollar_vol / vol_ann
            if (notional * entry) < MIN_ORDER_VALUE:
                notional = NOTIONAL_USD
        pnl = gross * notional - (FEE_PCT + SLIP_PCT) * notional
        rows.append({'timestamp': rr['timestamp'], 'symbol': rr['symbol'], 'entry': entry, 'exit': exit_p, 'pred_rich': rr['pred_rich'], 'vol_ann': vol_ann, 'notional': notional, 'pnl': pnl})

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print('Wrote', OUT)
print(out.to_string(index=False))
