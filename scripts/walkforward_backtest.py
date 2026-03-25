"""
Rolling walk-forward backtest using top-percentile triggers.
- Uses `datasets/engineered_features.parquet` as master timeline.
- Baseline: `models/lgbm_return_v1.pkl` (fixed notional)
- Rich: `models/lgbm_return_rich.pkl` + volatility sizing (uses rv_10_ann from rich features)

Outputs `backtests/walkforward_summary.csv` and prints summary.
"""
import os, sys, joblib, math
import pandas as pd
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), '..')
ENG = os.path.join(BASE, 'datasets', 'engineered_features.parquet')
RICH = os.path.join(BASE, 'datasets', 'rich_features.parquet')
MODEL_BASE = os.path.join(BASE, 'models', 'lgbm_return_v1.pkl')
MODEL_RICH = os.path.join(BASE, 'models', 'lgbm_return_rich.pkl')
OUT = os.path.join(BASE, 'backtests', 'walkforward_summary.csv')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# params
TOP_PCTS = [1,2,5,10]
TRAIN_FRAC = 0.6
TEST_FRAC = 0.2
STEP_FRAC = 0.2
# Ensure project root is on sys.path so `import config` works
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import config
NOTIONAL_USD = getattr(config, 'PRINCIPAL', 25.0)
FEE_PCT = getattr(config, 'ESTIMATED_FEE_PCT', 0.001)
SLIP_PCT = getattr(config, 'ESTIMATED_SLIPPAGE_PCT', 0.001)
TARGET_VOL = float(sys.argv[1]) if len(sys.argv)>1 else 0.02
MIN_ORDER_VALUE = getattr(__import__('config'), 'MIN_ORDER_VALUE', 0.0)

# Load master engineered features
if not os.path.exists(ENG):
    print('Missing engineered features:', ENG); sys.exit(1)
df = pd.read_parquet(ENG).sort_values(['timestamp','symbol']).reset_index(drop=True)
if 'price_next' not in df.columns:
    df['price_next'] = df.groupby('symbol')['price'].shift(-1)
df = df.dropna(subset=['price_next']).reset_index(drop=True)

# Load rich preds if available
rich_preds_map = {}
if os.path.exists(RICH):
    dr = pd.read_parquet(RICH).sort_values(['timestamp','symbol']).reset_index(drop=True)
    if 'rv_10_ann' not in dr.columns:
        dr['ret'] = dr.groupby('symbol')['price'].pct_change().fillna(0)
        ANNUALIZE = np.sqrt(252*24*4)
        dr['rv_10'] = dr.groupby('symbol')['ret'].rolling(10, min_periods=2).std().reset_index(level=0, drop=True)
        dr['rv_10_ann'] = dr['rv_10'] * ANNUALIZE
    # build map by (timestamp,symbol)
    for i,row in dr.iterrows():
        key = (row['timestamp'], row['symbol'])
        rich_preds_map[key] = {
            'rv_10_ann': float(row.get('rv_10_ann') or 0.0)
        }

# Load models
if not os.path.exists(MODEL_BASE):
    print('Missing baseline model', MODEL_BASE); sys.exit(1)
model_base = joblib.load(MODEL_BASE)
if not os.path.exists(MODEL_RICH):
    print('Missing rich model, training...')
    os.system(f'python "{os.path.join(BASE, "scripts", "train_save_rich.py")}"')
model_rich = joblib.load(MODEL_RICH) if os.path.exists(MODEL_RICH) else model_base

# Prepare features matrices
BASE_FEATS = ['event_id','qty','price','fee_usd','price_change','price_roll3']
for c in BASE_FEATS:
    if c not in df.columns: df[c]=0.0
X_base = df[BASE_FEATS].fillna(0.0)
pred_base_all = model_base.predict(X_base)

# For rich model, try to align
if os.path.exists(RICH):
    dr = pd.read_parquet(RICH).sort_values(['timestamp','symbol']).reset_index(drop=True)
    RICH_FEATS = ['qty','price','ret','rv_10','rv_10_ann','atr','vol_spike']
    for c in RICH_FEATS:
        if c not in dr.columns: dr[c]=0.0
    X_r = dr[RICH_FEATS].fillna(0.0)
    pred_rich_all = model_rich.predict(X_r)
    # align to df by order; if lengths equal assume correspondence
    if len(pred_rich_all) >= len(df):
        pred_rich_all = pred_rich_all[:len(df)]
    else:
        # pad with baseline mean
        pad = np.full(len(df)-len(pred_rich_all), np.mean(pred_rich_all))
        pred_rich_all = np.concatenate([pred_rich_all, pad])
else:
    pred_rich_all = pred_base_all

# attach preds to df
df['pred_base'] = pred_base_all
df['pred_rich'] = pred_rich_all

# unique sorted timestamps
uniq_ts = sorted(df['timestamp'].unique())
if not uniq_ts:
    print('No timestamps'); sys.exit(1)

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

results = []
for top_pct in TOP_PCTS:
    agg_base_pnl = 0.0
    agg_rich_pnl = 0.0
    agg_base_trades = 0
    agg_rich_trades = 0
    for (train_df, test_df) in folds:
        # baseline cutoff
        cutoff_b = np.percentile(train_df['pred_base'], 100-top_pct)
        mask_b = test_df['pred_base'] >= cutoff_b
        sel_b = test_df[mask_b].copy()
        sel_b['gross_ret'] = (sel_b['price_next'] - sel_b['price']) / sel_b['price']
        sel_b['pnl_net'] = sel_b['gross_ret'] * NOTIONAL_USD - (sel_b['fee_usd'].fillna(NOTIONAL_USD*(FEE_PCT+SLIP_PCT)))
        agg_base_pnl += sel_b['pnl_net'].sum()
        agg_base_trades += len(sel_b)

        # rich cutoff
        cutoff_r = np.percentile(train_df['pred_rich'], 100-top_pct)
        mask_r = test_df['pred_rich'] >= cutoff_r
        sel_r = test_df[mask_r].copy()
        pnls_r = []
        for _, rr in sel_r.iterrows():
            entry = float(rr['price']); exit_p = float(rr['price_next'])
            gross = (exit_p - entry) / entry
            key = (rr['timestamp'], rr['symbol'])
            vol_ann = rich_preds_map.get(key, {}).get('rv_10_ann', 0.0)
            if vol_ann <= 0:
                continue
            target_dollar_vol = TARGET_VOL * max(1.0, getattr(__import__('config'), 'INITIAL_CAPITAL', 100.0))
            notional = target_dollar_vol / vol_ann
            if (notional * entry) < MIN_ORDER_VALUE:
                # Fallback behavior for paper walk-forward: use fixed notional instead of skipping
                notional = NOTIONAL_USD
            pnl = gross * notional - (FEE_PCT + SLIP_PCT) * notional
            pnls_r.append(pnl)
        agg_rich_pnl += sum(pnls_r)
        agg_rich_trades += len(pnls_r)

    results.append({'top_pct': top_pct, 'base_trades': agg_base_trades, 'base_pnl': agg_base_pnl, 'rich_trades': agg_rich_trades, 'rich_pnl': agg_rich_pnl})

out = pd.DataFrame(results)
out.to_csv(OUT, index=False)
print('Wrote', OUT)
print(out.to_string(index=False))
