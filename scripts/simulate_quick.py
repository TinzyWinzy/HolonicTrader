"""
Naive simulation: use `models/lgbm_return_v1.pkl` to predict next-event returns.
If prediction >= threshold, enter at current price and exit at next event price (price_next).
Subtract estimated fees and slippage per trade. Report expectancy and win-rate.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import sys

# Ensure project root is on sys.path so `import config` works when running from venv
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
import config

BASE = os.path.join(os.path.dirname(__file__), '..')
FEAT_PARQ = os.path.join(BASE, 'datasets', 'engineered_features.parquet')
MODEL_PATH = os.path.join(BASE, 'models', 'lgbm_return_v1.pkl')

THRESHOLD = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0005
NOTIONAL_USD = getattr(config, 'PRINCIPAL', 25.0)
FEE_PCT = getattr(config, 'ESTIMATED_FEE_PCT', 0.001)
SLIP_PCT = getattr(config, 'ESTIMATED_SLIPPAGE_PCT', 0.001)

if not os.path.exists(FEAT_PARQ):
    print('Features missing:', FEAT_PARQ)
    sys.exit(1)
if not os.path.exists(MODEL_PATH):
    print('Model missing:', MODEL_PATH)
    sys.exit(1)

df = pd.read_parquet(FEAT_PARQ)
# Ensure price_next available: compute from raw ordering
df = df.sort_values(['symbol','timestamp']).reset_index(drop=True)

df['price_next'] = df.groupby('symbol')['price'].shift(-1)

df = df.dropna(subset=['price_next'])

# Load model
model = joblib.load(MODEL_PATH)

FEATURES = ['event_id','qty','price','fee_usd','price_change','price_roll3']
for c in FEATURES:
    if c not in df.columns:
        df[c] = 0.0
X = df[FEATURES].fillna(0.0)

preds = model.predict(X)

df['pred'] = preds

df['trade'] = df['pred'] >= THRESHOLD

trades = df[df['trade']]

results = []
for _, r in trades.iterrows():
    entry_price = r['price']
    exit_price = r['price_next']
    # assume long
    gross_ret = (exit_price - entry_price) / entry_price
    net_ret = gross_ret - FEE_PCT - SLIP_PCT
    pnl_usd = net_ret * NOTIONAL_USD
    results.append({'symbol': r['symbol'], 'entry_price': entry_price, 'exit_price': exit_price,
                    'gross_ret': gross_ret, 'net_ret': net_ret, 'pnl_usd': pnl_usd, 'pred': r['pred']})

res_df = pd.DataFrame(results)

if res_df.empty:
    print('No trades triggered at threshold', THRESHOLD)
    sys.exit(0)

n = len(res_df)
wins = (res_df['pnl_usd'] > 0).sum()
avg_pnl = res_df['pnl_usd'].mean()
expectancy_per_trade = avg_pnl
win_rate = wins / n

print('Simulation summary')
print('Threshold:', THRESHOLD)
print('Trades simulated:', n)
print('Win rate:', f'{win_rate:.2%}')
print('Average PnL per trade (USD):', f'{avg_pnl:.4f}')
print('Expectancy (USD/trade):', f'{expectancy_per_trade:.4f}')
print('Total PnL (USD):', f'{res_df["pnl_usd"].sum():.4f}')

# save results
out_csv = os.path.join(BASE, 'backtests', f'quick_sim_threshold_{int(THRESHOLD*1e6)}.csv')
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
res_df.to_csv(out_csv, index=False)
print('Saved per-trade results to', out_csv)
