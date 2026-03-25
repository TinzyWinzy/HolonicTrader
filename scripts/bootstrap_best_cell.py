"""
Bootstrap confidence intervals for the selected best cell trades.
Reads `backtests/per_trade_best_cell_filtered.csv` and writes `backtests/bootstrap_best_cell.csv`.
"""
import os, sys, numpy as np, pandas as pd

BASE = os.path.join(os.path.dirname(__file__), '..')
IN = os.path.join(BASE, 'backtests', 'per_trade_best_cell_filtered.csv')
OUT = os.path.join(BASE, 'backtests', 'bootstrap_best_cell.csv')
os.makedirs(os.path.dirname(OUT), exist_ok=True)

if not os.path.exists(IN):
    print('Missing input:', IN); raise SystemExit(1)

df = pd.read_csv(IN)
if 'pnl' not in df.columns:
    print('No pnl column in', IN); raise SystemExit(1)

pnls = df['pnl'].astype(float).values
n = len(pnls)
print(f'Sample size: {n}')

R = 10000
rng = np.random.default_rng(12345)
means = np.empty(R)
winrates = np.empty(R)
for i in range(R):
    sample = rng.choice(pnls, size=n, replace=True)
    means[i] = sample.mean()
    winrates[i] = (sample > 0).mean()

def ci(arr, alpha=0.05):
    lo = np.percentile(arr, 100*(alpha/2))
    hi = np.percentile(arr, 100*(1-alpha/2))
    return lo, hi

mean_est = pnls.mean()
mean_lo, mean_hi = ci(means)
wr_est = (pnls > 0).mean()
wr_lo, wr_hi = ci(winrates)

out = pd.DataFrame([
    {'metric':'mean_pnl_per_trade','estimate':mean_est,'ci_lo':mean_lo,'ci_hi':mean_hi,'n':n},
    {'metric':'win_rate','estimate':wr_est,'ci_lo':wr_lo,'ci_hi':wr_hi,'n':n},
])

out.to_csv(OUT, index=False)
print('Wrote', OUT)
print(out.to_string(index=False))
