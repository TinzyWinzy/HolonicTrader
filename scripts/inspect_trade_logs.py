"""
Inspect per-trade rows and find matching raw log lines in datasets (rich_features and raw_trades_snapshot).
Also produce a filtered per-trade CSV excluding rows where vol_ann == 0 (fallback trades).
"""
import os
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), '..')
PER_TRADE = os.path.join(BASE, 'backtests', 'per_trade_best_cell.csv')
RICH = os.path.join(BASE, 'datasets', 'rich_features.parquet')
RAW = os.path.join(BASE, 'datasets', 'raw_trades_snapshot.parquet')
OUT_FILTERED = os.path.join(BASE, 'backtests', 'per_trade_best_cell_filtered.csv')

if not os.path.exists(PER_TRADE):
    print('Missing per-trade CSV:', PER_TRADE)
    raise SystemExit(1)

pt = pd.read_csv(PER_TRADE)
print('Loaded per-trade rows:', len(pt))

# Normalize timestamp column to string
pt['timestamp'] = pt['timestamp'].astype(str)

# Load rich features if available
rich = None
if os.path.exists(RICH):
    rich = pd.read_parquet(RICH)
    rich['timestamp'] = rich['timestamp'].astype(str)

# Load raw trades snapshot
raw = None
if os.path.exists(RAW):
    raw = pd.read_parquet(RAW)
    raw['timestamp'] = raw['timestamp'].astype(str)

# Merge pt with rich on timestamp+symbol to get session_file/raw
merged = pt.copy()
if rich is not None:
    merged = merged.merge(rich[['timestamp','symbol','session_file','raw']], on=['timestamp','symbol'], how='left', suffixes=('','_rich'))

# Also merge raw snapshot to get original raw lines if any
if raw is not None:
    merged = merged.merge(raw[['timestamp','symbol','raw']], on=['timestamp','symbol'], how='left', suffixes=('','_raw'))

# Prefer raw_raw then raw_rich
def pick_raw(row):
    if pd.notna(row.get('raw_raw')):
        return row.get('raw_raw')
    if pd.notna(row.get('raw')):
        return row.get('raw')
    return None

# Print findings
for idx, r in merged.iterrows():
    print('\n---- Trade', idx+1, '----')
    print('timestamp:', r['timestamp'])
    print('symbol   :', r['symbol'])
    print('entry    :', r['entry'], 'exit:', r['exit'], 'pnl:', r['pnl'])
    print('vol_ann  :', r.get('vol_ann'))
    # raw from merged columns
    raw_rich = r.get('raw') if 'raw' in r else None
    raw_raw = r.get('raw_raw') if 'raw_raw' in r else None
    sess = r.get('session_file')
    if pd.notna(sess):
        print('session_file:', sess)
    if pd.notna(raw_raw):
        print('raw (snapshot):', raw_raw)
    elif pd.notna(raw_rich):
        print('raw (rich):', raw_rich)
    else:
        print('No raw log line found for this trade (might be aggregated/cleaned).')

# Create filtered CSV excluding vol_ann == 0
filtered = merged[merged['vol_ann'] > 0]
filtered.to_csv(OUT_FILTERED, index=False)
print('\nWrote filtered per-trade CSV to', OUT_FILTERED, ' (rows kept:', len(filtered), ')')
