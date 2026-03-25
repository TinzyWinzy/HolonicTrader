import csv
import glob
from collections import defaultdict

# find trades_dump.csv
paths = glob.glob('**/trades_dump.csv', recursive=True)
if not paths:
    print('trades_dump.csv not found')
    raise SystemExit(1)
path = paths[0]

sym_counts = defaultdict(int)
sym_pnl = defaultdict(float)
worst = []

with open(path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    total = 0
    losing = 0
    for r in reader:
        total += 1
        try:
            pnl = float(r.get('pnl', '0') or 0)
            pnl_pct = float(r.get('pnl_percent', '0') or 0)
        except Exception:
            continue
        sym = r.get('symbol')
        if pnl < 0:
            losing += 1
            sym_counts[sym] += 1
            sym_pnl[sym] += pnl
            worst.append((pnl_pct, pnl, sym, r.get('id'), r.get('timestamp'), r.get('price')))

print(f'Trades total: {total}')
print(f'Losing trades: {losing} ({losing/total:.2%} if total else 0)')
print('\nTop symbols by count of losing trades:')
for s, c in sorted(sym_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
    print(f'  {s}: {c} trades, total pnl {sym_pnl[s]:.2f}')

print('\nTop 10 worst trades by pnl_percent:')
for pct, pnl, sym, tid, ts, price in sorted(worst)[:10]:
    print(f'  id={tid} symbol={sym} pnl={pnl:.2f} pct={pct} time={ts} price={price}')
