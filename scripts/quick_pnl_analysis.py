#!/usr/bin/env python3
"""Quick PnL analysis of trades_dump.csv"""
import csv
from collections import defaultdict
from datetime import datetime

with open('trades_dump.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

print(f"Total trades: {len(rows)}")

total_pnl = sum(float(r.get('pnl', '0') or 0) for r in rows)
wins = [r for r in rows if float(r.get('pnl', '0') or 0) > 0]
losses = [r for r in rows if float(r.get('pnl', '0') or 0) < 0]
flat = [r for r in rows if float(r.get('pnl', '0') or 0) == 0]

print(f"Wins: {len(wins)} | Losses: {len(losses)} | Flat: {len(flat)}")
print(f"Win rate: {len(wins)/len(rows)*100:.1f}%")
print(f"Total PnL: ${total_pnl:.2f}")

win_pnl = sum(float(r.get('pnl', '0') or 0) for r in wins)
loss_pnl = sum(float(r.get('pnl', '0') or 0) for r in losses)
avg_win = win_pnl / len(wins) if wins else 0
avg_loss = loss_pnl / len(losses) if losses else 0
print(f"Gross Win: ${win_pnl:.2f} | Gross Loss: ${loss_pnl:.2f}")
print(f"Avg Win: ${avg_win:.4f} | Avg Loss: ${avg_loss:.4f}")
if loss_pnl:
    print(f"Profit factor: {abs(win_pnl/loss_pnl):.3f}")
    expectancy = (len(wins)/len(rows)) * avg_win + (len(losses)/len(rows)) * avg_loss
    print(f"Expectancy per trade: ${expectancy:.4f}")

# By direction
print("\n--- By Direction ---")
by_dir = defaultdict(lambda: {'count': 0, 'pnl': 0.0, 'wins': 0})
for r in rows:
    d = r.get('direction', '?')
    pnl = float(r.get('pnl', '0') or 0)
    by_dir[d]['count'] += 1
    by_dir[d]['pnl'] += pnl
    if pnl > 0:
        by_dir[d]['wins'] += 1
for d, v in sorted(by_dir.items()):
    wr = v['wins'] / v['count'] * 100 if v['count'] else 0
    print(f"  {d}: {v['count']} trades, PnL ${v['pnl']:.2f}, WR {wr:.1f}%")

# By symbol
print("\n--- Net PnL by Symbol ---")
sym_pnl = defaultdict(float)
sym_cnt = defaultdict(int)
sym_wins = defaultdict(int)
for r in rows:
    sym = r.get('symbol', '?')
    pnl = float(r.get('pnl', '0') or 0)
    sym_pnl[sym] += pnl
    sym_cnt[sym] += 1
    if pnl > 0:
        sym_wins[sym] += 1
for sym, pnl in sorted(sym_pnl.items(), key=lambda x: x[1]):
    wr = sym_wins[sym] / sym_cnt[sym] * 100 if sym_cnt[sym] else 0
    print(f"  {sym}: ${pnl:.2f} ({sym_cnt[sym]} trades, WR {wr:.0f}%)")

# Date range
dates = [r.get('timestamp', '')[:10] for r in rows if r.get('timestamp')]
if dates:
    print(f"\nDate range: {min(dates)} to {max(dates)}")

# Recent trades
print("\n--- Last 15 Trades ---")
for r in rows[-15:]:
    ts = r.get('timestamp', '?')[:19]
    sym = r.get('symbol', '?')
    side = r.get('direction', '?')
    pnl = float(r.get('pnl', 0) or 0)
    pnl_pct = float(r.get('pnl_percent', 0) or 0)
    print(f"  {ts} {sym:12s} {side:5s} pnl=${pnl:+.4f} ({pnl_pct:+.1f}%)")

# Weekly breakdown
print("\n--- Weekly PnL ---")
weekly = defaultdict(lambda: {'pnl': 0.0, 'count': 0, 'wins': 0})
for r in rows:
    ts = r.get('timestamp', '')
    if not ts:
        continue
    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        week = dt.strftime('%Y-W%W')
    except Exception:
        continue
    pnl = float(r.get('pnl', '0') or 0)
    weekly[week]['pnl'] += pnl
    weekly[week]['count'] += 1
    if pnl > 0:
        weekly[week]['wins'] += 1
for week in sorted(weekly):
    w = weekly[week]
    wr = w['wins'] / w['count'] * 100 if w['count'] else 0
    print(f"  {week}: ${w['pnl']:+.2f} ({w['count']} trades, WR {wr:.0f}%)")
