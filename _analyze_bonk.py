"""BONK/USDT deep dive analysis."""
import sqlite3, pandas as pd

conn = sqlite3.connect("HolonicTrader/holonic_trader.db")
df = pd.read_sql_query("SELECT * FROM trades WHERE symbol='BONK/USDT' ORDER BY id ASC", conn)
conn.close()

df['ts'] = pd.to_datetime(df['timestamp'], errors='coerce')
df['date'] = df['ts'].dt.date
df['hour'] = df['ts'].dt.hour

n = len(df)
w = df[df['pnl'] > 0]
l = df[df['pnl'] < 0]
s = df[df['pnl'].abs() < 1e-9]
wr = len(w)/n*100 if n else 0
aw = float(w['pnl'].mean()) if not w.empty else 0
al = float(abs(l['pnl'].mean())) if not l.empty else 0
wp = len(w)/n if n else 0
lp = len(l)/n if n else 0
exp = (wp*aw)-(lp*al)
gross_w = float(w['pnl'].sum()) if not w.empty else 0
gross_l = float(abs(l['pnl'].sum())) if not l.empty else 0
pf = gross_w / gross_l if gross_l > 0 else float('inf')
payoff = aw/al if al > 0 else float('inf')
be_payoff = (1-wr/100)/(wr/100) if wr > 0 else float('inf')

print("=" * 62)
print("  BONK/USDT DEEP DIVE")
print("=" * 62)
print(f"  Trades: {n}  |  Winners: {len(w)}  |  Losers: {len(l)}  |  Scratches: {len(s)}")
print(f"  Win Rate: {wr:.1f}%")
print(f"  Avg Win: ${aw:.4f}  |  Avg Loss: ${al:.4f}")
v = "OK" if payoff >= be_payoff else "SHORT"
print(f"  Payoff Ratio: {payoff:.2f}x  (breakeven needs {be_payoff:.2f}x)  {v}")
print(f"  Profit Factor: {pf:.3f}")
print(f"  Expectancy: ${exp:.4f}/trade")
print(f"  Net PnL: ${df['pnl'].sum():.4f}")
print(f"  Gross Win: ${gross_w:.4f}  |  Gross Loss: ${gross_l:.4f}")
print()

c = df['cost_usd']
print("  Position Sizing:")
print(f"    Min: ${c.min():.2f}  Max: ${c.max():.2f}  Mean: ${c.mean():.2f}  Median: ${c.median():.2f}")
print()

# Streaks
is_loss = (df['pnl'] < 0).astype(int)
g = (is_loss != is_loss.shift()).cumsum()
ls = is_loss.groupby(g).sum()
ls = ls[ls > 0]
max_ls = int(ls.max()) if not ls.empty else 0
is_win = (df['pnl'] > 0).astype(int)
wg = (is_win != is_win.shift()).cumsum()
ws = is_win.groupby(wg).sum()
ws = ws[ws > 0]
max_ws = int(ws.max()) if not ws.empty else 0
print(f"  Max Win Streak: {max_ws}  |  Max Loss Streak: {max_ls}")

days_span = (df['ts'].max() - df['ts'].min()).days + 1
print(f"  Active: {df['date'].nunique()} days  |  Span: {days_span} days  |  {n/days_span:.1f} trades/day")
print()

print("  PnL by Hour UTC:")
hourly = df.groupby('hour')['pnl'].agg(['sum', 'count', 'mean'])
for h, row in hourly.iterrows():
    if row['sum'] > 0:
        bar = "+" * max(1, int(abs(row['sum']) * 100))
    else:
        bar = "-" * max(1, int(abs(row['sum']) * 100))
    print(f"    {h:02d}:00  ${row['sum']:>8.4f}  ({int(row['count']):>2})  {bar}")
print()

print("  PnL by Date:")
daily = df.groupby('date')['pnl'].agg(['sum', 'count'])
cum = 0
for d, row in daily.iterrows():
    cum += row['sum']
    print(f"    {d}  ${row['sum']:>8.4f}  ({int(row['count']):>2} trades)  Cum: ${cum:.4f}")
print()

print("  All Trades:")
for _, r in df.iterrows():
    ts = str(r['timestamp'])[:19]
    print(f"    {ts}  PnL: ${r['pnl']:>8.4f}  Size: ${r['cost_usd']:>7.2f}  Ret: {r['pnl_percent']:>6.2f}%  Dir: {r['direction']}")
