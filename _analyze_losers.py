"""Forensic analysis of losing symbols — PAXG, XMR, BTC, XAUT."""
import sqlite3
import pandas as pd
from pathlib import Path

DB = Path("HolonicTrader/holonic_trader.db")
conn = sqlite3.connect(DB)
df = pd.read_sql_query("SELECT * FROM trades ORDER BY id ASC", conn)
conn.close()

BLEEDERS = ['PAXG/USDT', 'XMR/USDT', 'BTC/USDT', 'XAUT/USDT']
WINNERS  = ['TAO/USDT', 'BNB/USDT']
ALL = BLEEDERS + WINNERS

df['ts'] = pd.to_datetime(df['timestamp'], errors='coerce')
df['hour'] = df['ts'].dt.hour
df['date'] = df['ts'].dt.date

sub = df[df['symbol'].isin(ALL)].copy()

# === 1. Position Size Distribution ===
print("=" * 72)
print("  1. POSITION SIZE DISTRIBUTION (cost_usd)")
print("=" * 72)
for sym in ALL:
    t = sub[sub['symbol'] == sym]
    if t.empty: continue
    c = t['cost_usd']
    tag = "WINNER" if sym in WINNERS else "BLEEDER"
    print(f"\n  {sym} [{tag}] — {len(t)} trades")
    print(f"    Min: ${c.min():.2f}  Max: ${c.max():.2f}  Mean: ${c.mean():.2f}  Median: ${c.median():.2f}")

# === 2. Time-of-Day Loss Clustering ===
print("\n" + "=" * 72)
print("  2. TIME-OF-DAY ANALYSIS (Losses by Hour UTC)")
print("=" * 72)
for sym in BLEEDERS:
    t = sub[(sub['symbol'] == sym) & (sub['pnl'] < 0)]
    if t.empty: continue
    hourly = t.groupby('hour')['pnl'].agg(['sum', 'count'])
    worst_hour = hourly['sum'].idxmin()
    print(f"\n  {sym} — Worst hour: {worst_hour}:00 UTC (${hourly.loc[worst_hour, 'sum']:.2f} lost, {int(hourly.loc[worst_hour, 'count'])} trades)")
    for h, row in hourly.iterrows():
        bar = "#" * max(1, int(abs(row['sum']) / 5))
        print(f"    {h:02d}:00  ${row['sum']:>8.2f}  ({int(row['count']):>2} trades) {bar}")

# === 3. Loss Streak Analysis ===
print("\n" + "=" * 72)
print("  3. CONSECUTIVE LOSS STREAK ANALYSIS")
print("=" * 72)
for sym in ALL:
    t = sub[sub['symbol'] == sym].reset_index(drop=True)
    if t.empty: continue
    is_loss = (t['pnl'] < 0).astype(int)
    groups = (is_loss != is_loss.shift()).cumsum()
    loss_streaks = is_loss.groupby(groups).sum()
    loss_streaks = loss_streaks[loss_streaks > 0]
    max_streak = int(loss_streaks.max()) if not loss_streaks.empty else 0
    # also win streak
    is_win = (t['pnl'] > 0).astype(int)
    wgroups = (is_win != is_win.shift()).cumsum()
    win_streaks = is_win.groupby(wgroups).sum()
    win_streaks = win_streaks[win_streaks > 0]
    max_win = int(win_streaks.max()) if not win_streaks.empty else 0
    tag = "WINNER" if sym in WINNERS else "BLEEDER"
    print(f"  {sym:<14} [{tag}]  MaxLossStreak: {max_streak}  MaxWinStreak: {max_win}")

# === 4. Win/Loss Size Asymmetry ===
print("\n" + "=" * 72)
print("  4. WIN/LOSS SIZE ASYMMETRY (Payoff Ratio)")
print("=" * 72)
print(f"  {'Symbol':<14} {'WR%':>5} {'AvgWin':>9} {'AvgLoss':>9} {'Payoff':>7} {'Verdict':>20}")
print(f"  {'-'*14} {'-'*5} {'-'*9} {'-'*9} {'-'*7} {'-'*20}")
for sym in ALL:
    t = sub[sub['symbol'] == sym]
    if t.empty: continue
    w = t[t['pnl'] > 0]
    l = t[t['pnl'] < 0]
    wr = len(w) / len(t) * 100
    aw = float(w['pnl'].mean()) if not w.empty else 0
    al = float(abs(l['pnl'].mean())) if not l.empty else 0
    payoff = aw / al if al > 0 else float('inf')
    # Need payoff > (1-wr/100)/(wr/100) to be profitable
    breakeven_payoff = (1 - wr/100) / (wr/100) if wr > 0 else float('inf')
    verdict = "OK" if payoff >= breakeven_payoff else f"NEED {breakeven_payoff:.2f}x"
    print(f"  {sym:<14} {wr:>4.1f}% ${aw:>8.2f} ${al:>8.2f} {payoff:>6.2f}x {verdict:>20}")

# === 5. Trade Frequency ===
print("\n" + "=" * 72)
print("  5. TRADE FREQUENCY")
print("=" * 72)
for sym in ALL:
    t = sub[sub['symbol'] == sym]
    if t.empty: continue
    days = t['date'].nunique()
    first = t['ts'].min()
    last = t['ts'].max()
    span = (last - first).days + 1 if pd.notna(first) and pd.notna(last) else 1
    tpd = len(t) / span if span > 0 else 0
    tag = "WINNER" if sym in WINNERS else "BLEEDER"
    print(f"  {sym:<14} [{tag}]  {len(t)} trades / {span} days = {tpd:.1f}/day  (active {days} days)")

# === 6. Largest Individual Losses ===
print("\n" + "=" * 72)
print("  6. TOP 15 WORST INDIVIDUAL TRADES")
print("=" * 72)
bleeder_df = sub[sub['symbol'].isin(BLEEDERS)]
worst = bleeder_df.nsmallest(15, 'pnl')
for _, r in worst.iterrows():
    ts = r['timestamp'][:19] if isinstance(r['timestamp'], str) else 'N/A'
    print(f"  {r['symbol']:<14} PnL: ${r['pnl']:>8.2f}  Size: ${r['cost_usd']:>8.2f}  Return: {r['pnl_percent']:>6.2f}%  {ts}")

# === 7. Winners vs Losers Comparison ===
print("\n" + "=" * 72)
print("  7. STRUCTURAL COMPARISON: WINNERS vs BLEEDERS")
print("=" * 72)
print(f"\n  {'Metric':<25} {'TAO':>10} {'BNB':>10} {'PAXG':>10} {'XMR':>10} {'BTC':>10} {'XAUT':>10}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for metric_name, metric_fn in [
    ('Avg Position $', lambda t: f"${t['cost_usd'].mean():.1f}"),
    ('Max Position $', lambda t: f"${t['cost_usd'].max():.1f}"),
    ('Avg Win $', lambda t: f"${t[t['pnl']>0]['pnl'].mean():.2f}" if len(t[t['pnl']>0]) > 0 else "$0"),
    ('Avg Loss $', lambda t: f"${abs(t[t['pnl']<0]['pnl'].mean()):.2f}" if len(t[t['pnl']<0]) > 0 else "$0"),
    ('Win Rate %', lambda t: f"{len(t[t['pnl']>0])/len(t)*100:.1f}%"),
    ('Net PnL $', lambda t: f"${t['pnl'].sum():.1f}"),
    ('Trades', lambda t: f"{len(t)}"),
]:
    vals = []
    for sym in ['TAO/USDT', 'BNB/USDT', 'PAXG/USDT', 'XMR/USDT', 'BTC/USDT', 'XAUT/USDT']:
        t = sub[sub['symbol'] == sym]
        try:
            vals.append(metric_fn(t) if not t.empty else "N/A")
        except:
            vals.append("N/A")
    print(f"  {metric_name:<25} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10} {vals[4]:>10} {vals[5]:>10}")

# === 8. MFE/MAE Analysis ===
print("\n" + "=" * 72)
print("  8. MFE/MAE ANALYSIS")
print("=" * 72)
if 'mfe' in df.columns and 'mae' in df.columns:
    mfe_data = sub[sub['mfe'].notna() & (sub['mfe'] != 0)]
    if not mfe_data.empty:
        for sym in ALL:
            t = mfe_data[mfe_data['symbol'] == sym]
            if t.empty:
                print(f"  {sym}: No MFE/MAE data")
                continue
            avg_mfe = t['mfe'].mean()
            avg_mae = t['mae'].mean()
            avg_pnl = t['pnl'].mean()
            left_on_table = avg_mfe - avg_pnl
            print(f"  {sym:<14}  MFE: ${avg_mfe:.2f}  MAE: ${avg_mae:.2f}  ActualPnL: ${avg_pnl:.2f}  Left on table: ${left_on_table:.2f}")
    else:
        print("  MFE/MAE columns exist but all values are 0 or NULL")
else:
    print("  MFE/MAE columns not present in database")

# === 9. PnL Distribution (percentiles) for bleeders ===
print("\n" + "=" * 72)
print("  9. PNL DISTRIBUTION (Percentiles) FOR BLEEDERS")
print("=" * 72)
for sym in BLEEDERS:
    t = sub[sub['symbol'] == sym]['pnl']
    if t.empty: continue
    print(f"\n  {sym}:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"    P{p:02d}: ${t.quantile(p/100):.4f}")
