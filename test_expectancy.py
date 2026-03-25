"""Expectancy analysis — run from HolonicTrader repo root."""
import sqlite3
import pandas as pd
import math

DB = "HolonicTrader/holonic_trader.db"

conn = sqlite3.connect(DB)
df = pd.read_sql_query("SELECT * FROM trades ORDER BY id ASC", conn)
conn.close()

exits = df[df["cost_usd"] <= 1e-9].copy()
entries = df[df["cost_usd"] > 1e-9].copy()

print(f"Total rows: {len(df)}")
print(f"Entries (cost_usd > 0): {len(entries)}")
print(f"Exits (cost_usd ~ 0):  {len(exits)}")
print()

has_pnl = df[df["pnl"].abs() > 1e-9]
print(f"Rows with non-zero PnL: {len(has_pnl)}")
ent_pnl = has_pnl[has_pnl["cost_usd"] > 1e-9]
ext_pnl = has_pnl[has_pnl["cost_usd"] <= 1e-9]
print(f"  of which entries: {len(ent_pnl)}")
print(f"  of which exits:   {len(ext_pnl)}")
print()

trades = entries[entries["pnl"].notna()].copy()
print(f"Tradeable dataset: {len(trades)} entries with PnL")
print()

winners = trades[trades["pnl"] > 0]
losers = trades[trades["pnl"] < 0]
scratches = trades[trades["pnl"].abs() < 1e-9]

total = len(trades)
n_win = len(winners)
n_loss = len(losers)
n_scratch = len(scratches)

win_rate = n_win / total * 100 if total > 0 else 0
loss_rate = n_loss / total * 100 if total > 0 else 0

avg_win = float(winners["pnl"].mean()) if not winners.empty else 0
avg_loss = float(abs(losers["pnl"].mean())) if not losers.empty else 0

win_prob = n_win / total if total > 0 else 0
loss_prob = n_loss / total if total > 0 else 0
expectancy = (win_prob * avg_win) - (loss_prob * avg_loss)

gross_profit = float(winners["pnl"].sum()) if not winners.empty else 0
gross_loss = float(abs(losers["pnl"].sum())) if not losers.empty else 0
profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

payoff = avg_win / avg_loss if avg_loss > 0 else float("inf")

print("=" * 62)
print("  EXPECTANCY REPORT - ALL TRADES")
print("=" * 62)
print(f"  Total Trades:     {total}")
print(f"  Winners:          {n_win} ({win_rate:.1f}%)")
print(f"  Losers:           {n_loss} ({loss_rate:.1f}%)")
print(f"  Scratches:        {n_scratch}")
print()
print(f"  Avg Win:          ${avg_win:.4f}")
print(f"  Avg Loss:         ${avg_loss:.4f}")
print(f"  Payoff Ratio:     {payoff:.2f}:1")
print()
print(f"  Gross Profit:     ${gross_profit:.4f}")
print(f"  Gross Loss:       ${gross_loss:.4f}")
print(f"  Net P&L:          ${gross_profit - gross_loss:.4f}")
print(f"  Profit Factor:    {profit_factor:.3f}")
print()
tag = "POSITIVE" if expectancy > 0 else "NEGATIVE"
print(f"  ** EXPECTANCY:    ${expectancy:.4f} per trade **")
print(f"  ** {tag} **")
print()

# Per-symbol
print("=" * 62)
print("  PER-SYMBOL EXPECTANCY")
print("=" * 62)
symbols = trades["symbol"].unique()
rows = []
for sym in symbols:
    st = trades[trades["symbol"] == sym]
    sw = st[st["pnl"] > 0]
    sl = st[st["pnl"] < 0]
    n = len(st)
    wr = len(sw) / n * 100 if n > 0 else 0
    aw = float(sw["pnl"].mean()) if not sw.empty else 0
    al = float(abs(sl["pnl"].mean())) if not sl.empty else 0
    wp = len(sw) / n if n > 0 else 0
    lp = len(sl) / n if n > 0 else 0
    exp = (wp * aw) - (lp * al)
    net = float(st["pnl"].sum())
    rows.append((sym, n, wr, aw, al, exp, net))

rows.sort(key=lambda x: x[5], reverse=True)
hdr = f"  {'Symbol':<14} {'N':>4} {'Win%':>6} {'AvgW':>8} {'AvgL':>8} {'Expect':>8} {'NetPnL':>8}"
print(hdr)
print(f"  {'-'*14} {'-'*4} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for sym, n, wr, aw, al, exp, net in rows:
    print(f"  {sym:<14} {n:>4} {wr:>5.1f}% ${aw:>7.4f} ${al:>7.4f} ${exp:>7.4f} ${net:>7.4f}")

# Recent 50
print()
print("=" * 62)
print("  RECENT 50 TRADES EXPECTANCY")
print("=" * 62)
recent = trades.tail(50)
rw = recent[recent["pnl"] > 0]
rl = recent[recent["pnl"] < 0]
rn = len(recent)
r_wr = len(rw) / rn * 100 if rn > 0 else 0
r_aw = float(rw["pnl"].mean()) if not rw.empty else 0
r_al = float(abs(rl["pnl"].mean())) if not rl.empty else 0
r_wp = len(rw) / rn if rn > 0 else 0
r_lp = len(rl) / rn if rn > 0 else 0
r_exp = (r_wp * r_aw) - (r_lp * r_al)
print(f"  Win Rate:     {r_wr:.1f}%")
print(f"  Avg Win:      ${r_aw:.4f}")
print(f"  Avg Loss:     ${r_al:.4f}")
print(f"  Expectancy:   ${r_exp:.4f} per trade")
print(f"  Net PnL:      ${float(recent['pnl'].sum()):.4f}")
if r_exp > expectancy:
    print("  IMPROVING (recent > all-time)")
elif r_exp < expectancy:
    print("  DECLINING (recent < all-time)")
else:
    print("  STABLE")
