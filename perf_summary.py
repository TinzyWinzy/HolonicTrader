"""Simple performance metrics"""
import sqlite3

conn = sqlite3.connect('holonic_trader.db')
cursor = conn.cursor()

print("=== PERFORMANCE SUMMARY ===\n")

# Total trades
cursor.execute("SELECT COUNT(*) FROM trades")
print(f"Total Trades: {cursor.fetchone()[0]}")

# Buy vs Sell
cursor.execute("SELECT direction, COUNT(*) FROM trades GROUP BY direction")
for dir, count in cursor.fetchall():
    print(f"  {dir}: {count}")

# PnL
cursor.execute("SELECT SUM(pnl) FROM trades WHERE pnl IS NOT NULL")
total_pnl = cursor.fetchone()[0]
if total_pnl:
    print(f"\nTotal PnL: ${total_pnl:.2f}")

# Win rate
cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl > 0")
wins = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl < 0")
losses = cursor.fetchone()[0]
if wins + losses > 0:
    print(f"Win Rate: {wins/(wins+losses)*100:.1f}% ({wins}W/{losses}L)")

# Ledger stats
cursor.execute("SELECT COUNT(*) FROM ledger")
print(f"\nLedger Entries: {cursor.fetchone()[0]}")

cursor.execute("SELECT action, COUNT(*) FROM ledger GROUP BY action")
for action, count in cursor.fetchall():
    print(f"  {action}: {count}")

cursor.execute("SELECT regime, COUNT(*) FROM ledger GROUP BY regime")
print("\nRegimes:")
for regime, count in cursor.fetchall():
    print(f"  {regime}: {count}")

conn.close()
