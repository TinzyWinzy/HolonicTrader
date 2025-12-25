import sqlite3
import datetime

conn = sqlite3.connect('holonic_trader.db')
cursor = conn.cursor()

print("--- RECENT TRADES ---")
try:
    cursor.execute("SELECT timestamp, symbol, quantity, cost_usd, pnl FROM trades ORDER BY id DESC LIMIT 5")
    rows = cursor.fetchall()
    for row in rows:
        print(row)
except Exception as e:
    print(f"Error reading trades: {e}")

print("\n--- PORTFOLIO SUMMARY ---")
try:
    cursor.execute("SELECT balance_usd, held_assets FROM portfolio")
    row = cursor.fetchone()
    if row:
        print(f"Balance: ${row[0]:,.2f}")
        print(f"Assets: {row[1]}")
except Exception as e:
    print(f"Error reading portfolio: {e}")

conn.close()
