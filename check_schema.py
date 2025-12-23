"""Check database schema"""
import sqlite3

conn = sqlite3.connect('holonic_trader.db')
cursor = conn.cursor()

# Get schema for all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [t[0] for t in cursor.fetchall()]

for table in tables:
    print(f"\n{table}:")
    cursor.execute(f"PRAGMA table_info({table});")
    columns = cursor.fetchall()
    for col in columns:
        print(f"  {col[1]} ({col[2]})")

conn.close()
