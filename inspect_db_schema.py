import sqlite3
import pandas as pd

conn = sqlite3.connect('holonic_trader.db')
cursor = conn.cursor()

# Get columns
cursor.execute("PRAGMA table_info(trades)")
columns = cursor.fetchall()
print("Columns:")
for col in columns:
    print(col)

# Peek at data
print("\nFirst 5 rows:")
df = pd.read_sql_query("SELECT * FROM trades LIMIT 5", conn)
print(df)

conn.close()
