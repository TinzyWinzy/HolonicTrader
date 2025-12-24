
import sqlite3
import pandas as pd

def inspect_db():
    conn = sqlite3.connect('holonic_trader.db')
    cursor = conn.cursor()
    
    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables: {tables}")
    
    if ('trades',) in tables:
        # Get one row to see columns
        res = cursor.execute("SELECT * FROM trades LIMIT 1")
        cols = [description[0] for description in cursor.description]
        print(f"Trades columns: {cols}")
    
    conn.close()

if __name__ == "__main__":
    inspect_db()
