import sqlite3
import json

def update_balance(db_path='holonic_trader.db', new_balance=100.0):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Check if portfolio exists
    c.execute('SELECT balance_usd, held_assets, position_metadata FROM portfolio LIMIT 1')
    row = c.fetchone()
    
    if row:
        print(f"Current balance: ${row[0]:.2f}")
        c.execute('UPDATE portfolio SET balance_usd = ?', (new_balance,))
        print(f"Updated balance to: ${new_balance:.2f}")
    else:
        print("No portfolio found. Creating one...")
        # Insert initial portfolio
        held_assets = json.dumps({})
        metadata = json.dumps({})
        c.execute('INSERT INTO portfolio (balance_usd, held_assets, position_metadata) VALUES (?, ?, ?)', 
                  (new_balance, held_assets, metadata))
        print(f"Created portfolio with balance: ${new_balance:.2f}")
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    update_balance()
