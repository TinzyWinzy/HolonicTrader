
import sqlite3
import json
import os

DB_PATH = "holonic_trader.db"

def verify_knowledge():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rl_experiences';")
        if not cursor.fetchone():
            print("❌ Table 'rl_experiences' does not exist.")
            return

        # Count rows
        cursor.execute("SELECT COUNT(*) FROM rl_experiences")
        count = cursor.fetchone()[0]
        print(f"✅ Found {count} experiences in 'rl_experiences'.")

        if count > 0:
            print("\n--- Last 5 Experiences ---")
            cursor.execute("SELECT * FROM rl_experiences ORDER BY id DESC LIMIT 5")
            rows = cursor.fetchall()
            for row in rows:
                # Assuming schema: id, state, action, reward, next_state, done, timestamp (or similar)
                # Let's just print the raw row first to be safe, then format if needed.
                # Actually, I set the schema earlier. 
                # id INTEGER PRIMARY KEY AUTOINCREMENT, state TEXT, action_idx INTEGER, reward REAL, next_state TEXT, done INTEGER, timestamp DATETIME
                
                EXP_ID = row[0]
                STATE = row[1]
                ACTION = row[2]
                REWARD = row[3]
                NEXT_STATE = row[4]
                DONE = row[5]
                TIME = row[6]
                
                print(f"[{TIME}] ID: {EXP_ID} | Action: {ACTION} | Reward: {REWARD:.4f} | Done: {DONE}")
                # print(f"  State: {STATE}") # Optional: too verbose?
        else:
            print("⚠️ logic seems correct, but no experiences have been saved yet. (Has a trade closed?)")

    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    verify_knowledge()
