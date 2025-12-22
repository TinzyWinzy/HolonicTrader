
import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "holonic_trader.db"

def get_performance_data():
    """
    Fetch performance metrics and return as a dictionary.
    """
    data = {
        'total_trades': 0,
        'win_rate': 0.0,
        'realized_pnl': 0.0,
        'avg_pnl': 0.0,
        'best_trade': 0.0,
        'worst_trade': 0.0,
        'portfolio_usd': 0.0,
        'held_assets': {},
        'recent_trades': []
    }
    
    conn = sqlite3.connect(DB_PATH)
    try:
        # 1. TRADES
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY id ASC", conn)
        if not df.empty:
            sells = df[df['direction'] == 'SELL'].copy()
            total_trades = len(sells)
            data['total_trades'] = total_trades
            data['realized_pnl'] = sells['pnl'].sum()
            
            if total_trades > 0:
                winning_trades = sells[sells['pnl'] > 0]
                data['win_rate'] = (len(winning_trades) / total_trades) * 100
                data['avg_pnl'] = sells['pnl'].mean()
                data['best_trade'] = sells['pnl'].max()
                data['worst_trade'] = sells['pnl'].min()
            
            # Recent Trades (Last 10)
            recent = df.tail(10)[['timestamp', 'symbol', 'direction', 'price', 'quantity', 'pnl']].to_dict(orient='records')
            data['recent_trades'] = recent

        # 2. PORTFOLIO
        port = pd.read_sql_query("SELECT * FROM portfolio", conn)
        if not port.empty:
            data['portfolio_usd'] = port.iloc[0]['balance_usd']
            try:
                import json
                data['held_assets'] = json.loads(port.iloc[0].get('held_assets', '{}'))
            except:
                pass
                
    except Exception as e:
        print("Performance Tracker Error:", e)
    finally:
        conn.close()
        
    return data

def print_performance_report():
    data = get_performance_data()
    print("==========================================")
    print("   HOLONIC TRADER - PERFORMANCE REPORT    ")
    print("==========================================")
    print(f"Total Completed Trades: {data['total_trades']}")
    print(f"Win Rate:               {data['win_rate']:.2f}%")
    print(f"Total Realized PnL:     ${data['realized_pnl']:.2f}")
    print(f"Average PnL per Trade:  ${data['avg_pnl']:.2f}")
    
    print("\n--- Recent Activity ---")
    for t in data['recent_trades']:
        print(f"{t['timestamp']} {t['symbol']} {t['direction']} {t['pnl']}")
    
    print("\n--- Portfolio ---")
    print(f"USD Balance: ${data['portfolio_usd']:.2f}")
    print("==========================================")

if __name__ == "__main__":
    print_performance_report()
