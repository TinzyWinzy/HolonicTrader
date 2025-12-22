
import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "holonic_trader.db"

DB_PATH = "holonic_trader.db"

def calculate_omega_ratio(returns: list, threshold: float = 0.0) -> float:
    """
    Calculate Omega Ratio.
    Omega(L) = Sum(Gains - L) / Sum(L - Losses)
    Where discrete returns are split above and below threshold L.
    """
    if not returns:
        return 0.0
        
    gains = [r - threshold for r in returns if r > threshold]
    losses = [threshold - r for r in returns if r < threshold]
    
    sum_gains = sum(gains)
    sum_losses = sum(losses)
    
    if sum_losses == 0:
        return 100.0 if sum_gains > 0 else 0.0 # Infinite gain ratio
        
    return sum_gains / sum_losses

def get_performance_data():
    """
    Fetch performance metrics and return as a dictionary.
    """
    data = {
        'total_trades': 0,
        'win_rate': 0.0,
        'realized_pnl': 0.0,
        'avg_pnl': 0.0,
        'omega_ratio': 0.0,
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
                
                # Calculate Omega Ratio (Threshold = 0.0)
                # We need a list of PnL values (absolute dollars or %, here using dollars from 'pnl' column)
                pnl_list = sells['pnl'].tolist()
                data['omega_ratio'] = calculate_omega_ratio(pnl_list, threshold=0.0)
            
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
    print(f"Omega Ratio (Risk):     {data['omega_ratio']:.4f}")
    
    print("\n--- Recent Activity ---")
    for t in data['recent_trades']:
        print(f"{t['timestamp']} {t['symbol']} {t['direction']} {t['pnl']}")
    
    print("\n--- Portfolio ---")
    print(f"USD Balance: ${data['portfolio_usd']:.2f}")
    print("==========================================")

if __name__ == "__main__":
    print_performance_report()
