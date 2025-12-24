import sqlite3
import pandas as pd

def analyze_win_loss(db_path='holonic_trader.db'):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT pnl FROM trades WHERE pnl IS NOT NULL", conn)
    conn.close()
    
    if df.empty:
        print("No completed trades found.")
        return
        
    wins = df[df['pnl'] > 0]['pnl']
    losses = df[df['pnl'] < 0]['pnl']
    
    total_trades = len(df)
    win_count = len(wins)
    loss_count = len(losses)
    breakeven_count = len(df[df['pnl'] == 0])
    
    win_rate = win_count / total_trades if total_trades > 0 else 0
    avg_win = wins.mean() if not wins.empty else 0
    avg_loss = losses.mean() if not losses.empty else 0
    
    # Calculate R:R Ratio (Expected value vs actual)
    rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    
    print("==================================================")
    print("   WIN/LOSS STATISTICAL ANALYSIS")
    print("==================================================")
    print(f"Total Completed Trades: {total_trades}")
    print(f"Wins: {win_count} | Losses: {loss_count} | BE: {breakeven_count}")
    print(f"Historical Win Rate:    {win_rate*100:.2f}%")
    print("-" * 50)
    print(f"Average Win:            ${avg_win:.4f}")
    print(f"Average Loss:           ${avg_loss:.4f}")
    print(f"Risk/Reward Ratio:      {rr_ratio:.2f}:1")
    print("-" * 50)
    
    # PnL Expectancy Formula: (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
    expectancy = (win_rate * avg_win) + ((loss_count/total_trades) * avg_loss)
    print(f"PnL Expectancy per Trade: ${expectancy:.4f}")
    
    if expectancy > 0:
        print("✅ The strategy is mathematically profitable long-term.")
    else:
        print("❌ The strategy has negative expectancy.")
        print("   (Average losses are disproportionately large compared to wins)")
    
    print("==================================================")

if __name__ == "__main__":
    analyze_win_loss()
