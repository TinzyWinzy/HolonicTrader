import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def plot_decision_flow(db_path='holonic_trader.db'):
    conn = sqlite3.connect(db_path)
    
    # 1. Load Ledger (Entropy & Decisions)
    ledger_df = pd.read_sql_query("SELECT timestamp, entropy_score, regime, action FROM ledger ORDER BY timestamp", conn)
    ledger_df['timestamp'] = pd.to_datetime(ledger_df['timestamp'])
    
    # 2. Load Trades
    trades_df = pd.read_sql_query("SELECT timestamp, symbol, direction, pnl, pnl_percent, cost_usd FROM trades ORDER BY timestamp", conn)
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    
    conn.close()
    
    if ledger_df.empty:
        print("No ledger data found.")
        return

    # Filter for recent session (last 12 hours)
    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=12)
    ledger_df['timestamp'] = pd.to_datetime(ledger_df['timestamp'], utc=True)
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], utc=True)
    
    ledger_df = ledger_df[ledger_df['timestamp'] > cutoff]
    trades_df = trades_df[trades_df['timestamp'] > cutoff]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Plot 1: Entropy & Decisions
    ax1.plot(ledger_df['timestamp'], ledger_df['entropy_score'], color='blue', alpha=0.5, label='Entropy Score')
    
    # Background coloring by regime
    regimes = ledger_df['regime'].unique()
    colors = {'ORDERED': 'green', 'TRANSITION': 'yellow', 'CHAOTIC': 'red'}
    for regime in regimes:
        mask = ledger_df['regime'] == regime
        # Fill regions (simple version)
        ax1.scatter(ledger_df.loc[mask, 'timestamp'], ledger_df.loc[mask, 'entropy_score'], 
                   color=colors.get(regime, 'gray'), s=10, label=f'Regime: {regime}')

    # Markers for BUY/SELL
    buys = trades_df[trades_df['direction'] == 'BUY']
    sells = trades_df[trades_df['direction'] == 'SELL']
    
    ax1.scatter(buys['timestamp'], [0] * len(buys), marker='^', color='lime', s=100, label='BUY Signal', zorder=5)
    ax1.scatter(sells['timestamp'], [0] * len(sells), marker='v', color='darkred', s=100, label='SELL Signal', zorder=5)

    ax1.set_title("Market Entropy & Trade Decisions (Last 12 Hours)")
    ax1.set_ylabel("Entropy Score")
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cumulative PnL
    if not trades_df.empty:
        trades_df['cum_pnl'] = trades_df['pnl'].fillna(0).cumsum()
        ax2.fill_between(trades_df['timestamp'], trades_df['cum_pnl'], color='purple', alpha=0.3, label='Cumul. PnL ($)')
        ax2.plot(trades_df['timestamp'], trades_df['cum_pnl'], color='purple', linewidth=2)
        
        # Mark PnL per trade
        for i, row in trades_df.iterrows():
            if pd.notnull(row['pnl']):
                color = 'green' if row['pnl'] > 0 else 'red'
                ax2.text(row['timestamp'], row['cum_pnl'], f"${row['pnl']:.2f}", color=color, fontsize=8)

    ax2.set_title("Cumulative PnL ($)")
    ax2.set_ylabel("PnL ($)")
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax2.grid(True, alpha=0.3)

    # Date formatting
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plot_path = 'decision_flow_analysis.png'
    plt.savefig(plot_path)
    print(f"Analysis plot saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    plot_decision_flow()
