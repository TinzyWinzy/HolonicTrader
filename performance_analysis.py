"""
Performance Analysis Script for HolonicTrader V2
Analyzes all trading activity from the database.
"""

import sqlite3
from datetime import datetime

def analyze_performance():
    """Comprehensive performance analysis from database."""
    
    print("=" * 70)
    print("HOLONICTRADER V2 - PERFORMANCE ANALYSIS")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    
    conn = sqlite3.connect('holonic_trader.db')
    cursor = conn.cursor()
    
    # =========================================================================
    # 1. TRADE STATISTICS
    # =========================================================================
    print("=" * 70)
    print("TRADE STATISTICS")
    print("=" * 70)
    
    cursor.execute("SELECT COUNT(*) FROM trades")
    total_trades = cursor.fetchone()[0]
    print(f"Total Trades Executed: {total_trades}")
    
    if total_trades == 0:
        print("No trades found in database.")
        conn.close()
        return
    
    # Count by direction (BUY/SELL)
    cursor.execute("SELECT direction, COUNT(*) FROM trades GROUP BY direction")
    directions = cursor.fetchall()
    if directions:
        print("\nTrades by Direction:")
        for direction, count in directions:
            pct = (count / total_trades) * 100
            print(f"  {direction}: {count} ({pct:.1f}%)")
    
    # Count by symbol
    cursor.execute("SELECT symbol, COUNT(*) FROM trades GROUP BY symbol ORDER BY COUNT(*) DESC")
    symbols = cursor.fetchall()
    if symbols:
        print(f"\nAssets Traded: {len(symbols)}")
        print("\nTop 10 Most Traded:")
        for symbol, count in symbols[:10]:
            print(f"  {symbol}: {count} trades")
    
    print()
    
    # =========================================================================
    # 2. PNL ANALYSIS
    # =========================================================================
    print("=" * 70)
    print("PROFIT & LOSS ANALYSIS")
    print("=" * 70)
    
    # Aggregate PnL from trades
    cursor.execute("SELECT SUM(pnl), AVG(pnl_percent), COUNT(*) FROM trades WHERE pnl IS NOT NULL")
    total_pnl, avg_pnl_pct, pnl_count = cursor.fetchone()
    
    if total_pnl is not None:
        print(f"Total Realized PnL: ${total_pnl:,.2f}")
        print(f"Average PnL per Trade: {avg_pnl_pct:.2f}%")
        print(f"Trades with PnL Data: {pnl_count}")
        
        # Win/Loss ratio
        cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl > 0")
        wins = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl < 0")
        losses = cursor.fetchone()[0]
        
        if wins + losses > 0:
            win_rate = (wins / (wins + losses)) * 100
            print(f"\nWin Rate: {win_rate:.1f}% ({wins} wins / {losses} losses)")
    
    # Portfolio snapshot
    cursor.execute("SELECT balance_usd, updated_at FROM portfolio LIMIT 1")
    portfolio = cursor.fetchone()
    
    if portfolio:
        balance, updated = portfolio
        print(f"\nCurrent Portfolio Balance: ${balance:,.2f}")
        print(f"Last Updated: {updated}")
    
    print()
    
    # =========================================================================
    # 3. LEDGER ANALYSIS (Entropy-based decisions)
    # =========================================================================
    print("=" * 70)
    print("RISK MANAGEMENT (Entropy Ledger)")
    print("=" * 70)
    
    cursor.execute("SELECT COUNT(*) FROM ledger")
    total_decisions = cursor.fetchone()[0]
    print(f"Total Decisions Logged: {total_decisions}")
    
    if total_decisions > 0:
        # Action distribution
        cursor.execute("SELECT action, COUNT(*) FROM ledger GROUP BY action")
        actions = cursor.fetchall()
        if actions:
            print("\nDecision Distribution:")
            for action, count in actions:
                pct = (count / total_decisions) * 100
                print(f"  {action}: {count} ({pct:.1f}%)")
        
        # Regime distribution
        cursor.execute("SELECT regime, COUNT(*) FROM ledger GROUP BY regime")
        regimes = cursor.fetchall()
        if regimes:
            print("\nMarket Regime Distribution:")
            for regime, count in regimes:
                pct = (count / total_decisions) * 100
                print(f"  {regime}: {count} ({pct:.1f}%)")
        
        # Entropy statistics
        cursor.execute("SELECT AVG(entropy_score), MIN(entropy_score), MAX(entropy_score) FROM ledger")
        avg_ent, min_ent, max_ent = cursor.fetchone()
        if avg_ent:
            print(f"\nEntropy Statistics:")
            print(f"  Average: {avg_ent:.4f}")
            print(f"  Range: {min_ent:.4f} - {max_ent:.4f}")
    
    print()
    
    # =========================================================================
    # 4. RECENT ACTIVITY
    # =========================================================================
    print("=" * 70)
    print("RECENT ACTIVITY (Last 10 Trades)")
    print("=" * 70)
    
    cursor.execute("SELECT symbol, direction, quantity, price, pnl, timestamp FROM trades ORDER BY timestamp DESC LIMIT 10")
    recent_trades = cursor.fetchall()
    
    if recent_trades:
        for symbol, direction, quantity, price, pnl, timestamp in recent_trades:
            pnl_str = f"PnL: ${pnl:.2f}" if pnl else "PnL: N/A"
            print(f"  [{timestamp}] {symbol} - {direction} {quantity:.4f} @ ${price:.2f} | {pnl_str}")
    
    print()
    print("=" * 70)
    print("END OF REPORT")
    print("=" * 70)
    
    conn.close()

if __name__ == "__main__":
    analyze_performance()
