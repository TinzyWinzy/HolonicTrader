"""
Performance Analysis Script for HolonicTrader V2
Analyzes all trading activity from the database with Phase 12 metrics.
"""

import sqlite3
from datetime import datetime, timedelta

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
        cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl = 0")
        breakeven = cursor.fetchone()[0]
        
        if wins + losses > 0:
            win_rate = (wins / (wins + losses)) * 100
            print(f"\nWin Rate: {win_rate:.1f}% ({wins} wins / {losses} losses / {breakeven} BE)")
            
            # Average win vs average loss
            cursor.execute("SELECT AVG(pnl) FROM trades WHERE pnl > 0")
            avg_win = cursor.fetchone()[0] or 0
            cursor.execute("SELECT AVG(pnl) FROM trades WHERE pnl < 0")
            avg_loss = cursor.fetchone()[0] or 0
            
            if avg_loss != 0:
                risk_reward = abs(avg_win / avg_loss)
                print(f"Average Win: ${avg_win:.2f}")
                print(f"Average Loss: ${avg_loss:.2f}")
                print(f"Risk/Reward Ratio: {risk_reward:.2f}:1")
    
    # Portfolio snapshot
    cursor.execute("SELECT balance_usd, updated_at FROM portfolio LIMIT 1")
    portfolio = cursor.fetchone()
    
    if portfolio:
        balance, updated = portfolio
        print(f"\nCurrent Portfolio Balance: ${balance:,.2f}")
        print(f"Last Updated: {updated}")
        
        # Calculate ROI
        initial_capital = 10.0  # From config
        roi = ((balance - initial_capital) / initial_capital) * 100
        print(f"ROI: {roi:+.2f}% (from ${initial_capital})")
    
    print()
    
    # =========================================================================
    # 3. PHASE 12 RISK MANAGEMENT METRICS
    # =========================================================================
    print("=" * 70)
    print("PHASE 12: INSTITUTIONAL RISK MANAGEMENT")
    print("=" * 70)
    
    # Win rate for Kelly criterion
    if wins + losses > 0:
        print(f"Kelly Criterion Input:")
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  Risk/Reward: {risk_reward:.2f}:1" if 'risk_reward' in locals() else "  Risk/Reward: N/A")
        
        # Calculate Kelly fraction
        if 'risk_reward' in locals() and risk_reward > 0:
            p = win_rate / 100
            b = risk_reward
            kelly_fraction = (p * (b + 1) - 1) / b
            half_kelly = kelly_fraction * 0.5
            
            print(f"  Full Kelly: {kelly_fraction*100:.1f}%")
            print(f"  Half Kelly: {half_kelly*100:.1f}% (recommended)")
            
            if kelly_fraction <= 0:
                print(f"  ⚠️  Negative expectancy - system should not trade!")
            elif half_kelly > 0.25:
                print(f"  ⚠️  Kelly suggests >25% allocation - clamped for safety")
    
    # Minimax protection status
    if portfolio:
        balance = portfolio[0]
        principal = 10.0
        house_money = max(0, balance - principal)
        max_risk_1pct = balance * 0.01
        max_risk = min(house_money, max_risk_1pct)
        
        print(f"\nMinimax Constraint:")
        print(f"  Principal: ${principal:.2f} (protected)")
        print(f"  House Money: ${house_money:.2f}")
        print(f"  Max Risk (1%): ${max_risk:.2f}")
        
        if balance <= principal:
            print(f"  ⚠️  At principal - no risk allowed!")
    
    print()
    
    # =========================================================================
    # 4. LEDGER ANALYSIS (Entropy-based decisions)
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
            print(f"  Thresholds: ORDERED<0.67, TRANSITION<0.80, CHAOTIC>0.80")
    
    print()
    
    # =========================================================================
    # 5. TIME-BASED ANALYSIS
    # =========================================================================
    print("=" * 70)
    print("TIME-BASED PERFORMANCE")
    print("=" * 70)
    
    # Last 24 hours
    cursor.execute("""
        SELECT COUNT(*), SUM(pnl) 
        FROM trades 
        WHERE timestamp > datetime('now', '-1 day') AND pnl IS NOT NULL
    """)
    trades_24h, pnl_24h = cursor.fetchone()
    if trades_24h and trades_24h > 0:
        print(f"Last 24 Hours:")
        print(f"  Trades: {trades_24h}")
        print(f"  PnL: ${pnl_24h:.2f}" if pnl_24h else "  PnL: $0.00")
    
    # Last 7 days
    cursor.execute("""
        SELECT COUNT(*), SUM(pnl) 
        FROM trades 
        WHERE timestamp > datetime('now', '-7 days') AND pnl IS NOT NULL
    """)
    trades_7d, pnl_7d = cursor.fetchone()
    if trades_7d and trades_7d > 0:
        print(f"\nLast 7 Days:")
        print(f"  Trades: {trades_7d}")
        print(f"  PnL: ${pnl_7d:.2f}" if pnl_7d else "  PnL: $0.00")
    
    print()
    
    # =========================================================================
    # 6. RECENT ACTIVITY
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
