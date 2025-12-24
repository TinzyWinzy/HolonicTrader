
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

from run_backtest import run_backtest
import config

# Simple Mock Status Queue for backtest integration
class MockQueue:
    def put(self, item): pass
    def get_nowait(self): raise Exception("Queue empty")

def run_system_wide_benchmark():
    print("=" * 60)
    print("   HOLONIC TRADER - SYSTEM-WIDE PERFORMANCE BENCHMARK")
    print(f"   Assets: {len(config.ALLOWED_ASSETS)} | Date: {datetime.now().date()}")
    print("=" * 60)
    
    results = []
    
    for symbol in config.ALLOWED_ASSETS:
        # Check if historical data exists
        csv_name = symbol.replace('/', '').replace(':', '') + "_1h.csv"
        # Special case for BTC/USST -> BTCUSD mapping in downloader
        if symbol == 'BTC/USDT': csv_name = 'BTCUSD_1h.csv'
        
        filepath = os.path.join('market_data', csv_name)
        
        if not os.path.exists(filepath):
            print(f"⚠️ Skipping {symbol} (No historical CSV found at {filepath})")
            continue
            
        print(f"📈 Benchmarking {symbol}...")
        
        q = MockQueue()
        try:
            stats = run_backtest(q, symbol=symbol)
            if stats:
                results.append(stats)
        except Exception as e:
            print(f"❌ Error benchmarking {symbol}: {e}")
            
    if not results:
        print("❌ No benchmark results generated.")
        return

    # Aggregate Results
    df_results = pd.DataFrame(results)
    
    summary = {
        'total_assets': len(results),
        'avg_roi': df_results['roi'].mean(),
        'avg_win_rate': df_results['win_rate'].mean(),
        'total_trades': df_results['total_trades'].sum(),
        'profitable_assets': len(df_results[df_results['total_pnl'] > 0]),
        'total_combined_pnl': df_results['total_pnl'].sum()
    }
    
    print("\n" + "=" * 60)
    print("   BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total Assets Tested:   {summary['total_assets']}")
    print(f"Total Combined Trades: {summary['total_trades']}")
    print(f"Average ROI per Asset: {summary['avg_roi']:.2f}%")
    print(f"Average Win Rate:      {summary['avg_win_rate']:.2f}%")
    print(f"Profitable Assets:     {summary['profitable_assets']}/{summary['total_assets']}")
    print(f"Total Combined PnL:    ${summary['total_combined_pnl']:.2f}")
    print("=" * 60)
    
    # Save results to file
    with open('benchmark_final_results.txt', 'w') as f:
        f.write("HOLONIC TRADER SYSTEM BENCHMARK\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        f.write(f"Summary:\n{pd.Series(summary).to_string()}\n\n")
        f.write("Detail Table:\n")
        f.write(df_results[['symbol', 'roi', 'win_rate', 'total_trades', 'total_pnl']].to_string())
        
    print(f"\n✅ Benchmark complete. Results saved to benchmark_final_results.txt")

if __name__ == "__main__":
    run_system_wide_benchmark()
