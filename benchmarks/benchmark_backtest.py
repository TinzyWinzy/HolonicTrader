import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

from run_backtest import run_backtest
from database_manager import DatabaseManager
import config

# Simple Mock Status Queue for backtest integration
class MockQueue:
    def put(self, item): pass
    def get_nowait(self): raise Exception("Queue empty")

def run_comparative_backtest(symbol="XRP/USDT"):
    print("=" * 60)
    print(f"   FINANCIAL BENCHMARK: {symbol}")
    print("   Comparing 5% TP (Old) vs 3% TP (New)")
    print("=" * 60)
    
    results = []
    
    # Scenario A: Old Thresholds
    print("\nRunning Scenario A: 5% Take Profit / 3% Stop Loss...")
    config.PREDATOR_TAKE_PROFIT = 0.05
    config.SCAVENGER_STOP_LOSS = 0.03
    
    q_a = MockQueue()
    stats_a = run_backtest(q_a, symbol=symbol) 
    results.append(("5% TP / 3% SL", stats_a))
    
    # Scenario B: New Optimized Thresholds
    print("\nRunning Scenario B: 3% Take Profit / 3% Stop Loss...")
    config.PREDATOR_TAKE_PROFIT = 0.03
    config.SCAVENGER_STOP_LOSS = 0.03
    
    q_b = MockQueue()
    stats_b = run_backtest(q_b, symbol=symbol)
    results.append(("3% TP / 3% SL", stats_b))
    
    print("\nBenchmark Summary Table:")
    print("-" * 80)
    print(f"{'Strategy Configuration':<25} | {'ROI':<10} | {'Win Rate':<10} | {'PF':<10} | {'MDD':<10}")
    print("-" * 80)
    
    for label, stats in results:
        # Assuming stats is a dict with ROI, win_rate, profit_factor, max_drawdown
        if stats:
             print(f"{label:<25} | {stats.get('roi', 0):>8.2f}% | {stats.get('win_rate', 0):>8.2f}% | {stats.get('profit_factor', 0):>8.2f} | {stats.get('max_drawdown', 0):>8.2f}%")
        else:
             print(f"{label:<25} | {'ERROR':>10} | {'ERROR':>10} | {'ERROR':>10} | {'ERROR':>10}")
    
    print("-" * 80)
    print("=" * 60)

if __name__ == "__main__":
    # Note: symbol must have historical data in market_data/ symbol_1h.csv
    # Defaulting to BTC/USDT as it's likely present
    run_comparative_backtest("BTC/USDT")
