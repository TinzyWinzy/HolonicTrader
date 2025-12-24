import time
import psutil
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from HolonicTrader.agent_governor import GovernorHolon
from HolonicTrader.agent_oracle import EntryOracleHolon
from HolonicTrader.agent_executor import ExecutorHolon, TradeSignal
from database_manager import DatabaseManager
import config

def benchmark_operational_latency(iterations=50):
    print("=" * 60)
    print("   HOLONIC TRADER - OPERATIONAL LATENCY BENCHMARK")
    print("=" * 60)
    
    # Setup
    db = DatabaseManager()
    governor = GovernorHolon(db_manager=db)
    oracle = EntryOracleHolon()
    executor = ExecutorHolon(db_manager=db)
    
    # Mock Market Data (BTC/USDT 60 candles)
    mock_df = pd.DataFrame({
        'timestamp': pd.date_range(end=datetime.now(), periods=60, freq='H'),
        'open': np.random.uniform(60000, 61000, 60),
        'high': np.random.uniform(61000, 62000, 60),
        'low': np.random.uniform(59000, 60000, 60),
        'close': np.random.uniform(60000, 61000, 60),
        'volume': np.random.uniform(1, 10, 60)
    })
    mock_df.set_index(pd.DatetimeIndex(mock_df['timestamp']), inplace=True)
    
    mock_bb = {'middle': 60500, 'upper': 61500, 'lower': 59500}
    
    latencies = []
    cpu_usages = []
    mem_usages = []
    
    process = psutil.Process(os.getpid())
    
    print(f"Running {iterations} iterations of the 'Sense -> Think -> Govern -> Act' cycle...")
    
    for i in range(iterations):
        start_time = time.perf_counter()
        
        # 1. Oracle Calculation (Think)
        # Note: We simulate the core logic call
        signal = oracle.analyze_for_entry("BTC/USDT", mock_df, mock_bb, 0.001, 'PREDATOR')
        
        # 2. Governor Check (Govern)
        if signal:
            is_allowed, qty, lev = governor.calc_position_size("BTC/USDT", 60500.0, 0.001, 0.001)
        else:
            # Simulate a fallback check to use governor methods anyway for measurement
            _ = governor.calculate_max_risk(100.0)
            _ = governor.calculate_kelly_size(100.0)
            
        # 3. Executor Logic (Act)
        # Mock signal for executor test
        test_signal = TradeSignal("BTC/USDT", "BUY", 0.001, 60500.0)
        _ = executor.decide_trade(test_signal, "ORDERED", 0.1)
        
        end_time = time.perf_counter()
        
        latencies.append((end_time - start_time) * 1000) # ms
        cpu_usages.append(psutil.cpu_percent())
        mem_usages.append(process.memory_info().rss / (1024 * 1024)) # MB

    # Stats
    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    mean_cpu = np.mean(cpu_usages)
    mean_mem = np.mean(mem_usages)
    
    print("\nBenchmark Results:")
    print("-" * 30)
    print(f"Mean Cycle Latency:    {mean_lat:.2f} ms")
    print(f"P95 Cycle Latency:     {p95_lat:.2f} ms")
    print(f"Mean CPU Usage:        {mean_cpu:.2f} %")
    print(f"Peak Memory Usage:     {max(mem_usages):.2f} MB")
    print("-" * 30)
    
    # Verdict
    if mean_lat < 100:
        print("Verdict: ⚡ ULTRA-FAST (High-frequency ready)")
    elif mean_lat < 500:
        print("Verdict: ✅ OPTIMAL (Standard paper trading)")
    else:
        print("Verdict: ⚠️ SLUGGISH (Review complex holon logic)")

    print("=" * 60)

if __name__ == "__main__":
    benchmark_operational_latency()
