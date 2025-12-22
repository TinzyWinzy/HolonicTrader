import time
import pandas as pd
from HolonicTrader.agent_observer import ObserverHolon

def test_latency():
    print("--- Testing Observer Latency & Caching ---")
    observer = ObserverHolon(exchange_id='kucoin', symbol='BTC/USDT')
    
    # Test for BTC/USDT (Usually has a large file)
    symbol = "BTC/USDT"
    
    # 1. First Call (Miss)
    t0 = time.time()
    df1 = observer.fetch_market_data(symbol=symbol)
    t1 = time.time()
    print(f"1. Cold Fetch (Disk): {t1-t0:.4f}s - Rows: {len(df1)}")
    
    # 2. Second Call (Hit)
    t0 = time.time()
    df2 = observer.fetch_market_data(symbol=symbol)
    t2 = time.time()
    print(f"2. Warm Fetch (Cache): {t2-t0:.4f}s - Rows: {len(df2)}")
    
    # 3. Third Call
    t0 = time.time()
    df3 = observer.fetch_market_data(symbol=symbol)
    t3 = time.time()
    print(f"3. Warm Fetch (Cache): {t3-t0:.4f}s")

    if (t2-t0) < 0.1:
        print("\n✅ CACHING SUCCESS: Latency < 0.1s")
    else:
        print("\n❌ CACHING FAILED: Latency too high")

if __name__ == "__main__":
    test_latency()
