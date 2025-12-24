import sys
import pandas as pd
import numpy as np
from agent_observer import ObserverHolon

def test_observer():
    print("Testing ObserverHolon...")
    
    try:
        # Instantiate
        observer = ObserverHolon(exchange_id='kucoin', symbol='BTC/USDT')
        print(f"✅ Instantiated ObserverHolon: {observer.name}")
        
        # Test get_latest_price
        price = observer.get_latest_price()
        if isinstance(price, float) and price > 0:
            print(f"✅ get_latest_price returned valid price: {price}")
        else:
            print(f"❌ get_latest_price failed: {price}")
            
        # Test fetch_market_data
        print("Fetching market data...")
        df = observer.fetch_market_data(limit=10)
        
        # Check columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'returns']
        if all(col in df.columns for col in required_cols):
             print("✅ DataFrame has all required columns.")
        else:
             print(f"❌ Missing columns. Found: {df.columns}")
             return

        # Check NaNs
        if df.isnull().values.any():
            print("❌ DataFrame contains NaNs (should have been dropped).")
        else:
            print("✅ No NaNs found.")
            
        # Verify Log Returns manually for the last row
        # We need at least 2 rows to check calculation relative to previous
        if len(df) >= 2:
            last_close = df.iloc[-1]['close']
            prev_close = df.iloc[-2]['close']
            calc_return = np.log(last_close / prev_close)
            df_return = df.iloc[-1]['returns']
            
            if np.isclose(calc_return, df_return):
                print(f"✅ Log Return calculation verified: {calc_return} vs {df_return}")
            else:
                print(f"❌ Log Return calculation mismatch: {calc_return} vs {df_return}")
        else:
            print("⚠️ Not enough data to verify calculation manually.")
            
        print("ObserverHolon verification complete.")

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_observer()
