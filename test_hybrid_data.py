import pandas as pd
import config
from agent_observer import ObserverHolon

def test_hybrid_data_loading():
    print("Testing Hybrid Data Engine (Local CSV + Live CCXT)...")
    
    # Symbols to test
    symbols = ['ADA/USDT', 'BTC/USD', 'DOGE/USDT', 'SUI/USDT', 'XRP/USDT']
    
    observer = ObserverHolon(exchange_id='kucoin')
    
    for symbol in symbols:
        print(f"\n--- Processing {symbol} ---")
        
        # 1. Fetch Hybrid Data
        # We specify a small limit for live sync to be fast
        df = observer.fetch_market_data(symbol=symbol, limit=10)
        
        if not df.empty:
            print(f"✅ Success: Loaded {len(df)} candles for {symbol}")
            print(f"   History Start: {df['timestamp'].iloc[0]}")
            print(f"   Latest Candle: {df['timestamp'].iloc[-1]}")
            print(f"   Last Close: {df['close'].iloc[-1]:.4f}")
            
            # Verify returns
            if 'returns' in df.columns:
                 print(f"   Returns calculated: ✅")
            else:
                 print(f"   Returns missing: ❌")
        else:
            print(f"❌ Failed to load data for {symbol}")

if __name__ == "__main__":
    test_hybrid_data_loading()
