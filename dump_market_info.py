
import ccxt
import json

def dump_market():
    ex = ccxt.krakenfutures()
    ex.load_markets()
    
    sym = 'BTC/USD:USD' # The one that failed
    print(f"Inspecting {sym}...")
    
    if sym in ex.markets:
        m = ex.markets[sym]
        # Print keys and values cleanly
        print(json.dumps(m, indent=2, default=str))
        
        # Also try to fetch ticker and see what's in there
        try:
            ticker = ex.fetch_ticker(sym)
            print("\n--- Ticker Data ---")
            print(json.dumps(ticker, indent=2, default=str))
        except Exception as e:
            print(f"Ticker fetch failed: {e}")
            
    else:
        print(f"Symbol {sym} not found in markets.")

if __name__ == "__main__":
    dump_market()
