
import ccxt

def find_symbols():
    exchange = ccxt.kraken()
    try:
        markets = exchange.load_markets()
        print(f"Loaded {len(markets)} markets from Kraken.")
        
        targets = ['SUI', 'UNI', 'AAVE', 'PAXG', 'SHIB']
        
        for t in targets:
            print(f"\n--- Searching for {t} ---")
            found = [m for m in markets.keys() if t in m]
            for f in found:
                print(f"  {f}  -> ID: {markets[f]['id']}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    find_symbols()
