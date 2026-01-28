
import config
import ccxt

def check_map_and_pepe():
    # 1. Check Config Map
    try:
        print("--- checking config.KRAKEN_SYMBOL_MAP ---")
        if hasattr(config, 'KRAKEN_SYMBOL_MAP'):
            m = config.KRAKEN_SYMBOL_MAP
            print(f"Found Map with {len(m)} entries.")
            print(m)
        else:
            print("config.KRAKEN_SYMBOL_MAP does NOT exist.")
    except Exception as e:
        print(f"Error accessing config: {e}")

    # 2. Search for PEPE on Kraken Futures
    print("\n--- Searching for PEPE on Kraken Futures ---")
    ex = ccxt.krakenfutures()
    ex.load_markets()
    keys = list(ex.markets.keys())
    
    pepe_mkts = [k for k in keys if 'PEPE' in k]
    print(f"Found {len(pepe_mkts)} PEPE markets:")
    for m in pepe_mkts:
        print(f" - {m} (ID: {ex.markets[m]['id']})")
        # Fetch funding for found pepe
        try:
             data = ex.fetch_funding_rate(m)
             print(f"   -> Rate: {data.get('fundingRate')}")
        except:
             pass

if __name__ == "__main__":
    check_map_and_pepe()
