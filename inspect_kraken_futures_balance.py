
import ccxt
import config
import json

def inspect_futures_balance():
    print("Connecting to Kraken Futures...")
    try:
        # Kraken Futures
        exchange = ccxt.krakenfutures({
            'apiKey': config.KRAKEN_FUTURES_API_KEY,
            'secret': config.KRAKEN_FUTURES_PRIVATE_KEY,
            'enableRateLimit': True
        })
        
        balance = exchange.fetch_balance()
        
        print("\n--- Futures 'info' Data ---")
        if 'info' in balance:
            # Futures info might be a list or dict depending on endpoint
            print(json.dumps(balance['info'], indent=2))
        else:
            print("No 'info' field found.")

        print("\n--- Standard 'free' Data ---")
        print(json.dumps(balance.get('free', {}), indent=2))
        
        print("\n--- Standard 'total' Data ---")
        print(json.dumps(balance.get('total', {}), indent=2))
        
        print("\n--- XTZ Symbol Check ---")
        markets = exchange.load_markets()
        for s in markets:
            if 'XTZ' in s and markets[s].get('linear', False):
                print(f"Found: {s} -> {markets[s]['id']}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_futures_balance()
