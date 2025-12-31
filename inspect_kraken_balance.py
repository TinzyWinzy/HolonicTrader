
import ccxt
import config
import json

def inspect_balance():
    print("Connecting to Kraken (Spot/Margin)...")
    try:
        exchange = ccxt.kraken({
            'apiKey': config.API_KEY,
            'secret': config.API_SECRET,
            'enableRateLimit': True
        })
        
        balance = exchange.fetch_balance()
        
        # Print "info" which usually contains the raw exchange response
        # Kraken raw info typically has 'eb' (equivalent balance), 'tb' (trade balance), 'm' (margin level)
        print("\n--- Raw 'info' Data ---")
        if 'info' in balance:
            print(json.dumps(balance['info'], indent=2))
        else:
            print("No 'info' field found.")

        # Print standard fields
        print("\n--- Standard 'free' Data ---")
        print(json.dumps(balance.get('free', {}), indent=2))
        
        print("\n--- Standard 'total' Data ---")
        print(json.dumps(balance.get('total', {}), indent=2))

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_balance()
