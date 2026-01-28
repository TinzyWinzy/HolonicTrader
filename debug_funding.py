
import ccxt
import time
import sys
import traceback

def check_funding():
    try:
        print("Connecting to Kraken Futures...", flush=True)
        # Initialize Kraken Futures
        ex = ccxt.krakenfutures()
        ex.load_markets()
        
        print(f"Loaded {len(ex.markets)} markets.", flush=True)
        
        # Print first 10 symbols to check format
        keys = list(ex.markets.keys())
        print(f"Sample Markets: {keys[:10]}", flush=True)
        
        # Search for any USDT pairs
        usdt_markets = [k for k in keys if 'USDT' in k]
        print(f"Found {len(usdt_markets)} USDT markets:", flush=True)
        for m in usdt_markets:
            print(f" - {m} (ID: {ex.markets[m]['id']})", flush=True)

        if usdt_markets:
             test_sym = usdt_markets[0]
             print(f"Testing with: {test_sym}", flush=True)
             try:
                 data = ex.fetch_funding_rate(test_sym)
                 rate = data.get('fundingRate')
                 print(f"[{test_sym}] Rate: {rate}", flush=True)
             except Exception as e:
                 print(f"Error fetching rate: {e}", flush=True)
        else:
             print("No USDT markets found.", flush=True)

        # Test Linear USD pair
        linear_pair = "PF_XBTUSD"
        print(f"Testing Linear Pair: {linear_pair}", flush=True)
        try:
             data = ex.fetch_funding_rate(linear_pair)
             rate = data.get('fundingRate')
             print(f"[{linear_pair}] Rate: {rate}", flush=True)
        except Exception as e:
             print(f"Error fetching linear pair: {e}", flush=True)

        # Test BTC/USDT (The one the bot uses)
        bot_pair = "BTC/USDT"
        print(f"Testing Bot Pair: {bot_pair}", flush=True)
        try:
             data = ex.fetch_funding_rate(bot_pair)
             rate = data.get('fundingRate')
             print(f"[{bot_pair}] Rate: {rate}", flush=True)
        except Exception as e:
             print(f"Error fetching bot pair: {e}", flush=True)

    except Exception as e:
        print(f"Top Level Error: {e}", flush=True)
        traceback.print_exc()

if __name__ == "__main__":
    check_funding()
