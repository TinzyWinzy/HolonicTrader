
import ccxt
import config

def list_futures():
    print("Connecting to Kraken Futures...")
    try:
        # Kraken Futures is a separate exchange class in CCXT
        exchange = ccxt.krakenfutures({
            'enableRateLimit': True
        })
        
        markets = exchange.load_markets()
        print(f"Found {len(markets)} Futures Markets.")
        
        print("\n--- BTC Futures Deep Dive ---")
        # filter for BTC
        targets = ['BTC', 'XBT']
        
        for symbol in markets:
            market = markets[symbol]
            # print(market.keys()) # debug keys if needed
            
            # Look for BTC/XBT matches
            if any(t in symbol for t in targets) and market['active']:
                # Filter for Perps (Swaps) mainly, but show futures too
                print(f"Sym: {symbol}")
                print(f" - ID: {market['id']}")
                print(f" - Type: {market['type']}")
                print(f" - Linear: {market.get('linear', '?')}")
                print(f" - Inverse: {market.get('inverse', '?')}")
                print(f" - Settle: {market.get('settle', '?')}")
                print(f" - Quote: {market.get('quote', '?')}")
                print("---")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_futures()
