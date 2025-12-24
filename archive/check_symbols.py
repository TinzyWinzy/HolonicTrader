
import ccxt
import json

def check_exchanges():
    kraken = ccxt.kraken()
    kucoin = ccxt.kucoin()
    
    targets = ['PEPE', 'WIF', 'SHIB', 'PAXG', 'LTC', 'LINK', 'XMR', 'ALGO', 'UNI', 'AAVE']
    quotes = ['USDT', 'USD']
    
    # Load markets
    print("Loading KuCoin markets...")
    kucoin_mkts = kucoin.load_markets()
    print("Loading Kraken markets...")
    kraken_mkts = kraken.load_markets()
    
    results = {}
    
    for t in targets:
        results[t] = {
            'kucoin_usdt': f"{t}/USDT" in kucoin_mkts,
            'kraken_usdt': f"{t}/USDT" in kraken_mkts,
            'kraken_usd': f"{t}/USD" in kraken_mkts,
            'kraken_ticker_usdt': None,
            'kraken_ticker_usd': None
        }
        
        # Check for Kraken X-prefixes or other variances
        if not results[t]['kraken_usdt']:
            # Search for anything with the base
            matches = [m for m in kraken_mkts.keys() if m.startswith(t + "/")]
            if matches:
                results[t]['kraken_matches'] = matches

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    check_exchanges()
