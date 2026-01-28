
import ccxt

def test_majors():
    print("Connecting to Kraken Futures...")
    ex = ccxt.krakenfutures()
    ex.load_markets()
    
    # Symbols from config.KRAKEN_SYMBOL_MAP
    # 'BTC/USDT': 'BTC/USD:USD'
    # 'ETH/USDT': 'ETH/USD:USD'
    
    majors = ['BTC/USD:USD', 'ETH/USD:USD', 'SOL/USD:USD']
    
    print(f"Testing mapped majors: {majors}")
    
    for sym in majors:
        try:
            # Simulate new Observer logic
            ticker = ex.fetch_ticker(sym)
            rate = None
            if 'info' in ticker:
                if 'fundingRate' in ticker['info']:
                    rate = ticker['info']['fundingRate']
                elif 'lastFundingRate' in ticker['info']:
                    rate = ticker['info']['lastFundingRate']
            
            print(f"[{sym}] Ticker Rate: {rate}")
        except Exception as e:
            print(f"[{sym}] Error: {e}")

if __name__ == "__main__":
    test_majors()
