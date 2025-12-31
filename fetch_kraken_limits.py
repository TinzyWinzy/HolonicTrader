
import ccxt
import time

def fetch_limits():
    print("Connecting to Kraken to fetch limits...")
    exchange = ccxt.kraken()
    exchange.load_markets()
    
    sys_pairs = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 'DOGE/USDT',
        'BNB/USDT', 'LTC/USDT', 'LINK/USDT', 'SUI/USDT', 'UNI/USDT', 'AAVE/USDT',
        'SHIB/USDT', 'PAXG/USDT', 'XMR/USDT'
    ]
    
    print(f"{'PAIR':<12} | {'MIN AMOUNT':<12} | {'MIN COST ($)':<12} | {'PRICE ($)':<12}")
    print("-" * 60)
    
    for symbol in sys_pairs:
        try:
            # Kraken sometimes uses different symbols, ccxt handles mapping but let's be safe
            market = exchange.market(symbol)
            ticker = exchange.fetch_ticker(symbol)
            price = ticker['last']
            
            min_amount = market['limits']['amount']['min']
            min_cost = market['limits']['cost']['min'] if market['limits']['cost']['min'] else 0.0
            
            # Calculate implied min cost if min_amount is the constraint
            implied_cost = min_amount * price
            
            # Effective minimum cost/order value
            effective_min = max(min_cost, implied_cost)
            
            print(f"{symbol:<12} | {min_amount:<12.6f} | {effective_min:<12.2f} | {price:<12.2f}")
            
        except Exception as e:
            print(f"{symbol:<12} | ERROR: {e}")

if __name__ == "__main__":
    fetch_limits()
