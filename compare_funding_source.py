
import ccxt
import json

def compare_ticker_rates():
    ex = ccxt.krakenfutures()
    ex.load_markets()
    
    symbols = ['BTC/USD:USD', 'PEPE/USD:USD']
    
    for sym in symbols:
        print(f"--- {sym} ---")
        try:
            ticker = ex.fetch_ticker(sym)
            raw_info = ticker.get('info', {})
            
            funding_rate = raw_info.get('fundingRate')
            prediction = raw_info.get('fundingRatePrediction')
            
            print(f"Ticker Funding Rate: {funding_rate}")
            print(f"Ticker Prediction: {prediction}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    compare_ticker_rates()
