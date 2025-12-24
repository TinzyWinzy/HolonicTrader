
import asyncio
from HolonicTrader.agent_observer import ObserverHolon
import config

async def test_all_symbols():
    observer = ObserverHolon(exchange_id='kucoin') # Default from main_live_phase4
    print(f"Testing {len(config.ALLOWED_ASSETS)} symbols on {observer.exchange_id}...")
    
    for symbol in config.ALLOWED_ASSETS:
        try:
            print(f"Fetching {symbol}...", end=' ')
            data = observer.fetch_market_data(limit=5, symbol=symbol)
            if not data.empty:
                print(f"OK ({len(data)} candles) - Last Price: {data['close'].iloc[-1]}")
            else:
                print("FAILED (Empty DataFrame)")
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_all_symbols())
