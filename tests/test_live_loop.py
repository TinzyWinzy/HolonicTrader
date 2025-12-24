import signal
import time
from HolonicTrader.holon_core import Disposition, Message
from HolonicTrader.agent_trader import TraderHolon
from HolonicTrader.agent_observer import ObserverHolon
from HolonicTrader.agent_entropy import EntropyHolon
from HolonicTrader.agent_strategy import StrategyHolon
import config

# Mock Observer again to avoid API keys in test
class MockLiveObserver(ObserverHolon):
    def __init__(self, name, disposition):
        # SKIP super().__init__ to avoid ccxt load
        self.name = name
        self.disposition = disposition
        self.symbol = 'BTC/USDT'
        self.exchange = None

    def fetch_market_data(self, timeframe='1h', limit=100, symbol=None):
        raise Exception("Mock Data Fetch not implemented for full loop test yet - assuming connection failure is handled.")

    def get_latest_price(self, symbol=None):
        return 50000.0

def test_live_loop_interrupt():
    print("Testing LIVE Loop Logic (running for 3 seconds)...")
    
    # Setup
    observer = MockLiveObserver("MockObs", Disposition(0.5, 0.5))
    trader = TraderHolon("TraderNexus", sub_holons={'observer': observer})
    
    # Update config to 1 asset for speed
    config.ALLOWED_ASSETS = ['BTC/USDT'] 
    
    # Run in a way that we can break it? 
    # Actually, start_live_loop is infinite. 
    # We will override run_cycle to raise KeyboardInterrupt after 2 runs to simulate user stop.
    
    original_run = trader.run_cycle
    trader.cycle_count = 0
    
    def mock_cycle():
        trader.cycle_count += 1
        print(f"  -> Cycle {trader.cycle_count} Executed.")
        original_run() # Call original
        if trader.cycle_count >= 2:
            raise KeyboardInterrupt("Simulated Stop")
            
    trader.run_cycle = mock_cycle
    
    # Start Loop with fast interval
    trader.start_live_loop(interval_seconds=1)
    
    print("✅ Live Loop Test Passed (Clean Exit).")

if __name__ == "__main__":
    test_live_loop_interrupt()
