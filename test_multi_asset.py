import pandas as pd
import numpy as np
from HolonicTrader.holon_core import Message, Holon, Disposition
from HolonicTrader.agent_trader import TraderHolon
from agent_strategy import StrategyHolon
from agent_entropy import EntropyHolon
import config

# Update config for testing
config.ALLOWED_ASSETS = ['XRP/USDT', 'DOGE/USDT']

# Mock Observer
class MockObserver(Holon):
    def fetch_market_data(self, timeframe='1h', limit=100, symbol=None):
        print(f"[MockObserver] Fetching {symbol}...")
        
        # Generator synthetic data
        dates = pd.date_range(end=pd.Timestamp.now(), periods=limit, freq='1h')
        
        if symbol == 'XRP/USDT':
             # Flat market
             close = np.linspace(1.0, 1.05, limit)
             volume = np.random.normal(1000, 100, limit)
        else:
             # Volatile market
             close = np.linspace(0.1, 0.2, limit) + np.random.normal(0, 0.01, limit)
             volume = np.random.normal(5000, 500, limit)
             
        df = pd.DataFrame({
            'timestamp': dates,
            'open': close,
            'high': close * 1.01,
            'low': close * 0.99,
            'close': close,
            'volume': volume
        })
        
        # Returns
        df['returns'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        return df

    def get_latest_price(self, symbol=None):
        return 1.0

    def receive_message(self, sender, content):
        pass

def test_multi_asset_loop():
    print("Testing Multi-Asset Entropy Engine...")
    
    # 1. Setup Agents
    observer = MockObserver("MockObserver", Disposition(0.5, 0.5))
    entropy = EntropyHolon()
    strategy = StrategyHolon()
    
    trader = TraderHolon("TraderNexus", sub_holons={
        'observer': observer,
        'entropy': entropy,
        'strategy': strategy
    })
    
    # 2. Run Cycle
    trader.run_cycle()
    
    print("✅ Cycle Complete.")

if __name__ == "__main__":
    test_multi_asset_loop()
