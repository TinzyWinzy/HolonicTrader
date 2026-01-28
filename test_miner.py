
from HolonicTrader.agent_arbitrage import ArbitrageHolon
from HolonicTrader.agent_observer import ObserverHolon
import config
import time

def test_mining():
    print("Initialize Arbi...")
    arbi = ArbitrageHolon("TestMiner")
    
    # Mock data to simulate nugget
    # Scenario: SOL funding is positive (normal), spread < 1%
    arbi.funding_yields['SOL/USDT'] = 60.0 # 60% APY
    arbi.price_spreads['SOL/USDT'] = 0.002 # 0.2% Spread
    
    print("Testing Scan...")
    nuggets = arbi.scan_for_nuggets()
    
    if nuggets:
        print(f"SUCCESS: Found {len(nuggets)} nuggets.")
        print(nuggets[0])
    else:
        print("FAIL: No nuggets found.")

if __name__ == "__main__":
    test_mining()
