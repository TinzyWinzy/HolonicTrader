
import pandas as pd
import numpy as np
from agent_strategy import StrategyHolon
from agent_executor import ExecutorHolon, TradeSignal
from HolonicTrader.holon_core import Disposition
import config

def test_warp_speed_config():
    print("========================================")
    print("🚀 TESTING WARP SPEED CONFIGURATION")
    print("========================================")
    
    # 1. Verify Config
    print("\n[1] Checking Config...")
    print(f"   SCAVENGER_LEVERAGE: {config.SCAVENGER_LEVERAGE} (Expected: 20)")
    print(f"   PREDATOR_LEVERAGE: {config.PREDATOR_LEVERAGE} (Expected: 50)")
    print(f"   SCAVENGER_MAX_MARGIN: {config.SCAVENGER_MAX_MARGIN} (Expected: 8.0)")
    print(f"   KILL_ZONES: {config.KILL_ZONES} (Expected: [])")
    
    if config.SCAVENGER_LEVERAGE < 20 or config.SCAVENGER_MAX_MARGIN < 5.0:
        print("❌ Config NOT updated for Warp Speed.")
        return
    else:
        print("✅ Config looks Aggressive.")

    # 2. Verify Strategy Sensitivity
    print("\n[2] Testing Strategy Sensitivity (RSI=40, Price=1% above LowerBand)...")
    strategy = StrategyHolon()
    
    # Mock Data: Price falling but not touching lower band yet
    # Lower Band at 99.0, Price at 99.5 (0.5% above -> Should trigger with new tolerance)
    bb = {'upper': 110.0, 'middle': 105.0, 'lower': 99.0}
    current_price = 99.8 # 0.8% above lower band
    
    # Mock RSI
    # Should trigger because RSI 40 < 45 (New Threshold)
    # Old logic required < 30
    rsi = 40.0
    
    # Force mock internal state if needed or just instantiate
    # We need to mock 'calculate_rsi' or just trust the logic?
    # Actually we can just call 'check_scavenger_entry' directly for unit testing logic
    is_entry = strategy.check_scavenger_entry(current_price, bb, rsi)
    
    if is_entry:
        print(f"✅ Strategy Triggered! Price {current_price} (BB_Low {bb['lower']}), RSI {rsi}")
    else:
        print(f"❌ Strategy Failed to Trigger. Still too conservative.")

    # 3. Verify Position Sizing Calculation
    print("\n[3] Testing Execution Sizing...")
    executor = ExecutorHolon()
    executor.balance_usd = 10.0
    
    # Create a mock signal
    signal = TradeSignal(symbol="BTC/USDT", direction='BUY', size=1.0, price=10000.0)
    
    # Decide Trade (Assume ORDERED regime)
    decision = executor.decide_trade(signal, 'ORDERED', 0.5)
    
    # Check Disposition
    print(f"   Decision: {decision.action}")
    print(f"   Adjusted Size: {decision.adjusted_size}")
    
    # Note: Logic for 'SCAVENGER_MAX_MARGIN' is used inside 'execute_transaction' usually
    # or implicitly by 'fixed_stake' if use_compounding=False.
    # In 'execute_transaction', we check:
    # usd_to_spend = fixed_stake * decision.adjusted_size (if not compounding)
    # or balance * adjusted_size (if compounding).
    # Config has SCAVENGER_MAX_MARGIN = 8.0.
    # If using fixed stake, we should set fixed_stake = 8.0 manually or logic update?
    # Ah, ExecutorHolon default uses compounding=True? Let's check init.
    
    # ExecutorHolon.__init__(use_compounding=True) -> uses balance_usd * adjusted_size.
    # In ORDERED mode, adjusted_size = 1.0 (100%).
    # So it tries to use 10.0 USD.
    # But Governor might block it?
    # Let's assume no governor for this unit test.
    
    if decision.action == 'EXECUTE' and decision.adjusted_size >= 0.9:
        print("✅ Execution Logic is deploying Full Capital (subject to Governor).")
    else:
        print("❌ Execution Logic is holding back.")

    print("\n🏁 Warp Speed Simulation Complete.")

if __name__ == "__main__":
    test_warp_speed_config()
