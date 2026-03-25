"""
Arbitrage Integration Test - REAL DATA
Tests arb system with REAL KuCoin + Kraken data (NO MOCKS)
"""

import sys
import os
import time
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# FIX: Set UTF-8 encoding for Windows console compatibility
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import config
# Force FUTURES mode for validation
config.TRADING_MODE = 'FUTURES'

from HolonicTrader.agent_observer import ObserverHolon
from HolonicTrader.agent_arbitrage import ArbitrageHolon

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestArbitrage")

def test_arbitrage_integration():
    print("--- Starting Arbitrage Integration Test (REAL DATA) ---")

    # 1. Initialize Kraken Futures Observer (REAL)
    print("Initializing Kraken Observer...")
    try:
        kraken_observer = ObserverHolon(exchange_id='krakenfutures', symbol='BTC/USDT')
        print("  [OK] Kraken Futures Observer initialized")
    except Exception as e:
        print(f"  [FAIL] Kraken Observer: {e}")
        return

    # 2. Initialize KuCoin Observer (REAL - NO MOCKS)
    print("Initializing KuCoin Observer...")
    try:
        kucoin_observer = ObserverHolon(exchange_id='kucoin', symbol='BTC/USDT')
        print("  [OK] KuCoin Observer initialized")
    except Exception as e:
        print(f"  [FAIL] KuCoin Observer: {e}")
        return

    # 3. Initialize Arbitrage Holon
    arb_holon = ArbitrageHolon()

    # 4. Link Holons
    arb_holon.kraken_observer = kraken_observer
    arb_holon.kucoin_observer = kucoin_observer

    # 5. Define symbols to test (common to both exchanges)
    test_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']

    print(f"\nPerforming Sync for: {test_symbols}")

    # 6. Run Sync
    try:
        arb_holon.perform_sync(test_symbols)
        print("  [OK] Sync completed")
    except Exception as e:
        print(f"  [FAIL] Error during perform_sync: {e}")
        import traceback
        traceback.print_exc()
        return

    # 7. Inspect Results
    print("\n--- Funding Yields (8H Gross) ---")
    for sym in test_symbols:
        yield_8h = arb_holon.funding_yields.get(sym, 0)
        context = arb_holon.funding_context.get(sym, {})
        raw_rate = context.get('rate', 0)
        print(f"  {sym}: {yield_8h:+.4f}%/8h (raw: {raw_rate:+.6f})")

    print("\n--- Price Spreads (Realizable) ---")
    for sym in test_symbols:
        spread_data = arb_holon.price_spreads.get(sym, {})
        if isinstance(spread_data, dict):
            long_sp = spread_data.get('long', 0)
            short_sp = spread_data.get('short', 0)
            print(f"  {sym}: Long={long_sp:+.3f}%, Short={short_sp:+.3f}%")
        else:
            print(f"  {sym}: {spread_data}")

    # 8. Check Signals
    print("\n--- Active Signals ---")
    for sym in test_symbols:
        sig = arb_holon.get_active_signal(sym, 0.0)
        if sig:
            print(f"  {sym}: [{sig['direction']}] {sig['reason']} (Conf: {sig['confidence']:.0%})")
        else:
            print(f"  {sym}: No signal (no arb > threshold)")

    # 9. Validation
    print("\n--- Validation ---")
    issues = []
    for sym in test_symbols:
        spread_data = arb_holon.price_spreads.get(sym, {})
        if isinstance(spread_data, dict):
            if abs(spread_data.get('long', 0)) > 0.05:  # >5% = suspicious
                issues.append(f"{sym} spread too large")
        yield_8h = arb_holon.funding_yields.get(sym, 0)
        if abs(yield_8h) > 5.0:  # >5%/8h = extreme
            issues.append(f"{sym} funding extreme: {yield_8h:.2f}%")
    
    if issues:
        print("  [WARN] Issues found:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  [OK] All calculations within expected ranges")

    print("\n--- Test Complete ---")

if __name__ == "__main__":
    test_arbitrage_integration()
