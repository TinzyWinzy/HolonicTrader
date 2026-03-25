
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from HolonicTrader.agent_trader import TraderHolon
from HolonicTrader.agent_observer import ObserverHolon
import config
import logging

# Disable heavy logging
logging.basicConfig(level=logging.ERROR)

def test_full_pipeline():
    print("🧪 Starting Full Pipeline Stability Test...")
    trader = TraderHolon(name="TestNexus")
    
    # Mock data for 2 assets
    symbols = ["BTC/USDT", "LINK/USDT"]
    data = {}
    for s in symbols:
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2026-01-01', periods=100, freq='15min'),
            'open': np.random.normal(100, 1, 100),
            'high': np.random.normal(101, 1, 100),
            'low': np.random.normal(99, 1, 100),
            'close': np.random.normal(100, 1, 100),
            'volume': np.random.normal(1000, 100, 100)
        })
        data[s] = df

    # Test PATH 1: With Rust Signals (Simulated)
    print("\n🔹 Testing Path 1: Rust Accelerated...")
    rust_sigs = {
        'shannon_entropy': 2.1,
        'rsi': 45.0,
        'atr': 1.5,
        'bb_upper': 105.0,
        'bb_lower': 95.0,
        'ou_mu': 100.0
    }
    
    try:
        res1 = trader._analyze_asset(
            "BTC/USDT", data["BTC/USDT"], None, 0.5, {}, 0.0, rust_sigs=rust_sigs
        )
        if res1:
            print(f"✅ Path 1 Success: {res1['symbol']} | Action: {res1['entry_signal'].direction if res1['entry_signal'] else 'HOLD'}")
        else:
            print("❌ Path 1 Failed: Returned None")
    except Exception as e:
        print(f"❌ Path 1 CRASH: {e}")
        import traceback
        traceback.print_exc()

    # Test PATH 2: Legacy Fallback (No Rust Sigs)
    print("\n🔹 Testing Path 2: Legacy Fallback...")
    try:
        res2 = trader._analyze_asset(
            "LINK/USDT", data["LINK/USDT"], None, 0.5, {}, 0.0, rust_sigs=None
        )
        if res2:
            print(f"✅ Path 2 Success: {res2['symbol']} | Action: {res2['entry_signal'].direction if res2['entry_signal'] else 'HOLD'}")
        else:
            print("❌ Path 2 Failed: Returned None")
    except Exception as e:
        print(f"❌ Path 2 CRASH: {e}")
        import traceback
        traceback.print_exc()

    print("\n🏆 Stability Test Concluded.")

if __name__ == "__main__":
    test_full_pipeline()
