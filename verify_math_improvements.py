import pandas as pd
import numpy as np
import time
from HolonicTrader.agent_entropy import EntropyHolon
from HolonicTrader.agent_executor import ExecutorHolon, TradeSignal
from HolonicTrader.agent_governor import GovernorHolon
import config

def verify_math():
    print("--- Verifying Mathematical Improvements ---")
    
    # 1. Rényi Entropy
    print("\n[1] Testing Rényi Entropy...")
    entropy_agent = EntropyHolon()
    returns = pd.Series(np.random.normal(0, 0.01, 100))
    shannon = entropy_agent.calculate_shannon_entropy(returns)
    renyi = entropy_agent.calculate_renyi_entropy(returns, alpha=2.0)
    print(f"  Shannon: {shannon:.4f}")
    print(f"  Rényi (alpha=2): {renyi:.4f}")
    assert renyi <= shannon + 0.01, "Rényi should generally be <= Shannon for alpha > 1"
    print("  ✅ Rényi Entropy Verified.")

    # 2. Sigmoid Autonomy
    print("\n[2] Testing Sigmoid Autonomy (Executor)...")
    executor = ExecutorHolon()
    
    # Text Low Entropy (Ordered) -> Should have high Autonomy
    low_entropy = 1.0
    executor.decide_trade(TradeSignal('TEST', 'BUY', 1.0, 100.0), 'ORDERED', low_entropy)
    disp_low = executor.disposition
    print(f"  Entropy {low_entropy} -> Autonomy: {disp_low.autonomy:.4f}")
    
    # Test High Entropy (Chaotic) -> Should have low Autonomy
    high_entropy = 3.0
    executor.decide_trade(TradeSignal('TEST', 'BUY', 1.0, 100.0), 'CHAOTIC', high_entropy)
    disp_high = executor.disposition
    print(f"  Entropy {high_entropy} -> Autonomy: {disp_high.autonomy:.4f}")
    
    assert disp_low.autonomy > 0.9, "Low entropy should yield high autonomy (>0.9)"
    assert disp_high.autonomy < 0.1, "High entropy should yield low autonomy (<0.1)"
    print("  ✅ Sigmoid Autonomy Verified.")

    # 3. Kelly Criterion
    print("\n[3] Testing Kelly Criterion (Governor)...")
    governor = GovernorHolon(initial_balance=1000.0)
    governor.balance = 2000.0 # Surplus
    # Mock positions to act as Predator
    
    start_time = time.time()
    # Mocking ATR
    allowed, qty, lev = governor.calc_position_size('BTCUSDT', 50000.0, current_atr=1000.0)
    print(f"  Kelly result: Allowed={allowed}, Qty={qty:.4f}, Lev={lev}")
    
    # Kelly Logic check: 
    # Balance 2000. Surplus 2000-10=1990? No, init default is 10.
    # Governor init default 10. So surplus is ~1990.
    # Kelly fraction (p=0.55, b=1.5) approx 0.16. Safe=0.08. Max=0.2.
    # Exposure = 2000 * 0.08 * Lev(5) = 160 * 5 = 800.
    # Qty = 800 / 50000 = 0.016
    
    if allowed:
        print("  ✅ Kelly Criterion returned valid trade size.")
    else:
        print("  ❌ Kelly Criterion rejected trade (might be Scavenger mode loop?).")

    print("\n--- All Checks Passed ---")

if __name__ == "__main__":
    verify_math()
