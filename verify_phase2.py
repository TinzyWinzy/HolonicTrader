import numpy as np
import pandas as pd
from HolonicTrader.kalman import KalmanFilter1D
from performance_tracker import calculate_omega_ratio
from HolonicTrader.agent_strategy import StrategyHolon

def verify_phase2():
    print("--- Verifying Phase 2 Mathematical Improvements ---")
    
    # 1. Kalman Filter Test
    print("\n[1] Testing Kalman Filter...")
    # Generate synthetic price data (Sine wave + Noise)
    t = np.linspace(0, 10, 100)
    true_price = 100 + 10 * np.sin(t)
    noise = np.random.normal(0, 2, 100)
    noisy_price = true_price + noise
    
    # Tuned Q=1.0 to track the sine wave better (less lag), R=2.0 to reject noise
    kf = KalmanFilter1D(process_noise=1.0, measurement_noise=2.0)
    estimates = []
    
    for z in noisy_price:
        estimates.append(kf.update(z))
        
    estimates = np.array(estimates)
    mse_noisy = np.mean((noisy_price - true_price)**2)
    mse_kalman = np.mean((estimates - true_price)**2)
    
    print(f"  MSE Noisy: {mse_noisy:.4f}")
    print(f"  MSE Kalman: {mse_kalman:.4f}")
    
    assert mse_kalman < mse_noisy, "Kalman filter should reduce noise (MSE)"
    print("  ✅ Kalman Filter Verified (Noise Reduced).")
    
    # 2. Strategy Integration Test
    print("\n[2] Testing Strategy Integration...")
    strat = StrategyHolon()
    prices = pd.Series(noisy_price)
    # Warm up internal filter
    est = strat.get_kalman_estimate("TEST_SYM", prices)
    print(f"  Strategy returned Kalman Estimate: {est:.4f}")
    assert est != 0.0, "Strategy should return non-zero estimate"
    print("  ✅ Strategy Integration Verified.")

    # 3. Omega Ratio Test
    print("\n[3] Testing Omega Ratio...")
    # Scenario A: Good returns (High Omega)
    returns_good = [10, 20, -5, 15, -2, 30] 
    omega_good = calculate_omega_ratio(returns_good, threshold=0.0)
    print(f"  Omega (Good): {omega_good:.4f}")
    
    # Scenario B: Bad returns (Low Omega)
    returns_bad = [-10, -20, 5, -15, 2, -30]
    omega_bad = calculate_omega_ratio(returns_bad, threshold=0.0)
    print(f"  Omega (Bad): {omega_bad:.4f}")
    
    assert omega_good > 1.0, "Good returns should have Omega > 1"
    assert omega_bad < 1.0, "Bad returns should have Omega < 1"
    print("  ✅ Omega Ratio Verified.")

    print("\n--- All Phase 2 Checks Passed ---")

if __name__ == "__main__":
    verify_phase2()
