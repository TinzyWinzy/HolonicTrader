import numpy as np
import pandas as pd
from HolonicTrader.agent_entropy import EntropyHolon
import matplotlib.pyplot as plt

def generate_signals(n=500):
    t = np.linspace(0, 100, n)
    
    # 1. Sine Wave (Highly Ordered/Predictable patterns)
    # Derivative (returns) of sine is cosine, which is also deterministic.
    sine_wave = np.sin(t)
    sine_returns = pd.Series(np.diff(sine_wave)).dropna()

    # 2. Gaussian Noise (Normal Distribution - "Standard" Market randomness)
    # Most returns near 0, some tails.
    gaussian_noise = np.random.normal(0, 1, n)
    gaussian_returns = pd.Series(gaussian_noise)

    # 3. Uniform Noise (Max Chaos)
    # Equal probability of any return value within range.
    uniform_noise = np.random.uniform(-1, 1, n)
    uniform_returns = pd.Series(uniform_noise)

    # 4. Spiky/Clustered (Leptokurtic - Closer to real crypto market)
    # Mostly zeros/small moves, rare huge jumps.
    spiky = np.random.normal(0, 0.1, n)
    spiky[::50] = np.random.choice([-5, 5], size=n//50) # Occasional jumps
    spiky_returns = pd.Series(spiky)

    return {
        "Sine Wave": sine_returns,
        "Gaussian Noise": gaussian_returns,
        "Uniform Noise": uniform_returns,
        "Crypto-Like (Spiky)": spiky_returns
    }

def run_proof():
    agent = EntropyHolon()
    signals = generate_signals()
    
    print(f"{'SIGNAL TYPE':<25} | {'ENTROPY':<10} | {'REGIME':<12} | {'% of CHAOS (1.35)'}")
    print("-" * 65)
    
    chaos_thresh = 1.35
    
    for name, series in signals.items():
        # Clean infinite values if any (from log returns in real app, here we simulated returns directly)
        series = series.replace([np.inf, -np.inf], np.nan).dropna()
        
        ent = agent.calculate_shannon_entropy(series)
        regime = agent.determine_regime(ent)
        pct_chaos = (ent / chaos_thresh) * 100
        
        print(f"{name:<25} | {ent:<10.4f} | {regime:<12} | {pct_chaos:.1f}%")

if __name__ == "__main__":
    run_proof()
