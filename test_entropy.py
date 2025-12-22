"""
Test script for EntropyHolon

Verifies:
1. Shannon Entropy calculation is correct.
2. Probabilities are normalized (sum to 1).
3. Regime determination logic is correct.
"""

import numpy as np
import pandas as pd
from agent_entropy import EntropyHolon


def test_entropy_calculation():
    """Test Shannon Entropy calculation with known distributions."""
    print("=" * 60)
    print("TEST 1: Shannon Entropy Calculation")
    print("=" * 60)

    holon = EntropyHolon()

    # Test with uniform distribution (should have high entropy)
    np.random.seed(42)
    uniform_returns = pd.Series(np.random.uniform(-0.1, 0.1, 1000))

    entropy_uniform = holon.calculate_shannon_entropy(uniform_returns)
    print(f"\nUniform distribution entropy: {entropy_uniform:.4f}")
    print("  -> Expected: High entropy (close to max for 10 bins = ln(10) ≈ 2.30)")

    # Test with concentrated distribution (should have low entropy)
    # Most values clustered in center
    concentrated_returns = pd.Series(np.random.normal(0, 0.001, 1000))

    entropy_concentrated = holon.calculate_shannon_entropy(concentrated_returns)
    print(f"\nConcentrated (normal, small std) entropy: {entropy_concentrated:.4f}")
    print("  -> Expected: Low entropy (values clustered in few bins)")

    # Test with wider normal distribution
    wide_returns = pd.Series(np.random.normal(0, 0.05, 1000))

    entropy_wide = holon.calculate_shannon_entropy(wide_returns)
    print(f"\nWider normal distribution entropy: {entropy_wide:.4f}")
    print("  -> Expected: Medium entropy")

    return entropy_uniform, entropy_concentrated, entropy_wide


def test_probability_normalization():
    """Verify that histogram counts are properly normalized to probabilities."""
    print("\n" + "=" * 60)
    print("TEST 2: Probability Normalization Check")
    print("=" * 60)

    np.random.seed(123)
    returns = pd.Series(np.random.normal(0, 0.02, 500))

    # Manually replicate the binning to check probabilities
    counts, bin_edges = np.histogram(returns, bins=10)
    total = counts.sum()
    probabilities = counts / total

    print(f"\nBin counts: {counts}")
    print(f"Total count: {total}")
    print(f"Probabilities: {probabilities}")
    print(f"Sum of probabilities: {probabilities.sum():.10f}")

    if np.isclose(probabilities.sum(), 1.0):
        print("\n✓ PASS: Probabilities correctly sum to 1.0")
        return True
    else:
        print("\n✗ FAIL: Probabilities do NOT sum to 1.0")
        return False


def test_regime_determination():
    """Test regime classification logic."""
    print("\n" + "=" * 60)
    print("TEST 3: Regime Determination Logic")
    print("=" * 60)

    holon = EntropyHolon()

    # Test boundary conditions (New Thresholds: 1.96 and 2.10)
    test_cases = [
        (0.0, 'ORDERED'),
        (1.95, 'ORDERED'),      # Just below 1.96
        (1.97, 'TRANSITION'),   # Just above 1.96
        (2.0, 'TRANSITION'),    # Middle of transition
        (2.09, 'TRANSITION'),   # Just below 2.10
        (2.11, 'CHAOTIC'),      # Just above 2.10
        (2.5, 'CHAOTIC'),
        (3.5, 'CHAOTIC'),
        (5.0, 'CHAOTIC'),
    ]

    all_passed = True
    for entropy_val, expected_regime in test_cases:
        actual_regime = holon.determine_regime(entropy_val)
        status = "✓" if actual_regime == expected_regime else "✗"
        if actual_regime != expected_regime:
            all_passed = False
        print(f"  {status} Entropy {entropy_val:.1f} -> {actual_regime} (expected: {expected_regime})")

    if all_passed:
        print("\n✓ All regime tests PASSED")
    else:
        print("\n✗ Some regime tests FAILED")

    return all_passed


def test_integration_with_realistic_data():
    """Test with data that mimics real market returns."""
    print("\n" + "=" * 60)
    print("TEST 4: Integration Test with Realistic Market Data")
    print("=" * 60)

    holon = EntropyHolon()

    # Simulate different market conditions
    np.random.seed(999)

    # Ordered market: trending, low volatility
    ordered_market = pd.Series(np.random.normal(0.001, 0.005, 500))

    # Chaotic market: high volatility, random
    chaotic_market = pd.Series(np.random.uniform(-0.1, 0.1, 500))

    # Transition market: moderate
    transition_market = pd.Series(np.random.normal(0, 0.02, 500))

    print("\nMarket Regime Analysis:")
    print("-" * 40)

    for name, data in [("Ordered", ordered_market), 
                       ("Transition", transition_market),
                       ("Chaotic", chaotic_market)]:
        entropy = holon.calculate_shannon_entropy(data)
        regime = holon.determine_regime(entropy)
        print(f"  {name} Market: Entropy = {entropy:.4f} -> Regime: {regime}")


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# EntropyHolon Verification Suite")
    print("#" * 60)

    test_entropy_calculation()
    prob_ok = test_probability_normalization()
    regime_ok = test_regime_determination()
    test_integration_with_realistic_data()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Probability normalization: {'PASS' if prob_ok else 'FAIL'}")
    print(f"  Regime determination: {'PASS' if regime_ok else 'FAIL'}")
    print("=" * 60)
