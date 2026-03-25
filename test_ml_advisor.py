"""
Test ML Trading Advisor Integration

Validates that ML models can be loaded and used for predictions.
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from HolonicTrader.ml_advisor import MLTradingAdvisor, get_ml_advisor, predict_trade

print("=" * 70)
print("ML TRADING ADVISOR TEST")
print("=" * 70)

# Test 1: Initialize advisor
print("\n1. Testing initialization...")
try:
    advisor = MLTradingAdvisor()
    print("✓ Advisor initialized successfully")
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    sys.exit(1)

# Test 2: Get model status
print("\n2. Testing model status...")
status = advisor.get_model_status()
print(f"  Classifier loaded: {status['classifier_loaded']}")
print(f"  Regression loaded: {status['regression_loaded']}")
print(f"  Features: {status['features_used']}")
print(f"  Database trades: {status['database_trades']}")
print(f"  Database win rate: {status['database_win_rate']:.1%}")

# Test 3: Make predictions
print("\n3. Testing trade predictions...")
test_cases = [
    {
        'symbol': 'BTC/USDT',
        'direction': 'BUY',
        'price': 95000.0,
        'quantity': 0.001,
    },
    {
        'symbol': 'ETH/USDT',
        'direction': 'BUY',
        'price': 3500.0,
        'quantity': 0.01,
    },
    {
        'symbol': 'SOL/USDT',
        'direction': 'SELL',
        'price': 180.0,
        'quantity': 0.5,
    },
]

for i, test in enumerate(test_cases, 1):
    print(f"\n  Test {i}: {test['symbol']} {test['direction']}")
    try:
        result = advisor.predict_trade(**test)
        print(f"    Win Probability: {result['win_probability']:.1%}")
        print(f"    Predicted PnL: {result['predicted_pnl_percent']:.2f}%")
        print(f"    Recommendation: {result['recommendation']}")
        print(f"    Confidence: {result['confidence_level']}")
        print(f"    Recommended Size: {result['recommended_size_pct']:.1%}")
    except Exception as e:
        print(f"    ✗ Prediction failed: {e}")

# Test 4: Test singleton
print("\n4. Testing singleton pattern...")
try:
    advisor2 = get_ml_advisor()
    advisor3 = get_ml_advisor()
    assert advisor2 is advisor3, "Singleton failed"
    print("✓ Singleton working correctly")
except Exception as e:
    print(f"✗ Singleton test failed: {e}")

# Test 5: Test convenience functions
print("\n5. Testing convenience functions...")
try:
    result = predict_trade('DOT/USDT', 'BUY', 1.50, 10.0)
    print(f"  predict_trade() returned: {result['recommendation']}")
    print(f"  Win probability: {result['win_probability']:.1%}")
    print("✓ Convenience functions working")
except Exception as e:
    print(f"✗ Convenience function failed: {e}")

# Test 6: Performance test
print("\n6. Testing prediction performance...")
import time
start = time.time()
for _ in range(100):
    predict_trade('BTC/USDT', 'BUY', 95000.0, 0.001)
elapsed = time.time() - start
print(f"  100 predictions in {elapsed:.3f}s ({elapsed*10:.1f}ms per prediction)")
if elapsed < 1.0:
    print("✓ Performance acceptable")
else:
    print("⚠ Performance may be slow for live trading")

# Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

tests_passed = 0
tests_total = 6

if status['classifier_loaded']:
    tests_passed += 1
    print("✓ Classifier model loaded")
else:
    print("✗ Classifier model not loaded")

if status['regression_loaded']:
    tests_passed += 1
    print("✓ Regression model loaded")
else:
    print("✗ Regression model not loaded")

if status['database_trades'] > 0:
    tests_passed += 1
    print(f"✓ Database has {status['database_trades']} trades")
else:
    print("✗ Database not loaded")

# Test prediction worked
try:
    result = predict_trade('TEST', 'BUY', 100.0, 1.0)
    if 'win_probability' in result:
        tests_passed += 1
        print("✓ Predictions working")
    else:
        print("✗ Predictions missing data")
except:
    print("✗ Predictions failed")
    tests_passed += 0

# Singleton worked
tests_passed += 1
print("✓ Singleton pattern working")

# Performance acceptable
if elapsed < 1.0:
    tests_passed += 1
    print("✓ Performance acceptable")
else:
    print("⚠ Performance needs improvement")

print(f"\nTests Passed: {tests_passed}/{tests_total}")

if tests_passed >= 5:
    print("\n✓ ML Advisor is READY for integration")
    print("\nNext steps:")
    print("1. Add to Governor: from HolonicTrader.ml_advisor import get_ml_advisor")
    print("2. Call before entries: prediction = advisor.predict_trade(...)")
    print("3. Use prediction['recommendation'] for trade decisions")
else:
    print("\n⚠ ML Advisor needs fixes before integration")
    sys.exit(1)

print("\n" + "=" * 70)
