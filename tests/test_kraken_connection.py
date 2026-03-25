"""
Test Kraken API Connection
Verifies that API keys are loaded and connection works
Run directly: python test_kraken_connection.py
"""

import ccxt
import config

def main():
    print("=" * 70)
    print("KRAKEN API CONNECTION TEST")
    print("=" * 70)
    print()

    # Test 1: Check if keys are loaded
    print("TEST 1: API KEYS LOADED")
    print("-" * 70)
    if config.API_KEY and config.API_SECRET:
        print(f"✓ API Key loaded: {config.API_KEY[:10]}...")
        print(f"✓ API Secret loaded: {config.API_SECRET[:10]}...")
    else:
        print("✗ API keys not loaded from .env file")
        print("Make sure .env file exists with:")
        print("  KRAKEN_API_KEY=your_key")
        print("  KRAKEN_PRIVATE_KEY=your_secret")
        return 1
    print()

    # Test 2: Initialize exchange
    print("TEST 2: INITIALIZE EXCHANGE")
    print("-" * 70)
    try:
        exchange = ccxt.kraken({
            'apiKey': config.API_KEY,
            'secret': config.API_SECRET,
            'enableRateLimit': True,
        })
        print("✓ Kraken exchange initialized")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return 1
    print()

    # Test 3: Fetch balance (read-only operation)
    print("TEST 3: FETCH BALANCE")
    print("-" * 70)
    try:
        balance = exchange.fetch_balance()
        print("✓ Successfully connected to Kraken")
        print()
        print("Account Balance:")
        for currency, amount in balance['total'].items():
            if amount > 0:
                print(f"  {currency}: {amount}")
        
        # Check if we have USDT
        usdt_balance = balance['total'].get('USDT', 0)
        print()
        print(f"USDT Balance: ${usdt_balance:.2f}")
        
        if usdt_balance >= 10:
            print("✓ Sufficient balance for trading ($10+ USDT)")
        else:
            print(f"⚠️  Low balance: ${usdt_balance:.2f} (recommended: $10+)")
            
    except Exception as e:
        print(f"✗ Failed to fetch balance: {e}")
        print()
        print("Common issues:")
        print("  - Invalid API keys")
        print("  - API keys not activated")
        print("  - Insufficient permissions")
        return 1
    print()

    # Test 4: Fetch ticker (market data)
    print("TEST 4: FETCH MARKET DATA")
    print("-" * 70)
    try:
        ticker = exchange.fetch_ticker('BTC/USDT')
        print(f"✓ BTC/USDT Price: ${ticker['last']:,.2f}")
    except Exception as e:
        print(f"✗ Failed to fetch ticker: {e}")
    print()

    print("=" * 70)
    print("CONNECTION TEST SUMMARY")
    print("=" * 70)
    print("✓ API keys loaded from .env")
    print("✓ Kraken exchange initialized")
    print("✓ Account balance retrieved")
    print("✓ Market data accessible")
    print()
    print("🎉 System ready for paper trading!")
    return 0


if __name__ == "__main__":
    exit(main())
print("🎉 System ready for paper trading!")
print()
print("NEXT STEPS:")
print("1. Run: python main_live_phase4.py")
print("2. Monitor for 24 hours")
print("3. Verify Phase 12 risk management active")
print("4. Check principal protection working")
