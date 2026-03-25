import sys
import os
import pprint
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())
if os.path.isdir('HolonicTrader'):
    sys.path.append(os.path.join(os.getcwd(), 'HolonicTrader'))

try:
    import config
    # Force trading mode to FUTURES to ensure we hit the right logic if relevant, 
    # though agent_kraken initializes independently.
    config.TRADING_MODE = 'FUTURES'
    from HolonicTrader.agent_kraken import KrakenHolon
except ImportError:
    print("❌ Failed to import HolonicTrader modules.")
    sys.exit(1)

def main():
    print("🔬 INITIALIZING KRAKEN DEBUGGER...")
    kraken = KrakenHolon()
    
    print("\n📡 Fetching Raw Balance (Futures)...")
    try:
        balance = kraken.futures.fetch_balance()
        
        info = balance.get('info', {})
        accounts = info.get('accounts', {})
        flex = accounts.get('flex', {})
        
        print(f"\n[PnL Check] Flex Keys: {list(flex.keys())}")
        print(f"totalUnrealized: {flex.get('totalUnrealized')}")
        print(f"unrealizedPnL (Deprecated?): {flex.get('unrealizedPnL')}")

        print("\n📡 Fetching Positions via fetch_positions()...")
        try:
            positions = kraken.futures.fetch_positions()
            print(f"✅ Found {len(positions)} positions via fetch_positions()")
            if positions:
                pprint.pprint(positions[0]) # Print first one to see structure
                
                # Dump positions to file
                with open('kraken_raw_positions.json', 'w') as f:
                    json.dump(positions, f, indent=2, default=str)
            else:
                print("⚠️ fetch_positions returned empty list.")
                
        except Exception as e:
            print(f"❌ fetch_positions failed: {e}")

        # Dump balance to file
        with open('kraken_raw_balance.json', 'w') as f:
             json.dump(balance, f, indent=2, default=str)


    except Exception as e:
        print(f"❌ Error fetching balance: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
