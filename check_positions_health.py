
import sys
import os
import time
from datetime import datetime

# Ensure import paths
sys.path.append(os.getcwd())


import config
from HolonicTrader.agent_actuator import ActuatorHolon
from HolonicTrader.agent_guardian import ExitGuardianHolon

def check_health():
    print("🏥 Running Holonic Health Check...")
    print(f"Trading Mode: {config.TRADING_MODE}")
    
    # 1. Initialize Actuator (Connects to Exchange)
    try:
        actuator = ActuatorHolon()
        print("✅ Actuator Connected.")
    except Exception as e:
        print(f"❌ Actuator Connection Failed: {e}")
        return

    # 2. Fetch Positions
    try:
        positions = actuator.exchange.fetch_positions()
        active_positions = [p for p in positions if float(p['contracts']) > 0]
        print(f"✅ Positions Fetched: {len(active_positions)} Active")
    except Exception as e:
        print(f"❌ Failed to fetch positions: {e}")
        return

    if not active_positions:
        print("ℹ️ No Open Positions.")
        return

    # 3. Initialize Guardian for Logic
    guardian = ExitGuardianHolon()

    # 4. Analyze Each
    print("\n" + "="*60)
    print(f"{'SYMBOL':<15} {'SIDE':<6} {'SIZE':<10} {'ENTRY':<10} {'CURRENT':<10} {'LIQUIDITY':<15}")
    print("-" * 60)

    for p in active_positions:
        symbol = p['symbol']
        side = p['side'].upper() # long/short
        size = float(p['contracts'])
        entry = float(p['entryPrice'])
        
        # Current Price & Book
        try:
            # Map Symbol for Futures
            exec_symbol = symbol 
            # (Kraken returns 'PF_XRPUSD', Actuator uses 'PF_XRPUSD' for fetch)
            
            # Fetch Ticker for Price
            ticker = actuator.exchange.fetch_ticker(exec_symbol)
            current_price = float(ticker['last'])
            
            # Fetch Book for Liquidity
            book = actuator.fetch_order_book(exec_symbol)
            
            # Check Health
            # If Long -> We Sell -> Check Bids
            # If Short -> We Buy -> Check Asks
            exit_dir = 'SELL' if side == 'LONG' else 'BUY'
            
            health = guardian.check_liquidity_health(symbol, exit_dir, size, book)
            
            # Normalize Symbol for Display
            disp_sym = symbol
            if config.TRADING_MODE=='FUTURES':
                # Reverse map if possible or just show raw
                for k, v in config.KRAKEN_SYMBOL_MAP.items():
                    if v == symbol:
                        disp_sym = k
                        break
            
            print(f"{disp_sym:<15} {side:<6} {size:<10.4f} {entry:<10.4f} {current_price:<10.4f} {health:<15}")
            
        except Exception as e:
            print(f"{symbol:<15} ❌ Error: {e}")

    print("="*60 + "\n")

if __name__ == "__main__":
    check_health()
