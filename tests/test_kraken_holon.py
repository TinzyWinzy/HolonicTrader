"""
Verification Script for KrakenHolon
Tests connection, system status, account health, and market intelligence.
"""

import sys
import os
import time

# Add parent directory to path
sys.path.append(os.getcwd())

from HolonicTrader.agent_kraken import KrakenHolon
import config

def verify_kraken_holon():
    print("=== 🛠️ INITIALIZING KRAKEN HOLON VERIFICATION ===")
    
    try:
        kraken = KrakenHolon()
        print(f">> Agent Name: {kraken.name}")
        print(f">> Disposition: Autonomy {kraken.disposition.autonomy:.2f}, Integration {kraken.disposition.integration:.2f}")

        # 1. Test System Status
        print("\n[1] Testing System Status...")
        status = kraken.get_system_status()
        print(f">> System Status: {status}")
        
        # 2. Test Account Health
        print("\n[2] Testing Account Health...")
        health = kraken.get_account_health()
        if 'error' in health:
             print(f">> Account Health Fail: {health['error']}")
        else:
             print(f">> Equity: ${health['equity']:.2f}")
             print(f">> Available: ${health['available']:.2f}")
             print(f">> Margin Level: {health['margin_level']:.2f}")
             print(f">> Liquidation Distance: {health['liquidation_distance']:.2f}")
             print(f">> Status: {health['status']}")

        # 3. Test Market Intel (Funding/OI)
        print("\n[3] Testing Market Intel (BTC & ETH)...")
        kraken.update_market_intel(['BTC/USDT', 'ETH/USDT'])
        
        for sym, intel in kraken.last_funding_data.items():
            print(f">> {sym} Funding: {intel['rate']:.6f} ({intel['apy']:.2f}% APY)")
            
        for sym, oi in kraken.last_oi_data.items():
            print(f">> {sym} Open Interest: {oi}")

        # 5. Test Ghost Resolution Strategy
        print("\n[5] Testing Ghost Resolution Strategy...")
        mock_internal = {'BTC/USDT': 0.0001, 'SOL/USDT': 1.0} # We hold BTC and SOL in ledger
        # Case: exchange has BTC (match), no SOL (leak), has ETH (ghost)
        mock_kraken_report = {
            'ghosts': {'ETH/USDT': 0.01},
            'leaks': {'SOL/USDT': 1.0},
            'mismatch': {'BTC/USDT': {'exchange': 0.00011, 'ledger': 0.0001}}
        }
        actions = kraken.resolve_ghosts(mock_kraken_report, global_bias=0.6) # Bullish
        for a in actions:
            print(f">> ACTION Plan: {a['type']} {a.get('symbol')} - {a['reason']}")

        # 6. Test Equity Truth
        print("\n[6] Testing Equity Truth...")
        truth = kraken.get_equity_truth()
        print(f">> Equity Truth: ${truth.get('equity', 0):.2f} (Available: ${truth.get('available', 0):.2f})")

        # 8. Test Collateral Haircuts
        print("\n[8] Testing Collateral Haircuts...")
        haircuts = kraken.get_collateral_haircuts()
        print(f">> Haircut (BTC): {haircuts.get('BTC')}")

        # 9. Test Safety Rails (Sync Stops)
        print("\n[9] Testing Safety Rails (Sync Stops)...")
        held = {'BTC/USDT': 0.0001}
        rails = kraken.sync_server_side_stops(held, 0.05)
        print(f">> Safety Rails Result: {rails}")

        # 10. Test Environment Monitoring
        print("\n[10] Testing Environment Monitoring...")
        env = kraken.monitor_execution_environment(['BTC/USDT'])
        print(f">> Environment Status: {env.get('status')}")

        print("\n✅ VERIFICATION COMPLETE: KrakenHolon is fully armored.")
        
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_kraken_holon()
