import sys
import os
import pandas as pd
from datetime import datetime

# Add current dir to path
sys.path.append(os.getcwd())

from HolonicTrader.agent_governor import GovernorHolon
from HolonicTrader.agent_executor import ExecutorHolon, TradeSignal
from database_manager import DatabaseManager
import config

def validate_integration():
    print("==================================================")
    print("   HOLONIC TRADER - INTEGRATION VALIDATION")
    print("==================================================")
    
    # 1. Setup
    db_manager = DatabaseManager()
    governor = GovernorHolon(db_manager=db_manager)
    executor = ExecutorHolon(db_manager=db_manager)
    
    # 2. Check Balance & Metabolism
    # This should be $100 now after our update script
    portfolio = db_manager.get_portfolio()
    balance = portfolio['balance_usd']
    print(f"\n[1] Portfolio Balance: ${balance:.2f} (Principal: ${config.PRINCIPAL:.2f})")
    
    # Update Governor with current balance
    governor.update_balance(balance)
    metabolism = governor.get_metabolism_state()
    print(f"[2] Metabolic State: {metabolism} (Should be PREDATOR if > $90)")
    
    if balance <= config.PRINCIPAL:
        print(f"WARNING: Balance ${balance:.2f} <= Principal ${config.PRINCIPAL:.2f}. Minimax will REJECT all trades.")
    
    # 3. Test Position Sizing (Phase 12)
    # Simulate current ATR > ref ATR (Low Vol)
    current_atr = 0.001
    atr_ref = 0.0015 # 1.5x ref = Reduce size
    
    is_allowed, qty, lev = governor.calc_position_size("BTC/USDT", 60000.0, current_atr, atr_ref)
    
    print("\n[3] Governor Risk Check (Minimax + Kelly + Vol Scalar):")
    if is_allowed:
        print(f"  ✅ APPROVED: Qty={qty:.8f}, Lev={lev}x")
        # Max risk check
        max_risk = governor.calculate_max_risk(balance)
        print(f"  Minimax Max Risk: ${max_risk:.2f}")
    else:
        print("  ❌ REJECTED by Governor")

    # 4. Test Executor Decision (Entropy)
    signal = TradeSignal("BTC/USDT", "BUY", qty, 60000.0)
    # Simulate ORDERED regime (Low Entropy)
    decision = executor.decide_trade(signal, "ORDERED", 0.5)
    
    print("\n[4] Executor Decision Flow (Entropy 0.5):")
    print(f"  Decision Action: {decision.action}")
    print(f"  Disposition: Autonomy={decision.disposition.autonomy}, Integration={decision.disposition.integration}")

    # 5. Check Strategy Exit Thresholds
    print("\n[5] Strategy Exit Thresholds (Centralized Config):")
    print(f"  PREDATOR TP Target: {config.PREDATOR_TAKE_PROFIT*100}% (Requested: 3%)")
    print(f"  SCAVENGER SL Target: {config.SCAVENGER_STOP_LOSS*100}% (Requested: 3%)")
    
    if config.PREDATOR_TAKE_PROFIT == 0.03:
        print("  ✅ Strategy Targets Verified")
    else:
        print("  ❌ Strategy Targets MISMATCH")

    print("\n==================================================")
    print("   VALIDATION COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    validate_integration()
