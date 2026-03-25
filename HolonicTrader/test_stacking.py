
import logging
import time
import sys
import os

# Add parent dir to path to find HolonicTrader package if running from inside
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from HolonicTrader.agent_oracle import MacroOracle, EntryOracleHolon
from HolonicTrader.agent_signal_provider import TradeSignal
import config

# Setup dummy config for test
config.MACRO_TICKERS = ['^GSPC', '^IXIC', '^RUT', 'USDT-USD']
config.MACRO_STACK_WEIGHT = 0.20
config.STRUCTURE_DRIVES_TARGETS = True

def test_macro_stack():
    print("--- TESTING MACRO ORACLE ---")
    oracle = MacroOracle()
    ctx = oracle.fetch_macro_context()
    print(f"Macro Context: {ctx}")
    
    score = ctx.get('bias_score', 0.0)
    risk_on = ctx.get('risk_on', False)
    print(f"Bias Score: {score}")
    print(f"Risk On: {risk_on}")
    
    if score == 0.0 and not ctx.get('details'):
        print("⚠️ No data fetched? Check yfinance.")
    else:
        print("✅ Data fetched successfully.")

    print("\n--- TESTING APPLICATION TO SIGNAL ---")
    # Mock EntryOracle
    entry_oracle = EntryOracleHolon()
    entry_oracle.macro_oracle = oracle # Inject pre-warmed oracle
    
    # Create Dummy Signal (Long)
    sig_long = TradeSignal(
        symbol='BTC/USDT',
        direction='BUY',
        size=1.0,
        price=100000,
        conviction=0.5,
        metadata={'structure': {'pivots': {'R1': 105000, 'S1': 95000}}}
    )
    
    # Apply Physics
    res_long = entry_oracle.apply_market_physics('BTC/USDT', sig_long)
    
    if res_long:
        print(f"Original Conviction: 0.5")
        print(f"New Conviction: {res_long.conviction}")
        print(f"Stack Score: {res_long.metadata.get('macro_stack_score', 0)}")
        print(f"Target Price: {res_long.take_profit_price}")
        print(f"TP Type: {res_long.metadata.get('take_profit_type')}")
        
        expected_boost = score * config.MACRO_STACK_WEIGHT
        print(f"Expected Boost: {expected_boost}")
        
        if res_long.metadata.get('take_profit_type') == 'STRUCTURE_PIVOT':
             print("✅ Structure Target Applied.")
        else:
             print("❌ Structure Target Failed.")
    else:
        print("❌ Signal Vetoed entirely.")

if __name__ == "__main__":
    test_macro_stack()
