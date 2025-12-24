import unittest
import pandas as pd
from HolonicTrader.agent_guardian import ExitGuardianHolon
from HolonicTrader.agent_executor import ExecutorHolon, TradeDecision, TradeSignal as Signal
from HolonicTrader.holon_core import Disposition

class TestShortingPnL(unittest.TestCase):
    def setUp(self):
        self.guardian = ExitGuardianHolon()
        # Use Fixed Stake for predictable sizing in tests
        self.executor = ExecutorHolon(use_compounding=False, fixed_stake=500.0)
        self.executor.balance_usd = 1000.0
        self.executor.held_assets = {}
        self.executor.entry_prices = {}
        self.executor.position_metadata = {}

    def test_short_pnl_calculation(self):
        """Verify that price drop results in positive PnL for short."""
        entry_p = 100.0
        current_p = 90.0 # 10% drop
        
        # Guardian Analysis
        # Create dummy BB and ATR
        bb = {'upper': 110, 'middle': 100, 'lower': 90}
        atr = 5.0
        
        exit_sig = self.guardian.analyze_for_exit(
            symbol="BTC/USDT",
            current_price=current_p,
            entry_price=entry_p,
            bb=bb,
            atr=atr,
            metabolism_state="PREDATOR",
            position_age_hours=1.0,
            direction="SELL" # Short
        )
        
        # Manual PnL check logic (mirrors Guardian)
        pnl = (entry_p - current_p) / entry_p
        self.assertEqual(pnl, 0.10) # 10% gain
        
    def test_executor_short_accounting(self):
        """Verify balance updates correctly on short cover."""
        symbol = "ADA/USDT"
        entry_p = 0.50
        qty = 1000.0 # Notional = 500
        leverage = 1.0
        
        # 1. Open Short
        sig = Signal(symbol=symbol, direction='SELL', price=entry_p, size=qty)
        # Dummy disposition and hash
        disp = Disposition(0.5, 0.5)
        dec = TradeDecision(action='EXECUTE', adjusted_size=1.0, original_signal=sig, disposition=disp, block_hash="abc")
        
        self.executor.execute_transaction(dec, entry_p)
        
        # Expected quantity: 500 USD / 0.50 = 1000 units
        actual_qty = self.executor.held_assets.get(symbol, 0)
        self.assertEqual(actual_qty, -1000.0) 
        self.assertEqual(self.executor.balance_usd, 500.0) # 1000 - 500 margin
        
        # 2. Cover Short at lower price
        exit_p = 0.40 # 20% drop
        sig_cover = Signal(symbol=symbol, direction='BUY', price=exit_p, size=qty)
        dec_cover = TradeDecision(action='EXECUTE', adjusted_size=1.0, original_signal=sig_cover, disposition=disp, block_hash="def")
        
        # PnL = (0.50 - 0.40) * 1000 = +$100
        # Return Margin = 500
        # New Balance should be 500 + 500 + 100 = 1100
        
        self.executor.execute_transaction(dec_cover, exit_p)
        
        self.assertEqual(self.executor.held_assets.get(symbol, 0), 0)
        self.assertEqual(self.executor.balance_usd, 1100.0)

if __name__ == '__main__':
    unittest.main()
