
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from HolonicTrader.agent_trader import TraderHolon
from HolonicTrader.agent_executor import TradeSignal
import config

class TestArbHunterSizing(unittest.TestCase):
    def setUp(self):
        self.trader = TraderHolon("TestNexus")
        self.trader.verbose_logging = True
        
        # Mock Sub-Holons
        self.mock_governor = MagicMock()
        self.mock_governor.last_ratchet_time = 0
        self.mock_executor = MagicMock()
        self.mock_arbitrage = MagicMock()
        
        self.trader.medic = None # completely disable medic for this test

        self.trader.sub_holons = {
            'governor': self.mock_governor,
            'executor': self.mock_executor,
            'arbitrage': self.mock_arbitrage
        }
        
        # Setup Mock Data
        self.mock_executor.latest_prices = {'BTC/USDT': 90000.0}
        
    def test_arb_sizing_fix(self):
        """Verify that ArbHunter requests safe size from Governor instead of hardcoded 1.0"""
        
        # 1. Setup Mock Arb Nugget
        nugget = {
            'symbol': 'BTC/USDT',
            'direction': 'BUY',
            'confidence': 0.9,
            'reason': 'TEST_ARB'
        }
        self.mock_arbitrage.scan_for_nuggets.return_value = [nugget]
        
        # 2. Setup Governor Response (Safe Size = 0.001)
        # calc_position_size returns (is_approved, likely_qty, leverage)
        self.mock_governor.calc_position_size.return_value = (True, 0.001, 5.0)
        
        # 3. Setup Executor to capture the call
        self.mock_executor.decide_trade.return_value = MagicMock(action='EXECUTE')
        
        # 4. Run Cycle (just the Arb part effectively)
        # We need to bypass other parts of run_cycle or just check the calls
        # Since run_cycle is complex, let's just inspect the mocks after running it
        # But run_cycle has many side effects.
        # We can temporarily patch other phases out or just run it and catch exceptions if any (not ideal)
        
        # Strategy: Mock out methods that might block or fail
        with patch.object(self.trader, 'perform_health_check'), \
             patch.object(self.trader, '_run_scout_cycle'), \
             patch.object(self.trader, '_scan_for_genome_updates'):
             
             # Mock Actuator inside executor to prevent "Blind Mode" skip
             self.mock_executor.actuator.get_equity.return_value = 1000.0
             self.mock_executor.actuator.get_account_balance.return_value = 1000.0
             
             # PATCH: Prevent Analysis Phase from crashing due to missing mocks
             # We set active_session_whitelist to empty so the analysis loop doesn't run
             self.trader.active_session_whitelist = [] 
             
             self.trader.run_cycle()
             
        # 5. Verification
        
        # A. Verify Governor was asked for size
        self.mock_governor.calc_position_size.assert_called_once()
        args, kwargs = self.mock_governor.calc_position_size.call_args
        self.assertEqual(kwargs['symbol'], 'BTC/USDT')
        self.assertEqual(kwargs['asset_price'], 90000.0)
        self.assertTrue(kwargs['whale_confirmed']) # We forced this to True
        
        # B. Verify Executor received Signal with SAFE SIZE (0.001), not 1.0
        self.mock_executor.decide_trade.assert_called_once()
        call_args = self.mock_executor.decide_trade.call_args[0]
        signal = call_args[0]
        
        print(f"\nCaptured Signal Size: {signal.size}")
        self.assertAlmostEqual(signal.size, 0.001)
        self.assertNotEqual(signal.size, 1.0, "Bug Regression: Size is still 1.0!")
        
    def test_arb_sizing_veto(self):
        """Verify that ArbHunter skips trade if Governor vetoes sizing"""
         # 1. Setup Mock Arb Nugget
        nugget = {
            'symbol': 'ETH/USDT',
            'direction': 'BUY',
            'confidence': 0.8,
            'reason': 'TEST_ARB_VETO'
        }
        self.mock_arbitrage.scan_for_nuggets.return_value = [nugget]
        self.mock_executor.latest_prices = {'ETH/USDT': 3000.0}
        
        # 2. Setup Governor Response (VETO)
        self.mock_governor.calc_position_size.return_value = (False, 0.0, 0.0)
        
        # 3. Run
        with patch.object(self.trader, 'perform_health_check'), \
             patch.object(self.trader, '_run_scout_cycle'):
             
             self.mock_executor.actuator.get_equity.return_value = 1000.0
             self.mock_executor.actuator.get_account_balance.return_value = 1000.0
             self.trader.active_session_whitelist = [] 
             
             self.trader.run_cycle()
             
        # 4. Verify Executor was NOT called
        self.mock_executor.decide_trade.assert_not_called()
        print("\nconfirmed: Trade skipped after Governor Veto.")

if __name__ == '__main__':
    unittest.main()
