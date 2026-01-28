
import sys
import os
import unittest
from unittest.mock import MagicMock

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# MOCK DEPENDENCIES BEFORE IMPORT
sys.modules['tensorflow'] = MagicMock()
sys.modules['HolonicTrader.agent_ppo'] = MagicMock()

import config
from HolonicTrader.agent_governor import GovernorHolon

class TestRiskEngineOverhaul(unittest.TestCase):

    def setUp(self):
        # Initialize Governor with known state
        self.gov = GovernorHolon(name="TestGov", initial_balance=100.0)
        self.gov.regime_controller = MagicMock()
        self.gov.regime_controller.get_current_regime.return_value = 'SMALL' # 1.0x Cap
        
        # Mock Market Data
        self.gov.latest_prices = {
            'BTC/USDT': 100000.0,
            'ETH/USDT': 3000.0
        }

    def test_cross_margin_calculation(self):
        """Verify _calculate_portfolio_state aggregates margin correctly."""
        print("\n[Test] Cross-Margin Calculation")
        
        # Add a mock position: 0.0001 BTC (~$10)
        # Entry: 100k, Current: 100k. No PnL.
        self.gov.positions['BTC/USDT'] = {
            'quantity': 0.0001,
            'entry_price': 100000.0,
            'direction': 'BUY'
        }
        
        state = self.gov._calculate_portfolio_state()
        
        # Expected:
        # Equity = 100.0
        # Used Margin = (0.0001 * 100000) / 1.0 (Lev) = $10.0
        # Free Margin = 100 - 10 = 90.0
        
        print(f"State: {state}")
        self.assertAlmostEqual(state['equity'], 100.0)
        self.assertAlmostEqual(state['used_margin'], 10.0)
        self.assertAlmostEqual(state['free_margin'], 90.0)

    def test_solvency_veto(self):
        """Verify check_solvency rejects trades that breach limits."""
        print("\n[Test] Solvency Veto")
        
        # Current Balance $100. 
        # Try to open $90 position.
        # Future Used = $90. Equity = $100. Util = 90%.
        # Should Fail (>80% Limit).
        
        trade = {'size': 0.0009, 'price': 100000.0} # $90
        
        result = self.gov.check_solvency(trade)
        print(f"Trade $90 Result: {result} (Expected False)")
        self.assertFalse(result)
        
        # Try $50 position. Util 50%. Safe.
        trade_safe = {'size': 0.0005, 'price': 100000.0} # $50
        result_safe = self.gov.check_solvency(trade_safe)
        print(f"Trade $50 Result: {result_safe} (Expected True)")
        self.assertTrue(result_safe)

    def test_nano_position_sizing(self):
        """Verify new config function logic."""
        print("\n[Test] Nano Config Sizing")
        # $50 Balance. 5% = $2.50.
        # Min Trade $3.0.
        # Should Bump to $3.0.
        
        # Mock config values just in case
        # Assuming config.py is loaded correctly
        
        res = config.calculate_nano_position(50.0, 'XRP/USDT', 1.0) # Min XRP might be different
        # Let's use DOGE/USDT where min is known from my previous read (10.0 quantity, tick 0.01?)
        # Actually I saw MIN_TRADE_QTY keys: 'DOGE': 10.0.
        # If Price is $0.50, Min Notional $5.0.
        
        # If I use DOGE at $0.50. Min $5.
        # Balance $50. 5% = $2.5.
        # 10% Check = $5.0. 
        # $2.5 < $5 (Min). $5 <= $5 (Max). 
        # Should Allow $5.0 trade.
        
        res = config.calculate_nano_position(50.0, 'DOGE/USDT', 0.50)
        print(f"Nano Result: {res}")
        self.assertAlmostEqual(res['notional'], 5.0)
        self.assertEqual(res['leverage'], 1.0)

if __name__ == '__main__':
    unittest.main()
