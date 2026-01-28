
import unittest
import sys
import os

# Go up one level to import HolonicTrader modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config

class TestRiskEngine(unittest.TestCase):

    def setUp(self):
        # Reset config values to known states if needed
        pass

    def test_leverage_is_hardcoded_to_one(self):
        """Verify that all regime leverage caps are set to 1.0"""
        for regime, settings in config.REGIME_PERMISSIONS.items():
            self.assertEqual(settings['max_leverage'], 1.0, f"{regime} leverage should be 1.0")
        
        self.assertEqual(config.SATELLITE_LEVERAGE, 1.0, "Satellite leverage should be 1.0")
        self.assertEqual(config.SCAVENGER_LEVERAGE, 1.0, "Scavenger leverage should be 1.0")
        self.assertEqual(config.PREDATOR_LEVERAGE, 1.0, "Predator leverage should be 1.0")
        self.assertEqual(config.MICRO_HARD_LEVERAGE_LIMIT, 1.0, "Micro Hard Limit should be 1.0")
        self.assertEqual(config.NANO_MAX_LEVERAGE, 1.0, "Nano Max Limit should be 1.0")

    def test_calculate_nano_position_sufficient_capital(self):
        """Test calculation with enough capital ($100 balance, $3 min trade)"""
        balance = 100.0
        symbol = 'DOGE/USDT' # Min qty 10 (~$5)
        price = 0.50 # 10 * 0.5 = $5 min trade
        
        # 5% of 100 = $5. This matches min trade exactly.
        result = config.calculate_nano_position(balance, symbol, price)
        
        self.assertEqual(result['leverage'], 1.0)
        self.assertAlmostEqual(result['notional'], 5.0)
        self.assertAlmostEqual(result['quantity'], 10.0)

    def test_calculate_nano_position_small_capital_bump(self):
        """Test calculation where risk allocation < min trade but allowed (within 10%)"""
        balance = 60.0 # 5% = $3.0
        symbol = 'DOGE/USDT'
        price = 0.50 # Min trade $5
        
        # Risk alloc $3 < Min trade $5.
        # Check buffer: 10% of $60 = $6.0.
        # $5 < $6, so it should BUMP to $5.
        
        result = config.calculate_nano_position(balance, symbol, price)
        
        self.assertEqual(result['leverage'], 1.0)
        self.assertAlmostEqual(result['notional'], 5.0) # Bumped to min
        self.assertAlmostEqual(result['margin'], 5.0)

    def test_calculate_nano_position_insufficient_capital(self):
        """Test calculation where capital is too low for min trade"""
        balance = 40.0 # 5% = $2.0. 10% = $4.0
        symbol = 'DOGE/USDT'
        price = 0.50 # Min trade $5
        
        # Min trade $5 > 10% of Balance ($4). Should reject.
        
        result = config.calculate_nano_position(balance, symbol, price)
        
        self.assertEqual(result['quantity'], 0.0)
        self.assertIn('error', result)

if __name__ == '__main__':
    unittest.main()
