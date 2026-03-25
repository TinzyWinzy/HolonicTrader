import unittest
import time
from HolonicTrader.smce_capital_doctrine import SMCECapitalDoctrine

class TestSMCECapitalDoctrine(unittest.TestCase):
    def setUp(self):
        self.doctrine = SMCECapitalDoctrine()

    def test_exposure_limit_block(self):
        # SMALL tier limit is 30% total exposure ($120 for $400 equity)
        equity = 400.0
        # Already have 25% exposure ($100 margin)
        positions = {
            "BTC/USDT": MagicPosition(quantity=0.002, entry_price=50000.0, leverage=1.0) # $100 margin
        }
        
        # Propose adding $50 more margin
        allowed, reason, max_lev = self.doctrine.check_trade(
            symbol="ETH/USDT",
            direction="BUY",
            proposed_notional=50.0,
            proposed_leverage=1.0,
            equity=equity,
            smce_regime="HARVEST",
            current_positions=positions
        )
        # Total = 100 + 50 = 150 (37.5%) > 30% limit ($120)
        self.assertFalse(allowed)
        self.assertIn("Total exposure", reason)

    def test_leverage_capping(self):
        # NORMAL leverage for MEDIUM Harvest is 4.0
        equity = 1000.0
        allowed, reason, max_lev = self.doctrine.check_trade(
            symbol="ETH/USDT",
            direction="BUY",
            proposed_notional=100.0,
            proposed_leverage=5.0, # Attempt 5x
            equity=equity,
            smce_regime="HARVEST",
            current_positions={}
        )
        self.assertTrue(allowed)
        self.assertEqual(max_lev, 4.0)

    def test_defensive_cooldown_block(self):
        # Trigger defensive
        self.doctrine._trigger_defensive("Test Breach", 1000.0)
        
        allowed, reason, max_lev = self.doctrine.check_trade(
            symbol="BTC/USDT",
            direction="BUY",
            proposed_notional=100.0,
            proposed_leverage=1.0,
            equity=1000.0,
            smce_regime="HARVEST",
            current_positions={}
        )
        self.assertFalse(allowed)
        self.assertIn("DEFENSIVE cooldown", reason)

class MagicPosition:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

if __name__ == '__main__':
    unittest.main()
