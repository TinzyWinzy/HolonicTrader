
import unittest
import os
import time
from HolonicTrader.agent_governor import GovernorHolon
from database_manager import DatabaseManager

class TestPyramidingDecay(unittest.TestCase):
    def setUp(self):
        self.db_file = "test_decay.db"
        if os.path.exists(self.db_file): os.remove(self.db_file)
        self.db = DatabaseManager(self.db_file)
        self.governor = GovernorHolon(db_manager=self.db)
        self.governor.update_balance(10000.0)
        import config
        config.GOVERNOR_COOLDOWN_SECONDS = 0
        config.MAX_RISK_PCT = 0.5 # 50% risk allowed for test logic
        config.PRINCIPAL = 1000.0 # High buffer
        
    def tearDown(self):
        if os.path.exists(self.db_file):
            try:
                os.remove(self.db_file)
            except:
                pass

    def test_decay_scaling(self):
        """Verify that subsequent stacks are smaller than initial entry."""
        symbol = "BTC/USDT"
        price = 60000.0
        atr = 1000.0
        atr_ref = 1000.0
        
        # 1st Stack
        app1, qty1, lev1 = self.governor.calc_position_size(symbol, price, atr, atr_ref, conviction=1.0)
        self.assertTrue(app1)
        self.governor.receive_message(None, {'type': 'POSITION_FILLED', 'symbol': symbol, 'direction': 'LONG', 'price': price, 'quantity': qty1})
        
        # Manually ensure time moves forward slightly if needed, but stack_count is primary
        
        # 2nd Stack (Price + 1% to avoid proximity)
        price2 = price * 1.01
        app2, qty2, lev2 = self.governor.calc_position_size(symbol, price2, atr, atr_ref, conviction=1.0)
        self.assertTrue(app2)
        self.governor.receive_message(None, {'type': 'POSITION_FILLED', 'symbol': symbol, 'direction': 'LONG', 'price': price2, 'quantity': qty2})

        # 3rd Stack (Price + 2%)
        price3 = price * 1.02
        app3, qty3, lev3 = self.governor.calc_position_size(symbol, price3, atr, atr_ref, conviction=1.0)
        self.assertTrue(app3)
        
        print(f"Notionals: 1:{qty1*price:.2f}, 2:{qty2*price2:.2f}, 3:{qty3*price3:.2f}")
        
        # Verify decay on NOTIONAL: 1 > 2 > 3
        # Expected decay factor is 0.8 per stack
        self.assertAlmostEqual(qty2 * price2, (qty1 * price) * 0.8, delta=1.0)
        self.assertAlmostEqual(qty3 * price3, (qty1 * price) * 0.64, delta=1.0)
        self.assertTrue(qty1 > qty2)
        self.assertTrue(qty2 > qty3)
        
    def test_conviction_scaling(self):
        """Verify that higher conviction results in larger positions."""
        symbol = "ETH/USDT"
        price = 3000.0
        atr = 50.0
        atr_ref = 50.0
        
        # High Conviction (1.0) -> conv_scalar = 1.5
        app_h, qty_h, lev_h = self.governor.calc_position_size(symbol, price, atr, atr_ref, conviction=1.0)
        
        # Low Conviction (0.6) -> conv_scalar = 0.5 + (0.1 * 2) = 0.7
        app_l, qty_l, lev_l = self.governor.calc_position_size(symbol, price, atr, atr_ref, conviction=0.6)
        
        print(f"Conviction Sizes: High(1.0):{qty_h:.4f}, Low(0.6):{qty_l:.4f}")
        self.assertTrue(qty_h > qty_l)

if __name__ == '__main__':
    unittest.main()
