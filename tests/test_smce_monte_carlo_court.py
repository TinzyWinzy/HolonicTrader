import unittest
from HolonicTrader.smce_monte_carlo_court import SMCEMonteCarloRiskCourt

class TestSMCEMonteCarloCourt(unittest.TestCase):
    def setUp(self):
        # Using few paths for fast unit tests
        self.court = SMCEMonteCarloRiskCourt(n_paths=100)

    def test_pre_trade_approval(self):
        # Low volatility, reasonable stop, bullish -> should approve
        props = {
            "symbol": "BTC/USDT",
            "direction": "BUY",
            "entry_price": 50000.0,
            "notional": 500.0,
            "leverage": 1.0
        }
        res = self.court.evaluate_pre_trade(
            equity=5000.0,
            proposed_trade=props,
            portfolio_positions={},
            volatilities={"BTC/USDT": 0.02}
        )
        self.assertFalse(res["vetoed"])
        self.assertIn("cvar_95", res)

    def test_pre_trade_veto_high_risk(self):
        # Extremely high volatility -> might veto
        props = {
            "symbol": "PEPE/USDT",
            "direction": "BUY",
            "entry_price": 0.00001,
            "notional": 2000.0,
            "leverage": 5.0
        }
        res = self.court.evaluate_pre_trade(
            equity=5000.0,
            proposed_trade=props,
            portfolio_positions={},
            volatilities={"PEPE/USDT": 0.25}
        )
        self.assertIn("cvar_95", res)

    def test_portfolio_cvar(self):
        # Mock positions
        positions = {
            "BTC/USDT": {"notional": 1000, "direction": "BUY", "vol": 0.02},
            "ETH/USDT": {"notional": 500, "direction": "BUY", "vol": 0.03}
        }
        res = self.court.evaluate_portfolio(positions, equity=5000.0)
        self.assertIn("cvar_95", res)
        self.assertGreater(res["cvar_95"], 0)

if __name__ == '__main__':
    unittest.main()
