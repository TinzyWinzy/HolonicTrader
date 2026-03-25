import unittest
from HolonicTrader.smce_probability_engine import ProbabilityStackingEngine

class TestSMCEProbabilityEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ProbabilityStackingEngine()

    def test_strong_bullish_score(self):
        # Bullish structure + low entropy + positive momentum -> high score
        trade = {"direction": "BUY", "symbol": "BTC/USDT"}
        portfolio = {"equity": 5000.0, "cluster_exposure": 0.0, "cvar_95": 0.01}
        context = {
            "structure": "BULLISH",
            "momentum_aligned": True,
            "liquidity_status": "healthy",
            "entropy": 0.8,
            "correlation_idx": 0.3
        }
        res = self.engine.score_trade(trade, portfolio, context, "HARVEST")
        
        self.assertGreaterEqual(res["score"], 6) # At least structure(2) + mom(1) + liq(1) + entropy(1) + corr(1) + cvar(2) = 8
        self.assertTrue(res["eligible"])

    def test_high_entropy_penalty(self):
        # High entropy should penalize the score
        trade = {"direction": "BUY", "symbol": "BTC/USDT"}
        portfolio = {"equity": 5000.0, "cluster_exposure": 0.0, "cvar_95": 0.01}
        
        context_low = {
            "structure": "NEUTRAL",
            "momentum_aligned": True,
            "liquidity_status": "healthy",
            "entropy": 0.8,
            "correlation_idx": 0.3
        }
        score_low = self.engine.score_trade(trade, portfolio, context_low, "HARVEST")["score"]
        
        context_high = context_low.copy()
        context_high["entropy"] = 1.2 # > 1.0 for BUY
        score_high = self.engine.score_trade(trade, portfolio, context_high, "HARVEST")["score"]
        
        self.assertLess(score_high, score_low)

    def test_poor_liquidity_disqualify(self):
        trade = {"direction": "BUY", "symbol": "BTC/USDT"}
        portfolio = {"equity": 5000.0, "cluster_exposure": 0.0, "cvar_95": 0.01}
        context = {
            "structure": "BULLISH",
            "momentum_aligned": True,
            "liquidity_status": "critical",
            "entropy": 0.8,
            "correlation_idx": 0.3
        }
        res = self.engine.score_trade(trade, portfolio, context, "HARVEST")
        self.assertFalse(res["eligible"])
        self.assertIn("Liquidity CRITICAL", res["block_reason"])

if __name__ == '__main__':
    unittest.main()
