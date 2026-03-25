import unittest
from HolonicTrader.smce_regime_engine import SMCERegimeEngine

class TestSMCERegimeEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SMCERegimeEngine()

    def test_bullish_harvest(self):
        # Bullish structure, entropy <= 1.0 -> HARVEST
        regime = self.engine.classify(
            structure="BULLISH",
            entropy=0.8,
            liquidity_status="healthy",
            correlation_idx=0.5,
            drawdown_breach=False
        )
        self.assertEqual(regime, "HARVEST")

    def test_bearish_harvest(self):
        # Bearish structure, entropy <= 1.0 -> HARVEST
        regime = self.engine.classify(
            structure="BEARISH",
            entropy=0.8,
            liquidity_status="healthy",
            correlation_idx=0.5,
            drawdown_breach=False
        )
        self.assertEqual(regime, "HARVEST")

    def test_high_entropy_defensive(self):
        # High entropy (>1.5) -> DEFENSIVE
        regime = self.engine.classify(
            structure="BULLISH",
            entropy=1.6,
            liquidity_status="healthy",
            correlation_idx=0.5,
            drawdown_breach=False
        )
        self.assertEqual(regime, "DEFENSIVE")

    def test_drawdown_breach_defensive(self):
        # Drawdown breach -> DEFENSIVE
        regime = self.engine.classify(
            structure="BULLISH",
            entropy=1.1,
            liquidity_status="healthy",
            correlation_idx=0.5,
            drawdown_breach=True
        )
        self.assertEqual(regime, "DEFENSIVE")

    def test_low_liquidity_transition(self):
        # Poor liquidity -> TRANSITION
        regime = self.engine.classify(
            structure="BULLISH",
            entropy=1.1,
            liquidity_status="poor",
            correlation_idx=0.5,
            drawdown_breach=False
        )
        self.assertEqual(regime, "TRANSITION")

if __name__ == '__main__':
    unittest.main()
