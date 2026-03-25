import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from HolonicTrader.agent_executor import TradeSignal as Signal, Position


class TestShortingPnL(unittest.TestCase):
    """
    Tests short position PnL calculations using the current Position API.
    The executor now uses Position objects instead of raw dicts.
    """

    def test_short_pnl_calculation(self):
        """Verify that price drop results in positive PnL for short."""
        print("\n[Test] Short PnL Calculation")
        entry_p = 100.0
        current_p = 90.0  # 10% drop

        # Create a short position using the Position dataclass
        pos = Position(
            symbol="BTC/USDT",
            virt_key="BTC/USDT_DIRECTIONAL",
            direction="SELL",
            quantity=1.0,
            entry_price=entry_p,
            entry_timestamp="2026-01-01T00:00:00Z",
            leverage=1.0
        )

        # Verify PnL calculation via the Position.get_pnl_pct method
        pnl_pct = pos.get_pnl_pct(current_p)

        print(f"  Entry: ${entry_p}, Current: ${current_p}")
        print(f"  PnL%: {pnl_pct:.4f} (Expected: 0.10)")

        # Short PnL: (entry - current) / entry = (100 - 90) / 100 = 0.10
        self.assertAlmostEqual(pnl_pct, 0.10, places=4,
                               msg="Short PnL should be +10% when price drops 10%")

        # Verify is_short property
        self.assertTrue(pos.is_short, "Position should be identified as short")
        self.assertFalse(pos.is_long, "Position should NOT be identified as long")

    def test_short_pnl_negative(self):
        """Verify that price increase results in negative PnL for short."""
        print("\n[Test] Short PnL Negative (Price Up)")
        entry_p = 100.0
        current_p = 110.0  # 10% rise

        pos = Position(
            symbol="ETH/USDT",
            virt_key="ETH/USDT_DIRECTIONAL",
            direction="SELL",
            quantity=1.0,
            entry_price=entry_p,
            entry_timestamp="2026-01-01T00:00:00Z",
            leverage=1.0
        )

        pnl_pct = pos.get_pnl_pct(current_p)

        print(f"  Entry: ${entry_p}, Current: ${current_p}")
        print(f"  PnL%: {pnl_pct:.4f} (Expected: -0.10)")

        # Short PnL: (entry - current) / entry = (100 - 110) / 100 = -0.10
        self.assertAlmostEqual(pnl_pct, -0.10, places=4,
                               msg="Short PnL should be -10% when price rises 10%")

    def test_long_pnl_calculation(self):
        """Verify that price increase results in positive PnL for long."""
        print("\n[Test] Long PnL Calculation")
        entry_p = 100.0
        current_p = 115.0  # 15% rise

        pos = Position(
            symbol="BTC/USDT",
            virt_key="BTC/USDT_DIRECTIONAL",
            direction="BUY",
            quantity=1.0,
            entry_price=entry_p,
            entry_timestamp="2026-01-01T00:00:00Z",
            leverage=1.0
        )

        pnl_pct = pos.get_pnl_pct(current_p)

        print(f"  Entry: ${entry_p}, Current: ${current_p}")
        print(f"  PnL%: {pnl_pct:.4f} (Expected: 0.15)")

        self.assertAlmostEqual(pnl_pct, 0.15, places=4,
                               msg="Long PnL should be +15% when price rises 15%")
        self.assertTrue(pos.is_long, "Position should be identified as long")

if __name__ == '__main__':
    unittest.main()
