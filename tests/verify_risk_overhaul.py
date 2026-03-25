
import sys
import os
import unittest
from unittest.mock import MagicMock
from dataclasses import dataclass

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# MOCK DEPENDENCIES BEFORE IMPORT
sys.modules['tensorflow'] = MagicMock()
sys.modules['HolonicTrader.agent_ppo'] = MagicMock()

import config
from HolonicTrader.agent_governor import GovernorHolon


@dataclass
class MockPosition:
    """Mirrors the Position dataclass from agent_executor."""
    symbol: str
    quantity: float
    entry_price: float
    direction: str
    is_long: bool = True
    is_short: bool = False


class TestRiskEngineOverhaul(unittest.TestCase):

    def setUp(self):
        # Initialize Governor with known state
        self.gov = GovernorHolon(name="TestGov", initial_balance=100.0)
        self.gov.regime_controller = MagicMock()
        self.gov.regime_controller.get_current_regime.return_value = 'SMALL' # 1.0x Cap

        # Mock Executor with Position dict (current API)
        self.gov.executor = MagicMock()
        self.gov.executor.positions = {}

        # Mock Market Data
        self.gov.latest_prices = {
            'BTC/USDT': 100000.0,
            'ETH/USDT': 3000.0
        }

    def test_cross_margin_calculation(self):
        """Verify _calculate_portfolio_state aggregates margin correctly."""
        print("\n[Test] Cross-Margin Calculation")

        # Add a mock position: 0.0001 BTC (~$10 notional)
        mock_pos = MockPosition(
            symbol='BTC/USDT',
            quantity=0.0001,
            entry_price=100000.0,
            direction='BUY',
            is_long=True,
            is_short=False
        )
        self.gov.executor.positions = {'BTC/USDT_DIRECTIONAL': mock_pos}

        state = self.gov._calculate_portfolio_state()

        # Expected (SMALL regime = 5x leverage):
        # Equity = 100.0 (no unrealized PnL since mark == entry)
        # Used Margin = (0.0001 * 100000) / 5.0 (Lev) = $2.0
        # Free Margin = 100 - 2 = 98.0
        print(f"State: {state}")
        self.assertAlmostEqual(state['equity'], 100.0)
        self.assertAlmostEqual(state['used_margin'], 2.0)
        self.assertAlmostEqual(state['free_margin'], 98.0)

    def test_solvency_safe_trade(self):
        """Verify check_solvency allows safe trades."""
        print("\n[Test] Solvency Safe Trade")

        # $50 position. Balance $100. Util 50%. Safe.
        trade_safe = {'size': 0.0005, 'price': 100000.0, 'symbol': 'BTC/USDT', 'direction': 'BUY'}
        result_safe = self.gov.check_solvency(trade_safe)
        print(f"Trade $50 Result: {result_safe} (Expected True)")
        self.assertTrue(result_safe)

    def test_nano_position_sizing(self):
        """Verify new config function logic."""
        print("\n[Test] Nano Config Sizing")
        # $50 Balance. DOGE at $0.50. Min $5.
        # 5% of $50 = $2.5. Below min -> bumped to $5.0.
        res = config.calculate_nano_position(50.0, 'DOGE/USDT', 0.50)
        print(f"Nano Result: {res}")
        self.assertAlmostEqual(res['notional'], 5.0)
        self.assertEqual(res['leverage'], 1.0)

if __name__ == '__main__':
    unittest.main()
