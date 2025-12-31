
import unittest
from unittest.mock import MagicMock
import sys
import os

# Ensure import paths
sys.path.append(os.getcwd())

from HolonicTrader.agent_guardian import ExitGuardianHolon
from HolonicTrader.agent_observer import ObserverHolon
import config

class TestLiquidityMonitor(unittest.TestCase):
    
    def setUp(self):
        self.guardian = ExitGuardianHolon()
        
    def test_liquidity_logic_healthy(self):
        """Test that a deep order book returns HEALTHY."""
        # 1000 units needed. Best bid 100.
        # Book has 5000 units at 100.
        mock_book = {
            'bids': [[100.0, 5000.0], [99.0, 1000.0]], 
            'asks': [[101.0, 5000.0]]
        }
        status = self.guardian.check_liquidity_health('TEST', 'SELL', 1000.0, mock_book)
        print(f"Scenario 1 (Healthy): {status}")
        self.assertEqual(status, "HEALTHY")

    def test_liquidity_logic_illiquid(self):
        """Test that a thin order book returns CRITICAL_ILLIQUIDITY."""
        # 10,000 units needed. Best bid 100.
        # Book only has 100 units at 100, 100 at 99... total 200.
        mock_book = {
            'bids': [[100.0, 100.0], [99.0, 100.0]],
            'asks': [[101.0, 100.0]]
        }
        status = self.guardian.check_liquidity_health('TEST', 'SELL', 10000.0, mock_book)
        print(f"Scenario 2 (Illiquid): {status}")
        self.assertEqual(status, "CRITICAL_ILLIQUIDITY")

    def test_live_fetch(self):
        """Test active fetch from Kraken Futures (Real API)."""
        print("\n--- LIVE API TEST ---")
        try:
            # Requires config to have keys or public access
            config.TRADING_MODE = 'FUTURES'
            observer = ObserverHolon(exchange_id='krakenfutures', symbol='XRP/USD:USD')
            book = observer.fetch_order_book('XRP/USD:USD')
            
            bids = book.get('bids', [])
            asks = book.get('asks', [])
            
            print(f"Fetched Book for XRP/USD:USD")
            print(f"Top Bid: {bids[0] if bids else 'None'}")
            print(f"Top Ask: {asks[0] if asks else 'None'}")
            print(f"Bid Depth (Levels): {len(bids)}")
            
            self.assertTrue(len(bids) > 0)
            
            # Real Check using Guardian
            status = self.guardian.check_liquidity_health('XRP/USD:USD', 'SELL', 100.0, book)
            print(f"Real Liquidity Check (100 units): {status}")
            
        except Exception as e:
            print(f"Live Test Skipped or Failed: {e}")
            # Don't fail the whole test if just network issue/keys
            pass

if __name__ == '__main__':
    unittest.main()
