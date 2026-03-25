import unittest
import sys
import os
from unittest.mock import MagicMock

# --- MOCK HEAVY DEPENDENCIES BEFORE IMPORTS ---
sys.modules['tensorflow'] = MagicMock()
sys.modules['keras'] = MagicMock()
sys.modules['HolonicTrader.agent_ppo'] = MagicMock()

# Ensure we mock PPOHolon specifically if needed
mock_ppo = MagicMock()
sys.modules['HolonicTrader.agent_ppo'].PPOHolon = mock_ppo

# Mock rich to avoid console issues
sys.modules['rich'] = MagicMock()
sys.modules['rich.console'] = MagicMock()
# ----------------------------------------------

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # We might need to reload if standard import failed previously in same process (not applicable to new process run)
    from HolonicTrader.agent_governor import GovernorHolon
except ImportError:
    # Fallback if run from different context, try to adjust properly
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../HolonicTrader')))
    from HolonicTrader.agent_governor import GovernorHolon

class MockDbManager:
    def get_win_rate(self): return 0.5

class TestPaxgBan(unittest.TestCase):
    def setUp(self):
        self.governor = GovernorHolon(name="TestGov", initial_balance=2000.0, db_manager=MockDbManager())
        self.governor.DEBUG = True
        
        # Manually ensure blacklist is initialized if it wasn't in __init__ (Simulating current state vs future state)
        if not hasattr(self.governor, 'blacklist'):
            self.governor.blacklist = set()

    def test_paxg_allowed_high_equity(self):
        """Test that PAXG is allowed when equity is > $1000"""
        self.governor.set_live_balance(1500.0, 1500.0)
        
        # We need to simulate a price check.
        # is_trade_allowed(symbol, asset_price, silent, is_whale, funding_yield, is_arb)
        allowed = self.governor.is_trade_allowed("PAXG/USDT", 2000.0)
        # Expect TRUE currently (before fix) and TRUE after fix
        self.assertTrue(allowed, "PAXG/USDT should be allowed with > $1000 equity")

    def test_paxg_banned_low_equity(self):
        """Test that PAXG is banned when equity is < $1000"""
        self.governor.set_live_balance(900.0, 900.0)
        
        allowed = self.governor.is_trade_allowed("PAXG/USDT", 2000.0)
        
        # Before fix: This will likely be TRUE.
        # After fix: This MUST be FALSE.
        # For now, we assert FALSE to confirm failure.
        self.assertFalse(allowed, "PAXG/USDT should be banned with < $1000 equity")

    def test_other_assets_unaffected(self):
        """Test that other assets are not affected by equity check"""
        self.governor.set_live_balance(900.0, 900.0)
        
        # Mock other dependencies if is_trade_allowed fails on other checks
        allowed = self.governor.is_trade_allowed("BTC/USDT", 50000.0)
        self.assertTrue(allowed, "BTC/USDT should be allowed regardless of equity logic for PAXG")

    def test_recovery_unlock(self):
        """Test that PAXG is unbanned if equity recovers"""
        self.governor.set_live_balance(900.0, 900.0)
        
        # Allow fail here if we implemented it halfway
        # ...
        
        # Simulate recovery
        self.governor.set_live_balance(1200.0, 1200.0)
        allowed_high = self.governor.is_trade_allowed("PAXG/USDT", 2000.0)
        
        self.assertTrue(allowed_high, "PAXG/USDT should be unlocked after recovery")

if __name__ == '__main__':
    unittest.main()
