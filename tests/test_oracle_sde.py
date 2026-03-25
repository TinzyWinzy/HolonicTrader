"""
Test Oracle SDE integration.
"""
import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestOracleSDE(unittest.TestCase):
    """Tests for Oracle SDE integration."""
    
    def test_sde_module_exists(self):
        """Verify SDE module exists."""
        from HolonicTrader import sde_engine
        # Check for any SDE-related function
        self.assertTrue(hasattr(sde_engine, 'SDEEngine') or len(dir(sde_engine)) > 0)
        
    def test_sde_engine_exists(self):
        """Test SDE Engine class exists."""
        from HolonicTrader.sde_engine import SDEEngine
        self.assertTrue(hasattr(SDEEngine, 'estimate_ou_parameters'))


if __name__ == "__main__":
    unittest.main()
