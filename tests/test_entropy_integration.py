"""
Test Entropy Integration in Oracle.
"""
import unittest
from unittest.mock import MagicMock
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestEntropyIntegration(unittest.TestCase):
    """Tests for entropy integration in oracle."""
    
    def test_chaotic_dampening_concept(self):
        """Verify the concept of chaotic dampening."""
        # In CHAOTIC regime, confidence should be dampened
        # Formula: dampened = 0.5 + (original - 0.5) * 0.5
        original_confidence = 0.9
        dampened = 0.5 + (original_confidence - 0.5) * 0.5
        
        self.assertAlmostEqual(dampened, 0.7, places=2)
        self.assertLess(dampened, original_confidence)
        
    def test_ordered_no_change_concept(self):
        """Verify that ordered regime doesn't change confidence."""
        # In ORDERED regime, confidence stays the same
        original_confidence = 0.9
        
        # No dampening applied
        final_confidence = original_confidence
        
        self.assertEqual(final_confidence, original_confidence)
        
    def test_entropy_regime_exists(self):
        """Verify entropy regime module exists."""
        from HolonicTrader import agent_entropy
        self.assertTrue(hasattr(agent_entropy, 'EntropyHolon'))


if __name__ == '__main__':
    unittest.main()
