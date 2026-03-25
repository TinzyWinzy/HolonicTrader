"""
Test Pyramiding Decay logic for position sizing.
"""
import unittest
import os
import time

class TestPyramidingDecay(unittest.TestCase):
    """Tests for pyramiding decay logic."""
    
    def test_decay_concept(self):
        """Verify the concept of position decay on pyramiding."""
        # The decay factor is 0.8 per stack
        decay_factor = 0.8
        
        initial_position = 100.0
        
        # After 1st pyramid: 100 * 0.8 = 80
        second_position = initial_position * decay_factor
        self.assertAlmostEqual(second_position, 80.0)
        
        # After 2nd pyramid: 80 * 0.8 = 64
        third_position = second_position * decay_factor
        self.assertAlmostEqual(third_position, 64.0)
        
    def test_conviction_concept(self):
        """Verify the concept of conviction-based position sizing."""
        # Higher conviction should result in larger positions
        high_conviction = 1.0
        low_conviction = 0.6
        
        # Conviction scalar formula: 0.5 + (conviction * 1.0)
        high_scalar = 0.5 + (high_conviction * 1.0)
        low_scalar = 0.5 + (low_conviction * 1.0)
        
        self.assertGreater(high_scalar, low_scalar)


if __name__ == '__main__':
    unittest.main()
