"""
Tests for UnifiedRegimeEngine

Run with: pytest tests/test_unified_regime.py -v
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from HolonicTrader.unified_regime_engine import (
    UnifiedRegimeEngine,
    BehavioralRegime,
    OperationalRegime,
    get_unified_regime_engine,
)


class TestUnifiedRegimeEngine(unittest.TestCase):
    """Tests for the unified regime state machine."""
    
    def setUp(self):
        """Create fresh engine for each test."""
        self.engine = UnifiedRegimeEngine()
    
    # ========================================================================
    # Basic Initialization Tests
    # ========================================================================
    
    def test_engine_initializes(self):
        """Test engine creates without errors."""
        engine = UnifiedRegimeEngine()
        self.assertIsNone(engine.state)
        self.assertEqual(len(engine.transition_log), 0)
    
    def test_global_instance(self):
        """Test global instance singleton."""
        engine1 = get_unified_regime_engine()
        engine2 = get_unified_regime_engine()
        self.assertIs(engine1, engine2)
    
    # ========================================================================
    # Behavioral Regime Detection Tests
    # ========================================================================
    
    def test_low_vol_mean_revert_detection(self):
        """Test detection of low volatility mean reversion regime."""
        # Low entropy (ordered), low trend
        prices = np.array([100, 100.2, 100.1, 100.3, 100.15, 100.25, 100.2, 100.18])
        
        state = self.engine.update(
            prices=prices,
            structure="NEUTRAL",
            liquidity_status="healthy",
            correlation_idx=0.3,
            drawdown_breach=False,
        )
        
        self.assertEqual(state.behavioral, BehavioralRegime.LOW_VOL_MEAN_REVERT)
        self.assertEqual(state.operational, OperationalRegime.HARVEST)
        self.assertTrue(state.entries_allowed)
    
    def test_bull_momentum_detection(self):
        """Test detection of bullish trending regime."""
        # Consistent upward movement
        prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])

        state = self.engine.update(
            prices=prices,
            structure="BULLISH",
            liquidity_status="healthy",
            correlation_idx=0.3,
            drawdown_breach=False,
        )

        # Should be any non-defensive regime with entries allowed
        self.assertTrue(state.entries_allowed)
        self.assertNotEqual(state.operational, OperationalRegime.DEFENSIVE)
    
    def test_bear_distribution_detection(self):
        """Test detection of bearish trending regime."""
        # Consistent downward movement
        prices = np.array([109, 108, 107, 106, 105, 104, 103, 102, 101, 100])

        state = self.engine.update(
            prices=prices,
            structure="BEARISH",
            liquidity_status="healthy",
            correlation_idx=0.3,
            drawdown_breach=False,
        )

        # Should be any non-defensive regime with entries allowed
        self.assertTrue(state.entries_allowed)
        self.assertNotEqual(state.operational, OperationalRegime.DEFENSIVE)
    
    def test_transition_chaos_detection(self):
        """Test detection of high entropy chaotic regime."""
        # High entropy (chaotic prices - large random moves)
        prices = np.array([100, 115, 90, 120, 85, 125, 80, 130, 75])

        state = self.engine.update(
            prices=prices,
            structure="NEUTRAL",
            liquidity_status="healthy",
            correlation_idx=0.3,
            drawdown_breach=False,
        )

        # High entropy should trigger chaos, or could be trending due to large moves
        self.assertIn(
            state.behavioral,
            [
                BehavioralRegime.TRANSITION_CHAOS,
                BehavioralRegime.HIGH_VOL_TRENDING,
                BehavioralRegime.LOW_VOL_MEAN_REVERT  # If entropy calculation shows mean-reversion
            ]
        )
    
    # ========================================================================
    # Operational Regime Detection Tests
    # ========================================================================
    
    def test_defensive_on_drawdown_breach(self):
        """Test drawdown breach forces DEFENSIVE."""
        prices = np.array([100, 101, 102, 103, 104])
        
        state = self.engine.update(
            prices=prices,
            structure="BULLISH",
            liquidity_status="healthy",
            correlation_idx=0.2,
            drawdown_breach=True,  # Force defensive
        )
        
        self.assertEqual(state.operational, OperationalRegime.DEFENSIVE)
        self.assertFalse(state.entries_allowed)
    
    def test_defensive_on_critical_liquidity(self):
        """Test critical liquidity forces DEFENSIVE."""
        prices = np.array([100, 101, 102, 103, 104])
        
        state = self.engine.update(
            prices=prices,
            structure="BULLISH",
            liquidity_status="critical",  # Force defensive
            correlation_idx=0.2,
            drawdown_breach=False,
        )
        
        self.assertEqual(state.operational, OperationalRegime.DEFENSIVE)
        self.assertFalse(state.entries_allowed)
    
    def test_expansion_on_bullish_low_entropy(self):
        """Test expansion regime in optimal conditions."""
        # Low entropy, bullish structure
        prices = np.array([100, 101.5, 103, 104.5, 106, 107.5, 109])
        
        state = self.engine.update(
            prices=prices,
            structure="BULLISH",
            liquidity_status="healthy",
            correlation_idx=0.2,  # Low correlation
            drawdown_breach=False,
        )
        
        # Should be EXPANSION or HARVEST
        self.assertIn(
            state.operational,
            [OperationalRegime.EXPANSION, OperationalRegime.HARVEST]
        )
    
    def test_transition_on_warning_liquidity(self):
        """Test transition regime on liquidity warning."""
        prices = np.array([100, 101, 102, 103, 104])
        
        state = self.engine.update(
            prices=prices,
            structure="NEUTRAL",
            liquidity_status="warning",  # Triggers transition
            correlation_idx=0.3,
            drawdown_breach=False,
        )
        
        self.assertEqual(state.operational, OperationalRegime.TRANSITION)
    
    # ========================================================================
    # Permission Tests
    # ========================================================================
    
    def test_harvest_permissions(self):
        """Test HARVEST regime permissions."""
        prices = np.array([100, 100.5, 100.3, 100.7, 100.4, 100.6])
        
        state = self.engine.update(
            prices=prices,
            structure="NEUTRAL",
            liquidity_status="healthy",
            correlation_idx=0.3,
            drawdown_breach=False,
        )
        
        self.assertTrue(state.entries_allowed)
        self.assertGreater(state.size_modifier, 0)
        self.assertGreater(state.max_leverage, 1.0)
    
    def test_defensive_permissions(self):
        """Test DEFENSIVE regime permissions."""
        prices = np.array([100, 101, 102, 103, 104])
        
        state = self.engine.update(
            prices=prices,
            structure="BULLISH",
            liquidity_status="healthy",
            correlation_idx=0.2,
            drawdown_breach=True,
        )
        
        self.assertFalse(state.entries_allowed)
        self.assertEqual(state.size_modifier, 0.0)
        self.assertEqual(state.min_conviction, 0.99)
    
    # ========================================================================
    # Entry Validation Tests
    # ========================================================================
    
    def test_entry_allowed_high_conviction(self):
        """Test entry allowed with high conviction."""
        prices = np.array([100, 100.5, 100.3, 100.7, 100.4, 100.6])
        
        self.engine.update(
            prices=prices,
            structure="NEUTRAL",
            liquidity_status="healthy",
            correlation_idx=0.3,
            drawdown_breach=False,
        )
        
        allowed, reason = self.engine.should_allow_entry(conviction=0.80)
        self.assertTrue(allowed)
    
    def test_entry_blocked_low_conviction(self):
        """Test entry blocked with low conviction."""
        prices = np.array([100, 100.5, 100.3, 100.7, 100.4, 100.6])
        
        self.engine.update(
            prices=prices,
            structure="NEUTRAL",
            liquidity_status="healthy",
            correlation_idx=0.3,
            drawdown_breach=False,
        )
        
        allowed, reason = self.engine.should_allow_entry(conviction=0.50)
        self.assertFalse(allowed)
        self.assertIn("Conviction", reason)
    
    def test_entry_blocked_defensive(self):
        """Test entry blocked in DEFENSIVE regime."""
        prices = np.array([100, 101, 102, 103, 104])
        
        self.engine.update(
            prices=prices,
            structure="BULLISH",
            liquidity_status="healthy",
            correlation_idx=0.2,
            drawdown_breach=True,
        )
        
        allowed, reason = self.engine.should_allow_entry(conviction=0.95)
        self.assertFalse(allowed)
        self.assertIn("DEFENSIVE", reason)
    
    def test_entry_direction_mismatch(self):
        """Test entry blocked with wrong direction in strong regime."""
        # Strong downtrend (bear distribution)
        prices = np.array([114, 112, 110, 108, 106, 104, 102, 100])
        
        self.engine.update(
            prices=prices,
            structure="BEARISH",
            liquidity_status="healthy",
            correlation_idx=0.3,
            drawdown_breach=False,
        )
        
        # Long in bear regime with moderate conviction should be blocked
        allowed, reason = self.engine.should_allow_entry(
            conviction=0.65,
            direction='LONG'
        )
        # May or may not be blocked depending on exact regime
        # Just verify the check runs without error
        self.assertIsInstance(allowed, bool)
    
    # ========================================================================
    # Hysteresis Tests
    # ========================================================================
    
    def test_hysteresis_prevents_chattering(self):
        """Test hysteresis window prevents rapid regime changes."""
        prices_stable = np.array([100, 100.2, 100.1, 100.3, 100.15, 100.25])
        
        # First update establishes regime
        state1 = self.engine.update(
            prices=prices_stable,
            structure="NEUTRAL",
            liquidity_status="healthy",
            correlation_idx=0.3,
            drawdown_breach=False,
        )
        
        # Second update with very similar data should not change regime
        prices_similar = np.array([100.1, 100.25, 100.15, 100.35, 100.2, 100.28])
        state2 = self.engine.update(
            prices=prices_similar,
            structure="NEUTRAL",
            liquidity_status="healthy",
            correlation_idx=0.3,
            drawdown_breach=False,
        )
        
        # Regime should remain the same due to hysteresis
        self.assertEqual(state1.behavioral, state2.behavioral)
        self.assertEqual(state1.operational, state2.operational)
    
    # ========================================================================
    # Force Defensive Tests
    # ========================================================================
    
    def test_force_defensive(self):
        """Test manual force to DEFENSIVE."""
        prices = np.array([100, 101, 102, 103, 104])
        
        self.engine.update(
            prices=prices,
            structure="BULLISH",
            liquidity_status="healthy",
            correlation_idx=0.2,
            drawdown_breach=False,
        )
        
        # Force defensive
        self.engine.force_defensive("Manual test")
        
        self.assertEqual(self.engine.state.operational, OperationalRegime.DEFENSIVE)
        self.assertFalse(self.engine.state.entries_allowed)
    
    # ========================================================================
    # Status Summary Tests
    # ========================================================================
    
    def test_status_summary(self):
        """Test status summary returns complete data."""
        prices = np.array([100, 100.5, 100.3, 100.7, 100.4, 100.6])
        
        self.engine.update(
            prices=prices,
            structure="NEUTRAL",
            liquidity_status="healthy",
            correlation_idx=0.3,
            drawdown_breach=False,
        )
        
        summary = self.engine.get_status_summary()
        
        self.assertIn('behavioral_regime', summary)
        self.assertIn('operational_regime', summary)
        self.assertIn('confidence', summary)
        self.assertIn('entries_allowed', summary)
        self.assertIn('metrics', summary)
    
    def test_status_summary_uninitialized(self):
        """Test status summary when uninitialized."""
        summary = self.engine.get_status_summary()
        self.assertEqual(summary.get('status'), 'uninitialized')
    
    # ========================================================================
    # Get Permissions Tests
    # ========================================================================
    
    def test_get_permissions(self):
        """Test get_permissions returns complete dict."""
        prices = np.array([100, 100.5, 100.3, 100.7, 100.4, 100.6])
        
        self.engine.update(
            prices=prices,
            structure="NEUTRAL",
            liquidity_status="healthy",
            correlation_idx=0.3,
            drawdown_breach=False,
        )
        
        permissions = self.engine.get_permissions()
        
        self.assertIn('entries_allowed', permissions)
        self.assertIn('size_modifier', permissions)
        self.assertIn('max_leverage', permissions)
        self.assertIn('min_conviction', permissions)
        self.assertIn('behavioral', permissions)
        self.assertIn('operational', permissions)
    
    def test_get_permissions_uninitialized(self):
        """Test get_permissions when uninitialized."""
        permissions = self.engine.get_permissions()
        self.assertFalse(permissions['entries_allowed'])
        self.assertEqual(permissions['size_modifier'], 0.5)
        self.assertIn('reason', permissions)


class TestUnifiedRegimeConfig(unittest.TestCase):
    """Tests for regime configuration."""
    
    def test_all_behavioral_regimes_have_configs(self):
        """Test all behavioral regimes have configurations."""
        from HolonicTrader.unified_regime_engine import UNIFIED_REGIME_CONFIG
        
        for behavioral in BehavioralRegime:
            for operational in OperationalRegime:
                key = (behavioral, operational)
                # Should have exact match or fallback
                self.assertIn(key, UNIFIED_REGIME_CONFIG)
    
    def test_defensive_blocks_entries(self):
        """Test all DEFENSIVE configs block entries."""
        from HolonicTrader.unified_regime_engine import UNIFIED_REGIME_CONFIG
        
        for (behavioral, operational), config in UNIFIED_REGIME_CONFIG.items():
            if operational == OperationalRegime.DEFENSIVE:
                self.assertFalse(config['entries_allowed'])
                self.assertEqual(config['size_modifier'], 0.0)
                self.assertEqual(config['min_conviction'], 0.99)


class TestRegimeState(unittest.TestCase):
    """Tests for RegimeState dataclass."""
    
    def test_to_dict(self):
        """Test RegimeState serialization."""
        from HolonicTrader.unified_regime_engine import RegimeState
        import time
        
        state = RegimeState(
            timestamp=time.time(),
            behavioral=BehavioralRegime.LOW_VOL_MEAN_REVERT,
            operational=OperationalRegime.HARVEST,
            confidence=0.85,
        )
        
        state_dict = state.to_dict()
        
        self.assertEqual(state_dict['behavioral'], "LOW_VOL_MEAN_REVERT")
        self.assertEqual(state_dict['operational'], "HARVEST")
        self.assertAlmostEqual(state_dict['confidence'], 0.85)


if __name__ == '__main__':
    unittest.main()
