"""
Tests for SMCE Layer 1 – SMCERegimeEngine

Covers all 4 regime classifications and priority ordering.
Run with: pytest tests/smce/test_smce_regime_engine.py -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from HolonicTrader.smce_regime_engine import SMCERegimeEngine, REGIME_CONFIG


@pytest.fixture
def engine():
    return SMCERegimeEngine()


# ── HARVEST ──────────────────────────────────────────────────────────────────

class TestHarvestRegime:
    def test_harvest_low_entropy_neutral(self, engine):
        r = engine.classify("NEUTRAL", entropy=0.5, liquidity_status="healthy", correlation_idx=0.3)
        assert r == "HARVEST"

    def test_harvest_low_entropy_bearish(self, engine):
        """BEARISH + low entropy + healthy → HARVEST (not EXPANSION)."""
        r = engine.classify("BEARISH", entropy=0.6, liquidity_status="healthy", correlation_idx=0.3)
        assert r == "HARVEST"

    def test_harvest_new_entries_allowed(self, engine):
        engine.classify("NEUTRAL", 0.5, "healthy", 0.3)
        cfg = engine.get_permissions()
        assert cfg["new_entries_allowed"] is True
        assert "HARVEST" in cfg["strategy_modules"]


# ── EXPANSION ─────────────────────────────────────────────────────────────────

class TestExpansionRegime:
    def test_expansion_bullish_low_entropy(self, engine):
        r = engine.classify("BULLISH", entropy=0.5, liquidity_status="healthy", correlation_idx=0.4)
        assert r == "EXPANSION"

    def test_expansion_requires_bullish(self, engine):
        """NEUTRAL structure should NOT produce EXPANSION."""
        r = engine.classify("NEUTRAL", entropy=0.5, liquidity_status="healthy", correlation_idx=0.4)
        assert r != "EXPANSION"

    def test_expansion_min_prob_score(self, engine):
        engine.classify("BULLISH", 0.5, "healthy", 0.4)
        cfg = engine.get_permissions()
        assert cfg["min_prob_score"] == 6

    def test_expansion_blocked_by_high_entropy(self, engine):
        """High entropy should prevent EXPANSION even with BULLISH structure."""
        r = engine.classify("BULLISH", entropy=1.8, liquidity_status="healthy", correlation_idx=0.4)
        assert r != "EXPANSION"


# ── TRANSITION ────────────────────────────────────────────────────────────────

class TestTransitionRegime:
    def test_transition_moderate_entropy(self, engine):
        r = engine.classify("NEUTRAL", entropy=1.2, liquidity_status="healthy", correlation_idx=0.4)
        assert r == "TRANSITION"

    def test_transition_liquidity_warning(self, engine):
        r = engine.classify("NEUTRAL", entropy=0.5, liquidity_status="warning", correlation_idx=0.3)
        assert r == "TRANSITION"

    def test_transition_size_modifier_halved(self, engine):
        engine.classify("NEUTRAL", entropy=1.2, liquidity_status="healthy", correlation_idx=0.4)
        cfg = engine.get_permissions()
        assert cfg["size_modifier"] == 0.5

    def test_transition_only_harvest_strategy(self, engine):
        engine.classify("NEUTRAL", entropy=1.2, liquidity_status="healthy", correlation_idx=0.4)
        cfg = engine.get_permissions()
        assert "EXPANSION" not in cfg["strategy_modules"]
        assert "HARVEST" in cfg["strategy_modules"]


# ── DEFENSIVE ─────────────────────────────────────────────────────────────────

class TestDefensiveRegime:
    def test_defensive_drawdown_breach(self, engine):
        r = engine.classify("NEUTRAL", 0.5, "healthy", 0.3, drawdown_breach=True)
        assert r == "DEFENSIVE"

    def test_defensive_critical_liquidity(self, engine):
        r = engine.classify("NEUTRAL", 0.5, "critical", 0.3)
        assert r == "DEFENSIVE"

    def test_defensive_extreme_entropy(self, engine):
        r = engine.classify("NEUTRAL", entropy=1.3, liquidity_status="healthy", correlation_idx=0.4)
        assert r == "DEFENSIVE"

    def test_defensive_blocks_all_entries(self, engine):
        engine.classify("NEUTRAL", 0.5, "critical", 0.3)
        cfg = engine.get_permissions()
        assert cfg["new_entries_allowed"] is False
        assert cfg["strategy_modules"] == []

    def test_defensive_correlation_spike_warning(self, engine):
        """Correlation > 0.7 + liquidity warning → DEFENSIVE."""
        r = engine.classify("NEUTRAL", entropy=0.8, liquidity_status="warning", correlation_idx=0.75)
        assert r == "DEFENSIVE"


# ── Priority ordering ─────────────────────────────────────────────────────────

class TestPriorityOrdering:
    def test_defensive_trumps_expansion(self, engine):
        """Even BULLISH + low entropy → DEFENSIVE if drawdown_breach."""
        r = engine.classify("BULLISH", 0.5, "healthy", 0.3, drawdown_breach=True)
        assert r == "DEFENSIVE"

    def test_force_defensive(self, engine):
        engine.classify("BULLISH", 0.5, "healthy", 0.3)
        assert engine.current_regime == "EXPANSION"
        engine.force_defensive("Test override")
        assert engine.current_regime == "DEFENSIVE"

    def test_regime_transition_logged(self, engine):
        engine.classify("BULLISH", 0.5, "healthy", 0.3)
        engine.classify("NEUTRAL", 1.2, "healthy", 0.4)
        assert len(engine.transition_log) >= 1

    def test_leverage_small_tier_harvest(self, engine):
        engine.classify("NEUTRAL", 0.5, "healthy", 0.3)
        assert engine.get_max_leverage(equity=200.0) == 3.0   # HARVEST SMALL cap

    def test_leverage_defensive_always_1x(self, engine):
        engine.classify("NEUTRAL", 0.5, "critical", 0.3)
        assert engine.get_max_leverage(equity=200.0) == 1.0
