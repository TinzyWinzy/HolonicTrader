"""
Tests for SMCE Strategy Modules – HarvestStrategy and ExpansionStrategy

Covers entry parameter computation, size scaling, regime gating,
and stacking blocks.
Run with: pytest tests/smce/test_smce_strategies.py -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from HolonicTrader.smce_strategy_harvest import HarvestStrategy
from HolonicTrader.smce_strategy_expansion import ExpansionStrategy


# ═════════════════════════════════════════════════════════════════════════════
# HARVEST STRATEGY
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def harvest():
    return HarvestStrategy()


class TestHarvestEntry:
    def test_harvest_allowed_in_harvest_regime(self, harvest):
        r = harvest.compute_entry("BTC/USDT", 95000.0, equity=200.0,
                                  smce_regime="HARVEST")
        assert r["allowed"] is True

    def test_harvest_blocked_in_expansion(self, harvest):
        r = harvest.compute_entry("BTC/USDT", 95000.0, equity=200.0,
                                  smce_regime="EXPANSION")
        assert r["allowed"] is False
        assert "inactive" in r["block_reason"].lower()

    def test_harvest_blocked_in_defensive(self, harvest):
        r = harvest.compute_entry("BTC/USDT", 95000.0, equity=200.0,
                                  smce_regime="DEFENSIVE")
        assert r["allowed"] is False

    def test_harvest_no_stacking(self, harvest):
        r = harvest.compute_entry("BTC/USDT", 95000.0, equity=200.0,
                                  smce_regime="HARVEST",
                                  existing_position={"direction": "BUY"})
        assert r["allowed"] is False
        assert "stacking" in r["block_reason"].lower()

    def test_harvest_risk_reward_at_least_1(self, harvest):
        r = harvest.compute_entry("SOL/USDT", 250.0, equity=200.0,
                                  smce_regime="HARVEST", volatility_24h=0.03)
        assert r["allowed"] is True
        assert r["risk_reward"] >= 1.0

    def test_harvest_allocation_within_bounds(self, harvest):
        r = harvest.compute_entry("BTC/USDT", 95000.0, equity=200.0,
                                  smce_regime="HARVEST")
        assert 0.10 <= r["allocation_pct"] <= 0.20

    def test_harvest_leverage_capped_small_tier(self, harvest):
        r = harvest.compute_entry("BTC/USDT", 95000.0, equity=200.0,
                                  smce_regime="HARVEST")
        assert r["leverage"] <= 3.0

    def test_harvest_sl_above_zero(self, harvest):
        r = harvest.compute_entry("BTC/USDT", 95000.0, equity=200.0,
                                  smce_regime="HARVEST")
        assert r["stop_loss_px"] > 0

    def test_harvest_tp_above_entry(self, harvest):
        r = harvest.compute_entry("BTC/USDT", 95000.0, equity=200.0,
                                  smce_regime="HARVEST")
        assert r["take_profit_px"] > 95000.0

    def test_harvest_tier_small(self, harvest):
        r = harvest.compute_entry("BTC/USDT", 95000.0, equity=200.0,
                                  smce_regime="HARVEST")
        assert r["tier"] == "SMALL"

    def test_harvest_tier_medium(self, harvest):
        r = harvest.compute_entry("BTC/USDT", 95000.0, equity=1000.0,
                                  smce_regime="HARVEST")
        assert r["tier"] == "MEDIUM"

    def test_harvest_high_vol_tighter_leverage(self, harvest):
        """Higher vol → lower leverage."""
        r_low  = harvest.compute_entry("BTC/USDT", 95000.0, 200.0, smce_regime="HARVEST",
                                       volatility_24h=0.01)
        r_high = harvest.compute_entry("BTC/USDT", 95000.0, 200.0, smce_regime="HARVEST",
                                       volatility_24h=0.05)
        assert r_low["allowed"] and r_high["allowed"]
        assert r_high["leverage"] <= r_low["leverage"]


# ═════════════════════════════════════════════════════════════════════════════
# EXPANSION STRATEGY
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def expansion():
    return ExpansionStrategy()


class TestExpansionEntry:
    def test_expansion_allowed_in_expansion_regime(self, expansion):
        r = expansion.compute_entry("BTC/USDT", 95000.0, equity=1000.0,
                                    smce_regime="EXPANSION", structure="BULLISH",
                                    momentum_strong=True)
        assert r["allowed"] is True

    def test_expansion_blocked_in_harvest(self, expansion):
        r = expansion.compute_entry("BTC/USDT", 95000.0, equity=1000.0,
                                    smce_regime="HARVEST", structure="BULLISH",
                                    momentum_strong=True)
        assert r["allowed"] is False

    def test_expansion_requires_bullish(self, expansion):
        r = expansion.compute_entry("BTC/USDT", 95000.0, equity=1000.0,
                                    smce_regime="EXPANSION", structure="NEUTRAL",
                                    momentum_strong=True)
        assert r["allowed"] is False
        assert "BULLISH" in r["block_reason"]

    def test_expansion_requires_strong_momentum(self, expansion):
        r = expansion.compute_entry("BTC/USDT", 95000.0, equity=1000.0,
                                    smce_regime="EXPANSION", structure="BULLISH",
                                    momentum_strong=False)
        assert r["allowed"] is False
        assert "momentum" in r["block_reason"].lower()

    def test_expansion_high_vol_blocked(self, expansion):
        r = expansion.compute_entry("BTC/USDT", 95000.0, equity=1000.0,
                                    smce_regime="EXPANSION", structure="BULLISH",
                                    momentum_strong=True, volatility_24h=0.08)
        assert r["allowed"] is False
        assert "Volatility" in r["block_reason"]

    def test_expansion_no_scaling_in(self, expansion):
        r = expansion.compute_entry("BTC/USDT", 95000.0, equity=1000.0,
                                    smce_regime="EXPANSION", structure="BULLISH",
                                    momentum_strong=True,
                                    existing_positions=["BTC/USDT"])
        assert r["allowed"] is False
        assert "scaling" in r["block_reason"].lower()

    def test_expansion_leverage_within_tier_cap(self, expansion):
        r = expansion.compute_entry("BTC/USDT", 95000.0, equity=200.0,
                                    smce_regime="EXPANSION", structure="BULLISH",
                                    momentum_strong=True)
        if r["allowed"]:
            assert r["leverage"] <= 3.0   # SMALL tier cap

    def test_expansion_trailing_stop_below_entry(self, expansion):
        r = expansion.compute_entry("BTC/USDT", 95000.0, equity=1000.0,
                                    smce_regime="EXPANSION", structure="BULLISH",
                                    momentum_strong=True, atr=1000.0)
        if r["allowed"]:
            assert r["trailing_stop_px"] < 95000.0

    def test_expansion_swing_low_as_stop(self, expansion):
        r = expansion.compute_entry("BTC/USDT", 95000.0, equity=1000.0,
                                    smce_regime="EXPANSION", structure="BULLISH",
                                    momentum_strong=True, swing_low=93000.0)
        if r["allowed"]:
            assert r["trailing_stop_px"] == 93000.0
