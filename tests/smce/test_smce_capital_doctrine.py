"""
Tests for SMCE Layer 0 – SMCECapitalDoctrine

Covers: equity tier detection, leverage caps by tier+regime,
daily/weekly drawdown breach → DEFENSIVE, 48h cooldown,
stacking proximity block, exposure/cluster limits.
Run with: pytest tests/smce/test_smce_capital_doctrine.py -v
"""

import pytest
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from HolonicTrader.smce_capital_doctrine import SMCECapitalDoctrine


@pytest.fixture
def doc():
    d = SMCECapitalDoctrine()
    d.day_start_equity  = 200.0
    d.week_start_equity = 200.0
    return d


# ── Leverage caps ─────────────────────────────────────────────────────────────

class TestLeverageCaps:
    def test_small_harvest_max_3x(self, doc):
        assert doc.get_max_leverage(equity=200.0, smce_regime="HARVEST") == 3.0

    def test_small_expansion_max_3x(self, doc):
        assert doc.get_max_leverage(equity=200.0, smce_regime="EXPANSION") == 3.0

    def test_small_transition_max_2x(self, doc):
        assert doc.get_max_leverage(equity=200.0, smce_regime="TRANSITION") == 2.0

    def test_small_defensive_max_1x(self, doc):
        assert doc.get_max_leverage(equity=200.0, smce_regime="DEFENSIVE") == 1.0

    def test_medium_expansion_max_5x(self, doc):
        assert doc.get_max_leverage(equity=1000.0, smce_regime="EXPANSION") == 5.0

    def test_large_harvest_max_5x(self, doc):
        assert doc.get_max_leverage(equity=10000.0, smce_regime="HARVEST") == 5.0


# ── Exposure limits ────────────────────────────────────────────────────────────

class TestExposureLimits:
    def test_total_exposure_ceiling_small(self, doc):
        """$10 proposed notional on $200 equity = 5% < all caps → allowed."""
        allowed, reason, _ = doc.check_trade(
            symbol="ETH/USDT", direction="BUY",
            proposed_notional=10.0, proposed_leverage=1.0,
            equity=200.0, smce_regime="HARVEST",
            current_positions={},
        )
        assert allowed is True

    def test_total_exposure_blocked_when_over_limit(self, doc):
        """Existing $55 + proposed $10 = $65 > 30% of $200 ($60) → block."""
        mock_pos = {"direction": "BUY", "entry_price": 1000.0,
                    "quantity": 0.055, "notional": 55.0}
        allowed, reason, _ = doc.check_trade(
            symbol="ETH/USDT", direction="BUY",
            proposed_notional=10.0, proposed_leverage=1.0,
            equity=200.0, smce_regime="HARVEST",
            current_positions={"BTC/USDT": mock_pos},
        )
        assert allowed is False

    def test_per_asset_cap_blocks(self, doc):
        """Proposed notional > 10% of equity for SMALL → block."""
        # $200 * 10% = $20 cap; proposing $30 → block
        allowed, reason, _ = doc.check_trade(
            symbol="ETH/USDT", direction="BUY",
            proposed_notional=30.0, proposed_leverage=1.0,
            equity=200.0, smce_regime="HARVEST",
            current_positions={},
        )
        assert allowed is False
        assert "cap" in reason.lower()

    def test_cluster_cap_blocks(self, doc):
        """Cluster exposure > 15% for SMALL → block."""
        allowed, reason, _ = doc.check_trade(
            symbol="BTC/USDT", direction="BUY",
            proposed_notional=10.0, proposed_leverage=1.0,
            equity=200.0, smce_regime="HARVEST",
            current_positions={},
            cluster_exposure_pct=0.20,   # 20% > 15%
        )
        assert allowed is False
        assert "cluster" in reason.lower()


# ── Stacking block ─────────────────────────────────────────────────────────────

class TestStackingBlock:
    def test_stacking_same_symbol_blocked(self, doc):
        """Second entry in same symbol+direction → block."""
        mock_pos = type('P', (), {
            "direction":   "BUY",
            "entry_price": 50000.0,
            "quantity":    0.001,
            "notional":    50.0,
        })()
        allowed, reason, _ = doc.check_trade(
            symbol="BTC/USDT", direction="BUY",
            proposed_notional=10.0, proposed_leverage=1.0,
            equity=500.0, smce_regime="HARVEST",
            current_positions={"BTC/USDT": mock_pos},
        )
        assert allowed is False
        assert "stacking" in reason.lower() or "STACKING" in reason

    def test_opposite_direction_not_blocked(self, doc):
        """Short entry when holding long → not a stack, allow."""
        mock_pos = type('P', (), {
            "direction":   "BUY",
            "entry_price": 50000.0,
            "quantity":    0.001,
            "notional":    50.0,
        })()
        # Opposite direction should not trigger stacking block
        # (cluster/exposure caps may still block, but not stacking rule)
        allowed, reason, _ = doc.check_trade(
            symbol="BTC/USDT", direction="SELL",
            proposed_notional=10.0, proposed_leverage=1.0,
            equity=500.0, smce_regime="HARVEST",
            current_positions={"BTC/USDT": mock_pos},
        )
        assert "STACKING" not in reason


# ── Drawdown breach / DEFENSIVE cooldown ────────────────────────────────────────

class TestDrawdownEnforcement:
    def test_daily_drawdown_triggers_defensive(self, doc):
        """3% daily drawdown on $200 = $6 loss → DEFENSIVE."""
        doc.day_start_equity = 200.0
        status = doc.update(current_equity=193.0)   # 3.5% down
        assert status["in_defensive"] is True

    def test_weekly_drawdown_triggers_defensive(self, doc):
        """6% weekly drawdown on $200 → DEFENSIVE."""
        doc.day_start_equity  = 195.0   # today not a breach
        doc.week_start_equity = 200.0
        status = doc.update(current_equity=187.0)   # 6.5% weekly
        assert status["in_defensive"] is True

    def test_defensive_blocks_new_entries(self, doc):
        """After defensive triggered, new entry blocked."""
        doc.update(193.0)      # trigger
        # now try to enter
        allowed, reason, _ = doc.check_trade(
            symbol="BTC/USDT", direction="BUY",
            proposed_notional=10.0, proposed_leverage=1.0,
            equity=193.0, smce_regime="HARVEST",
            current_positions={},
        )
        assert allowed is False
        assert "cooldown" in reason.lower()

    def test_risk_multiplier_halved_on_breach(self, doc):
        doc.update(193.0)
        assert doc.risk_multiplier == 0.5

    def test_no_breach_below_limit(self, doc):
        doc.day_start_equity = 200.0
        status = doc.update(199.0)   # only 0.5% down
        assert status["in_defensive"] is False


# ── State persistence ──────────────────────────────────────────────────────────

class TestStatePersistence:
    def test_get_and_load_state_roundtrip(self, doc):
        doc.update(193.0)   # trigger defensive
        state = doc.get_state()
        doc2  = SMCECapitalDoctrine()
        doc2.load_state(state)
        assert doc2.defensive_until == doc.defensive_until
        assert doc2.risk_multiplier == doc.risk_multiplier
