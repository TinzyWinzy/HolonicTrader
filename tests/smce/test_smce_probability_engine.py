"""
Tests for SMCE Layer 2 – ProbabilityStackingEngine

Covers scoring components, regime thresholds, TRANSITION size halving,
and liquidity disqualification.
Run with: pytest tests/smce/test_smce_probability_engine.py -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from HolonicTrader.smce_probability_engine import ProbabilityStackingEngine


@pytest.fixture
def engine():
    return ProbabilityStackingEngine()


def _high_score_context():
    """Return inputs that produce maximum score."""
    trade   = {"symbol": "BTC/USDT", "direction": "BUY", "proposed_cluster_exposure": 0.0}
    port    = {"equity": 500.0, "cluster_exposure": 0.0, "cvar_95": 0.01}
    context = {"structure": "BULLISH", "momentum_aligned": True,
               "liquidity_status": "healthy", "entropy": 0.5, "correlation_idx": 0.3}
    return trade, port, context


# ── Scoring components ────────────────────────────────────────────────────────

class TestScoringComponents:
    def test_max_score_achievable(self, engine):
        trade, port, context = _high_score_context()
        r = engine.score_trade(trade, port, context, smce_regime="HARVEST")
        assert r["score"] == 8.0   # 2+1+1+1+1+2

    def test_structure_0_against_direction(self, engine):
        trade, port, context = _high_score_context()
        context["structure"] = "BEARISH"  # against BUY
        r = engine.score_trade(trade, port, context, smce_regime="HARVEST")
        assert r["breakdown"]["structure"] == 0

    def test_structure_1_neutral(self, engine):
        trade, port, context = _high_score_context()
        context["structure"] = "NEUTRAL"
        r = engine.score_trade(trade, port, context, smce_regime="HARVEST")
        assert r["breakdown"]["structure"] == 1

    def test_structure_2_aligned(self, engine):
        trade, port, context = _high_score_context()
        r = engine.score_trade(trade, port, context, smce_regime="HARVEST")
        assert r["breakdown"]["structure"] == 2

    def test_momentum_adds_point(self, engine):
        trade, port, context = _high_score_context()
        context["momentum_aligned"] = False
        r = engine.score_trade(trade, port, context, smce_regime="HARVEST")
        assert r["breakdown"]["momentum"] == 0

    def test_cvar_excellent_gets_2pts(self, engine):
        trade, port, context = _high_score_context()
        port["cvar_95"] = 0.01   # < 2%
        r = engine.score_trade(trade, port, context, smce_regime="HARVEST")
        assert r["breakdown"]["cvar"] == 2

    def test_cvar_good_gets_1pt(self, engine):
        trade, port, context = _high_score_context()
        port["cvar_95"] = 0.03   # < 4%
        r = engine.score_trade(trade, port, context, smce_regime="HARVEST")
        assert r["breakdown"]["cvar"] == 1

    def test_cvar_bad_gets_0pts(self, engine):
        trade, port, context = _high_score_context()
        port["cvar_95"] = 0.05   # ≥ 4%
        r = engine.score_trade(trade, port, context, smce_regime="HARVEST")
        assert r["breakdown"]["cvar"] == 0


# ── Regime thresholds ─────────────────────────────────────────────────────────

class TestRegimeThresholds:
    def test_harvest_threshold_5(self, engine):
        """Score ≥ 5 should pass HARVEST."""
        trade   = {"symbol": "BTC/USDT", "direction": "BUY", "proposed_cluster_exposure": 0.0}
        port    = {"equity": 200.0, "cluster_exposure": 0.0, "cvar_95": 0.01}  # 2pts cvar
        context = {"structure": "NEUTRAL", "momentum_aligned": True,
                   "liquidity_status": "healthy", "entropy": 0.5, "correlation_idx": 0.0}
        # structure=1, momentum=1, liquidity=1, entropy=1, corr=1, cvar=2 → 7
        r = engine.score_trade(trade, port, context, smce_regime="HARVEST")
        assert r["eligible"] is True
        assert r["score"] >= 5

    def test_harvest_below_threshold_blocked(self, engine):
        """Score < 5 should fail HARVEST."""
        trade   = {"symbol": "BTC/USDT", "direction": "BUY", "proposed_cluster_exposure": 0.0}
        port    = {"equity": 200.0, "cluster_exposure": 0.0, "cvar_95": 0.05}  # 0pts cvar
        context = {"structure": "BEARISH", "momentum_aligned": False,  # 0 struct, 0 mom
                   "liquidity_status": "healthy", "entropy": 0.5, "correlation_idx": 0.0}
        # structure=0, momentum=0, liquidity=1, entropy=1, corr=1, cvar=0 → 3  < 5
        r = engine.score_trade(trade, port, context, smce_regime="HARVEST")
        assert r["eligible"] is False

    def test_expansion_threshold_6(self, engine):
        """Score must be ≥ 6 for EXPANSION."""
        trade, port, context = _high_score_context()
        port["cvar_95"] = 0.01
        r = engine.score_trade(trade, port, context, smce_regime="EXPANSION")
        assert r["eligible"] is True
        assert r["score"] >= 6

    def test_defensive_always_blocked(self, engine):
        trade, port, context = _high_score_context()
        r = engine.score_trade(trade, port, context, smce_regime="DEFENSIVE")
        assert r["eligible"] is False
        assert "DEFENSIVE" in r["block_reason"]


# ── TRANSITION size modifier ──────────────────────────────────────────────────

class TestTransitionSizeModifier:
    def test_transition_halves_size(self, engine):
        trade, port, context = _high_score_context()
        port["cvar_95"] = 0.01  # high score
        r = engine.score_trade(trade, port, context, smce_regime="TRANSITION")
        if r["eligible"]:
            assert r["size_modifier"] == 0.5

    def test_transition_min_score_6(self, engine):
        """TRANSITION requires score ≥ 6, same as EXPANSION."""
        trade   = {"symbol": "BTC/USDT", "direction": "BUY", "proposed_cluster_exposure": 0.0}
        port    = {"equity": 200.0, "cluster_exposure": 0.0, "cvar_95": 0.05}
        context = {"structure": "NEUTRAL", "momentum_aligned": False,
                   "liquidity_status": "healthy", "entropy": 0.5, "correlation_idx": 0.0}
        # score → 3 < 6
        r = engine.score_trade(trade, port, context, smce_regime="TRANSITION")
        assert r["eligible"] is False


# ── Liquidity disqualification ────────────────────────────────────────────────

class TestLiquidityDisqualification:
    def test_critical_liquidity_disqualifies(self, engine):
        trade, port, context = _high_score_context()
        context["liquidity_status"] = "critical"
        r = engine.score_trade(trade, port, context, smce_regime="HARVEST")
        assert r["eligible"] is False
        assert "critical" in r["block_reason"].lower()
        assert r["score"] == 0

    def test_warning_liquidity_reduces_score(self, engine):
        trade, port, context = _high_score_context()
        context["liquidity_status"] = "warning"
        r = engine.score_trade(trade, port, context, smce_regime="HARVEST")
        assert r["breakdown"]["liquidity"] == 0
        assert r["eligible"] is not False  # not disqualified, just 0 for that component
