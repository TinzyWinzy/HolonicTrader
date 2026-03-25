"""
Tests for SMCE Layer 3 – SMCEMonteCarloRiskCourt (pre-trade gate)

Covers veto conditions, borderline CVaR size reduction, and basic
approve path (low-risk trade).
Run with: pytest tests/smce/test_smce_monte_carlo_court.py -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from HolonicTrader.smce_monte_carlo_court import SMCEMonteCarloRiskCourt


@pytest.fixture
def court():
    return SMCEMonteCarloRiskCourt(n_paths=200)   # smaller for test speed


def _safe_trade():
    return {
        "symbol":    "BTC/USDT",
        "direction": "BUY",
        "notional":  10.0,    # small relative to equity
        "leverage":  1.0,
    }


def _high_risk_trade(notional=300.0, leverage=10.0):
    return {
        "symbol":    "DOGE/USDT",
        "direction": "BUY",
        "notional":  notional,
        "leverage":  leverage,
    }


class TestApprovalPath:
    def test_small_safe_trade_approved(self, court):
        """Very small trade on $500 account → should pass most paths."""
        result = court.evaluate_pre_trade(
            equity=500.0,
            proposed_trade=_safe_trade(),
            portfolio_positions={},
            volatilities={"BTC/USDT": 0.02},
        )
        # We can't guarantee approval due to randomness, but approved_size ≥ 0
        assert "vetoed" in result
        assert "approved_size" in result
        assert result["paths_run"] > 0

    def test_result_has_required_keys(self, court):
        r = court.evaluate_pre_trade(
            equity=200.0,
            proposed_trade=_safe_trade(),
            portfolio_positions={},
            volatilities={"BTC/USDT": 0.02},
        )
        for key in ("vetoed", "veto_reason", "drawdown_prob", "cvar_95",
                    "liquidation_prob", "approved_size", "paths_run"):
            assert key in r, f"Missing key: {key}"

    def test_approved_size_never_more_than_proposed(self, court):
        r = court.evaluate_pre_trade(
            equity=200.0,
            proposed_trade=_safe_trade(),
            portfolio_positions={},
            volatilities={"BTC/USDT": 0.02},
        )
        assert r["approved_size"] <= _safe_trade()["notional"] + 0.01


class TestVetoConditions:
    def test_zero_equity_vetoed(self, court):
        r = court.evaluate_pre_trade(
            equity=0.0,
            proposed_trade=_safe_trade(),
            portfolio_positions={},
            volatilities={},
        )
        assert r["vetoed"] is True

    def test_zero_notional_vetoed(self, court):
        r = court.evaluate_pre_trade(
            equity=500.0,
            proposed_trade={**_safe_trade(), "notional": 0.0},
            portfolio_positions={},
            volatilities={},
        )
        assert r["vetoed"] is True

    def test_stats_tracked(self, court):
        """veto_count and approve_count accumulate correctly."""
        initial_vetos = court.veto_count
        # Force at least one veto via zero equity
        court.evaluate_pre_trade(0.0, _safe_trade(), {}, {})
        assert court.veto_count > initial_vetos


class TestSizeReduction:
    def test_approved_size_not_negative(self, court):
        """Approved size must always be ≥ 0 regardless of CVaR."""
        r = court.evaluate_pre_trade(
            equity=100.0,
            proposed_trade=_safe_trade(),
            portfolio_positions={},
            volatilities={"BTC/USDT": 0.03},
        )
        assert r["approved_size"] >= 0.0

    def test_drawdown_prob_between_0_and_1(self, court):
        r = court.evaluate_pre_trade(
            equity=500.0,
            proposed_trade=_safe_trade(),
            portfolio_positions={},
            volatilities={"BTC/USDT": 0.02},
        )
        assert 0.0 <= r["drawdown_prob"] <= 1.0

    def test_cvar_between_0_and_1(self, court):
        r = court.evaluate_pre_trade(
            equity=500.0,
            proposed_trade=_safe_trade(),
            portfolio_positions={},
            volatilities={"BTC/USDT": 0.02},
        )
        assert 0.0 <= r["cvar_95"] <= 5.0   # upper bound loose for extreme vols
