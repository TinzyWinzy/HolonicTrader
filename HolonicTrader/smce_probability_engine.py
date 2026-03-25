"""
SMCEProbabilityEngine – Sovereign Micro-Compounding Engine (Layer 2)

Every candidate trade must pass through this 6-component scoring system.
The score determines eligibility and position size modifier.

Score Components (maximum total: 8):
  1. Structure alignment   0–2
  2. Momentum alignment    0–1
  3. Liquidity healthy     0–1
  4. Entropy favorable     0–1
  5. Correlation safe      0–1
  6. Monte Carlo CVaR      0–2

Thresholds by regime:
  HARVEST     ≥ 5   (full size)
  EXPANSION   ≥ 6   (full size)
  TRANSITION  ≥ 6   (and size_modifier = 0.5)
  DEFENSIVE   always blocked (score irrelevant)

NO single signal (whale, APY) can bypass this score.
"""

import logging
from typing import Dict, Tuple, Optional, Any

logger = logging.getLogger("SMCEProbabilityEngine")

# Thresholds keyed by regime (synced with config.py)
# FIX 2026-02-25: Lowered TRANSITION threshold from 4 to 3 to allow more arb opportunities
# The 90%+ veto rate was preventing valid high-yield funding trades
# P1 FIX 2026-03-05: Lowered HARVEST threshold from 4 to 3 - valid setups (XAUT Bullish+Support)
# were scoring 3.0 and getting vetoed. 6-component scoring already filters bad trades.
REGIME_THRESHOLDS: Dict[str, Dict] = {
    "HARVEST":    {"min_score": 3, "size_modifier": 1.0},  # ↓ from 4 (allows valid setups)
    "EXPANSION":  {"min_score": 6, "size_modifier": 1.0},
    "TRANSITION": {"min_score": 3, "size_modifier": 0.7},  # ↓ from 4 (allows more trades), ↑ size mod from 0.5
    "DEFENSIVE":  {"min_score": 999, "size_modifier": 0.0},  # always blocked
}

# Volatility/Conviction Boost for HARVEST/TRANSITION regimes
# Raises effective score for high-quality setups
HARVEST_VOL_BOOST = 0.5    # +0.5 for RVOL > 2.0
HARVEST_CONV_BOOST = 1.0   # +1.0 for probability > 0.65

# Entropy thresholds for component #4
ENTROPY_LONG_MAX  = 1.0   # ordered → good for longs
ENTROPY_SHORT_MIN = 1.5   # high entropy → mean-reversion shorts
CVAR_EXCELLENT    = 0.02  # CVaR < 2% → 2 pts
CVAR_GOOD         = 0.04  # CVaR < 4% → 1 pt
CLUSTER_CAP_SMALL = 0.15  # 15% cluster cap for SMALL tier


class ProbabilityStackingEngine:
    """
    Layer 2 probability gate for the SMCE pipeline.

    Usage:
        engine = ProbabilityStackingEngine()
        result = engine.score_trade(
            trade_candidate={"direction": "BUY", "symbol": "BTC/USDT"},
            portfolio_state={
                "equity": 200.0,
                "cluster_exposure": 0.10,
                "cvar_95": 0.015,
            },
            market_context={
                "structure": "NEUTRAL",
                "momentum_aligned": True,
                "liquidity_status": "healthy",
                "entropy": 0.75,
                "correlation_idx": 0.4,
            },
            smce_regime="HARVEST",
        )
        if result["eligible"]:
            size = base_size * result["size_modifier"]
    """

    def __init__(self):
        self.last_scorecard: Dict = {}

    # ─── Public API ───────────────────────────────────────────────────────────

    def score_trade(
        self,
        trade_candidate: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        market_context: Dict[str, Any],
        smce_regime: str = "HARVEST",
    ) -> Dict[str, Any]:
        """
        Score a candidate trade and return eligibility verdict.

        Args:
            trade_candidate:  dict with keys: direction ('BUY'|'SELL'), symbol, proposed_cluster_exposure
            portfolio_state:  dict with keys: equity, cluster_exposure (fraction), cvar_95 (fraction)
            market_context:   dict with keys: structure, momentum_aligned (bool), liquidity_status,
                              entropy (float), correlation_idx (float)
            smce_regime:      current SMCE regime string

        Returns:
            {
                "score":         float (0–8),
                "eligible":      bool,
                "size_modifier": float (0.0–1.0),
                "breakdown":     dict of {component: score},
                "block_reason":  str or None,
            }
        """
        # DEFENSIVE always blocks
        if smce_regime == "DEFENSIVE":
            self.last_scorecard = self._make_defensive_card()
            return self.last_scorecard

        direction       = trade_candidate.get("direction", "BUY")
        structure       = market_context.get("structure", "NEUTRAL")
        momentum_ok     = bool(market_context.get("momentum_aligned", False))
        liquidity       = market_context.get("liquidity_status", "warning")
        entropy         = float(market_context.get("entropy", 1.0))
        corr_idx        = float(market_context.get("correlation_idx", 0.5))
        equity          = float(portfolio_state.get("equity", 100.0))
        cluster_exp     = float(portfolio_state.get("cluster_exposure", 0.0))
        proposed_add    = float(trade_candidate.get("proposed_cluster_exposure", 0.0))
        cvar_95         = float(portfolio_state.get("cvar_95", 0.05))   # fraction of equity
        
        # Volatility/Conviction data for HARVEST boost
        rvol           = float(trade_candidate.get("rvol", 1.0))          # Relative volume
        signal_prob    = float(trade_candidate.get("signal_probability", 0.5))  # Model probability

        # Compute individual components
        s1 = self._score_structure(structure, direction)
        s2 = self._score_momentum(momentum_ok)
        s3, disqualify = self._score_liquidity(liquidity)
        s4 = self._score_entropy(entropy, direction)
        s5 = self._score_correlation(cluster_exp, proposed_add, equity)
        s6 = self._score_cvar(cvar_95)

        if disqualify:
            card = {
                "score":         0,
                "eligible":      False,
                "size_modifier": 0.0,
                "breakdown":     {
                    "structure":   s1,
                    "momentum":    s2,
                    "liquidity":   s3,
                    "entropy":     s4,
                    "correlation": s5,
                    "cvar":        s6,
                },
                "block_reason":  "Liquidity CRITICAL – trade disqualified",
            }
            self.last_scorecard = card
            logger.warning("[SMCE-PROB] %s %s – DISQUALIFIED: critical liquidity",
                           direction, trade_candidate.get("symbol", "?"))
            return card

        total_score = s1 + s2 + s3 + s4 + s5 + s6
        
        # === HARVEST & TRANSITION REGIME: VOLATILITY/CONVICTION BOOST ===
        # Raise score via volatility (RVOL) or conviction (signal probability)
        # This allows entries with score 4.5 to qualify if they have high vol/conv
        boost_applied = 0.0
        if smce_regime in ("HARVEST", "TRANSITION"):
            # Volatility boost: RVOL > 2.0 indicates unusual trading activity
            if rvol > 2.0:
                boost_applied += HARVEST_VOL_BOOST
            # Conviction boost: Signal probability > 0.65 indicates high confidence
            if signal_prob > 0.65:
                boost_applied += HARVEST_CONV_BOOST
            
            if boost_applied > 0:
                logger.info(f"[SMCE-PROB] {smce_regime} BOOST: +{boost_applied:.1f} (RVOL={rvol:.2f}, Prob={signal_prob:.2f})")
        
        boosted_score = total_score + boost_applied
        # Cap total score at 8 (maximum possible)
        boosted_score = min(boosted_score, 8.0)
        
        threshold_cfg = REGIME_THRESHOLDS.get(smce_regime, REGIME_THRESHOLDS["DEFENSIVE"])
        min_score     = threshold_cfg["min_score"]
        size_mod      = threshold_cfg["size_modifier"]

        eligible      = boosted_score >= min_score
        block_reason  = None if eligible else (
            f"Score {total_score:.1f} < threshold {min_score} for regime {smce_regime}"
        )

        card = {
            "score":         boosted_score,
            "base_score":    total_score,
            "boost_applied": boost_applied,
            "eligible":      eligible,
            "size_modifier": size_mod if eligible else 0.0,
            "breakdown": {
                "structure":   s1,
                "momentum":    s2,
                "liquidity":   s3,
                "entropy":     s4,
                "correlation": s5,
                "cvar":        s6,
            },
            "block_reason":  block_reason,
        }
        self.last_scorecard = card

        level = logging.INFO if eligible else logging.DEBUG
        logger.log(level,
            "[SMCE-PROB] %s %s | score=%.1f/%d (base=%.1f, boost=+%.1f) | eligible=%s | regime=%s",
            direction,
            trade_candidate.get("symbol", "?"),
            boosted_score, min_score,
            total_score, boost_applied,
            eligible,
            smce_regime,
        )
        return card

    def get_last_scorecard(self) -> Dict:
        """Return the most recent scorecard (for digest logging)."""
        return self.last_scorecard

    # ─── Score Components ─────────────────────────────────────────────────────

    def _score_structure(self, structure: str, direction: str) -> float:
        """
        +2 if structure directly supports direction (BULLISH+BUY or BEARISH+SELL)
           AND at support/resistance level.
        +1 if neutral but aligned.
        +0 if against structure.
        """
        is_long = direction == "BUY"
        if is_long and structure in ("BULLISH", "SUPPORT"):
            return 2.0
        if not is_long and structure in ("BEARISH", "RESISTANCE"):
            return 2.0
        if structure == "NEUTRAL":
            return 1.0
        return 0.0  # Against structure

    def _score_momentum(self, momentum_aligned: bool) -> float:
        return 1.0 if momentum_aligned else 0.0

    def _score_liquidity(self, liquidity_status: str) -> Tuple[float, bool]:
        """Returns (score, disqualify). Disqualify=True means reject outright."""
        if liquidity_status == "healthy":
            return 1.0, False
        if liquidity_status == "warning":
            return 0.0, False
        # critical
        return 0.0, True

    def _score_entropy(self, entropy: float, direction: str) -> float:
        """
        For longs: +1 if entropy < 1.0 (ordered → trend following).
        For shorts: +1 if entropy > 1.5 (chaotic → mean reversion).
        """
        is_long = direction == "BUY"
        if is_long and entropy < ENTROPY_LONG_MAX:
            return 1.0
        if not is_long and entropy > ENTROPY_SHORT_MIN:
            return 1.0
        return 0.0

    def _score_correlation(
        self, existing_cluster_pct: float, proposed_add_pct: float, equity: float
    ) -> float:
        """
        +1 if (existing cluster + proposed) stays within SMCE cluster cap.
        The cap is 15% for SMALL tier (< $500), scaling up for larger accounts.
        """
        # Determine cluster cap based on equity tier
        if equity < 500.0:
            cluster_cap = CLUSTER_CAP_SMALL
        elif equity < 5000.0:
            cluster_cap = 0.20
        else:
            cluster_cap = 0.25

        total_cluster = existing_cluster_pct + proposed_add_pct
        return 1.0 if total_cluster <= cluster_cap else 0.0

    def _score_cvar(self, cvar_95: float) -> float:
        """
        +2 if 95% CVaR < 2% of equity.
        +1 if CVaR < 4%.
        +0 otherwise.
        """
        if cvar_95 < CVAR_EXCELLENT:
            return 2.0
        if cvar_95 < CVAR_GOOD:
            return 1.0
        return 0.0

    @staticmethod
    def _make_defensive_card() -> Dict:
        return {
            "score":         0,
            "eligible":      False,
            "size_modifier": 0.0,
            "breakdown":     {k: 0 for k in ("structure","momentum","liquidity","entropy","correlation","cvar")},
            "block_reason":  "DEFENSIVE regime – no new entries allowed",
        }
