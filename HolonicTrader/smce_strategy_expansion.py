"""
SMCE Strategy Modules – EXPANSION Mode (Rare, Trend-Capture Engine)

Goal: Capture trend acceleration – occasional larger wins.
Active ONLY in EXPANSION regime.

Constitutional parameters (SMALL tier < $500):
  - Allocation per trade: 15–20% of equity
  - Leverage:             up to 5x (volatility-gated)
  - Trailing stop:        based on structure swing low/high
  - No scaling in:        one entry, one exit
  - Trend-confirmed:      structure BULLISH + momentum strong (both required)
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("SMCE.ExpansionStrategy")

# ── Constitutional parameters by equity tier ──────────────────────────────────
TIER_PARAMS = {
    "SMALL":  {   # < $500
        "alloc_min":        0.15,
        "alloc_max":        0.20,
        "leverage_max":     3.0,    # SMALL hard cap from Layer 0
        "trailing_atr_mul": 2.0,
    },
    "MEDIUM": {   # $500–$5000
        "alloc_min":        0.15,
        "alloc_max":        0.20,
        "leverage_max":     5.0,
        "trailing_atr_mul": 2.5,
    },
    "LARGE":  {   # > $5000
        "alloc_min":        0.15,
        "alloc_max":        0.20,
        "leverage_max":     5.0,
        "trailing_atr_mul": 3.0,
    },
}

# High-volatility veto: reject EXPANSION if 24h vol too extreme
EXPANSION_MAX_VOL = 0.06   # 6% 24h vol → too risky for trend entry


class ExpansionStrategy:
    """
    EXPANSION mode strategy: trend-acceleration capture.

    One entry, one exit (no scaling in). Trailing stop based on
    structure-determined swing low (for longs) or swing high (for shorts).
    """

    def compute_entry(
        self,
        symbol: str,
        entry_price: float,
        equity: float,
        volatility_24h: float = 0.025,
        atr: float = 0.0,
        smce_regime: str = "EXPANSION",
        structure: str = "BULLISH",
        momentum_strong: bool = True,
        swing_low: Optional[float] = None,
        swing_high: Optional[float] = None,
        existing_positions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute EXPANSION entry parameters.

        Args:
            symbol:             Trading symbol
            entry_price:        Current market price
            equity:             Current account equity (USD)
            volatility_24h:     24h realized vol (fraction)
            atr:                Current ATR value in price units (0 = use vol proxy)
            smce_regime:        Must be 'EXPANSION'
            structure:          Must be 'BULLISH' (constitutional)
            momentum_strong:    True if momentum indicators confirm direction
            swing_low:          Recent structure swing low (for trailing stop longs)
            swing_high:         Recent structure swing high (for trailing stop shorts)
            existing_positions: Currently open symbol list (no scaling in check)

        Returns:
            {
                "allowed":              bool,
                "block_reason":         str or None,
                "allocation_pct":       float,
                "allocation_usd":       float,
                "leverage":             float,
                "trailing_stop_px":     float,
                "trailing_atr_mul":     float,
                "initial_stop_px":      float,
                "tier":                 str,
            }
        """
        if smce_regime != "EXPANSION":
            return self._blocked(f"ExpansionStrategy inactive in {smce_regime} regime")

        # Constitutional: EXPANSION requires BULLISH structure
        if structure not in ("BULLISH",):
            return self._blocked(
                f"EXPANSION requires BULLISH structure; got {structure}"
            )

        # Constitutional: momentum must be strong
        if not momentum_strong:
            return self._blocked("EXPANSION requires strong momentum confirmation")

        # High-volatility gate: don't enter trend during extreme vol
        if volatility_24h > EXPANSION_MAX_VOL:
            return self._blocked(
                f"Volatility {volatility_24h:.2%} > {EXPANSION_MAX_VOL:.0%} limit for EXPANSION"
            )

        # No scaling in – reject if symbol already held
        if existing_positions and symbol in existing_positions:
            return self._blocked(f"No scaling in: {symbol} already in portfolio")

        tier   = self._get_tier(equity)
        params = TIER_PARAMS[tier]

        # Leverage: reduce as volatility increases
        vol_norm = min(1.0, volatility_24h / EXPANSION_MAX_VOL)
        leverage = params["leverage_max"] * (1.0 - 0.4 * vol_norm)  # 60–100% of max
        leverage = round(max(1.0, min(params["leverage_max"], leverage)), 1)

        # Allocation: full band since this is a conviction trend play
        alloc_pct      = params["alloc_max"]
        allocation_usd = equity * alloc_pct

        # Trailing stop: based on structure swing low (for long)
        atr_mul = params["trailing_atr_mul"]
        if atr <= 0:
            # Proxy: use vol * entry_price as ATR
            atr = volatility_24h * entry_price

        if swing_low is not None:
            trailing_stop  = swing_low
            initial_stop   = swing_low
        else:
            # Fallback: ATR-based trailing stop below entry
            trailing_stop = entry_price - (atr * atr_mul)
            initial_stop  = trailing_stop

        result = {
            "allowed":          True,
            "block_reason":     None,
            "allocation_pct":   alloc_pct,
            "allocation_usd":   allocation_usd,
            "leverage":         leverage,
            "trailing_stop_px": trailing_stop,
            "initial_stop_px":  initial_stop,
            "trailing_atr_mul": atr_mul,
            "tier":             tier,
        }

        logger.debug(
            "[EXPANSION] %s tier=%s alloc=%.1f%% lev=%.1fx trail=%.4f",
            symbol, tier, alloc_pct * 100, leverage, trailing_stop
        )
        return result

    # ─── Private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _get_tier(equity: float) -> str:
        if equity < 500.0:
            return "SMALL"
        elif equity < 5000.0:
            return "MEDIUM"
        return "LARGE"

    @staticmethod
    def _blocked(reason: str) -> Dict:
        return {"allowed": False, "block_reason": reason}
