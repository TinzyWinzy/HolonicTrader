"""
SMCE Strategy Modules – HARVEST Mode (Primary Engine)

Goal: Small, frequent, low-risk gains (10–60¢ per trade).
Active ONLY in HARVEST regime.

Constitutional parameters (SMALL tier < $500):
  - Allocation per trade: 10–15% of equity
  - Leverage:            2–3x (lower in higher volatility)
  - Target:              0.2–0.6% from entry
  - Stop-loss:           0.4–0.8% (risk-reward ≥ 1:1)
  - No stacking:         one position per symbol at a time
  - No funding bias:     carry ignored; structure + momentum only
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("SMCE.HarvestStrategy")

# ── Constitutional parameters by equity tier ──────────────────────────────────
TIER_PARAMS = {
    "SMALL":  {   # < $500
        "alloc_min":   0.10,
        "alloc_max":   0.15,
        "leverage_min":2.0,
        "leverage_max":3.0,
        "target_min":  0.002,   # 0.2%
        "target_max":  0.006,   # 0.6%
        "sl_min":      0.004,   # 0.4%
        "sl_max":      0.008,   # 0.8%
    },
    "MEDIUM": {   # $500–$5000
        "alloc_min":   0.10,
        "alloc_max":   0.15,
        "leverage_min":2.0,
        "leverage_max":4.0,
        "target_min":  0.002,
        "target_max":  0.008,
        "sl_min":      0.003,
        "sl_max":      0.006,
    },
    "LARGE":  {   # > $5000
        "alloc_min":   0.10,
        "alloc_max":   0.15,
        "leverage_min":2.0,
        "leverage_max":5.0,
        "target_min":  0.002,
        "target_max":  0.010,
        "sl_min":      0.003,
        "sl_max":      0.005,
    },
}

# Min risk-reward ratio (constitutional non-negotiable)
MIN_RISK_REWARD = 1.0


class HarvestStrategy:
    """
    HARVEST mode strategy: micro-compounding entry parameters.

    This class computes ENTRY PARAMETERS only. It does not execute trades.
    It enforces the constitutional rules and returns a parameter dict that
    the Governor uses for final sizing.
    """

    def compute_entry(
        self,
        symbol: str,
        entry_price: float,
        equity: float,
        volatility_24h: float = 0.02,
        smce_regime: str = "HARVEST",
        structure: str = "NEUTRAL",
        momentum_aligned: bool = True,
        existing_position: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Compute HARVEST entry parameters.

        Args:
            symbol:            Trading symbol ('BTC/USDT')
            entry_price:       Current market price
            equity:            Current account equity (USD)
            volatility_24h:    24h realized volatility (fraction, e.g. 0.03 = 3%)
            smce_regime:       Must be 'HARVEST' or returns a blocked result
            structure:         Market structure ('BULLISH'/'BEARISH'/'NEUTRAL')
            momentum_aligned:  True if RSI/MACD confirm direction
            existing_position: If not None, check stacking rule → block if same symbol

        Returns:
            {
                "allowed":        bool,
                "block_reason":   str or None,
                "allocation_pct": float,   # fraction of equity to allocate
                "allocation_usd": float,   # USD amount
                "leverage":       float,
                "target_pct":     float,   # take-profit from entry
                "stop_loss_pct":  float,   # stop-loss from entry
                "take_profit_px": float,
                "stop_loss_px":   float,
                "risk_reward":    float,
                "tier":           str,
            }
        """
        # Only active in HARVEST (constitutional rule)
        if smce_regime != "HARVEST":
            return self._blocked(f"HarvestStrategy inactive in {smce_regime} regime")

        # No stacking – one position per symbol
        if existing_position is not None:
            return self._blocked(
                f"No stacking: position already held in {symbol}"
            )

        tier   = self._get_tier(equity)
        params = TIER_PARAMS[tier]

        # Scale leverage inversely with volatility
        leverage = self._scale_leverage(
            volatility_24h,
            params["leverage_min"],
            params["leverage_max"],
        )

        # Scale allocation – use lower end in TRANSITION/uncertain structure
        if structure in ("BEARISH", "RESISTANCE"):
            alloc_pct = params["alloc_min"]
        else:
            alloc_pct = params["alloc_max"] if momentum_aligned else (
                params["alloc_min"] + params["alloc_max"]) / 2.0

        allocation_usd = equity * alloc_pct

        # Target and SL scale with volatility (tighter in low-vol, wider in high-vol)
        vol_scale  = min(2.0, max(0.5, volatility_24h / 0.020))  # normalized to 2% baseline
        target_pct = min(
            params["target_max"],
            max(params["target_min"], params["target_min"] * vol_scale)
        )
        sl_pct = min(
            params["sl_max"],
            max(params["sl_min"], params["sl_min"] * vol_scale)
        )

        # Enforce minimum R:R
        if target_pct / sl_pct < MIN_RISK_REWARD:
            target_pct = sl_pct * MIN_RISK_REWARD

        risk_reward   = target_pct / sl_pct
        take_profit   = entry_price * (1 + target_pct)
        stop_loss     = entry_price * (1 - sl_pct)

        result = {
            "allowed":        True,
            "block_reason":   None,
            "allocation_pct": alloc_pct,
            "allocation_usd": allocation_usd,
            "leverage":       leverage,
            "target_pct":     target_pct,
            "stop_loss_pct":  sl_pct,
            "take_profit_px": take_profit,
            "stop_loss_px":   stop_loss,
            "risk_reward":    risk_reward,
            "tier":           tier,
        }

        logger.debug(
            "[HARVEST] %s tier=%s alloc=%.1f%% lev=%.1fx tp=%.3f%% sl=%.3f%% rr=%.2f",
            symbol, tier, alloc_pct * 100, leverage, target_pct * 100, sl_pct * 100, risk_reward
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
    def _scale_leverage(vol: float, lev_min: float, lev_max: float) -> float:
        """Higher volatility → lower leverage (inverse scaling)."""
        # vol=0.01 → max leverage; vol=0.06 → min leverage
        vol_norm = min(1.0, max(0.0, (vol - 0.01) / 0.05))
        leverage  = lev_max - vol_norm * (lev_max - lev_min)
        return round(max(lev_min, min(lev_max, leverage)), 1)

    @staticmethod
    def _blocked(reason: str) -> Dict:
        return {"allowed": False, "block_reason": reason}
