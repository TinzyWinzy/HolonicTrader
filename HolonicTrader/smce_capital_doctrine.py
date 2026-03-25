"""
SMCE Capital Doctrine – Layer 0 Enforcement

Hard-coded constitutional rules that CANNOT be overridden by any strategy,
whale signal, APY boost, or override mechanism.

Rules (SMALL tier < $500):
  - Max total exposure (all positions): 50% of current equity (100% for ARB)
  - Max per asset:                      20% of equity
  - Max correlated cluster:             25% of equity
  - Max leverage (NORMAL):              3x
  - Max leverage (TRANSITION):          3x (increased for arb viability)
  - Max leverage (HIGH_ENTROPY/1.0-1.5):1–2x (graduated)
  - No stacking within 0.5% of existing entry
  - Daily drawdown limit:               3% of day-start equity
  - Weekly drawdown limit:              6% of week-start equity

FIX BUG-007 2026-03-01:
  - ARB strategies (market-neutral) have 2x exposure limit (market-neutral risk)
  - ARB positions have different risk profile and should not be constrained same as directional

FIX 2026-03-03:
  - MICRO tier (<$200): 65% total exposure, 30% per asset (min order viability)
  - SMALL tier increased from 30% to 50% (small accounts need meaningful positions)
  - TRANSITION leverage increased from 2x to 3x (allow arb sizing)

Drawdown Breach Response:
  - Risk multiplier = 0.5
  - Enter DEFENSIVE mode (48h cooldown)
  - Block all new trades during cooldown
  - After cooldown: return to normal gradually
"""

import time
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone

logger = logging.getLogger("SMCECapitalDoctrine")

# ── Tier thresholds ───────────────────────────────────────────────────────────
TIER_SMALL_CEILING  = 500.0
TIER_MEDIUM_CEILING = 5000.0
TIER_MICRO_THRESHOLD = 200.0  # FIX 2026-03-03: Micro-tier for accounts <$200

# ── Constitutional limits by tier ─────────────────────────────────────────────
TIER_LIMITS = {
    "MICRO": {  # FIX 2026-03-03: For accounts <$200 - min order viability
        "max_total_exposure": 0.65,       # 65% of equity (was 30%)
        "max_per_asset":      0.30,       # 30% per asset (was 10%)
        "max_cluster":        0.35,       # 35% correlated cluster
        "max_leverage": {
            "HARVEST":    3.0,
            "EXPANSION":  3.0,
            "TRANSITION": 3.0,  # Increased from 2.0 for arb viability
            "DEFENSIVE":  1.0,
        },
    },
    "SMALL": {  # FIX 2026-03-03: Increased limits for meaningful positions
        "max_total_exposure": 0.50,       # 50% of equity (was 30%)
        "max_per_asset":      0.20,       # 20% per asset (was 10%)
        "max_cluster":        0.25,       # 25% correlated cluster (was 15%)
        "max_leverage": {
            "HARVEST":    3.0,
            "EXPANSION":  3.0,
            "TRANSITION": 3.0,  # Increased from 2.0 for arb viability
            "DEFENSIVE":  1.0,
        },
    },
    "MEDIUM": {
        "max_total_exposure": 0.40,
        "max_per_asset":      0.15,
        "max_cluster":        0.25,
        "max_leverage": {
            "HARVEST":    4.0,
            "EXPANSION":  5.0,
            "TRANSITION": 3.0,
            "DEFENSIVE":  1.0,
        },
    },
    "LARGE": {
        "max_total_exposure": 0.50,
        "max_per_asset":      0.20,
        "max_cluster":        0.30,
        "max_leverage": {
            "HARVEST":    5.0,
            "EXPANSION":  5.0,
            "TRANSITION": 3.0,
            "DEFENSIVE":  1.0,
        },
    },
}

DAILY_DD_LIMIT       = 0.03          # 3% of day-start equity
WEEKLY_DD_LIMIT      = 0.06          # 6% of week-start equity
DEFENSIVE_COOLDOWN_H = 48            # 48 hours cooldown after breach
STACKING_PRICE_BUF   = 0.005         # 0.5% price proximity block
RISK_MULT_DEFENSIVE  = 0.5           # risk multiplier during defensive


class SMCECapitalDoctrine:
    """
    Layer 0 enforcement object.

    Mount one instance inside GovernorHolon so that every sizing call
    passes through the constitutional rules before any regime or strategy
    logic is applied.

    State survives restarts via get_state() / load_state().
    """

    def __init__(self):
        now = time.time()

        # ── Drawdown tracking ──────────────────────────────────────────────────
        self.day_start_equity:    float = 0.0
        self.week_start_equity:   float = 0.0
        self.last_day_reset:      float = now
        self.last_week_reset:     float = now

        # ── Defensive cooldown ─────────────────────────────────────────────────
        self.defensive_until:     float = 0.0   # Unix timestamp
        self.risk_multiplier:     float = 1.0
        self.consecutive_clean_days: int = 0

        # ── Scaling tracker ────────────────────────────────────────────────────
        self.period_max_drawdown: float = 0.0
        self.weekly_returns:      list  = []     # list of weekly return fractions
        self._allocation_boost:   float = 0.0   # earned via 60-day streak

        # ── Violation counter (for digest) ────────────────────────────────────
        self.violations: list = []   # [(timestamp, rule, detail)]

    # ─── Public API ───────────────────────────────────────────────────────────

    def update(self, current_equity: float) -> Dict[str, Any]:
        """
        Called every cycle. Resets daily/weekly baselines and checks drawdown.
        Returns a status dict including whether DEFENSIVE mode is active.
        """
        now = time.time()
        self._reset_baselines(current_equity, now)

        breach, reason = self._check_drawdown(current_equity)
        if breach:
            self._trigger_defensive(reason, current_equity)

        in_cooldown = now < self.defensive_until
        if not in_cooldown and self.risk_multiplier < 1.0:
            # Gradually restore after cooldown
            self.risk_multiplier = min(1.0, self.risk_multiplier + 0.05)

        return {
            "in_defensive":    in_cooldown,
            "defensive_until": self.defensive_until,
            "risk_multiplier": self.risk_multiplier,
            "equity_tier":     self._get_tier(current_equity),
        }

    def check_trade(
        self,
        symbol: str,
        direction: str,
        proposed_notional: float,
        proposed_leverage: float,
        equity: float,
        smce_regime: str,
        current_positions: Dict[str, Any],
        cluster_exposure_pct: float = 0.0,
        strategy: str = 'DIRECTIONAL',  # FIX BUG-007: Add strategy parameter
    ) -> Tuple[bool, str, float]:
        """
        Constitutional pre-flight check for a proposed trade.

        Returns:
            (allowed: bool, reason: str, max_leverage: float)
        """
        now = time.time()

        # 1. Defensive cooldown: block all entries
        if now < self.defensive_until:
            hrs_remaining = (self.defensive_until - now) / 3600
            return False, f"DEFENSIVE cooldown: {hrs_remaining:.1f}h remaining", 1.0

        tier   = self._get_tier(equity)
        limits = TIER_LIMITS[tier]

        # FIX BUG-007: ARB strategies have higher exposure limits (market-neutral risk)
        is_arb_strategy = strategy in ['ARBITRAGE_GOLD', 'FUNDING_CARRY', 'BASIS_TRADE', 'ARB', 'ARBITRAGE'] or \
                          'ARBITRAGE' in strategy.upper() or 'FUNDING' in strategy.upper() or 'BASIS' in strategy.upper()
        
        # ARB positions get 2x exposure limit (60% vs 30% for SMALL tier)
        exposure_multiplier = 2.0 if is_arb_strategy else 1.0
        effective_max_exposure = limits["max_total_exposure"] * exposure_multiplier

        # DEBUG: Log exposure check context
        logger.info(f"[Layer0] check_trade: symbol={symbol}, equity=${equity:.2f}, "
                    f"proposed_notional=${proposed_notional:.2f}, proposed_leverage={proposed_leverage}, "
                    f"num_positions={len(current_positions)}, strategy={strategy}, is_arb={is_arb_strategy}")

        # 2. Leverage cap
        max_lev = limits["max_leverage"].get(smce_regime, 1.0)
        if proposed_leverage > max_lev:
            proposed_leverage = max_lev   # will be passed back as capped value
            # (Not a veto – just inform caller to cap)

        # 3. Max total exposure (using MARGIN, not notional)
        # DEBUG: Log position details for exposure calculation
        position_margins = []
        for p in current_positions.values():
            margin = self._pos_margin(p)
            position_margins.append((
                getattr(p, 'symbol', 'unknown'),
                getattr(p, 'quantity', 0),
                getattr(p, 'entry_price', 0),
                getattr(p, 'leverage', 1),
                margin
            ))
        if position_margins:
            logger.debug(f"[Layer0] Position margins: {position_margins}")

        total_margin = sum(
            abs(self._pos_margin(p))
            for p in current_positions.values()
        )
        proposed_margin = proposed_notional / max(proposed_leverage, 1.0)
        total_exposure = total_margin + proposed_margin
        max_exposure = equity * effective_max_exposure  # FIX BUG-007: Use effective limit
        if total_exposure > max_exposure:
            reason = (
                f"[Layer0] Total exposure ${total_exposure:.2f} > "
                f"${max_exposure:.2f} ({effective_max_exposure*100:.0f}% of equity)"
            )
            self._log_violation("EXPOSURE_CEILING", reason)
            return False, reason, max_lev

        # 4. Max per-asset (using MARGIN, not notional)
        proposed_margin = proposed_notional / max(proposed_leverage, 1.0)
        if proposed_margin > equity * limits["max_per_asset"]:
            reason = (
                f"[Layer0] {symbol} margin ${proposed_margin:.2f} > "
                f"${equity * limits['max_per_asset']:.2f} ({limits['max_per_asset']*100:.0f}% cap)"
            )
            self._log_violation("PER_ASSET_CAP", reason)
            return False, reason, max_lev

        # 5. Cluster cap
        if cluster_exposure_pct > limits["max_cluster"]:
            reason = (
                f"[Layer0] Cluster {cluster_exposure_pct*100:.1f}% > "
                f"{limits['max_cluster']*100:.0f}% cap"
            )
            self._log_violation("CLUSTER_CAP", reason)
            return False, reason, max_lev

        # 6. Stacking proximity (0.5% price buffer)
        stacking_block, stack_reason = self._check_stacking(
            symbol, direction, current_positions
        )
        if stacking_block:
            self._log_violation("STACKING_BLOCK", stack_reason)
            return False, stack_reason, max_lev

        return True, "OK", max_lev

    def clear_defensive(self, reason: str = "Manual override") -> bool:
        """
        FIX 2026-03-04: Clear a DEFENSIVE cooldown triggered by a false positive.

        Called by GovernorHolon.clear_defensive_cooldown() so both the Governor's
        own timestamp and this Doctrine object are cleared in sync.

        Returns True if a cooldown was active and cleared, False if already clear.
        """
        if time.time() < self.defensive_until:
            logger.warning(
                "[Layer0] ⚠️ DEFENSIVE COOLDOWN CLEARED: %s (was until %s)",
                reason,
                datetime.fromtimestamp(self.defensive_until, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            )
            self.defensive_until = 0.0
            self.risk_multiplier = 1.0
            return True
        return False

    def reset_baselines_now(self, equity: float, reason: str = "Manual reset") -> None:
        """
        FIX 2026-03-05: Force-reset daily and weekly equity baselines to current live equity.

        Called at startup after a confirmed equity sync to prevent stale DB values from
        triggering false drawdown calculations. The consecutive clean-day streak is preserved
        because a reset is not the same as a loss.
        """
        now = time.time()
        old_day  = self.day_start_equity
        old_week = self.week_start_equity
        self.day_start_equity  = equity
        self.week_start_equity = equity
        self.last_day_reset    = now
        self.last_week_reset   = now
        logger.warning(
            "[Layer0] 🔄 BASELINES RESET: %s | day $%.2f→$%.2f | week $%.2f→$%.2f",
            reason, old_day, equity, old_week, equity,
        )

    def get_max_leverage(self, equity: float, smce_regime: str) -> float:

        """Return the constitutional leverage cap for current tier + regime."""
        tier = self._get_tier(equity)
        return TIER_LIMITS[tier]["max_leverage"].get(smce_regime, 1.0)

    def get_allocation_pct(self, base_pct: float, equity: float) -> float:
        """Apply allocation boost (earned via 60-day streak) to base allocation."""
        tier = self._get_tier(equity)
        max_boost = 0.25 - base_pct          # never exceed 25% per trade in LARGE
        boosted   = base_pct + min(max_boost, self._allocation_boost)
        return round(min(0.25, boosted), 4)

    def record_scaling_check(self, max_drawdown_this_period: float, weekly_return: float):
        """
        Called weekly to check if the 60-day streak allows allocation scaling.
        Returns True if a +2% allocation boost was granted.
        """
        self.weekly_returns.append(weekly_return)
        self.period_max_drawdown = max(self.period_max_drawdown, abs(max_drawdown_this_period))

        # Keep only last 60 days of weekly returns (8.57 weeks ≈ 9 entries)
        if len(self.weekly_returns) > 9:
            self.weekly_returns.pop(0)

        # Check conditions for scaling
        if self._check_scaling_conditions():
            self._allocation_boost = min(0.10, self._allocation_boost + 0.02)  # +2%, max +10%
            logger.info(
                "[Layer0] 📈 Scaling boost applied: +2%% → total boost %.0f%%",
                self._allocation_boost * 100
            )
            return True
        return False

    def get_violations_today(self) -> list:
        """Return violations recorded in the last 24h for the digest."""
        cutoff = time.time() - 86400
        return [v for v in self.violations if v[0] > cutoff]

    def get_state(self) -> Dict:
        """Serialise state for DB persistence."""
        return {
            "day_start_equity":       self.day_start_equity,
            "week_start_equity":      self.week_start_equity,
            "last_day_reset":         self.last_day_reset,
            "last_week_reset":        self.last_week_reset,
            "defensive_until":        self.defensive_until,
            "risk_multiplier":        self.risk_multiplier,
            "consecutive_clean_days": self.consecutive_clean_days,
            "period_max_drawdown":    self.period_max_drawdown,
            "allocation_boost":       self._allocation_boost,
        }

    def load_state(self, state: Dict):
        """Restore state from DB after restart."""
        self.day_start_equity       = state.get("day_start_equity", 0.0)
        self.week_start_equity      = state.get("week_start_equity", 0.0)
        self.last_day_reset         = state.get("last_day_reset", time.time())
        self.last_week_reset        = state.get("last_week_reset", time.time())
        self.defensive_until        = state.get("defensive_until", 0.0)
        self.risk_multiplier        = state.get("risk_multiplier", 1.0)
        self.consecutive_clean_days = state.get("consecutive_clean_days", 0)
        self.period_max_drawdown    = state.get("period_max_drawdown", 0.0)
        self._allocation_boost      = state.get("allocation_boost", 0.0)

        if time.time() < self.defensive_until:
            logger.warning(
                "[Layer0] Resuming DEFENSIVE cooldown (%.1f h remaining)",
                (self.defensive_until - time.time()) / 3600
            )

    # ─── Private ──────────────────────────────────────────────────────────────

    def _reset_baselines(self, equity: float, now: float):
        """Reset daily/weekly equity baselines at UTC midnight / week boundary."""
        # Daily reset (every 24h)
        if now - self.last_day_reset >= 86400:
            if self.day_start_equity > 0 and equity >= self.day_start_equity:
                # Clean day – increment streak
                self.consecutive_clean_days += 1
            self.day_start_equity = equity
            self.last_day_reset   = now

        elif self.day_start_equity == 0:
            self.day_start_equity = equity

        # Weekly reset (every 7 days)
        if now - self.last_week_reset >= 604800:
            if self.week_start_equity > 0:
                weekly_ret = (equity - self.week_start_equity) / max(self.week_start_equity, 1)
                self.weekly_returns.append(weekly_ret)
            self.week_start_equity = equity
            self.last_week_reset   = now
            self.period_max_drawdown = 0.0   # reset period dd

        elif self.week_start_equity == 0:
            self.week_start_equity = equity

    def _check_drawdown(self, equity: float) -> Tuple[bool, str]:
        """Returns (breach, reason)."""
        if self.day_start_equity <= 0 or self.week_start_equity <= 0:
            return False, ""

        daily_dd  = (self.day_start_equity - equity) / self.day_start_equity
        weekly_dd = (self.week_start_equity - equity) / max(self.week_start_equity, 1)

        if daily_dd >= DAILY_DD_LIMIT:
            return True, f"Daily drawdown {daily_dd:.2%} ≥ {DAILY_DD_LIMIT:.0%} limit"
        if weekly_dd >= WEEKLY_DD_LIMIT:
            return True, f"Weekly drawdown {weekly_dd:.2%} ≥ {WEEKLY_DD_LIMIT:.0%} limit"

        # Track max drawdown for scaling check
        session_dd = max(daily_dd, weekly_dd)
        self.period_max_drawdown = max(self.period_max_drawdown, session_dd)
        return False, ""

    def _trigger_defensive(self, reason: str, equity: float):
        cooldown_end = time.time() + DEFENSIVE_COOLDOWN_H * 3600
        if cooldown_end > self.defensive_until:   # don't shorten existing cooldown
            self.defensive_until = cooldown_end
            self.risk_multiplier = RISK_MULT_DEFENSIVE
            self.consecutive_clean_days = 0   # reset streak
            logger.warning(
                "[Layer0] ⚠️ DEFENSIVE ACTIVATED (%s) | 48h cooldown until %s",
                reason,
                datetime.fromtimestamp(cooldown_end, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            )
            self._log_violation("DRAWDOWN_BREACH", reason)

    def _check_stacking(
        self,
        symbol: str,
        direction: str,
        current_positions: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Block if an existing position already exists in the same symbol+direction."""
        for sym, pos in current_positions.items():
            if sym != symbol:
                continue
            pos_dir = self._pos_get(pos, "direction", "BUY")
            if pos_dir != direction:
                continue   # opposite direction = not a stack
            entry = self._pos_get(pos, "entry_price", 0.0)
            if float(entry) <= 0:
                continue
            # "No stacking" in SMCE means: one position per symbol, period
            return True, (
                f"[Layer0] STACKING BLOCKED: position in {symbol} already exists "
                f"(entry={float(entry):.4f})"
            )
        return False, ""

    @staticmethod
    def _pos_get(pos: Any, key: str, default=None):
        """Duck-typed accessor for both dict and Position-like objects."""
        if isinstance(pos, dict):
            return pos.get(key, default)
        return getattr(pos, key, default)

    @staticmethod
    def _pos_notional(pos: Any) -> float:
        """Extract notional value from a dict or Position-like object."""
        if isinstance(pos, dict):
            qty = abs(pos.get("quantity", 0.0))
            price = pos.get("entry_price", 0.0)
            return pos.get("notional", qty * price)
        # Position object: try notional attribute first, then quantity*entry_price
        notional = getattr(pos, "notional", None)
        if notional is not None:
            return float(notional)
        qty   = abs(float(getattr(pos, "quantity", 0.0)))  # Use abs() to handle short positions
        price = float(getattr(pos, "entry_price", 0.0))
        return qty * price

    @staticmethod
    def _pos_margin(pos: Any) -> float:
        """Extract margin requirement (notional / leverage) from a position."""
        notional = SMCECapitalDoctrine._pos_notional(pos)
        leverage = SMCECapitalDoctrine._pos_get(pos, "leverage", 1.0)
        if leverage <= 0:
            leverage = 1.0
        return notional / leverage

    def _check_scaling_conditions(self) -> bool:
        """60-day streak + drawdown + variance conditions for allocation scaling."""
        if self.consecutive_clean_days < 60:
            return False
        if self.period_max_drawdown >= 0.08:    # max DD < 8%
            return False
        if len(self.weekly_returns) < 4:
            return False
        variance = sum((r - sum(self.weekly_returns)/len(self.weekly_returns))**2
                       for r in self.weekly_returns) / len(self.weekly_returns)
        if variance >= 0.05**2:                 # weekly variance < 5%
            return False
        return True

    def _log_violation(self, rule: str, detail: str):
        self.violations.append((time.time(), rule, detail))
        # Keep only last 1000 violations
        if len(self.violations) > 1000:
            self.violations = self.violations[-1000:]

    @staticmethod
    def _get_tier(equity: float) -> str:
        """Determine equity tier for constitutional limits."""
        if equity < TIER_MICRO_THRESHOLD:  # FIX 2026-03-03: Micro-tier <$200
            return "MICRO"
        elif equity < TIER_SMALL_CEILING:
            return "SMALL"
        elif equity < TIER_MEDIUM_CEILING:
            return "MEDIUM"
        return "LARGE"
