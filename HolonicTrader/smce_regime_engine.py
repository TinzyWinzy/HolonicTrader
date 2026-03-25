"""
SMCERegimeEngine – Sovereign Micro-Compounding Engine (Layer 1)

Classifies the current market regime based on four inputs:
  - structure:       BULLISh / BEARISH / NEUTRAL
  - entropy:         0.0–2.0 (lower = more ordered)
  - liquidity:       'healthy' / 'warning' / 'critical'
  - correlation_idx: 0.0–1.0 aggregate correlation across major assets

Outputs one of: HARVEST / EXPANSION / TRANSITION / DEFENSIVE

This is the macro gate. Strategies are only allowed to operate in
regimes that permit them. DEFENSIVE blocks ALL new entries.

Unlike the existing RegimeController (which manages capital tiers),
this engine classifies MARKET BEHAVIOUR for the SMCE pipeline.
Both systems run in parallel; both constrain risk.
"""

import time
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger("SMCERegimeEngine")

# ─── Regime Constants ─────────────────────────────────────────────────────────
HARVEST_ENTROPY_MAX       = 1.15
EXPANSION_ENTROPY_MAX     = 0.95
TRANSITION_ENTROPY_MIN    = 0.9
TRANSITION_ENTROPY_MAX    = 1.2
DEFENSIVE_ENTROPY_MIN     = 1.2
DEFENSIVE_CORRELATION_MIN = 0.7   # correlation spike threshold
HIGH_ENTROPY_LEV_SOFT     = 1.5   # SMCE_SMALL max lev when entropy in [1.0,1.5]

# Regime-level permissions returned to callers
REGIME_CONFIG: Dict[str, Dict] = {
    "HARVEST": {
        "new_entries_allowed": True,
        "strategy_modules":   ["HARVEST"],
        "min_prob_score":     5,
        "size_modifier":      1.0,
        "max_leverage_small": 3.0,
        "max_leverage_medium":4.0,
        "description":        "Low entropy, ordered market. Harvest micro-gains.",
    },
    "EXPANSION": {
        "new_entries_allowed": True,
        "strategy_modules":   ["EXPANSION"],
        "min_prob_score":     6,
        "size_modifier":      1.0,
        "max_leverage_small": 3.0,    # SMALL tier cap still applies
        "max_leverage_medium":5.0,
        "description":        "Low entropy + BULLISH. Capture trend acceleration.",
    },
    "TRANSITION": {
        "new_entries_allowed": True,
        "strategy_modules":   ["HARVEST"],   # only harvest allowed; expansion blocked
        "min_prob_score":     6,
        "size_modifier":      0.5,            # positions must be halved
        "max_leverage_small": 2.0,
        "max_leverage_medium":3.0,
        "description":        "Moderate entropy, mixed structure. Caution – size halved.",
    },
    "DEFENSIVE": {
        "new_entries_allowed": False,
        "strategy_modules":   [],
        "min_prob_score":     999,
        "size_modifier":      0.0,
        "max_leverage_small": 1.0,
        "max_leverage_medium":1.0,
        "description":        "High entropy / liquidity crisis / drawdown breach. No new trades.",
    },
}


class SMCERegimeEngine:
    """
    Layer 1 Macro Gate for the SMCE pipeline.

    Usage:
        engine = SMCERegimeEngine()
        regime = engine.classify(
            structure="NEUTRAL",
            entropy=0.8,
            liquidity_status="healthy",
            correlation_idx=0.4,
            drawdown_breach=False,
        )
        cfg = engine.get_permissions()   # dict from REGIME_CONFIG
    """

    def __init__(self):
        self.current_regime: str = "HARVEST"
        self.previous_regime: str = "HARVEST"
        self.last_update: float   = 0.0
        self.last_inputs: Dict    = {}
        self.transition_log: list = []   # [(timestamp, from, to, reason)]

    # ─── Public API ───────────────────────────────────────────────────────────

    def classify(
        self,
        structure: str,
        entropy: float,
        liquidity_status: str,
        correlation_idx: float,
        drawdown_breach: bool = False,
    ) -> str:
        """
        Classify the current market regime.

        Args:
            structure:        'BULLISH' | 'BEARISH' | 'NEUTRAL' | 'SUPPORT' | 'RESISTANCE'
            entropy:          0.0–2.0 float
            liquidity_status: 'healthy' | 'warning' | 'critical'
            correlation_idx:  0.0–1.0  aggregate cross-asset correlation
            drawdown_breach:  True if daily OR weekly drawdown limit hit

        Returns:
            Regime string: 'HARVEST' | 'EXPANSION' | 'TRANSITION' | 'DEFENSIVE'
        """
        now = time.time()
        self.last_inputs = {
            "structure":        structure,
            "entropy":          entropy,
            "liquidity_status": liquidity_status,
            "correlation_idx":  correlation_idx,
            "drawdown_breach":  drawdown_breach,
        }

        new_regime = self._compute_regime(
            structure, entropy, liquidity_status, correlation_idx, drawdown_breach
        )

        if new_regime != self.current_regime:
            reason = self._build_reason(
                structure, entropy, liquidity_status, correlation_idx, drawdown_breach
            )
            self.transition_log.append((now, self.current_regime, new_regime, reason))
            logger.info(
                "[SMCE-REGIME] %s → %s | %s",
                self.current_regime, new_regime, reason,
            )
            self.previous_regime = self.current_regime
            self.current_regime  = new_regime

        self.last_update = now
        return self.current_regime

    def get_permissions(self) -> Dict:
        """Return the config dict for the current regime."""
        return REGIME_CONFIG.get(self.current_regime, REGIME_CONFIG["DEFENSIVE"])

    def get_max_leverage(self, equity: float) -> float:
        """
        Return the regime-appropriate leverage cap for the given equity.
        The Capital Doctrine (Layer 0) imposes its own hard caps on top.
        """
        cfg = self.get_permissions()
        if equity < 500.0:          # SMALL tier
            return cfg["max_leverage_small"]
        elif equity < 5000.0:       # MEDIUM tier
            return cfg["max_leverage_medium"]
        else:                        # LARGE tier
            return min(cfg["max_leverage_medium"] + 2.0, 10.0)

    def get_status_summary(self) -> Dict:
        return {
            "regime":          self.current_regime,
            "previous_regime": self.previous_regime,
            "last_inputs":     self.last_inputs,
            "last_update":     self.last_update,
            "permissions":     self.get_permissions(),
            "recent_transitions": self.transition_log[-5:],
        }

    def force_defensive(self, reason: str = "External override"):
        """Hard-set DEFENSIVE for emergencies (drawdown breach, liquidity crisis)."""
        if self.current_regime != "DEFENSIVE":
            self.transition_log.append((time.time(), self.current_regime, "DEFENSIVE", reason))
            logger.warning("[SMCE-REGIME] Forced DEFENSIVE: %s", reason)
            self.previous_regime = self.current_regime
            self.current_regime  = "DEFENSIVE"

    # ─── Private ──────────────────────────────────────────────────────────────

    def _compute_regime(
        self,
        structure: str,
        entropy: float,
        liquidity_status: str,
        correlation_idx: float,
        drawdown_breach: bool,
    ) -> str:
        """Core classification logic – ordered from most restrictive to least."""

        # ── DEFENSIVE (highest priority) ──────────────────────────────────────
        if drawdown_breach:
            return "DEFENSIVE"
        if liquidity_status == "critical":
            return "DEFENSIVE"
        if entropy > DEFENSIVE_ENTROPY_MIN:
            return "DEFENSIVE"
        if correlation_idx > DEFENSIVE_CORRELATION_MIN and liquidity_status == "warning":
            return "DEFENSIVE"

        # ── EXPANSION ─────────────────────────────────────────────────────────
        # Requires: low entropy + BULLISH + healthy liquidity + low correlation
        if (
            entropy < EXPANSION_ENTROPY_MAX
            and structure in ("BULLISH",)
            and liquidity_status == "healthy"
            and correlation_idx < DEFENSIVE_CORRELATION_MIN
        ):
            return "EXPANSION"

        # ── TRANSITION ────────────────────────────────────────────────────────
        # Moderate entropy OR mixed signals
        if (
            TRANSITION_ENTROPY_MIN <= entropy <= TRANSITION_ENTROPY_MAX
            or correlation_idx >= DEFENSIVE_CORRELATION_MIN  # correlation rising
            or liquidity_status == "warning"
        ):
            return "TRANSITION"

        # ── HARVEST (default safe state) ──────────────────────────────────────
        # Low entropy, non-DEFENSIVE liquidity, any structure
        if entropy < HARVEST_ENTROPY_MAX and liquidity_status == "healthy":
            return "HARVEST"

        # Fallback – if none of the above matches cleanly, use TRANSITION
        return "TRANSITION"

    def _build_reason(
        self,
        structure: str,
        entropy: float,
        liquidity_status: str,
        correlation_idx: float,
        drawdown_breach: bool,
    ) -> str:
        parts = []
        if drawdown_breach:
            parts.append("drawdown_breach=True")
        parts.append(f"entropy={entropy:.2f}")
        parts.append(f"structure={structure}")
        parts.append(f"liquidity={liquidity_status}")
        parts.append(f"corr={correlation_idx:.2f}")
        return ", ".join(parts)
