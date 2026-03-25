"""
ATLAS HOLON — Profit Strategist (QUANT-OPS Agent)

Role: Path-to-profitability strategist and capital allocator.
Input: Helix fix proposals, system state, trade history
Output: Capital allocation, trading rules, scaling plan

Publishes to: quant_ops.strategy
Subscribes to: quant_ops.fixes (Helix action queue)
               quant_ops.cycle (triggered by QuantOps orchestrator)

Atlas decides WHERE capital flows. It wraps the existing AtlasProfitIntegration
and AtlasCapitalManager to produce structured strategy directives. Atlas proposes;
Governor/SMCE disposes (safety rules are never overridden).
"""

import time
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from HolonicTrader.holon_core import (
    Holon, Disposition, Message, MessageBus, get_message_bus
)

logger = logging.getLogger("Atlas.Holon")


class AtlasHolon(Holon):
    """
    Atlas Profit Architect — Holon-integrated capital allocation strategist.

    Consumes Helix fix proposals + system state to produce capital allocation
    and trading rule decisions. Publishes to quant_ops.strategy.

    CRITICAL SAFETY: Atlas's decisions are PROPOSALS. They must be clamped
    by SMCE tier limits and Governor veto logic before application.
    """

    # Scaling gates — Atlas only recommends scaling after these thresholds
    MIN_TRADES_FOR_SCALING = 30
    MIN_EXPECTANCY_FOR_SCALING = 0.001
    MAX_DRAWDOWN_FOR_SCALING = 0.10  # 10%

    def __init__(
        self,
        name: str = "AtlasAgent",
        atlas_integration: Optional[Any] = None,
        capital_manager: Optional[Any] = None,
        db_manager: Optional[Any] = None,
        message_bus: Optional[MessageBus] = None,
    ):
        super().__init__(
            name=name,
            disposition=Disposition(autonomy=0.9, integration=0.9),
            message_bus=message_bus or get_message_bus(),
        )
        self._atlas = atlas_integration
        self._capital_mgr = capital_manager
        self._db = db_manager
        self._last_helix: Optional[Dict] = None
        self._last_report: Optional[Dict] = None
        self._report_count = 0

        # Subscribe to Helix fixes and cycle events
        self.message_bus.subscribe("quant_ops.fixes", self._on_helix_report)
        self.message_bus.subscribe("quant_ops.cycle", self._on_cycle_event)

        logger.info(f"[{self.name}] Atlas Holon initialized")

    # ------------------------------------------------------------------
    # Core strategy generation
    # ------------------------------------------------------------------

    def generate_strategy(
        self,
        helix_report: Optional[Dict] = None,
        system_state: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate a capital allocation and trading strategy plan.

        Returns:
        {
            "agent": "atlas",
            "timestamp": ...,
            "capital_allocation": { "buy_strategy": 0.8, "reserve": 0.2, ... },
            "rules": [ ... ],
            "scaling_decision": { "allowed": bool, "reason": ... },
            "strategy_directives": [ ... ],
            "risk_overrides": [ ... ],
        }
        """
        h_report = helix_report or self._last_helix or {}
        state = system_state or self._gather_system_state()

        capital_allocation = self._compute_capital_allocation(h_report, state)
        rules = self._compute_trading_rules(h_report, state)
        scaling = self._evaluate_scaling(state)
        directives = self._generate_directives(h_report, state, capital_allocation)
        risk_overrides = self._compute_risk_overrides(h_report, state)

        report = {
            "agent": "atlas",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle_id": self._report_count,
            "system_state_summary": {
                "equity": state.get("equity", 0),
                "expectancy": state.get("expectancy", 0),
                "win_rate": state.get("win_rate", 0),
                "total_trades": state.get("total_trades", 0),
                "max_drawdown": state.get("max_drawdown", 0),
            },
            "capital_allocation": capital_allocation,
            "rules": rules,
            "scaling_decision": scaling,
            "strategy_directives": directives,
            "risk_overrides": risk_overrides,
            "helix_actions_consumed": len(h_report.get("actions", [])),
        }

        self._last_report = report
        self._report_count += 1
        return report

    def publish_report(self, report: Optional[Dict] = None) -> int:
        """Publish strategy report to the message bus."""
        if report is None:
            report = self.generate_strategy()

        msg = Message(
            sender=self.name,
            type="strategy_plan",
            payload=report,
        )
        notified = self.message_bus.publish("quant_ops.strategy", msg)
        logger.info(
            f"[{self.name}] Published strategy plan (directives={len(report.get('strategy_directives', []))}, "
            f"notified={notified})"
        )
        return notified

    # ------------------------------------------------------------------
    # Capital allocation logic
    # ------------------------------------------------------------------

    def _compute_capital_allocation(self, helix: Dict, state: Dict) -> Dict:
        """
        Determine capital allocation across strategies.

        Atlas Rule #2: Capital flows to proven edge only.
        """
        expectancy = state.get("expectancy", 0)
        win_rate = state.get("win_rate", 0.5)
        component_status = helix.get("component_status", {})

        # Base allocation
        buy_pct = 0.80
        sell_pct = 0.0  # Default: disabled until proven
        reserve_pct = 0.20

        # If buy strategy is compromised, reduce allocation
        if component_status.get("buy_strategy") == "DISABLED":
            buy_pct = 0.0
            reserve_pct = 1.0
        elif expectancy < 0:
            # Negative expectancy → conservative mode
            buy_pct = 0.50
            reserve_pct = 0.50
        elif expectancy > 0.002 and win_rate > 0.55:
            # Strong edge → allocate more
            buy_pct = 0.85
            reserve_pct = 0.15

        # Sell strategy only gets capital if explicitly enabled AND independently profitable
        sell_enabled = component_status.get("sell_strategy", "ACTIVE") == "ACTIVE"
        if sell_enabled and expectancy > 0.001 and win_rate > 0.55:
            sell_pct = 0.10
            buy_pct -= 0.10

        allocation = {
            "buy_strategy": round(buy_pct, 2),
            "sell_strategy": round(sell_pct, 2),
            "reserve": round(reserve_pct, 2),
            "total": round(buy_pct + sell_pct + reserve_pct, 2),
        }

        # If external Atlas capital manager available, sync
        if self._capital_mgr:
            try:
                equity = state.get("equity", 0)
                if equity > 0:
                    allocation["buy_capital_usd"] = round(equity * buy_pct, 2)
                    allocation["sell_capital_usd"] = round(equity * sell_pct, 2)
                    allocation["reserve_usd"] = round(equity * reserve_pct, 2)
            except Exception as e:
                logger.warning(f"[{self.name}] Capital manager sync failed: {e}")

        return allocation

    # ------------------------------------------------------------------
    # Trading rules
    # ------------------------------------------------------------------

    def _compute_trading_rules(self, helix: Dict, state: Dict) -> List[Dict]:
        """Generate trading rules based on system state and Helix recommendations."""
        rules = []
        actions = helix.get("actions", [])
        constraints = helix.get("constraints", [])
        expectancy = state.get("expectancy", 0)

        # Always-on rules
        rules.append({
            "rule": "only_trade_high_confidence",
            "description": "Only execute trades above conviction threshold",
            "active": True,
        })

        # Dynamic rules based on Helix analysis
        for action in actions:
            action_name = action.get("action", "")

            if action_name == "enable_regime_filter":
                rules.append({
                    "rule": "regime_alignment_required",
                    "description": "Only trade when strategy matches current market regime",
                    "active": True,
                })

            elif action_name == "reduce_chaos_exposure":
                rules.append({
                    "rule": "avoid_low_volatility",
                    "description": "Reduce size during CHAOS/TRANSITION regimes",
                    "active": True,
                    "size_reduction_pct": 0.50,
                })

        # Negative expectancy → defensive rules
        if expectancy < 0:
            rules.append({
                "rule": "reduced_position_sizing",
                "description": "Reduce position sizes by 50% until expectancy positive",
                "active": True,
                "size_multiplier": 0.5,
            })

        return rules

    # ------------------------------------------------------------------
    # Scaling evaluation
    # ------------------------------------------------------------------

    def _evaluate_scaling(self, state: Dict) -> Dict:
        """Determine if the system qualifies for scaling up."""
        total_trades = state.get("total_trades", 0)
        expectancy = state.get("expectancy", 0)
        max_dd = abs(state.get("max_drawdown", 0))

        if total_trades < self.MIN_TRADES_FOR_SCALING:
            return {
                "allowed": False,
                "reason": f"Insufficient trade count ({total_trades}/{self.MIN_TRADES_FOR_SCALING})",
                "recommendation": "Continue trading at current size",
            }

        if expectancy < self.MIN_EXPECTANCY_FOR_SCALING:
            return {
                "allowed": False,
                "reason": f"Expectancy too low ({expectancy:.4f} < {self.MIN_EXPECTANCY_FOR_SCALING})",
                "recommendation": "Fix signal quality before scaling",
            }

        if max_dd > self.MAX_DRAWDOWN_FOR_SCALING:
            return {
                "allowed": False,
                "reason": f"Drawdown too high ({max_dd:.1%} > {self.MAX_DRAWDOWN_FOR_SCALING:.1%})",
                "recommendation": "Reduce risk before scaling",
            }

        return {
            "allowed": True,
            "reason": "System meets scaling criteria",
            "recommendation": "Gradually increase position size by 10-25%",
            "suggested_size_increase": 0.15,
        }

    # ------------------------------------------------------------------
    # Strategy directives
    # ------------------------------------------------------------------

    def _generate_directives(self, helix: Dict, state: Dict, allocation: Dict) -> List[Dict]:
        """Generate high-level strategy directives for the trading system."""
        directives = []

        # Capital rebalance directive
        directives.append({
            "directive": "capital_rebalance",
            "buy_pct": allocation.get("buy_strategy", 0.8),
            "sell_pct": allocation.get("sell_strategy", 0.0),
            "reserve_pct": allocation.get("reserve", 0.2),
        })

        # Process Helix actions into directives
        for action in helix.get("actions", []):
            config_key = action.get("config_key")
            suggested_value = action.get("suggested_value")
            if config_key and suggested_value is not None:
                directives.append({
                    "directive": "config_update",
                    "config_key": config_key,
                    "value": suggested_value,
                    "reason": action.get("reason", ""),
                    "priority": action.get("priority", "MEDIUM"),
                })

        return directives

    # ------------------------------------------------------------------
    # Risk overrides (clamped by SMCE — these are proposals only)
    # ------------------------------------------------------------------

    def _compute_risk_overrides(self, helix: Dict, state: Dict) -> List[Dict]:
        """
        Propose risk parameter adjustments.

        SAFETY: These are PROPOSALS only. QuantOps orchestrator must
        clamp them against SMCE tier limits before applying.
        """
        overrides = []
        expectancy = state.get("expectancy", 0)

        if expectancy < -0.005:
            overrides.append({
                "param": "position_size_multiplier",
                "proposed_value": 0.5,
                "reason": "Strong negative expectancy — halve position sizes",
                "safety_note": "Must be clamped by SMCE tier limits",
            })

        return overrides

    # ------------------------------------------------------------------
    # System state gathering
    # ------------------------------------------------------------------

    def _gather_system_state(self) -> Dict:
        """Gather current system state from available sources."""
        state = {
            "equity": 0,
            "expectancy": 0,
            "win_rate": 0.5,
            "total_trades": 0,
            "max_drawdown": 0,
        }

        # From database
        if self._db:
            try:
                trades = self._db.get_recent_trades(limit=100)
                if trades:
                    state["total_trades"] = len(trades)
                    wins = [t for t in trades if t.get("pnl", 0) > 0]
                    losses = [t for t in trades if t.get("pnl", 0) < 0]
                    state["win_rate"] = len(wins) / len(trades) if trades else 0.5

                    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
                    avg_loss = abs(sum(t["pnl"] for t in losses) / len(losses)) if losses else 0
                    state["expectancy"] = (state["win_rate"] * avg_win) - ((1 - state["win_rate"]) * avg_loss)
            except Exception as e:
                logger.warning(f"[{self.name}] DB state gathering failed: {e}")

        # From Atlas integration
        if self._atlas:
            try:
                status = getattr(self._atlas, "integration_status", {})
                state["equity"] = status.get("account_balance", 0)
            except Exception:
                pass

        # From capital manager
        if self._capital_mgr:
            try:
                dd = getattr(self._capital_mgr, "drawdown_tracker", {})
                state["max_drawdown"] = dd.get("max_drawdown", 0)
                state["equity"] = state["equity"] or dd.get("current_equity", 0)
            except Exception:
                pass

        return state

    # ------------------------------------------------------------------
    # Holon ABC contract
    # ------------------------------------------------------------------

    def receive_message(self, sender: Any, content: Any) -> None:
        """
        Process incoming messages.

        Supported:
        - {"type": "generate_strategy"}                         → analyze + publish
        - {"type": "generate_strategy", "helix": {...}, "state": {...}}
        - {"type": "get_last_report"}                           → return cached
        """
        if isinstance(content, dict):
            msg_type = content.get("type", "")

            if msg_type == "generate_strategy":
                report = self.generate_strategy(
                    helix_report=content.get("helix"),
                    system_state=content.get("state"),
                )
                self.publish_report(report)

            elif msg_type == "get_last_report":
                return self._last_report

        logger.debug(f"[{self.name}] Received message from {sender}")

    # ------------------------------------------------------------------
    # Dashboard integration
    # ------------------------------------------------------------------

    def get_dashboard_state(self) -> dict:
        if self._last_report:
            alloc = self._last_report.get("capital_allocation", {})
            scaling = self._last_report.get("scaling_decision", {})
            return {
                "atlas_buy_pct": alloc.get("buy_strategy", 0),
                "atlas_reserve_pct": alloc.get("reserve", 0),
                "atlas_scaling_allowed": scaling.get("allowed", False),
                "atlas_directive_count": len(self._last_report.get("strategy_directives", [])),
                "atlas_report_count": self._report_count,
            }
        return {"atlas_buy_pct": 0, "atlas_report_count": 0}

    # ------------------------------------------------------------------
    # Internal event handlers
    # ------------------------------------------------------------------

    def _on_helix_report(self, message: Message) -> None:
        """Cache latest Helix report."""
        if isinstance(message.payload, dict) and message.payload.get("agent") == "helix":
            self._last_helix = message.payload

    def _on_cycle_event(self, message: Message) -> None:
        """Handle cycle trigger from QuantOps orchestrator."""
        payload = message.payload if isinstance(message.payload, dict) else {}
        if payload.get("target") in (None, "atlas", "all"):
            logger.info(f"[{self.name}] Cycle event received — generating strategy")
            report = self.generate_strategy()
            self.publish_report(report)
