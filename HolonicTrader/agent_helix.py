"""
HELIX HOLON — System Repair & Strategy Recovery (QUANT-OPS Agent)

Role: Quantitative system recovery engineer.
Input: Chronos forensic reports + Aegis security reports
Output: Fix proposals, constraint updates, config patches, action queue

Publishes to: quant_ops.fixes
Subscribes to: quant_ops.forensics (Chronos findings)
               quant_ops.security (Aegis findings)
               quant_ops.cycle (triggered by QuantOps orchestrator)

Helix does NOT assume the system is "almost working." It isolates broken
components, proposes disabling them, and outputs actionable fix instructions
that the QuantOps orchestrator or Atlas can apply.
"""

import time
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from HolonicTrader.holon_core import (
    Holon, Disposition, Message, MessageBus, get_message_bus
)

logger = logging.getLogger("Helix.Holon")


# ─── Fix action templates ────────────────────────────────────────────────────

FIX_ACTIONS = {
    # Signal quality fixes
    "raise_conviction_threshold": {
        "category": "SIGNAL",
        "description": "Raise minimum conviction threshold to filter weak signals",
        "config_key": "MINIMUM_CONVICTION_THRESHOLD",
        "priority": "HIGH",
    },
    "disable_sell_strategy": {
        "category": "SIGNAL",
        "description": "Disable SELL strategy due to negative expectancy",
        "config_key": "SELL_STRATEGY_ENABLED",
        "priority": "HIGH",
    },
    # Execution fixes
    "enforce_min_trade_threshold": {
        "category": "EXECUTION",
        "description": "Enforce minimum trade profitability threshold above execution costs",
        "config_key": "EXECUTION_COST_FILTER_ENABLED",
        "priority": "HIGH",
    },
    "increase_min_edge_multiple": {
        "category": "EXECUTION",
        "description": "Increase minimum edge multiple to cover costs",
        "config_key": "MIN_EDGE_MULTIPLE",
        "priority": "MEDIUM",
    },
    # Risk fixes
    "tighten_stop_loss": {
        "category": "RISK",
        "description": "Tighten stop-loss to reduce max single-trade loss",
        "config_key": "DEFAULT_STOP_LOSS_PCT",
        "priority": "MEDIUM",
    },
    "widen_stop_loss": {
        "category": "RISK",
        "description": "Widen stop-loss to reduce stop-hunting / friction exits",
        "config_key": "DEFAULT_STOP_LOSS_PCT",
        "priority": "MEDIUM",
    },
    "reduce_leverage": {
        "category": "RISK",
        "description": "Reduce leverage to limit drawdown magnitude",
        "config_key": "PREDATOR_LEVERAGE",
        "priority": "HIGH",
    },
    "close_stale_positions": {
        "category": "RISK",
        "description": "Force-close positions exceeding maximum age",
        "priority": "HIGH",
    },
    # Regime fixes
    "enable_regime_filter": {
        "category": "REGIME",
        "description": "Only trade when strategy matches detected market regime",
        "priority": "MEDIUM",
    },
    "reduce_chaos_exposure": {
        "category": "REGIME",
        "description": "Reduce position sizing during CHAOS/TRANSITION regimes",
        "priority": "MEDIUM",
    },
    # Security fixes (from Aegis)
    "reconcile_positions": {
        "category": "SECURITY",
        "description": "Reconcile internal positions with exchange positions immediately",
        "priority": "CRITICAL",
    },
    "switch_to_rest_api": {
        "category": "SECURITY",
        "description": "Switch to REST API due to WebSocket integrity issues",
        "config_key": "WS_FORCE_REST_ONLY",
        "priority": "HIGH",
    },
    "investigate_rl_reward": {
        "category": "SECURITY",
        "description": "Investigate RL reward signal for potential poisoning",
        "priority": "HIGH",
    },
}


class HelixHolon(Holon):
    """
    Helix Strategy Surgeon — Holon-integrated system recovery engine.

    Consumes Chronos + Aegis reports, synthesizes fix proposals,
    and publishes an action queue to quant_ops.fixes.
    """

    def __init__(
        self,
        name: str = "HelixAgent",
        message_bus: Optional[MessageBus] = None,
    ):
        super().__init__(
            name=name,
            disposition=Disposition(autonomy=0.85, integration=0.85),
            message_bus=message_bus or get_message_bus(),
        )
        self._last_chronos: Optional[Dict] = None
        self._last_aegis: Optional[Dict] = None
        self._last_report: Optional[Dict] = None
        self._report_count = 0
        self._applied_fixes: List[str] = []

        # Subscribe to upstream agents and cycle events
        self.message_bus.subscribe("quant_ops.forensics", self._on_chronos_report)
        self.message_bus.subscribe("quant_ops.security", self._on_aegis_report)
        self.message_bus.subscribe("quant_ops.cycle", self._on_cycle_event)

        logger.info(f"[{self.name}] Helix Holon initialized")

    # ------------------------------------------------------------------
    # Core diagnosis & fix generation
    # ------------------------------------------------------------------

    def generate_fixes(
        self,
        chronos_report: Optional[Dict] = None,
        aegis_report: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Analyze Chronos + Aegis reports and produce an action queue.

        Returns:
        {
            "agent": "helix",
            "timestamp": ...,
            "system_status": { ... },
            "actions": [ { "action": ..., "priority": ..., "config_key": ..., "value": ... } ],
            "constraints": [ ... ],
            "component_status": { "buy_strategy": ..., "sell_strategy": ..., ... },
        }
        """
        c_report = chronos_report or self._last_chronos or {}
        a_report = aegis_report or self._last_aegis or {}

        actions = []
        constraints = []
        component_status = {
            "buy_strategy": "ACTIVE",
            "sell_strategy": "ACTIVE",
            "execution_layer": "NOMINAL",
            "risk_management": "NOMINAL",
            "security": "SECURE",
        }

        # ── Analyze Chronos findings ──────────────────────────────────

        health = c_report.get("strategy_health", {})
        loss_causes = c_report.get("loss_causes", [])
        severity = c_report.get("severity", "UNKNOWN")
        expectancy = health.get("expectancy", 0)
        win_rate = health.get("win_rate", 0.5)
        veto = c_report.get("veto_attribution", {})

        # Negative expectancy → critical structural issue
        if expectancy < 0:
            constraints.append({
                "type": "expectancy_guard",
                "description": f"Negative expectancy ({expectancy:.4f}) — system loses money per trade",
                "severity": "CRITICAL",
            })

            # Check if SELL strategy is the problem
            for lc in loss_causes:
                if lc.get("category") == "SIGNAL" and lc.get("percentage", 0) > 30:
                    actions.append(self._build_action("raise_conviction_threshold", {
                        "suggested_value": 0.70,
                        "reason": f"Signal quality degraded — {lc['percentage']:.0f}% losses from signals",
                    }))

        # Execution cost eating edge
        exec_pct = sum(
            lc.get("percentage", 0) for lc in loss_causes
            if lc.get("category") == "EXECUTION"
        )
        if exec_pct > 25:
            actions.append(self._build_action("enforce_min_trade_threshold", {
                "reason": f"Execution costs cause {exec_pct:.0f}% of losses",
            }))
            actions.append(self._build_action("increase_min_edge_multiple", {
                "suggested_value": 4.0,
                "reason": "Need higher edge to overcome friction",
            }))
            component_status["execution_layer"] = "INEFFICIENT"

        # Risk management issues
        risk_pct = sum(
            lc.get("percentage", 0) for lc in loss_causes
            if lc.get("category") == "RISK"
        )
        if risk_pct > 25:
            risk_score = health.get("risk_management", 5)
            if risk_score < 5:
                actions.append(self._build_action("reduce_leverage", {
                    "suggested_value": 2.0,
                    "reason": f"Risk management score {risk_score:.1f}/10 — reduce exposure",
                }))
                component_status["risk_management"] = "COMPROMISED"

        # Regime mismatch
        regime_pct = sum(
            lc.get("percentage", 0) for lc in loss_causes
            if lc.get("category") == "REGIME"
        )
        if regime_pct > 20:
            actions.append(self._build_action("enable_regime_filter", {
                "reason": f"Regime mismatch causes {regime_pct:.0f}% of losses",
            }))
            actions.append(self._build_action("reduce_chaos_exposure", {
                "reason": "Reduce position size during volatile regimes",
            }))

        # Veto overprotection
        veto_assessment = veto.get("veto_assessment", "NORMAL")
        if veto_assessment in ("CRITICAL_OVERPROTECTION", "HIGH_OVERPROTECTION"):
            constraints.append({
                "type": "veto_overprotection",
                "description": f"System over-vetoing signals ({veto.get('pass_rate_pct', 0):.1f}% pass rate)",
                "severity": "HIGH",
                "suggested_action": "Lower holonic conviction threshold or reduce management mode duration",
            })

        # ── Analyze Aegis findings ────────────────────────────────────

        aegis_status = a_report.get("overall_status", "UNKNOWN")
        anomalies = a_report.get("anomalies", [])

        if aegis_status == "CRITICAL":
            component_status["security"] = "CRITICAL"
            for anomaly in anomalies:
                if anomaly.get("type") == "position_mismatch":
                    actions.append(self._build_action("reconcile_positions", {
                        "reason": anomaly.get("detail", "Position mismatch detected"),
                    }))
                elif anomaly.get("type") == "websocket_desync":
                    actions.append(self._build_action("switch_to_rest_api", {
                        "reason": anomaly.get("detail", "WebSocket integrity compromised"),
                    }))
                elif anomaly.get("type") == "reward_poisoning":
                    actions.append(self._build_action("investigate_rl_reward", {
                        "reason": anomaly.get("detail", "RL reward anomaly"),
                    }))
        elif aegis_status == "WARNING":
            component_status["security"] = "WARNING"

        # ── Sort actions by priority ──────────────────────────────────

        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        actions.sort(key=lambda a: priority_order.get(a.get("priority", "LOW"), 4))

        # Deduplicate by action name
        seen = set()
        deduped = []
        for a in actions:
            key = a.get("action", "")
            if key not in seen:
                seen.add(key)
                deduped.append(a)
        actions = deduped

        report = {
            "agent": "helix",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle_id": self._report_count,
            "system_status": {
                "chronos_severity": severity,
                "aegis_status": aegis_status,
                "expectancy": expectancy,
                "win_rate": win_rate,
            },
            "component_status": component_status,
            "actions": actions,
            "action_count": len(actions),
            "constraints": constraints,
            "constraint_count": len(constraints),
        }

        self._last_report = report
        self._report_count += 1
        return report

    def publish_report(self, report: Optional[Dict] = None) -> int:
        """Publish fix proposals to the message bus."""
        if report is None:
            report = self.generate_fixes()

        msg = Message(
            sender=self.name,
            type="fix_proposals",
            payload=report,
        )
        notified = self.message_bus.publish("quant_ops.fixes", msg)
        logger.info(
            f"[{self.name}] Published fix proposals (actions={report.get('action_count', 0)}, "
            f"constraints={report.get('constraint_count', 0)}, notified={notified})"
        )
        return notified

    # ------------------------------------------------------------------
    # Holon ABC contract
    # ------------------------------------------------------------------

    def receive_message(self, sender: Any, content: Any) -> None:
        """
        Process incoming messages.

        Supported:
        - {"type": "generate_fixes"}                → analyze + publish
        - {"type": "generate_fixes", "chronos": {...}, "aegis": {...}}  → with explicit inputs
        - {"type": "get_last_report"}               → return cached
        """
        if isinstance(content, dict):
            msg_type = content.get("type", "")

            if msg_type == "generate_fixes":
                report = self.generate_fixes(
                    chronos_report=content.get("chronos"),
                    aegis_report=content.get("aegis"),
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
            return {
                "helix_action_count": self._last_report.get("action_count", 0),
                "helix_constraint_count": self._last_report.get("constraint_count", 0),
                "helix_report_count": self._report_count,
                "helix_component_status": self._last_report.get("component_status", {}),
            }
        return {"helix_action_count": 0, "helix_report_count": 0}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_action(self, action_name: str, context: Dict) -> Dict:
        """Build a structured action from the FIX_ACTIONS registry."""
        template = FIX_ACTIONS.get(action_name, {})
        return {
            "action": action_name,
            "category": template.get("category", "OTHER"),
            "description": template.get("description", action_name),
            "priority": template.get("priority", "MEDIUM"),
            "config_key": template.get("config_key"),
            "suggested_value": context.get("suggested_value"),
            "reason": context.get("reason", ""),
        }

    def _on_chronos_report(self, message: Message) -> None:
        """Cache latest Chronos report."""
        if isinstance(message.payload, dict) and message.payload.get("agent") == "chronos":
            self._last_chronos = message.payload

    def _on_aegis_report(self, message: Message) -> None:
        """Cache latest Aegis report."""
        if isinstance(message.payload, dict) and message.payload.get("agent") == "aegis":
            self._last_aegis = message.payload

    def _on_cycle_event(self, message: Message) -> None:
        """Handle cycle trigger from QuantOps orchestrator."""
        payload = message.payload if isinstance(message.payload, dict) else {}
        if payload.get("target") in (None, "helix", "all"):
            logger.info(f"[{self.name}] Cycle event received — generating fixes")
            report = self.generate_fixes()
            self.publish_report(report)
