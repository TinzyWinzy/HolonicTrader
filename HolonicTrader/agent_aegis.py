"""
AEGIS HOLON — Security & Integrity Monitor (QUANT-OPS Agent)

Role: Red-team security sentinel for trading infrastructure.
Input: Execution logs, exchange API responses, position data, Chronos reports
Output: Vulnerabilities, anomalies, exploit risks, integrity scores

Publishes to: quant_ops.security
Subscribes to: quant_ops.forensics (reads Chronos findings)
               quant_ops.cycle (triggered by QuantOps orchestrator)

Consolidates the four AEGIS phases (log integrity, position reconciliation,
timestamp oracle, RL agent security) under one Holon interface.
"""

import time
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from HolonicTrader.holon_core import (
    Holon, Disposition, Message, MessageBus, get_message_bus
)

logger = logging.getLogger("Aegis.Holon")


class AegisHolon(Holon):
    """
    Aegis QuantSec — Holon-integrated security & integrity monitor.

    Wraps the existing AEGIS security components (log integrity, position
    reconciliation, timestamp oracle, RL security) and publishes consolidated
    security reports to quant_ops.security.
    """

    # Severity thresholds
    CRITICAL_ANOMALY_TYPES = {"causality_violation", "position_mismatch", "ledger_tamper"}
    HIGH_ANOMALY_TYPES = {"timestamp_drift", "websocket_desync", "reward_poisoning"}

    def __init__(
        self,
        name: str = "AegisAgent",
        aegis_components: Optional[Dict[str, Any]] = None,
        executor: Optional[Any] = None,
        message_bus: Optional[MessageBus] = None,
    ):
        super().__init__(
            name=name,
            disposition=Disposition(autonomy=0.7, integration=0.9),
            message_bus=message_bus or get_message_bus(),
        )
        self._components = aegis_components or {}
        self._executor = executor
        self._last_report: Optional[Dict] = None
        self._last_chronos_report: Optional[Dict] = None
        self._report_count = 0
        self._anomaly_log: List[Dict] = []

        # Subscribe to Chronos forensics and cycle events
        self.message_bus.subscribe("quant_ops.forensics", self._on_chronos_report)
        self.message_bus.subscribe("quant_ops.cycle", self._on_cycle_event)

        logger.info(f"[{self.name}] Aegis Holon initialized (components={'enabled' if self._components.get('enabled') else 'disabled'})")

    # ------------------------------------------------------------------
    # Component access helpers
    # ------------------------------------------------------------------

    def _get_log_manager(self):
        return self._components.get("log_manager")

    def _get_recon_engine(self):
        return self._components.get("reconciliation_engine")

    def _get_ws_monitor(self):
        return self._components.get("websocket_monitor")

    def _get_timestamp_oracle(self):
        return self._components.get("timestamp_oracle")

    def _get_rl_security(self) -> Dict:
        return self._components.get("rl_security", {})

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def run_security_scan(self) -> Dict[str, Any]:
        """
        Run a comprehensive security scan across all AEGIS phases.

        Returns structured report:
        {
            "agent": "aegis",
            "timestamp": ...,
            "overall_status": "SECURE" | "WARNING" | "CRITICAL",
            "anomalies": [ ... ],
            "integrity": { "log_chain": ..., "positions": ..., "websocket": ... },
            "rl_security": { ... },
            "recommendations": [ ... ],
            "chronos_correlation": { ... }
        }
        """
        anomalies = []
        integrity = {}
        rl_report = {}
        recommendations = []

        # --- Phase 1: Log Integrity ---
        log_mgr = self._get_log_manager()
        if log_mgr:
            try:
                log_report = log_mgr.get_integrity_report()
                integrity["log_chain"] = {
                    "status": "VERIFIED" if log_report.get("verified", True) else "TAMPERED",
                    "total_entries": log_report.get("total_entries", 0),
                    "hash_mismatches": log_report.get("hash_mismatches", 0),
                    "gaps_detected": log_report.get("gaps_detected", 0),
                }
                if log_report.get("hash_mismatches", 0) > 0:
                    anomalies.append({
                        "type": "ledger_tamper",
                        "severity": "CRITICAL",
                        "detail": f"{log_report['hash_mismatches']} hash mismatches in execution log",
                    })
                    recommendations.append("IMMEDIATE: Investigate log chain tampering — potential data integrity breach")
            except Exception as e:
                integrity["log_chain"] = {"status": "ERROR", "error": str(e)}
                logger.warning(f"[{self.name}] Log integrity check failed: {e}")
        else:
            integrity["log_chain"] = {"status": "NOT_INITIALIZED"}

        # --- Phase 2: Position Reconciliation ---
        recon = self._get_recon_engine()
        if recon:
            try:
                recon_report = recon.get_latest_report()
                if recon_report:
                    report_dict = recon_report.to_dict() if hasattr(recon_report, "to_dict") else {}
                    integrity["positions"] = {
                        "status": report_dict.get("status", "UNKNOWN"),
                        "ghost_positions": report_dict.get("ghost_positions", []),
                        "leak_positions": report_dict.get("leak_positions", []),
                        "mismatches": report_dict.get("mismatches", 0),
                    }
                    if report_dict.get("mismatches", 0) > 0:
                        anomalies.append({
                            "type": "position_mismatch",
                            "severity": "CRITICAL",
                            "detail": f"{report_dict['mismatches']} position mismatches vs exchange",
                        })
                        recommendations.append("CRITICAL: Reconcile positions with exchange immediately")
                else:
                    integrity["positions"] = {"status": "NO_REPORT"}
            except Exception as e:
                integrity["positions"] = {"status": "ERROR", "error": str(e)}
                logger.warning(f"[{self.name}] Position reconciliation check failed: {e}")
        else:
            integrity["positions"] = {"status": "NOT_INITIALIZED"}

        # --- Phase 3: Websocket & Timestamp Integrity ---
        ws_mon = self._get_ws_monitor()
        if ws_mon:
            try:
                ws_report = ws_mon.get_integrity_report()
                ws_dict = ws_report.to_dict() if hasattr(ws_report, "to_dict") else {}
                integrity["websocket"] = {
                    "status": ws_dict.get("status", "UNKNOWN"),
                    "disconnects": ws_dict.get("disconnects", 0),
                    "latency_p99_ms": ws_dict.get("latency_p99_ms", 0),
                    "book_mismatches": ws_dict.get("book_mismatches", 0),
                }
                if ws_dict.get("book_mismatches", 0) > 0:
                    anomalies.append({
                        "type": "websocket_desync",
                        "severity": "HIGH",
                        "detail": f"{ws_dict['book_mismatches']} orderbook mismatches (WS vs REST)",
                    })
                    recommendations.append("HIGH: WebSocket feed may be corrupted — switch to REST fallback")
            except Exception as e:
                integrity["websocket"] = {"status": "ERROR", "error": str(e)}
        else:
            integrity["websocket"] = {"status": "NOT_INITIALIZED"}

        ts_oracle = self._get_timestamp_oracle()
        if ts_oracle:
            try:
                drift = getattr(ts_oracle, "get_drift_ms", lambda: 0)()
                integrity["timestamp"] = {"drift_ms": drift, "status": "OK" if abs(drift) < 50 else "DRIFT"}
                if abs(drift) > 50:
                    anomalies.append({
                        "type": "timestamp_drift",
                        "severity": "HIGH",
                        "detail": f"Clock drift {drift:.0f}ms exceeds 50ms threshold",
                    })
            except Exception as e:
                integrity["timestamp"] = {"status": "ERROR", "error": str(e)}
        else:
            integrity["timestamp"] = {"status": "NOT_INITIALIZED"}

        # --- Phase 4: RL Agent Security ---
        rl_sec = self._get_rl_security()
        for agent_name, secured_agent in rl_sec.items():
            try:
                sec_report = secured_agent.get_security_report()
                agent_dict = sec_report.to_dict() if hasattr(sec_report, "to_dict") else {}
                rl_report[agent_name] = {
                    "status": agent_dict.get("status", "UNKNOWN"),
                    "reward_anomalies": agent_dict.get("reward_anomalies", 0),
                    "action_overrides": agent_dict.get("action_overrides", 0),
                }
                if agent_dict.get("reward_anomalies", 0) > 3:
                    anomalies.append({
                        "type": "reward_poisoning",
                        "severity": "HIGH",
                        "detail": f"RL agent '{agent_name}' shows {agent_dict['reward_anomalies']} reward anomalies",
                    })
                    recommendations.append(f"HIGH: Investigate reward signal for {agent_name} — possible poisoning")
            except Exception as e:
                rl_report[agent_name] = {"status": "ERROR", "error": str(e)}

        # --- Chronos Correlation ---
        chronos_correlation = {}
        if self._last_chronos_report:
            c_severity = self._last_chronos_report.get("severity", "UNKNOWN")
            c_health = self._last_chronos_report.get("strategy_health", {})
            chronos_correlation = {
                "chronos_severity": c_severity,
                "exploitation_risk": c_health.get("exploitation_risk", "UNKNOWN"),
                "loss_regime_mismatch": any(
                    lc.get("category") == "REGIME"
                    for lc in self._last_chronos_report.get("loss_causes", [])
                ),
            }

        # --- Overall Status ---
        if any(a["severity"] == "CRITICAL" for a in anomalies):
            overall_status = "CRITICAL"
        elif any(a["severity"] == "HIGH" for a in anomalies):
            overall_status = "WARNING"
        elif anomalies:
            overall_status = "ADVISORY"
        else:
            overall_status = "SECURE"

        report = {
            "agent": "aegis",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle_id": self._report_count,
            "overall_status": overall_status,
            "anomalies": anomalies,
            "anomaly_count": len(anomalies),
            "integrity": integrity,
            "rl_security": rl_report,
            "recommendations": recommendations,
            "chronos_correlation": chronos_correlation,
        }

        self._last_report = report
        self._report_count += 1

        # Track anomalies over time
        for a in anomalies:
            self._anomaly_log.append({**a, "timestamp": report["timestamp"]})
        # Trim anomaly log
        if len(self._anomaly_log) > 500:
            self._anomaly_log = self._anomaly_log[-500:]

        return report

    def publish_report(self, report: Optional[Dict] = None) -> int:
        """Publish security report to the message bus."""
        if report is None:
            report = self.run_security_scan()

        msg = Message(
            sender=self.name,
            type="security_report",
            payload=report,
        )
        notified = self.message_bus.publish("quant_ops.security", msg)
        logger.info(
            f"[{self.name}] Published security report (status={report.get('overall_status')}, "
            f"anomalies={report.get('anomaly_count', 0)}, notified={notified})"
        )
        return notified

    # ------------------------------------------------------------------
    # Holon ABC contract
    # ------------------------------------------------------------------

    def receive_message(self, sender: Any, content: Any) -> None:
        """
        Process incoming messages.

        Supported content types:
        - {"type": "run_scan"}             → run + publish security scan
        - {"type": "get_last_report"}      → return cached report
        - {"type": "inject_components", "components": {...}}  → late-bind AEGIS components
        """
        if isinstance(content, dict):
            msg_type = content.get("type", "")

            if msg_type == "run_scan":
                report = self.run_security_scan()
                self.publish_report(report)

            elif msg_type == "get_last_report":
                return self._last_report

            elif msg_type == "inject_components":
                self._components = content.get("components", self._components)
                logger.info(f"[{self.name}] AEGIS components injected/updated")

        logger.debug(f"[{self.name}] Received message from {sender}")

    # ------------------------------------------------------------------
    # Dashboard integration
    # ------------------------------------------------------------------

    def get_dashboard_state(self) -> dict:
        if self._last_report:
            return {
                "aegis_status": self._last_report.get("overall_status", "UNKNOWN"),
                "aegis_anomaly_count": self._last_report.get("anomaly_count", 0),
                "aegis_report_count": self._report_count,
            }
        return {"aegis_status": "NO_DATA", "aegis_report_count": 0}

    # ------------------------------------------------------------------
    # Internal event handlers
    # ------------------------------------------------------------------

    def _on_chronos_report(self, message: Message) -> None:
        """Cache latest Chronos report for correlation analysis."""
        if isinstance(message.payload, dict) and message.payload.get("agent") == "chronos":
            self._last_chronos_report = message.payload
            logger.debug(f"[{self.name}] Received Chronos report (severity={message.payload.get('severity')})")

    def _on_cycle_event(self, message: Message) -> None:
        """Handle cycle trigger from QuantOps orchestrator."""
        payload = message.payload if isinstance(message.payload, dict) else {}
        if payload.get("target") in (None, "aegis", "all"):
            logger.info(f"[{self.name}] Cycle event received — running security scan")
            report = self.run_security_scan()
            self.publish_report(report)
