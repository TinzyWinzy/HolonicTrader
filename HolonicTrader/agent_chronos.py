"""
CHRONOS HOLON — Quantitative Trading Loss Auditor (QUANT-OPS Agent)

Role: Forensic analyst for trading system failures.
Input: Trade logs, PnL data, session logs, signals
Output: Loss attribution, failure patterns, strategy health scores

Publishes to: quant_ops.forensics
Subscribes to: quant_ops.cycle (triggered by QuantOps orchestrator)

Wraps the existing ChronosForensicsEngine to provide a Holon-compatible
interface with structured JSON output for downstream agents (Aegis, Helix).
"""

import json
import time
import logging
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone

from HolonicTrader.holon_core import (
    Holon, Disposition, Message, MessageBus, get_message_bus
)

logger = logging.getLogger("Chronos.Holon")


class ChronosHolon(Holon):
    """
    Chronos Market Forensics — Holon-integrated quantitative loss auditor.

    Wraps ChronosForensicsEngine and publishes structured forensic reports
    to the quant_ops.forensics topic for consumption by Aegis and Helix.
    """

    def __init__(
        self,
        name: str = "ChronosAgent",
        db_path: str = "holonic_trader.db",
        log_dir: str = ".",
        message_bus: Optional[MessageBus] = None,
    ):
        super().__init__(
            name=name,
            disposition=Disposition(autonomy=0.8, integration=0.9),
            message_bus=message_bus or get_message_bus(),
        )
        self.db_path = db_path
        self.log_dir = log_dir
        self._engine = None  # Lazy init to avoid import-time DB access
        self._last_report: Optional[Dict] = None
        self._report_count = 0
        self._last_tier_calibration: float = 0.0   # epoch seconds
        self._tier_recalibration_interval: float = 4 * 3600  # 4 hours

        # Subscribe to cycle events from orchestrator
        self.message_bus.subscribe("quant_ops.cycle", self._on_cycle_event)

        logger.info(f"[{self.name}] Chronos Holon initialized (db={db_path})")

    # ------------------------------------------------------------------
    # Lazy engine initialization
    # ------------------------------------------------------------------

    def _get_engine(self):
        """Lazy-init the ChronosForensicsEngine to avoid import-time side effects."""
        if self._engine is None:
            try:
                from HolonicTrader.chronos_forensics import ChronosForensicsEngine
                self._engine = ChronosForensicsEngine(
                    db_path=self.db_path, log_dir=self.log_dir
                )
                logger.info(f"[{self.name}] Forensics engine initialized")
            except Exception as e:
                logger.error(f"[{self.name}] Failed to init forensics engine: {e}")
        return self._engine

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def run_forensics(self, log_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a complete forensic analysis cycle.

        Returns a structured JSON-compatible report dict:
        {
            "agent": "chronos",
            "timestamp": ...,
            "strategy_health": { ... },
            "loss_attribution": [ ... ],
            "veto_attribution": { ... },
            "loss_causes": [ ... ],
            "severity": "HIGH" | "MEDIUM" | "LOW",
            "executive_summary": "...",
            "recommendations": [ ... ],
            "next_actions": [ ... ]
        }
        """
        engine = self._get_engine()
        if engine is None:
            return self._empty_report("Engine initialization failed")

        try:
            full_report = engine.generate_forensic_report(log_path=log_path)
        except Exception as e:
            logger.error(f"[{self.name}] Forensic report generation failed: {e}")
            return self._empty_report(str(e))

        # Extract structured fields for downstream agents
        health = full_report.get("strategy_health", {})
        attributions = full_report.get("loss_attribution", [])
        veto = full_report.get("veto_attribution", {})

        # Compute severity from health score
        overall = health.get("overall_score", 5.0)
        if overall < 4.0:
            severity = "CRITICAL"
        elif overall < 6.0:
            severity = "HIGH"
        elif overall < 8.0:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # Flatten top loss causes for easy consumption
        loss_causes = []
        for attr in attributions[:5]:
            cause = attr.get("category", "UNKNOWN")
            pct = attr.get("percentage", 0)
            rec = attr.get("recommendation", "")
            loss_causes.append({
                "category": cause,
                "percentage": round(pct, 1),
                "recommendation": rec,
            })

        report = {
            "agent": "chronos",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle_id": self._report_count,
            "severity": severity,
            "strategy_health": health,
            "loss_attribution": attributions,
            "loss_causes": loss_causes,
            "veto_attribution": veto,
            "executive_summary": full_report.get("executive_summary", ""),
            "recommendations": full_report.get("recommendations", []),
            "next_actions": full_report.get("next_actions", []),
            "critical_findings": full_report.get("critical_findings", []),
        }

        self._last_report = report
        self._report_count += 1
        return report

    def publish_report(self, report: Optional[Dict] = None) -> int:
        """
        Publish a forensic report to the message bus.
        If no report provided, runs a fresh analysis first.
        Returns number of subscribers notified.
        """
        if report is None:
            report = self.run_forensics()

        msg = Message(
            sender=self.name,
            type="forensic_report",
            payload=report,
        )
        notified = self.message_bus.publish("quant_ops.forensics", msg)
        logger.info(
            f"[{self.name}] Published forensic report (severity={report.get('severity')}, "
            f"notified={notified} subscribers)"
        )
        return notified

    # ------------------------------------------------------------------
    # Holon ABC contract
    # ------------------------------------------------------------------

    def receive_message(self, sender: Any, content: Any) -> None:
        """
        Process incoming messages.

        Supported content types:
        - {"type": "run_forensics"}           → run + publish
        - {"type": "run_forensics", "log_path": "..."}  → targeted analysis
        - {"type": "get_last_report"}         → return cached report
        """
        if isinstance(content, dict):
            msg_type = content.get("type", "")

            if msg_type == "run_forensics":
                log_path = content.get("log_path")
                report = self.run_forensics(log_path=log_path)
                self.publish_report(report)

            elif msg_type == "get_last_report":
                return self._last_report

        logger.debug(f"[{self.name}] Received message from {sender}: {content}")

    # ------------------------------------------------------------------
    # Dashboard integration
    # ------------------------------------------------------------------

    def get_dashboard_state(self) -> dict:
        """Expose Chronos state to the GUI dashboard."""
        if self._last_report:
            health = self._last_report.get("strategy_health", {})
            return {
                "chronos_severity": self._last_report.get("severity", "UNKNOWN"),
                "chronos_health_score": health.get("overall_score", 0),
                "chronos_expectancy": health.get("expectancy", 0),
                "chronos_win_rate": health.get("win_rate", 0),
                "chronos_report_count": self._report_count,
            }
        return {"chronos_severity": "NO_DATA", "chronos_report_count": 0}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_cycle_event(self, message: Message) -> None:
        """Handle cycle trigger from QuantOps orchestrator."""
        payload = message.payload if isinstance(message.payload, dict) else {}
        if payload.get("target") in (None, "chronos", "all"):
            logger.info(f"[{self.name}] Cycle event received — running forensics")
            report = self.run_forensics()
            self.publish_report(report)

        # Periodic entropy tier recalibration (every 4h)
        self._maybe_recalibrate_entropy_tiers()

    def _maybe_recalibrate_entropy_tiers(self) -> None:
        """Trigger entropy tier recalibration if 4h interval has elapsed."""
        now = time.time()
        if now - self._last_tier_calibration >= self._tier_recalibration_interval:
            self.recalibrate_entropy_tiers()

    def recalibrate_entropy_tiers(self, scout_results: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Recompute ASSET_ENTROPY_TIERS from the latest scout SampleEntropy values.

        Uses empirical percentile splits: P50 = ORDERED boundary, P90 = CHAOTIC boundary.
        Updates config.ASSET_ENTROPY_TIERS and config.ASSET_ENTROPY_TIER_MAP in-place.

        Args:
            scout_results: Dict of {symbol: {'entropy': float, ...}} from EntropyScouter.
                           If None, tries to load from scout_status.json.
        Returns:
            Dict with calibration details (thresholds, tier assignments, timestamp).
        """
        import json, os
        import config

        # --- Load data ---
        if scout_results is None:
            scout_path = os.path.join(self.log_dir, 'scout_status.json')
            if not os.path.exists(scout_path):
                scout_path = 'scout_status.json'
            try:
                with open(scout_path) as f:
                    raw = json.load(f)
                scout_results = raw.get('results', raw)
            except Exception as e:
                logger.warning(f"[{self.name}] Tier recalibration: cannot load scout data: {e}")
                return {}

        if not scout_results:
            return {}

        # --- Compute empirical percentiles ---
        entries = [
            (sym, d['entropy'])
            for sym, d in scout_results.items()
            if isinstance(d, dict) and d.get('entropy', 0) > 0
        ]
        if len(entries) < 3:
            logger.warning(f"[{self.name}] Tier recalibration: insufficient data ({len(entries)} assets)")
            return {}

        entries.sort(key=lambda x: x[1])
        values = [v for _, v in entries]
        n = len(values)

        def percentile(data: list, p: float) -> float:
            idx = p * (len(data) - 1)
            lo, hi = int(idx), min(int(idx) + 1, len(data) - 1)
            return data[lo] + (data[hi] - data[lo]) * (idx - lo)

        p50 = percentile(values, 0.50)
        p90 = percentile(values, 0.90)

        # Clamp to sane range to avoid degenerate splits
        ordered_threshold  = max(0.50, min(p50, 0.90))
        chaotic_threshold  = max(ordered_threshold + 0.20, min(p90, 1.50))

        # --- Assign tiers ---
        new_ordered    = [s for s, v in entries if v <= ordered_threshold]
        new_transition = [s for s, v in entries if ordered_threshold < v <= chaotic_threshold]
        new_chaotic    = [s for s, v in entries if v > chaotic_threshold]

        # --- Mutate config in-place (safe: config is a flat mutable module) ---
        config.ASSET_ENTROPY_TIERS = {
            'ORDERED':    new_ordered,
            'TRANSITION': new_transition,
            'CHAOTIC':    new_chaotic,
        }
        config.ASSET_ENTROPY_TIER_MAP = {
            sym: tier
            for tier, syms in config.ASSET_ENTROPY_TIERS.items()
            for sym in syms
        }

        self._last_tier_calibration = time.time()

        result = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'n_assets': n,
            'p50': round(p50, 4),
            'p90': round(p90, 4),
            'ordered_threshold': round(ordered_threshold, 4),
            'chaotic_threshold': round(chaotic_threshold, 4),
            'tiers': config.ASSET_ENTROPY_TIERS,
        }
        logger.info(
            f"[{self.name}] Entropy tiers recalibrated: "
            f"ORDERED\u2264{ordered_threshold:.3f} ({len(new_ordered)}), "
            f"TRANSITION {ordered_threshold:.3f}\u2013{chaotic_threshold:.3f} ({len(new_transition)}), "
            f"CHAOTIC>{chaotic_threshold:.3f} ({len(new_chaotic)})"
        )

        # Publish summary to message bus
        msg = Message(sender=self.name, type="entropy_tiers_updated", payload=result)
        self.message_bus.publish("quant_ops.entropy_tiers", msg)
        return result

    def _empty_report(self, reason: str) -> Dict[str, Any]:
        """Return a minimal report when analysis cannot run."""
        return {
            "agent": "chronos",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle_id": self._report_count,
            "severity": "UNKNOWN",
            "strategy_health": {},
            "loss_attribution": [],
            "loss_causes": [],
            "veto_attribution": {},
            "executive_summary": f"Analysis unavailable: {reason}",
            "recommendations": [],
            "next_actions": [],
            "critical_findings": [reason],
        }
