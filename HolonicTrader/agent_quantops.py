"""
QUANT-OPS ORCHESTRATOR HOLON — Multi-Agent Intelligence Loop

The brain loop that coordinates all four QUANT-OPS persona agents:

    1. Trade executes → Logs captured
    2. Chronos analyzes losses (forensics)
    3. Aegis checks integrity / exploits (security)
    4. Helix proposes fixes (repair)
    5. Atlas decides strategy direction (profit)
    6. System updates rules → Next trades adapt

This supra-holon owns the four persona holons as sub_holons and runs
a sequential intelligence cycle after every N trading cycles (configurable
via config.QUANTOPS_CYCLE_INTERVAL).

CRITICAL SAFETY: Atlas's proposals are clamped by SMCE tier limits and
Governor veto logic before application. Atlas proposes, Governor disposes.
"""

import json
import os
import time
import logging
import threading
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from HolonicTrader.holon_core import (
    Holon, Disposition, Message, MessageBus, get_message_bus
)
from HolonicTrader.quantops_memory import QuantOpsMemory

logger = logging.getLogger("QuantOps.Orchestrator")


class QuantOpsHolon(Holon):
    """
    QUANT-OPS Multi-Agent Orchestrator — Supra-Holon for the intelligence loop.

    Owns four persona agents:
    - ChronosHolon (forensics)
    - AegisHolon (security)
    - HelixHolon (repair)
    - AtlasHolon (strategy)

    Runs a sequential analysis cycle: Chronos → Aegis → Helix → Atlas
    Then applies Atlas's final decisions via config mutations and
    Governor constraint updates.
    """

    def __init__(
        self,
        name: str = "QuantOpsAgent",
        cycle_interval: int = 5,
        memory_depth: int = 10,
        db_path: str = "holonic_trader.db",
        log_dir: str = ".",
        output_dir: str = "quant_ops_reports",
        message_bus: Optional[MessageBus] = None,
        # External references for feedback loop
        governor: Optional[Any] = None,
        executor: Optional[Any] = None,
        atlas_integration: Optional[Any] = None,
        capital_manager: Optional[Any] = None,
        db_manager: Optional[Any] = None,
        aegis_components: Optional[Dict] = None,
    ):
        super().__init__(
            name=name,
            disposition=Disposition(autonomy=0.95, integration=0.95),
            message_bus=message_bus or get_message_bus(),
        )

        self.cycle_interval = cycle_interval  # Run every N trading cycles
        self.memory_depth = memory_depth
        self.output_dir = output_dir
        self._governor = governor
        self._executor = executor

        # Cycle tracking
        self._trade_cycle_count = 0
        self._quantops_cycle_id = 0
        self._last_cycle_time = 0.0
        self._is_running = False

        # Initialize memory layer
        self.memory = QuantOpsMemory(db_path=db_path, memory_dir="memory")

        # Initialize persona agents (lazy — set up after construction)
        self._chronos = None
        self._aegis = None
        self._helix = None
        self._atlas = None

        # Store init params for lazy creation
        self._init_params = {
            "db_path": db_path,
            "log_dir": log_dir,
            "atlas_integration": atlas_integration,
            "capital_manager": capital_manager,
            "db_manager": db_manager,
            "aegis_components": aegis_components,
            "executor": executor,
        }

        # Latest cycle results
        self._last_cycle_reports: Dict[str, Dict] = {}
        self._last_strategy: Optional[Dict] = None

        # Ensure output directory
        os.makedirs(output_dir, exist_ok=True)

        logger.info(
            f"[{self.name}] QuantOps Orchestrator initialized "
            f"(cycle_interval={cycle_interval}, memory_depth={memory_depth})"
        )

    # ------------------------------------------------------------------
    # Lazy persona initialization
    # ------------------------------------------------------------------

    def _ensure_agents(self) -> None:
        """Lazily initialize all four persona agents."""
        if self._chronos is not None:
            return  # Already initialized

        params = self._init_params
        bus = self.message_bus

        try:
            from HolonicTrader.agent_chronos import ChronosHolon
            self._chronos = ChronosHolon(
                name="ChronosAgent",
                db_path=params["db_path"],
                log_dir=params["log_dir"],
                message_bus=bus,
            )
        except Exception as e:
            logger.error(f"[{self.name}] Failed to init ChronosHolon: {e}")

        try:
            from HolonicTrader.agent_aegis import AegisHolon
            self._aegis = AegisHolon(
                name="AegisAgent",
                aegis_components=params.get("aegis_components"),
                executor=params.get("executor"),
                message_bus=bus,
            )
        except Exception as e:
            logger.error(f"[{self.name}] Failed to init AegisHolon: {e}")

        try:
            from HolonicTrader.agent_helix import HelixHolon
            self._helix = HelixHolon(
                name="HelixAgent",
                message_bus=bus,
            )
        except Exception as e:
            logger.error(f"[{self.name}] Failed to init HelixHolon: {e}")

        try:
            from HolonicTrader.agent_atlas_strategist import AtlasHolon
            self._atlas = AtlasHolon(
                name="AtlasStrategist",
                atlas_integration=params.get("atlas_integration"),
                capital_manager=params.get("capital_manager"),
                db_manager=params.get("db_manager"),
                message_bus=bus,
            )
        except Exception as e:
            logger.error(f"[{self.name}] Failed to init AtlasHolon: {e}")

        logger.info(
            f"[{self.name}] Persona agents initialized: "
            f"Chronos={'OK' if self._chronos else 'FAIL'}, "
            f"Aegis={'OK' if self._aegis else 'FAIL'}, "
            f"Helix={'OK' if self._helix else 'FAIL'}, "
            f"Atlas={'OK' if self._atlas else 'FAIL'}"
        )

    # ------------------------------------------------------------------
    # Core intelligence cycle
    # ------------------------------------------------------------------

    def tick(self) -> Optional[Dict]:
        """
        Called every trading cycle by TraderHolon.

        Increments the trade cycle counter. When the counter reaches
        cycle_interval, runs the full QUANT-OPS intelligence cycle.

        Returns the Atlas strategy report if a cycle ran, else None.
        """
        self._trade_cycle_count += 1

        if self._trade_cycle_count >= self.cycle_interval:
            self._trade_cycle_count = 0
            return self.run_intelligence_cycle()

        return None

    def run_intelligence_cycle(self) -> Dict[str, Any]:
        """
        Execute the full sequential intelligence cycle:
            Chronos → Aegis → Helix → Atlas

        Each agent runs synchronously; outputs feed into the next.
        Final Atlas strategy is applied to the system.
        """
        self._ensure_agents()
        self._quantops_cycle_id += 1
        cycle_id = self._quantops_cycle_id
        cycle_start = time.time()
        reports = {}

        logger.info(f"[{self.name}] ═══════ QUANT-OPS CYCLE #{cycle_id} START ═══════")

        # ── Step 1: Chronos (Forensics) ────────────────────────────────
        chronos_report = {}
        if self._chronos:
            try:
                logger.info(f"[{self.name}] [1/4] Running Chronos forensics...")
                chronos_report = self._chronos.run_forensics()
                self._chronos.publish_report(chronos_report)
                reports["chronos"] = chronos_report
                logger.info(
                    f"[{self.name}] [1/4] Chronos complete: "
                    f"severity={chronos_report.get('severity')}, "
                    f"expectancy={chronos_report.get('strategy_health', {}).get('expectancy', 'N/A')}"
                )
            except Exception as e:
                logger.error(f"[{self.name}] Chronos failed: {e}")
                reports["chronos"] = {"agent": "chronos", "error": str(e)}

        # ── Step 2: Aegis (Security) ───────────────────────────────────
        aegis_report = {}
        if self._aegis:
            try:
                logger.info(f"[{self.name}] [2/4] Running Aegis security scan...")
                aegis_report = self._aegis.run_security_scan()
                self._aegis.publish_report(aegis_report)
                reports["aegis"] = aegis_report
                logger.info(
                    f"[{self.name}] [2/4] Aegis complete: "
                    f"status={aegis_report.get('overall_status')}, "
                    f"anomalies={aegis_report.get('anomaly_count', 0)}"
                )
            except Exception as e:
                logger.error(f"[{self.name}] Aegis failed: {e}")
                reports["aegis"] = {"agent": "aegis", "error": str(e)}

        # ── Step 3: Helix (Repair) ─────────────────────────────────────
        helix_report = {}
        if self._helix:
            try:
                logger.info(f"[{self.name}] [3/4] Running Helix fix generation...")
                helix_report = self._helix.generate_fixes(
                    chronos_report=chronos_report,
                    aegis_report=aegis_report,
                )
                self._helix.publish_report(helix_report)
                reports["helix"] = helix_report
                logger.info(
                    f"[{self.name}] [3/4] Helix complete: "
                    f"actions={helix_report.get('action_count', 0)}, "
                    f"constraints={helix_report.get('constraint_count', 0)}"
                )
            except Exception as e:
                logger.error(f"[{self.name}] Helix failed: {e}")
                reports["helix"] = {"agent": "helix", "error": str(e)}

        # ── Step 4: Atlas (Strategy) ───────────────────────────────────
        atlas_report = {}
        if self._atlas:
            try:
                logger.info(f"[{self.name}] [4/4] Running Atlas strategy generation...")
                atlas_report = self._atlas.generate_strategy(
                    helix_report=helix_report,
                )
                self._atlas.publish_report(atlas_report)
                reports["atlas"] = atlas_report
                logger.info(
                    f"[{self.name}] [4/4] Atlas complete: "
                    f"directives={len(atlas_report.get('strategy_directives', []))}, "
                    f"scaling={'YES' if atlas_report.get('scaling_decision', {}).get('allowed') else 'NO'}"
                )
            except Exception as e:
                logger.error(f"[{self.name}] Atlas failed: {e}")
                reports["atlas"] = {"agent": "atlas", "error": str(e)}

        # ── Apply Atlas decisions (with safety clamping) ───────────────
        if atlas_report and not atlas_report.get("error"):
            self._apply_strategy(atlas_report, helix_report)

        # ── Persist and record ─────────────────────────────────────────
        cycle_duration = time.time() - cycle_start
        self._last_cycle_reports = reports
        self._last_cycle_time = cycle_duration

        # Save to memory
        self.memory.save_cycle(cycle_id, reports)

        # Record patterns
        self._record_patterns(reports)

        # Save to disk as timestamped JSON
        self._save_cycle_to_disk(cycle_id, reports)

        logger.info(
            f"[{self.name}] ═══════ QUANT-OPS CYCLE #{cycle_id} COMPLETE "
            f"({cycle_duration:.1f}s) ═══════"
        )

        return {
            "cycle_id": cycle_id,
            "duration_sec": cycle_duration,
            "reports": reports,
            "strategy_applied": bool(atlas_report and not atlas_report.get("error")),
        }

    # ------------------------------------------------------------------
    # Strategy application (feedback loop)
    # ------------------------------------------------------------------

    def _apply_strategy(self, atlas_report: Dict, helix_report: Dict) -> None:
        """
        Apply Atlas strategy decisions to the live system.

        SAFETY: All changes are clamped by SMCE tier limits.
        Atlas proposes, Governor/SMCE disposes.
        """
        import config

        directives = atlas_report.get("strategy_directives", [])
        applied = []

        for directive in directives:
            dtype = directive.get("directive", "")

            # ── Config updates ───────────────────────────────────────
            if dtype == "config_update":
                config_key = directive.get("config_key", "")
                value = directive.get("value")
                priority = directive.get("priority", "MEDIUM")

                # Safety: Only allow specific config keys to be mutated
                ALLOWED_CONFIG_KEYS = {
                    "MINIMUM_CONVICTION_THRESHOLD",
                    "MIN_EDGE_MULTIPLE",
                    "EXECUTION_COST_FILTER_ENABLED",
                    "WS_FORCE_REST_ONLY",
                }

                if config_key in ALLOWED_CONFIG_KEYS and value is not None:
                    old_value = getattr(config, config_key, None)
                    setattr(config, config_key, value)
                    applied.append(f"{config_key}: {old_value} → {value}")
                    logger.info(
                        f"[{self.name}] CONFIG UPDATE: {config_key} = {value} "
                        f"(was {old_value}, priority={priority})"
                    )

            # ── Capital rebalance ────────────────────────────────────
            elif dtype == "capital_rebalance":
                buy_pct = directive.get("buy_pct", 0.8)
                reserve_pct = directive.get("reserve_pct", 0.2)
                logger.info(
                    f"[{self.name}] CAPITAL REBALANCE: BUY={buy_pct:.0%}, "
                    f"RESERVE={reserve_pct:.0%}"
                )
                # Store for Governor to read
                if hasattr(config, 'ATLAS_BUY_ALLOCATION'):
                    config.ATLAS_BUY_ALLOCATION = buy_pct
                if hasattr(config, 'ATLAS_RESERVE_ALLOCATION'):
                    config.ATLAS_RESERVE_ALLOCATION = reserve_pct

        # ── Governor constraint update ───────────────────────────────
        if self._governor and helix_report.get("constraints"):
            try:
                self._governor.receive_message(
                    self.name,
                    {
                        "type": "quantops_constraint_update",
                        "constraints": helix_report["constraints"],
                        "source": "QuantOps/Helix",
                    },
                )
                logger.info(
                    f"[{self.name}] Sent {len(helix_report['constraints'])} "
                    f"constraints to Governor"
                )
            except Exception as e:
                logger.warning(f"[{self.name}] Governor constraint update failed: {e}")

        # ── RL reward adjustment (from Atlas risk overrides) ─────────
        risk_overrides = atlas_report.get("risk_overrides", [])
        if risk_overrides and self._governor:
            try:
                self._governor.receive_message(
                    self.name,
                    {
                        "type": "quantops_risk_override",
                        "overrides": risk_overrides,
                        "source": "QuantOps/Atlas",
                    },
                )
            except Exception as e:
                logger.warning(f"[{self.name}] Risk override delivery failed: {e}")

        if applied:
            logger.info(f"[{self.name}] Applied {len(applied)} config changes: {applied}")

    # ------------------------------------------------------------------
    # Pattern recording
    # ------------------------------------------------------------------

    def _record_patterns(self, reports: Dict[str, Dict]) -> None:
        """Extract and record failure/success patterns from cycle reports."""
        chronos = reports.get("chronos", {})
        health = chronos.get("strategy_health", {})
        expectancy = health.get("expectancy", 0)

        # Record failures
        for lc in chronos.get("loss_causes", []):
            if lc.get("percentage", 0) > 20:
                self.memory.record_failure({
                    "category": lc.get("category"),
                    "description": lc.get("recommendation", ""),
                    "percentage": lc.get("percentage"),
                    "cycle_id": self._quantops_cycle_id,
                })

        # Record successes (positive expectancy cycle)
        if expectancy > 0:
            self.memory.record_success({
                "strategy": "SYSTEM",
                "conditions": {
                    "expectancy": expectancy,
                    "win_rate": health.get("win_rate", 0),
                    "overall_score": health.get("overall_score", 0),
                },
                "result": {"expectancy": expectancy},
                "cycle_id": self._quantops_cycle_id,
            })

    # ------------------------------------------------------------------
    # Disk persistence
    # ------------------------------------------------------------------

    def _save_cycle_to_disk(self, cycle_id: int, reports: Dict[str, Dict]) -> None:
        """Save cycle reports as a timestamped JSON file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cycle_{cycle_id:04d}_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)

            output = {
                "cycle_id": cycle_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reports": reports,
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, default=str)

            logger.debug(f"[{self.name}] Saved cycle report to {filepath}")
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to save cycle to disk: {e}")

    # ------------------------------------------------------------------
    # Holon ABC contract
    # ------------------------------------------------------------------

    def receive_message(self, sender: Any, content: Any) -> None:
        """
        Process incoming messages.

        Supported:
        - {"type": "run_cycle"}          → force immediate intelligence cycle
        - {"type": "tick"}               → increment trade counter (called by TraderHolon)
        - {"type": "inject_governor", "governor": ...}
        - {"type": "inject_aegis", "components": ...}
        - {"type": "get_status"}         → return orchestrator status
        """
        if isinstance(content, dict):
            msg_type = content.get("type", "")

            if msg_type == "run_cycle":
                self.run_intelligence_cycle()

            elif msg_type == "tick":
                self.tick()

            elif msg_type == "inject_governor":
                self._governor = content.get("governor")
                logger.info(f"[{self.name}] Governor injected")

            elif msg_type == "inject_aegis":
                components = content.get("components")
                if self._aegis and components:
                    self._aegis._components = components
                    logger.info(f"[{self.name}] AEGIS components injected into AegisHolon")

            elif msg_type == "get_status":
                return self.get_status()

    def get_status(self) -> Dict:
        """Get orchestrator status summary."""
        return {
            "name": self.name,
            "cycle_id": self._quantops_cycle_id,
            "trade_cycles_since_last": self._trade_cycle_count,
            "cycle_interval": self.cycle_interval,
            "last_cycle_duration": self._last_cycle_time,
            "agents": {
                "chronos": "ACTIVE" if self._chronos else "NOT_INITIALIZED",
                "aegis": "ACTIVE" if self._aegis else "NOT_INITIALIZED",
                "helix": "ACTIVE" if self._helix else "NOT_INITIALIZED",
                "atlas": "ACTIVE" if self._atlas else "NOT_INITIALIZED",
            },
            "memory": {
                "total_cycles": self.memory.get_latest_cycle_id(),
                "failure_count": len(self.memory.get_failures()),
                "success_count": len(self.memory.get_successes()),
            },
        }

    # ------------------------------------------------------------------
    # Dashboard integration
    # ------------------------------------------------------------------

    def get_dashboard_state(self) -> dict:
        """Expose QuantOps state to the GUI dashboard."""
        status = self.get_status()
        state = {
            "quantops_cycle_id": status["cycle_id"],
            "quantops_interval": status["cycle_interval"],
            "quantops_trade_counter": status["trade_cycles_since_last"],
            "quantops_last_duration": round(status["last_cycle_duration"], 1),
        }

        # Add agent summaries
        if self._chronos:
            state.update(self._chronos.get_dashboard_state())
        if self._aegis:
            state.update(self._aegis.get_dashboard_state())
        if self._helix:
            state.update(self._helix.get_dashboard_state())
        if self._atlas:
            state.update(self._atlas.get_dashboard_state())

        return state

    # ------------------------------------------------------------------
    # Property access to persona agents
    # ------------------------------------------------------------------

    @property
    def chronos(self):
        self._ensure_agents()
        return self._chronos

    @property
    def aegis(self):
        self._ensure_agents()
        return self._aegis

    @property
    def helix(self):
        self._ensure_agents()
        return self._helix

    @property
    def atlas(self):
        self._ensure_agents()
        return self._atlas
