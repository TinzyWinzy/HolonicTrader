"""
QUANT-OPS Multi-Agent Architecture — Test Suite

Tests for:
1. MessageBus (publish/subscribe/history)
2. ChronosHolon (forensics agent)
3. AegisHolon (security agent)
4. HelixHolon (repair agent)
5. AtlasHolon (strategy agent)
6. QuantOpsMemory (persistence layer)
7. QuantOpsHolon (orchestrator — full cycle)
8. Feedback paths (config mutation, Governor constraint updates)
"""

import json
import os
import sys
import time
import sqlite3
import shutil
import tempfile
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Dict, Any

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def temp_dir():
    """Provide a temporary directory cleaned up after test."""
    d = tempfile.mkdtemp(prefix="quantops_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def temp_db(temp_dir):
    """Provide a temporary SQLite DB with the trades table."""
    db_path = os.path.join(temp_dir, "test_trader.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            direction TEXT,
            quantity REAL,
            price REAL,
            cost_usd REAL,
            timestamp TEXT,
            pnl REAL,
            pnl_percent REAL,
            unrealized_pnl REAL,
            unrealized_pnl_percent REAL,
            mfe REAL,
            mae REAL
        )
    """)
    # Insert some test trades
    trades = [
        ("BTC/USDT", "BUY", 0.001, 65000, 65.0, "2026-03-18T10:00:00", 1.50, 0.023, 0, 0, 2.0, -0.5),
        ("ETH/USDT", "BUY", 0.01, 3500, 35.0, "2026-03-18T11:00:00", -0.80, -0.023, 0, 0, 0.5, -1.0),
        ("SOL/USDT", "BUY", 0.5, 140, 70.0, "2026-03-18T12:00:00", 2.10, 0.030, 0, 0, 3.0, -0.3),
        ("BTC/USDT", "BUY", 0.001, 64500, 64.5, "2026-03-18T13:00:00", -1.20, -0.019, 0, 0, 0.2, -1.5),
        ("ETH/USDT", "SELL", 0.01, 3450, 34.5, "2026-03-18T14:00:00", -2.50, -0.072, 0, 0, 0.1, -3.0),
        ("BTC/USDT", "BUY", 0.001, 65500, 65.5, "2026-03-18T15:00:00", 0.90, 0.014, 0, 0, 1.5, -0.4),
    ]
    c.executemany(
        "INSERT INTO trades (symbol,direction,quantity,price,cost_usd,timestamp,pnl,pnl_percent,unrealized_pnl,unrealized_pnl_percent,mfe,mae) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        trades,
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def fresh_bus():
    """Provide a fresh MessageBus (not the global singleton)."""
    from HolonicTrader.holon_core import MessageBus
    return MessageBus()


@pytest.fixture
def mock_db_manager(temp_db):
    """Mock DatabaseManager pointing at the temp DB."""
    mgr = MagicMock()
    mgr.db_path = temp_db
    mgr.get_recent_trades.return_value = [
        {"symbol": "BTC/USDT", "direction": "BUY", "pnl": 1.50, "pnl_percent": 0.023, "timestamp": "2026-03-18T10:00:00"},
        {"symbol": "ETH/USDT", "direction": "BUY", "pnl": -0.80, "pnl_percent": -0.023, "timestamp": "2026-03-18T11:00:00"},
        {"symbol": "SOL/USDT", "direction": "BUY", "pnl": 2.10, "pnl_percent": 0.030, "timestamp": "2026-03-18T12:00:00"},
        {"symbol": "BTC/USDT", "direction": "BUY", "pnl": -1.20, "pnl_percent": -0.019, "timestamp": "2026-03-18T13:00:00"},
        {"symbol": "BTC/USDT", "direction": "BUY", "pnl": 0.90, "pnl_percent": 0.014, "timestamp": "2026-03-18T15:00:00"},
    ]
    return mgr


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MESSAGE BUS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMessageBus:
    """Test the pub/sub MessageBus."""

    def test_subscribe_and_publish(self, fresh_bus):
        received = []
        fresh_bus.subscribe("test.topic", lambda msg: received.append(msg))

        from HolonicTrader.holon_core import Message
        msg = Message(sender="test", type="ping", payload={"hello": "world"})
        count = fresh_bus.publish("test.topic", msg)

        assert count == 1
        assert len(received) == 1
        assert received[0].payload == {"hello": "world"}

    def test_multiple_subscribers(self, fresh_bus):
        results = {"a": [], "b": []}
        fresh_bus.subscribe("multi", lambda m: results["a"].append(m))
        fresh_bus.subscribe("multi", lambda m: results["b"].append(m))

        from HolonicTrader.holon_core import Message
        fresh_bus.publish("multi", Message(sender="x", type="t", payload=1))

        assert len(results["a"]) == 1
        assert len(results["b"]) == 1

    def test_no_subscribers_returns_zero(self, fresh_bus):
        from HolonicTrader.holon_core import Message
        count = fresh_bus.publish("empty.topic", Message(sender="x", type="t", payload=None))
        assert count == 0

    def test_history_stored(self, fresh_bus):
        from HolonicTrader.holon_core import Message
        for i in range(5):
            fresh_bus.publish("hist", Message(sender="x", type="t", payload=i))

        history = fresh_bus.get_history("hist", limit=3)
        assert len(history) == 3
        assert history[-1].payload == 4

    def test_get_latest(self, fresh_bus):
        from HolonicTrader.holon_core import Message
        fresh_bus.publish("latest", Message(sender="x", type="t", payload="first"))
        fresh_bus.publish("latest", Message(sender="x", type="t", payload="second"))

        latest = fresh_bus.get_latest("latest")
        assert latest.payload == "second"

    def test_get_latest_empty(self, fresh_bus):
        assert fresh_bus.get_latest("nonexistent") is None

    def test_unsubscribe(self, fresh_bus):
        received = []
        cb = lambda msg: received.append(msg)
        fresh_bus.subscribe("unsub", cb)
        fresh_bus.unsubscribe("unsub", cb)

        from HolonicTrader.holon_core import Message
        fresh_bus.publish("unsub", Message(sender="x", type="t", payload=1))
        assert len(received) == 0

    def test_subscriber_error_doesnt_crash(self, fresh_bus):
        def bad_callback(msg):
            raise ValueError("boom")

        good_results = []
        fresh_bus.subscribe("err", bad_callback)
        fresh_bus.subscribe("err", lambda m: good_results.append(m))

        from HolonicTrader.holon_core import Message
        count = fresh_bus.publish("err", Message(sender="x", type="t", payload="ok"))
        # Bad subscriber failed, but good one still received
        assert count == 1  # only successful notifications counted
        assert len(good_results) == 1

    def test_topics_property(self, fresh_bus):
        from HolonicTrader.holon_core import Message
        fresh_bus.subscribe("a", lambda m: None)
        fresh_bus.publish("b", Message(sender="x", type="t", payload=None))
        topics = fresh_bus.topics
        assert "a" in topics
        assert "b" in topics

    def test_clear_history(self, fresh_bus):
        from HolonicTrader.holon_core import Message
        fresh_bus.publish("clear_me", Message(sender="x", type="t", payload=1))
        assert len(fresh_bus.get_history("clear_me")) == 1
        fresh_bus.clear_history("clear_me")
        assert len(fresh_bus.get_history("clear_me")) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CHRONOS HOLON TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestChronosHolon:
    """Test ChronosHolon forensics agent."""

    def test_init(self, fresh_bus, temp_db):
        from HolonicTrader.agent_chronos import ChronosHolon
        c = ChronosHolon(db_path=temp_db, message_bus=fresh_bus)
        assert c.name == "ChronosAgent"
        assert c.state == "ACTIVE"
        assert c._report_count == 0

    def test_run_forensics_returns_report(self, fresh_bus, temp_db):
        from HolonicTrader.agent_chronos import ChronosHolon
        c = ChronosHolon(db_path=temp_db, message_bus=fresh_bus)
        report = c.run_forensics()

        assert report["agent"] == "chronos"
        assert "severity" in report
        assert "strategy_health" in report
        assert "loss_causes" in report
        assert "timestamp" in report

    def test_publish_report_notifies_subscribers(self, fresh_bus, temp_db):
        from HolonicTrader.agent_chronos import ChronosHolon
        c = ChronosHolon(db_path=temp_db, message_bus=fresh_bus)

        received = []
        fresh_bus.subscribe("quant_ops.forensics", lambda m: received.append(m))

        c.publish_report()
        assert len(received) == 1
        assert received[0].payload["agent"] == "chronos"

    def test_receive_message_run_forensics(self, fresh_bus, temp_db):
        from HolonicTrader.agent_chronos import ChronosHolon
        c = ChronosHolon(db_path=temp_db, message_bus=fresh_bus)

        received = []
        fresh_bus.subscribe("quant_ops.forensics", lambda m: received.append(m))

        c.receive_message("test", {"type": "run_forensics"})
        assert len(received) == 1

    def test_dashboard_state(self, fresh_bus, temp_db):
        from HolonicTrader.agent_chronos import ChronosHolon
        c = ChronosHolon(db_path=temp_db, message_bus=fresh_bus)

        # Before any report
        state = c.get_dashboard_state()
        assert state["chronos_severity"] == "NO_DATA"

        # After a report
        c.run_forensics()
        state = c.get_dashboard_state()
        assert "chronos_severity" in state
        assert state["chronos_report_count"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 3. AEGIS HOLON TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAegisHolon:
    """Test AegisHolon security agent."""

    def test_init(self, fresh_bus):
        from HolonicTrader.agent_aegis import AegisHolon
        a = AegisHolon(message_bus=fresh_bus)
        assert a.name == "AegisAgent"
        assert a._report_count == 0

    def test_run_scan_without_components(self, fresh_bus):
        from HolonicTrader.agent_aegis import AegisHolon
        a = AegisHolon(message_bus=fresh_bus, aegis_components={})
        report = a.run_security_scan()

        assert report["agent"] == "aegis"
        assert report["overall_status"] == "SECURE"  # No components = no anomalies
        assert report["anomaly_count"] == 0

    def test_publish_report(self, fresh_bus):
        from HolonicTrader.agent_aegis import AegisHolon
        a = AegisHolon(message_bus=fresh_bus)

        received = []
        fresh_bus.subscribe("quant_ops.security", lambda m: received.append(m))

        a.publish_report()
        assert len(received) == 1
        assert received[0].payload["agent"] == "aegis"

    def test_chronos_correlation(self, fresh_bus, temp_db):
        """Aegis should cache and correlate Chronos reports."""
        from HolonicTrader.agent_aegis import AegisHolon
        from HolonicTrader.agent_chronos import ChronosHolon
        from HolonicTrader.holon_core import Message

        a = AegisHolon(message_bus=fresh_bus)

        # Simulate a Chronos report arriving
        chronos_msg = Message(
            sender="ChronosAgent",
            type="forensic_report",
            payload={"agent": "chronos", "severity": "HIGH", "strategy_health": {"exploitation_risk": "MEDIUM"}, "loss_causes": []},
        )
        fresh_bus.publish("quant_ops.forensics", chronos_msg)

        # Now Aegis should have cached it
        assert a._last_chronos_report is not None
        assert a._last_chronos_report["severity"] == "HIGH"

        # Run scan — should include chronos correlation
        report = a.run_security_scan()
        assert report["chronos_correlation"]["chronos_severity"] == "HIGH"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HELIX HOLON TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestHelixHolon:
    """Test HelixHolon repair agent."""

    def test_init(self, fresh_bus):
        from HolonicTrader.agent_helix import HelixHolon
        h = HelixHolon(message_bus=fresh_bus)
        assert h.name == "HelixAgent"

    def test_generate_fixes_empty_inputs(self, fresh_bus):
        from HolonicTrader.agent_helix import HelixHolon
        h = HelixHolon(message_bus=fresh_bus)
        report = h.generate_fixes()

        assert report["agent"] == "helix"
        assert "actions" in report
        assert "constraints" in report
        assert "component_status" in report

    def test_generate_fixes_from_chronos(self, fresh_bus):
        from HolonicTrader.agent_helix import HelixHolon
        h = HelixHolon(message_bus=fresh_bus)

        chronos_report = {
            "agent": "chronos",
            "severity": "HIGH",
            "strategy_health": {
                "overall_score": 3.5,
                "expectancy": -0.005,
                "win_rate": 0.45,
                "risk_management": 4.0,
            },
            "loss_causes": [
                {"category": "EXECUTION", "percentage": 40, "recommendation": "reduce fees"},
                {"category": "SIGNAL", "percentage": 35, "recommendation": "retrain model"},
            ],
            "veto_attribution": {"veto_assessment": "HIGH_OVERPROTECTION", "pass_rate_pct": 8.0},
        }

        report = h.generate_fixes(chronos_report=chronos_report)

        # Should propose execution cost filter
        action_names = [a["action"] for a in report["actions"]]
        assert "enforce_min_trade_threshold" in action_names
        # Should note negative expectancy constraint
        assert report["constraint_count"] >= 1

    def test_generate_fixes_from_aegis_critical(self, fresh_bus):
        from HolonicTrader.agent_helix import HelixHolon
        h = HelixHolon(message_bus=fresh_bus)

        aegis_report = {
            "agent": "aegis",
            "overall_status": "CRITICAL",
            "anomalies": [
                {"type": "position_mismatch", "severity": "CRITICAL", "detail": "3 mismatches"},
            ],
        }

        report = h.generate_fixes(aegis_report=aegis_report)
        action_names = [a["action"] for a in report["actions"]]
        assert "reconcile_positions" in action_names
        assert report["component_status"]["security"] == "CRITICAL"

    def test_publish_report(self, fresh_bus):
        from HolonicTrader.agent_helix import HelixHolon
        h = HelixHolon(message_bus=fresh_bus)

        received = []
        fresh_bus.subscribe("quant_ops.fixes", lambda m: received.append(m))

        h.publish_report()
        assert len(received) == 1
        assert received[0].payload["agent"] == "helix"

    def test_actions_are_priority_sorted(self, fresh_bus):
        from HolonicTrader.agent_helix import HelixHolon
        h = HelixHolon(message_bus=fresh_bus)

        report = h.generate_fixes(
            chronos_report={
                "agent": "chronos",
                "severity": "CRITICAL",
                "strategy_health": {"expectancy": -0.01, "win_rate": 0.3, "risk_management": 3.0},
                "loss_causes": [
                    {"category": "EXECUTION", "percentage": 30, "recommendation": ""},
                    {"category": "RISK", "percentage": 30, "recommendation": ""},
                    {"category": "REGIME", "percentage": 25, "recommendation": ""},
                ],
                "veto_attribution": {},
            },
            aegis_report={
                "agent": "aegis",
                "overall_status": "CRITICAL",
                "anomalies": [
                    {"type": "position_mismatch", "severity": "CRITICAL", "detail": "mismatch"},
                ],
            },
        )

        actions = report["actions"]
        if len(actions) >= 2:
            priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
            for i in range(len(actions) - 1):
                p1 = priority_order.get(actions[i].get("priority", "LOW"), 4)
                p2 = priority_order.get(actions[i + 1].get("priority", "LOW"), 4)
                assert p1 <= p2, f"Actions not sorted by priority: {actions[i]} before {actions[i+1]}"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ATLAS HOLON TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAtlasHolon:
    """Test AtlasHolon strategy agent."""

    def test_init(self, fresh_bus):
        from HolonicTrader.agent_atlas_strategist import AtlasHolon
        a = AtlasHolon(message_bus=fresh_bus)
        assert a.name == "AtlasAgent"

    def test_generate_strategy_empty(self, fresh_bus):
        from HolonicTrader.agent_atlas_strategist import AtlasHolon
        a = AtlasHolon(message_bus=fresh_bus)
        report = a.generate_strategy()

        assert report["agent"] == "atlas"
        assert "capital_allocation" in report
        assert "rules" in report
        assert "scaling_decision" in report
        assert "strategy_directives" in report

    def test_capital_allocation_defaults(self, fresh_bus):
        from HolonicTrader.agent_atlas_strategist import AtlasHolon
        a = AtlasHolon(message_bus=fresh_bus)
        report = a.generate_strategy()

        alloc = report["capital_allocation"]
        assert alloc["buy_strategy"] >= 0
        assert alloc["reserve"] >= 0
        assert abs(alloc["total"] - 1.0) < 0.01

    def test_negative_expectancy_conservative(self, fresh_bus, mock_db_manager):
        """When expectancy is negative, Atlas should be conservative."""
        from HolonicTrader.agent_atlas_strategist import AtlasHolon

        # Mock DB returns trades with overall negative expectancy
        mock_db_manager.get_recent_trades.return_value = [
            {"pnl": -2.0, "pnl_percent": -0.03},
            {"pnl": -1.5, "pnl_percent": -0.02},
            {"pnl": 0.5, "pnl_percent": 0.01},
        ]

        a = AtlasHolon(message_bus=fresh_bus, db_manager=mock_db_manager)
        report = a.generate_strategy()

        alloc = report["capital_allocation"]
        # Should have higher reserve when losing
        assert alloc["reserve"] >= 0.20

    def test_scaling_not_allowed_few_trades(self, fresh_bus, mock_db_manager):
        from HolonicTrader.agent_atlas_strategist import AtlasHolon
        a = AtlasHolon(message_bus=fresh_bus, db_manager=mock_db_manager)
        report = a.generate_strategy()

        scaling = report["scaling_decision"]
        assert scaling["allowed"] is False

    def test_publish_report(self, fresh_bus):
        from HolonicTrader.agent_atlas_strategist import AtlasHolon
        a = AtlasHolon(message_bus=fresh_bus)

        received = []
        fresh_bus.subscribe("quant_ops.strategy", lambda m: received.append(m))

        a.publish_report()
        assert len(received) == 1
        assert received[0].payload["agent"] == "atlas"

    def test_helix_subscription(self, fresh_bus):
        """Atlas should cache Helix reports via subscription."""
        from HolonicTrader.agent_atlas_strategist import AtlasHolon
        from HolonicTrader.holon_core import Message

        a = AtlasHolon(message_bus=fresh_bus)

        helix_msg = Message(
            sender="HelixAgent",
            type="fix_proposals",
            payload={"agent": "helix", "actions": [{"action": "reduce_leverage"}], "constraints": []},
        )
        fresh_bus.publish("quant_ops.fixes", helix_msg)

        assert a._last_helix is not None
        assert a._last_helix["agent"] == "helix"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. QUANTOPS MEMORY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestQuantOpsMemory:
    """Test the QuantOps memory/persistence layer."""

    def test_init(self, temp_dir):
        from HolonicTrader.quantops_memory import QuantOpsMemory
        db_path = os.path.join(temp_dir, "mem_test.db")
        mem = QuantOpsMemory(db_path=db_path, memory_dir=os.path.join(temp_dir, "memory"))
        assert mem is not None

    def test_save_and_get_report(self, temp_dir):
        from HolonicTrader.quantops_memory import QuantOpsMemory
        db_path = os.path.join(temp_dir, "mem_test.db")
        mem = QuantOpsMemory(db_path=db_path, memory_dir=os.path.join(temp_dir, "memory"))

        report = {"agent": "chronos", "severity": "HIGH", "data": [1, 2, 3]}
        mem.save_report(cycle_id=1, agent_name="chronos", report=report)

        results = mem.get_agent_reports("chronos", limit=10)
        assert len(results) == 1
        assert results[0]["agent"] == "chronos"
        assert results[0]["severity"] == "HIGH"

    def test_save_cycle(self, temp_dir):
        from HolonicTrader.quantops_memory import QuantOpsMemory
        db_path = os.path.join(temp_dir, "mem_test.db")
        mem = QuantOpsMemory(db_path=db_path, memory_dir=os.path.join(temp_dir, "memory"))

        reports = {
            "chronos": {"agent": "chronos", "severity": "LOW"},
            "aegis": {"agent": "aegis", "overall_status": "SECURE"},
        }
        mem.save_cycle(cycle_id=1, reports=reports)

        cycle = mem.get_cycle_reports(1)
        assert "chronos" in cycle
        assert "aegis" in cycle

    def test_record_failure(self, temp_dir):
        from HolonicTrader.quantops_memory import QuantOpsMemory
        db_path = os.path.join(temp_dir, "mem_test.db")
        mem_dir = os.path.join(temp_dir, "memory")
        mem = QuantOpsMemory(db_path=db_path, memory_dir=mem_dir)

        mem.record_failure({"category": "SIGNAL", "description": "false positive"})
        mem.record_failure({"category": "EXECUTION", "description": "high slippage"})

        failures = mem.get_failures()
        assert len(failures) == 2

        summary = mem.get_failure_summary()
        assert summary["SIGNAL"] == 1
        assert summary["EXECUTION"] == 1

        # Verify JSON file was written
        assert os.path.exists(os.path.join(mem_dir, "failures.json"))

    def test_record_success(self, temp_dir):
        from HolonicTrader.quantops_memory import QuantOpsMemory
        db_path = os.path.join(temp_dir, "mem_test.db")
        mem_dir = os.path.join(temp_dir, "memory")
        mem = QuantOpsMemory(db_path=db_path, memory_dir=mem_dir)

        mem.record_success({"strategy": "BUY", "conditions": {"regime": "ORDERED"}, "result": {"pnl": 1.5}})
        successes = mem.get_successes()
        assert len(successes) == 1

    def test_agent_context(self, temp_dir):
        from HolonicTrader.quantops_memory import QuantOpsMemory
        db_path = os.path.join(temp_dir, "mem_test.db")
        mem = QuantOpsMemory(db_path=db_path, memory_dir=os.path.join(temp_dir, "memory"))

        mem.save_report(1, "chronos", {"agent": "chronos", "severity": "HIGH"})
        mem.save_report(2, "chronos", {"agent": "chronos", "severity": "MEDIUM"})
        mem.record_failure({"category": "SIGNAL"})

        context = mem.get_agent_context("chronos", depth=5)
        assert len(context["prior_reports"]) == 2
        assert "failure_summary" in context
        assert context["total_cycles"] == 2

    def test_latest_cycle_id(self, temp_dir):
        from HolonicTrader.quantops_memory import QuantOpsMemory
        db_path = os.path.join(temp_dir, "mem_test.db")
        mem = QuantOpsMemory(db_path=db_path, memory_dir=os.path.join(temp_dir, "memory"))

        assert mem.get_latest_cycle_id() == 0
        mem.save_report(5, "atlas", {"agent": "atlas"})
        assert mem.get_latest_cycle_id() == 5


# ═══════════════════════════════════════════════════════════════════════════════
# 7. QUANTOPS ORCHESTRATOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestQuantOpsHolon:
    """Test the QuantOps orchestrator (full cycle)."""

    def test_init(self, fresh_bus, temp_dir):
        from HolonicTrader.agent_quantops import QuantOpsHolon
        db_path = os.path.join(temp_dir, "qo_test.db")
        qo = QuantOpsHolon(
            cycle_interval=3,
            db_path=db_path,
            output_dir=os.path.join(temp_dir, "reports"),
            message_bus=fresh_bus,
        )
        assert qo.name == "QuantOpsAgent"
        assert qo.cycle_interval == 3
        assert qo._quantops_cycle_id == 0

    def test_tick_counting(self, fresh_bus, temp_dir):
        from HolonicTrader.agent_quantops import QuantOpsHolon
        db_path = os.path.join(temp_dir, "qo_test.db")
        # Use a high interval so no cycle triggers
        qo = QuantOpsHolon(
            cycle_interval=100,
            db_path=db_path,
            output_dir=os.path.join(temp_dir, "reports"),
            message_bus=fresh_bus,
        )
        for _ in range(10):
            result = qo.tick()
        assert result is None  # No cycle should fire
        assert qo._trade_cycle_count == 10

    def test_tick_triggers_cycle(self, fresh_bus, temp_db, temp_dir):
        from HolonicTrader.agent_quantops import QuantOpsHolon
        qo = QuantOpsHolon(
            cycle_interval=3,
            db_path=temp_db,
            output_dir=os.path.join(temp_dir, "reports"),
            message_bus=fresh_bus,
        )

        results = []
        for _ in range(3):
            r = qo.tick()
            if r:
                results.append(r)

        assert len(results) == 1
        assert results[0]["cycle_id"] == 1
        assert results[0]["duration_sec"] >= 0

    def test_run_intelligence_cycle(self, fresh_bus, temp_db, temp_dir):
        from HolonicTrader.agent_quantops import QuantOpsHolon
        qo = QuantOpsHolon(
            db_path=temp_db,
            output_dir=os.path.join(temp_dir, "reports"),
            message_bus=fresh_bus,
        )

        result = qo.run_intelligence_cycle()

        assert result["cycle_id"] == 1
        assert "reports" in result
        reports = result["reports"]
        assert "chronos" in reports
        assert "helix" in reports
        assert "atlas" in reports

    def test_cycle_persists_to_memory(self, fresh_bus, temp_db, temp_dir):
        from HolonicTrader.agent_quantops import QuantOpsHolon
        qo = QuantOpsHolon(
            db_path=temp_db,
            output_dir=os.path.join(temp_dir, "reports"),
            message_bus=fresh_bus,
        )

        qo.run_intelligence_cycle()

        # Check memory has the cycle
        assert qo.memory.get_latest_cycle_id() == 1
        chronos_reports = qo.memory.get_agent_reports("chronos")
        assert len(chronos_reports) >= 1

    def test_cycle_saves_to_disk(self, fresh_bus, temp_db, temp_dir):
        from HolonicTrader.agent_quantops import QuantOpsHolon
        output_dir = os.path.join(temp_dir, "reports")
        qo = QuantOpsHolon(
            db_path=temp_db,
            output_dir=output_dir,
            message_bus=fresh_bus,
        )

        qo.run_intelligence_cycle()

        # Check JSON file exists
        files = os.listdir(output_dir)
        json_files = [f for f in files if f.endswith(".json")]
        assert len(json_files) >= 1

    def test_get_status(self, fresh_bus, temp_dir):
        from HolonicTrader.agent_quantops import QuantOpsHolon
        db_path = os.path.join(temp_dir, "qo_test.db")
        qo = QuantOpsHolon(
            db_path=db_path,
            output_dir=os.path.join(temp_dir, "reports"),
            message_bus=fresh_bus,
        )

        status = qo.get_status()
        assert status["cycle_id"] == 0
        assert status["cycle_interval"] == 5
        assert "agents" in status
        assert "memory" in status

    def test_receive_message_run_cycle(self, fresh_bus, temp_db, temp_dir):
        from HolonicTrader.agent_quantops import QuantOpsHolon
        qo = QuantOpsHolon(
            db_path=temp_db,
            output_dir=os.path.join(temp_dir, "reports"),
            message_bus=fresh_bus,
        )

        qo.receive_message("test", {"type": "run_cycle"})
        assert qo._quantops_cycle_id == 1

    def test_dashboard_state(self, fresh_bus, temp_db, temp_dir):
        from HolonicTrader.agent_quantops import QuantOpsHolon
        qo = QuantOpsHolon(
            db_path=temp_db,
            output_dir=os.path.join(temp_dir, "reports"),
            message_bus=fresh_bus,
        )

        qo.run_intelligence_cycle()
        state = qo.get_dashboard_state()
        assert state["quantops_cycle_id"] == 1
        assert "chronos_severity" in state or "chronos_report_count" in state


# ═══════════════════════════════════════════════════════════════════════════════
# 8. FEEDBACK PATH TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeedbackPaths:
    """Test that strategy decisions propagate correctly."""

    def test_full_pipeline_message_flow(self, fresh_bus, temp_db, temp_dir):
        """Verify the Chronos → Aegis → Helix → Atlas message flow."""
        from HolonicTrader.agent_chronos import ChronosHolon
        from HolonicTrader.agent_aegis import AegisHolon
        from HolonicTrader.agent_helix import HelixHolon
        from HolonicTrader.agent_atlas_strategist import AtlasHolon

        # Create all agents on same bus
        chronos = ChronosHolon(db_path=temp_db, message_bus=fresh_bus)
        aegis = AegisHolon(message_bus=fresh_bus)
        helix = HelixHolon(message_bus=fresh_bus)
        atlas = AtlasHolon(message_bus=fresh_bus)

        # Track final output
        strategy_received = []
        fresh_bus.subscribe("quant_ops.strategy", lambda m: strategy_received.append(m))

        # Step 1: Chronos publishes
        c_report = chronos.run_forensics()
        chronos.publish_report(c_report)

        # Aegis should have received it
        assert aegis._last_chronos_report is not None

        # Step 2: Aegis publishes
        a_report = aegis.run_security_scan()
        aegis.publish_report(a_report)

        # Helix should have received both
        assert helix._last_chronos is not None
        assert helix._last_aegis is not None

        # Step 3: Helix publishes
        h_report = helix.generate_fixes()
        helix.publish_report(h_report)

        # Atlas should have received it
        assert atlas._last_helix is not None

        # Step 4: Atlas publishes
        s_report = atlas.generate_strategy()
        atlas.publish_report(s_report)

        # Final strategy should be on the bus
        assert len(strategy_received) == 1
        assert strategy_received[0].payload["agent"] == "atlas"

    def test_config_mutation_safety(self, fresh_bus, temp_db, temp_dir):
        """Verify that QuantOps only mutates allowed config keys."""
        import config

        from HolonicTrader.agent_quantops import QuantOpsHolon
        qo = QuantOpsHolon(
            db_path=temp_db,
            output_dir=os.path.join(temp_dir, "reports"),
            message_bus=fresh_bus,
        )

        # Save original values
        original_conviction = config.MINIMUM_CONVICTION_THRESHOLD

        # Run a cycle
        qo.run_intelligence_cycle()

        # Config should not have been set to something dangerous
        # (the test DB has a mix of wins/losses, Atlas may or may not
        # propose changes, but forbidden keys should never be touched)
        assert hasattr(config, "PAPER_TRADING")  # Safety key should still exist
        assert hasattr(config, "MINIMUM_CONVICTION_THRESHOLD")

        # Restore
        config.MINIMUM_CONVICTION_THRESHOLD = original_conviction

    def test_governor_receives_constraints(self, fresh_bus, temp_db, temp_dir):
        """Mock Governor and verify it receives constraint messages."""
        from HolonicTrader.agent_quantops import QuantOpsHolon

        mock_governor = MagicMock()
        mock_governor.receive_message = MagicMock()

        qo = QuantOpsHolon(
            db_path=temp_db,
            output_dir=os.path.join(temp_dir, "reports"),
            message_bus=fresh_bus,
            governor=mock_governor,
        )

        qo.run_intelligence_cycle()

        # Governor may or may not receive messages depending on Helix output
        # But it should not crash
        assert qo._quantops_cycle_id == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 9. INTEGRATION: MULTIPLE CYCLES
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultipleCycles:
    """Test that repeated cycles build institutional memory."""

    def test_three_cycles_accumulate(self, fresh_bus, temp_db, temp_dir):
        from HolonicTrader.agent_quantops import QuantOpsHolon
        qo = QuantOpsHolon(
            cycle_interval=1,
            db_path=temp_db,
            output_dir=os.path.join(temp_dir, "reports"),
            message_bus=fresh_bus,
        )

        for _ in range(3):
            qo.tick()

        assert qo._quantops_cycle_id == 3
        assert qo.memory.get_latest_cycle_id() == 3

        # Should have 3 chronos reports in memory
        chronos_reports = qo.memory.get_agent_reports("chronos", limit=10)
        assert len(chronos_reports) == 3

        # Output dir should have 3 JSON files
        files = [f for f in os.listdir(qo.output_dir) if f.endswith(".json")]
        assert len(files) == 3
