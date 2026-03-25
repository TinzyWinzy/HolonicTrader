"""
QUANT-OPS Memory Layer — Persistence for Multi-Agent Intelligence

Stores agent reports, cross-cycle learnings, failure patterns, and
successful strategies in SQLite + JSON files for institutional memory.

Used by QuantOpsHolon to provide context to each agent (e.g., Chronos
gets last N forensic reports as "memory").
"""

import json
import os
import sqlite3
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from collections import defaultdict

logger = logging.getLogger("QuantOps.Memory")

# Default paths
DEFAULT_DB_PATH = "holonic_trader.db"
DEFAULT_MEMORY_DIR = "memory"
FAILURES_FILE = "failures.json"
SUCCESSES_FILE = "successful_patterns.json"


class QuantOpsMemory:
    """
    Persistence layer for QUANT-OPS multi-agent reports and learnings.

    Storage:
    1. SQLite `quantops_reports` table — per-cycle agent outputs
    2. memory/failures.json — accumulated failure patterns
    3. memory/successful_patterns.json — accumulated winning patterns
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        memory_dir: str = DEFAULT_MEMORY_DIR,
    ):
        self.db_path = db_path
        self.memory_dir = memory_dir
        self._ensure_dirs()
        self._init_table()

        # In-memory caches
        self._failures: List[Dict] = self._load_json(FAILURES_FILE)
        self._successes: List[Dict] = self._load_json(SUCCESSES_FILE)

        logger.info(f"QuantOps Memory initialized (db={db_path}, memory={memory_dir})")

    # ------------------------------------------------------------------
    # SQLite table initialization
    # ------------------------------------------------------------------

    def _init_table(self):
        """Create the quantops_reports table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS quantops_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id INTEGER NOT NULL,
                agent_name TEXT NOT NULL,
                report_json TEXT NOT NULL,
                severity TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        # Index for fast lookups
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_qo_agent_time
            ON quantops_reports (agent_name, timestamp DESC)
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_qo_cycle
            ON quantops_reports (cycle_id)
        """)
        conn.commit()
        conn.close()

    def _ensure_dirs(self):
        """Ensure the memory directory exists."""
        os.makedirs(self.memory_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Report persistence
    # ------------------------------------------------------------------

    def save_report(self, cycle_id: int, agent_name: str, report: Dict) -> None:
        """Save an agent report to the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO quantops_reports (cycle_id, agent_name, report_json, severity, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    cycle_id,
                    agent_name,
                    json.dumps(report, default=str),
                    report.get("severity", report.get("overall_status", "UNKNOWN")),
                    report.get("timestamp", datetime.now(timezone.utc).isoformat()),
                ),
            )
            conn.commit()
            conn.close()
            logger.debug(f"Saved {agent_name} report for cycle {cycle_id}")
        except Exception as e:
            logger.error(f"Failed to save report ({agent_name}, cycle {cycle_id}): {e}")

    def save_cycle(self, cycle_id: int, reports: Dict[str, Dict]) -> None:
        """Save all agent reports for a complete cycle."""
        for agent_name, report in reports.items():
            self.save_report(cycle_id, agent_name, report)

    def get_agent_reports(
        self,
        agent_name: str,
        limit: int = 10,
    ) -> List[Dict]:
        """Retrieve recent reports for a specific agent."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute(
                """
                SELECT report_json, cycle_id, timestamp
                FROM quantops_reports
                WHERE agent_name = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (agent_name, limit),
            )
            rows = c.fetchall()
            conn.close()

            results = []
            for row in rows:
                try:
                    report = json.loads(row[0])
                    report["_db_cycle_id"] = row[1]
                    report["_db_timestamp"] = row[2]
                    results.append(report)
                except json.JSONDecodeError:
                    pass
            return results
        except Exception as e:
            logger.error(f"Failed to get reports for {agent_name}: {e}")
            return []

    def get_cycle_reports(self, cycle_id: int) -> Dict[str, Dict]:
        """Retrieve all agent reports for a specific cycle."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute(
                """
                SELECT agent_name, report_json
                FROM quantops_reports
                WHERE cycle_id = ?
                """,
                (cycle_id,),
            )
            rows = c.fetchall()
            conn.close()

            return {
                row[0]: json.loads(row[1])
                for row in rows
            }
        except Exception as e:
            logger.error(f"Failed to get cycle {cycle_id} reports: {e}")
            return {}

    def get_latest_cycle_id(self) -> int:
        """Get the most recent cycle ID."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT MAX(cycle_id) FROM quantops_reports")
            row = c.fetchone()
            conn.close()
            return row[0] if row and row[0] is not None else 0
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Pattern accumulation (failures & successes)
    # ------------------------------------------------------------------

    def record_failure(self, pattern: Dict) -> None:
        """
        Record a failure pattern for cross-cycle learning.

        pattern: {
            "category": "SIGNAL" | "EXECUTION" | "RISK" | ...,
            "description": "...",
            "cycle_id": ...,
            "timestamp": ...,
            "context": { ... }
        }
        """
        pattern.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self._failures.append(pattern)

        # Keep bounded
        if len(self._failures) > 500:
            self._failures = self._failures[-500:]

        self._save_json(FAILURES_FILE, self._failures)

    def record_success(self, pattern: Dict) -> None:
        """
        Record a successful pattern for reinforcement.

        pattern: {
            "strategy": "BUY" | "SELL",
            "conditions": { ... },
            "result": { ... },
            "cycle_id": ...,
            "timestamp": ...,
        }
        """
        pattern.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self._successes.append(pattern)

        if len(self._successes) > 500:
            self._successes = self._successes[-500:]

        self._save_json(SUCCESSES_FILE, self._successes)

    def get_failures(self, limit: int = 50) -> List[Dict]:
        """Get recent failure patterns."""
        return self._failures[-limit:]

    def get_successes(self, limit: int = 50) -> List[Dict]:
        """Get recent success patterns."""
        return self._successes[-limit:]

    def get_failure_summary(self) -> Dict[str, int]:
        """Summarize failures by category."""
        summary = defaultdict(int)
        for f in self._failures:
            summary[f.get("category", "UNKNOWN")] += 1
        return dict(summary)

    # ------------------------------------------------------------------
    # Context generation for agents
    # ------------------------------------------------------------------

    def get_agent_context(self, agent_name: str, depth: int = 5) -> Dict:
        """
        Build a context payload for an agent, including its prior reports
        and relevant cross-cycle patterns.

        This gives each agent "institutional memory" across cycles.
        """
        prior_reports = self.get_agent_reports(agent_name, limit=depth)
        failure_summary = self.get_failure_summary()

        context = {
            "prior_reports": prior_reports,
            "failure_summary": failure_summary,
            "recent_failures": self.get_failures(limit=10),
            "recent_successes": self.get_successes(limit=10),
            "total_cycles": self.get_latest_cycle_id(),
        }
        return context

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    def _load_json(self, filename: str) -> List[Dict]:
        """Load a JSON file from the memory directory."""
        path = os.path.join(self.memory_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load {path}: {e}")
        return []

    def _save_json(self, filename: str, data: List[Dict]) -> None:
        """Save data to a JSON file in the memory directory."""
        path = os.path.join(self.memory_dir, filename)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Failed to save {path}: {e}")
