import unittest
from unittest.mock import MagicMock, patch
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from HolonicTrader.agent_actuator import ActuatorHolon
from HolonicTrader.agent_monitor import MonitorHolon
import config

class TestSystemStability(unittest.TestCase):

    def test_circuit_breaker_activates(self):
        print("\n--- Testing Circuit Breaker ---")
        actuator = ActuatorHolon(name="TestActuator", paper_mode=True)
        actuator.exchange = MagicMock()

        # Override threshold to 5 for deterministic testing
        actuator.MAX_CONSECUTIVE_ERRORS = 5

        # Simulate 5 Failures
        print("1. Simulating 5 consecutive API errors...")
        for i in range(5):
            actuator.report_failure("503 Service Unavailable")

        self.assertTrue(actuator.circuit_open, "Circuit should be OPEN after 5 failures")
        self.assertTrue(actuator.hibernate_until > time.time(), "Hibernate time should be set")

        # Verify blockade (paper mode returns config.INITIAL_CAPITAL, but circuit breaker check should return 0.0)
        print("2. Verifying blockade...")
        # check_circuit_breaker should return False when circuit is open
        self.assertFalse(actuator.check_circuit_breaker(), "Circuit breaker should block operations")

        print("✅ Circuit Breaker Test Passed")

    def test_monitor_healthy_equity(self):
        """Test that a small drawdown is healthy."""
        print("\n--- Testing Monitor Healthy Check ---")
        with patch.object(MonitorHolon, '_load_state', lambda self: None):
            monitor = MonitorHolon(name="TestMonitor", principal=1000.0)

            # Override to known state
            monitor._save_state = lambda: None
            monitor.daily_start_balance = 1000.0
            monitor.last_day_reset = time.time()
            monitor.is_system_healthy = True
            monitor._stale_check_done = True  # Skip stale detection

            # 1. Healthy Check ($990 = 1% drawdown)
            print("1. Testing Healthy Equity ($990)...")
            health, msg = monitor.perform_live_check(990.0)
            self.assertTrue(health, f"Should be healthy (1% drawdown). Got msg: {msg}")
            print("✅ Healthy Equity Test Passed")

    def test_monitor_fever_check(self):
        """Test that large drawdown triggers FEVER."""
        print("\n--- Testing Monitor Fever Check ---")
        # Patch _load_state so disk persistence doesn't override test values
        with patch.object(MonitorHolon, '_load_state', lambda self: None):
            monitor = MonitorHolon(name="TestMonitor", principal=1000.0)

            # Override persistence to prevent disk state from interfering
            monitor._save_state = lambda: None

            # Set known state AFTER construction
            monitor.daily_start_balance = 1000.0
            monitor.last_day_reset = time.time()
            monitor.is_system_healthy = True
            monitor._stale_check_done = True  # Skip stale detection

            # Fever Check ($750 = 25% drawdown, exceeding 20% IMMUNE_MAX_DAILY_DRAWDOWN)
            print("1. Testing Fever Equity ($750)...")
            health, msg = monitor.perform_live_check(750.0)
            self.assertFalse(health, f"Should be UNHEALTHY (25% drawdown > 20% limit). Got msg: {msg}")
            self.assertIn("FEVER", msg)
            self.assertFalse(monitor.is_system_healthy, "Internal state should be unhealthy")

            print(f"   Msg: {msg}")
            print("✅ Monitor Fever Check Test Passed")

if __name__ == '__main__':
    unittest.main()
