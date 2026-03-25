"""
StateAggregator - Clean dashboard state collection.

Replaces the monolithic construct_system_state() by delegating to
each agent's get_dashboard_state() method.
"""

import time
import json
import os
import logging
from datetime import datetime
import config

logger = logging.getLogger("StateAggregator")


class StateAggregator:
    """
    Collects dashboard state from all agents via their get_dashboard_state() method.
    Server-owned state (equity history, logs, scanning) is managed here directly.
    Status file reads remain as fallback for data not yet on a live holon.
    """

    MAX_EQUITY_POINTS = 100
    MAX_LOG_ENTRIES = 50

    STATUS_FILES = {
        'dashboard': 'dashboard_status.json',
        'evolution': 'evolution_status.json',
        'order_flow': 'order_flow_status.json',
        'scout': 'scout_status.json',
        'monte_carlo': 'monte_carlo_results.json',
    }

    def __init__(self, holon_stack=None):
        self.stack = holon_stack
        self.equity_history = []
        self.system_logs = []
        self.scanning_active = False
        self.last_scan_time = None
        self.latest_report = []
        # Search in both current dir and parent dir for status files
        self._search_paths = [
            os.path.dirname(__file__),
            os.path.dirname(os.path.dirname(__file__)),
        ]

    # ------------------------------------------------------------------
    # Core Collection
    # ------------------------------------------------------------------

    def collect(self) -> dict:
        """Build the complete hub_state payload."""
        state = {
            'status': 'ACTIVE',
            'timestamp': time.time(),
        }

        # --- Agent-owned slices ---
        if self.stack:
            # Governor → positions, prices, portfolio_health
            if self.stack.governor:
                state.update(self.stack.governor.get_dashboard_state())

            # Arbitrage → arbitrage array
            arb = self.stack.holons.get('arbitrage')
            if arb:
                state.update(arb.get_dashboard_state())

            # Doomsday → doomsday object
            doom = self.stack.holons.get('doomsday')
            if doom:
                try:
                    doom.run_doomsday_scan(state.get('prices', {}))
                except Exception:
                    pass
                state.update(doom.get_dashboard_state())

            # SignalProvider → order_flow, monte_carlo
            if self.stack.signal_provider:
                state.update(self.stack.signal_provider.get_dashboard_state())

            # Overwatch → overwatch_state, sitrep
            overwatch = self.stack.holons.get('overwatch')
            if overwatch:
                state.update(overwatch.get_dashboard_state())

        # --- Defaults for missing agent data ---
        state.setdefault('positions', [])
        state.setdefault('prices', {})
        state.setdefault('portfolio_health', {})
        state.setdefault('arbitrage', [])
        state.setdefault('doomsday', {'defcon_level': 5, 'crisis_active': False})
        state.setdefault('order_flow', {})
        state.setdefault('monte_carlo', {})

        # --- Status-file-backed data (fallback / non-holon sources) ---
        status_data = self._load_status_data()
        dashboard = status_data.get('dashboard') or {}

        state['system_status'] = dashboard.get('solvency_status', 'DISCONNECTED')
        state['health_score'] = dashboard.get('health_score', 0) * 100
        state['equity'] = dashboard.get('equity') or dashboard.get('balance') or 0
        state['pnl'] = self._parse_pnl(dashboard.get('pnl', '0'))
        state['regime'] = dashboard.get('regime', 'UNKNOWN')

        # Evolution (file-backed, no live holon yet)
        if not state.get('evolution'):
            state['evolution'] = status_data.get('evolution') or {}

        # Order flow / Monte Carlo fallback from files if agent data empty
        if not state.get('order_flow'):
            state['order_flow'] = status_data.get('order_flow') or {}
        if not state.get('monte_carlo'):
            state['monte_carlo'] = status_data.get('monte_carlo') or {}

        # --- Server-owned state ---
        self._update_equity_history(state['equity'])
        state['equity_history'] = self.equity_history
        state['radar'] = self.latest_report or (dashboard.get('scout_data') or [])
        state['logs'] = self.system_logs
        state['scanning'] = self.scanning_active
        state['last_scan'] = self.last_scan_time

        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_equity_history(self, equity_val):
        """Append equity value to the time-series buffer (one per minute)."""
        if equity_val and isinstance(equity_val, (int, float)):
            now_str = datetime.now().strftime("%H:%M")
            if not self.equity_history or self.equity_history[-1]['t'] != now_str:
                self.equity_history.append({'t': now_str, 'y': float(equity_val)})
                if len(self.equity_history) > self.MAX_EQUITY_POINTS:
                    self.equity_history.pop(0)

    def _load_status_data(self) -> dict:
        """Read status JSON files from disk (searches multiple paths)."""
        data = {}
        for key, filename in self.STATUS_FILES.items():
            # Try each search path
            for search_path in self._search_paths:
                try:
                    path = os.path.join(search_path, filename)
                    if os.path.exists(path):
                        # Check file is recent (within last 30 minutes)
                        mtime = os.path.getmtime(path)
                        if time.time() - mtime < 1800:  # 30 min
                            with open(path, 'r', encoding='utf-8') as f:
                                data[key] = json.load(f)
                            break  # Found recent file, stop searching
                except Exception:
                    continue
            # Fallback to empty dict if not found
            if key not in data:
                data[key] = {}
        return data

    @staticmethod
    def _parse_pnl(raw) -> float:
        """Robustly parse PnL from various formats."""
        if isinstance(raw, (int, float)):
            return float(raw)
        try:
            return float(str(raw).replace('$', '').replace(',', ''))
        except (ValueError, TypeError):
            return 0.0

    def add_log(self, record_msg: str, level: str = 'INFO'):
        """Add a log entry to the buffer."""
        self.system_logs.append({
            'time': datetime.now().strftime("%H:%M:%S"),
            'msg': record_msg,
            'level': level,
        })
        if len(self.system_logs) > self.MAX_LOG_ENTRIES:
            self.system_logs.pop(0)
    def get_config_state(self) -> dict:
        """Expose key configuration parameters."""
        return {
            'max_allocation': getattr(config, 'SIZE_MAX_ALLOCATION', 0.20),
            'leverage_cap': getattr(config, 'MAX_TOTAL_LEVERAGE', 10.0),
            'scavenger_mode': getattr(config, 'SCAVENGER_MODE', False), # Inferred
            'trading_mode': getattr(config, 'TRADING_MODE', 'FUTURES'),
            'reinvest_profits': getattr(config, 'REINVEST_PROFITS', False),
        }

    def update_config(self, updates: dict) -> dict:
        """
        Runtime configuration updates (In-Memory Override).
        Note: This does NOT persist to config.py on disk yet, 
        but allows runtime tuning for the active session.
        """
        updated = {}
        for key, value in updates.items():
            # Map frontend keys to config variable names
            if key == 'max_allocation':
                config.SIZE_MAX_ALLOCATION = float(value)
                updated['max_allocation'] = config.SIZE_MAX_ALLOCATION
            elif key == 'leverage_cap':
                config.MAX_TOTAL_LEVERAGE = float(value)
                updated['leverage_cap'] = config.MAX_TOTAL_LEVERAGE
        
        return updated
