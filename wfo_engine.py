import os
import time
import json
import logging
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List

import config
from HolonicTrader.agent_observer import ObserverHolon
from utf8_logging import get_logger

class WalkForwardOptimizer:
    """
    Volatility Regime Sensor (demoted from parameter optimizer 2026-03-20).
    Runs periodically in a background thread to detect the current market regime
    (HIGH/NORMAL/LOW volatility). Exposes regime info for other systems.
    Parameter optimization is handled by the Evolution Engine.
    """
    def __init__(self, observer: ObserverHolon, state_file: str = "wfo_state.json"):
        self.logger = get_logger("WFOEngine")
        self.observer = observer
        self.state_file = os.path.join(os.getcwd(), state_file)
        self.is_running = False
        self._thread = None
        self._state_lock = threading.Lock()
        
        # Default starting state
        self.active_state = self._load_state()

    def start(self, cycle_hours: float = 4.0):
        if self.is_running: return
        self.is_running = True
        self.cycle_seconds = cycle_hours * 3600
        
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="WFO_Thread")
        self._thread.start()
        self.logger.info(f"🧬 Walk-Forward Optimization Engine started. Cycle: {cycle_hours} hours.")

    def stop(self):
        self.is_running = False
        self.logger.info("🛑 WFO Engine stopping...")

    def _run_loop(self):
        # We start with an immediate initial run, then sleep
        while self.is_running:
            try:
                self.logger.info("🧬 WFO: Starting optimization cycle...")
                new_params = self._optimize_parameters()
                if new_params:
                    self._save_state(new_params)
                    self.active_state = new_params
                    self.logger.info("🧬 WFO: Optimization complete. New parameters broadcast.")
            except Exception as e:
                self.logger.error(f"❌ WFO Engine Error: {e}")
            
            # Sleep in chucks to allow clean shutdown
            for _ in range(int(self.cycle_seconds)):
                if not self.is_running: break
                time.sleep(1.0)

    def _optimize_parameters(self) -> Dict[str, Any]:
        """
        Detect current volatility regime from BTC 15m data.
        No longer sets parameters — just reports regime and volatility.
        """
        try:
             symbol = 'BTC/USDT'
             df = self.observer.fetch_market_data(timeframe='15m', limit=200, symbol=symbol)
             if df.empty or 'returns' not in df:
                  return None
                  
             # Calculate 50-period annualized volatility
             returns = df['returns'].iloc[-50:]
             recent_vol = np.std(returns) * np.sqrt(35040)  # Annualized 15m
             
             # Classify regime
             if recent_vol > 0.80:
                  regime = 'HIGH'
             elif recent_vol < 0.30:
                  regime = 'LOW'
             else:
                  regime = 'NORMAL'
                  
             state = {
                  "last_updated": datetime.now(timezone.utc).isoformat(),
                  "recent_volatility": float(recent_vol),
                  "regime": regime,
                  "parameters": {}  # Empty — evolution handles parameters now
             }
             self.logger.info(f"🧬 WFO Regime: {regime} (vol={recent_vol:.3f})")
             return state
             
        except Exception as e:
             self.logger.error(f"WFO Regime Detection Failed: {e}")
             return None

    def _save_state(self, state: Dict[str, Any]):
        with self._state_lock:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=4)

    def _load_state(self) -> Dict[str, Any]:
        with self._state_lock:
            if os.path.exists(self.state_file):
                try:
                    with open(self.state_file, 'r') as f:
                        return json.load(f)
                except Exception:
                    pass
        return {}

    def get_current_parameters(self) -> Dict[str, float]:
        """Returns the currently active dynamic parameters (empty since 2026-03-20 — evolution handles params)."""
        return self.active_state.get('parameters', {})

    def get_regime(self) -> str:
        """Returns the current volatility regime: 'HIGH', 'NORMAL', or 'LOW'."""
        return self.active_state.get('regime', 'NORMAL')

    def get_volatility(self) -> float:
        """Returns the latest annualized BTC volatility reading."""
        return self.active_state.get('recent_volatility', 0.5)
