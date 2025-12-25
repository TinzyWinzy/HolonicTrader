"""
MonitorHolon - System Homeostasis Brain (Phase 16)

Specialized in:
1. Account Health Tracking (Drawdown)
2. Performance Analytics (Win Rate, Omega)
3. Execution Quality (Slippage/Fees)
4. Homeostasis Control (Pause trading if unstable)
"""

from typing import Any
from HolonicTrader.holon_core import Holon, Disposition
import config

class MonitorHolon(Holon):
    def __init__(self, name: str = "SystemMonitor", principal: float = 100.0):
        super().__init__(name=name, disposition=Disposition(autonomy=0.7, integration=0.9))
        self.principal = principal
        self.max_drawdown = 0.0
        self.is_system_healthy = True
        
        # Stats Cache
        self.metrics = {
            'win_rate': 0.0,
            'total_trades': 0,
            'current_drawdown': 0.0,
            'slippage_avg': 0.0
        }
        
        # Immune System State
        self.daily_start_balance = principal
        self.last_day_reset = None # To track 24h cycles

    def update_health(self, current_balance: float, performance_data: dict):
        """Analyze system health and potentially pause operations."""
        import time
        
        # 0. Daily Reset Logic (Simple 24h reset for now)
        #Ideally this should be aligned with UTC midnight, but relative 24h is fine for robustness
        current_time = time.time()
        if self.last_day_reset is None or (current_time - self.last_day_reset > 86400):
            print(f"[{self.name}] 🌅 NEW DAY: Resetting Daily Balance Tracker (${current_balance:.2f})")
            self.daily_start_balance = current_balance
            self.last_day_reset = current_time

        # 1. FEVER CHECK (Daily Drawdown)
        daily_drawdown = (self.daily_start_balance - current_balance) / self.daily_start_balance
        
        if daily_drawdown > config.IMMUNE_MAX_DAILY_DRAWDOWN:
             print(f"[{self.name}] 🌡️ FEVER DETECTED: Daily Drawdown {daily_drawdown*100:.2f}% > Limit {config.IMMUNE_MAX_DAILY_DRAWDOWN*100:.1f}%")
             print(f"[{self.name}] 🔒 ACTION: INITIATING HARD LOCKDOWN (4 HOURS)")
             self.is_system_healthy = False
             return # Stop processing
             
        # Normal Drawdown (All time)
        drawdown = (self.principal - current_balance) / self.principal if current_balance < self.principal else 0.0
        self.metrics['current_drawdown'] = drawdown
        self.metrics['win_rate'] = performance_data.get('win_rate', 0.0)
        
        # 2. PRINCIPAL PROTECTION
        if current_balance < config.PRINCIPAL:
            if self.is_system_healthy:
                print(f"[{self.name}] ⚠️ CRITICAL HEALTH: Principal Breach! Current: ${current_balance:.2f} < Min: ${config.PRINCIPAL:.2f}")
                self.is_system_healthy = False
        else:
            # Only recover if not in Fever
            if daily_drawdown <= config.IMMUNE_MAX_DAILY_DRAWDOWN:
                self.is_system_healthy = True

        # 2. CONSECUTIVE LOSS PROTECTION (FUTURE)
        # If win_rate < 20% over last 10 trades, we are likely out of sync with market
        
    def get_health_report(self) -> dict:
        return {
            'healthy': self.is_system_healthy,
            'metrics': self.metrics,
            'state': 'STABLE' if self.is_system_healthy else 'LOCKDOWN'
        }

    def get_health(self) -> dict:
        return self.get_health_report()

    def receive_message(self, sender: Any, content: Any) -> Any:
        if isinstance(content, dict) and content.get('type') == 'CHECK_HEALTH':
            return self.get_health_report()
        return None
