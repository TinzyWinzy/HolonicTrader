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

    def update_health(self, current_balance: float, performance_data: dict):
        """Analyze system health and potentially pause operations."""
        drawdown = (self.principal - current_balance) / self.principal if current_balance < self.principal else 0.0
        self.metrics['current_drawdown'] = drawdown
        self.metrics['win_rate'] = performance_data.get('win_rate', 0.0)
        
        # 1. PRINCIPAL PROTECTION
        if current_balance < config.PRINCIPAL:
            if self.is_system_healthy:
                print(f"[{self.name}] ⚠️ CRITICAL HEALTH: Principal Breach! Current: ${current_balance:.2f} < Min: ${config.PRINCIPAL:.2f}")
                self.is_system_healthy = False
        else:
            self.is_system_healthy = True

        # 2. CONSECUTIVE LOSS PROTECTION (FUTURE)
        # If win_rate < 20% over last 10 trades, we are likely out of sync with market
        
    def get_health_report(self) -> dict:
        return {
            'healthy': self.is_system_healthy,
            'metrics': self.metrics,
            'state': 'STABLE' if self.is_system_healthy else 'HIBERNATE'
        }

    def get_health(self) -> dict:
        return self.get_health_report()

    def receive_message(self, sender: Any, content: Any) -> Any:
        if isinstance(content, dict) and content.get('type') == 'CHECK_HEALTH':
            return self.get_health_report()
        return None
