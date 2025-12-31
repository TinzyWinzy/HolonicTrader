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

    def update_health(self, executor_summary: dict, performance_data: dict) -> bool:
        """
        Analyze system health. Returns False if CRITICAL FAILURE (Liquidation).
        """
        import time
        
        current_balance = executor_summary.get('balance', 0.0)
        current_equity = executor_summary.get('equity', 0.0)
        margin_used = executor_summary.get('margin_used', 0.0)
        
        # 0. Daily Reset Logic
        current_time = time.time()
        if self.last_day_reset is None or (current_time - self.last_day_reset > 86400):
            print(f"[{self.name}] 🌅 NEW DAY: Resetting Daily Balance Tracker (${current_equity:.2f})")
            self.daily_start_balance = current_equity
            self.last_day_reset = current_time

        # --- PHASE 36: LIQUIDATION ENGINE (Solvency Check) ---
        maintenance_margin = margin_used * config.MAINTENANCE_MARGIN_RATE
        if current_equity < maintenance_margin and margin_used > 0:
            print(f"[{self.name}] ☠️ INSOLVENCY DETECTED: Equity ${current_equity:.2f} < Maint.Margin ${maintenance_margin:.2f}")
            print(f"[{self.name}] 🩸 LIQUIDATING ALL POSITIONS TO PROTECT EXCHANGE.")
            return False # FATAL HEALTH FAILURE

        # 1. FEVER CHECK (Daily Drawdown)
        # BUGFIX: Use Equity (Unrealized) not Balance (Realized) for Drawdown to allow open positions
        daily_drawdown = (self.daily_start_balance - current_equity) / self.daily_start_balance
        if daily_drawdown > config.IMMUNE_MAX_DAILY_DRAWDOWN:
             print(f"[{self.name}] 🌡️ FEVER DETECTED: Daily Drawdown {daily_drawdown*100:.2f}% > Limit {config.IMMUNE_MAX_DAILY_DRAWDOWN*100:.1f}%")
             self.is_system_healthy = False
             # Just lock, don't liquidate yet? Or halt?
             # For now, we return True (Alive) but set healthy=False (Fever)
             return True 

             
        # Normal Drawdown (All time)
        drawdown = (self.principal - current_balance) / self.principal if current_balance < self.principal else 0.0
        self.metrics['current_drawdown'] = drawdown
        self.metrics['win_rate'] = performance_data.get('win_rate', 0.0)
        
        # 2. PRINCIPAL PROTECTION
        # BUGFIX: Use Equity (Total Value) to check Principal, otherwise opening positions (spending cash) triggers panic.
        if current_equity < config.PRINCIPAL:
            if self.is_system_healthy:
                print(f"[{self.name}] ⚠️ CRITICAL HEALTH: Principal Breach! Equity: ${current_equity:.2f} < Min: ${config.PRINCIPAL:.2f}")
                self.is_system_healthy = False
        else:
            # Only recover if not in Fever
            if daily_drawdown <= config.IMMUNE_MAX_DAILY_DRAWDOWN:
                self.is_system_healthy = True

        # 2. CONSECUTIVE LOSS PROTECTION (FUTURE)
        # If win_rate < 20% over last 10 trades, we are likely out of sync with market
        
        return self.is_system_healthy
        
    def get_health_report(self) -> dict:
        return {
            'healthy': self.is_system_healthy,
            'metrics': self.metrics,
            'state': 'STABLE' if self.is_system_healthy else 'LOCKDOWN'
        }

    def get_health(self) -> dict:
        return self.get_health_report()

    def check_vital_signs(self) -> tuple[bool, str]:
        """
        Non-intrusive health check for the Kill Switch.
        Returns: (is_healthy, risk_message)
        """
        if not self.is_system_healthy:
            return False, "System Unhealthy (Previous Trigger)"
            
        # Check Daily Drawdown (using cached state)
        if hasattr(self, 'daily_start_balance') and self.daily_start_balance > 0:
            # We need current balance access. If not available, use last known metrics?
            # Ideally we pass current balance. But for the Kill Switch at start of loop, we might not have it.
            # Reliance on internal state 'is_system_healthy' set by previous update_health call is safest.
            pass
            
        return self.is_system_healthy, "OK"

    def receive_message(self, sender: Any, content: Any) -> Any:
        if isinstance(content, dict) and content.get('type') == 'CHECK_HEALTH':
            return self.get_health_report()
        return None
