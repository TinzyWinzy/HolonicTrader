import logging
import time
from typing import Dict, List, Any

logger = logging.getLogger("EvoMonitor")

class EvolutionMonitor:
    def __init__(self):
        self.alert_thresholds = {
            'fitness_inflation': 2.0,    # 2x jump in one cycle
            'equity_divergence': 0.5,     # fitness up, equity down
            'sharpe_ceiling': 4.0,        # suspected overfitting
            'sim_live_divergence': 0.30   # FIX #6: Track sim vs live PnL gap
        }
        self.history = {} # island_name -> history_dicts
        # FIX #6: Live PnL tracking
        self.live_pnl_history = []  # Track actual trading PnL
        self.simulated_pnl_history = []  # Track simulated evolution PnL
        
    def record_live_pnl(self, pnl_pct: float, timestamp: float = None):
        """FIX #6: Record actual live trading PnL for comparison."""
        if timestamp is None:
            timestamp = time.time()
        self.live_pnl_history.append({
            'timestamp': timestamp,
            'pnl_pct': pnl_pct
        })
        # Keep last 100 entries
        self.live_pnl_history = self.live_pnl_history[-100:]
        
    def record_simulated_pnl(self, pnl_pct: float, island_name: str, timestamp: float = None):
        """FIX #6: Record simulated evolution PnL by island."""
        if timestamp is None:
            timestamp = time.time()
        self.simulated_pnl_history.append({
            'timestamp': timestamp,
            'island': island_name,
            'pnl_pct': pnl_pct
        })
        # Keep last 100 entries
        self.simulated_pnl_history = self.simulated_pnl_history[-100:]
        
    def get_pnl_divergence(self) -> Dict[str, Any]:
        """FIX #6: Calculate divergence between simulated and live PnL."""
        if not self.live_pnl_history or not self.simulated_pnl_history:
            return {'divergence': 0.0, 'status': 'INSUFFICIENT_DATA'}
            
        # Average live PnL (last 20 entries)
        recent_live = self.live_pnl_history[-20:]
        avg_live_pnl = sum(p['pnl_pct'] for p in recent_live) / len(recent_live)
        
        # Average simulated PnL (last 20 entries)
        recent_sim = self.simulated_pnl_history[-20:]
        avg_sim_pnl = sum(p['pnl_pct'] for p in recent_sim) / len(recent_sim)
        
        divergence = avg_sim_pnl - avg_live_pnl
        
        status = 'OK'
        if abs(divergence) > self.alert_thresholds['sim_live_divergence']:
            status = 'CRITICAL_DIVERGENCE'
        elif abs(divergence) > 0.15:
            status = 'WARNING'
            
        return {
            'divergence': divergence,
            'avg_live_pnl': avg_live_pnl,
            'avg_sim_pnl': avg_sim_pnl,
            'status': status,
            'live_samples': len(recent_live),
            'sim_samples': len(recent_sim)
        }

    def check_health(self, island_name: str, current_metrics: Dict[str, Any]) -> List[str]:
        alerts = []
        if island_name not in self.history:
            self.history[island_name] = []
            self.history[island_name].append(current_metrics)
            return alerts

        prev = self.history[island_name][-1]

        # 1. Fitness Inflation (without ROI growth)
        if current_metrics['fitness'] > prev['fitness'] * self.alert_thresholds['fitness_inflation']:
            if current_metrics['roi'] <= prev['roi']:
                alerts.append(f"☢️ FITNESS INFLATION: {island_name} fitness doubled while ROI stalled!")

        # 2. Equity/Fitness Divergence
        if current_metrics['fitness'] > prev['fitness'] and current_metrics.get('final_equity', 0) < prev.get('final_equity', 0):
             alerts.append(f"📉 DIVERGENCE: {island_name} fitness up, but equity down.")

        # 3. Sharpe Ceiling
        if current_metrics.get('sharpe', 0) > self.alert_thresholds['sharpe_ceiling']:
             alerts.append(f"🚩 OVERFIT SUSPECT: {island_name} Sharpe {current_metrics['sharpe']:.2f} exceeds ceiling.")

        # 4. FIX #6: Sim/Live PnL Divergence Alert
        pnl_div = self.get_pnl_divergence()
        if pnl_div['status'] == 'CRITICAL_DIVERGENCE':
            alerts.append(f"🚨 SIM/LIVE DIVERGENCE: Sim PnL {pnl_div['avg_sim_pnl']*100:.1f}% vs Live PnL {pnl_div['avg_live_pnl']*100:.1f}% (Gap: {pnl_div['divergence']*100:.1f}%)")

        # Record simulated PnL for tracking
        self.record_simulated_pnl(current_metrics.get('roi', 0), island_name)

        # Update History
        self.history[island_name].append(current_metrics)
        return alerts

    def get_summary(self):
        summary = "--- EVO MONITOR SUMMARY ---\n"
        for island, logs in self.history.items():
            if logs:
                last = logs[-1]
                summary += f"{island}: Fit {last['fitness']:.2f} | ROI {last['roi']*100:.1f}%\n"
        
        # FIX #6: Add PnL Divergence Status
        pnl_div = self.get_pnl_divergence()
        summary += f"\nFIX #6 - SIM/LIVE DIVERGENCE: {pnl_div['status']}\n"
        if pnl_div['status'] != 'INSUFFICIENT_DATA':
            summary += f"  Sim Avg PnL: {pnl_div['avg_sim_pnl']*100:.2f}% | Live Avg PnL: {pnl_div['avg_live_pnl']*100:.2f}%\n"
            summary += f"  Gap: {pnl_div['divergence']*100:.2f}%\n"
        return summary
