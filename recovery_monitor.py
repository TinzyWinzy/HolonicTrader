#!/usr/bin/env python3
"""
Helix Strategy Surgeon - Recovery Monitoring System
Real-time monitoring of system recovery progress
"""

import json
import time
from datetime import datetime, timedelta
import threading
from collections import deque
import numpy as np

class RecoveryMonitor:
    """Monitor recovery progress and enforce constraints"""
    
    def __init__(self):
        self.load_config()
        self.metrics_history = deque(maxlen=1000)
        self.alerts = []
        self.recovery_phase = "STABILIZATION"
        self.start_time = datetime.now()
        
        # Performance thresholds
        self.thresholds = {
            'expectancy': 0.001,      # Positive expectancy required
            'win_rate': 0.55,         # Minimum win rate
            'cost_ratio': 0.5,        # Costs < 50% of profits
            'max_drawdown': -0.15,    # Maximum 15% drawdown
            'profit_factor': 1.5      # $1.50 profit per $1 loss
        }
        
    def load_config(self):
        """Load recovery configuration"""
        try:
            with open('recovery_config.json', 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {
                'recovery_phase': 'STABILIZATION',
                'monitoring_interval_minutes': 5,
                'allowed_strategies': ['LONG_ONLY']
            }
    
    def calculate_metrics(self, trade_data):
        """Calculate current performance metrics"""
        if not trade_data:
            return {}
        
        # Convert to numeric
        pnls = [float(t['pnl']) for t in trade_data]
        pnl_pct = [float(t['pnl_percent']) for t in trade_data]
        
        # Basic metrics
        total_trades = len(pnls)
        winning_trades = sum(1 for p in pnls if p > 0)
        losing_trades = sum(1 for p in pnls if p < 0)
        
        if total_trades == 0:
            return {}
        
        win_rate = winning_trades / total_trades
        
        # Expectancy
        avg_win = np.mean([p for p in pnls if p > 0]) if winning_trades > 0 else 0
        avg_loss = abs(np.mean([p for p in pnls if p < 0])) if losing_trades > 0 else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Profit factor
        total_profits = sum(p for p in pnls if p > 0)
        total_losses = abs(sum(p for p in pnls if p < 0))
        profit_factor = total_profits / total_losses if total_losses > 0 else float('inf')
        
        # Drawdown (simplified)
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-10)
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_trades': total_trades,
            'win_rate': win_rate,
            'expectancy': expectancy,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': avg_win / avg_loss if avg_loss > 0 else 0,
            'total_pnl': sum(pnls),
            'avg_pnl_percent': np.mean(pnl_pct)
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def check_thresholds(self, metrics):
        """Check metrics against recovery thresholds"""
        violations = []
        
        if metrics.get('expectancy', 0) < self.thresholds['expectancy']:
            violations.append(f"Expectancy below threshold: {metrics['expectancy']:.4f}")
            
        if metrics.get('win_rate', 0) < self.thresholds['win_rate']:
            violations.append(f"Win rate below threshold: {metrics['win_rate']:.2%}")
            
        if metrics.get('profit_factor', float('inf')) < self.thresholds['profit_factor']:
            violations.append(f"Profit factor below threshold: {metrics['profit_factor']:.2f}")
            
        if metrics.get('max_drawdown', 0) < self.thresholds['max_drawdown']:
            violations.append(f"Max drawdown exceeded: {metrics['max_drawdown']:.2%}")
        
        return violations
    
    def generate_recovery_report(self):
        """Generate comprehensive recovery report"""
        if not self.metrics_history:
            return {
                'status': 'NO_DATA',
                'message': 'No trade data available for analysis'
            }
        
        latest_metrics = self.metrics_history[-1]
        violations = self.check_thresholds(latest_metrics)
        
        # Calculate recovery progress
        recovery_duration = datetime.now() - self.start_time
        
        # Determine recovery phase
        if violations:
            recovery_phase = "STABILIZATION"
        elif latest_metrics['expectancy'] > 0.002 and latest_metrics['win_rate'] > 0.6:
            recovery_phase = "OPTIMIZATION"
        else:
            recovery_phase = "VALIDATION"
        
        report = {
            'recovery_phase': recovery_phase,
            'recovery_duration_hours': recovery_duration.total_seconds() / 3600,
            'current_metrics': latest_metrics,
            'threshold_violations': violations,
            'system_status': 'HEALTHY' if not violations else 'UNHEALTHY',
            'recommended_actions': self.generate_actions(violations, recovery_phase),
            'monitoring_active': True,
            'last_update': datetime.now().isoformat()
        }
        
        return report
    
    def generate_actions(self, violations, recovery_phase):
        """Generate recommended actions based on violations"""
        actions = []
        
        if recovery_phase == "STABILIZATION":
            actions.append("Maintain SELL strategy disabled")
            actions.append("Focus on BUY strategy optimization")
            actions.append("Monitor execution costs closely")
            
        if "Expectancy below threshold" in violations:
            actions.append("Review entry/exit logic")
            actions.append("Check for signal lag")
            actions.append("Validate stop-loss placement")
            
        if "Win rate below threshold" in violations:
            actions.append("Improve signal filtering")
            actions.append("Add confirmation indicators")
            actions.append("Review market regime compatibility")
            
        if "Profit factor below threshold" in violations:
            actions.append("Increase reward:risk ratio")
            actions.append("Reduce position sizes")
            actions.append("Improve take-profit levels")
            
        if recovery_phase == "OPTIMIZATION":
            actions.append("Consider gradual SELL strategy reintroduction")
            actions.append("Test redesigned short strategy in paper trading")
            actions.append("Monitor regime detection effectiveness")
        
        return actions
    
    def enforce_constraints(self, trade_signal):
        """Enforce recovery constraints on trade signals"""
        constraints = {
            'allowed': True,
            'reasons': []
        }
        
        # Check recovery phase constraints
        if self.recovery_phase == "STABILIZATION":
            # Only allow LONG trades during stabilization
            if trade_signal.get('direction', '').upper() == 'SELL':
                constraints['allowed'] = False
                constraints['reasons'].append("SELL_STRATEGY_DISABLED_IN_STABILIZATION")
        
        # Load execution constraints
        try:
            with open('execution_constraints.json', 'r') as f:
                exec_constraints = json.load(f)
                
                # Check minimum trade size
                min_size = exec_constraints.get('min_trade_size_usd', 25.0)
                trade_size = trade_signal.get('size_usd', 0)
                
                if trade_size < min_size:
                    constraints['allowed'] = False
                    constraints['reasons'].append(f"TRADE_SIZE_BELOW_MINIMUM_{trade_size}<{min_size}")
                    
        except FileNotFoundError:
            pass
        
        return constraints
    
    def start_monitoring(self, trade_data_callback, interval_minutes=5):
        """Start continuous monitoring"""
        def monitor_loop():
            while True:
                try:
                    # Get latest trade data
                    trade_data = trade_data_callback()
                    
                    # Calculate metrics
                    metrics = self.calculate_metrics(trade_data)
                    
                    if metrics:
                        # Check thresholds
                        violations = self.check_thresholds(metrics)
                        
                        # Generate report
                        report = self.generate_recovery_report()
                        
                        # Save report
                        with open('recovery_status.json', 'w') as f:
                            json.dump(report, f, indent=2)
                        
                        # Log alerts if any
                        if violations:
                            alert_msg = f"Threshold violations detected: {violations}"
                            self.alerts.append({
                                'timestamp': datetime.now().isoformat(),
                                'message': alert_msg,
                                'severity': 'WARNING'
                            })
                            print(f"[RECOVERY MONITOR] WARNING: {alert_msg}")
                        
                except Exception as e:
                    print(f"[RECOVERY MONITOR] Error: {e}")
                
                time.sleep(interval_minutes * 60)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
        print(f"[RECOVERY MONITOR] Started with {interval_minutes} minute interval")
        return monitor_thread

# Utility function to load trade data
def load_recent_trades():
    """Load recent trades for monitoring"""
    try:
        import csv
        trades = []
        with open('trades_dump_since_20260314.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                trades.append(row)
        return trades[-50:]  # Last 50 trades
    except FileNotFoundError:
        return []

if __name__ == "__main__":
    print("Helix Recovery Monitoring System")
    print("=" * 50)
    
    # Create monitor
    monitor = RecoveryMonitor()
    
    # Load recent trades
    trades = load_recent_trades()
    print(f"Loaded {len(trades)} recent trades")
    
    # Calculate current metrics
    metrics = monitor.calculate_metrics(trades)
    
    if metrics:
        print("\nCURRENT METRICS:")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.2%}")
        print(f"  Expectancy: ${metrics['expectancy']:.4f}")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  Total PnL: ${metrics['total_pnl']:.4f}")
        
        # Check thresholds
        violations = monitor.check_thresholds(metrics)
        if violations:
            print("\nTHRESHOLD VIOLATIONS:")
            for violation in violations:
                print(f"  • {violation}")
        else:
            print("\nALL THRESHOLDS MET ✓")
        
        # Generate report
        report = monitor.generate_recovery_report()
        print(f"\nRECOVERY PHASE: {report['recovery_phase']}")
        print(f"SYSTEM STATUS: {report['system_status']}")
        
        if report['recommended_actions']:
            print("\nRECOMMENDED ACTIONS:")
            for action in report['recommended_actions']:
                print(f"  • {action}")
    
    print("\nMonitoring system ready - reports saved to 'recovery_status.json'")