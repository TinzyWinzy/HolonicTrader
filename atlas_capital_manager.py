#!/usr/bin/env python3
"""
Atlas Profit Architect - Capital Efficiency Manager
Manages capital allocation for maximum profitability
"""

import json
import numpy as np
from datetime import datetime, timedelta
from collections import deque

class CapitalEfficiencyManager:
    """
    Atlas Rule #2: Capital flows to proven edge only
    """
    
    def __init__(self, config_path='atlas_profit_config.json'):
        self.load_config(config_path)
        self.performance_history = deque(maxlen=100)
        self.capital_allocation = {
            'buy_strategy': 0.0,
            'reserve': 0.0,
            'total_deployed': 0.0
        }
        self.drawdown_tracker = {
            'peak_equity': 0.0,
            'current_equity': 0.0,
            'max_drawdown': 0.0,
            'drawdown_start': None
        }
        
    def load_config(self, config_path):
        """Load profit configuration"""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {
                'capital_allocation': {
                    'buy_strategy_capital': 0.8,
                    'reserve_capital': 0.2,
                    'max_position_size_pct': 0.05,
                    'daily_loss_limit_pct': 0.02,
                    'max_drawdown_pct': 0.15
                }
            }
    
    def initialize_capital(self, account_balance):
        """Initialize capital allocation based on account balance"""
        buy_capital = account_balance * self.config['capital_allocation']['buy_strategy_capital']
        reserve_capital = account_balance * self.config['capital_allocation']['reserve_capital']
        
        self.capital_allocation = {
            'buy_strategy': buy_capital,
            'reserve': reserve_capital,
            'total_deployed': 0.0,
            'account_balance': account_balance,
            'last_update': datetime.now().isoformat()
        }
        
        self.drawdown_tracker['peak_equity'] = account_balance
        self.drawdown_tracker['current_equity'] = account_balance
        
        return self.capital_allocation
    
    def update_performance(self, trade_result):
        """Update performance metrics after a trade"""
        self.performance_history.append(trade_result)
        
        # Update equity for drawdown tracking
        if 'pnl' in trade_result:
            self.drawdown_tracker['current_equity'] += trade_result['pnl']
            
            # Update peak equity
            if self.drawdown_tracker['current_equity'] > self.drawdown_tracker['peak_equity']:
                self.drawdown_tracker['peak_equity'] = self.drawdown_tracker['current_equity']
                self.drawdown_tracker['drawdown_start'] = None
            else:
                # Calculate current drawdown
                drawdown = (self.drawdown_tracker['current_equity'] - self.drawdown_tracker['peak_equity']) / \
                          self.drawdown_tracker['peak_equity']
                
                if drawdown < self.drawdown_tracker['max_drawdown']:
                    self.drawdown_tracker['max_drawdown'] = drawdown
                    if self.drawdown_tracker['drawdown_start'] is None:
                        self.drawdown_tracker['drawdown_start'] = datetime.now()
    
    def calculate_position_size(self, strategy_type, win_rate, win_loss_ratio, volatility_pct=0.01):
        """
        Calculate optimal position size using adaptive Kelly Criterion
        
        Atlas Rule: Scale only what works, survive before scaling
        """
        # Get available capital for this strategy
        if strategy_type.upper() == 'BUY':
            available_capital = self.capital_allocation.get('buy_strategy', 0)
        else:
            available_capital = 0  # SELL strategy disabled during isolation
        
        if available_capital <= 0:
            return 0.0
        
        # Calculate base Kelly fraction
        if win_loss_ratio > 0 and win_rate > 0:
            # Kelly formula: f* = (bp - q) / b
            b = win_loss_ratio
            p = win_rate
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b
            
            # Apply multiple safety factors:
            # 1. Conservative fraction (25% of Kelly)
            conservative_fraction = kelly_fraction * 0.25
            
            # 2. Volatility adjustment (reduce size in high volatility)
            vol_adjustment = 1.0 / (1.0 + volatility_pct * 10)
            
            # 3. Drawdown adjustment (reduce size during drawdown)
            drawdown = abs(self.drawdown_tracker['max_drawdown'])
            drawdown_adjustment = max(0.5, 1.0 - (drawdown / 0.15))  # Reduce up to 50%
            
            # 4. Performance consistency adjustment
            performance_score = self._calculate_performance_score()
            
            # Combined adjustment
            adjusted_fraction = conservative_fraction * vol_adjustment * \
                              drawdown_adjustment * performance_score
            
            # Ensure positive but not excessive
            adjusted_fraction = max(0.01, min(adjusted_fraction, 0.1))  # 1-10% range
            
        else:
            # Default conservative size if no performance data
            adjusted_fraction = 0.02  # 2%
        
        # Calculate position size
        position_size = available_capital * adjusted_fraction
        
        # Apply absolute limits
        min_size = 100.0  # $100 minimum for profitability
        max_size_pct = self.config['capital_allocation']['max_position_size_pct']
        max_size = available_capital * max_size_pct
        
        position_size = max(min_size, min(position_size, max_size))
        
        return position_size
    
    def _calculate_performance_score(self):
        """Calculate performance consistency score (0-1)"""
        if len(self.performance_history) < 10:
            return 0.7  # Default score for small sample
        
        # Calculate win rate consistency
        recent_trades = list(self.performance_history)[-20:]  # Last 20 trades
        wins = [t for t in recent_trades if t.get('pnl', 0) > 0]
        win_rate = len(wins) / len(recent_trades) if recent_trades else 0
        
        # Calculate expectancy consistency
        pnls = [t.get('pnl', 0) for t in recent_trades]
        expectancy = np.mean(pnls) if pnls else 0
        
        # Calculate Sharpe-like ratio (return/risk)
        if np.std(pnls) > 0:
            sharpe_ratio = expectancy / np.std(pnls)
        else:
            sharpe_ratio = 0
        
        # Combine metrics into score
        win_rate_score = min(1.0, win_rate / 0.7)  # Normalize to 70% target
        expectancy_score = min(1.0, abs(expectancy) * 100)  # $0.01 expectancy = 1.0
        sharpe_score = min(1.0, sharpe_ratio * 2)  # Sharpe 0.5 = 1.0
        
        # Weighted average
        performance_score = (
            win_rate_score * 0.4 +
            expectancy_score * 0.4 +
            sharpe_score * 0.2
        )
        
        return max(0.3, min(1.0, performance_score))  # Keep between 0.3-1.0
    
    def check_daily_loss_limit(self):
        """Check if daily loss limit is exceeded"""
        daily_loss_limit = self.config['capital_allocation']['daily_loss_limit_pct']
        max_daily_loss = self.capital_allocation.get('account_balance', 10000) * daily_loss_limit
        
        # Calculate today's PnL (simplified)
        today = datetime.now().date()
        today_trades = [
            t for t in self.performance_history 
            if datetime.fromisoformat(t.get('timestamp', '')).date() == today
        ]
        
        today_pnl = sum(t.get('pnl', 0) for t in today_trades)
        
        if today_pnl < -max_daily_loss:
            return False, f'DAILY_LOSS_LIMIT_EXCEEDED_{today_pnl:.2f}'
        
        return True, ''
    
    def check_drawdown_limit(self):
        """Check if maximum drawdown limit is exceeded"""
        max_drawdown_limit = self.config['capital_allocation']['max_drawdown_pct']
        current_drawdown = abs(self.drawdown_tracker['max_drawdown'])
        
        if current_drawdown > max_drawdown_limit:
            # Calculate drawdown duration
            duration = ""
            if self.drawdown_tracker['drawdown_start']:
                days = (datetime.now() - self.drawdown_tracker['drawdown_start']).days
                duration = f" for {days} days"
            
            return False, f'MAX_DRAWDOWN_EXCEEDED_{current_drawdown:.1%}{duration}'
        
        return True, ''
    
    def should_scale_capital(self):
        """
        Determine if capital should be scaled up
        
        Atlas Rule: Scale only after proven consistency
        """
        # Minimum trades for scaling consideration
        if len(self.performance_history) < 30:
            return False, 'INSUFFICIENT_TRADES_FOR_SCALING'
        
        # Check performance consistency
        recent_trades = list(self.performance_history)[-30:]  # Last 30 trades
        
        # Calculate metrics
        pnls = [t.get('pnl', 0) for t in recent_trades]
        wins = [p for p in pnls if p > 0]
        
        win_rate = len(wins) / len(pnls) if pnls else 0
        expectancy = np.mean(pnls) if pnls else 0
        profit_factor = sum(wins) / abs(sum(p for p in pnls if p < 0)) if any(p < 0 for p in pnls) else float('inf')
        
        # Scaling criteria
        criteria = {
            'min_trades': len(recent_trades) >= 30,
            'min_win_rate': win_rate >= 0.6,
            'min_expectancy': expectancy >= 0.002,
            'min_profit_factor': profit_factor >= 1.5,
            'max_drawdown': abs(self.drawdown_tracker['max_drawdown']) <= 0.1,
            'daily_loss_ok': self.check_daily_loss_limit()[0]
        }
        
        # Check all criteria
        all_met = all(criteria.values())
        
        if all_met:
            return True, 'SCALING_CRITERIA_MET'
        else:
            failed = [k for k, v in criteria.items() if not v]
            return False, f'SCALING_CRITERIA_FAILED_{failed}'
    
    def get_capital_allocation_report(self):
        """Generate capital allocation report"""
        report = {
            'capital_allocation': self.capital_allocation.copy(),
            'drawdown_metrics': self.drawdown_tracker.copy(),
            'performance_summary': {
                'total_trades': len(self.performance_history),
                'recent_win_rate': self._calculate_recent_win_rate(),
                'recent_expectancy': self._calculate_recent_expectancy(),
                'sharpe_ratio': self._calculate_sharpe_ratio()
            },
            'scaling_readiness': self.should_scale_capital(),
            'risk_limits': {
                'daily_loss_ok': self.check_daily_loss_limit()[0],
                'drawdown_ok': self.check_drawdown_limit()[0]
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def _calculate_recent_win_rate(self, n_trades=20):
        """Calculate win rate for recent trades"""
        if len(self.performance_history) < n_trades:
            n_trades = len(self.performance_history)
            
        recent = list(self.performance_history)[-n_trades:]
        wins = [t for t in recent if t.get('pnl', 0) > 0]
        
        return len(wins) / len(recent) if recent else 0
    
    def _calculate_recent_expectancy(self, n_trades=20):
        """Calculate expectancy for recent trades"""
        if len(self.performance_history) < n_trades:
            n_trades = len(self.performance_history)
            
        recent = list(self.performance_history)[-n_trades:]
        pnls = [t.get('pnl', 0) for t in recent]
        
        return np.mean(pnls) if pnls else 0
    
    def _calculate_sharpe_ratio(self, n_trades=30):
        """Calculate Sharpe-like ratio"""
        if len(self.performance_history) < n_trades:
            return 0
            
        recent = list(self.performance_history)[-n_trades:]
        pnls = [t.get('pnl', 0) for t in recent]
        
        if np.std(pnls) > 0:
            return np.mean(pnls) / np.std(pnls)
        return 0

# Example usage
if __name__ == "__main__":
    print("Atlas Capital Efficiency Manager - Testing")
    print("=" * 60)
    
    manager = CapitalEfficiencyManager()
    
    # Initialize with $10,000 account
    allocation = manager.initialize_capital(10000)
    print(f"Initial Capital Allocation:")
    print(f"  Buy Strategy: ${allocation['buy_strategy']:.2f}")
    print(f"  Reserve: ${allocation['reserve']:.2f}")
    print(f"  Total Account: ${allocation['account_balance']:.2f}")
    
    # Simulate some trades
    for i in range(5):
        trade_result = {
            'pnl': 25.0 if i % 3 != 0 else -15.0,
            'timestamp': datetime.now().isoformat(),
            'strategy': 'BUY'
        }
        manager.update_performance(trade_result)
    
    # Calculate position size
    position_size = manager.calculate_position_size(
        strategy_type='BUY',
        win_rate=0.65,
        win_loss_ratio=1.5,
        volatility_pct=0.01
    )
    print(f"\nRecommended Position Size: ${position_size:.2f}")
    
    # Check scaling readiness
    should_scale, reason = manager.should_scale_capital()
    print(f"\nScaling Ready: {should_scale} ({reason})")
    
    # Generate report
    report = manager.get_capital_allocation_report()
    print(f"\nPerformance Summary:")
    print(f"  Total Trades: {report['performance_summary']['total_trades']}")
    print(f"  Recent Win Rate: {report['performance_summary']['recent_win_rate']:.1%}")
    print(f"  Recent Expectancy: ${report['performance_summary']['recent_expectancy']:.4f}")
    
    print("\nCapital manager ready for integration")