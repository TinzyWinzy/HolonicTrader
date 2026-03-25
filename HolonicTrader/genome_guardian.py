"""
Genome Guardian - Monitors Live Brain Performance

Enforces tight monitoring on current genome:
- Alert at -3% drawdown
- Switch after 2nd loss
- Require 50%+ win rate over next 5 trades
- Auto-switch to Genome #2 if thresholds breached

Usage:
    from HolonicTrader.genome_guardian import GenomeGuardian
    
    guardian = GenomeGuardian()
    guardian.check_genome_performance(trade_result)
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class GenomeGuardian:
    """
    Tight monitoring for current live genome
    Ready to switch to Genome #2 if performance degrades
    """
    
    def __init__(self):
        self.hall_of_fame_path = Path('hall_of_fame.json')
        self.live_genome_path = Path('live_genome.json')
        self.backup_genome_path = Path('live_genome.backup.json')
        
        # Monitoring thresholds
        self.max_drawdown_alert = -0.03  # Alert at -3%
        self.max_drawdown_switch = -0.05  # Switch at -5%
        self.max_consecutive_losses = 2   # Switch after 2 losses
        self.min_win_rate_5trades = 0.50  # 50% win rate over 5 trades
        self.min_trades_for_evaluation = 5
        
        # State tracking
        self.state = {
            'initial_equity': 100.0,
            'peak_equity': 100.0,
            'current_equity': 100.0,
            'trades': [],  # List of {pnl_usd, pnl_percent, timestamp}
            'consecutive_losses': 0,
            'genome_switched': False,
            'switch_reason': None,
            'monitoring_start': datetime.now().isoformat()
        }
        
        # Load state if exists
        self._load_state()
        
        print("🛡️ Genome Guardian initialized")
        print(f"   Alert drawdown: {self.max_drawdown_alert:.1%}")
        print(f"   Switch drawdown: {self.max_drawdown_switch:.1%}")
        print(f"   Max consecutive losses: {self.max_consecutive_losses}")
        print(f"   Min win rate (5 trades): {self.min_win_rate_5trades:.0%}")
    
    def record_trade(self, pnl_usd: float, pnl_percent: float, symbol: str, equity: float):
        """Record trade outcome and check thresholds"""
        
        trade = {
            'pnl_usd': pnl_usd,
            'pnl_percent': pnl_percent,
            'symbol': symbol,
            'equity': equity,
            'timestamp': datetime.now().isoformat()
        }
        
        self.state['trades'].append(trade)
        self.state['current_equity'] = equity
        
        # Update peak equity
        if equity > self.state['peak_equity']:
            self.state['peak_equity'] = equity
        
        # Update consecutive losses
        if pnl_percent < 0:
            self.state['consecutive_losses'] += 1
        else:
            self.state['consecutive_losses'] = 0
        
        # Check all thresholds
        alerts = []
        switch_required = False
        switch_reason = None
        
        # Check 1: Drawdown alert
        drawdown = (self.state['peak_equity'] - equity) / self.state['peak_equity']
        if drawdown >= abs(self.max_drawdown_alert):
            alerts.append(f'⚠️ DRAWDOWN ALERT: {-drawdown:.1%}')
        
        # Check 2: Drawdown switch
        if drawdown >= abs(self.max_drawdown_switch):
            switch_required = True
            switch_reason = f'Drawdown {-drawdown:.1%} exceeded {-self.max_drawdown_switch:.1%} limit'
        
        # Check 3: Consecutive losses
        if self.state['consecutive_losses'] >= self.max_consecutive_losses:
            switch_required = True
            switch_reason = f'{self.state["consecutive_losses"]} consecutive losses'
        
        # Check 4: Win rate after 5 trades
        if len(self.state['trades']) >= self.min_trades_for_evaluation:
            recent_trades = self.state['trades'][-self.min_trades_for_evaluation:]
            wins = sum(1 for t in recent_trades if t['pnl_percent'] > 0)
            win_rate = wins / len(recent_trades)
            
            if win_rate < self.min_win_rate_5trades:
                switch_required = True
                switch_reason = f'Win rate {win_rate:.0%} < {self.min_win_rate_5trades:.0%} minimum'
        
        # Log status
        self._log_status(alerts, switch_required, switch_reason)
        
        # Execute switch if required
        if switch_required and not self.state['genome_switched']:
            self._execute_switch(switch_reason)
            return {
                'action': 'SWITCH',
                'reason': switch_reason,
                'backup_genome': self._get_backup_genome()
            }
        
        # Return alerts
        return {
            'action': 'MONITOR',
            'alerts': alerts,
            'trades': len(self.state['trades']),
            'win_rate': self._get_recent_win_rate(),
            'drawdown': -drawdown
        }
    
    def _log_status(self, alerts: list, switch_required: bool, switch_reason: str):
        """Log current status"""
        
        if alerts:
            for alert in alerts:
                print(f"🛡️ [GENOME GUARDIAN] {alert}")
        
        if switch_required:
            print(f"🛡️ [GENOME GUARDIAN] 🚨 SWITCH REQUIRED: {switch_reason}")
        
        # Print summary every 5 trades
        if len(self.state['trades']) % 5 == 0:
            win_rate = self._get_recent_win_rate()
            drawdown = (self.state['peak_equity'] - self.state['current_equity']) / self.state['peak_equity']
            print(f"🛡️ [GENOME GUARDIAN] Status: {len(self.state['trades'])} trades, "
                  f"{win_rate:.0%} win rate, {-drawdown:.1%} drawdown")
    
    def _execute_switch(self, reason: str):
        """Switch to backup genome"""
        
        print(f"\n🛡️ [GENOME GUARDIAN] 🚨 EXECUTING GENOME SWITCH")
        print(f"   Reason: {reason}")
        print(f"   Switching to Genome #2 (11 trades, 18% win rate)")
        
        # Backup current genome
        if self.live_genome_path.exists():
            import shutil
            shutil.copy(self.live_genome_path, self.backup_genome_path)
            print(f"   ✅ Backed up current genome to live_genome.backup.json")
        
        # Load Genome #2 from Hall of Fame
        backup_genome = self._get_backup_genome()
        
        if backup_genome:
            # Write as new live genome
            live_data = {
                'genome': backup_genome['genome'],
                'final_equity': backup_genome['final_equity'],
                'roi': backup_genome['roi'],
                'win_rate': backup_genome['win_rate'],
                'trades': backup_genome['trades'],
                'max_dd': backup_genome['max_dd'],
                'sharpe': backup_genome['sharpe'],
                'sortino': backup_genome['sortino'],
                'fitness': backup_genome['fitness'],
                'validation_roi': backup_genome.get('validation_roi', 0),
                'validation_trades': backup_genome.get('validation_trades', 0),
                'test_score': backup_genome.get('test_score', 1.0),
                'violation': backup_genome.get('violation', False),
                'timestamp': backup_genome['timestamp'],
                'source_island': backup_genome['source_island'],
                'switched_by': 'GenomeGuardian',
                'switch_reason': reason,
                'switch_timestamp': datetime.now().isoformat()
            }
            
            with open(self.live_genome_path, 'w') as f:
                json.dump(live_data, f, indent=2)
            
            print(f"   ✅ Genome #2 deployed successfully")
            print(f"      RSI: {backup_genome['genome']['rsi_buy']:.1f}/{backup_genome['genome']['rsi_sell']:.1f}")
            print(f"      SL/TP: {backup_genome['genome']['stop_loss']:.1%} / {backup_genome['genome']['take_profit']:.1%}")
            print(f"      Leverage: {backup_genome['genome']['leverage_cap']:.1f}x")
            
            self.state['genome_switched'] = True
            self.state['switch_reason'] = reason
            self._save_state()
        else:
            print(f"   ❌ Failed to load backup genome")
    
    def _get_backup_genome(self) -> Optional[Dict[str, Any]]:
        """Get Genome #2 from Hall of Fame"""
        
        if not self.hall_of_fame_path.exists():
            return None
        
        with open(self.hall_of_fame_path) as f:
            hof = json.load(f)
        
        if len(hof) < 2:
            return None
        
        # Return #2 (index 1) - most reliable with 11 trades
        return hof[1]
    
    def _get_recent_win_rate(self) -> float:
        """Calculate win rate over last 5 trades"""
        if len(self.state['trades']) < 1:
            return 0.0
        
        recent = self.state['trades'][-min(5, len(self.state['trades'])):]
        wins = sum(1 for t in recent if t['pnl_percent'] > 0)
        return wins / len(recent)
    
    def _load_state(self):
        """Load state from file"""
        state_path = Path('genome_guardian_state.json')
        if state_path.exists():
            try:
                with open(state_path) as f:
                    saved_state = json.load(f)
                self.state.update(saved_state)
                print(f"   📊 Loaded state: {len(self.state['trades'])} trades recorded")
            except:
                pass
    
    def _save_state(self):
        """Save state to file"""
        state_path = Path('genome_guardian_state.json')
        with open(state_path, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            'trades': len(self.state['trades']),
            'win_rate': self._get_recent_win_rate(),
            'consecutive_losses': self.state['consecutive_losses'],
            'drawdown': -(self.state['peak_equity'] - self.state['current_equity']) / self.state['peak_equity'],
            'genome_switched': self.state['genome_switched'],
            'switch_reason': self.state['switch_reason']
        }


# Singleton
_guardian_instance = None

def get_genome_guardian() -> GenomeGuardian:
    """Get guardian singleton"""
    global _guardian_instance
    if _guardian_instance is None:
        _guardian_instance = GenomeGuardian()
    return _guardian_instance


# Convenience function for integration
def monitor_trade(pnl_usd: float, pnl_percent: float, symbol: str, equity: float) -> Dict[str, Any]:
    """Monitor trade and check if genome switch required"""
    guardian = get_genome_guardian()
    return guardian.record_trade(pnl_usd, pnl_percent, symbol, equity)


print("🛡️ Genome Guardian loaded - Tight monitoring active")
