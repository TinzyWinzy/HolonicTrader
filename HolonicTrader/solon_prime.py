"""
Solon Prime - Capital Intelligence Architect

Enforces:
1. Capital Preservation (Babylon Core)
2. Structural Trade Validation (CTKS Core)
3. Expectancy Enforcement
4. Entropy Monitoring (AEHML)
5. Adaptive Learning Loop

Usage:
    from HolonicTrader.solon_prime import SolonPrime
    
    solon = SolonPrime()
    decision = solon.evaluate_trade(signal, portfolio_state)
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

class SolonPrime:
    """
    Capital Intelligence Architect
    Enforces discipline, structure, and adaptation for profitability
    """
    
    def __init__(self):
        # Babylon Core - Capital Preservation
        self.risk_per_trade = 0.01  # ≤1%
        self.max_daily_loss = 0.03  # ≤3%
        self.max_drawdown = 0.10    # ≤10%
        
        # CTKS Core - Structural Validation
        self.min_rr_ratio = 2.0     # Minimum 2:1 reward:risk
        self.min_expectancy = 0.0   # Must be positive
        
        # AEHML Core - Entropy Monitoring
        self.entropy_threshold = 0.7  # High entropy = reduce trading
        self.entropy_window = 10      # Last N trades for entropy calc
        
        # Tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.recent_trades = []  # Last N trades for entropy
        self.last_reset = datetime.now().date()
        
        print("🏛️ Solon Prime initialized - Capital Intelligence Active")
    
    def evaluate_trade(self, signal: Dict[str, Any], portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate trade through Solon's 7-layer filter
        
        Returns:
        {
            'action': 'APPROVE' | 'REJECT' | 'HALT',
            'reason': str,
            'risk_params': dict,
            'entropy_level': float
        }
        """
        # Layer 1: Capital Preservation Check
        capital_check = self._check_capital_preservation(portfolio)
        if capital_check['action'] != 'APPROVE':
            return capital_check
        
        # Layer 2: Structural Validation
        structural_check = self._validate_structure(signal)
        if structural_check['action'] != 'APPROVE':
            return structural_check
        
        # Layer 3: Expectancy Check
        expectancy_check = self._check_expectancy(signal)
        if expectancy_check['action'] != 'APPROVE':
            return expectancy_check
        
        # Layer 4: Entropy Check
        entropy_check = self._check_entropy()
        if entropy_check['action'] != 'APPROVE':
            return entropy_check
        
        # Layer 5: Trade Quality Filter
        quality_check = self._filter_trade_quality(signal)
        if quality_check['action'] != 'APPROVE':
            return quality_check
        
        # Layer 6: Strategic Alignment
        alignment_check = self._check_strategic_alignment(signal, portfolio)
        if alignment_check['action'] != 'APPROVE':
            return alignment_check
        
        # All layers passed - APPROVE with risk parameters
        return {
            'action': 'APPROVE',
            'reason': 'All Solon layers passed',
            'risk_params': {
                'position_size': self._calculate_position_size(signal, portfolio),
                'stop_loss': signal.get('stop_loss'),
                'take_profit': signal.get('take_profit'),
                'max_risk_usd': portfolio.get('equity', 0) * self.risk_per_trade
            },
            'entropy_level': self._calculate_entropy(),
            'system_state': self.get_system_state()
        }
    
    def _check_capital_preservation(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Layer 1: Babylon Core - Protect Capital First"""
        
        # Check daily loss limit
        if abs(self.daily_pnl) >= self.max_daily_loss * portfolio.get('equity', 100):
            return {
                'action': 'HALT',
                'reason': f'Daily loss limit hit (${self.daily_pnl:.2f})',
                'next_step': 'Resume trading tomorrow'
            }
        
        # Check max drawdown
        current_dd = portfolio.get('drawdown', 0)
        if current_dd >= self.max_drawdown:
            return {
                'action': 'HALT',
                'reason': f'Max drawdown exceeded ({current_dd:.1%})',
                'next_step': 'Diagnostic cycle required'
            }
        
        # Check consecutive losses
        consecutive_losses = portfolio.get('consecutive_losses', 0)
        if consecutive_losses >= 3:
            return {
                'action': 'REJECT',
                'reason': f'{consecutive_losses} consecutive losses - reduce size',
                'risk_reduction': 0.5  # 50% size reduction
            }
        
        return {'action': 'APPROVE'}
    
    def _validate_structure(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Layer 2: CTKS Core - Structural Validation"""
        
        # Must have defined entry level
        if not signal.get('entry_price'):
            return {
                'action': 'REJECT',
                'reason': 'No defined entry price'
            }
        
        # Must have stop loss
        if not signal.get('stop_loss'):
            return {
                'action': 'REJECT',
                'reason': 'No stop loss defined'
            }
        
        # Must have take profit
        if not signal.get('take_profit'):
            return {
                'action': 'REJECT',
                'reason': 'No take profit defined'
            }
        
        return {'action': 'APPROVE'}
    
    def _check_expectancy(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Layer 3: Expectancy Enforcement"""
        
        # Calculate reward:risk ratio
        entry = signal.get('entry_price', 0)
        stop = signal.get('stop_loss', 0)
        target = signal.get('take_profit', 0)
        
        if entry == 0 or stop == 0 or target == 0:
            return {
                'action': 'REJECT',
                'reason': 'Invalid price levels for expectancy calc'
            }
        
        risk = abs(entry - stop)
        reward = abs(target - entry)
        
        if risk == 0:
            return {
                'action': 'REJECT',
                'reason': 'Zero risk (stop = entry)'
            }
        
        rr_ratio = reward / risk
        
        if rr_ratio < self.min_rr_ratio:
            return {
                'action': 'REJECT',
                'reason': f'RR ratio {rr_ratio:.2f}:1 < minimum {self.min_rr_ratio}:1'
            }
        
        # Calculate expectancy (simplified - uses win rate from ML)
        win_prob = signal.get('ml_win_prob', 0.5)
        expectancy = (win_prob * reward) - ((1 - win_prob) * risk)
        
        if expectancy <= 0:
            return {
                'action': 'REJECT',
                'reason': f'Negative expectancy (${expectancy:.2f} per trade)'
            }
        
        return {
            'action': 'APPROVE',
            'expectancy': expectancy,
            'rr_ratio': rr_ratio
        }
    
    def _check_entropy(self) -> Dict[str, Any]:
        """Layer 4: AEHML Entropy Monitoring"""
        
        entropy = self._calculate_entropy()
        
        if entropy > self.entropy_threshold:
            return {
                'action': 'REJECT',
                'reason': f'High entropy ({entropy:.2f}) - system disorder detected',
                'recommendation': 'Reduce trading frequency, tighten filters'
            }
        
        # Moderate entropy - reduce size
        if entropy > self.entropy_threshold * 0.7:
            return {
                'action': 'APPROVE',
                'risk_reduction': 0.7,  # 30% size reduction
                'entropy_level': entropy
            }
        
        return {'action': 'APPROVE', 'entropy_level': entropy}
    
    def _calculate_entropy(self) -> float:
        """Calculate system entropy from recent trades"""
        
        if len(self.recent_trades) < 3:
            return 0.3  # Default low entropy
        
        # Entropy from win/loss consistency
        wins = sum(1 for t in self.recent_trades if t > 0)
        losses = len(self.recent_trades) - wins
        
        # Perfect consistency = 0 entropy, 50/50 = 1.0 entropy
        consistency = abs(wins - losses) / len(self.recent_trades)
        entropy = 1.0 - consistency
        
        # Add PnL variance component
        if len(self.recent_trades) >= 5:
            import numpy as np
            pnl_std = np.std(self.recent_trades)
            pnl_mean = abs(np.mean(self.recent_trades))
            variance_component = min(1.0, pnl_std / max(pnl_mean, 0.001))
            entropy = (entropy + variance_component) / 2
        
        return entropy
    
    def _filter_trade_quality(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Layer 5: Trade Quality Filter (Anti-Noise)"""
        
        # ML confidence check
        ml_conf = signal.get('ml_confidence', 'MEDIUM')
        ml_prob = signal.get('ml_win_prob', 0.5)
        
        if ml_prob < 0.45:
            return {
                'action': 'REJECT',
                'reason': f'Low ML confidence ({ml_prob:.1%})'
            }
        
        # Quality score check
        quality = signal.get('quality', 'MEDIUM')
        if quality == 'VETOED':
            return {
                'action': 'REJECT',
                'reason': 'Signal quality vetoed'
            }
        
        return {'action': 'APPROVE'}
    
    def _check_strategic_alignment(self, signal: Dict[str, Any], portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Layer 6: Strategic Alignment"""
        
        # Check if trading against current regime
        regime = signal.get('regime', 'UNKNOWN')
        if regime == 'CHAOTIC':
            return {
                'action': 'REJECT',
                'reason': 'Chaotic regime - stand aside'
            }
        
        # Check correlation with existing positions
        symbol = signal.get('symbol', '')
        existing_positions = portfolio.get('positions', {})
        
        # Don't add more than 2 positions in same family
        family = self._get_symbol_family(symbol)
        family_count = sum(1 for s in existing_positions if self._get_symbol_family(s) == family)
        
        if family_count >= 2:
            return {
                'action': 'REJECT',
                'reason': f'Already {family_count} positions in {family} family'
            }
        
        return {'action': 'APPROVE'}
    
    def _get_symbol_family(self, symbol: str) -> str:
        """Group symbols by family for correlation check"""
        families = {
            'BTC': 'CRYPTO_MAJOR',
            'ETH': 'CRYPTO_MAJOR',
            'SOL': 'L1_ALT',
            'AVAX': 'L1_ALT',
            'DOT': 'L1_ALT',
            'TAO': 'AI_TOKEN',
            'LDO': 'DEFI',
            'AAVE': 'DEFI',
            'XRP': 'PAYMENT',
            'DOGE': 'MEME',
            'PEPE': 'MEME',
            'WIF': 'MEME'
        }
        
        base = symbol.split('/')[0]
        return families.get(base, 'OTHER')
    
    def _calculate_position_size(self, signal: Dict[str, Any], portfolio: Dict[str, Any]) -> float:
        """Calculate position size based on risk parameters"""
        
        equity = portfolio.get('equity', 100)
        risk_amount = equity * self.risk_per_trade
        
        entry = signal.get('entry_price', 0)
        stop = signal.get('stop_loss', 0)
        
        if entry == 0 or stop == 0:
            return 0
        
        risk_per_unit = abs(entry - stop)
        
        if risk_per_unit == 0:
            return 0
        
        position_size = risk_amount / risk_per_unit
        
        return position_size
    
    def record_trade_outcome(self, symbol: str, pnl_usd: float, pnl_percent: float):
        """Record trade outcome for entropy tracking"""
        
        self.recent_trades.append(pnl_percent)
        
        # Keep only last N trades
        if len(self.recent_trades) > self.entropy_window:
            self.recent_trades = self.recent_trades[-self.entropy_window:]
        
        # Update daily PnL
        self.daily_pnl += pnl_usd
        self.daily_trades += 1
        
        # Check if day reset needed
        if datetime.now().date() != self.last_reset:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset = datetime.now().date()
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get current system state"""
        
        entropy = self._calculate_entropy()
        
        if entropy > self.entropy_threshold:
            state = 'HIGH_ENTROPY'
        elif entropy > self.entropy_threshold * 0.7:
            state = 'MODERATE_ENTROPY'
        else:
            state = 'LOW_ENTROPY'
        
        return {
            'capital_health': 'STABLE' if abs(self.daily_pnl) < self.max_daily_loss else 'AT_RISK',
            'entropy_level': state,
            'entropy_score': entropy,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'recent_win_rate': sum(1 for t in self.recent_trades if t > 0) / max(len(self.recent_trades), 1)
        }


# Singleton instance
_solon_instance = None

def get_solon() -> SolonPrime:
    """Get Solon Prime singleton"""
    global _solon_instance
    if _solon_instance is None:
        _solon_instance = SolonPrime()
    return _solon_instance


# Convenience function
def evaluate_trade(signal: Dict[str, Any], portfolio: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate trade through Solon's filters"""
    solon = get_solon()
    return solon.evaluate_trade(signal, portfolio)


print("🏛️ Solon Prime loaded - Capital Intelligence Active")
