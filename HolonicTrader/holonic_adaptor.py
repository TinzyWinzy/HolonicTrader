"""
Holonic Adaptor - Market Resonance Layer (Phase 48)

Purpose:
1. Detect market regime (volatility, trend, entropy)
2. Adapt all parameters to resonate with current regime
3. Provide unified signal routing through Oracle
4. Learn which frequencies work in which conditions

Philosophy:
- Market is not an adversary to defeat
- Market is an ecosystem to harmonize with
- Parameters should flow WITH market, not against it
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from collections import deque
import json
import os

try:
    import config
except ImportError:
    import sys
    sys.path.append('..')
    import config


# ============================================================================
# REGIME DEFINITIONS
# ============================================================================

# CHRONOS FIX: Use config-based conviction threshold (was hardcoded 0.50-0.80)
# Base conviction threshold from config, regime-specific adjustments applied below
BASE_MIN_CONVICTION = getattr(config, 'MINIMUM_CONVICTION_THRESHOLD', 0.65)

MARKET_REGIMES = {
    'HIGH_VOL_TRENDING': {
        'description': 'High volatility, strong trends (whale moves)',
        'trailing_mult': 5.5,      # Wide room for noise
        'profit_targets': {'rapid': 0.025, 'normal': 0.06, 'runner': 0.12},
        'position_size_mult': 1.25,  # Confidence in trends
        'cooldown_type': 'state',    # Wait for structure reset
        'min_conviction': BASE_MIN_CONVICTION - 0.05,  # Lower threshold (trends are clear)
    },
    'LOW_VOL_MEAN_REVERT': {
        'description': 'Low volatility, range-bound (choppy)',
        'trailing_mult': 3.0,      # Tight captures
        'profit_targets': {'rapid': 0.015, 'normal': 0.03, 'runner': 0.06},
        'position_size_mult': 0.75,  # Reduce size (chop kills)
        'cooldown_type': 'time',     # Time-based ok in ranges
        # FIX 2026-03-15 (Chronos v2): Lowered from BASE+0.10 (0.75) to BASE+0.05 (0.70).
        # OLD: 0.75 caused 34.7% of all session vetoes (CRITICAL_OVERPROTECTION).
        # Still above BASE, so false-break protection is maintained.
        'min_conviction': BASE_MIN_CONVICTION + 0.05,
    },
    'TRANSITION_CHAOS': {
        'description': 'Regime change, high entropy (dangerous)',
        'trailing_mult': 4.5,      # Balanced
        'profit_targets': {'rapid': 0.02, 'normal': 0.04, 'runner': 0.08},
        'position_size_mult': 0.50,  # Reduce size (uncertainty)
        'cooldown_type': 'state',    # Wait for clarity
        'min_conviction': BASE_MIN_CONVICTION + 0.15,  # Highest threshold (danger)
    },
    'BULL_MOMENTUM': {
        'description': 'Sustained upward momentum (risk-on)',
        'trailing_mult': 4.0,      # Let runners go
        'profit_targets': {'rapid': 0.02, 'normal': 0.05, 'runner': 0.15},
        'position_size_mult': 1.50,  # Aggressive (tailwind)
        'cooldown_type': 'state',
        'min_conviction': BASE_MIN_CONVICTION,  # Base threshold
    },
    'BEAR_DISTRIBUTION': {
        'description': 'Sustained downward pressure (risk-off)',
        'trailing_mult': 4.0,      # Tight stops
        'profit_targets': {'rapid': 0.015, 'normal': 0.03, 'runner': 0.06},
        'position_size_mult': 0.60,  # Defensive
        'cooldown_type': 'state',
        'min_conviction': BASE_MIN_CONVICTION + 0.10,  # Higher threshold
    }
}


class HolonicAdaptor:
    """
    Market Resonance Layer
    
    Continuously monitors market state and adjusts system parameters
    to maintain harmony with current conditions.
    """
    
    def __init__(self):
        self.current_regime = 'TRANSITION_CHAOS'  # Default (safe)
        self.regime_confidence = 0.5
        self.last_regime_change = time.time()
        
        # Entropy tracking
        self.entropy_window = deque(maxlen=50)
        self.volatility_window = deque(maxlen=50)
        self.trend_strength_window = deque(maxlen=50)
        
        # Holonic memory (learning)
        self.holonic_memory = self._load_holonic_memory()
        
        # Parameter cache (avoid recalculating)
        self._cached_params = None
        self._cache_timestamp = 0
        self._cache_ttl = 60  # 1 minute cache
        
        # Entropy thresholds
        self.HIGH_ENTROPY_THRESHOLD = 0.75
        self.LOW_ENTROPY_THRESHOLD = 0.35
        
    def _load_holonic_memory(self) -> Dict:
        """Load historical regime performance"""
        memory_path = os.path.join(os.path.dirname(__file__), 'holonic_memory.json')
        if os.path.exists(memory_path):
            try:
                with open(memory_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'regime_performance': {},  # {regime: {signal_type: {win_rate, avg_win, avg_loss}}}
            'last_updated': time.time()
        }
    
    def save_holonic_memory(self):
        """Persist learning to disk"""
        memory_path = os.path.join(os.path.dirname(__file__), 'holonic_memory.json')
        self.holonic_memory['last_updated'] = time.time()
        try:
            with open(memory_path, 'w') as f:
                json.dump(self.holonic_memory, f, indent=2)
        except Exception as e:
            print(f"[HolonicAdaptor] Memory save failed: {e}")
    
    def record_trade_feedback(self, regime: str, signal_type: str, 
                              win: bool, pnl_pct: float, metadata: Dict = None):
        """
        Record trade outcome for regime-specific learning
        
        This builds the "holonic memory" - what works in which regimes
        """
        if regime not in self.holonic_memory['regime_performance']:
            self.holonic_memory['regime_performance'][regime] = {}
        
        if signal_type not in self.holonic_memory['regime_performance'][regime]:
            self.holonic_memory['regime_performance'][regime][signal_type] = {
                'trades': 0,
                'wins': 0,
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'win_rate': 0.0
            }
        
        stats = self.holonic_memory['regime_performance'][regime][signal_type]
        stats['trades'] += 1
        stats['total_pnl'] += pnl_pct
        
        if win:
            stats['wins'] += 1
            # Running average for wins
            n = stats['wins']
            stats['avg_win'] = ((n-1) * stats['avg_win'] + pnl_pct) / n
        else:
            # Running average for losses
            losses = stats['trades'] - stats['wins']
            stats['avg_loss'] = ((losses-1) * stats['avg_loss'] + abs(pnl_pct)) / losses
        
        stats['win_rate'] = stats['wins'] / stats['trades']
        
        # Persist periodically
        if stats['trades'] % 10 == 0:
            self.save_holonic_memory()
    
    def calculate_market_entropy(self, prices: np.ndarray) -> float:
        """
        Calculate market entropy (disorder/chaos measure)
        
        Uses permutation entropy - higher = more chaotic/unpredictable
        """
        if len(prices) < 10:
            return 0.5  # Default (insufficient data)
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Normalize to [0, 1]
        if returns.std() > 0:
            returns_norm = (returns - returns.min()) / (returns.max() - returns.min() + 1e-10)
        else:
            return 0.5
        
        # Permutation entropy (simplified)
        # Divide into bins and measure distribution
        n_bins = 5
        hist, _ = np.histogram(returns_norm, bins=n_bins, range=(0, 1))
        probs = hist / hist.sum()
        
        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10)) / np.log2(n_bins)
        
        return float(entropy)
    
    def detect_regime(self, market_data: Dict[str, Any]) -> str:
        """
        Detect current market regime based on multiple signals
        
        market_data should contain:
        - prices: recent price array
        - volumes: recent volume array
        - atr: average true range
        - trend_indicator: e.g., ADX, or calculated
        """
        prices = market_data.get('prices', np.array([]))
        volumes = market_data.get('volumes', np.array([]))
        atr = market_data.get('atr', 0.0)
        
        if len(prices) < 20:
            return 'TRANSITION_CHAOS'
        
        # 1. Calculate Entropy
        entropy = self.calculate_market_entropy(prices)
        self.entropy_window.append(entropy)
        
        # 2. Calculate Volatility (rolling std of returns)
        returns = np.diff(prices) / prices[:-1]
        volatility = returns.std()
        self.volatility_window.append(volatility)
        
        # 3. Calculate Trend Strength (simplified: |mean return| / std return)
        if returns.std() > 0:
            trend_strength = abs(returns.mean()) / returns.std()
        else:
            trend_strength = 0.0
        self.trend_strength_window.append(trend_strength)
        
        # Get rolling averages
        avg_entropy = np.mean(self.entropy_window) if self.entropy_window else 0.5
        avg_volatility = np.mean(self.volatility_window) if self.volatility_window else 0.02
        avg_trend = np.mean(self.trend_strength_window) if self.trend_strength_window else 0.5
        
        # 4. Regime Classification Logic
        regime = 'TRANSITION_CHAOS'
        confidence = 0.5
        
        if avg_entropy > self.HIGH_ENTROPY_THRESHOLD:
            # Chaotic - reduce risk
            regime = 'TRANSITION_CHAOS'
            confidence = 1.0 - avg_entropy
        elif avg_entropy < self.LOW_ENTROPY_THRESHOLD:
            # Ordered - can be more aggressive
            if avg_trend > 1.0:
                regime = 'BULL_MOMENTUM' if returns.mean() > 0 else 'BEAR_DISTRIBUTION'
            else:
                regime = 'LOW_VOL_MEAN_REVERT'
            confidence = 1.0 - avg_entropy
        else:
            # Medium entropy
            if avg_trend > 0.8 and avg_volatility > 0.03:
                regime = 'HIGH_VOL_TRENDING'
                confidence = avg_trend
            elif avg_trend < 0.3:
                regime = 'LOW_VOL_MEAN_REVERT'
                confidence = 1.0 - avg_trend
        
        # Detect regime change
        if regime != self.current_regime:
            old_regime = self.current_regime
            self.current_regime = regime
            self.regime_confidence = confidence
            self.last_regime_change = time.time()
            self._cached_params = None  # Invalidate cache
            
            print(f"[HolonicAdaptor] 🌊 REGIME CHANGE: {old_regime} → {regime} (confidence: {confidence:.2f})")
        else:
            self.regime_confidence = 0.9 * self.regime_confidence + 0.1 * confidence
        
        return regime
    
    def get_adaptive_parameters(self, market_data: Dict[str, Any] = None) -> Dict:
        """
        Get parameters adapted to current market regime
        
        Returns dict with all adjustable parameters
        """
        # Check cache
        now = time.time()
        if self._cached_params and (now - self._cache_timestamp) < self._cache_ttl:
            return self._cached_params
        
        # Detect regime if data provided
        if market_data:
            self.detect_regime(market_data)
        
        # Get base regime parameters
        regime_params = MARKET_REGIMES.get(self.current_regime, MARKET_REGIMES['TRANSITION_CHAOS'])
        
        # Apply holonic learning adjustments
        learned_adjustments = self._get_learned_adjustments()
        
        # Build adaptive parameters
        adaptive_params = {
            'regime': self.current_regime,
            'regime_confidence': self.regime_confidence,
            
            # Trailing stops (adapted)
            'PREDATOR_TRAILING_STOP_ATR_MULT': regime_params['trailing_mult'],
            
            # Profit targets (adapted)
            'PROFIT_TARGETS': regime_params['profit_targets'],
            
            # Position sizing (adapted)
            'POSITION_SIZE_MULTIPLIER': regime_params['position_size_mult'],
            
            # Cooldown type
            'COOLDOWN_TYPE': regime_params['cooldown_type'],
            
            # Conviction threshold
            'MIN_CONVICTION': regime_params['min_conviction'],
            
            # Apply learned adjustments
            **learned_adjustments
        }
        
        # Cache and return
        self._cached_params = adaptive_params
        self._cache_timestamp = now
        
        return adaptive_params
    
    def _get_learned_adjustments(self) -> Dict:
        """
        Get parameter adjustments based on holonic memory
        
        If certain signal types perform poorly in current regime,
        automatically adjust to compensate
        """
        adjustments = {}
        
        regime_perf = self.holonic_memory['regime_performance'].get(self.current_regime, {})
        
        # Check if we have enough data
        total_trades = sum(
            stats.get('trades', 0) 
            for stats in regime_perf.values()
        )
        
        if total_trades < 20:
            return adjustments  # Not enough data
        
        # Calculate overall regime performance
        total_wins = sum(stats.get('wins', 0) for stats in regime_perf.values())
        overall_win_rate = total_wins / total_trades if total_trades > 0 else 0.5
        
        # If win rate is poor, increase conviction requirement
        if overall_win_rate < 0.35:
            adjustments['MIN_CONVICTION'] = adjustments.get('MIN_CONVICTION', 0.65) + 0.10
            adjustments['POSITION_SIZE_MULTIPLIER'] = adjustments.get('POSITION_SIZE_MULTIPLIER', 1.0) * 0.75
        
        # If win rate is excellent, can be more aggressive
        elif overall_win_rate > 0.55:
            adjustments['POSITION_SIZE_MULTIPLIER'] = adjustments.get('POSITION_SIZE_MULTIPLIER', 1.0) * 1.15
        
        return adjustments
    
    def should_allow_trade(self, signal_type: str, current_conviction: float, 
                          metadata: Dict = None) -> Tuple[bool, str]:
        """
        Holonic trade permission check
        
        Considers:
        1. Current regime
        2. Signal type performance in this regime
        3. Conviction level
        4. Recent learning
        """
        params = self.get_adaptive_parameters()
        
        # 1. Check conviction threshold
        min_conviction = params.get('MIN_CONVICTION', 0.65)
        if current_conviction < min_conviction:
            return False, f"Conviction {current_conviction:.2f} < {min_conviction:.2f} ({self.current_regime})"
        
        # 2. Check historical performance for this signal type in this regime
        regime_perf = self.holonic_memory['regime_performance'].get(self.current_regime, {})
        signal_perf = regime_perf.get(signal_type, None)
        
        if signal_perf and signal_perf.get('trades', 0) >= 10:
            # Have enough history
            if signal_perf['win_rate'] < 0.30:
                return False, f"{signal_type} underperforming in {self.current_regime} ({signal_perf['win_rate']:.1%})"
        
        # 3. Regime-specific checks
        if self.current_regime == 'TRANSITION_CHAOS':
            # Extra cautious in chaos
            if current_conviction < 0.75:
                return False, "Chaos regime requires 0.75+ conviction"
        
        elif self.current_regime == 'LOW_VOL_MEAN_REVERT':
            # FIX 2026-03-15 (Chronos v2): Removed redundant secondary conviction check.
            # min_conviction from MARKET_REGIMES (BASE+0.05=0.70) already handles filtering.
            # Old double-check was blocking 34.7% of all signals (CRITICAL_OVERPROTECTION).
            pass  # Conviction gate already applied above via min_conviction

        return True, "Holonic approval granted"
    
    def get_regime_context(self) -> Dict:
        """Get full regime context for logging/debugging"""
        return {
            'current_regime': self.current_regime,
            'regime_description': MARKET_REGIMES[self.current_regime]['description'],
            'confidence': self.regime_confidence,
            'time_in_regime': time.time() - self.last_regime_change,
            'entropy_avg': np.mean(self.entropy_window) if self.entropy_window else 0.5,
            'volatility_avg': np.mean(self.volatility_window) if self.volatility_window else 0.02,
            'trend_avg': np.mean(self.trend_strength_window) if self.trend_strength_window else 0.5,
            'adaptive_params': self.get_adaptive_parameters()
        }


# ============================================================================
# GLOBAL INSTANCE (for import)
# ============================================================================

_holonic_adaptor_instance = None

def get_holonic_adaptor() -> HolonicAdaptor:
    """Get or create global HolonicAdaptor instance"""
    global _holonic_adaptor_instance
    if _holonic_adaptor_instance is None:
        _holonic_adaptor_instance = HolonicAdaptor()
    return _holonic_adaptor_instance
