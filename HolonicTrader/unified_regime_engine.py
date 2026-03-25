"""
UnifiedRegimeEngine - Merged Market Regime State Machine

Purpose:
    Consolidates the dual regime systems (HolonicAdaptor + SMCERegimeEngine)
    into a single unified state machine to eliminate conflicts and simplify
    regime-based decision making.

Design Philosophy:
    - Single source of truth for market regime
    - Two-dimensional regime state: (Behavioral, Operational)
    - Behavioral: HOW the market is behaving (volatility, entropy, trend)
    - Operational: WHAT operations are allowed (entry permissions, sizing)
    - Deterministic transitions with hysteresis to prevent chattering

Author: Chronos Market Forensics v3
Date: 2026-03-16
"""

import time
import logging
import numpy as np
from typing import Dict, Optional, Tuple, Any, List
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("UnifiedRegimeEngine")


# ============================================================================
# ENUMS: Regime Dimensions
# ============================================================================

class BehavioralRegime(Enum):
    """
    Describes HOW the market is behaving.
    Based on: entropy, volatility, trend strength
    Maps from old HolonicAdaptor regimes.
    """
    LOW_VOL_MEAN_REVERT = "LOW_VOL_MEAN_REVERT"      # Low entropy, low trend (choppy)
    HIGH_VOL_TRENDING = "HIGH_VOL_TRENDING"          # Medium entropy, high trend (directional)
    BULL_MOMENTUM = "BULL_MOMENTUM"                  # Low entropy, strong uptrend
    BEAR_DISTRIBUTION = "BEAR_DISTRIBUTION"          # Low entropy, strong downtrend
    TRANSITION_CHAOS = "TRANSITION_CHAOS"            # High entropy (dangerous)


class OperationalRegime(Enum):
    """
    Describes WHAT operations are allowed.
    Based on: structure, liquidity, correlation, drawdown
    Maps from old SMCERegimeEngine regimes.
    """
    HARVEST = "HARVEST"           # Full operations, normal sizing
    EXPANSION = "EXPANSION"       # Full operations, aggressive sizing
    TRANSITION = "TRANSITION"     # Restricted operations, reduced sizing
    DEFENSIVE = "DEFENSIVE"       # No new entries, exit only


# ============================================================================
# Unified Regime State
# ============================================================================

@dataclass
class RegimeState:
    """Complete regime state at a point in time."""
    timestamp: float
    behavioral: BehavioralRegime
    operational: OperationalRegime
    confidence: float  # 0.0 - 1.0
    
    # Input metrics
    entropy: float = 0.5
    volatility: float = 0.02
    trend_strength: float = 0.5
    structure: str = "NEUTRAL"
    liquidity_status: str = "healthy"
    correlation_idx: float = 0.3
    drawdown_breach: bool = False
    
    # Derived permissions
    entries_allowed: bool = True
    size_modifier: float = 1.0
    max_leverage: float = 3.0
    min_conviction: float = 0.65
    trailing_stop_mult: float = 4.0
    profit_targets: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'behavioral': self.behavioral.value,
            'operational': self.operational.value,
            'confidence': self.confidence,
            'entropy': self.entropy,
            'volatility': self.volatility,
            'trend_strength': self.trend_strength,
            'structure': self.structure,
            'liquidity_status': self.liquidity_status,
            'correlation_idx': self.correlation_idx,
            'drawdown_breach': self.drawdown_breach,
            'entries_allowed': self.entries_allowed,
            'size_modifier': self.size_modifier,
            'max_leverage': self.max_leverage,
            'min_conviction': self.min_conviction,
            'trailing_stop_mult': self.trailing_stop_mult,
        }


# ============================================================================
# Configuration Constants
# ============================================================================

# Entropy thresholds (from HolonicAdaptor)
ENTROPY_HIGH_THRESHOLD = 0.75   # Above = chaotic
ENTROPY_LOW_THRESHOLD = 0.35    # Below = ordered

# SMCE entropy thresholds (different scale - raw vs normalized)
SMCE_DEFENSIVE_ENTROPY_MIN = 1.2
SMCE_HARVEST_ENTROPY_MAX = 1.15
SMCE_TRANSITION_ENTROPY_MIN = 0.9
SMCE_TRANSITION_ENTROPY_MAX = 1.2

# Correlation thresholds
CORRELATION_DEFENSIVE_MIN = 0.7  # Above = dangerous correlation spike

# Trend strength thresholds
TREND_STRONG_THRESHOLD = 1.0     # Above = strong trend
TREND_WEAK_THRESHOLD = 0.3       # Below = no clear trend

# Volatility thresholds
VOLATILITY_HIGH_THRESHOLD = 0.03  # Above = high volatility

# Regime transition hysteresis (prevent chattering)
HYSTERESIS_WINDOW = 0.05  # Require 5% change before transitioning


# ============================================================================
# Unified Regime Configuration
# ============================================================================

UNIFIED_REGIME_CONFIG: Dict[Tuple[BehavioralRegime, OperationalRegime], Dict] = {
    # =========================================================================
    # HARVEST operational regimes (normal operations)
    # =========================================================================
    (BehavioralRegime.LOW_VOL_MEAN_REVERT, OperationalRegime.HARVEST): {
        'description': 'Choppy but safe - mean reversion opportunities',
        'entries_allowed': True,
        'size_modifier': 0.75,       # Reduce size in chop
        'max_leverage': 3.0,
        'min_conviction': 0.65,      # BASE + 0.05 (lowered from 0.70)
        'trailing_stop_mult': 3.0,   # Tight stops for ranges
        'profit_targets': {'rapid': 0.015, 'normal': 0.03, 'runner': 0.06},
        'cooldown_type': 'time',
    },
    (BehavioralRegime.HIGH_VOL_TRENDING, OperationalRegime.HARVEST): {
        'description': 'High vol trending - follow the move',
        'entries_allowed': True,
        'size_modifier': 1.25,
        'max_leverage': 3.0,
        'min_conviction': 0.60,      # BASE - 0.05 (trends are clear)
        'trailing_stop_mult': 5.5,   # Wide room for noise
        'profit_targets': {'rapid': 0.025, 'normal': 0.06, 'runner': 0.12},
        'cooldown_type': 'state',
    },
    (BehavioralRegime.BULL_MOMENTUM, OperationalRegime.HARVEST): {
        'description': 'Bull momentum - aggressive long bias',
        'entries_allowed': True,
        'size_modifier': 1.50,
        'max_leverage': 4.0,
        'min_conviction': 0.65,      # BASE
        'trailing_stop_mult': 4.0,
        'profit_targets': {'rapid': 0.02, 'normal': 0.05, 'runner': 0.15},
        'cooldown_type': 'state',
    },
    (BehavioralRegime.BEAR_DISTRIBUTION, OperationalRegime.HARVEST): {
        'description': 'Bear distribution - short bias',
        'entries_allowed': True,
        'size_modifier': 0.75,       # Increased from 0.60
        'max_leverage': 3.0,
        'min_conviction': 0.60,      # Reduced from 0.70 (was blocking shorts)
        'trailing_stop_mult': 4.0,
        'profit_targets': {'rapid': 0.015, 'normal': 0.03, 'runner': 0.06},
        'cooldown_type': 'state',
    },
    (BehavioralRegime.TRANSITION_CHAOS, OperationalRegime.HARVEST): {
        'description': 'Chaotic but liquid - moderate caution',
        'entries_allowed': True,
        'size_modifier': 0.65,       # Increased from 0.50 (more size in chaos)
        'max_leverage': 3.0,         # Increased from 2.0
        'min_conviction': 0.60,      # Reduced from 0.75 (was blocking most signals)
        'trailing_stop_mult': 4.0,   # Slightly wider stops
        'profit_targets': {'rapid': 0.02, 'normal': 0.04, 'runner': 0.08},
        'cooldown_type': 'state',
    },
    
    # =========================================================================
    # EXPANSION operational regimes (aggressive growth)
    # =========================================================================
    (BehavioralRegime.BULL_MOMENTUM, OperationalRegime.EXPANSION): {
        'description': 'Optimal long conditions - press advantage',
        'entries_allowed': True,
        'size_modifier': 2.0,
        'max_leverage': 5.0,
        'min_conviction': 0.60,      # Lower for high-probability setup
        'trailing_stop_mult': 4.5,
        'profit_targets': {'rapid': 0.025, 'normal': 0.06, 'runner': 0.18},
        'cooldown_type': 'state',
    },
    (BehavioralRegime.HIGH_VOL_TRENDING, OperationalRegime.EXPANSION): {
        'description': 'Strong trend - expansion mode',
        'entries_allowed': True,
        'size_modifier': 1.5,
        'max_leverage': 4.0,
        'min_conviction': 0.60,
        'trailing_stop_mult': 5.5,
        'profit_targets': {'rapid': 0.03, 'normal': 0.07, 'runner': 0.15},
        'cooldown_type': 'state',
    },
    # Fallback: if behavioral doesn't match exactly, use HARVEST config
    (BehavioralRegime.LOW_VOL_MEAN_REVERT, OperationalRegime.EXPANSION): {
        'description': 'Expansion regime but choppy market',
        'entries_allowed': True,
        'size_modifier': 1.0,
        'max_leverage': 3.0,
        'min_conviction': 0.65,
        'trailing_stop_mult': 3.5,
        'profit_targets': {'rapid': 0.02, 'normal': 0.04, 'runner': 0.10},
        'cooldown_type': 'time',
    },
    (BehavioralRegime.TRANSITION_CHAOS, OperationalRegime.EXPANSION): {
        'description': 'Expansion regime but chaotic - stay cautious',
        'entries_allowed': True,
        'size_modifier': 0.50,
        'max_leverage': 2.0,
        'min_conviction': 0.75,
        'trailing_stop_mult': 4.5,
        'profit_targets': {'rapid': 0.02, 'normal': 0.04, 'runner': 0.08},
        'cooldown_type': 'state',
    },
    (BehavioralRegime.BEAR_DISTRIBUTION, OperationalRegime.EXPANSION): {
        'description': 'Expansion regime but bearish - short bias only',
        'entries_allowed': True,
        'size_modifier': 0.75,
        'max_leverage': 3.0,
        'min_conviction': 0.70,
        'trailing_stop_mult': 4.0,
        'profit_targets': {'rapid': 0.015, 'normal': 0.03, 'runner': 0.08},
        'cooldown_type': 'state',
    },
    
    # =========================================================================
    # TRANSITION operational regimes (restricted)
    # =========================================================================
    (BehavioralRegime.LOW_VOL_MEAN_REVERT, OperationalRegime.TRANSITION): {
        'description': 'Transition + choppy - wait for clarity',
        'entries_allowed': True,
        'size_modifier': 0.35,       # Halved from normal
        'max_leverage': 2.0,
        'min_conviction': 0.70,
        'trailing_stop_mult': 3.5,
        'profit_targets': {'rapid': 0.01, 'normal': 0.02, 'runner': 0.04},
        'cooldown_type': 'state',
    },
    (BehavioralRegime.HIGH_VOL_TRENDING, OperationalRegime.TRANSITION): {
        'description': 'Transition + trending - reduced size',
        'entries_allowed': True,
        'size_modifier': 0.60,
        'max_leverage': 2.0,
        'min_conviction': 0.65,
        'trailing_stop_mult': 5.0,
        'profit_targets': {'rapid': 0.02, 'normal': 0.04, 'runner': 0.08},
        'cooldown_type': 'state',
    },
    (BehavioralRegime.TRANSITION_CHAOS, OperationalRegime.TRANSITION): {
        'description': 'Double transition - maximum caution',
        'entries_allowed': True,
        'size_modifier': 0.25,
        'max_leverage': 1.5,
        'min_conviction': 0.80,
        'trailing_stop_mult': 5.0,
        'profit_targets': {'rapid': 0.015, 'normal': 0.03, 'runner': 0.06},
        'cooldown_type': 'state',
    },
    (BehavioralRegime.BULL_MOMENTUM, OperationalRegime.TRANSITION): {
        'description': 'Transition but bullish - selective longs',
        'entries_allowed': True,
        'size_modifier': 0.50,
        'max_leverage': 2.0,
        'min_conviction': 0.70,
        'trailing_stop_mult': 4.0,
        'profit_targets': {'rapid': 0.02, 'normal': 0.04, 'runner': 0.10},
        'cooldown_type': 'state',
    },
    (BehavioralRegime.BEAR_DISTRIBUTION, OperationalRegime.TRANSITION): {
        'description': 'Transition but bearish - selective shorts',
        'entries_allowed': True,
        'size_modifier': 0.40,
        'max_leverage': 2.0,
        'min_conviction': 0.75,
        'trailing_stop_mult': 4.0,
        'profit_targets': {'rapid': 0.015, 'normal': 0.03, 'runner': 0.06},
        'cooldown_type': 'state',
    },
    
    # =========================================================================
    # DEFENSIVE operational regimes (no entries)
    # =========================================================================
    # All behavioral regimes map to same defensive config
    (BehavioralRegime.LOW_VOL_MEAN_REVERT, OperationalRegime.DEFENSIVE): {
        'description': 'DEFENSIVE - no new entries',
        'entries_allowed': False,
        'size_modifier': 0.0,
        'max_leverage': 1.0,
        'min_conviction': 0.99,      # Effectively blocks all
        'trailing_stop_mult': 2.0,   # Tighten existing positions
        'profit_targets': {'rapid': 0.01, 'normal': 0.02, 'runner': 0.03},
        'cooldown_type': 'state',
    },
    (BehavioralRegime.HIGH_VOL_TRENDING, OperationalRegime.DEFENSIVE): {
        'description': 'DEFENSIVE - no new entries',
        'entries_allowed': False,
        'size_modifier': 0.0,
        'max_leverage': 1.0,
        'min_conviction': 0.99,
        'trailing_stop_mult': 2.5,
        'profit_targets': {'rapid': 0.01, 'normal': 0.02, 'runner': 0.03},
        'cooldown_type': 'state',
    },
    (BehavioralRegime.BULL_MOMENTUM, OperationalRegime.DEFENSIVE): {
        'description': 'DEFENSIVE - no new entries',
        'entries_allowed': False,
        'size_modifier': 0.0,
        'max_leverage': 1.0,
        'min_conviction': 0.99,
        'trailing_stop_mult': 2.5,
        'profit_targets': {'rapid': 0.01, 'normal': 0.02, 'runner': 0.03},
        'cooldown_type': 'state',
    },
    (BehavioralRegime.BEAR_DISTRIBUTION, OperationalRegime.DEFENSIVE): {
        'description': 'DEFENSIVE - no new entries',
        'entries_allowed': False,
        'size_modifier': 0.0,
        'max_leverage': 1.0,
        'min_conviction': 0.99,
        'trailing_stop_mult': 2.5,
        'profit_targets': {'rapid': 0.01, 'normal': 0.02, 'runner': 0.03},
        'cooldown_type': 'state',
    },
    (BehavioralRegime.TRANSITION_CHAOS, OperationalRegime.DEFENSIVE): {
        'description': 'DEFENSIVE - no new entries',
        'entries_allowed': False,
        'size_modifier': 0.0,
        'max_leverage': 1.0,
        'min_conviction': 0.99,
        'trailing_stop_mult': 3.0,
        'profit_targets': {'rapid': 0.01, 'normal': 0.02, 'runner': 0.03},
        'cooldown_type': 'state',
    },
}


# ============================================================================
# Unified Regime Engine
# ============================================================================

class UnifiedRegimeEngine:
    """
    Unified Market Regime State Machine
    
    Combines behavioral detection (HolonicAdaptor) with operational gating
    (SMCERegimeEngine) into a single coherent system.
    
    Usage:
        engine = UnifiedRegimeEngine()
        
        # Update with market data
        state = engine.update(
            prices=price_array,
            volumes=volume_array,
            atr=atr_value,
            structure="NEUTRAL",
            liquidity_status="healthy",
            correlation_idx=0.35,
            drawdown_breach=False,
        )
        
        # Get permissions
        if state.entries_allowed and conviction >= state.min_conviction:
            # Trade with state.size_modifier and state.max_leverage
    """
    
    def __init__(self):
        self.state: Optional[RegimeState] = None
        self.previous_state: Optional[RegimeState] = None
        
        # Rolling windows for regime detection
        self.entropy_window = deque(maxlen=50)
        self.volatility_window = deque(maxlen=50)
        self.trend_window = deque(maxlen=50)
        
        # Transition tracking
        self.transition_log: List[Tuple[float, str, str, str]] = []  # (ts, from, to, reason)
        self.last_update: float = 0.0
        
        # Hysteresis tracking (prevent chattering)
        self._last_entropy_avg: float = 0.5
        self._last_trend_avg: float = 0.5
        
        logger.info("[UnifiedRegimeEngine] Initialized")
    
    # ========================================================================
    # Public API
    # ========================================================================
    
    def update(
        self,
        prices: np.ndarray,
        volumes: np.ndarray = None,
        atr: float = None,
        structure: str = "NEUTRAL",
        liquidity_status: str = "healthy",
        correlation_idx: float = 0.3,
        drawdown_breach: bool = False,
    ) -> RegimeState:
        """
        Update regime state based on current market data.
        
        Args:
            prices: Recent price array (OHLC close or mid prices)
            volumes: Recent volume array (optional)
            atr: Average True Range (optional)
            structure: Market structure ('BULLISH', 'BEARISH', 'NEUTRAL', 'SUPPORT', 'RESISTANCE')
            liquidity_status: Liquidity condition ('healthy', 'warning', 'critical')
            correlation_idx: Aggregate correlation index (0.0-1.0)
            drawdown_breach: True if daily/weekly drawdown limit hit
        
        Returns:
            RegimeState with current regime and permissions
        """
        now = time.time()
        
        # Calculate metrics
        entropy = self._calculate_entropy(prices)
        volatility = self._calculate_volatility(prices)
        trend_strength = self._calculate_trend_strength(prices)
        
        # Update rolling windows
        self.entropy_window.append(entropy)
        self.volatility_window.append(volatility)
        self.trend_window.append(trend_strength)
        
        # Get rolling averages
        avg_entropy = np.mean(self.entropy_window) if self.entropy_window else 0.5
        avg_volatility = np.mean(self.volatility_window) if self.volatility_window else 0.02
        avg_trend = np.mean(self.trend_window) if self.trend_window else 0.5
        
        # Check hysteresis (skip regime change if within window)
        skip_behavioral_change = False
        if self.state:
            entropy_change = abs(avg_entropy - self._last_entropy_avg)
            trend_change = abs(avg_trend - self._last_trend_avg)
            if entropy_change < HYSTERESIS_WINDOW and trend_change < HYSTERESIS_WINDOW:
                skip_behavioral_change = True
        
        # Determine behavioral regime
        if skip_behavioral_change and self.state:
            behavioral = self.state.behavioral
            confidence = self.state.confidence * 0.95 + 0.05  # Decay slightly
        else:
            behavioral, confidence = self._compute_behavioral_regime(
                avg_entropy, avg_volatility, avg_trend, prices
            )
            self._last_entropy_avg = avg_entropy
            self._last_trend_avg = avg_trend
        
        # Determine operational regime
        operational = self._compute_operational_regime(
            structure, entropy, liquidity_status, correlation_idx, drawdown_breach
        )
        
        # Get unified config
        config = self._get_unified_config(behavioral, operational)
        
        # Build state
        new_state = RegimeState(
            timestamp=now,
            behavioral=behavioral,
            operational=operational,
            confidence=confidence,
            entropy=avg_entropy,
            volatility=avg_volatility,
            trend_strength=avg_trend,
            structure=structure,
            liquidity_status=liquidity_status,
            correlation_idx=correlation_idx,
            drawdown_breach=drawdown_breach,
            entries_allowed=config['entries_allowed'],
            size_modifier=config['size_modifier'],
            max_leverage=config['max_leverage'],
            min_conviction=config['min_conviction'],
            trailing_stop_mult=config['trailing_stop_mult'],
            profit_targets=config['profit_targets'],
        )
        
        # Check for regime change
        if self.state is None or self._is_significant_change(self.state, new_state):
            self._log_transition(new_state)
            self.previous_state = self.state
            self.state = new_state
        else:
            # Smooth confidence
            if self.state:
                self.state.confidence = self.state.confidence * 0.9 + confidence * 0.1
        
        self.last_update = now
        return self.state
    
    def get_permissions(self) -> Dict:
        """Get current trading permissions."""
        if not self.state:
            return {
                'entries_allowed': False,
                'size_modifier': 0.5,
                'max_leverage': 1.0,
                'min_conviction': 0.75,
                'reason': 'No regime state yet',
            }
        
        return {
            'entries_allowed': self.state.entries_allowed,
            'size_modifier': self.state.size_modifier,
            'max_leverage': self.state.max_leverage,
            'min_conviction': self.state.min_conviction,
            'trailing_stop_mult': self.state.trailing_stop_mult,
            'profit_targets': self.state.profit_targets,
            'behavioral': self.state.behavioral.value,
            'operational': self.state.operational.value,
            'confidence': self.state.confidence,
        }
    
    def should_allow_entry(
        self,
        conviction: float,
        direction: str = None,
        symbol: str = None,
    ) -> Tuple[bool, str]:
        """
        Check if an entry should be allowed under current regime.
        
        Args:
            conviction: Signal conviction score (0.0-1.0)
            direction: Trade direction ('LONG' or 'SHORT'), optional
            symbol: Trading symbol, optional
        
        Returns:
            (allowed, reason) tuple
        """
        if not self.state:
            return False, "Regime state not initialized"
        
        # Check operational gate
        if not self.state.entries_allowed:
            return False, f"DEFENSIVE regime - entries blocked"
        
        # Check conviction
        if conviction < self.state.min_conviction:
            return False, f"Conviction {conviction:.2f} < {self.state.min_conviction:.2f} ({self.state.behavioral.value})"
        
        # Check direction vs regime alignment
        if direction:
            if self.state.behavioral == BehavioralRegime.BEAR_DISTRIBUTION and direction == 'LONG':
                if conviction < 0.75:
                    return False, "Bear regime - long requires 0.75+ conviction"
            elif self.state.behavioral == BehavioralRegime.BULL_MOMENTUM and direction == 'SHORT':
                if conviction < 0.75:
                    return False, "Bull regime - short requires 0.75+ conviction"
        
        return True, "Entry approved"
    
    def get_status_summary(self) -> Dict:
        """Get full regime status for logging/monitoring."""
        if not self.state:
            return {'status': 'uninitialized'}
        
        return {
            'behavioral_regime': self.state.behavioral.value,
            'operational_regime': self.state.operational.value,
            'confidence': self.state.confidence,
            'entries_allowed': self.state.entries_allowed,
            'size_modifier': self.state.size_modifier,
            'max_leverage': self.state.max_leverage,
            'min_conviction': self.state.min_conviction,
            'metrics': {
                'entropy': self.state.entropy,
                'volatility': self.state.volatility,
                'trend_strength': self.state.trend_strength,
                'structure': self.state.structure,
                'liquidity': self.state.liquidity_status,
                'correlation': self.state.correlation_idx,
                'drawdown_breach': self.state.drawdown_breach,
            },
            'recent_transitions': self.transition_log[-5:],
        }
    
    def force_defensive(self, reason: str = "External override"):
        """Force DEFENSIVE operational regime (emergency stop)."""
        if not self.state:
            return
        
        if self.state.operational != OperationalRegime.DEFENSIVE:
            old_operational = self.state.operational
            self.state.operational = OperationalRegime.DEFENSIVE
            self.state.entries_allowed = False
            self.state.size_modifier = 0.0
            self.state.min_conviction = 0.99
            
            self.transition_log.append((
                time.time(),
                f"{old_operational.value}",
                "DEFENSIVE",
                reason
            ))
            logger.warning(f"[UnifiedRegimeEngine] Forced DEFENSIVE: {reason}")
    
    # ========================================================================
    # Private Methods: Metric Calculations
    # ========================================================================
    
    def _calculate_entropy(self, prices: np.ndarray) -> float:
        """Calculate permutation entropy of price returns."""
        if len(prices) < 10:
            return 0.5
        
        returns = np.diff(prices) / prices[:-1]
        
        if returns.std() < 1e-10:
            return 0.5
        
        returns_norm = (returns - returns.min()) / (returns.max() - returns.min() + 1e-10)
        
        n_bins = 5
        hist, _ = np.histogram(returns_norm, bins=n_bins, range=(0, 1))
        probs = hist / hist.sum()
        
        entropy = -np.sum(probs * np.log2(probs + 1e-10)) / np.log2(n_bins)
        return float(entropy)
    
    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """Calculate rolling volatility (std of returns)."""
        if len(prices) < 2:
            return 0.02
        
        returns = np.diff(prices) / prices[:-1]
        return float(returns.std())
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength (|mean| / std of returns)."""
        if len(prices) < 2:
            return 0.5
        
        returns = np.diff(prices) / prices[:-1]
        std = returns.std()
        
        if std < 1e-10:
            return 0.0
        
        return float(abs(returns.mean()) / std)
    
    # ========================================================================
    # Private Methods: Regime Computation
    # ========================================================================
    
    def _compute_behavioral_regime(
        self,
        avg_entropy: float,
        avg_volatility: float,
        avg_trend: float,
        prices: np.ndarray,
    ) -> Tuple[BehavioralRegime, float]:
        """
        Determine behavioral regime from market metrics.
        
        Returns:
            (BehavioralRegime, confidence) tuple
        """
        # High entropy = chaotic
        if avg_entropy > ENTROPY_HIGH_THRESHOLD:
            return BehavioralRegime.TRANSITION_CHAOS, 1.0 - avg_entropy
        
        # Low entropy = ordered
        if avg_entropy < ENTROPY_LOW_THRESHOLD:
            if avg_trend > TREND_STRONG_THRESHOLD:
                # Strong trend - check direction
                returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else np.array([0])
                if returns.mean() > 0:
                    return BehavioralRegime.BULL_MOMENTUM, 1.0 - avg_entropy
                else:
                    return BehavioralRegime.BEAR_DISTRIBUTION, 1.0 - avg_entropy
            else:
                return BehavioralRegime.LOW_VOL_MEAN_REVERT, 1.0 - avg_entropy
        
        # Medium entropy
        if avg_trend > 0.8 and avg_volatility > VOLATILITY_HIGH_THRESHOLD:
            return BehavioralRegime.HIGH_VOL_TRENDING, avg_trend
        elif avg_trend < TREND_WEAK_THRESHOLD:
            return BehavioralRegime.LOW_VOL_MEAN_REVERT, 1.0 - avg_trend
        
        # Default
        return BehavioralRegime.TRANSITION_CHAOS, 0.5
    
    def _compute_operational_regime(
        self,
        structure: str,
        entropy: float,
        liquidity_status: str,
        correlation_idx: float,
        drawdown_breach: bool,
    ) -> OperationalRegime:
        """
        Determine operational regime from macro conditions.
        
        This is the primary gating mechanism - DEFENSIVE overrides everything.
        """
        # DEFENSIVE conditions (highest priority)
        if drawdown_breach:
            return OperationalRegime.DEFENSIVE
        if liquidity_status == "critical":
            return OperationalRegime.DEFENSIVE
        if entropy > SMCE_DEFENSIVE_ENTROPY_MIN:
            return OperationalRegime.DEFENSIVE
        if correlation_idx > CORRELATION_DEFENSIVE_MIN and liquidity_status == "warning":
            return OperationalRegime.DEFENSIVE
        
        # EXPANSION conditions (requires bullish structure)
        if (
            entropy < SMCE_HARVEST_ENTROPY_MAX
            and structure in ("BULLISH", "SUPPORT")
            and liquidity_status == "healthy"
            and correlation_idx < CORRELATION_DEFENSIVE_MIN
        ):
            return OperationalRegime.EXPANSION
        
        # TRANSITION conditions
        if (
            SMCE_TRANSITION_ENTROPY_MIN <= entropy <= SMCE_TRANSITION_ENTROPY_MAX
            or correlation_idx >= CORRELATION_DEFENSIVE_MIN
            or liquidity_status == "warning"
        ):
            return OperationalRegime.TRANSITION
        
        # HARVEST (default safe state)
        if entropy < SMCE_HARVEST_ENTROPY_MAX and liquidity_status == "healthy":
            return OperationalRegime.HARVEST
        
        # Fallback
        return OperationalRegime.TRANSITION
    
    def _get_unified_config(
        self,
        behavioral: BehavioralRegime,
        operational: OperationalRegime,
    ) -> Dict:
        """Get unified config for regime combination."""
        key = (behavioral, operational)
        
        if key in UNIFIED_REGIME_CONFIG:
            return UNIFIED_REGIME_CONFIG[key].copy()
        
        # Fallback: try to find config with same operational but any behavioral
        for (b, o), config in UNIFIED_REGIME_CONFIG.items():
            if o == operational:
                return config.copy()
        
        # Ultimate fallback: safe defaults
        return {
            'entries_allowed': False,
            'size_modifier': 0.5,
            'max_leverage': 1.0,
            'min_conviction': 0.75,
            'trailing_stop_mult': 4.0,
            'profit_targets': {'rapid': 0.01, 'normal': 0.02, 'runner': 0.04},
            'cooldown_type': 'state',
        }
    
    def _is_significant_change(
        self,
        old_state: RegimeState,
        new_state: RegimeState,
    ) -> bool:
        """Check if regime change is significant (exceeds hysteresis)."""
        # Always significant if either regime dimension changed
        if old_state.behavioral != new_state.behavioral:
            return True
        if old_state.operational != new_state.operational:
            return True
        
        # Check confidence change
        confidence_change = abs(old_state.confidence - new_state.confidence)
        if confidence_change > 0.15:
            return True
        
        return False
    
    def _log_transition(self, new_state: RegimeState):
        """Log regime transition."""
        if not self.state:
            logger.info(f"[UnifiedRegimeEngine] Initial state: {new_state.behavioral.value} + {new_state.operational.value}")
            return
        
        old_key = f"{self.state.behavioral.value} + {self.state.operational.value}"
        new_key = f"{new_state.behavioral.value} + {new_state.operational.value}"
        
        reasons = []
        if self.state.behavioral != new_state.behavioral:
            reasons.append(f"behavioral: {self.state.behavioral.value} → {new_state.behavioral.value}")
        if self.state.operational != new_state.operational:
            reasons.append(f"operational: {self.state.operational.value} → {new_state.operational.value}")
        
        reason = "; ".join(reasons)
        self.transition_log.append((time.time(), old_key, new_key, reason))
        
        logger.info(f"[UnifiedRegimeEngine] REGIME CHANGE: {old_key} → {new_key} | {reason}")


# ============================================================================
# Global Instance
# ============================================================================

_unified_engine: Optional[UnifiedRegimeEngine] = None

def get_unified_regime_engine() -> UnifiedRegimeEngine:
    """Get or create global UnifiedRegimeEngine instance."""
    global _unified_engine
    if _unified_engine is None:
        _unified_engine = UnifiedRegimeEngine()
    return _unified_engine
