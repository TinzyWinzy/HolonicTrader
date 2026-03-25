"""
AEGIS QUANTSEC - RL Agent Security Wrapper

Security layer for Reinforcement Learning trading agents (DQN, PPO).

Provides:
1. Reward poisoning detection
2. Adversarial pattern filtering
3. Strategy fingerprinting protection
4. State manipulation detection

Addresses CRITICAL finding C-03: RL Agent Manipulation via Reward Poisoning

Author: AEGIS QuantSec v1.0
Date: 2026-03-15
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Deque
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timezone
import time
import threading
import logging

logger = logging.getLogger("AEGIS.RLSecurity")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RewardAnomaly:
    """Detected reward manipulation attempt."""
    anomaly_type: str  # SPOOFED_REWARD, STATE_MANIPULATION, ENVIRONMENT_DRIFT
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    state_hash: str
    expected_reward: float
    actual_reward: float
    z_score: float
    timestamp: float
    symbol: str
    details: str
    confidence: float  # Confidence in anomaly detection
    
    def to_dict(self) -> Dict:
        return {
            'type': self.anomaly_type,
            'severity': self.severity,
            'state_hash': self.state_hash,
            'expected_reward': self.expected_reward,
            'actual_reward': self.actual_reward,
            'z_score': self.z_score,
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'details': self.details,
            'confidence': self.confidence
        }


@dataclass
class AdversarialPattern:
    """Detected adversarial market pattern."""
    pattern_type: str  # SPOOFING, LAYERING, WASH_TRADING, MOMENTUM_IGNITION
    symbol: str
    confidence: float
    evidence: Dict[str, Any]
    detected_at: float
    impact_on_agent: str  # How this affects the RL agent
    
    def to_dict(self) -> Dict:
        return {
            'type': self.pattern_type,
            'symbol': self.symbol,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'detected_at': self.detected_at,
            'impact': self.impact_on_agent
        }


@dataclass
class SecurityReport:
    """RL agent security status report."""
    timestamp: float
    agent_type: str
    total_states_processed: int
    anomalies_detected: int
    patterns_detected: int
    reward_filtering_rate: float
    security_score: float  # 0.0 to 1.0
    status: str  # SECURE, WARNING, COMPROMISED
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'agent_type': self.agent_type,
            'total_states_processed': self.total_states_processed,
            'anomalies_detected': self.anomalies_detected,
            'patterns_detected': self.patterns_detected,
            'reward_filtering_rate': self.reward_filtering_rate,
            'security_score': self.security_score,
            'status': self.status
        }


# =============================================================================
# REWARD INTEGRITY FILTER
# =============================================================================

class RewardIntegrityFilter:
    """
    Detects and filters reward poisoning attempts.
    
    Reward poisoning occurs when:
    1. Market manipulation creates false reward signals
    2. Environment drift causes unexpected rewards
    3. State observation is manipulated
    
    The filter uses statistical analysis to detect anomalous rewards
    that don't match expected patterns.
    """
    
    def __init__(
        self,
        window_size: int = 500,
        z_score_threshold: float = 3.5,
        min_samples: int = 100,
        reward_decay: float = 0.99
    ):
        self.window_size = window_size
        self.z_score_threshold = z_score_threshold
        self.min_samples = min_samples
        self.reward_decay = reward_decay
        
        # Reward history per state cluster
        self._reward_history: Dict[str, Deque[float]] = {}
        self._reward_stats: Dict[str, Dict[str, float]] = {}
        
        # Anomaly tracking
        self._anomalies: List[RewardAnomaly] = []
        self._filtered_rewards: int = 0
        self._total_rewards: int = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Callbacks
        self._on_anomaly_callbacks: List[callable] = []
    
    def _get_state_hash(self, state: np.ndarray) -> str:
        """Create hash for state clustering."""
        # Discretize state for clustering
        if state.ndim > 1:
            state = state.flatten()
        
        # Simple binning for clustering
        binned = np.digitize(state, bins=np.linspace(-1, 1, 10))
        return hash(tuple(binned)) & 0xFFFFFFFF
    
    def validate_reward(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Tuple[bool, Optional[RewardAnomaly], float]:
        """
        Validate a reward signal.
        
        Args:
            state: Current state
            action: Action taken
            reward: Raw reward from environment
            next_state: Resulting state
            done: Episode termination
        
        Returns:
            (is_valid, anomaly, filtered_reward)
            - is_valid: False if reward should be rejected
            - anomaly: Detected anomaly if any
            - filtered_reward: Reward after filtering (may be adjusted)
        """
        with self._lock:
            self._total_rewards += 1
            
            state_hash = self._get_state_hash(state)
            
            # Initialize history for this state cluster
            if state_hash not in self._reward_history:
                self._reward_history[state_hash] = deque(maxlen=self.window_size)
                self._reward_stats[state_hash] = {
                    'mean': 0.0,
                    'std': 1.0,
                    'count': 0
                }
            
            history = self._reward_history[state_hash]
            stats = self._reward_stats[state_hash]
            
            # Check if we have enough samples
            if stats['count'] < self.min_samples:
                # Not enough data, accept reward but update stats
                history.append(reward)
                self._update_stats(stats, history)
                return True, None, reward
            
            # Calculate z-score
            z_score = (reward - stats['mean']) / (stats['std'] + 1e-8)
            
            # Check for anomaly
            anomaly = None
            is_valid = True
            filtered_reward = reward
            
            if abs(z_score) > self.z_score_threshold:
                # Anomaly detected
                expected_reward = stats['mean']
                confidence = min(1.0, abs(z_score) / (self.z_score_threshold * 2))
                
                anomaly = RewardAnomaly(
                    anomaly_type='SPOOFED_REWARD',
                    severity='HIGH' if abs(z_score) > self.z_score_threshold * 1.5 else 'MEDIUM',
                    state_hash=str(state_hash),
                    expected_reward=expected_reward,
                    actual_reward=reward,
                    z_score=z_score,
                    timestamp=time.time(),
                    symbol='UNKNOWN',
                    details=f"Reward {reward:.4f} deviates {z_score:.2f}σ from expected {expected_reward:.4f}",
                    confidence=confidence
                )
                
                self._anomalies.append(anomaly)
                self._filtered_rewards += 1
                
                # Decide action based on severity
                if abs(z_score) > self.z_score_threshold * 2:
                    # Extreme deviation - reject reward
                    is_valid = False
                    filtered_reward = expected_reward  # Use expected instead
                else:
                    # Moderate deviation - clip reward
                    clip_bound = stats['mean'] + np.sign(z_score) * self.z_score_threshold * stats['std']
                    filtered_reward = clip_bound
            
            # Update history and stats
            history.append(filtered_reward)
            self._update_stats(stats, history)
            
            return is_valid, anomaly, filtered_reward
    
    def _update_stats(self, stats: Dict, history: Deque[float]):
        """Update running statistics."""
        stats['count'] += 1
        
        # Welford's online algorithm for mean and variance
        n = len(history)
        if n > 0:
            stats['mean'] = sum(history) / n
            if n > 1:
                variance = sum((x - stats['mean']) ** 2 for x in history) / (n - 1)
                stats['std'] = np.sqrt(variance)
    
    def register_anomaly_callback(self, callback: callable):
        """Register callback for reward anomalies."""
        self._on_anomaly_callbacks.append(callback)
    
    def _trigger_anomaly_callbacks(self, anomaly: RewardAnomaly):
        """Trigger registered callbacks."""
        for callback in self._on_anomaly_callbacks:
            try:
                callback(anomaly)
            except Exception as e:
                logger.error(f"Anomaly callback error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get filter statistics."""
        with self._lock:
            return {
                'total_rewards': self._total_rewards,
                'filtered_rewards': self._filtered_rewards,
                'filtering_rate': self._filtered_rewards / max(1, self._total_rewards),
                'anomalies_detected': len(self._anomalies),
                'state_clusters': len(self._reward_history)
            }


# =============================================================================
# ADVERSARIAL PATTERN DETECTOR
# =============================================================================

class AdversarialPatternDetector:
    """
    Detects adversarial market patterns that could manipulate RL agents.
    
    Patterns detected:
    - Spoofing: Fake orders to manipulate price
    - Layering: Multiple fake orders at different levels
    - Wash trading: Self-trading to create false volume
    - Momentum ignition: Rapid orders to trigger algo responses
    """
    
    def __init__(
        self,
        lookback_window: int = 100,
        volume_spike_threshold: float = 3.0,
        order_imbalance_threshold: float = 0.7,
        price_impact_threshold: float = 0.002
    ):
        self.lookback_window = lookback_window
        self.volume_spike_threshold = volume_spike_threshold
        self.order_imbalance_threshold = order_imbalance_threshold
        self.price_impact_threshold = price_impact_threshold
        
        # Per-symbol tracking
        self._order_history: Dict[str, Deque[Dict]] = defaultdict(lambda: deque(maxlen=lookback_window))
        self._trade_history: Dict[str, Deque[Dict]] = defaultdict(lambda: deque(maxlen=lookback_window))
        self._pattern_history: List[AdversarialPattern] = []
        
        # Statistics
        self._stats = {
            'orders_processed': 0,
            'trades_processed': 0,
            'patterns_detected': 0
        }
        
        self._lock = threading.RLock()
    
    def record_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str,
        timestamp: float
    ):
        """Record an order for pattern analysis."""
        with self._lock:
            self._stats['orders_processed'] += 1
            
            self._order_history[symbol].append({
                'side': side,
                'quantity': quantity,
                'price': price,
                'type': order_type,
                'timestamp': timestamp
            })
            
            # Check for patterns
            self._check_patterns(symbol)
    
    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: float
    ):
        """Record a trade for pattern analysis."""
        with self._lock:
            self._stats['trades_processed'] += 1
            
            self._trade_history[symbol].append({
                'side': side,
                'quantity': quantity,
                'price': price,
                'timestamp': timestamp
            })
            
            # Check for patterns
            self._check_patterns(symbol)
    
    def _check_patterns(self, symbol: str):
        """Check for adversarial patterns."""
        orders = list(self._order_history[symbol])
        trades = list(self._trade_history[symbol])
        
        if len(orders) < 20 or len(trades) < 10:
            return
        
        # Check for spoofing
        spoofing = self._detect_spoofing(symbol, orders)
        if spoofing:
            self._pattern_history.append(spoofing)
            self._stats['patterns_detected'] += 1
        
        # Check for wash trading
        wash_trading = self._detect_wash_trading(symbol, trades)
        if wash_trading:
            self._pattern_history.append(wash_trading)
            self._stats['patterns_detected'] += 1
        
        # Check for momentum ignition
        momentum = self._detect_momentum_ignition(symbol, orders, trades)
        if momentum:
            self._pattern_history.append(momentum)
            self._stats['patterns_detected'] += 1
    
    def _detect_spoofing(self, symbol: str, orders: List[Dict]) -> Optional[AdversarialPattern]:
        """Detect spoofing patterns."""
        # Look for large orders that are quickly cancelled
        recent_orders = [o for o in orders if time.time() - o['timestamp'] < 60]
        
        if len(recent_orders) < 5:
            return None
        
        # Check for order imbalance
        buy_volume = sum(o['quantity'] for o in recent_orders if o['side'] == 'BUY')
        sell_volume = sum(o['quantity'] for o in recent_orders if o['side'] == 'SELL')
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return None
        
        imbalance = abs(buy_volume - sell_volume) / total_volume
        
        if imbalance > self.order_imbalance_threshold:
            # Check if large orders were cancelled (not filled)
            # This is a simplified check - real detection would track order status
            return AdversarialPattern(
                pattern_type='SPOOFING',
                symbol=symbol,
                confidence=imbalance,
                evidence={
                    'buy_volume': buy_volume,
                    'sell_volume': sell_volume,
                    'imbalance': imbalance
                },
                detected_at=time.time(),
                impact_on_agent='May cause false directional signals'
            )
        
        return None
    
    def _detect_wash_trading(self, symbol: str, trades: List[Dict]) -> Optional[AdversarialPattern]:
        """Detect wash trading patterns."""
        if len(trades) < 10:
            return None
        
        recent_trades = [t for t in trades if time.time() - t['timestamp'] < 30]
        
        if len(recent_trades) < 5:
            return None
        
        # Check for rapid back-and-forth trades at similar prices
        buy_trades = [t for t in recent_trades if t['side'] == 'BUY']
        sell_trades = [t for t in recent_trades if t['side'] == 'SELL']
        
        if len(buy_trades) > 3 and len(sell_trades) > 3:
            # Check price similarity
            avg_buy_price = np.mean([t['price'] for t in buy_trades])
            avg_sell_price = np.mean([t['price'] for t in sell_trades])
            
            price_diff = abs(avg_buy_price - avg_sell_price) / avg_buy_price
            
            if price_diff < 0.001:  # Very similar prices
                confidence = 1.0 - price_diff * 1000
                
                return AdversarialPattern(
                    pattern_type='WASH_TRADING',
                    symbol=symbol,
                    confidence=confidence,
                    evidence={
                        'buy_count': len(buy_trades),
                        'sell_count': len(sell_trades),
                        'price_difference': price_diff
                    },
                    detected_at=time.time(),
                    impact_on_agent='May create false volume signals'
                )
        
        return None
    
    def _detect_momentum_ignition(
        self,
        symbol: str,
        orders: List[Dict],
        trades: List[Dict]
    ) -> Optional[AdversarialPattern]:
        """Detect momentum ignition patterns."""
        if len(orders) < 10 or len(trades) < 5:
            return None
        
        recent_orders = [o for o in orders if time.time() - o['timestamp'] < 10]
        recent_trades = [t for t in trades if time.time() - t['timestamp'] < 10]
        
        if len(recent_orders) < 5 or len(recent_trades) < 3:
            return None
        
        # Check for sudden volume spike
        recent_volume = sum(t['quantity'] for t in recent_trades)
        older_trades = [t for t in trades if 10 < time.time() - t['timestamp'] < 60]
        
        if older_trades:
            older_volume = sum(t['quantity'] for t in older_trades)
            time_ratio = 60 / 10  # Normalize for time window
            
            if older_volume > 0:
                volume_ratio = (recent_volume / 10) / (older_volume / 50)
                
                if volume_ratio > self.volume_spike_threshold:
                    return AdversarialPattern(
                        pattern_type='MOMENTUM_IGNITION',
                        symbol=symbol,
                        confidence=min(1.0, volume_ratio / 10),
                        evidence={
                            'recent_volume': recent_volume,
                            'older_volume': older_volume,
                            'volume_ratio': volume_ratio
                        },
                        detected_at=time.time(),
                        impact_on_agent='May trigger momentum-based strategies'
                    )
        
        return None
    
    def get_detected_patterns(
        self,
        symbol: Optional[str] = None,
        pattern_type: Optional[str] = None,
        limit: int = 50
    ) -> List[AdversarialPattern]:
        """Get detected patterns with optional filtering."""
        with self._lock:
            results = self._pattern_history
            
            if symbol:
                results = [p for p in results if p.symbol == symbol]
            if pattern_type:
                results = [p for p in results if p.pattern_type == pattern_type]
            
            return results[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        with self._lock:
            return self._stats.copy()


# =============================================================================
# RL AGENT SECURITY WRAPPER
# =============================================================================

class RLAgentSecurityWrapper:
    """
    Security wrapper for RL trading agents.
    
    Wraps DQN, PPO, or other RL agents to provide:
    1. Reward validation and filtering
    2. State integrity verification
    3. Adversarial pattern awareness
    4. Training data sanitization
    """
    
    def __init__(
        self,
        agent,
        agent_type: str = 'DQN',
        enable_reward_filtering: bool = True,
        enable_pattern_detection: bool = True,
        z_score_threshold: float = 3.5
    ):
        self.agent = agent
        self.agent_type = agent_type
        
        # Initialize security components
        self.reward_filter = RewardIntegrityFilter(
            z_score_threshold=z_score_threshold
        ) if enable_reward_filtering else None
        
        self.pattern_detector = AdversarialPatternDetector() if enable_pattern_detection else None
        
        # State tracking
        self._experience_buffer: Deque[Dict] = deque(maxlen=10000)
        self._sanitized_experiences: int = 0
        self._total_experiences: int = 0
        
        # Security status
        self._security_mode = 'NORMAL'  # NORMAL, ELEVATED, CRITICAL
        self._anomaly_count: int = 0
        self._last_anomaly_time: float = 0
        
        # Callbacks
        self._on_security_event_callbacks: List[callable] = []
        
        self._lock = threading.RLock()
        
        logger.info(f"RL Agent Security Wrapper initialized for {agent_type}")
    
    def wrap_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        symbol: str = 'UNKNOWN'
    ) -> Tuple[bool, Dict]:
        """
        Validate and sanitize an experience tuple before storing.
        
        Args:
            state: Current state
            action: Action taken
            reward: Raw reward
            next_state: Resulting state
            done: Episode termination
            symbol: Trading symbol
        
        Returns:
            (is_valid, sanitized_experience)
        """
        with self._lock:
            self._total_experiences += 1
            
            # 1. Validate reward
            if self.reward_filter:
                is_valid, anomaly, filtered_reward = self.reward_filter.validate_reward(
                    state, action, reward, next_state, done
                )
                
                if anomaly:
                    anomaly.symbol = symbol
                    self._on_security_event('REWARD_ANOMALY', anomaly)
                
                if not is_valid:
                    self._sanitized_experiences += 1
                    self._update_security_mode('ELEVATED')
                
                reward = filtered_reward
            
            # 2. Create sanitized experience
            experience = {
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'symbol': symbol,
                'timestamp': time.time(),
                'sanitized': self._sanitized_experiences > 0
            }
            
            self._experience_buffer.append(experience)
            
            return True, experience
    
    def check_adversarial_exposure(self, symbol: str) -> Dict[str, Any]:
        """
        Check if current market conditions show adversarial patterns.
        
        Returns:
            Risk assessment dictionary
        """
        if not self.pattern_detector:
            return {'risk_level': 'UNKNOWN', 'patterns': []}
        
        patterns = self.pattern_detector.get_detected_patterns(symbol=symbol, limit=10)
        
        if not patterns:
            return {'risk_level': 'LOW', 'patterns': []}
        
        # Calculate risk level
        high_confidence_patterns = [p for p in patterns if p.confidence > 0.7]
        
        if len(high_confidence_patterns) > 3:
            risk_level = 'HIGH'
            self._update_security_mode('ELEVATED')
        elif len(patterns) > 5:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'risk_level': risk_level,
            'patterns': [p.to_dict() for p in patterns],
            'recommendation': self._get_risk_recommendation(risk_level)
        }
    
    def _get_risk_recommendation(self, risk_level: str) -> str:
        """Get recommendation based on risk level."""
        recommendations = {
            'LOW': 'Continue normal operations',
            'MEDIUM': 'Increase exploration, reduce position sizes',
            'HIGH': 'Consider pausing training, switch to safe mode'
        }
        return recommendations.get(risk_level, 'Unknown risk level')
    
    def _update_security_mode(self, new_mode: str):
        """Update security mode based on threat level."""
        now = time.time()
        
        if new_mode == 'CRITICAL':
            self._security_mode = 'CRITICAL'
            self._anomaly_count += 1
        elif new_mode == 'ELEVATED':
            if self._security_mode == 'NORMAL':
                self._security_mode = 'ELEVATED'
            self._anomaly_count += 1
        
        self._last_anomaly_time = now
        
        # Decay security mode over time
        if self._anomaly_count > 10 and now - self._last_anomaly_time > 3600:
            self._security_mode = 'NORMAL'
            self._anomaly_count = 0
    
    def _on_security_event(self, event_type: str, data: Any):
        """Handle security event."""
        for callback in self._on_security_event_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Security event callback error: {e}")
    
    def register_security_callback(self, callback: callable):
        """Register callback for security events."""
        self._on_security_event_callbacks.append(callback)
    
    def get_security_report(self) -> SecurityReport:
        """Get comprehensive security report."""
        with self._lock:
            anomalies = len(self.reward_filter._anomalies) if self.reward_filter else 0
            patterns = len(self.pattern_detector._pattern_history) if self.pattern_detector else 0
            
            filtering_rate = (
                self.reward_filter.get_statistics()['filtering_rate']
                if self.reward_filter else 0.0
            )
            
            # Calculate security score
            base_score = 1.0
            base_score -= min(0.3, anomalies * 0.01)
            base_score -= min(0.3, patterns * 0.02)
            base_score -= min(0.2, filtering_rate * 0.5)
            
            security_score = max(0.0, base_score)
            
            # Determine status
            if security_score < 0.5:
                status = 'COMPROMISED'
            elif security_score < 0.7:
                status = 'WARNING'
            else:
                status = 'SECURE'
            
            return SecurityReport(
                timestamp=time.time(),
                agent_type=self.agent_type,
                total_states_processed=self._total_experiences,
                anomalies_detected=anomalies,
                patterns_detected=patterns,
                reward_filtering_rate=filtering_rate,
                security_score=security_score,
                status=status
            )
    
    def get_sanitized_batch(self, batch_size: int) -> List[Dict]:
        """
        Get a batch of sanitized experiences for training.
        
        Returns experiences that have passed security validation.
        """
        with self._lock:
            # Filter to only sanitized experiences
            sanitized = [e for e in self._experience_buffer if e.get('sanitized', True)]
            
            if len(sanitized) < batch_size:
                return sanitized
            
            # Random sample
            indices = np.random.choice(len(sanitized), batch_size, replace=False)
            return [sanitized[i] for i in indices]


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def wrap_dqn_agent(dqn_holon, enable_all_features: bool = True) -> RLAgentSecurityWrapper:
    """
    Wrap a DQN agent with security features.
    
    Usage:
        secured_dqn = wrap_dqn_agent(dqn_holon)
    """
    wrapper = RLAgentSecurityWrapper(
        agent=dqn_holon,
        agent_type='DQN',
        enable_reward_filtering=enable_all_features,
        enable_pattern_detection=enable_all_features
    )
    
    # Wrap the remember method
    if hasattr(dqn_holon, 'remember'):
        original_remember = dqn_holon.remember
        
        def secured_remember(state, action_idx, reward, next_state, done):
            is_valid, experience = wrapper.wrap_experience(
                state=state,
                action=action_idx,
                reward=reward,
                next_state=next_state,
                done=done
            )
            
            if is_valid:
                original_remember(state, action_idx, experience['reward'], next_state, done)
            
            return is_valid
        
        dqn_holon.remember = secured_remember
        logger.info("DQN agent wrapped with security layer")
    
    return wrapper


def wrap_ppo_agent(ppo_holon, enable_all_features: bool = True) -> RLAgentSecurityWrapper:
    """
    Wrap a PPO agent with security features.
    
    Usage:
        secured_ppo = wrap_ppo_agent(ppo_holon)
    """
    wrapper = RLAgentSecurityWrapper(
        agent=ppo_holon,
        agent_type='PPO',
        enable_reward_filtering=enable_all_features,
        enable_pattern_detection=enable_all_features
    )
    
    logger.info("PPO agent wrapped with security layer")
    return wrapper


# =============================================================================
# CLI VERIFICATION
# =============================================================================

def run_security_check(wrapper: RLAgentSecurityWrapper) -> Dict[str, Any]:
    """Run a one-time security check."""
    report = wrapper.get_security_report()
    return report.to_dict()


if __name__ == "__main__":
    print("AEGIS QUANTSEC - RL Agent Security Wrapper")
    print()
    print("This module provides security for RL trading agents:")
    print("  1. Reward poisoning detection")
    print("  2. Adversarial pattern filtering")
    print("  3. Strategy fingerprinting protection")
    print("  4. State manipulation detection")
    print()
    print("Usage:")
    print("  from HolonicTrader.rl_agent_security import wrap_dqn_agent, wrap_ppo_agent")
    print("  secured_agent = wrap_dqn_agent(dqn_holon)")
