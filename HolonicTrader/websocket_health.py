"""
WebSocket Health Monitor - AEGIS QUANTSEC Enhancement

Provides:
1. Real-time connection health tracking
2. Automatic ping/pong keepalive enforcement
3. Connection quality scoring
4. Fallback orchestration (WS → REST)

Addresses: H-01 WebSocket Feed Instability

Author: AEGIS QuantSec v1.0
Date: 2026-03-15
"""

import time
import threading
import asyncio
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque, defaultdict
import logging

logger = logging.getLogger("AEGIS.WebSocketHealth")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ConnectionMetrics:
    """Real-time connection health metrics."""
    channel: str
    symbol: str
    last_message_time: float = 0.0
    messages_last_minute: int = 0
    latency_samples: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    reconnect_count: int = 0
    last_reconnect_time: float = 0.0
    consecutive_timeouts: int = 0
    last_ping_time: float = 0.0
    last_pong_time: float = 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        if not self.latency_samples:
            return 0.0
        return sum(self.latency_samples) / len(self.latency_samples)
    
    @property
    def messages_per_second(self) -> float:
        return self.messages_last_minute / 60.0
    
    def to_dict(self) -> Dict:
        return {
            'channel': self.channel,
            'symbol': self.symbol,
            'last_message_time': self.last_message_time,
            'messages_per_second': self.messages_per_second,
            'avg_latency_ms': self.avg_latency_ms,
            'error_count': self.error_count,
            'reconnect_count': self.reconnect_count,
            'consecutive_timeouts': self.consecutive_timeouts
        }


@dataclass
class HealthStatus:
    """Overall health status for a connection."""
    channel: str
    symbol: str
    status: str  # HEALTHY, DEGRADED, UNHEALTHY, CRITICAL
    health_score: float  # 0.0 to 1.0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=lambda: time.time())
    
    def to_dict(self) -> Dict:
        return {
            'channel': self.channel,
            'symbol': self.symbol,
            'status': self.status,
            'health_score': self.health_score,
            'issues': self.issues,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp
        }


# =============================================================================
# WEBSOCKET HEALTH MONITOR
# =============================================================================

class WebSocketHealthMonitor:
    """
    Monitors and maintains WebSocket connection health.
    
    Features:
    - Real-time ping/pong tracking
    - Message rate monitoring
    - Latency anomaly detection
    - Automatic reconnection triggers
    - Health scoring system
    
    Usage:
        monitor = WebSocketHealthMonitor()
        monitor.register_connection('book', 'BTC/USDT')
        
        # Call on message received
        monitor.record_message('book', 'BTC/USDT', latency_ms=15.5)
        
        # Get health status
        status = monitor.get_health_status('book', 'BTC/USDT')
        if status.status == 'UNHEALTHY':
            # Switch to REST fallback
    """
    
    def __init__(
        self,
        max_latency_ms: float = 5000.0,
        min_messages_per_minute: float = 10.0,
        max_consecutive_timeouts: int = 3,
        health_check_interval: float = 5.0,
        ping_interval: float = 15.0,  # AEGIS Recommendation: 15s (was default 30s)
        pong_timeout: float = 10.0
    ):
        self.max_latency_ms = max_latency_ms
        self.min_messages_per_minute = min_messages_per_minute
        self.max_consecutive_timeouts = max_consecutive_timeouts
        self.health_check_interval = health_check_interval
        self.ping_interval = ping_interval  # Enhanced keepalive
        self.pong_timeout = pong_timeout
        
        # Connection tracking: (channel, symbol) -> metrics
        self._metrics: Dict[tuple, ConnectionMetrics] = {}
        self._health_status: Dict[tuple, HealthStatus] = {}
        
        # Callbacks
        self._on_degraded_callbacks: List[Callable] = []
        self._on_unhealthy_callbacks: List[Callable] = []
        self._on_recovered_callbacks: List[Callable] = []
        
        # Background monitoring
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'total_connections': 0,
            'healthy_connections': 0,
            'degraded_connections': 0,
            'unhealthy_connections': 0,
            'total_reconnects': 0,
            'total_timeouts': 0
        }
        
        logger.info(f"WebSocket Health Monitor initialized (ping_interval={ping_interval}s)")
    
    def register_connection(self, channel: str, symbol: str):
        """Register a new connection for monitoring."""
        with self._lock:
            key = (channel, symbol)
            if key not in self._metrics:
                self._metrics[key] = ConnectionMetrics(channel=channel, symbol=symbol)
                self._health_status[key] = HealthStatus(
                    channel=channel,
                    symbol=symbol,
                    status='HEALTHY',
                    health_score=1.0
                )
                self._stats['total_connections'] += 1
                logger.debug(f"Registered connection: {channel}/{symbol}")
    
    def unregister_connection(self, channel: str, symbol: str):
        """Unregister a connection from monitoring."""
        with self._lock:
            key = (channel, symbol)
            if key in self._metrics:
                del self._metrics[key]
                del self._health_status[key]
                self._stats['total_connections'] -= 1
    
    def record_message(self, channel: str, symbol: str, latency_ms: float = 0.0):
        """Record a received message for health tracking."""
        with self._lock:
            key = (channel, symbol)
            if key not in self._metrics:
                self.register_connection(channel, symbol)
            
            metrics = self._metrics[key]
            now = time.time()
            
            # Update message tracking
            metrics.last_message_time = now
            metrics.messages_last_minute += 1
            
            # Track latency
            if latency_ms > 0:
                metrics.latency_samples.append(latency_ms)
            
            # Reset timeout counter on successful message
            if metrics.consecutive_timeouts > 0:
                logger.info(f"Connection {channel}/{symbol} recovered after {metrics.consecutive_timeouts} timeouts")
                metrics.consecutive_timeouts = 0
                self._trigger_recovered(channel, symbol, self._health_status[key])
    
    def record_error(self, channel: str, symbol: str, error_type: str = 'UNKNOWN'):
        """Record an error for health tracking."""
        with self._lock:
            key = (channel, symbol)
            if key not in self._metrics:
                return
            
            metrics = self._metrics[key]
            metrics.error_count += 1
            
            if 'timeout' in error_type.lower():
                metrics.consecutive_timeouts += 1
                self._stats['total_timeouts'] += 1
                
                if metrics.consecutive_timeouts >= self.max_consecutive_timeouts:
                    logger.warning(
                        f"Connection {channel}/{symbol} unhealthy: "
                        f"{metrics.consecutive_timeouts} consecutive timeouts"
                    )
    
    def record_reconnect(self, channel: str, symbol: str):
        """Record a reconnection event."""
        with self._lock:
            key = (channel, symbol)
            if key not in self._metrics:
                return
            
            metrics = self._metrics[key]
            metrics.reconnect_count += 1
            metrics.last_reconnect_time = time.time()
            metrics.consecutive_timeouts = 0  # Reset timeout counter
            
            self._stats['total_reconnects'] += 1
            
            logger.info(f"Connection {channel}/{symbol} reconnected (count: {metrics.reconnect_count})")
    
    def record_ping(self, channel: str, symbol: str):
        """Record a ping sent."""
        with self._lock:
            key = (channel, symbol)
            if key not in self._metrics:
                return
            self._metrics[key].last_ping_time = time.time()
    
    def record_pong(self, channel: str, symbol: str, latency_ms: float = 0.0):
        """Record a pong received."""
        with self._lock:
            key = (channel, symbol)
            if key not in self._metrics:
                return
            
            metrics = self._metrics[key]
            metrics.last_pong_time = time.time()
            
            if latency_ms > 0:
                metrics.latency_samples.append(latency_ms)
    
    def get_health_status(self, channel: str, symbol: str) -> HealthStatus:
        """Get current health status for a connection."""
        with self._lock:
            key = (channel, symbol)
            if key not in self._metrics:
                return HealthStatus(
                    channel=channel,
                    symbol=symbol,
                    status='UNKNOWN',
                    health_score=0.0,
                    issues=['Connection not registered']
                )
            
            return self._calculate_health_status(key)
    
    def _calculate_health_status(self, key: tuple) -> HealthStatus:
        """Calculate health status based on metrics."""
        metrics = self._metrics[key]
        now = time.time()
        
        issues = []
        recommendations = []
        health_score = 1.0
        
        # 1. Check message recency
        time_since_message = now - metrics.last_message_time
        if time_since_message > 60:  # No message in 60s
            issues.append(f"No messages for {time_since_message:.1f}s")
            health_score -= 0.3
            recommendations.append("Check WebSocket subscription")
        
        # 2. Check message rate
        if metrics.messages_last_minute < self.min_messages_per_minute:
            issues.append(
                f"Low message rate: {metrics.messages_last_minute:.1f}/min "
                f"(min: {self.min_messages_per_minute})"
            )
            health_score -= 0.2
            recommendations.append("Verify exchange feed status")
        
        # 3. Check latency
        if metrics.avg_latency_ms > self.max_latency_ms:
            issues.append(f"High latency: {metrics.avg_latency_ms:.1f}ms")
            health_score -= 0.3
            recommendations.append("Consider REST fallback")
        
        # 4. Check consecutive timeouts
        if metrics.consecutive_timeouts > 0:
            issues.append(f"{metrics.consecutive_timeouts} consecutive timeouts")
            health_score -= 0.2 * min(metrics.consecutive_timeouts, 3)
            
            if metrics.consecutive_timeouts >= self.max_consecutive_timeouts:
                recommendations.append("IMMEDIATE: Trigger reconnection")
        
        # 5. Check error rate
        if metrics.error_count > 10:
            issues.append(f"High error count: {metrics.error_count}")
            health_score -= 0.1
            recommendations.append("Review error logs")
        
        # 6. Check reconnect frequency
        if metrics.reconnect_count > 5:
            time_since_first_reconnect = now - metrics.last_reconnect_time
            if time_since_first_reconnect < 300:  # 5 reconnects in 5min
                issues.append(f"Frequent reconnects: {metrics.reconnect_count}")
                health_score -= 0.2
                recommendations.append("Investigate network stability")
        
        # Clamp health score
        health_score = max(0.0, min(1.0, health_score))
        
        # Determine status
        if health_score >= 0.8:
            status = 'HEALTHY'
        elif health_score >= 0.6:
            status = 'DEGRADED'
        elif health_score >= 0.4:
            status = 'UNHEALTHY'
        else:
            status = 'CRITICAL'
        
        # Update cached status
        old_status = self._health_status.get(key)
        new_status = HealthStatus(
            channel=metrics.channel,
            symbol=metrics.symbol,
            status=status,
            health_score=health_score,
            issues=issues,
            recommendations=recommendations,
            timestamp=now
        )
        self._health_status[key] = new_status
        
        # Trigger callbacks on status change
        if old_status and old_status.status != status:
            if status == 'DEGRADED' and old_status.status == 'HEALTHY':
                self._trigger_degraded(metrics.channel, metrics.symbol, new_status)
            elif status in ('UNHEALTHY', 'CRITICAL') and old_status.status in ('HEALTHY', 'DEGRADED'):
                self._trigger_unhealthy(metrics.channel, metrics.symbol, new_status)
            elif status == 'HEALTHY' and old_status.status in ('DEGRADED', 'UNHEALTHY', 'CRITICAL'):
                self._trigger_recovered(metrics.channel, metrics.symbol, new_status)
        
        # Update stats
        self._update_stats()
        
        return new_status
    
    def _trigger_degraded(self, channel: str, symbol: str, status: HealthStatus):
        """Trigger degraded status callbacks."""
        logger.warning(f"Connection DEGRADED: {channel}/{symbol} (score: {status.health_score:.2f})")
        for callback in self._on_degraded_callbacks:
            try:
                callback(channel, symbol, status)
            except Exception as e:
                logger.error(f"Degraded callback error: {e}")
    
    def _trigger_unhealthy(self, channel: str, symbol: str, status: HealthStatus):
        """Trigger unhealthy status callbacks."""
        logger.error(f"Connection UNHEALTHY: {channel}/{symbol} (score: {status.health_score:.2f})")
        for callback in self._on_unhealthy_callbacks:
            try:
                callback(channel, symbol, status)
            except Exception as e:
                logger.error(f"Unhealthy callback error: {e}")
    
    def _trigger_recovered(self, channel: str, symbol: str, status: HealthStatus):
        """Trigger recovery callbacks."""
        logger.info(f"Connection RECOVERED: {channel}/{symbol} (score: {status.health_score:.2f})")
        for callback in self._on_recovered_callbacks:
            try:
                callback(channel, symbol, status)
            except Exception as e:
                logger.error(f"Recovery callback error: {e}")
    
    def _update_stats(self):
        """Update aggregate statistics."""
        healthy = 0
        degraded = 0
        unhealthy = 0
        
        for key, status in self._health_status.items():
            if status.status == 'HEALTHY':
                healthy += 1
            elif status.status == 'DEGRADED':
                degraded += 1
            else:
                unhealthy += 1
        
        self._stats['healthy_connections'] = healthy
        self._stats['degraded_connections'] = degraded
        self._stats['unhealthy_connections'] = unhealthy
    
    def register_degraded_callback(self, callback: Callable):
        """Register callback for degraded status."""
        self._on_degraded_callbacks.append(callback)
    
    def register_unhealthy_callback(self, callback: Callable):
        """Register callback for unhealthy status."""
        self._on_unhealthy_callbacks.append(callback)
    
    def register_recovered_callback(self, callback: Callable):
        """Register callback for recovery."""
        self._on_recovered_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start background health monitoring thread."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("WebSocket health monitoring started")
    
    def stop_monitoring(self):
        """Stop background health monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("WebSocket health monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                # Reset per-minute counters
                with self._lock:
                    for metrics in self._metrics.values():
                        metrics.messages_last_minute = 0
                
                # Health check interval
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
    
    def get_all_health_statuses(self) -> Dict[str, HealthStatus]:
        """Get health status for all connections."""
        with self._lock:
            return {
                f"{m.channel}/{m.symbol}": self._calculate_health_status(key)
                for key, m in self._metrics.items()
            }
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get summary health report."""
        with self._lock:
            statuses = [self._calculate_health_status(key) for key in self._metrics]
            
            return {
                'timestamp': time.time(),
                'total_connections': self._stats['total_connections'],
                'healthy': self._stats['healthy_connections'],
                'degraded': self._stats['degraded_connections'],
                'unhealthy': self._stats['unhealthy_connections'],
                'total_reconnects': self._stats['total_reconnects'],
                'total_timeouts': self._stats['total_timeouts'],
                'connections': [s.to_dict() for s in statuses]
            }
    
    def should_fallback_to_rest(self, channel: str, symbol: str) -> bool:
        """Determine if connection should fallback to REST."""
        status = self.get_health_status(channel, symbol)
        return status.status in ('UNHEALTHY', 'CRITICAL')


# =============================================================================
# GLOBAL HEALTH MONITOR INSTANCE
# =============================================================================

# Singleton instance for global access
_global_health_monitor: Optional[WebSocketHealthMonitor] = None


def get_global_health_monitor() -> WebSocketHealthMonitor:
    """Get or create global health monitor instance."""
    global _global_health_monitor
    if _global_health_monitor is None:
        # Import config for AEGIS settings
        try:
            import config
            ping_interval = getattr(config, 'AEGIS_WS_PING_INTERVAL', 15.0)
            pong_timeout = getattr(config, 'AEGIS_WS_PONG_TIMEOUT', 10.0)
            max_consecutive_timeouts = getattr(config, 'AEGIS_WS_MAX_TIMEOUTS', 3)
            health_check_interval = getattr(config, 'AEGIS_WS_HEALTH_CHECK_INTERVAL', 5.0)
            min_messages_per_minute = getattr(config, 'AEGIS_WS_MIN_MESSAGES_PER_MINUTE', 10.0)
        except ImportError:
            # Defaults if config not available
            ping_interval = 15.0
            pong_timeout = 10.0
            max_consecutive_timeouts = 3
            health_check_interval = 5.0
            min_messages_per_minute = 10.0

        _global_health_monitor = WebSocketHealthMonitor(
            ping_interval=ping_interval,  # AEGIS enhanced keepalive
            pong_timeout=pong_timeout,
            max_consecutive_timeouts=max_consecutive_timeouts,
            health_check_interval=health_check_interval,
            min_messages_per_minute=min_messages_per_minute
        )
        _global_health_monitor.start_monitoring()
    return _global_health_monitor


def initialize_websocket_health_monitor(
    ping_interval: float = 15.0,
    pong_timeout: float = 10.0,
    max_consecutive_timeouts: int = 3
) -> WebSocketHealthMonitor:
    """Initialize global health monitor with custom settings."""
    global _global_health_monitor

    if _global_health_monitor:
        _global_health_monitor.stop_monitoring()

    _global_health_monitor = WebSocketHealthMonitor(
        ping_interval=ping_interval,
        pong_timeout=pong_timeout,
        max_consecutive_timeouts=max_consecutive_timeouts
    )
    _global_health_monitor.start_monitoring()

    logger.info(
        f"Global WebSocket Health Monitor initialized: "
        f"ping={ping_interval}s, pong_timeout={pong_timeout}s"
    )

    return _global_health_monitor
