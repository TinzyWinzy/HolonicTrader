"""
AEGIS QUANTSEC - Dedicated Websocket Integrity Monitor

Standalone module for websocket feed integrity monitoring.

Provides:
1. Sequence number validation per channel
2. Gap detection and automatic recovery
3. Message reordering buffer
4. Latency monitoring
5. Connection health tracking

Addresses CRITICAL finding C-02: Timing Oracle Vulnerability

Author: AEGIS QuantSec v1.0
Date: 2026-03-15
"""

import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict, deque
import logging

logger = logging.getLogger("AEGIS.WebsocketIntegrity")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class WebsocketEvent:
    """A websocket message with integrity metadata."""
    channel: str
    symbol: str
    event_type: str
    sequence_num: int
    data: Dict[str, Any]
    timestamp_ns: int
    received_ns: int = field(default_factory=lambda: time.time_ns())
    
    @property
    def latency_ms(self) -> float:
        """Message latency in milliseconds."""
        return (self.received_ns - self.timestamp_ns) / 1e6
    
    def to_dict(self) -> Dict:
        return {
            'channel': self.channel,
            'symbol': self.symbol,
            'event_type': self.event_type,
            'sequence_num': self.sequence_num,
            'timestamp_ns': self.timestamp_ns,
            'received_ns': self.received_ns,
            'latency_ms': self.latency_ms
        }


@dataclass
class SequenceGap:
    """A detected gap in sequence numbers."""
    channel: str
    symbol: str
    expected_seq: int
    actual_seq: int
    missing_count: int
    detected_at: float
    recovered: bool = False
    recovery_method: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'channel': self.channel,
            'symbol': self.symbol,
            'expected_seq': self.expected_seq,
            'actual_seq': self.actual_seq,
            'missing_count': self.missing_count,
            'recovered': self.recovered,
            'recovery_method': self.recovery_method
        }


@dataclass
class ConnectionHealth:
    """Health status of a websocket connection."""
    channel: str
    symbol: str
    last_message_time: float
    messages_per_second: float
    error_count: int
    reconnect_count: int
    status: str  # HEALTHY, DEGRADED, UNHEALTHY
    
    def to_dict(self) -> Dict:
        return {
            'channel': self.channel,
            'symbol': self.symbol,
            'last_message_time': self.last_message_time,
            'messages_per_second': self.messages_per_second,
            'error_count': self.error_count,
            'reconnect_count': self.reconnect_count,
            'status': self.status
        }


@dataclass
class IntegrityReport:
    """Comprehensive websocket integrity report."""
    timestamp: float
    channels_monitored: int
    total_events_processed: int
    sequence_gaps: List[SequenceGap]
    latency_stats: Dict[str, float]
    connection_health: List[ConnectionHealth]
    health_status: str  # HEALTHY, WARNING, DEGRADED, CRITICAL
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'channels_monitored': self.channels_monitored,
            'total_events_processed': self.total_events_processed,
            'sequence_gaps': [g.to_dict() for g in self.sequence_gaps],
            'latency_stats': self.latency_stats,
            'connection_health': [h.to_dict() for h in self.connection_health],
            'health_status': self.health_status
        }


# =============================================================================
# WEBSOCKET INTEGRITY MONITOR
# =============================================================================

class WebsocketIntegrityMonitor:
    """
    Monitors websocket feed integrity for quantitative trading.
    
    Features:
    - Sequence number tracking per channel/symbol
    - Gap detection with automatic recovery
    - Message buffering and reordering
    - Latency monitoring
    - Connection health tracking
    
    Usage:
        monitor = WebsocketIntegrityMonitor()
        monitor.register_channel('book', 'BTC/USDT', initial_sequence=0)
        
        # Process incoming messages
        is_valid, event = monitor.process_message(
            channel='book',
            symbol='BTC/USDT',
            data={'price': 50000},
            sequence_num=12345
        )
        
        # Get integrity report
        report = monitor.get_integrity_report()
        print(f"Health: {report.health_status}")
    """
    
    def __init__(
        self,
        max_latency_ms: float = 5000.0,
        max_timestamp_drift_ms: float = 1000.0,
        buffer_size: int = 1000,
        gap_recovery_window_ms: float = 5000.0,
        health_check_interval: float = 10.0
    ):
        self.max_latency_ms = max_latency_ms
        self.max_timestamp_drift_ms = max_timestamp_drift_ms
        self.buffer_size = buffer_size
        self.gap_recovery_window_ms = gap_recovery_window_ms
        self.health_check_interval = health_check_interval
        
        # Sequence tracking: channel -> symbol -> expected_seq
        self._expected_sequences: Dict[str, Dict[str, int]] = defaultdict(dict)
        
        # Message buffers: channel -> symbol -> deque
        self._sequence_buffers: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=buffer_size)))
        
        # Gap tracking
        self._sequence_gaps: List[SequenceGap] = []
        self._pending_gaps: Dict[str, List[int]] = defaultdict(list)
        
        # Connection health tracking
        self._message_times: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=1000)))
        self._error_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._reconnect_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Statistics
        self._stats = {
            'total_events': 0,
            'valid_events': 0,
            'rejected_events': 0,
            'gaps_detected': 0,
            'gaps_recovered': 0,
            'duplicates': 0,
            'out_of_order': 0,
            'latency_violations': 0
        }
        
        # Callbacks
        self._on_gap_callbacks: List[Callable] = []
        self._on_health_alert_callbacks: List[Callable] = []
        
        # State
        self._running = False
        self._health_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        logger.info("Websocket Integrity Monitor initialized")
    
    def register_channel(
        self,
        channel: str,
        symbol: str,
        initial_sequence: int = 0
    ):
        """
        Register a channel/symbol for monitoring.
        
        Args:
            channel: Websocket channel (e.g., 'book', 'trades')
            symbol: Trading symbol (e.g., 'BTC/USDT')
            initial_sequence: Starting sequence number
        """
        with self._lock:
            self._expected_sequences[channel][symbol] = initial_sequence
            self._message_times[channel][symbol] = deque(maxlen=1000)
            logger.info(f"Registered channel {channel}/{symbol} (seq={initial_sequence})")
    
    def unregister_channel(self, channel: str, symbol: str):
        """Unregister a channel/symbol from monitoring."""
        with self._lock:
            if channel in self._expected_sequences and symbol in self._expected_sequences[channel]:
                del self._expected_sequences[channel][symbol]
                logger.info(f"Unregistered channel {channel}/{symbol}")
    
    def process_message(
        self,
        channel: str,
        symbol: str,
        data: Dict[str, Any],
        sequence_num: Optional[int] = None,
        timestamp_ns: Optional[int] = None,
        event_type: str = 'MARKET_DATA'
    ) -> Tuple[bool, Optional[WebsocketEvent]]:
        """
        Process an incoming websocket message.
        
        Args:
            channel: Websocket channel
            symbol: Trading symbol
            data: Message data
            sequence_num: Sequence number from exchange
            timestamp_ns: Message timestamp (nanoseconds)
            event_type: Type of event
        
        Returns:
            (is_valid, event)
            - is_valid: False if message should be rejected
            - event: Processed event (may be None if buffered)
        """
        received_ns = time.time_ns()
        timestamp_ns = timestamp_ns or received_ns
        
        event = WebsocketEvent(
            channel=channel,
            symbol=symbol,
            event_type=event_type,
            sequence_num=sequence_num or 0,
            data=data,
            timestamp_ns=timestamp_ns,
            received_ns=received_ns
        )
        
        with self._lock:
            self._stats['total_events'] += 1
            
            # Track message timing for health
            self._message_times[channel][symbol].append(time.time())
            
            # 1. Validate sequence number (if provided)
            if sequence_num is not None:
                seq_valid = self._validate_sequence(channel, symbol, sequence_num, event)
                if not seq_valid:
                    self._stats['rejected_events'] += 1
                    return False, None
            
            # 2. Check latency
            if event.latency_ms > self.max_latency_ms:
                self._stats['latency_violations'] += 1
                logger.warning(
                    f"High latency: {channel}/{symbol} "
                    f"latency={event.latency_ms:.2f}ms"
                )
            
            # 3. Buffer and reorder (only if we have gaps)
            if self._pending_gaps.get(f"{channel}:{symbol}"):
                buffered_event = self._buffer_and_reorder(channel, symbol, event)
            else:
                buffered_event = event

            if buffered_event:
                self._stats['valid_events'] += 1

            return buffered_event is not None, buffered_event
    
    def _validate_sequence(
        self,
        channel: str,
        symbol: str,
        sequence_num: int,
        event: WebsocketEvent
    ) -> bool:
        """Validate sequence number and detect gaps."""
        expected = self._expected_sequences[channel].get(symbol, 0)

        if sequence_num < expected:
            # Duplicate or old message
            self._stats['duplicates'] += 1
            logger.debug(
                f"Duplicate message: seq {sequence_num} < expected {expected}"
            )
            return False

        if sequence_num > expected:
            # Gap detected!
            missing_count = sequence_num - expected
            self._stats['gaps_detected'] += 1

            gap = SequenceGap(
                channel=channel,
                symbol=symbol,
                expected_seq=expected,
                actual_seq=sequence_num,
                missing_count=missing_count,
                detected_at=time.time()
            )
            self._sequence_gaps.append(gap)
            self._pending_gaps[f"{channel}:{symbol}"].extend(
                range(expected, sequence_num)
            )

            logger.warning(
                f"Sequence gap on {channel}/{symbol}: "
                f"expected {expected}, got {sequence_num}, "
                f"missing {missing_count} messages"
            )

            # Trigger gap callbacks
            self._trigger_gap_callbacks(gap)

            # Buffer this event and wait for missing messages
            self._sequence_buffers[channel][symbol].append(event)

            return False  # Don't process yet

        # Sequence is correct - update expected for next message
        self._expected_sequences[channel][symbol] = sequence_num + 1

        # Check if this fills pending gaps
        self._fill_pending_gaps(channel, symbol, sequence_num)

        return True
    
    def _buffer_and_reorder(
        self,
        channel: str,
        symbol: str,
        event: WebsocketEvent
    ) -> Optional[WebsocketEvent]:
        """Buffer events and release in order."""
        buffer = self._sequence_buffers[channel][symbol]
        buffer.append(event)
        
        # Sort by sequence number
        sorted_buffer = sorted(buffer, key=lambda e: e.sequence_num)
        self._sequence_buffers[channel][symbol] = deque(sorted_buffer, maxlen=self.buffer_size)
        
        # Release events in order
        return self._release_buffered_events(channel, symbol)
    
    def _release_buffered_events(
        self,
        channel: str,
        symbol: str
    ) -> Optional[WebsocketEvent]:
        """Release oldest buffered event if it's in order."""
        buffer = self._sequence_buffers[channel][symbol]
        
        if not buffer:
            return None
        
        expected = self._expected_sequences[channel].get(symbol, 0)
        oldest = buffer[0]
        
        # Check if oldest event has correct sequence
        if oldest.sequence_num == expected:
            buffer.popleft()
            self._expected_sequences[channel][symbol] = expected + 1
            return oldest
        
        return None
    
    def _fill_pending_gaps(
        self,
        channel: str,
        symbol: str,
        received_seq: int
    ):
        """Mark gaps as filled when we receive missing sequences."""
        key = f"{channel}:{symbol}"
        
        if key in self._pending_gaps:
            if received_seq - 1 in self._pending_gaps[key]:
                self._pending_gaps[key].remove(received_seq - 1)
                
                # Check if all gaps are filled
                if not self._pending_gaps[key]:
                    # Mark gaps as recovered
                    for gap in self._sequence_gaps:
                        if (gap.channel == channel and 
                            gap.symbol == symbol and 
                            not gap.recovered):
                            gap.recovered = True
                            gap.recovery_method = 'LATE_MESSAGE_RECEIVED'
                            self._stats['gaps_recovered'] += 1
    
    def handle_reconnect(
        self,
        channel: str,
        symbol: str,
        new_sequence: int
    ):
        """
        Handle websocket reconnect - reset sequence tracking.
        
        Args:
            channel: Websocket channel
            symbol: Trading symbol
            new_sequence: Sequence number after reconnect
        """
        with self._lock:
            logger.info(
                f"Reconnect on {channel}/{symbol}, "
                f"resetting to seq {new_sequence}"
            )
            
            # Clear buffer
            self._sequence_buffers[channel][symbol].clear()
            
            # Update expected sequence
            self._expected_sequences[channel][symbol] = new_sequence
            
            # Mark pending gaps as unrecoverable
            for gap in self._sequence_gaps:
                if (gap.channel == channel and 
                    gap.symbol == symbol and 
                    not gap.recovered):
                    gap.recovery_method = 'RECONNECT_RESET'
            
            # Increment reconnect count
            self._reconnect_counts[channel][symbol] += 1
    
    def record_error(self, channel: str, symbol: str, error: str):
        """Record a websocket error."""
        with self._lock:
            self._error_counts[channel][symbol] += 1
            logger.error(f"WS error on {channel}/{symbol}: {error}")
    
    def _trigger_gap_callbacks(self, gap: SequenceGap):
        """Trigger registered gap callbacks."""
        for callback in self._on_gap_callbacks:
            try:
                callback(gap)
            except Exception as e:
                logger.error(f"Gap callback error: {e}")
    
    def register_gap_callback(self, callback: Callable):
        """Register callback for sequence gaps."""
        self._on_gap_callbacks.append(callback)
    
    def register_health_callback(self, callback: Callable):
        """Register callback for health alerts."""
        self._on_health_alert_callbacks.append(callback)
    
    def start_health_monitoring(self):
        """Start background health monitoring thread."""
        if self._running:
            return
        
        self._running = True
        self._health_thread = threading.Thread(
            target=self._health_monitoring_loop,
            daemon=True
        )
        self._health_thread.start()
        logger.info("Health monitoring started")
    
    def stop_health_monitoring(self):
        """Stop background health monitoring."""
        self._running = False
        if self._health_thread:
            self._health_thread.join(timeout=2.0)
        logger.info("Health monitoring stopped")
    
    def _health_monitoring_loop(self):
        """Background health monitoring loop."""
        while self._running:
            time.sleep(self.health_check_interval)
            
            # Check for stale connections
            now = time.time()
            for channel in self._expected_sequences:
                for symbol in self._expected_sequences[channel]:
                    times = self._message_times[channel][symbol]
                    if times:
                        last_msg = max(times)
                        if now - last_msg > self.health_check_interval * 3:
                            logger.warning(
                                f"Stale connection: {channel}/{symbol} "
                                f"(last msg {now - last_msg:.1f}s ago)"
                            )
    
    def get_integrity_report(self) -> IntegrityReport:
        """Get comprehensive integrity report."""
        with self._lock:
            # Calculate latency stats
            latencies = []
            for channel_buffer in self._sequence_buffers.values():
                for symbol_buffer in channel_buffer.values():
                    for event in symbol_buffer:
                        latencies.append(event.latency_ms)
            
            latency_stats = {
                'avg_ms': sum(latencies) / len(latencies) if latencies else 0,
                'max_ms': max(latencies) if latencies else 0,
                'min_ms': min(latencies) if latencies else 0,
                'p99_ms': sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 100 else 0
            }
            
            # Calculate connection health
            connection_health = []
            now = time.time()
            for channel in self._expected_sequences:
                for symbol in self._expected_sequences[channel]:
                    times = list(self._message_times[channel][symbol])
                    
                    # Calculate messages per second
                    if len(times) > 1:
                        time_span = max(times) - min(times)
                        mps = len(times) / time_span if time_span > 0 else 0
                    else:
                        mps = 0
                    
                    # Determine status
                    last_msg = max(times) if times else 0
                    errors = self._error_counts[channel][symbol]
                    reconnects = self._reconnect_counts[channel][symbol]
                    
                    if now - last_msg > 60:
                        status = 'UNHEALTHY'
                    elif errors > 10 or reconnects > 3:
                        status = 'DEGRADED'
                    elif errors > 0:
                        status = 'WARNING'
                    else:
                        status = 'HEALTHY'
                    
                    connection_health.append(ConnectionHealth(
                        channel=channel,
                        symbol=symbol,
                        last_message_time=last_msg,
                        messages_per_second=mps,
                        error_count=errors,
                        reconnect_count=reconnects,
                        status=status
                    ))
            
            # Determine overall health status
            unrecovered_gaps = sum(1 for g in self._sequence_gaps if not g.recovered)
            unhealthy_connections = sum(1 for h in connection_health if h.status == 'UNHEALTHY')
            
            if unrecovered_gaps > 5 or unhealthy_connections > 0:
                health_status = 'CRITICAL'
            elif unrecovered_gaps > 0:
                health_status = 'DEGRADED'
            elif self._stats['latency_violations'] > 100:
                health_status = 'WARNING'
            else:
                health_status = 'HEALTHY'
            
            return IntegrityReport(
                timestamp=time.time(),
                channels_monitored=len(self._expected_sequences),
                total_events_processed=self._stats['total_events'],
                sequence_gaps=list(self._sequence_gaps[-50:]),
                latency_stats=latency_stats,
                connection_health=connection_health,
                health_status=health_status
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        with self._lock:
            return self._stats.copy()
    
    def get_channel_status(self, channel: str, symbol: str) -> Dict[str, Any]:
        """Get status for a specific channel/symbol."""
        with self._lock:
            expected_seq = self._expected_sequences.get(channel, {}).get(symbol, 0)
            buffer_size = len(self._sequence_buffers.get(channel, {}).get(symbol, []))
            errors = self._error_counts.get(channel, {}).get(symbol, 0)
            reconnects = self._reconnect_counts.get(channel, {}).get(symbol, 0)
            
            return {
                'channel': channel,
                'symbol': symbol,
                'expected_sequence': expected_seq,
                'buffer_size': buffer_size,
                'error_count': errors,
                'reconnect_count': reconnects,
                'pending_gaps': len(self._pending_gaps.get(f"{channel}:{symbol}", []))
            }


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def create_websocket_monitor(
    max_latency_ms: float = 5000.0,
    enable_alerts: bool = True,
    telegram_bot=None,
    chat_id: str = None
) -> WebsocketIntegrityMonitor:
    """
    Create and configure a Websocket Integrity Monitor.
    
    Args:
        max_latency_ms: Maximum acceptable latency
        enable_alerts: Enable Telegram alerts
        telegram_bot: Telegram bot instance
        chat_id: Telegram chat ID
    
    Returns:
        Configured WebsocketIntegrityMonitor
    """
    monitor = WebsocketIntegrityMonitor(max_latency_ms=max_latency_ms)
    
    if enable_alerts and telegram_bot and chat_id:
        def on_gap(gap: SequenceGap):
            if gap.missing_count > 10:
                message = f"""
⚠️ WEBSOCKET SEQUENCE GAP

Channel: {gap.channel}
Symbol: {gap.symbol}
Missing: {gap.missing_count} messages
Expected: {gap.expected_seq}
Received: {gap.actual_seq}
Recovery: {gap.recovery_method}
"""
                try:
                    telegram_bot.send_message(
                        chat_id=chat_id,
                        text=message,
                        parse_mode='Markdown'
                    )
                except Exception as e:
                    logger.error(f"Failed to send gap alert: {e}")
        
        monitor.register_gap_callback(on_gap)
        logger.info("Telegram gap alerts enabled")
    
    return monitor


if __name__ == "__main__":
    print("AEGIS QUANTSEC - Websocket Integrity Monitor")
    print()
    print("Features:")
    print("  - Sequence number validation")
    print("  - Gap detection and recovery")
    print("  - Message reordering buffer")
    print("  - Latency monitoring")
    print("  - Connection health tracking")
    print()
    print("Usage:")
    print("  from HolonicTrader.websocket_integrity import WebsocketIntegrityMonitor")
    print("  monitor = WebsocketIntegrityMonitor()")
    print("  monitor.register_channel('book', 'BTC/USDT')")
    print("  is_valid, event = monitor.process_message(...)")
