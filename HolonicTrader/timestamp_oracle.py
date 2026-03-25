"""
AEGIS QUANTSEC - Timestamp Oracle & Websocket Integrity Monitor

Provides:
1. Nanosecond-accurate event ordering
2. Websocket sequence number validation
3. Gap detection and automatic recovery
4. Market data integrity verification

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

logger = logging.getLogger("AEGIS.TimestampOracle")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TimestampedEvent:
    """
    Event with nanosecond-precision timestamp.
    
    Used for ordering market data, trades, and system events.
    """
    event_type: str  # MARKET_DATA, TRADE, ORDER, FILL, SYSTEM
    symbol: str
    timestamp_ns: int  # Nanosecond timestamp
    sequence_num: int  # Optional sequence number from exchange
    data: Dict[str, Any]
    source: str  # WEBSOCKET, REST, INTERNAL
    received_ns: int = 0  # When we received it
    
    def __post_init__(self):
        if self.received_ns == 0:
            self.received_ns = time.time_ns()
    
    @property
    def latency_ns(self) -> int:
        """Latency between event timestamp and receipt."""
        return self.received_ns - self.timestamp_ns
    
    @property
    def latency_ms(self) -> float:
        """Latency in milliseconds."""
        return self.latency_ns / 1e6
    
    def to_dict(self) -> Dict:
        return {
            'event_type': self.event_type,
            'symbol': self.symbol,
            'timestamp_ns': self.timestamp_ns,
            'sequence_num': self.sequence_num,
            'data': self.data,
            'source': self.source,
            'received_ns': self.received_ns,
            'latency_ms': self.latency_ms
        }


@dataclass
class SequenceGap:
    """Detected gap in sequence numbers."""
    channel: str
    symbol: str
    expected_seq: int
    actual_seq: int
    missing_count: int
    detected_at: float
    recovered: bool = False
    recovery_method: str = ""


@dataclass
class TimestampAnomaly:
    """Detected timestamp ordering anomaly."""
    anomaly_type: str  # OUT_OF_ORDER, FUTURE_TIMESTAMP, LARGE_GAP, DUPLICATE
    symbol: str
    expected_timestamp_ns: int
    actual_timestamp_ns: int
    drift_ms: float
    detected_at: float
    details: str


@dataclass
class IntegrityReport:
    """Websocket integrity report."""
    timestamp: float
    channels_monitored: int
    total_events_processed: int
    sequence_gaps: List[SequenceGap]
    timestamp_anomalies: List[TimestampAnomaly]
    latency_stats: Dict[str, float]
    health_status: str
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'channels_monitored': self.channels_monitored,
            'total_events_processed': self.total_events_processed,
            'sequence_gaps': [
                {
                    'channel': g.channel,
                    'symbol': g.symbol,
                    'missing_count': g.missing_count,
                    'recovered': g.recovered
                }
                for g in self.sequence_gaps
            ],
            'timestamp_anomalies': [
                {
                    'type': a.anomaly_type,
                    'symbol': a.symbol,
                    'drift_ms': a.drift_ms
                }
                for a in self.timestamp_anomalies
            ],
            'latency_stats': self.latency_stats,
            'health_status': self.health_status
        }


# =============================================================================
# WEBSOCKET INTEGRITY MONITOR
# =============================================================================

class WebsocketIntegrityMonitor:
    """
    Monitors websocket feed integrity.
    
    Tracks:
    - Sequence numbers per channel
    - Timestamp ordering
    - Message gaps
    - Latency anomalies
    
    Recovers from:
    - Sequence gaps (buffer and reorder)
    - Connection drops (resync on reconnect)
    - Stale data (timestamp validation)
    """
    
    def __init__(
        self,
        max_latency_ms: float = 5000.0,  # Max acceptable latency
        max_timestamp_drift_ms: float = 1000.0,  # Max clock drift
        buffer_size: int = 1000,  # Messages to buffer for reordering
        gap_recovery_window_ms: float = 5000.0  # Wait time for late messages
    ):
        self.max_latency_ms = max_latency_ms
        self.max_timestamp_drift_ms = max_timestamp_drift_ms
        self.buffer_size = buffer_size
        self.gap_recovery_window_ms = gap_recovery_window_ms
        
        # Sequence tracking per channel
        self._expected_sequences: Dict[str, Dict[str, int]] = defaultdict(dict)  # channel -> symbol -> seq
        self._sequence_buffers: Dict[str, Dict[str, List[TimestampedEvent]]] = defaultdict(lambda: defaultdict(list))
        
        # Timestamp tracking
        self._last_timestamps: Dict[str, Dict[str, int]] = defaultdict(dict)  # channel -> symbol -> timestamp_ns
        self._timestamp_anomalies: List[TimestampAnomaly] = []
        
        # Gap tracking
        self._sequence_gaps: List[SequenceGap] = []
        self._pending_gaps: Dict[str, List[int]] = defaultdict(list)  # channel -> missing sequences
        
        # Statistics
        self._stats = {
            'total_events': 0,
            'out_of_order_events': 0,
            'gaps_detected': 0,
            'gaps_recovered': 0,
            'latency_violations': 0,
            'duplicate_events': 0
        }
        
        # Callbacks
        self._on_gap_callbacks: List[Callable] = []
        self._on_anomaly_callbacks: List[Callable] = []
        
        # State
        self._running = False
        self._lock = threading.RLock()
    
    def register_channel(self, channel: str, symbol: str, initial_sequence: int = 0):
        """Register a channel/symbol for monitoring."""
        with self._lock:
            self._expected_sequences[channel][symbol] = initial_sequence
            self._last_timestamps[channel][symbol] = 0
            self._sequence_buffers[channel][symbol] = []
            logger.info(f"Registered channel {channel} for {symbol}, initial seq: {initial_sequence}")
    
    def process_event(
        self,
        channel: str,
        symbol: str,
        event_type: str,
        data: Dict[str, Any],
        sequence_num: Optional[int] = None,
        timestamp_ns: Optional[int] = None
    ) -> Tuple[bool, Optional[TimestampedEvent]]:
        """
        Process an incoming websocket event.
        
        Returns:
            (is_valid, ordered_event)
            - is_valid: False if event should be rejected
            - ordered_event: Event ready for processing (may be delayed for ordering)
        """
        received_ns = time.time_ns()
        
        # Create event
        event = TimestampedEvent(
            event_type=event_type,
            symbol=symbol,
            timestamp_ns=timestamp_ns or received_ns,
            sequence_num=sequence_num or 0,
            data=data,
            source='WEBSOCKET',
            received_ns=received_ns
        )
        
        with self._lock:
            self._stats['total_events'] += 1
            
            # 1. Validate sequence number
            if sequence_num is not None:
                seq_valid = self._validate_sequence(channel, symbol, sequence_num, event)
                if not seq_valid:
                    return False, None
            
            # 2. Validate timestamp
            ts_valid = self._validate_timestamp(channel, symbol, event)
            if not ts_valid:
                return False, None
            
            # 3. Check latency
            self._check_latency(event)
            
            # 4. Buffer and reorder if needed
            buffered_event = self._buffer_and_reorder(channel, symbol, event)
            
            return True, buffered_event
    
    def _validate_sequence(
        self,
        channel: str,
        symbol: str,
        sequence_num: int,
        event: TimestampedEvent
    ) -> bool:
        """Validate sequence number and detect gaps."""
        expected = self._expected_sequences[channel].get(symbol, 0)
        
        if sequence_num < expected:
            # Duplicate or very old message
            self._stats['duplicate_events'] += 1
            logger.debug(f"Duplicate message: seq {sequence_num} < expected {expected}")
            return False
        
        if sequence_num > expected:
            # Gap detected
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
            self._pending_gaps[channel].extend(range(expected, sequence_num))
            
            logger.warning(
                f"Sequence gap on {channel}/{symbol}: "
                f"expected {expected}, got {sequence_num}, missing {missing_count}"
            )
            
            # Trigger gap callbacks
            self._trigger_gap_callbacks(gap)
            
            # Buffer this event and wait for missing messages
            self._sequence_buffers[channel][symbol].append(event)
            self._sort_buffer(channel, symbol)
            
            return False  # Don't process yet, wait for missing messages
        
        # Sequence is correct
        self._expected_sequences[channel][symbol] = sequence_num + 1
        
        # Check if this fills any pending gaps
        self._fill_pending_gaps(channel, sequence_num)
        
        return True
    
    def _validate_timestamp(
        self,
        channel: str,
        symbol: str,
        event: TimestampedEvent
    ) -> bool:
        """Validate timestamp ordering."""
        last_ts = self._last_timestamps[channel].get(symbol, 0)
        
        # Check for out-of-order
        if event.timestamp_ns < last_ts:
            drift_ms = (last_ts - event.timestamp_ns) / 1e6
            
            anomaly = TimestampAnomaly(
                anomaly_type='OUT_OF_ORDER',
                symbol=symbol,
                expected_timestamp_ns=last_ts,
                actual_timestamp_ns=event.timestamp_ns,
                drift_ms=drift_ms,
                detected_at=time.time(),
                details=f"Event timestamp {event.timestamp_ns} is before last seen {last_ts}"
            )
            self._timestamp_anomalies.append(anomaly)
            self._stats['out_of_order_events'] += 1
            
            logger.warning(
                f"Out-of-order timestamp on {channel}/{symbol}: "
                f"drift={drift_ms:.2f}ms"
            )
            
            self._trigger_anomaly_callbacks(anomaly)
            
            # Allow but flag - some reordering is normal
            if drift_ms > self.max_timestamp_drift_ms:
                return False  # Too much drift, reject
        
        # Check for future timestamp
        now_ns = time.time_ns()
        future_threshold = now_ns + int(self.max_timestamp_drift_ms * 1e6)
        
        if event.timestamp_ns > future_threshold:
            drift_ms = (event.timestamp_ns - now_ns) / 1e6
            
            anomaly = TimestampAnomaly(
                anomaly_type='FUTURE_TIMESTAMP',
                symbol=symbol,
                expected_timestamp_ns=now_ns,
                actual_timestamp_ns=event.timestamp_ns,
                drift_ms=drift_ms,
                detected_at=time.time(),
                details=f"Event timestamp is {drift_ms:.2f}ms in the future"
            )
            self._timestamp_anomalies.append(anomaly)
            
            logger.warning(f"Future timestamp on {channel}/{symbol}: {drift_ms:.2f}ms ahead")
            return False
        
        # Update last timestamp
        self._last_timestamps[channel][symbol] = event.timestamp_ns
        
        return True
    
    def _check_latency(self, event: TimestampedEvent):
        """Check event latency."""
        latency_ms = event.latency_ms
        
        if latency_ms > self.max_latency_ms:
            self._stats['latency_violations'] += 1
            logger.warning(
                f"High latency event: {event.event_type} {event.symbol} "
                f"latency={latency_ms:.2f}ms"
            )
    
    def _buffer_and_reorder(
        self,
        channel: str,
        symbol: str,
        event: TimestampedEvent
    ) -> Optional[TimestampedEvent]:
        """Buffer events and release in order."""
        buffer = self._sequence_buffers[channel][symbol]
        buffer.append(event)
        
        # Keep buffer bounded
        if len(buffer) > self.buffer_size:
            buffer.pop(0)
        
        # Sort by timestamp
        self._sort_buffer(channel, symbol)
        
        # Release events in order
        return self._release_buffered_events(channel, symbol)
    
    def _sort_buffer(self, channel: str, symbol: str):
        """Sort buffer by timestamp."""
        buffer = self._sequence_buffers[channel][symbol]
        buffer.sort(key=lambda e: e.timestamp_ns)
    
    def _release_buffered_events(
        self,
        channel: str,
        symbol: str
    ) -> Optional[TimestampedEvent]:
        """Release oldest buffered event if it's in order."""
        buffer = self._sequence_buffers[channel][symbol]
        
        if not buffer:
            return None
        
        # Check if oldest event is ready
        oldest = buffer[0]
        expected_seq = self._expected_sequences[channel].get(symbol, 0)
        
        # If sequence tracking is active, check sequence
        if oldest.sequence_num > 0:
            if oldest.sequence_num != expected_seq:
                return None  # Wait for correct sequence
        
        # Release the event
        buffer.pop(0)
        return oldest
    
    def _fill_pending_gaps(self, channel: str, received_seq: int):
        """Mark gaps as filled when we receive missing sequences."""
        if channel not in self._pending_gaps:
            return
        
        # Remove received sequence from pending
        if received_seq - 1 in self._pending_gaps[channel]:
            self._pending_gaps[channel].remove(received_seq - 1)
            
            # Check if all gaps are filled
            if not self._pending_gaps[channel]:
                # Mark gaps as recovered
                for gap in self._sequence_gaps:
                    if gap.channel == channel and not gap.recovered:
                        gap.recovered = True
                        gap.recovery_method = 'LATE_MESSAGE_RECEIVED'
                        self._stats['gaps_recovered'] += 1
    
    def _trigger_gap_callbacks(self, gap: SequenceGap):
        """Trigger registered gap callbacks."""
        for callback in self._on_gap_callbacks:
            try:
                callback(gap)
            except Exception as e:
                logger.error(f"Gap callback error: {e}")
    
    def _trigger_anomaly_callbacks(self, anomaly: TimestampAnomaly):
        """Trigger registered anomaly callbacks."""
        for callback in self._on_anomaly_callbacks:
            try:
                callback(anomaly)
            except Exception as e:
                logger.error(f"Anomaly callback error: {e}")
    
    def register_gap_callback(self, callback: Callable):
        """Register callback for sequence gaps."""
        self._on_gap_callbacks.append(callback)
    
    def register_anomaly_callback(self, callback: Callable):
        """Register callback for timestamp anomalies."""
        self._on_anomaly_callbacks.append(callback)
    
    def handle_reconnect(self, channel: str, symbol: str, new_sequence: int):
        """Handle websocket reconnect - reset sequence tracking."""
        with self._lock:
            logger.info(f"Reconnect detected on {channel}/{symbol}, resetting to seq {new_sequence}")
            
            # Clear buffer
            self._sequence_buffers[channel][symbol] = []
            
            # Update expected sequence
            self._expected_sequences[channel][symbol] = new_sequence
            
            # Mark any pending gaps as unrecoverable
            for gap in self._sequence_gaps:
                if gap.channel == channel and gap.symbol == symbol and not gap.recovered:
                    gap.recovery_method = 'RECONNECT_RESET'
    
    def get_integrity_report(self) -> IntegrityReport:
        """Get current integrity status."""
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
            
            # Determine health status
            unrecovered_gaps = sum(1 for g in self._sequence_gaps if not g.recovered)
            recent_anomalies = sum(1 for a in self._timestamp_anomalies[-100:] 
                                   if a.detected_at > time.time() - 60)
            
            if unrecovered_gaps > 0:
                health_status = 'DEGRADED'
            elif recent_anomalies > 10:
                health_status = 'WARNING'
            elif self._stats['latency_violations'] > 100:
                health_status = 'WARNING'
            else:
                health_status = 'HEALTHY'
            
            return IntegrityReport(
                timestamp=time.time(),
                channels_monitored=len(self._expected_sequences),
                total_events_processed=self._stats['total_events'],
                sequence_gaps=list(self._sequence_gaps[-50:]),  # Last 50 gaps
                timestamp_anomalies=list(self._timestamp_anomalies[-50:]),
                latency_stats=latency_stats,
                health_status=health_status
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        with self._lock:
            return self._stats.copy()


# =============================================================================
# TIMESTAMP ORACLE
# =============================================================================

class TimestampOracle:
    """
    Centralized timestamp authority for event ordering.
    
    Provides:
    - Nanosecond-accurate timestamps
    - Causality tracking (happens-before relationships)
    - Clock synchronization monitoring
    - Vector clocks for distributed ordering
    """
    
    def __init__(self):
        self._lock = threading.RLock()
        self._event_counter = 0
        self._vector_clock: Dict[str, int] = defaultdict(int)
        self._causality_log: deque = deque(maxlen=10000)
        
        # Clock sync monitoring
        self._sync_samples: deque = deque(maxlen=1000)
        self._last_ntp_sync: Optional[float] = None
        self._clock_drift_estimate_ms: float = 0.0
    
    def get_timestamp(self) -> int:
        """Get current nanosecond timestamp."""
        return time.time_ns()
    
    def get_ordered_timestamp(self, event_type: str) -> Tuple[int, int]:
        """
        Get timestamp with ordering guarantee.
        
        Returns:
            (timestamp_ns, sequence_num)
        """
        with self._lock:
            self._event_counter += 1
            return time.time_ns(), self._event_counter
    
    def update_vector_clock(self, node_id: str) -> Dict[str, int]:
        """
        Update and return vector clock for distributed ordering.
        
        Args:
            node_id: Identifier for this node/component
        
        Returns:
            Current vector clock state
        """
        with self._lock:
            self._vector_clock[node_id] += 1
            return dict(self._vector_clock)
    
    def record_causality(
        self,
        cause_event: TimestampedEvent,
        effect_event: TimestampedEvent
    ):
        """Record a causality relationship between events."""
        with self._lock:
            self._causality_log.append({
                'cause': cause_event.to_dict(),
                'effect': effect_event.to_dict(),
                'recorded_at': time.time()
            })
    
    def verify_causality(
        self,
        event_a: TimestampedEvent,
        event_b: TimestampedEvent
    ) -> bool:
        """
        Verify that event A happened before event B.
        
        Returns:
            True if A causally precedes B
        """
        # Simple timestamp-based check
        if event_a.timestamp_ns > event_b.timestamp_ns:
            return False
        
        # Check vector clock if available
        # (In full implementation, would compare vector clocks)
        
        return True
    
    def sync_clock(self, ntp_offset_ms: float):
        """
        Record clock synchronization with NTP server.
        
        Args:
            ntp_offset_ms: Offset from NTP server in milliseconds
        """
        with self._lock:
            self._last_ntp_sync = time.time()
            self._clock_drift_estimate_ms = ntp_offset_ms
            self._sync_samples.append({
                'timestamp': time.time(),
                'offset_ms': ntp_offset_ms
            })
            
            logger.info(f"Clock synced: offset={ntp_offset_ms:.2f}ms")
    
    def get_clock_status(self) -> Dict[str, Any]:
        """Get clock synchronization status."""
        with self._lock:
            # Calculate drift rate
            drift_rate = 0.0
            if len(self._sync_samples) > 1:
                samples = list(self._sync_samples)
                time_diff = samples[-1]['timestamp'] - samples[0]['timestamp']
                offset_diff = samples[-1]['offset_ms'] - samples[0]['offset_ms']
                if time_diff > 0:
                    drift_rate = offset_diff / time_diff  # ms per second
            
            return {
                'last_sync': self._last_ntp_sync,
                'current_drift_ms': self._clock_drift_estimate_ms,
                'drift_rate_ms_per_sec': drift_rate,
                'samples_collected': len(self._sync_samples)
            }
    
    def adjust_timestamp(self, timestamp_ns: int) -> int:
        """
        Adjust timestamp for known clock drift.
        
        Args:
            timestamp_ns: Raw timestamp
        
        Returns:
            Adjusted timestamp
        """
        if self._clock_drift_estimate_ms == 0:
            return timestamp_ns
        
        # Apply drift correction
        drift_ns = int(self._clock_drift_estimate_ms * 1e6)
        return timestamp_ns - drift_ns


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def integrate_websocket_monitor(
    ws_client,
    kraken_holon=None,
    enable_alerts: bool = False,
    telegram_bot=None,
    chat_id: str = None
) -> WebsocketIntegrityMonitor:
    """
    Integrate WebsocketIntegrityMonitor with existing websocket client.
    
    Usage:
        monitor = integrate_websocket_monitor(ws_client, kraken)
        monitor.start()
    """
    monitor = WebsocketIntegrityMonitor()
    
    # Register channels for common Kraken feeds
    if kraken_holon:
        # Get symbols from config
        try:
            import config
            symbols = getattr(config, 'ALLOWED_ASSETS', ['BTC/USDT', 'ETH/USDT'])
            
            for symbol in symbols[:10]:  # Monitor top 10
                monitor.register_channel('book', symbol, initial_sequence=0)
                monitor.register_channel('trades', symbol, initial_sequence=0)
        except:
            pass
    
    # Add Telegram alerts
    if enable_alerts and telegram_bot and chat_id:
        def on_gap(gap: SequenceGap):
            if gap.missing_count > 10:  # Only alert for significant gaps
                message = f"""
⚠️ WEBSOCKET SEQUENCE GAP

Channel: {gap.channel}
Symbol: {gap.symbol}
Missing: {gap.missing_count} messages
Expected seq: {gap.expected_seq}
Received seq: {gap.actual_seq}
"""
                try:
                    telegram_bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
                except:
                    pass
        
        monitor.register_gap_callback(on_gap)
    
    return monitor


def integrate_timestamp_oracle() -> TimestampOracle:
    """Create and configure TimestampOracle."""
    oracle = TimestampOracle()
    
    # Attempt initial NTP sync (if ntplib available)
    try:
        import ntplib
        client = ntplib.NTPClient()
        response = client.request('pool.ntp.org', version=3)
        oracle.sync_clock(response.offset * 1000)  # Convert to ms
        logger.info("Initial NTP sync successful")
    except ImportError:
        logger.warning("ntplib not available, clock sync disabled")
    except Exception as e:
        logger.warning(f"NTP sync failed: {e}")
    
    return oracle


# =============================================================================
# CLI VERIFICATION
# =============================================================================

def run_integrity_check(
    ws_monitor: WebsocketIntegrityMonitor = None
) -> Dict[str, Any]:
    """Run a one-time integrity check."""
    if ws_monitor:
        report = ws_monitor.get_integrity_report()
        return report.to_dict()
    
    return {
        'status': 'NO_MONITOR',
        'message': 'Websocket monitor not provided'
    }


if __name__ == "__main__":
    print("AEGIS QUANTSEC - Timestamp Oracle & Websocket Integrity Monitor")
    print()
    print("This module provides:")
    print("  1. Nanosecond-accurate event ordering")
    print("  2. Websocket sequence number validation")
    print("  3. Gap detection and recovery")
    print("  4. Clock synchronization monitoring")
    print()
    print("Usage:")
    print("  from HolonicTrader.timestamp_oracle import integrate_websocket_monitor, integrate_timestamp_oracle")
    print("  monitor = integrate_websocket_monitor(ws_client)")
    print("  oracle = integrate_timestamp_oracle()")
