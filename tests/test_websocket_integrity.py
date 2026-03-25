"""
Tests for AEGIS Websocket Integrity Monitor

Run: pytest tests/test_websocket_integrity.py -v
"""

import pytest
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HolonicTrader.websocket_integrity import (
    WebsocketIntegrityMonitor,
    WebsocketEvent,
    SequenceGap,
    create_websocket_monitor
)


class TestWebsocketEvent:
    """Test WebsocketEvent data structure."""
    
    def test_event_creation(self):
        """Event should create with auto-timestamp."""
        event = WebsocketEvent(
            channel='book',
            symbol='BTC/USDT',
            event_type='MARKET_DATA',
            sequence_num=100,
            data={'price': 50000},
            timestamp_ns=time.time_ns()
        )
        
        assert event.channel == 'book'
        assert event.sequence_num == 100
        assert event.received_ns > 0
    
    def test_latency_calculation(self):
        """Should calculate latency correctly."""
        now = time.time_ns()
        event = WebsocketEvent(
            channel='book',
            symbol='BTC/USDT',
            event_type='MARKET_DATA',
            sequence_num=100,
            data={},
            timestamp_ns=now - int(50 * 1e6),  # 50ms ago
            received_ns=now
        )
        
        assert 45 < event.latency_ms < 55  # Allow some tolerance


class TestWebsocketIntegrityMonitor:
    """Test Websocket Integrity Monitor."""
    
    def test_monitor_creation(self):
        """Monitor should initialize correctly."""
        monitor = WebsocketIntegrityMonitor()
        assert monitor is not None
        assert monitor._stats['total_events'] == 0
    
    def test_register_channel(self):
        """Should register channels for monitoring."""
        monitor = WebsocketIntegrityMonitor()
        monitor.register_channel('book', 'BTC/USDT', initial_sequence=0)
        
        assert 'book' in monitor._expected_sequences
        assert 'BTC/USDT' in monitor._expected_sequences['book']
        assert monitor._expected_sequences['book']['BTC/USDT'] == 0
    
    def test_process_valid_message(self):
        """Should accept valid messages."""
        monitor = WebsocketIntegrityMonitor()
        monitor.register_channel('book', 'BTC/USDT', initial_sequence=0)
        
        is_valid, event = monitor.process_message(
            channel='book',
            symbol='BTC/USDT',
            data={'price': 50000},
            sequence_num=0
        )
        
        assert is_valid
        assert event is not None
        assert event.sequence_num == 0
    
    def test_sequence_gap_detection(self):
        """Should detect sequence gaps."""
        monitor = WebsocketIntegrityMonitor()
        monitor.register_channel('book', 'BTC/USDT', initial_sequence=0)
        
        # First valid message
        monitor.process_message(
            channel='book', symbol='BTC/USDT',
            data={}, sequence_num=0
        )
        
        # Gap: skip to sequence 5
        is_valid, event = monitor.process_message(
            channel='book', symbol='BTC/USDT',
            data={}, sequence_num=5
        )
        
        assert not is_valid  # Should reject due to gap
        assert monitor._stats['gaps_detected'] >= 1
    
    def test_duplicate_detection(self):
        """Should detect duplicate messages."""
        monitor = WebsocketIntegrityMonitor()
        monitor.register_channel('book', 'BTC/USDT', initial_sequence=0)
        
        # First message
        monitor.process_message(
            channel='book', symbol='BTC/USDT',
            data={}, sequence_num=0
        )
        
        # Duplicate (old sequence)
        is_valid, event = monitor.process_message(
            channel='book', symbol='BTC/USDT',
            data={}, sequence_num=0
        )
        
        assert not is_valid
        assert monitor._stats['duplicates'] >= 1
    
    def test_gap_recovery(self):
        """Should recover from gaps when missing messages arrive."""
        monitor = WebsocketIntegrityMonitor(gap_recovery_window_ms=10000)
        monitor.register_channel('book', 'BTC/USDT', initial_sequence=0)
        
        # First message
        monitor.process_message(
            channel='book', symbol='BTC/USDT',
            data={}, sequence_num=0
        )
        
        # Gap: receive sequence 5 (missing 1-4)
        monitor.process_message(
            channel='book', symbol='BTC/USDT',
            data={}, sequence_num=5
        )
        
        # Now receive missing sequence 1
        is_valid, event = monitor.process_message(
            channel='book', symbol='BTC/USDT',
            data={}, sequence_num=1
        )
        
        # Should still be buffered, waiting for correct order
        # But gap should eventually be marked as recoverable
        gaps = monitor.get_integrity_report().sequence_gaps
        assert len(gaps) >= 1
    
    def test_reconnect_handling(self):
        """Should handle reconnects correctly."""
        monitor = WebsocketIntegrityMonitor()
        monitor.register_channel('book', 'BTC/USDT', initial_sequence=0)
        
        # Process some messages
        for i in range(5):
            monitor.process_message(
                channel='book', symbol='BTC/USDT',
                data={}, sequence_num=i
            )
        
        # Simulate reconnect
        monitor.handle_reconnect('book', 'BTC/USDT', new_sequence=100)
        
        # Next message should be accepted from new sequence
        is_valid, event = monitor.process_message(
            channel='book', symbol='BTC/USDT',
            data={}, sequence_num=100
        )
        
        assert is_valid
        assert monitor._expected_sequences['book']['BTC/USDT'] == 101
    
    def test_error_recording(self):
        """Should record errors."""
        monitor = WebsocketIntegrityMonitor()
        monitor.register_channel('book', 'BTC/USDT')
        
        monitor.record_error('book', 'BTC/USDT', 'Connection timeout')
        
        assert monitor._error_counts['book']['BTC/USDT'] == 1
    
    def test_integrity_report(self):
        """Should generate integrity report."""
        monitor = WebsocketIntegrityMonitor()
        monitor.register_channel('book', 'BTC/USDT')
        monitor.register_channel('trades', 'BTC/USDT')
        
        # Process some messages
        for i in range(10):
            monitor.process_message(
                channel='book', symbol='BTC/USDT',
                data={}, sequence_num=i
            )
        
        report = monitor.get_integrity_report()
        
        assert report.channels_monitored == 2
        assert report.total_events_processed >= 10
        assert report.health_status in ['HEALTHY', 'WARNING', 'DEGRADED', 'CRITICAL']
    
    def test_channel_status(self):
        """Should get channel status."""
        monitor = WebsocketIntegrityMonitor()
        monitor.register_channel('book', 'BTC/USDT', initial_sequence=5)
        
        status = monitor.get_channel_status('book', 'BTC/USDT')
        
        assert status['channel'] == 'book'
        assert status['symbol'] == 'BTC/USDT'
        assert status['expected_sequence'] == 5
    
    def test_statistics(self):
        """Should track statistics."""
        monitor = WebsocketIntegrityMonitor()
        monitor.register_channel('book', 'BTC/USDT')
        
        for i in range(20):
            monitor.process_message(
                channel='book', symbol='BTC/USDT',
                data={}, sequence_num=i
            )
        
        stats = monitor.get_statistics()
        
        assert stats['total_events'] == 20
        assert stats['valid_events'] == 20
        assert stats['rejected_events'] == 0
    
    def test_unregister_channel(self):
        """Should unregister channels."""
        monitor = WebsocketIntegrityMonitor()
        monitor.register_channel('book', 'BTC/USDT')
        monitor.unregister_channel('book', 'BTC/USDT')
        
        assert 'BTC/USDT' not in monitor._expected_sequences.get('book', {})
    
    def test_concurrent_message_processing(self):
        """Should handle concurrent messages safely."""
        import threading
        
        monitor = WebsocketIntegrityMonitor()
        monitor.register_channel('book', 'BTC/USDT')
        
        errors = []
        
        def process_messages():
            try:
                for i in range(50):
                    monitor.process_message(
                        channel='book', symbol='BTC/USDT',
                        data={}, sequence_num=i
                    )
            except Exception as e:
                errors.append(e)
        
        # Run concurrently
        threads = [threading.Thread(target=process_messages) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        
        assert len(errors) == 0


class TestCreateWebsocketMonitor:
    """Test monitor creation helper."""
    
    def test_create_basic_monitor(self):
        """Should create basic monitor."""
        monitor = create_websocket_monitor(
            max_latency_ms=3000.0,
            enable_alerts=False
        )
        
        assert monitor is not None
        assert monitor.max_latency_ms == 3000.0
    
    def test_create_monitor_with_alerts(self):
        """Should create monitor with alerts."""
        # Mock telegram bot
        class MockBot:
            def send_message(self, **kwargs):
                pass
        
        monitor = create_websocket_monitor(
            enable_alerts=True,
            telegram_bot=MockBot(),
            chat_id='123456'
        )
        
        assert monitor is not None
        assert len(monitor._on_gap_callbacks) > 0


class TestIntegration:
    """Integration tests."""
    
    def test_full_workflow(self):
        """Test complete monitoring workflow."""
        monitor = WebsocketIntegrityMonitor()
        
        # Register channels
        monitor.register_channel('book', 'BTC/USDT', initial_sequence=0)
        monitor.register_channel('book', 'ETH/USDT', initial_sequence=0)
        
        # Process normal messages
        for i in range(10):
            is_valid, event = monitor.process_message(
                channel='book', symbol='BTC/USDT',
                data={'price': 50000 + i},
                sequence_num=i
            )
            assert is_valid
        
        # Simulate gap
        is_valid, event = monitor.process_message(
            channel='book', symbol='BTC/USDT',
            data={'price': 50100},
            sequence_num=20  # Gap: missing 10-19
        )
        assert not is_valid
        
        # Get report
        report = monitor.get_integrity_report()
        assert report.health_status in ['HEALTHY', 'DEGRADED', 'CRITICAL']
        assert len(report.sequence_gaps) >= 1
        
        # Get statistics
        stats = monitor.get_statistics()
        assert stats['gaps_detected'] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
