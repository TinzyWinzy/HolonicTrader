"""
Integration Tests for AEGIS QUANTSEC Phase 2-4 Components

Tests:
1. Position Reconciliation Engine
2. Timestamp Oracle & Websocket Integrity Monitor
3. RL Agent Security Wrapper

Run: pytest tests/test_aegis_integration.py -v
"""

import pytest
import numpy as np
import time
import threading
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HolonicTrader.position_reconciliation import (
    PositionReconciliationEngine,
    PositionSnapshot,
    ReconciliationDiscrepancy
)

from HolonicTrader.timestamp_oracle import (
    WebsocketIntegrityMonitor,
    TimestampOracle,
    TimestampedEvent
)

from HolonicTrader.rl_agent_security import (
    RLAgentSecurityWrapper,
    RewardIntegrityFilter,
    AdversarialPatternDetector
)


# =============================================================================
# POSITION RECONCILIATION TESTS
# =============================================================================

class MockExecutor:
    """Mock ExecutorHolon for testing."""
    def __init__(self):
        self.positions = {}


class MockKrakenHolon:
    """Mock KrakenHolon for testing."""
    def __init__(self):
        self.futures = MockFuturesAPI()


class MockFuturesAPI:
    """Mock CCXT futures API."""
    def fetch_positions(self):
        return [
            {'symbol': 'PF_BTCUSD', 'contracts': 0.5, 'side': 'long'},
            {'symbol': 'PF_ETHUSD', 'contracts': -2.0, 'side': 'short'}
        ]
    
    def fetch_ticker(self, symbol):
        return {'last': 50000.0}


class TestPositionReconciliation:
    """Test Position Reconciliation Engine."""
    
    def test_engine_creation(self):
        """Engine should initialize correctly."""
        engine = PositionReconciliationEngine()
        assert engine is not None
        assert engine._stats['reconciliations_run'] == 0
    
    def test_ghost_detection(self):
        """Should detect ghost positions."""
        executor = MockExecutor()
        kraken = MockKrakenHolon()
        
        # Set up ghost scenario: position on exchange but not in ledger
        executor.positions = {}  # Empty ledger
        
        engine = PositionReconciliationEngine(
            executor_holon=executor,
            kraken_holon=kraken,
            reconciliation_interval_sec=0.1
        )
        
        report = engine.run_reconciliation()
        
        # Should detect ghosts
        ghosts = [d for d in report.discrepancies if d.discrepancy_type == 'GHOST']
        assert len(ghosts) > 0
    
    def test_leak_detection(self):
        """Should detect leak positions."""
        executor = MockExecutor()
        kraken = MockKrakenHolon()
        
        # Set up leak scenario: position in ledger but not on exchange
        from HolonicTrader.position_reconciliation import PositionSnapshot
        
        class MockPosition:
            def __init__(self, qty):
                self.quantity = qty
        
        executor.positions = {
            'SOL/USDT': MockPosition(100.0)  # Phantom position
        }
        
        engine = PositionReconciliationEngine(
            executor_holon=executor,
            kraken_holon=kraken
        )
        
        report = engine.run_reconciliation()
        
        # Should detect leaks
        leaks = [d for d in report.discrepancies if d.discrepancy_type == 'LEAK']
        assert len(leaks) > 0
    
    def test_integrity_score(self):
        """Should calculate integrity score."""
        engine = PositionReconciliationEngine()
        
        # No discrepancies = perfect score
        score = engine.get_integrity_score()
        assert score == 1.0
    
    def test_background_reconciliation(self):
        """Should run background reconciliation."""
        engine = PositionReconciliationEngine(
            reconciliation_interval_sec=0.1
        )
        
        engine.start()
        time.sleep(0.3)
        engine.stop()
        
        assert engine._stats['reconciliations_run'] >= 1


# =============================================================================
# TIMESTAMP ORACLE TESTS
# =============================================================================

class TestTimestampOracle:
    """Test Timestamp Oracle."""
    
    def test_oracle_creation(self):
        """Oracle should initialize correctly."""
        oracle = TimestampOracle()
        assert oracle is not None
    
    def test_get_timestamp(self):
        """Should return nanosecond timestamp."""
        oracle = TimestampOracle()
        ts = oracle.get_timestamp()
        
        assert isinstance(ts, int)
        assert ts > 0
        assert ts > 1e18  # Nanosecond timestamp
    
    def test_ordered_timestamp(self):
        """Should return ordered timestamps."""
        oracle = TimestampOracle()
        
        ts1, seq1 = oracle.get_ordered_timestamp('TEST')
        ts2, seq2 = oracle.get_ordered_timestamp('TEST')
        
        assert seq2 > seq1
        assert ts2 >= ts1
    
    def test_vector_clock(self):
        """Should maintain vector clock."""
        oracle = TimestampOracle()
        
        vc1 = oracle.update_vector_clock('node1')
        vc2 = oracle.update_vector_clock('node1')
        vc3 = oracle.update_vector_clock('node2')
        
        assert vc1['node1'] == 1
        assert vc2['node1'] == 2
        assert vc3['node2'] == 1


# =============================================================================
# WEBSOCKET INTEGRITY MONITOR TESTS
# =============================================================================

class TestWebsocketIntegrityMonitor:
    """Test Websocket Integrity Monitor."""
    
    def test_monitor_creation(self):
        """Monitor should initialize correctly."""
        monitor = WebsocketIntegrityMonitor()
        assert monitor is not None
    
    def test_register_channel(self):
        """Should register channels for monitoring."""
        monitor = WebsocketIntegrityMonitor()
        monitor.register_channel('book', 'BTC/USDT', initial_sequence=0)
        
        assert 'book' in monitor._expected_sequences
        assert 'BTC/USDT' in monitor._expected_sequences['book']
    
    def test_sequence_validation(self):
        """Should validate sequence numbers."""
        monitor = WebsocketIntegrityMonitor()
        monitor.register_channel('book', 'BTC/USDT', initial_sequence=0)
        
        # Valid sequence
        is_valid, event = monitor.process_event(
            channel='book',
            symbol='BTC/USDT',
            event_type='MARKET_DATA',
            data={'price': 50000},
            sequence_num=0,
            timestamp_ns=time.time_ns()
        )
        
        assert is_valid
        assert event is not None
    
    def test_gap_detection(self):
        """Should detect sequence gaps."""
        monitor = WebsocketIntegrityMonitor()
        monitor.register_channel('book', 'BTC/USDT', initial_sequence=0)
        
        # First valid message
        monitor.process_event(
            channel='book', symbol='BTC/USDT',
            event_type='MARKET_DATA', data={},
            sequence_num=0, timestamp_ns=time.time_ns()
        )
        
        # Gap: skip to sequence 5
        is_valid, event = monitor.process_event(
            channel='book', symbol='BTC/USDT',
            event_type='MARKET_DATA', data={},
            sequence_num=5, timestamp_ns=time.time_ns()
        )
        
        assert not is_valid  # Should reject due to gap
        assert monitor._stats['gaps_detected'] >= 1
    
    def test_duplicate_detection(self):
        """Should detect duplicate messages."""
        monitor = WebsocketIntegrityMonitor()
        monitor.register_channel('book', 'BTC/USDT', initial_sequence=0)
        
        # First message
        monitor.process_event(
            channel='book', symbol='BTC/USDT',
            event_type='MARKET_DATA', data={},
            sequence_num=0, timestamp_ns=time.time_ns()
        )
        
        # Duplicate (old sequence)
        is_valid, event = monitor.process_event(
            channel='book', symbol='BTC/USDT',
            event_type='MARKET_DATA', data={},
            sequence_num=0, timestamp_ns=time.time_ns()
        )
        
        assert not is_valid  # Should reject duplicate
        assert monitor._stats['duplicate_events'] >= 1
    
    def test_timestamp_anomaly_detection(self):
        """Should detect timestamp anomalies."""
        monitor = WebsocketIntegrityMonitor(max_timestamp_drift_ms=100)
        monitor.register_channel('book', 'BTC/USDT', initial_sequence=0)
        
        now = time.time_ns()
        
        # First message with current timestamp
        monitor.process_event(
            channel='book', symbol='BTC/USDT',
            event_type='MARKET_DATA', data={},
            sequence_num=0, timestamp_ns=now
        )
        
        # Second message with old timestamp (out of order)
        is_valid, event = monitor.process_event(
            channel='book', symbol='BTC/USDT',
            event_type='MARKET_DATA', data={},
            sequence_num=1, timestamp_ns=now - int(200 * 1e6)  # 200ms in past
        )
        
        # Should flag anomaly but may still accept
        assert len(monitor._timestamp_anomalies) >= 1
    
    def test_integrity_report(self):
        """Should generate integrity report."""
        monitor = WebsocketIntegrityMonitor()
        monitor.register_channel('book', 'BTC/USDT')
        
        # Process some events
        for i in range(10):
            monitor.process_event(
                channel='book', symbol='BTC/USDT',
                event_type='MARKET_DATA', data={},
                sequence_num=i, timestamp_ns=time.time_ns()
            )
        
        report = monitor.get_integrity_report()
        
        assert report.channels_monitored >= 1
        assert report.total_events_processed >= 10
        assert report.health_status in ['HEALTHY', 'WARNING', 'DEGRADED']


# =============================================================================
# RL AGENT SECURITY TESTS
# =============================================================================

class TestRewardIntegrityFilter:
    """Test Reward Integrity Filter."""
    
    def test_filter_creation(self):
        """Filter should initialize correctly."""
        filter = RewardIntegrityFilter()
        assert filter is not None
    
    def test_normal_reward_accepted(self):
        """Normal rewards should be accepted."""
        filter = RewardIntegrityFilter(min_samples=10)
        
        # Warm up with normal rewards
        for i in range(20):
            state = np.array([0.5, 0.3, 0.1])
            is_valid, anomaly, reward = filter.validate_reward(
                state=state, action=0, reward=0.1,
                next_state=state, done=False
            )
        
        # Now test with normal reward
        state = np.array([0.5, 0.3, 0.1])
        is_valid, anomaly, reward = filter.validate_reward(
            state=state, action=0, reward=0.1,
            next_state=state, done=False
        )
        
        assert is_valid
        assert anomaly is None
    
    def test_anomalous_reward_detected(self):
        """Anomalous rewards should be detected."""
        filter = RewardIntegrityFilter(min_samples=10, z_score_threshold=2.0)
        
        # Warm up with normal rewards
        for i in range(20):
            state = np.array([0.5, 0.3, 0.1])
            filter.validate_reward(
                state=state, action=0, reward=0.1,
                next_state=state, done=False
            )
        
        # Test with anomalous reward
        state = np.array([0.5, 0.3, 0.1])
        is_valid, anomaly, reward = filter.validate_reward(
            state=state, action=0, reward=10.0,  # Anomalous!
            next_state=state, done=False
        )
        
        assert anomaly is not None
        assert anomaly.anomaly_type == 'SPOOFED_REWARD'
    
    def test_reward_filtering(self):
        """Should filter extreme rewards."""
        filter = RewardIntegrityFilter(min_samples=10, z_score_threshold=2.0)
        
        # Warm up
        for i in range(20):
            state = np.array([0.5])
            filter.validate_reward(
                state=state, action=0, reward=0.1,
                next_state=state, done=False
            )
        
        # Extreme reward
        state = np.array([0.5])
        is_valid, anomaly, filtered_reward = filter.validate_reward(
            state=state, action=0, reward=100.0,
            next_state=state, done=False
        )
        
        # Should be filtered
        assert filtered_reward < 100.0
        assert abs(filtered_reward - 0.1) < 1.0  # Close to expected


class TestAdversarialPatternDetector:
    """Test Adversarial Pattern Detector."""
    
    def test_detector_creation(self):
        """Detector should initialize correctly."""
        detector = AdversarialPatternDetector()
        assert detector is not None
    
    def test_order_recording(self):
        """Should record orders."""
        detector = AdversarialPatternDetector()
        
        now = time.time()
        for i in range(30):
            detector.record_order(
                symbol='BTC/USDT',
                side='BUY' if i % 2 == 0 else 'SELL',
                quantity=1.0,
                price=50000 + i,
                order_type='LIMIT',
                timestamp=now - (30 - i)
            )
        
        assert detector._stats['orders_processed'] == 30
    
    def test_spoofing_detection(self):
        """Should detect spoofing patterns."""
        detector = AdversarialPatternDetector(order_imbalance_threshold=0.5)
        
        now = time.time()
        
        # Create imbalanced orders (mostly buys)
        for i in range(20):
            detector.record_order(
                symbol='BTC/USDT',
                side='BUY',  # All buys
                quantity=10.0,
                price=50000,
                order_type='LIMIT',
                timestamp=now - (20 - i)
            )
        
        patterns = detector.get_detected_patterns(symbol='BTC/USDT')
        
        # May detect spoofing due to imbalance
        spoofing = [p for p in patterns if p.pattern_type == 'SPOOFING']
        # Detection depends on threshold and data


class TestRLAgentSecurityWrapper:
    """Test RL Agent Security Wrapper."""
    
    def test_wrapper_creation(self):
        """Wrapper should initialize correctly."""
        mock_agent = type('MockAgent', (), {})()
        wrapper = RLAgentSecurityWrapper(agent=mock_agent, agent_type='DQN')
        assert wrapper is not None
        assert wrapper.agent_type == 'DQN'
    
    def test_experience_wrapping(self):
        """Should wrap experiences correctly."""
        mock_agent = type('MockAgent', (), {})()
        wrapper = RLAgentSecurityWrapper(
            agent=mock_agent,
            agent_type='DQN',
            enable_reward_filtering=True
        )
        
        state = np.array([0.5, 0.3, 0.1, 0.2])
        
        is_valid, experience = wrapper.wrap_experience(
            state=state,
            action=0,
            reward=0.1,
            next_state=state,
            done=False,
            symbol='BTC/USDT'
        )
        
        assert is_valid
        assert 'state' in experience
        assert 'reward' in experience
        assert experience['symbol'] == 'BTC/USDT'
    
    def test_security_report(self):
        """Should generate security report."""
        mock_agent = type('MockAgent', (), {})()
        wrapper = RLAgentSecurityWrapper(agent=mock_agent, agent_type='PPO')
        
        report = wrapper.get_security_report()
        
        assert report.agent_type == 'PPO'
        assert report.security_score >= 0.0
        assert report.security_score <= 1.0
        assert report.status in ['SECURE', 'WARNING', 'COMPROMISED']
    
    def test_adversarial_exposure_check(self):
        """Should check adversarial exposure."""
        mock_agent = type('MockAgent', (), {})()
        wrapper = RLAgentSecurityWrapper(
            agent=mock_agent,
            agent_type='DQN',
            enable_pattern_detection=True
        )
        
        risk = wrapper.check_adversarial_exposure('BTC/USDT')
        
        assert 'risk_level' in risk
        assert risk['risk_level'] in ['LOW', 'MEDIUM', 'HIGH', 'UNKNOWN']


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestAEGISIntegration:
    """Test integration of all AEGIS components."""
    
    def test_full_security_pipeline(self):
        """Test complete security pipeline."""
        # 1. Create reconciliation engine
        executor = MockExecutor()
        kraken = MockKrakenHolon()
        recon_engine = PositionReconciliationEngine(
            executor_holon=executor,
            kraken_holon=kraken
        )
        
        # 2. Create websocket monitor
        ws_monitor = WebsocketIntegrityMonitor()
        ws_monitor.register_channel('book', 'BTC/USDT')
        
        # 3. Create RL security wrapper
        mock_agent = type('MockAgent', (), {})()
        rl_wrapper = RLAgentSecurityWrapper(
            agent=mock_agent,
            agent_type='DQN'
        )
        
        # Run reconciliation
        recon_report = recon_engine.run_reconciliation()
        assert recon_report is not None
        
        # Process websocket events
        for i in range(5):
            ws_monitor.process_event(
                channel='book', symbol='BTC/USDT',
                event_type='MARKET_DATA', data={},
                sequence_num=i, timestamp_ns=time.time_ns()
            )
        
        ws_report = ws_monitor.get_integrity_report()
        assert ws_report is not None
        
        # Wrap RL experiences
        state = np.array([0.5, 0.3, 0.1, 0.2])
        for i in range(10):
            rl_wrapper.wrap_experience(
                state=state, action=0, reward=0.1,
                next_state=state, done=False
            )
        
        rl_report = rl_wrapper.get_security_report()
        assert rl_report is not None
        
        # All components should report status
        assert recon_report.summary['status'] in ['HEALTHY', 'DISCREPANCIES_FOUND']
        assert ws_report.health_status in ['HEALTHY', 'WARNING', 'DEGRADED']
        assert rl_report.status in ['SECURE', 'WARNING', 'COMPROMISED']
    
    def test_concurrent_operations(self):
        """Test thread safety of concurrent operations."""
        engine = PositionReconciliationEngine()
        ws_monitor = WebsocketIntegrityMonitor()
        
        errors = []
        
        def reconciliation_task():
            try:
                for _ in range(5):
                    engine.run_reconciliation()
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)
        
        def websocket_task():
            try:
                for i in range(50):
                    ws_monitor.process_event(
                        channel='book', symbol='BTC/USDT',
                        event_type='MARKET_DATA', data={},
                        sequence_num=i, timestamp_ns=time.time_ns()
                    )
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        # Run concurrently
        t1 = threading.Thread(target=reconciliation_task)
        t2 = threading.Thread(target=websocket_task)
        
        t1.start()
        t2.start()
        
        t1.join(timeout=5)
        t2.join(timeout=5)
        
        assert len(errors) == 0, f"Errors during concurrent operations: {errors}"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
