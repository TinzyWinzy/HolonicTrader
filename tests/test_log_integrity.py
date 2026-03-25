"""
Unit Tests for Log Integrity Verification Engine

Tests cover:
- Hash chain integrity
- Tamper detection
- Merkle tree proofs
- Blockchain anchoring
- Entry serialization

Run with: pytest tests/test_log_integrity.py -v
"""

import pytest
import json
import os
import sys
import time
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HolonicTrader.log_integrity import (
    LogEntry,
    LogIntegrityManager,
    MerkleTree,
    BlockchainAnchorer,
    AnchorRecord,
    TamperDetectionEngine,
    IntegrityViolation,
    compute_hash,
    compute_block_hash
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_log_dir():
    """Create temporary directory for test logs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def integrity_manager(temp_log_dir):
    """Create LogIntegrityManager with temp storage."""
    storage_path = os.path.join(temp_log_dir, "test_integrity_log.json")
    return LogIntegrityManager(
        storage_path=storage_path,
        auto_anchor_interval=10  # Anchor every 10 entries for testing
    )


# =============================================================================
# HASH FUNCTION TESTS
# =============================================================================

class TestHashFunctions:
    """Test cryptographic hash functions."""
    
    def test_compute_hash_deterministic(self):
        """Hash should be deterministic."""
        data = "test_data_123"
        hash1 = compute_hash(data)
        hash2 = compute_hash(data)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 hex chars
    
    def test_compute_hash_unique(self):
        """Different inputs should produce different hashes."""
        hash1 = compute_hash("data1")
        hash2 = compute_hash("data2")
        assert hash1 != hash2
    
    def test_compute_block_hash(self):
        """Block hash should include all fields."""
        hash1 = compute_block_hash(
            timestamp="2026-03-15T12:00:00Z",
            entropy_score=0.5,
            regime="ORDERED",
            action="EXECUTE",
            prev_hash="abc123"
        )
        
        hash2 = compute_block_hash(
            timestamp="2026-03-15T12:00:00Z",
            entropy_score=0.5,
            regime="ORDERED",
            action="EXECUTE",
            prev_hash="abc123"
        )
        
        # Same inputs = same hash
        assert hash1 == hash2
        
        # Different entropy = different hash
        hash3 = compute_block_hash(
            timestamp="2026-03-15T12:00:00Z",
            entropy_score=0.6,  # Changed
            regime="ORDERED",
            action="EXECUTE",
            prev_hash="abc123"
        )
        assert hash1 != hash3


# =============================================================================
# LOG ENTRY TESTS
# =============================================================================

class TestLogEntry:
    """Test LogEntry data structure."""
    
    def test_entry_creation(self):
        """LogEntry should compute hash on creation."""
        entry = LogEntry(
            sequence_num=0,
            timestamp="2026-03-15T12:00:00Z",
            timestamp_ns=1234567890,
            event_type="TRADE",
            symbol="BTC/USDT",
            data={"action": "BUY"},
            prev_hash="GENESIS"
        )
        
        assert entry.entry_hash != ""
        assert len(entry.entry_hash) == 64
    
    def test_entry_hash_changes_with_data(self):
        """Entry hash should change if any field changes."""
        entry1 = LogEntry(
            sequence_num=0,
            timestamp="2026-03-15T12:00:00Z",
            timestamp_ns=1234567890,
            event_type="TRADE",
            symbol="BTC/USDT",
            data={"action": "BUY"},
            prev_hash="GENESIS"
        )
        
        entry2 = LogEntry(
            sequence_num=0,
            timestamp="2026-03-15T12:00:00Z",
            timestamp_ns=1234567890,
            event_type="TRADE",
            symbol="BTC/USDT",
            data={"action": "SELL"},  # Changed
            prev_hash="GENESIS"
        )
        
        assert entry1.entry_hash != entry2.entry_hash
    
    def test_entry_serialization(self):
        """Entry should serialize and deserialize correctly."""
        original = LogEntry(
            sequence_num=5,
            timestamp="2026-03-15T12:00:00Z",
            timestamp_ns=1234567890,
            event_type="ORDER",
            symbol="ETH/USDT",
            data={"price": 3000, "qty": 1.5},
            prev_hash="prev_hash_123"
        )
        
        # Serialize
        d = original.to_dict()
        
        # Deserialize
        restored = LogEntry.from_dict(d)
        
        assert restored.sequence_num == original.sequence_num
        assert restored.event_type == original.event_type
        assert restored.data == original.data
        assert restored.entry_hash == original.entry_hash


# =============================================================================
# MERKLE TREE TESTS
# =============================================================================

class TestMerkleTree:
    """Test Merkle tree implementation."""
    
    def test_merkle_root_single_leaf(self):
        """Tree with single leaf should have that leaf as root."""
        hashes = ["abc123"]
        tree = MerkleTree(hashes)
        assert tree.root == hashes[0]
    
    def test_merkle_root_deterministic(self):
        """Same inputs should produce same root."""
        hashes = ["h1", "h2", "h3", "h4"]
        tree1 = MerkleTree(hashes)
        tree2 = MerkleTree(hashes)
        assert tree1.root == tree2.root
    
    def test_merkle_proof_verification(self):
        """Merkle proofs should verify correctly."""
        hashes = [compute_hash(f"leaf_{i}") for i in range(4)]
        tree = MerkleTree(hashes)
        
        # Get proof for leaf 2
        proof = tree.get_proof(2)
        
        # Verify proof
        is_valid = MerkleTree.verify_proof(
            leaf_hash=hashes[2],
            index=2,
            proof=proof,
            expected_root=tree.root
        )
        
        assert is_valid
    
    def test_merkle_proof_invalid(self):
        """Invalid proofs should fail verification."""
        hashes = [compute_hash(f"leaf_{i}") for i in range(4)]
        tree = MerkleTree(hashes)
        
        # Try to verify wrong leaf
        is_valid = MerkleTree.verify_proof(
            leaf_hash=compute_hash("wrong_leaf"),
            index=2,
            proof=tree.get_proof(2),
            expected_root=tree.root
        )
        
        assert not is_valid


# =============================================================================
# LOG INTEGRITY MANAGER TESTS
# =============================================================================

class TestLogIntegrityManager:
    """Test main LogIntegrityManager."""
    
    def test_manager_creation(self, integrity_manager):
        """Manager should initialize correctly."""
        assert integrity_manager.entries == []
        assert integrity_manager.sequence_counter == 0
    
    def test_log_event(self, integrity_manager):
        """Logging an event should create entry."""
        entry = integrity_manager.log_event(
            event_type="TRADE",
            symbol="BTC/USDT",
            data={"action": "BUY", "price": 50000}
        )
        
        assert entry.sequence_num == 0
        assert entry.event_type == "TRADE"
        assert entry.symbol == "BTC/USDT"
        assert len(integrity_manager.entries) == 1
    
    def test_log_event_chain(self, integrity_manager):
        """Multiple events should form valid chain."""
        integrity_manager.log_event("TRADE", "BTC/USDT", {"action": "BUY"})
        integrity_manager.log_event("TRADE", "ETH/USDT", {"action": "SELL"})
        integrity_manager.log_event("SIGNAL", "SOL/USDT", {"type": "LONG"})
        
        # Verify chain integrity
        is_valid, violations = integrity_manager.verify_integrity()
        
        assert is_valid
        assert len(violations) == 0
        assert len(integrity_manager.entries) == 3
    
    def test_hash_chain_integrity(self, integrity_manager):
        """Each entry should link to previous via hash."""
        entry1 = integrity_manager.log_event("TRADE", "BTC/USDT", {"action": "BUY"})
        entry2 = integrity_manager.log_event("TRADE", "ETH/USDT", {"action": "SELL"})
        
        # Entry 2's prev_hash should equal entry1's hash
        assert entry2.prev_hash == entry1.entry_hash
    
    def test_verify_integrity_valid(self, integrity_manager):
        """Valid chain should pass verification."""
        for i in range(5):
            integrity_manager.log_event("TRADE", "BTC/USDT", {"index": i})
        
        is_valid, violations = integrity_manager.verify_integrity()
        
        assert is_valid
        assert len(violations) == 0
    
    def test_get_entries_by_type(self, integrity_manager):
        """Should filter entries by type."""
        integrity_manager.log_event("TRADE", "BTC/USDT", {})
        integrity_manager.log_event("SIGNAL", "ETH/USDT", {})
        integrity_manager.log_event("TRADE", "SOL/USDT", {})
        integrity_manager.log_event("ERROR", "SYSTEM", {})
        
        trades = integrity_manager.get_entries_by_type("TRADE")
        assert len(trades) == 2
        
        errors = integrity_manager.get_entries_by_type("ERROR")
        assert len(errors) == 1
    
    def test_persistence(self, temp_log_dir):
        """Log should persist to disk and reload correctly."""
        path1 = os.path.join(temp_log_dir, "log1.json")
        manager1 = LogIntegrityManager(storage_path=path1)
        
        manager1.log_event("TRADE", "BTC/USDT", {"action": "BUY"})
        manager1.log_event("SIGNAL", "ETH/USDT", {"type": "LONG"})
        
        # Create new manager (should load from disk)
        manager2 = LogIntegrityManager(storage_path=path1)
        
        assert len(manager2.entries) == 2
        assert manager2.entries[0].symbol == "BTC/USDT"
        assert manager2.entries[1].symbol == "ETH/USDT"
    
    def test_integrity_report(self, integrity_manager):
        """Should generate comprehensive report."""
        integrity_manager.log_event("TRADE", "BTC/USDT", {})
        integrity_manager.log_event("SIGNAL", "ETH/USDT", {})
        
        report = integrity_manager.get_integrity_report()
        
        assert report['status'] == 'HEALTHY'
        assert report['total_entries'] == 2
        assert report['sequence_range']['last'] == 1


# =============================================================================
# TAMPER DETECTION TESTS
# =============================================================================

class TestTamperDetection:
    """Test tamper detection engine."""
    
    def test_detect_hash_modification(self, integrity_manager):
        """Should detect if entry data is modified."""
        integrity_manager.log_event("TRADE", "BTC/USDT", {"action": "BUY"})
        integrity_manager.log_event("TRADE", "ETH/USDT", {"action": "SELL"})
        
        # Tamper with entry data
        integrity_manager.entries[0].data["action"] = "SELL"
        
        is_valid, violations = integrity_manager.verify_integrity()
        
        assert not is_valid
        assert any(v.violation_type == "HASH_MISMATCH" for v in violations)
    
    def test_detect_entry_deletion(self, integrity_manager):
        """Should detect if entry is deleted."""
        integrity_manager.log_event("TRADE", "BTC/USDT", {"action": "BUY"})
        integrity_manager.log_event("TRADE", "ETH/USDT", {"action": "SELL"})
        integrity_manager.log_event("TRADE", "SOL/USDT", {"action": "BUY"})
        
        # Delete middle entry (simulate tampering)
        del integrity_manager.entries[1]
        
        is_valid, violations = integrity_manager.verify_integrity()
        
        assert not is_valid
        assert any(v.violation_type in ["SEQUENCE_GAP", "CHAIN_BREAK"] for v in violations)
    
    def test_detect_entry_insertion(self, integrity_manager):
        """Should detect if entry is inserted."""
        entry1 = integrity_manager.log_event("TRADE", "BTC/USDT", {"action": "BUY"})
        entry2 = integrity_manager.log_event("TRADE", "ETH/USDT", {"action": "SELL"})
        
        # Create fake entry with wrong prev_hash
        fake_entry = LogEntry(
            sequence_num=1,  # Inserted between 0 and 2
            timestamp="2026-03-15T12:00:00Z",
            timestamp_ns=1234567890,
            event_type="TRADE",
            symbol="FAKE",
            data={},
            prev_hash="WRONG_HASH"  # Doesn't match entry0's hash
        )
        
        # Insert fake entry
        integrity_manager.entries.insert(1, fake_entry)
        
        is_valid, violations = integrity_manager.verify_integrity()
        
        assert not is_valid
        assert any(v.violation_type == "CHAIN_BREAK" for v in violations)
    
    def test_detect_timestamp_anomaly(self, integrity_manager):
        """Should detect out-of-order timestamps."""
        integrity_manager.log_event(
            "TRADE", "BTC/USDT", {},
            timestamp_ns=2000000000  # Later timestamp
        )
        integrity_manager.log_event(
            "TRADE", "ETH/USDT", {},
            timestamp_ns=1000000000  # Earlier timestamp (anomaly)
        )
        
        is_valid, violations = integrity_manager.verify_integrity()
        
        assert any(v.violation_type == "TIMESTAMP_ANOMALY" for v in violations)


# =============================================================================
# BLOCKCHAIN ANCHORING TESTS
# =============================================================================

class TestBlockchainAnchoring:
    """Test blockchain anchoring system."""
    
    def test_create_anchor(self, integrity_manager):
        """Should create anchor record."""
        for i in range(5):
            integrity_manager.log_event("TRADE", "BTC/USDT", {"index": i})
        
        anchor = integrity_manager.create_anchor()
        
        assert anchor is not None
        assert anchor.log_sequence_num == 4
        assert anchor.merkle_root is not None
        assert anchor.external_txid is not None
    
    def test_anchor_persistence(self, temp_log_dir):
        """Anchors should persist to disk."""
        path = os.path.join(temp_log_dir, "log.json")
        manager = LogIntegrityManager(storage_path=path, auto_anchor_interval=10)

        for i in range(15):  # Trigger auto-anchor
            manager.log_event("TRADE", "BTC/USDT", {"index": i})

        # Reload
        manager2 = LogIntegrityManager(storage_path=path, auto_anchor_interval=10)

        assert len(manager2.anchorer.anchor_history) >= 1
    
    def test_anchor_verification(self, integrity_manager):
        """Should verify anchor integrity."""
        for i in range(5):
            integrity_manager.log_event("TRADE", "BTC/USDT", {"index": i})
        
        anchor = integrity_manager.create_anchor()
        
        is_valid = integrity_manager.anchorer.verify_anchor(anchor)
        assert is_valid


# =============================================================================
# CONVENIENCE METHOD TESTS
# =============================================================================

class TestConvenienceMethods:
    """Test convenience logging methods."""
    
    def test_log_trade(self, integrity_manager):
        """log_trade should create proper entry."""
        entry = integrity_manager.log_trade(
            symbol="BTC/USDT",
            action="BUY",
            quantity=0.1,
            price=50000,
            order_id="order_123"
        )
        
        assert entry.event_type == "TRADE"
        assert entry.data['action'] == "BUY"
        assert entry.data['quantity'] == 0.1
        assert entry.data['order_id'] == "order_123"
    
    def test_log_signal(self, integrity_manager):
        """log_signal should create proper entry."""
        entry = integrity_manager.log_signal(
            symbol="ETH/USDT",
            signal_type="LONG",
            conviction=0.85,
            strategy="MOMENTUM"
        )
        
        assert entry.event_type == "SIGNAL"
        assert entry.data['conviction'] == 0.85
        assert entry.data['strategy'] == "MOMENTUM"
    
    def test_log_error(self, integrity_manager):
        """log_error should create proper entry."""
        entry = integrity_manager.log_error(
            error_type="API_ERROR",
            message="Connection timeout",
            symbol="SYSTEM",
            traceback="stack trace here"
        )
        
        assert entry.event_type == "ERROR"
        assert entry.data['error_type'] == "API_ERROR"
        assert "traceback" in entry.data


# =============================================================================
# EXPORT TESTS
# =============================================================================

class TestExport:
    """Test export functionality."""
    
    def test_export_for_audit(self, integrity_manager, temp_log_dir):
        """Should export complete audit data."""
        for i in range(5):
            integrity_manager.log_event("TRADE", "BTC/USDT", {"index": i})
        
        output_path = os.path.join(temp_log_dir, "audit_export.json")
        result_path = integrity_manager.export_for_audit(output_path)
        
        assert result_path == output_path
        assert os.path.exists(output_path)
        
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert 'entries' in data
        assert 'integrity_report' in data
        assert len(data['entries']) == 5


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_log_verification(self, integrity_manager):
        """Empty log should be valid."""
        is_valid, violations = integrity_manager.verify_integrity()
        assert is_valid
        assert len(violations) == 0
    
    def test_single_entry_log(self, integrity_manager):
        """Single entry log should verify correctly."""
        integrity_manager.log_event("TRADE", "BTC/USDT", {})
        
        is_valid, violations = integrity_manager.verify_integrity()
        assert is_valid
    
    def test_get_nonexistent_entry(self, integrity_manager):
        """Should return None for nonexistent entry."""
        entry = integrity_manager.get_entry(999)
        assert entry is None
    
    def test_large_log(self, temp_log_dir):
        """Should handle large logs efficiently."""
        path = os.path.join(temp_log_dir, "large_log.json")
        manager = LogIntegrityManager(
            storage_path=path,
            auto_anchor_interval=100
        )
        
        # Log 1000 entries
        for i in range(1000):
            manager.log_event("TRADE", "BTC/USDT", {"index": i})
        
        assert len(manager.entries) == 1000
        
        # Verify should still be fast
        start = time.time()
        is_valid, violations = manager.verify_integrity()
        elapsed = time.time() - start
        
        assert is_valid
        assert elapsed < 5.0  # Should complete in under 5 seconds


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests simulating real usage."""
    
    def test_full_trading_session(self, temp_log_dir):
        """Simulate complete trading session."""
        path = os.path.join(temp_log_dir, "session.json")
        manager = LogIntegrityManager(storage_path=path)
        
        # Session start
        manager.log_event("SESSION_START", "SYSTEM", {"timestamp": time.time()})
        
        # Trading loop
        trades = [
            ("BTC/USDT", "BUY", 0.1, 50000),
            ("ETH/USDT", "SELL", 1.0, 3000),
            ("SOL/USDT", "BUY", 10.0, 100),
        ]
        
        for symbol, action, qty, price in trades:
            # Log signal
            manager.log_signal(symbol, action, 0.75, "MOMENTUM")
            
            # Log trade
            manager.log_trade(symbol, action, qty, price, f"order_{symbol}")
        
        # Session end
        manager.log_event("SESSION_END", "SYSTEM", {"trades": len(trades)})
        
        # Create anchor
        anchor = manager.create_anchor()
        
        # Verify everything
        is_valid, violations = manager.verify_integrity()
        
        assert is_valid
        assert len(manager.entries) == 8  # Start + 3*(signal+trade) + End
        assert anchor is not None
    
    def test_multi_session_persistence(self, temp_log_dir):
        """Multiple sessions should persist correctly."""
        path = os.path.join(temp_log_dir, "multi_session.json")
        
        # Session 1
        manager1 = LogIntegrityManager(storage_path=path)
        manager1.log_event("TRADE", "BTC/USDT", {"session": 1})
        manager1.log_event("TRADE", "ETH/USDT", {"session": 1})
        del manager1  # Simulate shutdown
        
        # Session 2
        manager2 = LogIntegrityManager(storage_path=path)
        manager2.log_event("TRADE", "SOL/USDT", {"session": 2})
        
        # Session 3
        manager3 = LogIntegrityManager(storage_path=path)
        
        assert len(manager3.entries) == 3
        assert manager3.entries[0].data['session'] == 1
        assert manager3.entries[2].data['session'] == 2


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
