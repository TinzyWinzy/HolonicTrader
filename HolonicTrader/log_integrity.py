"""
Log Integrity Verification Engine - AEGIS QUANTSEC

Tamper-evident logging system with cryptographic hash chaining
and external blockchain anchoring for audit trail integrity.

Author: Aegis QuantSec v1.0
Date: 2026-03-15
"""

import hashlib
import json
import time
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import threading
import struct

# Optional: Rust acceleration for hash computation
try:
    import holonic_speed
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("[LogIntegrity] ℹ️ Rust acceleration not available, using Python fallback")


# =============================================================================
# CRYPTOGRAPHIC PRIMITIVES
# =============================================================================

def compute_hash_python(data: str) -> str:
    """Compute SHA-256 hash using Python stdlib."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def compute_hash_rust(data: str) -> str:
    """Compute hash using Rust acceleration if available."""
    try:
        # holonic_speed typically exposes a hash function
        return holonic_speed.compute_hash(data)
    except (ImportError, AttributeError):
        return compute_hash_python(data)


def compute_hash(data: str) -> str:
    """Compute SHA-256 hash, preferring Rust implementation."""
    if RUST_AVAILABLE:
        return compute_hash_rust(data)
    return compute_hash_python(data)


def compute_block_hash(
    timestamp: str,
    entropy_score: float,
    regime: str,
    action: str,
    prev_hash: str
) -> str:
    """
    Compute hash for a ledger block with all critical fields.
    
    This creates a cryptographic commitment to the block's state,
    making any tampering detectable.
    """
    # Use struct for deterministic float serialization
    entropy_bytes = struct.pack('>d', entropy_score)  # Big-endian double
    entropy_hex = entropy_bytes.hex()
    
    block_data = {
        'timestamp': timestamp,
        'entropy_score': entropy_hex,
        'regime': regime,
        'action': action,
        'prev_hash': prev_hash
    }
    block_string = json.dumps(block_data, sort_keys=True, separators=(',', ':'))
    return compute_hash(block_string)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LogEntry:
    """
    A single tamper-evident log entry.
    
    Each entry is cryptographically linked to the previous entry
    via hash chaining, creating an immutable audit trail.
    """
    sequence_num: int
    timestamp: str  # ISO 8601 format
    timestamp_ns: int  # Nanosecond precision for HFT forensics
    event_type: str  # TRADE, SIGNAL, ORDER, FILL, ERROR, etc.
    symbol: str
    data: Dict[str, Any]
    prev_hash: str
    entry_hash: str = ""
    
    def __post_init__(self):
        """Compute hash if not already set."""
        if not self.entry_hash:
            self.entry_hash = self.compute_hash()
    
    def compute_hash(self) -> str:
        """Compute SHA-256 hash of entry contents."""
        # Serialize data deterministically
        data_json = json.dumps(self.data, sort_keys=True, separators=(',', ':'))
        
        entry_data = {
            'sequence_num': self.sequence_num,
            'timestamp': self.timestamp,
            'timestamp_ns': self.timestamp_ns,
            'event_type': self.event_type,
            'symbol': self.symbol,
            'data': data_json,
            'prev_hash': self.prev_hash
        }
        
        entry_string = json.dumps(entry_data, sort_keys=True, separators=(',', ':'))
        return compute_hash(entry_string)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'sequence_num': self.sequence_num,
            'timestamp': self.timestamp,
            'timestamp_ns': self.timestamp_ns,
            'event_type': self.event_type,
            'symbol': self.symbol,
            'data': self.data,
            'prev_hash': self.prev_hash,
            'entry_hash': self.entry_hash
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'LogEntry':
        """Reconstruct LogEntry from dictionary."""
        return cls(
            sequence_num=d['sequence_num'],
            timestamp=d['timestamp'],
            timestamp_ns=d['timestamp_ns'],
            event_type=d['event_type'],
            symbol=d['symbol'],
            data=d['data'],
            prev_hash=d['prev_hash'],
            entry_hash=d.get('entry_hash', '')
        )


@dataclass
class AnchorRecord:
    """
    Record of external blockchain anchoring.
    
    Periodically commits the log chain's state to an external
    blockchain (e.g., Bitcoin OP_RETURN) for tamper evidence.
    """
    anchor_timestamp: str
    log_sequence_num: int  # Last sequence number anchored
    log_tail_hash: str  # Hash of last entry at anchor time
    merkle_root: str  # Merkle root of anchored entries
    external_txid: Optional[str] = None  # Blockchain transaction ID
    external_chain: str = "SIMULATED"  # BITCOIN, ETHEREUM, SIMULATED
    block_height: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'AnchorRecord':
        return cls(**d)


# =============================================================================
# MERKLE TREE IMPLEMENTATION
# =============================================================================

class MerkleTree:
    """
    Merkle tree for efficient verification of log entry inclusion.
    
    Allows proving that a specific log entry was part of an anchored
    batch without revealing the entire batch contents.
    """
    
    def __init__(self, hashes: List[str]):
        """Build Merkle tree from list of hashes."""
        if not hashes:
            self.root = compute_hash("EMPTY")
            self.levels = []
            return
        
        # Pad to power of 2
        n = len(hashes)
        next_pow2 = 1
        while next_pow2 < n:
            next_pow2 *= 2
        hashes = hashes + [hashes[-1]] * (next_pow2 - n)
        
        self.levels = [hashes]
        
        # Build tree bottom-up
        current_level = hashes
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                combined = current_level[i] + current_level[i + 1]
                next_level.append(compute_hash(combined))
            self.levels.append(next_level)
            current_level = next_level
        
        self.root = self.levels[-1][0] if self.levels else compute_hash("EMPTY")
    
    def get_proof(self, index: int) -> List[Tuple[str, str]]:
        """
        Get Merkle proof for entry at given index.
        
        Returns list of (hash, direction) tuples where direction
        is 'L' or 'R' indicating position of sibling.
        """
        if index < 0 or index >= len(self.levels[0]):
            raise IndexError("Index out of range")
        
        proof = []
        idx = index
        
        for level in self.levels[:-1]:
            if idx % 2 == 0:
                # Sibling is on right
                sibling_idx = idx + 1
                direction = 'R'
            else:
                # Sibling is on left
                sibling_idx = idx - 1
                direction = 'L'
            
            if sibling_idx < len(level):
                proof.append((level[sibling_idx], direction))
            
            idx //= 2
        
        return proof
    
    @staticmethod
    def verify_proof(
        leaf_hash: str,
        index: int,
        proof: List[Tuple[str, str]],
        expected_root: str
    ) -> bool:
        """Verify a Merkle proof."""
        current_hash = leaf_hash
        idx = index
        
        for sibling_hash, direction in proof:
            if direction == 'R':
                combined = current_hash + sibling_hash
            else:
                combined = sibling_hash + current_hash
            current_hash = compute_hash(combined)
            idx //= 2
        
        return current_hash == expected_root


# =============================================================================
# BLOCKCHAIN ANCHORING
# =============================================================================

class BlockchainAnchorer:
    """
    Anchors log chain state to external blockchain.
    
    Supports multiple backends:
    - SIMULATED: For testing/development
    - BITCOIN: OP_RETURN transactions
    - ETHEREUM: Smart contract events
    """
    
    def __init__(self, mode: str = "SIMULATED"):
        self.mode = mode
        self.anchor_history: List[AnchorRecord] = []
        
        # For Bitcoin anchoring (future implementation)
        self.bitcoin_rpc_url = None
        self.bitcoin_wallet = None
    
    def create_anchor(
        self,
        log_sequence_num: int,
        log_tail_hash: str,
        merkle_root: str
    ) -> AnchorRecord:
        """Create an anchor record."""
        anchor = AnchorRecord(
            anchor_timestamp=datetime.now(timezone.utc).isoformat(),
            log_sequence_num=log_sequence_num,
            log_tail_hash=log_tail_hash,
            merkle_root=merkle_root,
            external_chain=self.mode
        )
        
        if self.mode == "SIMULATED":
            # Simulate blockchain commitment
            anchor.external_txid = self._simulate_transaction(merkle_root)
            anchor.block_height = self._simulate_block_height()
        
        elif self.mode == "BITCOIN":
            # TODO: Implement actual Bitcoin OP_RETURN anchoring
            anchor.external_txid = self._anchor_to_bitcoin(merkle_root)
        
        self.anchor_history.append(anchor)
        return anchor
    
    def _simulate_transaction(self, data_hash: str) -> str:
        """Simulate a blockchain transaction ID."""
        # In production, this would be a real txid
        simulated_txid = compute_hash(f"SIMULATED_{data_hash}_{time.time()}")[:64]
        return simulated_txid
    
    def _simulate_block_height(self) -> int:
        """Simulate a block height."""
        # In production, this would be actual block height
        base_height = 850000  # Approximate current BTC block
        return base_height + int(time.time() - 1700000000) // 600
    
    def _anchor_to_bitcoin(self, data: str) -> Optional[str]:
        """
        Anchor data to Bitcoin blockchain via OP_RETURN.
        
        TODO: Implement with actual Bitcoin RPC integration.
        """
        print(f"[BlockchainAnchorer] Bitcoin anchoring not yet implemented")
        return None
    
    def verify_anchor(self, anchor: AnchorRecord) -> bool:
        """Verify an anchor record's integrity."""
        if anchor.external_chain == "SIMULATED":
            # Verify simulated txid format
            if not anchor.external_txid or len(anchor.external_txid) != 64:
                return False
            return True
        
        # TODO: Implement actual blockchain verification
        return False
    
    def get_latest_anchor(self) -> Optional[AnchorRecord]:
        """Get the most recent anchor record."""
        if not self.anchor_history:
            return None
        return self.anchor_history[-1]


# =============================================================================
# TAMPER DETECTION ENGINE
# =============================================================================

@dataclass
class IntegrityViolation:
    """Record of a detected integrity violation."""
    violation_type: str  # HASH_MISMATCH, SEQUENCE_GAP, TIMESTAMP_ANOMALY, etc.
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    sequence_num: int
    expected_hash: str
    actual_hash: str
    timestamp: str
    details: str
    remediation: str = ""


class TamperDetectionEngine:
    """
    Detects tampering attempts in the log chain.
    
    Monitors for:
    - Hash chain breaks
    - Sequence number gaps
    - Timestamp anomalies (out-of-order, future timestamps)
    - Entry modification
    - Deletion attempts
    """
    
    def __init__(self):
        self.violations: List[IntegrityViolation] = []
        self.max_timestamp_drift_ms = 1000  # 1 second allowed drift
    
    def verify_chain(self, entries: List[LogEntry], genesis_hash: str = "GENESIS") -> Tuple[bool, List[IntegrityViolation]]:
        """
        Verify integrity of entire log chain.

        Returns:
            (is_valid, list_of_violations)
        """
        violations = []

        if not entries:
            return True, []

        prev_hash = genesis_hash
        prev_seq = -1
        prev_timestamp_ns = 0
        
        for entry in entries:
            # 1. Verify entry's own hash
            computed_hash = entry.compute_hash()
            if computed_hash != entry.entry_hash:
                violations.append(IntegrityViolation(
                    violation_type="HASH_MISMATCH",
                    severity="CRITICAL",
                    sequence_num=entry.sequence_num,
                    expected_hash=computed_hash,
                    actual_hash=entry.entry_hash,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    details=f"Entry {entry.sequence_num} hash mismatch. Entry may have been modified.",
                    remediation="Restore from backup or investigate tampering"
                ))
            
            # 2. Verify hash chain linkage
            if entry.prev_hash != prev_hash:
                violations.append(IntegrityViolation(
                    violation_type="CHAIN_BREAK",
                    severity="CRITICAL",
                    sequence_num=entry.sequence_num,
                    expected_hash=prev_hash,
                    actual_hash=entry.prev_hash,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    details=f"Hash chain broken at sequence {entry.sequence_num}. Entries may have been inserted or deleted.",
                    remediation="Investigate chain break, restore from last valid anchor"
                ))
            
            # 3. Verify sequence continuity
            if entry.sequence_num != prev_seq + 1:
                violations.append(IntegrityViolation(
                    violation_type="SEQUENCE_GAP",
                    severity="HIGH",
                    sequence_num=entry.sequence_num,
                    expected_hash="N/A",
                    actual_hash="N/A",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    details=f"Sequence gap detected: expected {prev_seq + 1}, got {entry.sequence_num}",
                    remediation="Investigate missing entries"
                ))
            
            # 4. Verify timestamp ordering
            if entry.timestamp_ns < prev_timestamp_ns:
                violations.append(IntegrityViolation(
                    violation_type="TIMESTAMP_ANOMALY",
                    severity="MEDIUM",
                    sequence_num=entry.sequence_num,
                    expected_hash="N/A",
                    actual_hash="N/A",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    details=f"Out-of-order timestamp: entry {entry.sequence_num} has earlier timestamp than previous entry",
                    remediation="Check system clock synchronization"
                ))
            
            # 5. Check for future timestamps (clock skew)
            now_ns = int(time.time() * 1e9)
            if entry.timestamp_ns > now_ns + (self.max_timestamp_drift_ms * 1e6):
                violations.append(IntegrityViolation(
                    violation_type="FUTURE_TIMESTAMP",
                    severity="MEDIUM",
                    sequence_num=entry.sequence_num,
                    expected_hash="N/A",
                    actual_hash="N/A",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    details=f"Future timestamp detected: {entry.timestamp_ns}ns vs current {now_ns}ns",
                    remediation="Check system clock synchronization"
                ))
            
            prev_hash = entry.entry_hash
            prev_seq = entry.sequence_num
            prev_timestamp_ns = entry.timestamp_ns
        
        self.violations.extend(violations)
        return len(violations) == 0, violations
    
    def verify_entry_inclusion(
        self,
        entry: LogEntry,
        anchor: AnchorRecord,
        proof: List[Tuple[str, str]]
    ) -> bool:
        """
        Verify that an entry was part of an anchored batch.
        
        Uses Merkle proof to prove inclusion without revealing batch contents.
        """
        return MerkleTree.verify_proof(
            entry.entry_hash,
            entry.sequence_num,
            proof,
            anchor.merkle_root
        )
    
    def get_violation_summary(self) -> Dict:
        """Get summary of detected violations."""
        if not self.violations:
            return {'status': 'HEALTHY', 'violations': []}
        
        by_severity = {}
        by_type = {}
        
        for v in self.violations:
            by_severity[v.severity] = by_severity.get(v.severity, 0) + 1
            by_type[v.violation_type] = by_type.get(v.violation_type, 0) + 1
        
        return {
            'status': 'COMPROMISED',
            'total_violations': len(self.violations),
            'by_severity': by_severity,
            'by_type': by_type,
            'latest_violation': asdict(self.violations[-1]) if self.violations else None
        }


# =============================================================================
# MAIN LOG INTEGRITY MANAGER
# =============================================================================

class LogIntegrityManager:
    """
    Main interface for tamper-evident logging.
    
    Usage:
        manager = LogIntegrityManager(storage_path="logs/integrity_log.json")
        
        # Log an event
        manager.log_event(
            event_type="TRADE",
            symbol="BTC/USDT",
            data={"action": "BUY", "qty": 0.1, "price": 50000}
        )
        
        # Verify integrity
        is_valid, violations = manager.verify_integrity()
        
        # Create blockchain anchor
        anchor = manager.create_anchor()
    """
    
    def __init__(
        self,
        storage_path: str = "logs/integrity_log.json",
        anchor_mode: str = "SIMULATED",
        auto_anchor_interval: int = 1000,  # Anchor every N entries
        enable_tamper_detection: bool = True
    ):
        self.storage_path = storage_path
        self.auto_anchor_interval = auto_anchor_interval
        self.enable_tamper_detection = enable_tamper_detection
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize components
        self.tamper_engine = TamperDetectionEngine() if enable_tamper_detection else None
        self.anchorer = BlockchainAnchorer(mode=anchor_mode)
        
        # Log state
        self.entries: List[LogEntry] = []
        self.sequence_counter = 0
        self.last_anchor_seq = 0
        
        # Genesis entry
        self._genesis_hash = compute_hash("HOLONIC_TRADER_GENESIS_" + str(time.time()))

        # Load existing log
        self._load_log()

    def _load_log(self):
        """Load existing log from disk."""
        if not os.path.exists(self.storage_path):
            print(f"[LogIntegrity] Creating new integrity log at {self.storage_path}")
            self._save_log()
            return

        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.entries = [LogEntry.from_dict(e) for e in data.get('entries', [])]
            self.sequence_counter = data.get('sequence_counter', len(self.entries))
            self.last_anchor_seq = data.get('last_anchor_seq', 0)

            # Load genesis hash if persisted (for chain verification continuity)
            if 'genesis_hash' in data:
                self._genesis_hash = data['genesis_hash']

            # Load anchor history
            for a in data.get('anchor_history', []):
                self.anchorer.anchor_history.append(AnchorRecord.from_dict(a))

            # Update anchor's last_anchor_seq if anchors exist
            if self.anchorer.anchor_history:
                self.last_anchor_seq = self.anchorer.anchor_history[-1].log_sequence_num + 1
            
            print(f"[LogIntegrity] Loaded {len(self.entries)} entries from disk")

            # Verify integrity on load
            if self.enable_tamper_detection and self.entries:
                is_valid, violations = self.tamper_engine.verify_chain(self.entries, genesis_hash=self._genesis_hash)
                if not is_valid:
                    print(f"[LogIntegrity] INTEGRITY VIOLATION DETECTED ON LOAD!")
                    for v in violations:
                        print(f"  - {v.severity}: {v.violation_type} at seq {v.sequence_num}")
                else:
                    print(f"[LogIntegrity] Log integrity verified")
        
        except Exception as e:
            print(f"[LogIntegrity] Failed to load log: {e}")
            # Start fresh on error
            self.entries = []
            self.sequence_counter = 0
    
    def _save_log(self):
        """Persist log to disk."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

        data = {
            'entries': [e.to_dict() for e in self.entries],
            'sequence_counter': self.sequence_counter,
            'last_anchor_seq': self.last_anchor_seq,
            'anchor_history': [a.to_dict() for a in self.anchorer.anchor_history],
            'genesis_hash': self._genesis_hash,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }

        # Write atomically (write to temp, then rename)
        temp_path = self.storage_path + '.tmp'
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        os.replace(temp_path, self.storage_path)
    
    def log_event(
        self,
        event_type: str,
        symbol: str,
        data: Dict[str, Any],
        timestamp_ns: Optional[int] = None
    ) -> LogEntry:
        """
        Log a tamper-evident event.
        
        Args:
            event_type: Type of event (TRADE, SIGNAL, ORDER, FILL, ERROR, etc.)
            symbol: Asset symbol
            data: Event-specific data
            timestamp_ns: Nanosecond timestamp (auto-generated if not provided)
        
        Returns:
            The created LogEntry
        """
        with self.lock:
            now_ns = timestamp_ns if timestamp_ns else int(time.time() * 1e9)
            now_iso = datetime.now(timezone.utc).isoformat()
            
            # Get previous hash
            prev_hash = self.entries[-1].entry_hash if self.entries else self._genesis_hash
            
            # Create entry
            entry = LogEntry(
                sequence_num=self.sequence_counter,
                timestamp=now_iso,
                timestamp_ns=now_ns,
                event_type=event_type,
                symbol=symbol,
                data=data,
                prev_hash=prev_hash
            )
            
            # Add to chain
            self.entries.append(entry)
            self.sequence_counter += 1
            
            # Auto-anchor if needed
            if (self.sequence_counter - self.last_anchor_seq) >= self.auto_anchor_interval:
                self.create_anchor()
            
            # Persist
            self._save_log()
            
            return entry
    
    def log_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        order_id: str,
        metadata: Optional[Dict] = None
    ) -> LogEntry:
        """Log a trade execution."""
        data = {
            'action': action,
            'quantity': quantity,
            'price': price,
            'order_id': order_id,
            **(metadata or {})
        }
        return self.log_event("TRADE", symbol, data)
    
    def log_signal(
        self,
        symbol: str,
        signal_type: str,
        conviction: float,
        strategy: str,
        metadata: Optional[Dict] = None
    ) -> LogEntry:
        """Log a trading signal."""
        data = {
            'signal_type': signal_type,
            'conviction': conviction,
            'strategy': strategy,
            **(metadata or {})
        }
        return self.log_event("SIGNAL", symbol, data)
    
    def log_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_id: Optional[str] = None
    ) -> LogEntry:
        """Log an order placement."""
        data = {
            'order_type': order_type,
            'side': side,
            'quantity': quantity,
            'price': price,
            'order_id': order_id,
        }
        return self.log_event("ORDER", symbol, data)
    
    def log_error(
        self,
        error_type: str,
        message: str,
        symbol: str = "SYSTEM",
        traceback: Optional[str] = None
    ) -> LogEntry:
        """Log an error event."""
        data = {
            'error_type': error_type,
            'message': message,
            'traceback': traceback
        }
        return self.log_event("ERROR", symbol, data)
    
    def create_anchor(self) -> Optional[AnchorRecord]:
        """
        Create a blockchain anchor for current log state.
        
        Returns:
            AnchorRecord if successful, None otherwise
        """
        with self.lock:
            if not self.entries:
                return None
            
            # Compute Merkle root of all entries since last anchor
            anchor_start = self.last_anchor_seq
            hashes_to_anchor = [e.entry_hash for e in self.entries[anchor_start:]]
            
            if not hashes_to_anchor:
                return None
            
            merkle_tree = MerkleTree(hashes_to_anchor)
            tail_hash = self.entries[-1].entry_hash
            
            anchor = self.anchorer.create_anchor(
                log_sequence_num=self.sequence_counter - 1,
                log_tail_hash=tail_hash,
                merkle_root=merkle_tree.root
            )
            
            self.last_anchor_seq = self.sequence_counter
            self._save_log()
            
            print(f"[LogIntegrity] Created anchor at seq {anchor.log_sequence_num}")
            print(f"   Merkle Root: {anchor.merkle_root[:16]}...")
            print(f"   TxID: {anchor.external_txid[:16] if anchor.external_txid else 'N/A'}...")
            
            return anchor
    
    def verify_integrity(self) -> Tuple[bool, List[IntegrityViolation]]:
        """
        Verify integrity of entire log chain.

        Returns:
            (is_valid, list_of_violations)
        """
        if not self.enable_tamper_detection:
            return True, []

        with self.lock:
            return self.tamper_engine.verify_chain(self.entries, genesis_hash=self._genesis_hash)
    
    def verify_entry(
        self,
        sequence_num: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify a specific entry's integrity.
        
        Returns:
            (is_valid, error_message)
        """
        if sequence_num < 0 or sequence_num >= len(self.entries):
            return False, f"Sequence {sequence_num} not found"
        
        entry = self.entries[sequence_num]
        
        # Verify entry hash
        computed = entry.compute_hash()
        if computed != entry.entry_hash:
            return False, f"Hash mismatch at seq {sequence_num}"
        
        # Verify chain linkage
        expected_prev = self.entries[sequence_num - 1].entry_hash if sequence_num > 0 else self._genesis_hash
        if entry.prev_hash != expected_prev:
            return False, f"Chain break at seq {sequence_num}"
        
        return True, None
    
    def get_entry(self, sequence_num: int) -> Optional[LogEntry]:
        """Get entry by sequence number."""
        if 0 <= sequence_num < len(self.entries):
            return self.entries[sequence_num]
        return None
    
    def get_entries_by_type(
        self,
        event_type: str,
        symbol: Optional[str] = None
    ) -> List[LogEntry]:
        """Get all entries of a specific type."""
        entries = [e for e in self.entries if e.event_type == event_type]
        if symbol:
            entries = [e for e in entries if e.symbol == symbol]
        return entries
    
    def get_integrity_report(self) -> Dict:
        """Generate comprehensive integrity report."""
        with self.lock:
            is_valid, violations = self.verify_integrity()
            
            report = {
                'status': 'HEALTHY' if is_valid else 'COMPROMISED',
                'total_entries': len(self.entries),
                'sequence_range': {
                    'first': 0 if self.entries else None,
                    'last': len(self.entries) - 1 if self.entries else None
                },
                'anchors': len(self.anchorer.anchor_history),
                'last_anchor_seq': self.last_anchor_seq,
                'violations': [asdict(v) for v in violations] if violations else [],
                'tamper_detection_enabled': self.enable_tamper_detection
            }
            
            if self.anchorer.anchor_history:
                latest = self.anchorer.anchor_history[-1]
                report['latest_anchor'] = {
                    'timestamp': latest.anchor_timestamp,
                    'sequence_num': latest.log_sequence_num,
                    'txid': latest.external_txid,
                    'chain': latest.external_chain
                }
            
            return report
    
    def export_for_audit(
        self,
        output_path: str,
        include_proofs: bool = True
    ) -> str:
        """
        Export log for external audit.
        
        Includes Merkle proofs for entry verification.
        """
        with self.lock:
            export_data = {
                'export_timestamp': datetime.now(timezone.utc).isoformat(),
                'integrity_report': self.get_integrity_report(),
                'entries': [e.to_dict() for e in self.entries],
                'anchors': [a.to_dict() for a in self.anchorer.anchor_history]
            }
            
            if include_proofs and self.entries:
                # Add Merkle proofs for each anchor batch
                proofs = []
                for i, anchor in enumerate(self.anchorer.anchor_history):
                    start_seq = self.anchorer.anchor_history[i-1].log_sequence_num + 1 if i > 0 else 0
                    end_seq = anchor.log_sequence_num
                    
                    hashes = [self.entries[s].entry_hash for s in range(start_seq, end_seq + 1)]
                    merkle_tree = MerkleTree(hashes)
                    
                    for seq in range(start_seq, end_seq + 1):
                        idx = seq - start_seq
                        proof = merkle_tree.get_proof(idx)
                        proofs.append({
                            'sequence_num': seq,
                            'anchor_seq': anchor.log_sequence_num,
                            'proof': proof
                        })
                
                export_data['merkle_proofs'] = proofs
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            print(f"[LogIntegrity] Exported audit data to {output_path}")
            return output_path


# =============================================================================
# INTEGRATION WITH EXISTING EXECUTOR LEDGER
# =============================================================================

class ExecutorLedgerIntegrator:
    """
    Integrates LogIntegrityManager with existing ExecutorHolon AuditLedger.
    
    Wraps the existing ledger to add tamper-evident logging.
    """
    
    def __init__(
        self,
        executor_holon,
        integrity_manager: Optional[LogIntegrityManager] = None
    ):
        self.executor = executor_holon
        self.integrity_manager = integrity_manager or LogIntegrityManager()
    
    def log_trade_decision(
        self,
        entropy_score: float,
        regime: str,
        action: str,
        block_hash: str
    ) -> LogEntry:
        """Log a trade decision from ExecutorHolon."""
        return self.integrity_manager.log_event(
            event_type="DECISION",
            symbol="SYSTEM",
            data={
                'entropy_score': entropy_score,
                'regime': regime,
                'action': action,
                'block_hash': block_hash
            }
        )
    
    def log_position_open(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        entry_price: float,
        leverage: float,
        order_id: str
    ) -> LogEntry:
        """Log a position opening."""
        return self.integrity_manager.log_trade(
            symbol=symbol,
            action="OPEN_" + direction,
            quantity=quantity,
            price=entry_price,
            order_id=order_id,
            metadata={'leverage': leverage}
        )
    
    def log_position_close(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        exit_price: float,
        pnl_usd: float,
        pnl_pct: float,
        order_id: str
    ) -> LogEntry:
        """Log a position closing."""
        return self.integrity_manager.log_trade(
            symbol=symbol,
            action="CLOSE_" + direction,
            quantity=quantity,
            price=exit_price,
            order_id=order_id,
            metadata={
                'pnl_usd': pnl_usd,
                'pnl_pct': pnl_pct
            }
        )
    
    def log_ghost_detection(
        self,
        symbol: str,
        exchange_qty: float,
        ledgerQty: float,
        action: str
    ) -> LogEntry:
        """Log a ghost/leak detection event."""
        return self.integrity_manager.log_event(
            event_type="GHOST_DETECTED",
            symbol=symbol,
            data={
                'exchange_qty': exchange_qty,
                'ledger_qty': ledgerQty,
                'discrepancy': abs(exchange_qty - ledgerQty),
                'action_taken': action
            }
        )
    
    def log_equity_divergence(
        self,
        internal_equity: float,
        exchange_equity: float,
        divergence_pct: float
    ) -> LogEntry:
        """Log an equity divergence event."""
        return self.integrity_manager.log_event(
            event_type="EQUITY_DIVERGENCE",
            symbol="SYSTEM",
            data={
                'internal_equity': internal_equity,
                'exchange_equity': exchange_equity,
                'divergence_pct': divergence_pct,
                'timestamp': time.time()
            }
        )
    
    def verify_execution_integrity(self) -> Dict:
        """Verify integrity of execution logs."""
        return self.integrity_manager.get_integrity_report()


# =============================================================================
# TELEGRAM ALERT INTEGRATION
# =============================================================================

class IntegrityAlertHandler:
    """
    Sends Telegram alerts on integrity violations.
    """
    
    def __init__(
        self,
        integrity_manager: LogIntegrityManager,
        telegram_bot,
        chat_id: str
    ):
        self.integrity_manager = integrity_manager
        self.telegram_bot = telegram_bot
        self.chat_id = chat_id
        self.last_alert_time = 0
        self.alert_cooldown_sec = 60  # Prevent alert spam
    
    def check_and_alert(self) -> List[IntegrityViolation]:
        """Check integrity and send alerts for new violations."""
        now = time.time()
        if now - self.last_alert_time < self.alert_cooldown_sec:
            return []
        
        is_valid, violations = self.integrity_manager.verify_integrity()
        
        if violations:
            for v in violations:
                self._send_alert(v)
            self.last_alert_time = now
        
        return violations
    
    def _send_alert(self, violation: IntegrityViolation):
        """Send Telegram alert for a violation."""
        emoji = {
            'CRITICAL': '[CRIT]',
            'HIGH': '[HIGH]',
            'MEDIUM': '[MED]',
            'LOW': '[LOW]'
        }.get(violation.severity, '📢')
        
        message = f"""
{emoji} *LOG INTEGRITY ALERT*

*Severity:* {violation.severity}
*Type:* {violation.violation_type}
*Sequence:* {violation.sequence_num}
*Time:* {violation.timestamp}

*Details:*
{violation.details}

*Remediation:*
{violation.remediation}
"""
        
        try:
            if self.telegram_bot:
                self.telegram_bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode='Markdown'
                )
        except Exception as e:
            print(f"[IntegrityAlert] Failed to send Telegram alert: {e}")


# =============================================================================
# CLI VERIFICATION TOOL
# =============================================================================

def verify_log_cli(storage_path: str = "logs/integrity_log.json") -> int:
    """
    CLI tool for verifying log integrity.
    
    Usage:
        python -m HolonicTrader.log_integrity verify --path logs/integrity_log.json
    
    Returns:
        0 if valid, 1 if compromised
    """
    print("=" * 60)
    print("AEGIS QUANTSEC - LOG INTEGRITY VERIFIER")
    print("=" * 60)
    print()
    
    if not os.path.exists(storage_path):
        print(f"Log file not found: {storage_path}")
        return 1

    manager = LogIntegrityManager(storage_path=storage_path)
    report = manager.get_integrity_report()

    print(f"Log Status: {report['status']}")
    print(f"Total Entries: {report['total_entries']}")
    print(f"Anchors: {report['anchors']}")
    
    if report.get('latest_anchor'):
        anchor = report['latest_anchor']
        print(f"Last Anchor: Seq {anchor['sequence_num']}")
        print(f"   TxID: {anchor['txid'][:32] if anchor['txid'] else 'N/A'}...")

    print()

    if report['violations']:
        print(f"VIOLATIONS DETECTED: {len(report['violations'])}")
        print()
        for v in report['violations']:
            print(f"  {v['severity']}: {v['violation_type']} at seq {v['sequence_num']}")
            print(f"    {v['details']}")
        return 1
    else:
        print("No integrity violations detected")
        return 0


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        path = sys.argv[2] if len(sys.argv) > 2 else "logs/integrity_log.json"
        exit_code = verify_log_cli(path)
        sys.exit(exit_code)
    else:
        # Demo mode
        print("AEGIS QUANTSEC Log Integrity Engine")
        print()
        print("Usage:")
        print("  python -m HolonicTrader.log_integrity verify [path]")
        print()
        print("Programmatic usage:")
        print("  from HolonicTrader.log_integrity import LogIntegrityManager")
        print("  manager = LogIntegrityManager()")
        print("  manager.log_trade('BTC/USDT', 'BUY', 0.1, 50000, 'order123')")
        print("  is_valid, violations = manager.verify_integrity()")
