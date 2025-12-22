"""
Tests for ExecutorHolon - The Executor Agent (Phase 4)

Tests cover:
1. Ledger integrity (hash chain validation)
2. Correct disposition setting per regime
3. Correct action returned per regime
4. Block addition verification
"""

import pytest
from agent_executor import ExecutorHolon, TradeSignal, TradeDecision


class TestAuditLedger:
    """Tests for the AuditLedger pseudo-blockchain."""

    def test_ledger_initialization(self):
        """Ledger should start empty."""
        executor = ExecutorHolon()
        assert len(executor.ledger) == 0
        assert executor.ledger.chain == []

    def test_genesis_block_prev_hash(self):
        """First block should have prev_hash = '0'."""
        executor = ExecutorHolon()
        block = executor.ledger.add_block(
            entropy_score=2.0,
            regime='ORDERED',
            action='EXECUTE'
        )
        assert block.prev_hash == "0"

    def test_block_chaining(self):
        """Each block should reference the previous block's hash."""
        executor = ExecutorHolon()
        
        # Add first block
        block1 = executor.ledger.add_block(
            entropy_score=2.0,
            regime='ORDERED',
            action='EXECUTE'
        )
        
        # Add second block
        block2 = executor.ledger.add_block(
            entropy_score=4.0,
            regime='CHAOTIC',
            action='HALT'
        )
        
        # Verify chain linkage
        assert block2.prev_hash == block1.hash

    def test_block_hash_computation(self):
        """Block hash should be deterministic based on contents."""
        executor = ExecutorHolon()
        block = executor.ledger.add_block(
            entropy_score=3.0,
            regime='TRANSITION',
            action='REDUCE'
        )
        
        # Hash should be a 64-character hex string (SHA-256)
        assert len(block.hash) == 64
        assert all(c in '0123456789abcdef' for c in block.hash)
        
        # Recomputing should give same result
        assert block.hash == block.compute_hash()

    def test_chain_verification_valid(self):
        """A properly constructed chain should verify as valid."""
        executor = ExecutorHolon()
        
        # Add multiple blocks
        executor.ledger.add_block(2.0, 'ORDERED', 'EXECUTE')
        executor.ledger.add_block(4.0, 'CHAOTIC', 'HALT')
        executor.ledger.add_block(3.0, 'TRANSITION', 'REDUCE')
        
        assert executor.ledger.verify_chain() is True

    def test_chain_verification_empty(self):
        """Empty chain should be considered valid."""
        executor = ExecutorHolon()
        assert executor.ledger.verify_chain() is True

    def test_chain_verification_tampered_hash(self):
        """Tampering with a block's hash should invalidate the chain."""
        executor = ExecutorHolon()
        
        executor.ledger.add_block(2.0, 'ORDERED', 'EXECUTE')
        executor.ledger.add_block(4.0, 'CHAOTIC', 'HALT')
        
        # Tamper with the first block's hash
        executor.ledger._chain[0].hash = "tampered_hash"
        
        assert executor.ledger.verify_chain() is False


class TestExecutorDispositionLogic:
    """Tests for disposition setting based on regime."""

    def test_chaotic_regime_disposition(self):
        """CHAOTIC regime should set high integration, low autonomy."""
        executor = ExecutorHolon()
        signal = TradeSignal(direction='BUY', size=100, price=50000)
        
        result = executor.decide_trade(signal, 'CHAOTIC', 4.5)
        
        assert executor.disposition.autonomy == 0.1
        assert executor.disposition.integration == 0.9

    def test_ordered_regime_disposition(self):
        """ORDERED regime should set high autonomy, low integration."""
        executor = ExecutorHolon()
        signal = TradeSignal(direction='BUY', size=100, price=50000)
        
        result = executor.decide_trade(signal, 'ORDERED', 2.0)
        
        assert executor.disposition.autonomy == 0.9
        assert executor.disposition.integration == 0.1

    def test_transition_regime_disposition(self):
        """TRANSITION regime should set balanced autonomy and integration."""
        executor = ExecutorHolon()
        signal = TradeSignal(direction='BUY', size=100, price=50000)
        
        result = executor.decide_trade(signal, 'TRANSITION', 3.0)
        
        assert executor.disposition.autonomy == 0.5
        assert executor.disposition.integration == 0.5


class TestExecutorTradeActions:
    """Tests for trade actions based on regime."""

    def test_chaotic_regime_halt(self):
        """CHAOTIC regime should HALT trade and set size to 0."""
        executor = ExecutorHolon()
        signal = TradeSignal(direction='BUY', size=100, price=50000)
        
        result = executor.decide_trade(signal, 'CHAOTIC', 4.5)
        
        assert result.action == 'HALT'
        assert result.adjusted_size == 0.0

    def test_ordered_regime_execute(self):
        """ORDERED regime should EXECUTE trade at full size."""
        executor = ExecutorHolon()
        signal = TradeSignal(direction='SELL', size=50, price=48000)
        
        result = executor.decide_trade(signal, 'ORDERED', 2.0)
        
        assert result.action == 'EXECUTE'
        assert result.adjusted_size == 50

    def test_transition_regime_reduce(self):
        """TRANSITION regime should REDUCE trade size by 50%."""
        executor = ExecutorHolon()
        signal = TradeSignal(direction='BUY', size=100, price=50000)
        
        result = executor.decide_trade(signal, 'TRANSITION', 3.0)
        
        assert result.action == 'REDUCE'
        assert result.adjusted_size == 50.0  # 100 * 0.5


class TestExecutorLedgerIntegration:
    """Tests for ledger logging on trade decisions."""

    def test_decision_logged_to_ledger(self):
        """Every trade decision should be logged to the ledger."""
        executor = ExecutorHolon()
        signal = TradeSignal(direction='BUY', size=100, price=50000)
        
        # Make a decision
        result = executor.decide_trade(signal, 'ORDERED', 2.0)
        
        # Verify block was added
        assert len(executor.ledger) == 1
        
        # Verify block contents
        block = executor.ledger.chain[0]
        assert block.entropy_score == 2.0
        assert block.regime == 'ORDERED'
        assert block.action == 'EXECUTE'

    def test_multiple_decisions_logged(self):
        """Multiple decisions should create a valid chain."""
        executor = ExecutorHolon()
        signal = TradeSignal(direction='BUY', size=100, price=50000)
        
        # Make multiple decisions
        executor.decide_trade(signal, 'ORDERED', 2.0)
        executor.decide_trade(signal, 'CHAOTIC', 4.5)
        executor.decide_trade(signal, 'TRANSITION', 3.0)
        
        assert len(executor.ledger) == 3
        assert executor.ledger.verify_chain() is True

    def test_decision_returns_block_hash(self):
        """Trade decision should return the block hash for verification."""
        executor = ExecutorHolon()
        signal = TradeSignal(direction='BUY', size=100, price=50000)
        
        result = executor.decide_trade(signal, 'ORDERED', 2.0)
        
        # Result should contain the block hash
        assert result.block_hash is not None
        assert len(result.block_hash) == 64
        
        # Hash should match the ledger's last block
        assert result.block_hash == executor.ledger.chain[-1].hash


class TestExecutorLedgerSummary:
    """Tests for the ledger summary functionality."""

    def test_ledger_summary_empty(self):
        """Empty ledger summary should show zero blocks."""
        executor = ExecutorHolon()
        summary = executor.get_ledger_summary()
        
        assert summary['total_blocks'] == 0
        assert summary['chain_valid'] is True
        assert summary['last_block'] is None

    def test_ledger_summary_with_blocks(self):
        """Summary should reflect ledger state after trades."""
        executor = ExecutorHolon()
        signal = TradeSignal(direction='BUY', size=100, price=50000)
        
        executor.decide_trade(signal, 'ORDERED', 2.0)
        executor.decide_trade(signal, 'CHAOTIC', 4.5)
        
        summary = executor.get_ledger_summary()
        
        assert summary['total_blocks'] == 2
        assert summary['chain_valid'] is True
        assert summary['last_block']['action'] == 'HALT'
        assert summary['last_block']['regime'] == 'CHAOTIC'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
