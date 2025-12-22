"""
ExecutorHolon - The Executor Agent (Phase 4)

This agent acts as the 'trade executor' of the AEHML core.
It executes trades only if the Disposition allows it and maintains
a local pseudo-blockchain ledger for audit purposes.

Key Features:
1. Pseudo-Blockchain Ledger (AuditLedger) with SHA-256 hashing
2. Disposition-based trade execution logic
"""

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Literal, List, Optional
import math

from HolonicTrader.holon_core import Holon, Disposition


@dataclass
class TradeSignal:
    """
    Represents a trading signal.
    """
    symbol: str
    direction: Literal['BUY', 'SELL']
    size: float
    price: float


@dataclass
class TradeDecision:
    """
    Represents the result of a trade decision.
    """
    action: Literal['EXECUTE', 'HALT', 'REDUCE']
    original_signal: TradeSignal
    adjusted_size: float
    disposition: Disposition
    block_hash: str


class ExecutorHolon(Holon):
    """
    ExecutorHolon is the 'Executor' that decides whether to execute trades
    based on market regime and entropy levels. All decisions are logged
    to a tamper-evident pseudo-blockchain ledger.
    """

    @dataclass
    class LedgerBlock:
        """
        A single block in the audit ledger.
        """
        timestamp: str
        entropy_score: float
        regime: Literal['ORDERED', 'CHAOTIC', 'TRANSITION']
        action: Literal['EXECUTE', 'HALT', 'REDUCE']
        prev_hash: str
        hash: str = ""

        def compute_hash(self) -> str:
            """
            Compute SHA-256 hash of the block's contents (excluding current hash).
            """
            block_data = {
                'timestamp': self.timestamp,
                'entropy_score': self.entropy_score,
                'regime': self.regime,
                'action': self.action,
                'prev_hash': self.prev_hash
            }
            block_string = json.dumps(block_data, sort_keys=True)
            return hashlib.sha256(block_string.encode()).hexdigest()

    class AuditLedger:
        """
        A pseudo-blockchain ledger for maintaining an immutable audit trail
        of all trading decisions.
        
        Each block is chained to the previous via SHA-256 hashing.
        """

        def __init__(self):
            self._chain: List['ExecutorHolon.LedgerBlock'] = []

        @property
        def chain(self) -> List['ExecutorHolon.LedgerBlock']:
            """Return the chain as a read-only property."""
            return list(self._chain)

        def add_block(
            self,
            entropy_score: float,
            regime: Literal['ORDERED', 'CHAOTIC', 'TRANSITION'],
            action: Literal['EXECUTE', 'HALT', 'REDUCE']
        ) -> 'ExecutorHolon.LedgerBlock':
            """
            Add a new block to the ledger.
            
            Args:
                entropy_score: The entropy value at decision time
                regime: The market regime (ORDERED, CHAOTIC, TRANSITION)
                action: The action taken (EXECUTE, HALT, REDUCE)
                
            Returns:
                The newly created and added block
            """
            # Get previous hash (genesis block uses "0")
            prev_hash = self._chain[-1].hash if self._chain else "0"

            # Create new block
            block = ExecutorHolon.LedgerBlock(
                timestamp=datetime.now(timezone.utc).isoformat(),
                entropy_score=entropy_score,
                regime=regime,
                action=action,
                prev_hash=prev_hash
            )

            # Compute and set the hash
            block.hash = block.compute_hash()

            # Add to chain
            self._chain.append(block)

            return block

        def verify_chain(self) -> bool:
            """
            Verify the integrity of the entire chain.
            
            Returns:
                True if the chain is valid, False otherwise
            """
            if not self._chain:
                return True

            # Check genesis block
            if self._chain[0].prev_hash != "0":
                return False

            # Verify each block
            for i, block in enumerate(self._chain):
                # Verify block's own hash
                if block.hash != block.compute_hash():
                    return False

                # Verify chain linkage (skip genesis)
                if i > 0 and block.prev_hash != self._chain[i - 1].hash:
                    return False

            return True

        def __len__(self) -> int:
            return len(self._chain)

    def __init__(
        self, 
        name: str = "ExecutorAgent", 
        initial_capital: float = 10.0,
        use_compounding: bool = True,
        fixed_stake: float = 10.0,
        db_manager: Any = None,
        governor: Any = None,
        actuator: Any = None
    ):
        """
        Initialize the ExecutorHolon with a neutral disposition and starting capital.
        
        Args:
            name: Agent name
            initial_capital: Starting balance in USD
            use_compounding: If True, uses % of portfolio. If False, uses fixed_stake.
            fixed_stake: Amount in USD to bet per trade if use_compounding is False.
            db_manager: Optional DatabaseManager instance for persistence.
            governor: Optional GovernorHolon instance for risk management.
            actuator: Optional ActuatorHolon instance for execution.
        """
        # Start with balanced disposition; will be adjusted per trade decision
        default_disposition = Disposition(autonomy=0.5, integration=0.5)
        super().__init__(name=name, disposition=default_disposition)
        
        # Initialize the audit ledger
        self.ledger = self.AuditLedger()
        
        # Persistence & Risk & Execution
        self.db_manager = db_manager
        self.governor = governor
        self.actuator = actuator
        
        # Portfolio Management
        self.initial_capital = initial_capital
        self.balance_usd = initial_capital
        
        # Multi-Asset Tracking
        self.held_assets = {}   # symbol -> quantity
        self.entry_prices = {}  # symbol -> entry_price (for SL/TP)
        self.latest_prices = {} # symbol -> last_seen_price (for valuation)
        self.position_metadata = {} # symbol -> {'leverage': float}
        
        # Stop-Loss / Take-Profit Parameters (Widened for more breathing room)
        self.stop_loss_pct = 0.03    # 3% below entry (was 2%)
        self.take_profit_pct = 0.05  # 5% above entry (was 3%)
        
        # Sizing Strategy
        self.use_compounding = use_compounding
        self.fixed_stake = fixed_stake
        
        # Load state from DB if available
        if self.db_manager:
            self._load_state()

    def _persist_portfolio(self):
        """Helper to save current balance and assets to DB."""
        if self.db_manager:
            self.db_manager.save_portfolio(self.balance_usd, self.held_assets, self.position_metadata)

    def _load_state(self):
        """Load portfolio and ledger state from database."""
        # Load Portfolio
        portfolio = self.db_manager.get_portfolio()
        if portfolio:
            self.balance_usd = portfolio['balance_usd']
            self.held_assets = portfolio.get('held_assets', {})
            self.position_metadata = portfolio.get('position_metadata', {})
            
            # Restore entry_prices from metadata
            for sym, meta in self.position_metadata.items():
                if 'entry_price' in meta:
                    self.entry_prices[sym] = meta['entry_price']
            
            # Reconstruct entry prices from metadata if possible, or leave empty
            # Note: A real system would persist entry prices perfectly. 
            # For now, we accept risk of losing SL/TP reference on hard restart if not in metadata.
            # Improvement: Store entry_prices in metadata?
            # Doing a quick fix: If we have position but no entry price, assume current market price on next tick to avoid crash,
            # but ideally we should persist entry_prices too.
            # Let's add entry_prices to save_portfolio next time, but for now held_assets is key.
            
            print(f"[{self.name}] Loaded portfolio: ${self.balance_usd:.2f} USD")
            print(f"[{self.name}] Assets: {json.dumps(self.held_assets)}")
        
        # Load Last Block
        last_block = self.db_manager.get_last_block()
        if last_block:
            restored_block = self.LedgerBlock(
                timestamp=last_block['timestamp'],
                entropy_score=last_block['entropy_score'],
                regime=last_block['regime'],
                action=last_block['action'],
                prev_hash=last_block['prev_hash'],
                hash=last_block['hash']
            )
            self.ledger._chain.append(restored_block)
            print(f"[{self.name}] Restored ledger tip: {restored_block.hash[:8]}...")



    def check_stop_loss_take_profit(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Check if current price triggers Stop-Loss or Take-Profit for a specific symbol.
        """
        entry_price = self.entry_prices.get(symbol)
        qty = self.held_assets.get(symbol, 0.0)
        
        if entry_price is None or qty <= 0:
            return None
        
        # Calculate price change from entry
        price_change_pct = (current_price - entry_price) / entry_price
        
        # Stop-Loss triggered (price dropped below threshold)
        if price_change_pct <= -self.stop_loss_pct:
            print(f"[{self.name}] {symbol} STOP-LOSS triggered at {price_change_pct*100:.2f}%")
            return 'STOP_LOSS'
        
        # Take-Profit triggered (price rose above threshold)
        if price_change_pct >= self.take_profit_pct:
            print(f"[{self.name}] {symbol} TAKE-PROFIT triggered at {price_change_pct*100:.2f}%")
            return 'TAKE_PROFIT'
        
        return None

    def decide_trade(
        self,
        signal: TradeSignal,
        current_regime: Literal['ORDERED', 'CHAOTIC', 'TRANSITION'],
        entropy_score: float
    ) -> TradeDecision:
        """
        Decide whether to execute a trade based on market regime and entropy.
        
        Disposition Logic:
            - CHAOTIC: Autonomy=0.1, Integration=0.9 → HALT (reject trade)
            - ORDERED: Autonomy=0.9, Integration=0.1 → EXECUTE (full trade)
            - TRANSITION: Autonomy=0.5, Integration=0.5 → REDUCE (50% size)
        
        Constraint: The decision is ALWAYS hashed and saved to the ledger
        before returning.
        
        Args:
            signal: The trade signal to evaluate
            current_regime: The current market regime
            entropy_score: The current entropy score
            
        Returns:
            TradeDecision containing action, adjusted size, and block hash
        """
        action: Literal['EXECUTE', 'HALT', 'REDUCE']
        adjusted_size: float

        # Apply disposition logic based on continuous Sigmoid function
        # Autonomy = 1 / (1 + e^(k * (Entropy - Threshold)))
        # k=5 (steepness), Threshold=2.0 (Chaos inflection point)
        k = 5.0
        threshold = 2.0
        
        # Calculate continuous autonomy (0.0 to 1.0)
        autonomy = 1.0 / (1.0 + math.exp(k * (entropy_score - threshold)))
        
        # Integration is the inverse
        integration = 1.0 - autonomy
        
        self.disposition = Disposition(autonomy=autonomy, integration=integration)
        
        # Map continuous autonomy to discrete actions for Ledger/Protocol compliance
        # Autonomy > 0.6 -> EXECUTE (High Independence)
        # Autonomy < 0.4 -> HALT (High Safety)
        # 0.4 <= Autonomy <= 0.6 -> REDUCE (Balanced)
        
        if autonomy > 0.6:
            action = 'EXECUTE'
            adjusted_size = signal.size * autonomy  # Scale size by confidence/autonomy? Or full? 
            # Let's scale slightly by autonomy to reflect "conviction"
            # But kept close to 1.0 for high autonomy.
            # Actually, let's keep it simple: EXECUTE = Full Size
            adjusted_size = signal.size
            
        elif autonomy < 0.4:
            action = 'HALT'
            adjusted_size = 0.0
            
        else:
            action = 'REDUCE'
            # Proportional reduction
            adjusted_size = signal.size * 0.5

        # Check for sufficient funds/assets
        # Note: adjusted_size here is a multiplier (0.0 to 1.0) of the signal
        # The signal size is usually 1.0 (100% of intended move), but we interpret it as
        # "Target allocation of available execution power".
        # Real logic happens in execute_transaction, but we should update action if impossible.
        
        # For simplicity in this simulation, we keep the decision logic "pure" to the regime,
        # but the EXECUTION logic below handles the constraints.
        
        # CONSTRAINT: Always hash and save to ledger BEFORE returning
        block = self.ledger.add_block(
            entropy_score=entropy_score,
            regime=current_regime,
            action=action
        )
        
        # Persist Block
        if self.db_manager:
            self.db_manager.add_block(asdict(block))

        return TradeDecision(
            action=action,
            original_signal=signal,
            adjusted_size=adjusted_size,
            disposition=self.disposition,
            block_hash=block.hash
        )

    def record_external_decision(
        self,
        signal: TradeSignal,
        current_regime: Literal['ORDERED', 'CHAOTIC', 'TRANSITION'],
        entropy_score: float,
        action: Literal['EXECUTE', 'HALT', 'REDUCE']
    ) -> TradeDecision:
        """
        Record a decision made by an external agent (e.g., RL Agent) into the ledger.
        """
        # Set disposition based on the action (reverse engineer or just set based on regime?)
        # For consistency, let's just set a "delegated" disposition or match the action's typical one.
        
        if action == 'EXECUTE':
             self.disposition = Disposition(autonomy=0.9, integration=0.1)
             adjusted_size = signal.size
        elif action == 'HALT':
             self.disposition = Disposition(autonomy=0.1, integration=0.9)
             adjusted_size = 0.0
        else: # REDUCE
             self.disposition = Disposition(autonomy=0.5, integration=0.5)
             adjusted_size = signal.size * 0.5

        # Save to ledger
        block = self.ledger.add_block(
            entropy_score=entropy_score,
            regime=current_regime,
            action=action
        )
        
        # Persist Block
        if self.db_manager:
            self.db_manager.add_block(asdict(block))

        return TradeDecision(
            action=action,
            original_signal=signal,
            adjusted_size=adjusted_size,
            disposition=self.disposition,
            block_hash=block.hash
        )

    def execute_transaction(self, decision: TradeDecision, current_price: float) -> Optional[float]:
        """
        Execute the trade decision against the portfolio.
        Returns pnl_pct if it was a SELL, else None.
        """
        symbol = decision.original_signal.symbol
        direction = decision.original_signal.direction
        pnl_to_return = None
        
        if decision.action == 'HALT' or decision.adjusted_size <= 0:
            return

        # Determine transaction value based on directive
        direction = decision.original_signal.direction
        
        # Direction Logic
        if direction == 'BUY':
            # 1. Calculate raw intent
            if self.use_compounding:
                # Compounding: adjusted_size is % of TOTAL AVAILABLE USD
                usd_to_spend = self.balance_usd * decision.adjusted_size
            else:
                # Fixed: adjusted_size is % of FIXED STAKE
                usd_to_spend = self.fixed_stake * decision.adjusted_size
                if usd_to_spend > self.balance_usd:
                    usd_to_spend = self.balance_usd
            
            # --- GOVERNOR INTERVENTION (Risk Check via Messaging) ---
            if self.governor:
                # Sync balance first (Direct state sync for performance, or could be a message too)
                total_equity = self.get_portfolio_value()
                self.governor.update_balance(total_equity)
                
                # Check 1: Is trade allowed?
                msg = {
                    'type': 'VALIDATE_TRADE', 
                    'price': current_price,
                    'symbol': symbol
                }
                
                # In a strict Holon system, this might be async. 
                # Here we use the direct receive_message return value for synchronous control.
                is_approved, safe_qty, leverage = self.governor.receive_message(sender=self, content=msg)
                
                if not is_approved:
                    print(f"  [RISK] Governor BLOCKED Buy Trade.")
                    return # Stop execution

                # Check 2: Position Sizing (Constraint)
                intended_qty = usd_to_spend / current_price
                
                if intended_qty > safe_qty:
                    # If Governor limits us (e.g. Risk Mgmt), respect safe_qty
                    # Note: safe_qty is the total asset size allowed.
                    print(f"  [RISK] Governor RESIZED Buy: {intended_qty:.4f} -> {safe_qty:.4f}")
                    # Recalculate spending based on safe_qty (Total Notional)
                    # We will pay this / leverage.
                    usd_to_spend_notional = safe_qty * current_price
                else:
                    usd_to_spend_notional = usd_to_spend
                
                # If no Governor, leverage defaults to 1
                if not self.governor:
                    leverage = 1.0

            else:
                # No Governor attached
                usd_to_spend_notional = usd_to_spend
                leverage = 1.0

            # Execute Buy
            # We check notional >= 0.01 (min trade value)
            if usd_to_spend_notional > 0.01: 
                # Calculate quantity
                asset_amount = usd_to_spend_notional / current_price
                
                if self.actuator:
                    # Delegate
                    order = self.actuator.place_limit_order(
                        symbol=symbol, 
                        direction='BUY', 
                        quantity=asset_amount, 
                        limit_price=current_price
                    )
                    
                    # SIMULATION: Immediate check against current price (Fill at Touch)
                    fills = self.actuator.check_fills(candle_low=current_price, candle_high=current_price)
                    
                    if fills:
                        res = fills[0]
                        
                        # Apply Leverage to Cost Deduction
                        # We only pay the Initial Margin (Cost / Leverage)
                        margin_cost = res['cost_usd'] / leverage
                        self.balance_usd -= margin_cost
                        
                        # Update Asset Holdings (Position Stacking / Pyramiding)
                        current_holding = self.held_assets.get(symbol, 0.0)
                        new_qty = current_holding + res['filled_qty']
                        
                        if current_holding > 0:
                            # Weighted Average Price calculation
                            old_avg = self.entry_prices.get(symbol, current_price)
                            consolidated_avg = ((current_holding * old_avg) + (res['filled_qty'] * current_price)) / new_qty
                            self.entry_prices[symbol] = consolidated_avg
                            print(f"[{self.name}] STACKED {symbol}: New Avg Price ${consolidated_avg:.4f}")
                        else:
                            self.entry_prices[symbol] = current_price  # Fresh entry
                        
                        self.held_assets[symbol] = new_qty
                        
                        # WARP SPEED: Store metadata for accurate valuation and leverage tracking
                        self.position_metadata[symbol] = {
                            'leverage': leverage,
                            'entry_price': self.entry_prices[symbol]
                        }
                        
                        # Notify Governor
                        if self.governor:
                            self.governor.receive_message(self, {
                                'type': 'POSITION_FILLED',
                                'symbol': symbol,
                                'direction': 'LONG',
                                'price': current_price,
                                'quantity': res['filled_qty']
                            })
                        
                        # Log Trade
                        if self.db_manager:
                            self.db_manager.save_trade({
                                'symbol': symbol,
                                'direction': 'BUY',
                                'quantity': res['filled_qty'],
                                'price': current_price,
                                'cost_usd': margin_cost,  # Log actual spend (margin)
                                'leverage': leverage,
                                'notional_value': res['cost_usd'],
                                'timestamp': datetime.now(timezone.utc).isoformat(),
                                'pnl': 0.0,
                                'pnl_percent': 0.0
                            })
                            self._persist_portfolio()
                else:
                    # Direct
                    # margin_cost = notional / leverage
                    margin_cost = usd_to_spend_notional / leverage
                    
                    self.balance_usd -= margin_cost
                    current_holding = self.held_assets.get(symbol, 0.0)
                    new_qty = current_holding + asset_amount
                    
                    if current_holding > 0:
                        # Weighted Average Price calculation
                        old_avg = self.entry_prices.get(symbol, current_price)
                        consolidated_avg = ((current_holding * old_avg) + (asset_amount * current_price)) / new_qty
                        self.entry_prices[symbol] = consolidated_avg
                        print(f"[{self.name}] STACKED {symbol}: New Avg Price ${consolidated_avg:.4f}")
                    else:
                        self.entry_prices[symbol] = current_price  # Fresh entry
                        
                    self.held_assets[symbol] = new_qty
                    
                    # WARP SPEED: Store metadata for accurate valuation and leverage tracking
                    self.position_metadata[symbol] = {
                        'leverage': leverage,
                        'entry_price': self.entry_prices[symbol]
                    }
                    
                    # Notify Governor (Direct Mode)
                    if self.governor:
                        self.governor.receive_message(self, {
                            'type': 'POSITION_FILLED',
                            'symbol': symbol,
                            'direction': 'LONG',
                            'price': current_price,
                            'quantity': asset_amount
                        })
                    
                    # Log Trade
                    if self.db_manager:
                        self.db_manager.save_trade({
                            'symbol': symbol,
                            'direction': 'BUY',
                            'quantity': asset_amount,
                            'price': current_price,
                            'cost_usd': margin_cost,
                            'leverage': leverage,
                            'notional_value': usd_to_spend_notional,
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'pnl': 0.0,
                            'pnl_percent': 0.0
                        })

        elif direction == 'SELL':
            # Logic for selling: Sell entire position of this symbol
            asset_to_sell = self.held_assets.get(symbol, 0.0) * decision.adjusted_size
            
            entry_p = self.entry_prices.get(symbol)
            
            # Retrieve leverage for this position to calculate returned margin
            meta = self.position_metadata.get(symbol, {})
            leverage = meta.get('leverage', 1.0)
            
            pnl = 0.0
            pnl_pct = 0.0
            if entry_p and entry_p > 0:
                pnl = (current_price - entry_p) * asset_to_sell
                pnl_pct = (current_price - entry_p) / entry_p

            if asset_to_sell > 0.00000001: 
                if self.actuator:
                    # Delegate
                    order = self.actuator.place_limit_order(
                        symbol=symbol, 
                        direction='SELL', 
                        quantity=asset_to_sell, 
                        limit_price=current_price
                    )
                    
                    # SIMULATION: Immediate check against current price
                    fills = self.actuator.check_fills(candle_low=current_price, candle_high=current_price)
                    
                    if fills:
                        res = fills[0]
                        # Update Holdings
                        current_holding = self.held_assets.get(symbol, 0.0)
                        self.held_assets[symbol] = max(0.0, current_holding - res['filled_qty'])
                        
                        # Credit Balance:
                        # 1. Return Margin: (Quantity * EntryPrice) / Leverage
                        # 2. Add PnL: (CurrentPrice - EntryPrice) * Quantity
                        
                        market_val_sold = res['filled_qty'] * current_price
                        entry_val_sold = res['filled_qty'] * entry_p if entry_p else 0
                        margin_released = entry_val_sold / leverage
                        realized_pnl = market_val_sold - entry_val_sold
                        
                        self.balance_usd += (margin_released + realized_pnl)
                        
                        # Notify Governor
                        if self.governor:
                            self.governor.receive_message(self, {
                                'type': 'POSITION_CLOSED',
                                'symbol': symbol
                            })
                            
                        # Log Trade
                        if self.db_manager:
                            self.db_manager.save_trade({
                                'symbol': symbol,
                                'direction': 'SELL',
                                'quantity': res['filled_qty'],
                                'price': current_price,
                                'cost_usd': res['cost_usd'], # Notional sold
                                'timestamp': datetime.now(timezone.utc).isoformat(),
                                'pnl': realized_pnl,
                                'pnl_percent': pnl_pct
                            })
                            
                        if self.held_assets[symbol] <= 0:
                            if symbol in self.entry_prices: del self.entry_prices[symbol]
                            if symbol in self.position_metadata: del self.position_metadata[symbol]
                        
                        pnl_to_return = pnl_pct
                else:
                    # Direct
                    market_val_sold = asset_to_sell * current_price
                    entry_val_sold = asset_to_sell * entry_p if entry_p else 0
                    margin_released = entry_val_sold / leverage
                    realized_pnl = market_val_sold - entry_val_sold
                    
                    current_holding = self.held_assets.get(symbol, 0.0)
                    self.held_assets[symbol] = max(0.0, current_holding - asset_to_sell)
                    
                    self.balance_usd += (margin_released + realized_pnl)
                    
                    # Log Trade
                    if self.db_manager:
                        self.db_manager.save_trade({
                            'symbol': symbol,
                            'direction': 'SELL',
                            'quantity': asset_to_sell,
                            'price': current_price,
                            'cost_usd': market_val_sold,
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'pnl': realized_pnl,
                            'pnl_percent': pnl_pct
                        })
                        
                    if self.held_assets[symbol] <= 0:
                        if symbol in self.entry_prices: del self.entry_prices[symbol]
                        if symbol in self.position_metadata: del self.position_metadata[symbol]
                    
                    pnl_to_return = pnl_pct

        # Persist Portfolio State
        self._persist_portfolio()
        return pnl_to_return

    def get_portfolio_value(self, current_price_ref: float = 0.0) -> float:
        """
        Calculate total portfolio value in USD based on all held assets (Leveraged Equity).
        Equity = Free Balance + Margin Used + Unrealized PnL
        """
        equity = self.balance_usd
        
        for sym, qty in self.held_assets.items():
            current_price = self.latest_prices.get(sym, 0.0)
            entry_price = self.entry_prices.get(sym, 0.0)
            meta = self.position_metadata.get(sym, {})
            leverage = meta.get('leverage', 1.0)
            
            if qty > 0 and current_price > 0 and entry_price > 0:
                # Margin currently locked in this position
                margin_used = (qty * entry_price) / leverage
                
                # Unrealized PnL
                unrealized_pnl = (current_price - entry_price) * qty
                
                equity += (margin_used + unrealized_pnl)
                
        return equity

    def receive_message(self, sender: Any, content: Any) -> None:
        """
        Handle incoming messages.
        """
        if isinstance(content, Message):
            if content.type == 'EXECUTE':
                 pass # Logic to trigger execution via message
        else:
            pass

    def get_ledger_summary(self) -> dict:
        """
        Get a summary of the audit ledger.
        
        Returns:
            Dictionary with chain length, validity, and last block info
        """
        return {
            'total_blocks': len(self.ledger),
            'chain_valid': self.ledger.verify_chain(),
            'last_block': asdict(self.ledger.chain[-1]) if self.ledger.chain else None
        }
