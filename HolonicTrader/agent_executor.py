"""
ExecutorHolon - The Executor Agent (Phase 4)

This agent acts as the 'trade executor' of the AEHML core.
It executes trades only if the Disposition allows it and maintains
a local pseudo-blockchain ledger for audit purposes.

Key Features:
1. Pseudo-Blockchain Ledger (AuditLedger) with SHA-256 hashing
2. Disposition-based trade execution logic
"""

import logging
import hashlib
import json
import config
import time
import pandas as pd  # Added for ATR calculation
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Literal, List, Optional, Dict, Tuple
import math
import threading
from copy import deepcopy

from HolonicTrader.holon_core import Holon, Disposition, Message, PositionState
from HolonicTrader.exceptions import DeadMansSwitchTriggered
from utf8_logging import get_logger

# Setup Logger with UTF-8 encoding support
logger = get_logger("HolonicTrader.Executor")


@dataclass
class TradeSignal:
    """
    Represents a trading signal.
    """
    symbol: str
    direction: Literal['BUY', 'SELL']
    size: float
    price: float = 0.0  # Default to 0.0, will be filled by executor if not provided
    conviction: float = 0.5
    metadata: Dict = field(default_factory=dict)
    stop_loss_price: Optional[float] = None # Dynamic Stop Value
    take_profit_price: Optional[float] = None # Dynamic TP Value


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
    entropy_score: float = 0.0


@dataclass
class Position:
    """
    Represents a single trading position.

    FIX 2026-03-14: Added individual stack tracking with PnL per stack.
    """
    symbol: str
    virt_key: str
    direction: Literal['BUY', 'SELL']
    quantity: float
    entry_price: float
    entry_timestamp: str
    leverage: float = 1.0
    strategy: str = 'DIRECTIONAL'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    take_profit_type: str = 'FIXED'
    mfe: float = 0.0 # Max Favorable Excursion
    mae: float = 0.0 # Max Adverse Excursion
    stack_count: int = 1
    ppo_state: Optional[Any] = None
    ppo_conviction: Optional[float] = None
    metadata: Dict = field(default_factory=dict)
    state: PositionState = PositionState.ACTIVE
    # 2026-03-21: Entry context for winning pattern analysis
    entry_entropy: Optional[float] = None
    entry_regime: Optional[str] = None
    entry_conviction: Optional[float] = None
    entry_strategy: Optional[str] = None
    quality_score: Optional[float] = None
    is_whitelisted: bool = False

    # === NEW: Individual Stack Tracking ===
    # List of dicts: [{'entry_price': float, 'quantity': float, 'entry_time': float, 'pnl_usd': float, 'pnl_pct': float}, ...]
    stacks: List[Dict] = field(default_factory=list)

    @property
    def is_long(self) -> bool:
        return self.direction == 'BUY'

    @property
    def is_short(self) -> bool:
        return self.direction == 'SELL'

    def get_pnl_pct(self, current_price: float) -> float:
        if self.entry_price <= 0: return 0.0
        if self.is_long:
            return (current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - current_price) / self.entry_price

    def get_pnl_usd(self, current_price: float) -> float:
        return self.get_pnl_pct(current_price) * abs(self.quantity) * self.entry_price

    def get_stack_pnl_details(self, current_price: float) -> List[Dict]:
        """
        Returns PnL details for each individual stack.

        FIX 2026-03-14: Track PnL per stack for granular exit decisions.
        """
        if not self.stacks:
            # Legacy: No stack data, return synthetic single stack
            pnl_pct = self.get_pnl_pct(current_price)
            pnl_usd = pnl_pct * abs(self.quantity) * self.entry_price
            return [{
                'stack_id': 0,
                'entry_price': self.entry_price,
                'quantity': self.quantity,
                'entry_time': 0,
                'pnl_pct': pnl_pct,
                'pnl_usd': pnl_usd,
                'is_winner': pnl_usd > 0
            }]

        stack_details = []
        for idx, stack in enumerate(self.stacks):
            entry_price = stack.get('entry_price', 0)
            quantity = stack.get('quantity', 0)

            if entry_price <= 0 or quantity <= 0:
                continue

            # Calculate PnL for this stack
            if self.is_long:
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            pnl_usd = pnl_pct * abs(quantity) * entry_price

            stack_details.append({
                'stack_id': idx + 1,
                'entry_price': entry_price,
                'quantity': quantity,
                'entry_time': stack.get('entry_time', 0),
                'pnl_pct': pnl_pct,
                'pnl_usd': pnl_usd,
                'is_winner': pnl_usd > 0,
                'age_seconds': time.time() - stack.get('entry_time', time.time())
            })

        return stack_details

    def get_worst_stack(self, current_price: float) -> Optional[Dict]:
        """
        Returns the worst-performing stack (for hygiene exits).
        """
        details = self.get_stack_pnl_details(current_price)
        if not details:
            return None

        # Sort by PnL USD (ascending - worst first)
        details.sort(key=lambda x: x['pnl_usd'])
        return details[0]

    def get_best_stack(self, current_price: float) -> Optional[Dict]:
        """
        Returns the best-performing stack (for target exits).
        """
        details = self.get_stack_pnl_details(current_price)
        if not details:
            return None

        # Sort by PnL USD (descending - best first)
        details.sort(key=lambda x: x['pnl_usd'], reverse=True)
        return details[0]

    def get(self, key: str, default=None):
        """
        Dict-like access for compatibility with legacy code.
        Allows code like: pos.get('quantity', 0) to work on Position objects.
        """
        # Special handling for 'metadata' key - return the metadata dict itself
        if key == 'metadata':
            return getattr(self, 'metadata', default)
        return getattr(self, key, default)

    def to_dict(self) -> Dict:
        """Convert Position to dictionary for compatibility."""
        return {
            'symbol': self.symbol,
            'virt_key': self.virt_key,
            'direction': self.direction,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'leverage': self.leverage,
            'strategy': self.strategy,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'take_profit_type': self.take_profit_type,
            'mfe': self.mfe,
            'mae': self.mae,
            'stack_count': self.stack_count,
            'ppo_state': self.ppo_state,
            'ppo_conviction': self.ppo_conviction,
            'metadata': self.metadata,
            'state': self.state.name if self.state else 'ACTIVE',
            'stacks': self.stacks  # NEW: Include stack details
        }


class ExecutorHolon(Holon):
    """
    ExecutorHolon is the 'Executor' that decides whether to execute trades
    based on market regime and entropy levels. All decisions are logged
    to a tamper-evident pseudo-blockchain ledger.
    """
    # --- CONSTANTS (Magic Number Remediation) ---
    WORKING_ORDER_TIMEOUT = 30       # Seconds before a working order is considered stale
    DIVERGENCE_CHECK_INTERVAL = 300  # Seconds between equity divergence checks (5 mins)
    DIVERGENCE_THRESHOLD = 0.1       # 10% divergence triggers correction
    MIN_ORDER_VALUE = config.MIN_ORDER_VALUE if hasattr(config, 'MIN_ORDER_VALUE') else 10.0
    SPOT_SWEEP_THRESHOLD = 400.0     # When Futures hits $400
    SPOT_SWEEP_AMOUNT = 100.0        # Move $100 to Spot

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
            Uses Rust for performance if available.
            """
            try:
                import holonic_speed
                return holonic_speed.compute_block_hash(
                    self.timestamp,
                    self.entropy_score,
                    self.regime,
                    self.action,
                    self.prev_hash
                )
            except ImportError:
                # Fallback to Python
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

            # --- MEMORY SAFETY ---
            # Prune ledger if it exceeds 10,000 blocks to prevent memory leaks in long-running sessions.
            # We keep the genesis block or just truncate the tail. Pruning tail is safer for continuity logic.
            # Actually, prev_hash depends on immediate predecessor, so sliding window is fine.
            if len(self._chain) > 10000:
                self._chain = self._chain[-5000:] # Keep last 5000
                print(f"[Executor] 🧹 Pruned Ledger to last 5000 blocks.")


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
        market: Any = None,
        actuator: Any = None, # NEW: Actuator Link
        gui_queue: Any = None # NEW: Dashboard Link
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
        
        # Thread-safety for Positions
        self.position_lock = threading.RLock()

        # Initialize the audit ledger
        self.ledger = self.AuditLedger()
        
        # Persistence & Risk & Execution
        self.initial_capital = initial_capital
        self.use_compounding = use_compounding
        self.fixed_stake = fixed_stake
        self.db_manager = db_manager
        self.governor = governor
        self.market = market
        self.actuator = actuator
        self.gui_queue = gui_queue
        
        # Throttling for Divergence Warnings
        self.last_divergence_check = 0.0
        self.DIVERGENCE_CHECK_INTERVAL = 60.0 # 60 seconds
 # Store reference
        
        # Portfolio Management
        self.initial_capital = initial_capital
        self.balance_usd = initial_capital

        # FIX: Fee and Slippage Tracking for Ledger Drift Prevention
        self.cumulative_fees_usd = 0.0  # Total fees paid
        self.cumulative_slippage_usd = 0.0  # Total slippage cost
        self.last_ledger_sync_ts = time.time()  # Last time ledger was synced with exchange

        # === FIX 2026-03-04: PER-ASSET SLIPPAGE TRACKING ===
        # Track slippage by asset to identify worst offenders
        self._slippage_by_asset = {}  # {symbol: {'count': N, 'total_slippage': $, 'avg_slippage_pct': %}}
        
        # === DEATH BY THOUSAND CUTS PROTECTION (2026-03-03) ===
        self._cumulative_drift = 0.0  # Track cumulative drift
        self._cumulative_funding_pnl = 0.0  # Track funding payments
        self._trade_timestamps = []  # Track trade times for frequency limiting
        self._trades_today = 0  # Count trades today

        # === FEE TRACKING AUDIT (Fix for excessive fee detection) ===
        self._last_fee_reset_date = datetime.now(timezone.utc).date()
        self._fee_anomaly_threshold = 0.10  # 10% of balance
        self._fee_reset_count = 0  # Track how many times fees have been reset
        self._fee_halt_active = False  # FIX 2026-02-28: Flag to enforce trading halt on excessive fees

        # Daily fee tracking for audit
        self._today_fees_usd = 0.0
        self._yesterday_fees_usd = 0.0

        # FIX 2026-02-28: Per-trade fee logging for double-counting investigation
        self._trade_fee_log = []  # List of dicts: {'symbol', 'side', 'fee_usd', 'timestamp', 'source'}
        self._max_fee_log_size = 100  # Keep last 100 trades for comparison

        # === STARTUP FEE RESET ===
        # Reset fees on every fresh startup to prevent carryover from previous sessions
        self.cumulative_fees_usd = 0.0
        self._today_fees_usd = 0.0
        # FIX 2026-02-28: Grace period to suppress fee accumulation during startup
        # First N syncs compare DB-restored balance vs exchange balance, causing false drift.
        self._startup_sync_count = 0  # Counts sync_balance() calls during startup
        self._startup_grace_syncs = 3  # Ignore fee tracking for first 3 syncs
        
        # === FIX 2026-03-04: DRIFT ACCOUNTING FIX ===
        # Reset drift on startup - drift from baseline DB/exchange mismatch is PHANTOM
        # Only track drift from actual trades executed THIS session
        self._session_drift = 0.0  # Drift from this session's trades only
        self._startup_drift_ignored = 0.0  # Drift from startup sync (ignored)
        print(f"[{self.name}] 💰 Fee Counters Reset on Startup: $0.00")

        # Unified State Management
        # symbol/virt_key -> Position object
        self.positions: Dict[str, Position] = {}

        # Legacy mappings (will be migrated to self.positions)
        self.latest_prices: Dict[str, float] = {} # symbol -> last_seen_price
        self.held_assets: Dict[str, float] = {}  # symbol -> quantity (legacy format)
        self.entry_prices: Dict[str, float] = {}  # symbol -> entry_price (legacy format)
        self.entry_timestamps: Dict[str, str] = {}  # symbol -> timestamp (legacy format)
        self.position_metadata: Dict[str, dict] = {}  # symbol -> metadata dict (legacy format)

        # Stop-Loss / Take-Profit Parameters (Synced with config)
        self.stop_loss_pct = config.SCAVENGER_STOP_LOSS
        self.take_profit_pct = config.PREDATOR_TAKE_PROFIT
        
        # Sizing Strategy
        self.use_compounding = use_compounding
        self.fixed_stake = fixed_stake
        
        # FIX 2026-03-01: Compounding Pool for profit reinvestment
        self._compounding_pool = 0.0  # Accumulated profits for auto-reinvestment

        # Dashboard Details
        self.last_order_details = "NONE"
        
        # NEW: Error tracking for retry diagnostics
        self._last_execution_error = "Unknown"

        # Load state from DB if available
        if self.db_manager:
            self._load_state()
            
        # --- RECONCILIATION: Sync with Reality (Exchange) ---
        if self.market:
            # Check for forced HARD sync config
            if getattr(config, 'FORCE_HARD_SYNC_ON_STARTUP', False):
                logger.warning(f"[{self.name}] ⚠️ FORCE_HARD_SYNC_ON_STARTUP enabled - performing HARD reset")
                self.sync_with_exchange(mode='HARD')
            else:
                try:
                    self.sync_with_exchange(mode='SOFT')
                except DeadMansSwitchTriggered as e:
                    # DMS triggered during SOFT sync - stale DB position likely
                    logger.critical(f"[{self.name}] ☠️ DMS TRIGGERED: {e}")
                    logger.warning(f"[{self.name}] ⚠️ Attempting HARD sync recovery...")
                    try:
                        self.sync_with_exchange(mode='HARD')
                        logger.info(f"[{self.name}] ✅ HARD sync recovery successful")
                    except Exception as hard_error:
                        logger.critical(f"[{self.name}] ☠️ HARD sync also failed: {hard_error}")
                        raise
        # ----------------------------------------------------

        # Debug Flag
        self.DEBUG = getattr(config, 'DEBUG_MODE', False)
        if self.DEBUG:
            logger.setLevel(logging.DEBUG)

    def get_positions_snapshot(self) -> Dict[str, Position]:
        """Provides a thread-safe snapshot of all active positions."""
        with self.position_lock:
            return deepcopy(self.positions)

    def track_slippage(self, symbol: str, expected_price: float, fill_price: float, quantity: float):
        """
        FIX 2026-03-04: Track slippage per asset for diagnostics.
        
        Args:
            symbol: Asset symbol
            expected_price: Expected fill price
            fill_price: Actual fill price
            quantity: Trade quantity
        """
        if expected_price <= 0 or fill_price <= 0:
            return
            
        # Calculate slippage
        slippage_pct = abs(fill_price - expected_price) / expected_price
        slippage_usd = abs(fill_price - expected_price) * abs(quantity)
        
        # Update cumulative
        self.cumulative_slippage_usd += slippage_usd
        
        # Update per-asset tracking
        if symbol not in self._slippage_by_asset:
            self._slippage_by_asset[symbol] = {
                'count': 0,
                'total_slippage': 0.0,
                'avg_slippage_pct': 0.0
            }
        
        data = self._slippage_by_asset[symbol]
        data['count'] += 1
        data['total_slippage'] += slippage_usd
        # Running average
        data['avg_slippage_pct'] = (
            (data['avg_slippage_pct'] * (data['count'] - 1) + slippage_pct) / data['count']
        )
        
        # Log if significant slippage (>0.5%)
        if slippage_pct > 0.005:
            logger.warning(f"[{self.name}] 📉 SLIPPAGE ALERT: {symbol} {slippage_pct*100:.2f}% (${slippage_usd:.2f})")

    def check_trade_frequency_limit(self) -> Tuple[bool, str]:
        """
        FIX 2026-03-04: Check if trade frequency limits are exceeded.
        
        Returns:
            (allowed, reason): Tuple of whether trade is allowed and reason
        """
        now = time.time()
        current_date = datetime.now(timezone.utc).date()
        
        # Get limits from config
        max_trades_per_hour = getattr(config, 'MAX_TRADES_PER_HOUR', 10)
        max_trades_per_day = getattr(config, 'MAX_TRADES_PER_DAY', 100)
        min_time_between_trades = getattr(config, 'MIN_TIME_BETWEEN_TRADES_SEC', 30)
        
        # Initialize if needed
        if not hasattr(self, '_trade_timestamps'):
            self._trade_timestamps = []
        if not hasattr(self, '_trades_today'):
            self._trades_today = 0
        if not hasattr(self, '_last_trade_date'):
            self._last_trade_date = current_date
        if not hasattr(self, '_last_trade_time'):
            self._last_trade_time = 0
        
        # Reset daily counter if new day
        if current_date != self._last_trade_date:
            self._trades_today = 0
            self._last_trade_date = current_date
        
        # Clean old timestamps (keep only last hour)
        one_hour_ago = now - 3600
        self._trade_timestamps = [ts for ts in self._trade_timestamps if ts > one_hour_ago]
        
        # Check trades per hour
        if len(self._trade_timestamps) >= max_trades_per_hour:
            return False, f"Max trades per hour exceeded ({max_trades_per_hour})"
        
        # Check trades per day
        if self._trades_today >= max_trades_per_day:
            return False, f"Max trades per day exceeded ({max_trades_per_day})"
        
        # Check minimum time between trades
        if self._last_trade_time > 0:
            time_since_last = now - self._last_trade_time
            if time_since_last < min_time_between_trades:
                remaining = min_time_between_trades - time_since_last
                return False, f"Cooldown active: {remaining:.0f}s remaining"
        
        return True, "OK"

    def record_trade_executed(self, symbol: str):
        """
        FIX 2026-03-04: Record a trade for frequency tracking.
        
        Args:
            symbol: Asset symbol that was traded
        """
        now = time.time()
        current_date = datetime.now(timezone.utc).date()
        
        # Reset daily counter if new day
        if not hasattr(self, '_last_trade_date') or current_date != self._last_trade_date:
            self._trades_today = 0
            self._last_trade_date = current_date
        
        # Record timestamp
        if not hasattr(self, '_trade_timestamps'):
            self._trade_timestamps = []
        self._trade_timestamps.append(now)
        self._trades_today += 1
        self._last_trade_time = now
        
        logger.info(f"[{self.name}] 📊 Trade recorded: {symbol} (Today: {self._trades_today}, Last hour: {len(self._trade_timestamps)})")

    def compute_size_by_volatility(self, symbol: str, price: float, target_vol: float, timeframe: str = None, period: int = 14) -> Optional[float]:
        """
        Compute an execution quantity sized to target annualized volatility.

        Returns quantity (units of asset) or None if sizing failed.
        Formula: notional = (target_vol * account_equity) / realized_vol_ann
                 qty = notional / price
        """
        try:
            tf = timeframe or getattr(config, 'TIMEFRAME', '15m')
            # derive minutes from timeframe
            if isinstance(tf, str) and tf.endswith('m'):
                minutes = int(tf[:-1])
            elif isinstance(tf, str) and tf.endswith('h'):
                minutes = int(tf[:-1]) * 60
            else:
                minutes = 15

            # Attempt to fetch recent OHLCV from market or observer
            ohlcv = None
            exec_sym = config.KRAKEN_SYMBOL_MAP.get(symbol, symbol)
            if self.market:
                # Prefer a market-level fetch_ohlcv if present
                if hasattr(self.market, 'fetch_ohlcv'):
                    try:
                        ohlcv = self.market.fetch_ohlcv(exec_sym, tf, since=None, limit=period+1)
                    except Exception:
                        ohlcv = None
                # Fallback to observer cache if available
                if (not ohlcv) and hasattr(self.market, 'observer') and hasattr(self.market.observer, '_fetch_ohlcv_resilient'):
                    try:
                        ohlcv = self.market.observer._fetch_ohlcv_resilient(exec_sym, tf, since=None, limit=period+1)
                    except Exception:
                        ohlcv = None

            if not ohlcv or len(ohlcv) < 3:
                return None

            # OHLCV format: [timestamp, open, high, low, close, volume]
            closes = [float(c[4]) for c in ohlcv if len(c) >= 5]
            if len(closes) < 3:
                return None

            import numpy as _np
            rets = _np.diff(_np.log(_np.array(closes)))
            vol = float(_np.std(rets, ddof=1))
            if vol <= 0:
                return None

            # Annualize: factor = sqrt(trading_periods_per_year)
            periods_per_day = (24 * 60) / max(1, minutes)
            ann_factor = _np.sqrt(252 * periods_per_day)
            vol_ann = vol * ann_factor

            if vol_ann <= 0:
                return None

            # Dollar notional sized to target volatility
            target_dollar_vol = target_vol * max(1.0, self.balance_usd)
            notional = target_dollar_vol / vol_ann
            qty = notional / max(1e-12, float(price))

            # Enforce exchange min order value
            if self.market and hasattr(self.market, 'get_min_order_value'):
                try:
                    min_val = float(self.market.get_min_order_value(symbol))
                    if (qty * price) < min_val:
                        # Too small to place
                        return None
                except Exception:
                    pass

            return float(qty)
        except Exception:
            logger.exception(f"[{self.name}] ⚠️ Volatility sizing failed for {symbol}")
            return None

    def _map_symbol(self, symbol: str, to_exchange: bool = True) -> str:
        """
        Helper to map symbols between internal and exchange formats.
        """
        if to_exchange:
            # Internal (BTC/USDT) -> Exchange (XBT/USDT:USDT or similar)
            return config.KRAKEN_SYMBOL_MAP.get(symbol, symbol)
        else:
            # Exchange -> Internal
            # 1. Direct reverse lookup
            for k, v in config.KRAKEN_SYMBOL_MAP.items():
                if v == symbol: return k
            
            # 2. Heuristic for Kraken Futures (PF_XBTUSD -> BTC/USDT)
            raw = symbol.upper()
            if raw.startswith('PF_'):
                base = raw[3:6]
                if base == 'XBT': base = 'BTC'
                guess = f"{base}/USDT"
                if guess in config.ALLOWED_ASSETS: return guess
            
            # 3. Strip suffix (XRP/USD:USD -> XRP/USDT)
            if '/' in symbol:
                base = symbol.split('/')[0]
                if base == 'XBT': base = 'BTC'
                guess = f"{base}/USDT"
                if guess in config.ALLOWED_ASSETS: return guess
                
            return symbol

    def sync_with_exchange(self, mode: Literal['SOFT', 'HARD'] = 'SOFT'):
        """
        Unified reconciliation with the exchange.
        SOFT: Updates existing positions and adds new ones. Removes 'ghosts' found only internally.
        HARD: Wipes local state and re-imports everything from the exchange.
        """
        if not self.market:
            logger.warning(f"[{self.name}] ⚠️ Cannot sync: No Market linked.")
            return

        logger.info(f"[{self.name}] 🔄 SYNCING STATE ({mode}) with Exchange...")
        
        # DEBUG: Log internal positions before sync
        if self.positions:
            logger.debug(f"[{self.name}] 📋 Internal positions before sync: {list(self.positions.keys())}")
            for vk, pos in self.positions.items():
                pos_value = abs(pos.quantity * pos.entry_price) if pos.entry_price > 0 else 0
                logger.debug(f"[{self.name}]   - {vk}: qty={pos.quantity}, entry=${pos.entry_price}, value=${pos_value:.2f}")
        else:
            logger.debug(f"[{self.name}] 📋 No internal positions before sync")

        try:
            # 1. Sync Cash Balance (Equity & Wallet)
            try:
                real_equity = self.market.get_equity()
                real_wallet = self.market.get_wallet_balance() or real_equity
                if real_wallet and real_wallet > 0:
                    if abs(self.balance_usd - real_wallet) > 0.01:
                        logger.info(f"[{self.name}] 🏦 Syncing Balance: ${self.balance_usd:.2f} -> ${real_wallet:.2f}")
                        self.balance_usd = real_wallet
            except Exception as e:
                logger.debug(f"[{self.name}] Balance sync failed: {e}")

            # 2. Fetch Open Positions
            positions_raw = self.market.fetch_positions()
            active_exchange = {}
            for p in positions_raw:
                size = float(p.get('contracts') or 0.0)
                if size == 0: continue

                exch_symbol = p['symbol']
                internal_symbol = self._map_symbol(exch_symbol, to_exchange=False)

                direction = 'BUY' if p['side'] == 'long' else 'SELL'
                active_exchange[internal_symbol] = {
                    'symbol': internal_symbol,
                    'direction': direction,
                    'quantity': size if direction == 'BUY' else -size,
                    'entry_price': float(p.get('entryPrice') or 0.0),
                    'mark_price': float(p.get('markPrice') or 0.0),
                    'leverage': float(p.get('leverage') or 1.0)
                }
            
            # DEBUG: Log exchange positions
            logger.debug(f"[{self.name}] 📋 Exchange positions: {list(active_exchange.keys())}")
            for sym, data in active_exchange.items():
                logger.debug(f"[{self.name}]   - {sym}: qty={data['quantity']}, entry=${data['entry_price']}")

            with self.position_lock: # Acquire lock for all position modifications
                if mode == 'HARD':
                    logger.warning(f"[{self.name}] ⚠️ HARD RESET: Clearing all local positions.")
                    self.positions.clear()

                # 3. Synchronize
                seen_internal = set()
                for int_sym, data in active_exchange.items():
                    # We need to find the correct virt_key.
                    # If we have multiple positions for the same symbol (different strategies), 
                    # this simple logic might need refinement. 
                    # For now, we update the primary or first match.
                    
                    matches = [k for k in self.positions.keys() if k.startswith(f"{int_sym}:") or k == int_sym]
                    
                    if matches:
                        # Update existing (use first match if multiple)
                        vk = matches[0]
                        pos = self.positions[vk]
                        if abs(pos.quantity - data['quantity']) > 1e-8:
                            logger.info(f"[{self.name}] ⚖️ Adjusting {vk}: {pos.quantity} -> {data['quantity']}")
                            pos.quantity = data['quantity']
                        
                        # Update entry price if internal is 0
                        if pos.entry_price <= 0:
                            pos.entry_price = data['entry_price']
                        
                        pos.leverage = data['leverage']
                        seen_internal.add(vk)
                    else:
                        # New position found on exchange
                        # --- DEAD MAN's SWITCH: MASSIVE GHOST CHECK ---
                        if mode == 'SOFT' and data['quantity'] > (self.balance_usd * 0.5):
                            # If we find a ghost position whose size is > 50% of account equity
                            # This might mean a manual trade went completely rogue, or the API is returning bad data.
                            logger.critical(f"[{self.name}] ☠️ DEAD MAN'S SWITCH: Massive un-tracked position found '{int_sym}' ({data['quantity']}). Refusing to adopt.")
                            raise DeadMansSwitchTriggered(f"Massive Un-tracked Ghost Position: {int_sym}")
                        # ----------------------------------------------
                        
                        vk = int_sym # Default virt_key to internal symbol
                        logger.info(f"[{self.name}] ➕ Importing {vk}: {data['quantity']} @ {data['entry_price']}")
                        self.positions[vk] = Position(
                            symbol=int_sym,
                            virt_key=vk,
                            direction=data['direction'],
                            quantity=data['quantity'],
                            entry_price=data['entry_price'],
                            entry_timestamp=datetime.now(timezone.utc).isoformat(),
                            leverage=data['leverage'],
                            strategy='RECOVERED'
                        )
                        seen_internal.add(vk)

                # 4. Cleanup Ghosts (Internal but not on Exchange)
                for vk in list(self.positions.keys()):
                    if vk not in seen_internal:
                        pos = self.positions[vk]

                        # --- DEAD MAN's SWITCH: MASSIVE PURGE CHECK ---
                        if mode == 'SOFT':
                            # If it's a very large position that suddenly isn't on the exchange,
                            # it could be a catastrophic liquidation, OR an API glitch.
                            # Purging it here might mask the issue and let the bot open another one.
                            pos_value = abs(pos.quantity * pos.entry_price)
                            if pos_value > (self.balance_usd * 0.5): # 50% equity threshold
                                logger.critical(f"[{self.name}] ☠️ DEAD MAN'S SWITCH: Massive internal position '{vk}' suddenly missing from exchange. Possible liquidation or API desync.")
                                logger.critical(f"[{self.name}]   Position details: qty={pos.quantity}, entry=${pos.entry_price}, value=${pos_value:.2f}")
                                logger.critical(f"[{self.name}]   Account equity: ${self.balance_usd:.2f}, Threshold (50%): ${self.balance_usd * 0.5:.2f}")
                                logger.critical(f"[{self.name}]   Exchange positions: {list(active_exchange.keys()) if active_exchange else 'NONE'}")
                                logger.critical(f"[{self.name}]   ACTION: HARD sync will be attempted automatically to recover from stale DB state")
                                raise DeadMansSwitchTriggered(f"Massive Internal Position missing from exchange data: {vk} (value=${pos_value:.2f}, equity=${self.balance_usd:.2f})")
                        # ----------------------------------------------

                        logger.warning(f"[{self.name}] 👻 GHOST REMOVED: {vk} ({pos.quantity}) - not on exchange")
                        
                        # --- NEW: REGISTER GHOST REMOVAL AS A WIN/LOSS ---
                        try:
                            # Estimate exit price: Stop Loss hit? Target hit? Or latest market price?
                            exit_price = pos.metadata.get('stop_loss', pos.entry_price)
                            if hasattr(pos, 'stop_loss_price') and pos.stop_loss_price:
                                exit_price = pos.stop_loss_price
                            if exit_price == pos.entry_price and self.governor and hasattr(self.governor, 'latest_prices'):
                                exit_price = self.governor.latest_prices.get(pos.symbol, pos.entry_price)
                                
                            # Calculate PnL locally
                            qty = abs(pos.quantity)
                            if pos.direction == 'BUY':
                                pnl_usd = (exit_price - pos.entry_price) * qty
                            else:
                                pnl_usd = (pos.entry_price - exit_price) * qty

                            # Only log if it's a meaningful quantity
                            if qty > 0.0001:
                                self._record_trade_result(
                                    symbol=pos.symbol,
                                    direction=pos.direction,
                                    entry_price=pos.entry_price,
                                    exit_price=exit_price,
                                    quantity=qty,
                                    pnl_usd=pnl_usd,
                                    entry_time=pos.entry_timestamp,
                                    exit_time=datetime.now(timezone.utc).isoformat(),
                                    strategy=pos.strategy,
                                    mfe=getattr(pos, 'mfe', 0.0),
                                    mae=getattr(pos, 'mae', 0.0),
                                    exit_reason='GHOST_REMOVAL',
                                    entropy_score=getattr(pos, 'entry_entropy', None),
                                    regime=getattr(pos, 'entry_regime', None),
                                    conviction=getattr(pos, 'entry_conviction', None),
                                    quality_score=getattr(pos, 'quality_score', None),
                                    is_whitelisted=getattr(pos, 'is_whitelisted', False),
                                )
                                logger.info(f"[{self.name}] ✅ POSITION CLOSED ON EXCHANGE: {pos.symbol} (Estimated PnL: ${pnl_usd:.2f} at {exit_price:.4f})")
                        except Exception as e:
                            logger.error(f"[{self.name}] Error logging ghost removal PnL: {e}")
                        # --------------------------------------------------

                        del self.positions[vk]
                        if self.governor and hasattr(self.governor, 'close_position'):
                            try: self.governor.close_position(pos.symbol)
                            except: pass

            # 5. Finalize
            self._persist_portfolio()
            if self.governor and hasattr(self.governor, 'reconcile_with_executor'):
                self.governor.reconcile_with_executor(self.get_positions_snapshot())


            logger.info(f"[{self.name}] ✅ Sync Complete. {len(self.positions)} active positions.")

        except DeadMansSwitchTriggered:
            raise  # Re-raise DMS exceptions - they are critical safety events
        except Exception as e:
            logger.error(f"[{self.name}] ❌ Sync Failed: {e}")
            if mode == 'HARD':
                logger.info(f"[{self.name}] Retrying with SOFT sync...")
                self.sync_with_exchange(mode='SOFT')

    def _persist_portfolio(self):
        """Helper to save current balance and assets to DB."""
        if self.db_manager:
            try:
                f_bal = self.governor.fortress_balance if self.governor else 0.0

                # Update legacy structures to match current state of new structure
                self.held_assets.clear()
                self.position_metadata.clear()

                with self.position_lock: # Acquire lock for reading positions for persistence
                    for k, p in self.positions.items():
                        self.held_assets[p.symbol] = p.quantity
                        d = asdict(p)
                        if 'state' in d and hasattr(d['state'], 'name'):
                            d['state'] = d['state'].name
                        self.position_metadata[p.symbol] = d

                # Use the legacy structures for saving (for compatibility with other parts of the system)
                self.db_manager.save_portfolio(
                    self.balance_usd,
                    self.held_assets,
                    self.position_metadata,
                    fortress_balance=f_bal
                )
            except Exception as e:
                logger.error(f"[{self.name}] ❌ Failed to persist portfolio: {e}")

    def save_state(self):
        """Public method to force persistence (e.g., on shutdown)."""
        logger.info(f"[{self.name}] 💾 Force-Saving Portfolio State...")
        self._persist_portfolio()

    def _load_state(self):
        """Premium State Restoration: Reconstructs portfolio and records from DB."""
        # Load Portfolio
        portfolio = self.db_manager.get_portfolio()
        if portfolio:
            stored_balance = portfolio.get('balance_usd', self.initial_capital)

            # --- SYNC OVERRIDE CHECK ---
            # If initial_capital passed to constructor (from Main Sync) differs significantly
            # from stored DB balance, prefer the FRESH SYNC value.
            if abs(stored_balance - self.initial_capital) > 0.05 and self.initial_capital > 0:
                 logger.info(f"[{self.name}] 🔄 SYNC PRIORITY: Overriding Stored Balance ${stored_balance:.2f} with Fresh Sync ${self.initial_capital:.2f}")
                 self.balance_usd = self.initial_capital
            else:
                 self.balance_usd = stored_balance

            # Legacy fields from DB
            held_assets = portfolio.get('held_assets', {})
            position_metadata = portfolio.get('position_metadata', {})

            with self.position_lock: # Acquire lock for all position modifications during load
                # Clear current state
                self.positions.clear()
                self.held_assets.clear()
                self.entry_prices.clear()
                self.entry_timestamps.clear()
                self.position_metadata.clear()

                # Reconstruct Positions (both legacy and new structures)
                for k, meta in position_metadata.items():
                    if not isinstance(meta, dict): continue

                    # Handle migration from old format or reconstruct from new
                    symbol = meta.get('symbol', k.split(':')[0] if ':' in k else k)
                    virt_key = meta.get('virt_key', k)
                    direction = meta.get('direction')
                    if not direction:
                        qty = held_assets.get(k, meta.get('quantity', 0.0))
                        direction = 'BUY' if qty > 0 else 'SELL'

                    # Create Position object for new structure
                    position_obj = Position(
                        symbol=symbol,
                        virt_key=virt_key,
                        direction=direction,
                        quantity=held_assets.get(k, meta.get('quantity', 0.0)),
                        entry_price=meta.get('entry_price', meta.get('price', 0.0)),
                        entry_timestamp=meta.get('entry_timestamp', meta.get('timestamp', datetime.now(timezone.utc).isoformat())),
                        leverage=meta.get('leverage', 1.0),
                        strategy=meta.get('strategy', 'DIRECTIONAL'),
                        stop_loss=meta.get('stop_loss'),
                        take_profit=meta.get('take_profit'),
                        take_profit_type=meta.get('take_profit_type', 'FIXED'),
                        mfe=meta.get('mfe', 0.0),
                        mae=meta.get('mae', 0.0),
                        stack_count=meta.get('stack_count', 1),
                        ppo_state=meta.get('ppo_state'),
                        ppo_conviction=meta.get('ppo_conviction'),
                        metadata=meta.get('metadata', {})
                    )

                    self.positions[k] = position_obj

                    # Populate legacy structures for backward compatibility
                    self.held_assets[symbol] = position_obj.quantity
                    self.entry_prices[symbol] = position_obj.entry_price
                    self.entry_timestamps[symbol] = position_obj.entry_timestamp
                    self.position_metadata[symbol] = meta

                # If there are held_assets in the DB but not in position_metadata, add them too
                for symbol, qty in held_assets.items():
                    if symbol not in self.held_assets:
                        # This is a position that exists in legacy format but not in new format
                        # Create a minimal position object
                        self.held_assets[symbol] = qty
                        # Add a minimal entry for this position
                        self.entry_prices[symbol] = 0.0  # Will be updated when price is known
                        self.entry_timestamps[symbol] = datetime.now(timezone.utc).isoformat()
                        self.position_metadata[symbol] = {
                            'symbol': symbol,
                            'quantity': qty,
                            'direction': 'BUY' if qty > 0 else 'SELL',
                            'strategy': 'LEGACY_RECOVERY'
                        }

            # Sync metadata back to Governor if linked (using legacy format for now as Governor expects it)
            if self.governor:
                self.governor.sync_fortress(portfolio.get('fortress_balance', 0.0))

            logger.info(f"[{self.name}] 🏦 Portfolio Restored: ${self.balance_usd:.2f} USD")
            active_list = [f"{p.symbol}({p.quantity:.4f})" for p in self.positions.values() if abs(p.quantity) > 1e-8]
            if active_list:
                logger.debug(f"[{self.name}] 📦 Active Positions: {', '.join(active_list)}")

        # Load Last Block
        try:
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
                logger.info(f"[{self.name}] ⛓️ Ledger Tip Restored: {restored_block.hash[:8]}...")
        except Exception as e:
            logger.warning(f"[{self.name}] ⚠️ Could not restore ledger tip: {e}")

    def reconcile_exchange_positions(self):
        """
        Fetch REAL positions from Exchange and sync them to Brain.
        HARD RESET: Wipes local state and re-imports entirely from exchange truth.
        """
        if not self.actuator or not getattr(self.actuator, 'exchange', None):
            return

        # 1. Sync Cash Balance First
        try:
            real_cash = self.actuator.get_wallet_balance()
            if real_cash > 0:
                print(f"[{self.name}] 🏦 Syncing Ledger Balance: ${self.balance_usd:.2f} -> ${real_cash:.2f}")
                self.balance_usd = real_cash
                self.initial_capital = real_cash
                
                # --- AUTO-SWEEP TO SPOT (DUAL MODE) ---
                if getattr(config, 'TRADING_MODE', 'FUTURES') == 'DUAL':
                    if real_cash >= self.SPOT_SWEEP_THRESHOLD:
                        # Only sweep once per hour max to prevent spamming the API
                        if time.time() - self._last_spot_sweep > 3600:
                            print(f"[{self.name}] 🚀 SWEEP TRIGGERED: Futures Balance ${real_cash:.2f} >= ${self.SPOT_SWEEP_THRESHOLD}")
                            try:
                                # Execute transfer
                                response = self.actuator.exchange.transfer("USDT", self.SPOT_SWEEP_AMOUNT, "futures", "spot")
                                print(f"[{self.name}] ✅ SUCCESS: Swept ${self.SPOT_SWEEP_AMOUNT} USDT to Spot! {response}")
                                self._last_spot_sweep = time.time()
                                # Deduct locally immediately so we don't think we lost money before next fetch
                                self.balance_usd -= self.SPOT_SWEEP_AMOUNT
                                self.initial_capital -= self.SPOT_SWEEP_AMOUNT
                            except Exception as e:
                                print(f"[{self.name}] ❌ SWEEP FAILED: {e}")
                # --------------------------------------
        except: pass

        try:
            positions = self.actuator.exchange.fetch_positions()

            with self.position_lock: # Acquire lock for all position modifications
                # --- HARD RESET: Wipe all local positions first ---
                old_positions = dict(self.held_assets)
                self.held_assets.clear()
                self.entry_prices.clear()
                self.entry_timestamps.clear()
                self.position_metadata.clear()
                self.positions.clear()  # Clear new structure too

                synced_count = 0
                for p in positions:
                    # Safe cast to float to prevent 'NoneType' errors
                    size_raw = p.get('contracts', 0)
                    size = float(size_raw if size_raw is not None else 0)
                    if size == 0: continue

                    # 1. Map Exchange Symbol -> Internal Symbol
                    exchange_sym = p['symbol'] # 'XRP/USD:USD'
                    internal_sym = None

                    # Try simple match first
                    if exchange_sym in config.ALLOWED_ASSETS:
                        internal_sym = exchange_sym
                    else:
                        # Reverse Lookup in KRAKEN_SYMBOL_MAP
                        for k, v in config.KRAKEN_SYMBOL_MAP.items():
                            if v == exchange_sym:
                                internal_sym = k
                                break

                    if not internal_sym:
                        # HEURISTIC: Try to guess (XRP/USD:USD -> XRP/USDT)
                        raw = exchange_sym.split('/')[0]
                        if raw.startswith('PF_'): raw = raw[3:6]
                        guess = f"{raw}/USDT"
                        if guess in config.ALLOWED_ASSETS:
                            internal_sym = guess

                    if not internal_sym:
                        print(f"[{self.name}] WARNING Unknown Position Found: {exchange_sym} ({size}). Skipping.")
                        continue

                    # 2. Import this position from exchange
                    direction = 'BUY' if p['side'] == 'long' else 'SELL'

                    # Safe casting for Kraken/CCXT variants that might return None
                    entry_price_raw = p.get('entryPrice', 0.0)
                    entry_price = float(entry_price_raw if entry_price_raw is not None else 0.0)

                    mark_price_raw = p.get('markPrice', entry_price)
                    mark_price = float(mark_price_raw if mark_price_raw is not None else entry_price)

                    leverage_raw = p.get('leverage', 1.0)
                    leverage = float(leverage_raw if leverage_raw is not None else 1.0)

                    # --- PATCH: INVALID ENTRY PRICE FALLBACK ---
                    if entry_price <= 0:
                        entry_price = mark_price
                    if entry_price <= 0:
                        entry_price = self.latest_prices.get(internal_sym, 0.0)

                    # FIX 2026-03-21: Skip sub-minimum positions that can't be managed
                    # These positions are too small for the exchange to place stop-loss orders on.
                    base_asset = internal_sym.split('/')[0] if '/' in internal_sym else internal_sym
                    min_qty = config.MIN_TRADE_QTY.get(base_asset, 0.0)
                    if min_qty > 0 and size < min_qty:
                        print(f"[{self.name}] ⚠️ SUB-MINIMUM POSITION SKIPPED: {internal_sym} size {size} < min {min_qty} — cannot manage, ignoring")
                        continue

                    # Check if this differs from what we had before
                    old_qty = old_positions.get(internal_sym, 0.0)
                    if abs(old_qty - size) > 0.0001 or old_qty == 0:
                        print(f"[{self.name}] 📥 Importing: {internal_sym} ({direction}) Size: {size} Entry: {entry_price:.8f}")

                    # FIX 2026-02-28: Try to restore strategy/pool_type from existing DB metadata
                    # before defaulting to 'RECOVERED'. This preserves position strategy across restarts.
                    existing_meta = self.position_metadata.get(internal_sym, {})
                    preserved_strategy = existing_meta.get('strategy', 'RECOVERED')
                    preserved_pool_type = existing_meta.get('pool_type', None)
                    preserved_stack_count = existing_meta.get('stack_count', 1)
                    preserved_tp_type = existing_meta.get('take_profit_type', 'FIXED')
                    preserved_metadata = existing_meta.get('metadata', {})

                    # Update legacy attributes
                    self.held_assets[internal_sym] = size if direction == 'BUY' else -size
                    self.entry_prices[internal_sym] = entry_price
                    # Preserve entry timestamp from existing metadata if available (don't overwrite with restart time)
                    existing_timestamp = existing_meta.get('entry_timestamp')
                    if existing_timestamp:
                        self.entry_timestamps[internal_sym] = existing_timestamp
                    else:
                        self.entry_timestamps[internal_sym] = datetime.now(timezone.utc).isoformat()

                    # Reconstruct Metadata
                    direction_mult = 1.0 if direction == 'BUY' else -1.0
                    # Preserve existing stop-loss/take-profit levels if they exist
                    existing_sl = existing_meta.get('stop_loss')
                    existing_tp = existing_meta.get('take_profit')
                    sl_price = existing_sl if existing_sl else entry_price * (1.0 - (config.PREDATOR_STOP_LOSS * direction_mult))
                    tp_price = existing_tp if existing_tp else entry_price * (1.0 + (config.PREDATOR_TAKE_PROFIT * direction_mult))

                    self.position_metadata[internal_sym] = {
                        'symbol': internal_sym,
                        'direction': direction,
                        'quantity': size,
                        'entry_price': entry_price,
                        'entry_timestamp': self.entry_timestamps[internal_sym],
                        'leverage': leverage,
                        'stop_loss': sl_price,
                        'take_profit': tp_price,
                        'strategy': preserved_strategy,
                        'pool_type': preserved_pool_type,
                        'stack_count': preserved_stack_count,
                        'take_profit_type': preserved_tp_type,
                        'metadata': preserved_metadata
                    }

                    # Update new structure
                    virt_key = internal_sym  # Default virt_key
                    self.positions[virt_key] = Position(
                        symbol=internal_sym,
                        virt_key=virt_key,
                        direction=direction,
                        quantity=size if direction == 'BUY' else -size,
                        entry_price=entry_price,
                        entry_timestamp=self.entry_timestamps[internal_sym],
                        leverage=leverage,
                        strategy=preserved_strategy,
                        stop_loss=sl_price,
                        take_profit=tp_price,
                        take_profit_type=preserved_tp_type,
                        stack_count=preserved_stack_count,
                        state=PositionState.ACTIVE,
                        metadata=preserved_metadata
                    )

                    # Ensure protective orders exist for imported live positions.
                    # This prevents "naked" legacy exposure after restarts (tail-loss driver).
                    # FIX 2026-03-19: Place TP BEFORE SL on Kraken Futures to avoid wouldNotReducePosition error.
                    # On Kraken Futures, reduce-only orders reserve position quantity. Placing SL first reserves
                    # the entire position, causing the subsequent TP order to fail. By placing TP first, both
                    # orders can coexist as conditional reduce-only orders.
                    try:
                        already_ensured = bool(existing_meta.get('protective_orders_ensured', False))
                        if not already_ensured:
                            # FIX 2026-03-19: Cancel any surviving reduce-only orders from the previous
                            # session before placing new SL/TP. On Kraken Futures, leftover SL orders
                            # reserve the full position quantity, causing new TP orders to fail with
                            # wouldNotReducePosition. Cancelling first clears the reservation.
                            is_kraken = 'kraken' in str(self.actuator.exchange.id).lower() if self.actuator and hasattr(self.actuator, 'exchange') else False
                            if is_kraken and self.actuator and hasattr(self.actuator, 'cancel_all_orders'):
                                try:
                                    self.actuator.cancel_all_orders(internal_sym)
                                    print(f"[{self.name}] 🧹 Cleared stale protective orders for {internal_sym} before re-placing")
                                except Exception as _ce:
                                    print(f"[{self.name}] ⚠️ Could not cancel stale orders for {internal_sym}: {_ce}")
                            # Place TP first on Kraken Futures to avoid position reservation conflict
                            if is_kraken:
                                tp_result = self.place_take_profit_order(internal_sym, direction, abs(size), tp_price, leverage=leverage)
                                sl_result = self.place_stop_loss_order(internal_sym, direction, abs(size), sl_price, leverage=leverage)
                            else:
                                sl_result = self.place_stop_loss_order(internal_sym, direction, abs(size), sl_price, leverage=leverage)
                                tp_result = self.place_take_profit_order(internal_sym, direction, abs(size), tp_price, leverage=leverage)
                            if sl_result and tp_result:
                                self.position_metadata[internal_sym]['protective_orders_ensured'] = True
                                print(f"[{self.name}] 🛡️ PROTECTIVE ORDERS ENSURED: {internal_sym} SL/TP placed on import")
                            elif sl_result and not tp_result:
                                self.position_metadata[internal_sym]['protective_orders_ensured'] = False
                                print(f"[{self.name}] ⚠️ PROTECTION PARTIAL: {internal_sym} SL placed, TP failed on import")
                            elif not sl_result and tp_result:
                                self.position_metadata[internal_sym]['protective_orders_ensured'] = False
                                print(f"[{self.name}] ⚠️ PROTECTION PARTIAL: {internal_sym} TP placed, SL failed on import")
                            else:
                                self.position_metadata[internal_sym]['protective_orders_ensured'] = False
                                print(f"[{self.name}] ⚠️ PROTECTION FAILED: {internal_sym} SL/TP failed on import")
                    except Exception as _e:
                        print(f"[{self.name}] ⚠️ Protective order ensure failed for {internal_sym}: {_e}")

                    # --- EQUITY REPAIR FIX: Pre-populate latest_prices ---
                    # Use current markPrice from exchange if available, else entryPrice
                    mark_price_final = p.get('markPrice', entry_price)
                    mark_price = float(mark_price_final if mark_price_final is not None else entry_price)

                    if mark_price > 0:
                        self.latest_prices[internal_sym] = mark_price
                    else:
                        self.latest_prices[internal_sym] = entry_price
                    # -----------------------------------------------------

                    synced_count += 1

                # Report any positions that were in old_positions but NOT re-imported (i.e., ghosts)
                for old_sym, old_qty in old_positions.items():
                    if abs(old_qty) > 0.00000001 and old_sym not in self.held_assets:
                        print(f"[{self.name}] 👻 GHOST CLEARED: {old_sym} ({old_qty}) - not on exchange")
                        # Also notify Governor to remove from its tracking
                        if self.governor and hasattr(self.governor, 'close_position'):
                            try:
                                self.governor.close_position(old_sym)
                            except:
                                pass  # Governor might not have this method or position

            if synced_count > 0:
                print(f"[{self.name}] SUCCESS Imported {synced_count} positions from Exchange.")
            else:
                print(f"[{self.name}] SUCCESS No open positions on Exchange.")

            if self.db_manager:
                self.db_manager.save_portfolio(self.balance_usd, self.held_assets, self.position_metadata)

            # Notify Governor
            if self.governor:
                self.governor.sync_positions(self.held_assets, self.position_metadata)

        except Exception as e:
            print(f"[{self.name}] ERROR Reconciliation Failed: {e}")
            # Try a more gentle reconciliation
            self._soft_reconcile_with_exchange()

    def _soft_reconcile_with_exchange(self):
        """
        Gentle reconciliation that doesn't wipe everything, just updates differences.
        """
        if not self.actuator or not getattr(self.actuator, 'exchange', None):
            return

        try:
            positions = self.actuator.exchange.fetch_positions()

            # Create a map of current exchange positions
            exchange_positions = {}
            for p in positions:
                size_raw = p.get('contracts', 0)
                size = float(size_raw if size_raw is not None else 0)
                if size != 0:
                    exchange_sym = p['symbol']
                    # Map to internal symbol
                    internal_sym = None
                    if exchange_sym in config.ALLOWED_ASSETS:
                        internal_sym = exchange_sym
                    else:
                        for k, v in config.KRAKEN_SYMBOL_MAP.items():
                            if v == exchange_sym:
                                internal_sym = k
                                break

                    if internal_sym:
                        direction = 'BUY' if p['side'] == 'long' else 'SELL'
                        exchange_positions[internal_sym] = {
                            'quantity': size if direction == 'BUY' else -size,
                            'entry_price': float(p.get('entryPrice', 0.0) or 0.0),
                            'leverage': float(p.get('leverage', 1.0) or 1.0)
                        }

            with self.position_lock: # Acquire lock for all position modifications
                # Compare with our internal positions (legacy structure)
                for internal_sym, internal_qty in list(self.held_assets.items()):
                    if internal_sym in exchange_positions:
                        # Position exists on both sides, check if quantities match
                        exchange_qty = exchange_positions[internal_sym]['quantity']
                        if abs(internal_qty - exchange_qty) > 0.00000001:
                            print(f"[{self.name}] 🔄 POSITION ADJUSTMENT: {internal_sym} {internal_qty} -> {exchange_qty}")
                            self.held_assets[internal_sym] = exchange_qty

                            # Update corresponding position in new structure
                            matches = [k for k in self.positions.keys() if k.startswith(f"{internal_sym}:") or k == internal_sym]
                            if matches:
                                vk = matches[0]
                                if vk in self.positions:
                                    self.positions[vk].quantity = exchange_qty
                    else:
                        # Position exists in our records but not on exchange - GHOST
                        print(f"[{self.name}] 👻 GHOST DETECTED: {internal_sym} ({internal_qty}) - removing")

                        # Remove from legacy structure
                        del self.held_assets[internal_sym]
                        if internal_sym in self.entry_prices:
                            del self.entry_prices[internal_sym]
                        if internal_sym in self.entry_timestamps:
                            del self.entry_timestamps[internal_sym]
                        if internal_sym in self.position_metadata:
                            del self.position_metadata[internal_sym]

                        # Remove from new structure
                        matches = [k for k in self.positions.keys() if k.startswith(f"{internal_sym}:") or k == internal_sym]
                        for match_vk in matches:
                            if match_vk in self.positions:
                                del self.positions[match_vk]

                # Add any new positions from exchange that we don't have
                for internal_sym, pos_data in exchange_positions.items():
                    if internal_sym not in self.held_assets:
                        print(f"[{self.name}] ➕ NEW EXCHANGE POS: {internal_sym} ({pos_data['quantity']}) - adding")
                        self.held_assets[internal_sym] = pos_data['quantity']
                        self.entry_prices[internal_sym] = pos_data['entry_price']
                        self.entry_timestamps[internal_sym] = datetime.now(timezone.utc).isoformat()
                        self.position_metadata[internal_sym] = {
                            'symbol': internal_sym,
                            'direction': 'BUY' if pos_data['quantity'] > 0 else 'SELL',
                            'quantity': abs(pos_data['quantity']),
                            'entry_price': pos_data['entry_price'],
                            'entry_timestamp': datetime.now(timezone.utc).isoformat(),
                            'leverage': pos_data['leverage'],
                            'strategy': 'SYNCED',
                            'pool': 'A (DIR)'
                        }

                        # Add to new structure
                        virt_key = internal_sym  # Default virt_key
                        self.positions[virt_key] = Position(
                            symbol=internal_sym,
                            virt_key=virt_key,
                            direction='BUY' if pos_data['quantity'] > 0 else 'SELL',
                            quantity=pos_data['quantity'],
                            entry_price=pos_data['entry_price'],
                            entry_timestamp=datetime.now(timezone.utc).isoformat(),
                            leverage=pos_data['leverage'],
                            strategy='SYNCED',
                            state=PositionState.ACTIVE
                        )

            # Update DB
            if self.db_manager:
                self.db_manager.save_portfolio(self.balance_usd, self.held_assets, self.position_metadata)

            # Notify Governor
            if self.governor:
                self.governor.sync_positions(self.held_assets, self.position_metadata)

        except Exception as e:
            print(f"[{self.name}] ERROR Soft reconciliation failed: {e}")


    def adopt_position(self, symbol: str, qty: float, price: float, reason: str = "Manual Adoption"):
        """
        Manually add a position to the internal ledger (Adopting a Ghost).
        """
        logger.info(f"[{self.name}] 🧬 ADOPTING POSITION: {symbol} @ {price} (Qty: {qty}) - Reason: {reason}")
        
        direction = 'BUY' if qty > 0 else 'SELL'
        vk = symbol # Default virt_key
        
        with self.position_lock:
            self.positions[vk] = Position(
                symbol=symbol,
                virt_key=vk,
                direction=direction,
                quantity=qty,
                entry_price=price,
                entry_timestamp=datetime.now(timezone.utc).isoformat(),
                strategy='MANUAL_ADOPTION',
                metadata={'reason': reason}
            )
        
        # 2. Add to Ledger (Blockchain)
        self.ledger.add_block(
            entropy_score=0.0,
            regime='ORDERED',
            action='EXECUTE'
        )
        
        self._persist_portfolio()

    def purge_position(self, symbol: str, reason: str = "Leak Detected"):
        """
        Forcefully remove a position from the internal ledger (Purging a Leak).
        """
        with self.position_lock:
            matches = [vk for vk in self.positions.values() if vk.symbol == symbol]
            if matches:
                logger.info(f"[{self.name}] 🕳️ PURGING POSITION: {symbol} - Reason: {reason}")
                # Remove all positions for this symbol
                keys_to_del = [vk for vk, p in self.positions.items() if p.symbol == symbol]
                for k in keys_to_del:
                    del self.positions[k]
        
                self._persist_portfolio()
                if self.governor and hasattr(self.governor, 'reconcile_with_executor'):
                    self.governor.reconcile_with_executor(self.get_positions_snapshot())

    def reconcile_position_size(self, symbol: str, new_qty: float, reason: str = "Size Correction"):
        """
        Correct the internal ledger quantity to match reality.
        """
        with self.position_lock:
            # Find first matching symbol
            vk = next((k for k, p in self.positions.items() if p.symbol == symbol), None)
            if vk:
                old_qty = self.positions[vk].quantity
                logger.info(f"[{self.name}] ⚖️ RECONCILING SIZE: {symbol} {old_qty} -> {new_qty} - Reason: {reason}")
                self.positions[vk].quantity = new_qty
                self._persist_portfolio()
                if self.governor and hasattr(self.governor, 'reconcile_with_executor'):
                    self.governor.reconcile_with_executor(self.get_positions_snapshot())

    # === ATR-BASED STOP-LOSS (Recovery Plan 2026-03-12) ===
    def get_atr_for_symbol(self, symbol: str, period: int = 14) -> float:
        """
        Get ATR value for a symbol from latest market data.

        Returns:
            float: ATR value (price terms), or 2% of price as fallback
        """
        try:
            # Try to fetch from observer's latest data
            if hasattr(self, 'observer') and self.observer:
                df = self.observer.fetch_market_data(symbol=symbol, timeframe='15m', limit=50)
                if not df.empty:
                    high = df['high']
                    low = df['low']
                    close = df['close']

                    # Calculate True Range
                    tr1 = high - low
                    tr2 = abs(high - close.shift(1))
                    tr3 = abs(low - close.shift(1))
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

                    # Calculate ATR (Simple Moving Average of TR)
                    atr = tr.rolling(period).mean().iloc[-1]
                    return float(atr)
        except Exception as e:
            print(f"[{self.name}] ⚠️ ATR fetch failed for {symbol}: {e}")

        # Fallback: return 2% of price as rough estimate
        latest_prices = getattr(self, 'latest_prices', {})
        price = latest_prices.get(symbol, 100.0)
        return price * 0.02

    def calculate_atr_stop_loss(self, symbol: str, entry_price: float, direction: str,
                                atr_value: float = None) -> float:
        """
        Calculate dynamic stop-loss based on ATR.

        Args:
            symbol: Trading pair
            entry_price: Entry price
            direction: 'BUY' or 'SELL'
            atr_value: Pre-calculated ATR (optional)

        Returns:
            float: Stop-loss price
        """
        if atr_value is None or atr_value <= 0:
            atr_value = self.get_atr_for_symbol(symbol)

        # Get ATR multiplier from config
        atr_mult = getattr(config, 'PREDATOR_STOP_LOSS_ATR_MULT', 2.5)

        # Calculate stop distance
        stop_distance = atr_value * atr_mult

        # Apply direction
        if direction == 'BUY':
            stop_price = entry_price - stop_distance
        else:
            stop_price = entry_price + stop_distance

        # Apply min/max floors (prevent too-tight or too-wide stops)
        min_stop_pct = getattr(config, 'MIN_STOP_LOSS_PCT', 0.015)
        max_stop_pct = getattr(config, 'MAX_STOP_LOSS_PCT', 0.08)

        if direction == 'BUY':
            min_stop = entry_price * (1 - min_stop_pct)
            max_stop = entry_price * (1 - max_stop_pct)
            stop_price = max(min(stop_price, max_stop), min_stop)
        else:
            min_stop = entry_price * (1 + min_stop_pct)
            max_stop = entry_price * (1 + max_stop_pct)
            stop_price = min(max(stop_price, max_stop), min_stop)

        return stop_price

    def calculate_atr_take_profit(self, symbol: str, entry_price: float, direction: str,
                                  atr_value: float = None) -> float:
        """
        Calculate dynamic take-profit based on ATR (2:1 reward/risk).

        Returns:
            float: Take-profit price
        """
        if atr_value is None or atr_value <= 0:
            atr_value = self.get_atr_for_symbol(symbol)

        # Get ATR multiplier for TP (typically 2x the stop distance for 2:1 R/R)
        atr_mult = getattr(config, 'DEFAULT_TAKE_PROFIT_ATR_MULT', 4.0)

        # Calculate TP distance
        tp_distance = atr_value * atr_mult

        # Apply direction
        if direction == 'BUY':
            tp_price = entry_price + tp_distance
        else:
            tp_price = entry_price - tp_distance

        # === CHRONOS: EXPECTANCY REPAIR (PHASE 2) ===
        # Clamp TP distance to a minimum % so winners aren't harvested at dust levels
        # that get erased by fees/spread. Mirrors stop-loss min/max floors.
        min_tp_pct = getattr(config, 'MIN_TAKE_PROFIT_PCT', 0.01)  # 1.0% default
        max_tp_pct = getattr(config, 'MAX_TAKE_PROFIT_PCT', 0.25)  # 25% default safety cap

        if direction == 'BUY':
            min_tp = entry_price * (1 + min_tp_pct)
            max_tp = entry_price * (1 + max_tp_pct)
            tp_price = max(min(tp_price, max_tp), min_tp)
        else:
            min_tp = entry_price * (1 - min_tp_pct)
            max_tp = entry_price * (1 - max_tp_pct)
            tp_price = min(max(tp_price, max_tp), min_tp)

        return tp_price
    # === END ATR-BASED STOP-LOSS ===

    def sync_balance(self, confirmed_balance: float):
        """
        Force-update the internal balance to match Reality (Exchange).
        Overrides any stale DB state.
        FIX 2026-02-24: Added debounce and realistic fee estimation.
        """
        drift = confirmed_balance - self.balance_usd

        # FIX: Add debounce - ignore small drifts (< $0.50 or < 0.5%)
        drift_pct = abs(drift / self.balance_usd) if self.balance_usd > 0 else 0
        if abs(drift) < 0.50 and drift_pct < 0.005:
            print(f"[{self.name}] VERIFIED Ledger Balance: ${self.balance_usd:.2f} (drift ${drift:.2f} ignored)")
            return

        # Track cumulative drift with realistic fee estimation
        # FIX: Don't attribute all drift to fees - most is likely sync timing differences
        if abs(drift) > 0.50:
            # FIX 2026-02-28: Skip fee accumulation during startup grace period.
            # The first few syncs compare stale DB balance vs live exchange balance,
            # producing large phantom drift that is NOT real fees.
            self._startup_sync_count = getattr(self, '_startup_sync_count', 0) + 1
            is_startup_grace = self._startup_sync_count <= getattr(self, '_startup_grace_syncs', 3)

            # Hoist these so they are always defined (used below for today's fee tracking)
            recent_trade_count = 0
            fee_estimate = 0.0

            if is_startup_grace:
                # Startup drift — just silently update the balance baseline, no fee tracking
                print(f"[{self.name}] 🌅 STARTUP SYNC #{self._startup_sync_count}: Drift ${drift:.2f} attributed to DB/exchange baseline difference (no fee tracking).")
                
                # FIX 2026-03-01: Explicitly reset fee counters during startup to prevent phantom accumulation
                self.cumulative_fees_usd = 0.0
                self._today_fees_usd = 0.0

                # FIX 2026-03-04: Auto-reset fee halt inherited from a prior session
                if self._fee_halt_active:
                    self._fee_halt_active = False
                    print(f"[{self.name}] ✅ AUTO-RESET FEE HALT on startup (was set by prior TCT drift)")
            else:
                # Only track as fees if there were actual trades in this cycle
                recent_trade_count = len([p for p in self.positions.values() if hasattr(p, 'entry_timestamp')])

                if recent_trade_count > 0:
                    # FIX 2026-03-03: Use actual trade volume for fee estimation instead of arbitrary drift percentage
                    # Kraken futures taker fee is ~0.02-0.05%, use 0.05% conservative estimate
                    trade_volume = sum(abs(p.quantity * p.entry_price) for p in self.positions.values() 
                                      if hasattr(p, 'entry_timestamp') and abs(p.quantity) > 1e-9)
                    fee_estimate = trade_volume * 0.0005  # 0.05% taker fee
                    self.cumulative_fees_usd += fee_estimate
                    # Remaining drift is slippage + timing differences
                    self.cumulative_slippage_usd += max(0, abs(drift) - fee_estimate) * 0.3
                else:
                    # No trades - drift is likely funding payments or sync timing
                    # Track funding separately instead of lumping into slippage
                    self.cumulative_slippage_usd += abs(drift) * 0.2
                    # Initialize funding tracker if not exists
                    self.cumulative_funding_payments = getattr(self, 'cumulative_funding_payments', 0.0)
                    self.cumulative_funding_payments += abs(drift) * 0.5  # 50% likely funding/PnL

            self.last_ledger_sync_ts = time.time()

            # === FEE AUDIT FIX (2026-02-24) ===
            # Check for daily reset
            now = datetime.now(timezone.utc)
            if now.date() != self._last_fee_reset_date:
                old_fees = self.cumulative_fees_usd
                self._yesterday_fees_usd = self._today_fees_usd
                self.cumulative_fees_usd = 0.0
                self._today_fees_usd = 0.0
                self._last_fee_reset_date = now.date()
                self._fee_reset_count += 1
                print(f"[{self.name}] 💰 Daily Fee Reset: Yesterday ${old_fees:.2f} -> Today $0.00 (Reset #{self._fee_reset_count})")

            # FIX 2026-03-03: Fee decay mechanism - 50% decay per hour prevents permanent halt
            current_time = time.time()
            if hasattr(self, '_last_fee_decay_time'):
                time_elapsed = current_time - self._last_fee_decay_time
                decay_factor = 0.5 ** (time_elapsed / 3600)  # 50% per hour
                old_fees = self.cumulative_fees_usd
                self.cumulative_fees_usd *= decay_factor
                if old_fees - self.cumulative_fees_usd > 0.01:
                    print(f"[{self.name}] 📉 Fee Decay: ${old_fees:.2f} -> ${self.cumulative_fees_usd:.2f} ({decay_factor*100:.1f}% remaining)")
            self._last_fee_decay_time = current_time

            # Track today's fees (only applies outside startup grace, where fee_estimate > 0)
            if recent_trade_count > 0:
                self._today_fees_usd += fee_estimate

            # FIX 2026-03-03: Reset fee counter after successful equity verification (trust rebuild)
            # If equity is within 5% of starting and we have no recent trades, reduce fees by 50%
            if recent_trade_count == 0 and self.balance_usd > self.initial_capital * 0.95:
                if self.cumulative_fees_usd > 1.0:  # Only if there are fees to reduce
                    self.cumulative_fees_usd *= 0.5
                    print(f"[{self.name}] ✅ Fee Counter Reduced: ${self.cumulative_fees_usd:.2f} (50% decay on equity verification)")

            # FIX 2026-02-28: Compare actual exchange fees vs estimated drift-based fees
            total_actual_fees = sum(f['fee_usd'] for f in self._trade_fee_log)
            estimated_fees_from_drift = self.cumulative_fees_usd
            if total_actual_fees > 0:
                ratio = estimated_fees_from_drift / total_actual_fees if total_actual_fees > 0 else 0
                log_msg = f"[{self.name}] 📊 FEE AUDIT: Actual Exchange Fees=${total_actual_fees:.4f} | Est. from Drift=${estimated_fees_from_drift:.4f} (Ratio: {ratio:.2f}x)"
                if ratio > 2.0 or ratio < 0.5:
                    print(f"{log_msg} ⚠️ POTENTIAL DOUBLE-COUNTING DETECTED!")
                else:
                    print(log_msg)

            # Anomaly detection
            fee_pct = self.cumulative_fees_usd / self.balance_usd if self.balance_usd > 0 else 0
            if fee_pct > self._fee_anomaly_threshold:
                print(f"[{self.name}] 🚨 FEE ANOMALY: Cumulative fees ${self.cumulative_fees_usd:.2f} ({fee_pct*100:.1f}%) seems excessive. Audit required.")
            # ===================================

            # FIX: Add fee alert if exceeding threshold
            fee_pct = self.cumulative_fees_usd / self.initial_capital if self.initial_capital > 0 else 0
            if fee_pct > getattr(config, 'MAX_DAILY_FEES_PCT', 0.05):
                print(f"[{self.name}] ⚠️ FEE ALERT: Cumulative fees ${self.cumulative_fees_usd:.2f} ({fee_pct*100:.2f}%) exceed threshold!")

            # FIX 2026-03-03: Increased hard limit from 10% to 25% to prevent false halts
            # Fee decay mechanism should handle runaway fees instead of hard halt
            if fee_pct > getattr(config, 'MAX_DAILY_FEES_HARD_LIMIT', 0.25):
                print(f"[{self.name}] 🛑 FEE HALT: Cumulative fees ${self.cumulative_fees_usd:.2f} ({fee_pct*100:.2f}%) exceed HARD LIMIT! Trading HALTED.")
                self._fee_halt_active = True

            # FIX 2026-03-03: DEATH BY THOUSAND CUTS PROTECTION
            # Track cumulative drift (absolute value to catch both positive and negative)
            # FIX 2026-03-04: Skip drift accumulation during startup grace period.
            # Startup syncs compare stale DB balance vs live exchange — this is NOT real trading loss.
            if not is_startup_grace:
                # Only accumulate drift if there were recent trades (real trading activity)
                if recent_trade_count > 0:
                    self._session_drift += abs(drift) * 0.3  # Only 30% attributed to drift, rest is fees/slippage
                else:
                    # No trades = drift is funding/timing, not trading loss
                    self._startup_drift_ignored += abs(drift)
                
                # For backward compatibility, keep cumulative_drift but only from session activity
                self._cumulative_drift = self._session_drift

            # Calculate Total Cost of Trading (TCT)
            funding_pnl = getattr(self, 'cumulative_funding_payments', 0.0)
            total_cost_of_trading = (
                self.cumulative_fees_usd +
                self.cumulative_slippage_usd +
                abs(self._session_drift) +  # Only session drift, not startup phantom drift
                max(0, -funding_pnl)  # Only count negative funding as cost
            )
            tct_pct = total_cost_of_trading / self.initial_capital * 100 if self.initial_capital > 0 else 0

            # TCT Dashboard
            print(f"[{self.name}] 📊 TOTAL COST OF TRADING: ${total_cost_of_trading:.2f} ({tct_pct:.2f}%)")
            print(f"   ├─ Fees: ${self.cumulative_fees_usd:.2f}")
            print(f"   ├─ Slippage: ${self.cumulative_slippage_usd:.2f}")
            print(f"   ├─ Session Drift: ${abs(self._session_drift):.2f} (Startup Ignored: ${self._startup_drift_ignored:.2f})")
            print(f"   └─ Negative Funding: ${max(0, -funding_pnl):.2f}")
            
            # === FIX 2026-03-04: SLIPPAGE BY ASSET ===
            # Show top 3 worst offenders for slippage
            if hasattr(self, '_slippage_by_asset') and self._slippage_by_asset:
                sorted_assets = sorted(
                    self._slippage_by_asset.items(),
                    key=lambda x: x[1].get('total_slippage', 0),
                    reverse=True
                )[:3]  # Top 3
                if sorted_assets:
                    print(f"[{self.name}] 📊 TOP SLIPPAGE OFFENDERS:")
                    for asset, data in sorted_assets:
                        if data.get('total_slippage', 0) > 0:
                            print(f"   - {asset}: ${data['total_slippage']:.2f} ({data['count']} trades, avg {data.get('avg_slippage_pct', 0)*100:.2f}%)")
            
            # TCT Alerts
            if tct_pct > 10.0:
                print(f"[{self.name}] 🚨 TCT CRITICAL: Total costs exceed 10% of capital! Consider stopping.")
            if tct_pct > 15.0:
                print(f"[{self.name}] 🛑 TCT HALT: Total costs {tct_pct:.2f}% exceed 15% limit! Trading HALTED.")
                self._fee_halt_active = True  # Reuse fee halt mechanism

            # Drift alert
            drift_threshold = 0.05 * self.initial_capital  # 5% of equity
            if abs(self._cumulative_drift) > drift_threshold:
                print(f"[{self.name}] 🚨 DRIFT ALERT: Cumulative drift ${abs(self._cumulative_drift):.2f} exceeds 5% threshold!")

            # Funding PnL alert
            if funding_pnl < -0.02 * self.initial_capital:
                print(f"[{self.name}] 🚨 FUNDING ALERT: Negative funding PnL ${funding_pnl:.2f} exceeds 2% threshold!")

            # FIX 2026-03-03: Include funding payments in drift analysis for complete picture
            funding_display = getattr(self, 'cumulative_funding_payments', 0.0)
            print(f"[{self.name}] 📊 DRIFT ANALYSIS: Drift=${drift:.2f} ({drift_pct*100:.2f}%) | Cumulative Fees=${self.cumulative_fees_usd:.2f} | Cumulative Slippage=${self.cumulative_slippage_usd:.2f} | Funding PnL=${funding_display:.2f}")

            self.balance_usd = confirmed_balance
            self.initial_capital = confirmed_balance  # Update basis too
            
            # Update Governor with real balance too
            if self.governor:
                self.governor.balance = confirmed_balance
                self.governor.available_balance = confirmed_balance
            self._persist_portfolio()
        else:
             print(f"[{self.name}] VERIFIED Ledger Balance Verified: ${self.balance_usd:.2f}")

    def force_full_sync(self):
        """
        Perform a complete synchronization with the exchange to resolve divergences.
        """
        self.sync_with_exchange(mode='HARD')

    def gc_reconcile_positions(self) -> List[str]:
        """
        Garbage Collector: Full bidirectional position sync with exchange.
        Returns list of ghost positions that were zeroed out.
        """
        before_keys = set(self.positions.keys())
        self.sync_with_exchange(mode='SOFT')
        after_keys = set(self.positions.keys())

        ghosts_found = list(before_keys - after_keys)
        if ghosts_found:
            logger.info(f"[GC Monitor] ✅ Position Reconciliation: {len(ghosts_found)} ghosts cleared: {ghosts_found}")
        else:
            logger.debug(f"[GC Monitor] ✅ Position Reconciliation: Clean - no discrepancies.")

        return ghosts_found

    def reset_fee_halt(self):
        """
        FIX 2026-02-28: Manual reset for fee halt after user intervention.
        Call this after reviewing fee anomaly and deciding to resume trading.
        """
        if self._fee_halt_active:
            self._fee_halt_active = False
            logger.info(f"[{self.name}] ✅ FEE HALT RESET: Trading resumed after manual intervention.")
            print(f"[{self.name}] ✅ FEE HALT RESET: Trading resumed. Cumulative fees: ${self.cumulative_fees_usd:.2f}")
        return self._fee_halt_active


    def get_portfolio_stats(self, prices: Dict[str, float] = None) -> Dict[str, float]:
        """
        Single Source of Truth for Portfolio Risk Metrics.
        Returns exposure, margin usage, and solvency ratios.
        Args:
            prices: Optional dict of current prices {symbol: price}. Uses self.latest_prices if None.
        """
        # 1. Update Prices if provided
        if prices:
            self.latest_prices.update(prices)
            
        # 2. Calculate Aggregates
        total_exposure = 0.0
        margin_used = 0.0
        
        # Use Actuator's equity if available as primary truth for balance
        equity = self.balance_usd
        if self.actuator:
             equity = self.actuator.get_equity() or self.balance_usd
             
        active_count = 0
        with self.position_lock: # Acquire lock for reading positions
            for p in self.positions.values():
                if abs(p.quantity) < 1e-8: continue
                active_count += 1
                
                # Use current price if available, else entry
                # self.latest_prices is updated by sync_with_exchange and potentially passed in
                price = self.latest_prices.get(p.symbol, p.entry_price)
                if price <= 0: price = p.entry_price
                
                notional = abs(p.quantity) * price
                total_exposure += notional
                # Used Margin = Notional / Leverage
                margin_used += notional / max(1.0, p.leverage)

                # FIX 2026-03-21: Track MFE/MAE (Max Favorable/Adverse Excursion)
                if p.entry_price > 0:
                    pnl_pct = p.get_pnl_pct(price)
                    if pnl_pct > p.mfe:
                        p.mfe = pnl_pct
                    if pnl_pct < p.mae:
                        p.mae = pnl_pct

        # 3. Derived Metrics
        margin_free = max(0.0, equity - margin_used)
        # Prevent division by zero
        margin_level = (equity / margin_used) if margin_used > 0.01 else 999.0
        
        return {
            'equity': equity,
            'balance': self.balance_usd,
            'total_exposure': total_exposure,
            'margin_used': margin_used,
            'margin_free': margin_free,
            'margin_level': margin_level,
            'position_count': active_count
        }

    def get_execution_summary(self) -> dict:
        """Returns a high-level summary of execution status and portfolio health."""
        stats = self.get_portfolio_stats()
        
        return {
            'balance': stats['balance'],
            'equity': stats['equity'],
            'margin_used': stats['margin_used'],
            'margin_available': stats['margin_free'], 
            'active_positions': stats['position_count'],
            'ledger_size': len(self.ledger)
        }



    def get_balance_details(self) -> Tuple[float, float]:
        """
        Returns (Total Equity, Free Margin).
        Used for Solvency Checks.
        """
        if self.actuator:
            # Equity = Total Net Worth (use specific equity method)
            equity = self.actuator.get_equity()
            # Free = Account Balance (Cash/Available Margin)
            free = self.actuator.get_account_balance()
            return equity, free
        else:
            return self.balance_usd, self.balance_usd

    @property
    def total_value(self) -> float:
        """Alias for Portfolio Value (Equity) to support Trader access."""
        return self.get_portfolio_value()

    # check_stop_loss_take_profit removed. 
    # Logic centralized in ExitGuardianHolon to prevent redundancy and race conditions.

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

        # FIX BUG-004: Compliance reductions must ALWAYS execute - skip disposition logic
        # These are risk-reducing trades that should never be blocked
        signal_metadata = signal.metadata if hasattr(signal, 'metadata') and signal.metadata else {}
        bypass_validation = signal_metadata.get('bypass_validation', False)
        is_compliance_reduce = signal_metadata.get('reason', '') == 'COMPLIANCE_REDUCE'
        
        if bypass_validation or is_compliance_reduce:
            # Compliance reduction - always execute, skip all safety checks
            action = 'EXECUTE'
            adjusted_size = signal.size
            if is_compliance_reduce:
                print(f"[{self.name}] ⚖️ COMPLIANCE REDUCE: Bypassing disposition logic for risk-reducing trade")
        else:
            # Apply disposition logic based on continuous Sigmoid function
            # Autonomy = 1 / (1 + e^(k * (Entropy - Threshold)))
            # k=5 (steepness), Threshold=0.75 (calibrated for live data)
            #
            # CALIBRATION NOTE (Phase 11):
            # - Original threshold: 2.0 (for backtest data with max entropy ~2.25)
            # - Live data max entropy: ~1.85
            # - Adjusted to 0.75 to enable HALT/REDUCE triggers        # - Live data max entropy: ~1.85
            # - Gaussian Noise: ~1.40
            # - Phase 34 Tuned Threshold: 1.1 (Midpoint of new Transition zone)
            # - Phase 35 Update: Raised to 1.5 to handle high-entropy assets (DOT, XTZ showing 1.8-1.9)
            # - UNLEASHED (Phase 36): Raised to 2.2 to basically disable the Halt. (Max Entropy ~2.3)
            k = 5.0
            threshold = 2.2  # UNLEASHED MODE: High tolerance for Chaos

            # Ranges:
            # Entropy < 1.0 (Ordered) -> Autonomy > 0.7
            # Entropy > 1.35 (Chaotic) -> Autonomy < 0.3

            autonomy = 1.0 / (1.0 + math.exp(k * (entropy_score - threshold)))

            # Integration is the inverse
            integration = 1.0 - autonomy

            self.disposition = Disposition(autonomy=autonomy, integration=integration)

            # Map continuous autonomy to discrete actions for Ledger/Protocol compliance
            # Autonomy > 0.6 -> EXECUTE (High Independence)
            # Autonomy < 0.4 -> HALT (High Safety)
            # 0.4 <= Autonomy <= 0.6 -> REDUCE (Balanced)

            if autonomy > 0.5:  # Relaxed from 0.6 to allow more entries
                action = 'EXECUTE'
                adjusted_size = signal.size

                # --- SAFETY NET (Phase 5) ---
                if self.governor and hasattr(self.governor, 'check_solvency'):
                     trade_meta = {
                         'symbol': signal.symbol,
                         'size': signal.size, # Quantity
                         'price': signal.price,
                         'direction': signal.direction
                     }
                     if not self.governor.check_solvency(trade_meta):
                         action = 'HALT'
                         adjusted_size = 0.0
                         print(f"[{self.name}] 🛡️ SAFETY NET: Governor Vetoed Trade Solvency. Action -> HALT.")
                # ----------------------------

            elif autonomy < 0.05:  # Very relaxed - only HALT in extreme chaos
                action = 'HALT'
                adjusted_size = 0.0
                print(f"[{self.name}] ⚠️ HALT: Autonomy too low ({autonomy:.3f}), Entropy: {entropy_score:.2f}")

            else:
                # SOFT-HALT / REDUCE Range (0.05 - 0.5 autonomy)
                action = 'REDUCE'
                # Scale participation: 25% minimum, up to 100% near 0.5
                adjusted_size = signal.size * max(0.25, autonomy)

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
            block_hash=block.hash,
            entropy_score=entropy_score
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
            block_hash=block.hash,
            entropy_score=entropy_score
        )

    def _get_virt_key(self, symbol: str, strategy: str) -> str:
        """Helper to get the correct virtual key for a symbol and strategy."""
        virt_key = f"{symbol}:{strategy}"
        
        # Deduce pool if not found
        with self.position_lock: # Acquire lock for reading positions
            if virt_key not in self.positions or abs(self.positions[virt_key].quantity) < 1e-9:
                matches = [k for k, p in self.positions.items() if p.symbol == symbol and abs(p.quantity) > 1e-9]
                if len(matches) == 1:
                    virt_key = matches[0]
                    logger.debug(f"[{self.name}] 🧠 DEDUCED POOL: {symbol} -> {virt_key}")
        
        return virt_key

    def _get_minimum_order_quantity(self, symbol: str) -> float:
        """
        FIX 2026-02-28: Get minimum order quantity for a symbol from config or exchange market data.
        Prevents order rejections due to precision/minimum amount violations.
        
        Returns:
            float: Minimum quantity required, or 0.01 default if unknown
        """
        # FIX 2026-03-01: Use config MIN_TRADE_QTY first (most reliable)
        base_asset = symbol.split('/')[0]
        
        # Check config MIN_TRADE_QTY dictionary
        min_qty = getattr(config, 'MIN_TRADE_QTY', {}).get(base_asset, 0.0)
        if min_qty > 0:
            return min_qty
        
        # Fallback to exchange market data if not in config
        if not self.market or not hasattr(self.market, 'exchange'):
            recent_price = getattr(self, 'latest_prices', {}).get(symbol, 0.0)
            if recent_price > 0:
                return getattr(config, 'MIN_ORDER_VALUE', 5.0) / recent_price
            return 0.01  # Conservative default
        
        try:
            # Map internal symbol to exchange format
            exec_symbol = config.KRAKEN_SYMBOL_MAP.get(symbol, symbol)
            
            # Fetch market data for this symbol
            markets = getattr(self.market.exchange, 'markets', {})
            if not markets:
                # Try to fetch markets
                self.market.exchange.load_markets()
                markets = getattr(self.market.exchange, 'markets', {})
            
            market = markets.get(exec_symbol)
            if not market:
                # Try alternative lookup
                for k, v in markets.items():
                    if k.startswith(base_asset):
                        market = v
                        break
            
            if market:
                # Get minimum amount from limits or precision
                limits = market.get('limits', {})
                amount_limits = limits.get('amount', {})
                min_amount = amount_limits.get('min', 0.0)
                
                # Also check precision as fallback
                if min_amount <= 0:
                    precision = market.get('precision', {})
                    amount_precision = precision.get('amount', 0.0)
                    if amount_precision > 0:
                        min_amount = amount_precision
                
                if min_amount > 0:
                    return min_amount
            
            # Dynamic fallback using config.MIN_ORDER_VALUE and recent price
            recent_price = getattr(self, 'latest_prices', {}).get(symbol, 0.0)
            if recent_price > 0:
                return getattr(config, 'MIN_ORDER_VALUE', 5.0) / recent_price
                
            return 0.01  # Ultimate Fallback if price is zero
            
        except Exception as e:
            logger.debug(f"[{self.name}] Could not fetch min qty for {symbol}: {e}")
            recent_price = getattr(self, 'latest_prices', {}).get(symbol, 0.0)
            if recent_price > 0:
                return getattr(config, 'MIN_ORDER_VALUE', 5.0) / recent_price
            return 0.01  # Conservative default

    def _attempt_margin_release(self, required_margin_usd: float) -> bool:
        """
        Attempts to free up margin by closing a slice of an existing profitable position.
        """
        if not self.market:
            return False
            
        try:
            free_margin = self.market.get_balance()
            target_margin = required_margin_usd + 1.0 # Buffer
            shortfall = target_margin - free_margin
            
            if shortfall <= 0: return True
            
            logger.info(f"[{self.name}] 🔓 MARGIN RELEASE: Shortfall ${shortfall:.2f}. Scanning candidates...")
            
            # Find profitable candidates
            candidates = []
            with self.position_lock: # Acquire lock for reading positions
                for vk, pos in self.positions.items():
                    if abs(pos.quantity) < 1e-8: continue
                    
                    price = self.latest_prices.get(pos.symbol, pos.entry_price)
                    pnl_pct = pos.get_pnl_pct(price)
                    
                    if pnl_pct < 0.005: continue # 0.5% min profit
                    
                    notional = abs(pos.quantity) * price
                    if notional < (config.MIN_ORDER_VALUE * 3): continue
                    
                    candidates.append({
                        'vk': vk,
                        'notional': notional,
                        'pos': pos,
                        'pnl': pnl_pct
                    })
                
            candidates.sort(key=lambda x: x['notional'])
            
            if not candidates:
                logger.warning(f"[{self.name}] ❌ MARGIN RELEASE: No eligible profitable candidates found.")
                return False
                
            # Release 2x shortfall or 30% of position
            target = candidates[0]
            qty_to_close = (shortfall * 2.0 / target['notional']) * abs(target['pos'].quantity)
            qty_to_close = min(qty_to_close, abs(target['pos'].quantity) * 0.5)
            
            logger.info(f"[{self.name}] 🔪 RELEASING MARGIN: Closing {qty_to_close:.4f} {target['vk']} (~${shortfall:.2f}+)")
            
            self.market.place_order(
                symbol=target['pos'].symbol,
                direction='SELL' if target['pos'].is_long else 'BUY',
                quantity=qty_to_close,
                price=0,
                order_type='market',
                leverage=target['pos'].leverage,
                reduce_only=True,
                urgent=True
            )
            time.sleep(2.0)
            return True
        except Exception as e:
            logger.error(f"[{self.name}] ❌ Margin release failed: {e}")
            return False

    def _validate_trade_constraints(self, signal: TradeSignal) -> bool:
        """Checks for pending orders and stale states before executing."""
        if not self.market: return True
        
        WORKING_ORDER_TIMEOUT = 30
        current_time = time.time()
        has_pending = False
        
        pending_orders = getattr(self.market, 'pending_orders', [])
        for o in list(pending_orders):
            if o.get('symbol') == signal.symbol:
                age = current_time - o.get('timestamp', current_time)
                if age > WORKING_ORDER_TIMEOUT:
                    logger.info(f"[{self.name}] ⏰ STALE WORKING ORDER: {signal.symbol} ({age:.0f}s old). Clearing.")
                    try: pending_orders.remove(o)
                    except: pass
                else:
                    has_pending = True

        if has_pending:
            if signal.metadata.get('reason') == 'STOP_LOSS':
                logger.warning(f"[{self.name}] 🚨 EMERGENCY STOP LOSS: Overriding working order for {signal.symbol}.")
                return True
            logger.debug(f"[{self.name}] ⏳ Working Order exists for {signal.symbol}. Skipping.")
            return False
            
        return True

    def _calculate_execution_qty(self, decision: TradeDecision, current_price: float) -> Tuple[float, float]:
        """Calculates final execution quantity and leverage based on Governor and solvency."""
        symbol = decision.original_signal.symbol
        direction = decision.original_signal.direction

        # Evaluate robust exit logic (match fee halt bypass logic)
        sig_meta = decision.original_signal.metadata or {}
        sig_reason = str(sig_meta.get('reason', '')).upper()

        # === FIX 2026-03-12: Better Entry/Exit Detection ===
        # Whale entries should NOT be treated as exits
        is_entry_signal = sig_reason in ('WHALE_BID_WALL', 'WHALE_ACCUMULATION', 'WHALE_SQUEEZE',
                                          'WHALE_SCALPER', 'WHALE_SCALPER_FORCED', 'DIP',
                                          'PACK_HUNT', 'STRUCTURAL_RESONANCE', 'TREND',
                                          'FUNDING_ARB', 'VOLATILITY_SQUEEZE', 'SYSTEM')

        is_exit_trade = (
            sig_meta.get('is_exit', False) or
            sig_meta.get('reduce_only', False) or
            (sig_reason in ('STOP_LOSS', 'TAKE_PROFIT', 'EMERGENCY_STOP', 'COMPLIANCE_REDUCE',
                           'ADAPTIVE_STOP', 'VOL_TRAIL', 'VOL_STOP', 'MEAN_REV',
                           'THESIS', 'PROFIT_PRESERVE', 'TREND_EXPIRY', 'TIME_EXIT',
                           'RAPID_TP', 'NORMAL_TP', 'RUNNER_TP', 'SATELLITE_TP',
                           'SATELLITE_SL', 'MONTE_CARLO_CLOSURE', 'MANAGEMENT_MODE_LOSS_CUT')
             and not is_entry_signal)  # Whale entries override exit detection
        )

        # Debug log for entry/exit classification
        logger.info(f"[{self.name}] 🐛 DEBUG: {symbol} sig_reason={sig_reason} is_entry={is_entry_signal} is_exit={is_exit_trade} action={decision.action}")
        # === END FIX ===

        # === FIX 2026-03-12 #2: Handle REDUCE action for entries ===
        # Whale signals with action='REDUCE' should still be treated as entries if is_entry_signal=True
        is_forced_entry = is_entry_signal and not is_exit_trade

        # Entries
        # Updated condition: Also allow REDUCE action if it's a forced entry (whale signal)
        if (decision.action in ['EXECUTE', 'BUY', 'SELL', 'REDUCE'] and not is_exit_trade) or is_forced_entry:
            requested_qty = decision.adjusted_size
            if self.governor:
                self.governor.update_balance(self.get_portfolio_value())
                is_approved, safe_qty, leverage = self.governor.receive_message(self, {
                    'type': 'VALIDATE_TRADE',
                    'price': current_price,
                    'symbol': symbol,
                    'direction': direction,
                    'crisis_score': decision.entropy_score,
                    'strategy': decision.original_signal.metadata.get('strategy', 'DIRECTIONAL'),
                    'metadata': decision.original_signal.metadata
                })
                # FIX 2026-02-28: Check both approval flag AND quantity
                if not is_approved or safe_qty <= 0:
                    logger.warning(f"[{self.name}] ⚠️ GOVERNOR REJECT: {symbol} (approved={is_approved}, qty={safe_qty})")
                    return 0.0, 1.0
                exec_qty = min(requested_qty, safe_qty)
            else:
                exec_qty = requested_qty
                leverage = 1.0

            # FIX 2026-02-28: Validate quantity before solvency check
            if exec_qty <= 0:
                logger.warning(f"[{self.name}] ⚠️ INVALID QTY: {symbol} calculated qty={exec_qty:.8f}")
                return 0.0, 1.0

            # Minimum Quantity Check for Entries
            min_qty = self._get_minimum_order_quantity(symbol)
            if exec_qty < min_qty:
                # FIX 2026-03-01: Round up to minimum instead of rejecting for nano accounts
                is_nano_account = self.balance_usd < 100.0
                if is_nano_account and min_qty > 0:
                    logger.info(f"[{self.name}] 📈 NANO FIX: {symbol} qty={exec_qty:.8f} rounded up to min={min_qty}")
                    exec_qty = min_qty
                else:
                    logger.warning(f"[{self.name}] ⚠️ BELOW MINIMUM: {symbol} qty={exec_qty:.8f} < min={min_qty}. Rejecting.")
                    return 0.0, 1.0

            # Solvency
            avail_capital = self.market.get_buying_power(leverage=5.0) if self.market else self.balance_usd * 5.0
            safe_capital = avail_capital * 0.90
            # Reserve estimated fees and slippage when sizing to avoid underestimating notional
            fee_est = getattr(config, 'ESTIMATED_FEE_PCT', 0.001)  # 0.1% default
            slip_est = getattr(config, 'ESTIMATED_SLIPPAGE_PCT', 0.001)  # 0.1% default
            safe_capital = safe_capital * (1.0 - fee_est - slip_est)
            notional_req = exec_qty * current_price

            if notional_req > safe_capital:
                exec_qty = safe_capital / current_price
                if (exec_qty * current_price) < config.MIN_ORDER_VALUE:
                    if self._attempt_margin_release(config.MIN_ORDER_VALUE / (leverage or 5.0)):
                        return self._calculate_execution_qty(decision, current_price)
                    return 0.0, 1.0
            # === VOLATILITY TARGET SIZING ===
            # If the signal specifies a `target_vol` (annual fraction), prefer volatility-sized qty
            sig_target_vol = None
            try:
                sig_target_vol = float(sig_meta.get('target_vol')) if sig_meta and sig_meta.get('target_vol') is not None else None
            except Exception:
                sig_target_vol = None

            if sig_target_vol and sig_target_vol > 0:
                vol_qty = self.compute_size_by_volatility(symbol, current_price, sig_target_vol)
                if vol_qty and vol_qty > 0:
                    logger.info(f"[{self.name}] ⚖️ VOL-SIZED: {symbol} override qty {exec_qty:.6f} -> {vol_qty:.6f} using target_vol={sig_target_vol}")
                    exec_qty = vol_qty
            return exec_qty, leverage
            
        # Exits
        # FIX 2026-03-01 #7: Improved position lookup for exits with strategy fallback
        virt_key = self._get_virt_key(symbol, decision.original_signal.metadata.get('strategy', 'DIRECTIONAL'))
        with self.position_lock:
            pos = self.positions.get(virt_key)
        
        # FIX: If position not found with strategy key, search all positions for symbol
        if not pos:
            for vk, p in self.positions.items():
                if p.symbol == symbol and abs(p.quantity) > 1e-9:
                    pos = p
                    virt_key = vk
                    logger.warning(f"[{self.name}] 🔄 EXIT QTY FIX: Found {symbol} position via fallback search (key={virt_key})")
                    break
        
        if not pos:
            logger.warning(f"[{self.name}] ⚠️ EXIT QTY=0: No position found for {symbol} (expected exit)")
            
            # 🧹 PHANTOM PURGE: If Guardian/Trader thinks it has a position but Executor doesn't,
            # we must aggressively clear it from all tracking structures to break the retry loop.
            logger.warning(f"[{self.name}] 🧹 PHANTOM PURGE: Automatically clearing {symbol} from internal tracking")
            with self.position_lock:
                if virt_key in self.positions:
                    del self.positions[virt_key]
                if symbol in self.held_assets:
                    del self.held_assets[symbol]
                if symbol in self.position_metadata:
                    del self.position_metadata[symbol]
                    
            return 0.0, 1.0

        if decision.original_signal.metadata.get('is_percent'):
            exec_qty = abs(pos.quantity) * decision.adjusted_size
        else:
            exec_qty = min(decision.adjusted_size, abs(pos.quantity))

        if decision.action == 'EXIT': exec_qty = abs(pos.quantity)

        # FIX 2026-03-01: Ensure exits meet minimum quantity for exchange routing
        min_qty = self._get_minimum_order_quantity(symbol)
        if 0 < exec_qty < min_qty:
            logger.info(f"[{self.name}] 📉 EXIT QTY ADJUST: {symbol} qty={exec_qty:.6f} rounded to min={min_qty} for closure")
            exec_qty = min_qty

        
        # If an exit/close, we may want to round the exec_qty to exchange precision
        # FIX 2026-03-01: Ensure exits meet minimum quantity for exchange routing
        min_qty = self._get_minimum_order_quantity(symbol)
        if 0 < exec_qty < min_qty:
            logger.info(f"[{self.name}] 📉 EXIT QTY ADJUST: {symbol} qty={exec_qty:.6f} rounded to min={min_qty} for closure")
            exec_qty = min_qty

        # Enforce exchange precision rounding if possible (avoid actuator rounding surprises)
        try:
            exec_sym = config.KRAKEN_SYMBOL_MAP.get(symbol, symbol)
            if self.market and hasattr(self.market, 'exchange') and hasattr(self.market.exchange, 'amount_to_precision'):
                qty_str = self.market.exchange.amount_to_precision(exec_sym, exec_qty)
                exec_qty = float(qty_str)
        except Exception:
            pass

        return exec_qty, pos.leverage

    def _update_state_post_fill(self, vk: str, fill: dict, leverage: float, strategy: str, sig_meta: dict = None):
        """Updates internal position state and balance following a successful fill."""
        symbol = fill['symbol']
        actual_qty = fill['filled_qty']
        actual_price = fill['price']
        direction = fill['direction']
        # FIX 2026-02-28: Capture and log actual fee from fill data
        actual_fee_usd = fill.get('fee_usd', 0.0)

        realized_pnl = 0.0

        # Pre-evaluate whether this is a closing fill (pos may be None for entries)
        pos = self.positions.get(vk)
        is_closing = (direction == 'SELL' and pos and pos.is_long) or (direction == 'BUY' and pos and pos.is_short)

        if actual_fee_usd > 0:
            # Log per-trade fee for double-counting investigation
            self._trade_fee_log.append({
                'symbol': symbol,
                'side': direction,
                'fee_usd': actual_fee_usd,
                'timestamp': fill.get('timestamp', datetime.now(timezone.utc).isoformat()),
                'source': 'exchange',
                'is_exit': is_closing
            })
            # Trim log if too large
            if len(self._trade_fee_log) > self._max_fee_log_size:
                self._trade_fee_log.pop(0)
            logger.info(f"[{self.name}] 💰 TRADE FEE: {symbol} {direction} Fee=${actual_fee_usd:.4f} (exchange)")

        with self.position_lock: # Acquire lock for all position modifications
            pos = self.positions.get(vk)

            if is_closing:
                # PnL Calculation
                realized_pnl = pos.get_pnl_usd(actual_price) * (actual_qty / abs(pos.quantity))
                self.balance_usd += realized_pnl

                # Record trade result for win rate tracking
                self._record_trade_result(
                    symbol=symbol,
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=actual_price,
                    quantity=abs(actual_qty),
                    pnl_usd=realized_pnl,
                    entry_time=pos.entry_timestamp,
                    exit_time=datetime.now(timezone.utc).isoformat(),
                    strategy=pos.strategy,
                    mfe=getattr(pos, 'mfe', 0.0),
                    mae=getattr(pos, 'mae', 0.0),
                    exit_reason=pos.metadata.get('exit_reason', 'NORMAL'),
                    entropy_score=getattr(pos, 'entry_entropy', None),
                    regime=getattr(pos, 'entry_regime', None),
                    conviction=getattr(pos, 'entry_conviction', None),
                    quality_score=getattr(pos, 'quality_score', None),
                    is_whitelisted=getattr(pos, 'is_whitelisted', False),
                )

                pos.quantity -= (actual_qty if direction == 'SELL' else -actual_qty)
                pos.quantity = round(pos.quantity, 8)
                if abs(pos.quantity) < 1e-8:
                    del self.positions[vk]
                    # FIX 2026-03-21: Clean up ALL legacy tracking dicts on full close
                    # Without this, entry_prices stays stale and guardian fires phantom TREND EXPIRY
                    if symbol in self.entry_prices:
                        del self.entry_prices[symbol]
                    if symbol in self.entry_timestamps:
                        del self.entry_timestamps[symbol]
                    if symbol in self.held_assets:
                        del self.held_assets[symbol]
                    if symbol in self.position_metadata:
                        del self.position_metadata[symbol]
                logger.info(f"[{self.name}] 📉 EXIT {symbol}: Realized ${realized_pnl:.2f}")
                # Deduct actual exchange fee from account balance to reflect net PnL
                if actual_fee_usd and actual_fee_usd > 0:
                    self.balance_usd -= actual_fee_usd
                    logger.info(f"[{self.name}] 📉 EXIT FEE SUBTRACTED: ${actual_fee_usd:.4f} | New Balance=${self.balance_usd:.2f}")
            else:
                # Entry / Stacking
                if not pos:
                    # === ATR-BASED STOP-LOSS (Recovery Plan 2026-03-12) ===
                    # Calculate stop loss and take profit prices using ATR
                    import config

                    # Fetch ATR for this symbol
                    atr_value = self.get_atr_for_symbol(symbol)

                    # Calculate ATR-based stops
                    stop_loss_price = self.calculate_atr_stop_loss(symbol, actual_price, direction, atr_value)
                    take_profit_price = self.calculate_atr_take_profit(symbol, actual_price, direction, atr_value)

                    # Strategy-specific override: use oracle-provided structural SL if present.
                    # This is critical for MARKET_OPEN_FVG where ATR-based stops can be too wide,
                    # creating the observed ~-3% tail losses.
                    sig_meta = sig_meta or {}
                    sig_strategy = str(sig_meta.get('strategy', '') or '').upper()
                    sl_override = sig_meta.get('sl_price')
                    if sig_strategy == 'MARKET_OPEN_FVG' and isinstance(sl_override, (int, float)) and float(sl_override) > 0:
                        sl_override = float(sl_override)
                        if direction == 'BUY' and sl_override < actual_price:
                            stop_loss_price = sl_override
                        elif direction != 'BUY' and sl_override > actual_price:
                            stop_loss_price = sl_override

                    # Log ATR value for diagnostics
                    stop_distance_pct = abs(actual_price - stop_loss_price) / actual_price * 100
                    tp_distance_pct = abs(take_profit_price - actual_price) / actual_price * 100

                    logger.info(f"[{self.name}] 📊 ATR STOP CALC: {symbol} | ATR=${atr_value:.4f} | SL Dist={stop_distance_pct:.2f}% | TP Dist={tp_distance_pct:.2f}%")
                    # === END ATR-BASED STOP-LOSS ===

                    self.positions[vk] = Position(
                        symbol=symbol,
                        virt_key=vk,
                        direction=direction,
                        quantity=actual_qty if direction == 'BUY' else -actual_qty,
                        entry_price=actual_price,
                        entry_timestamp=datetime.now(timezone.utc).isoformat(),
                        leverage=leverage,
                        strategy=strategy,
                        stop_loss=stop_loss_price,
                        take_profit=take_profit_price,
                        metadata=fill.get('metadata', {}),
                        # 2026-03-21: Entry context for winning pattern analysis
                        entry_entropy=sig_meta.get('entropy') if sig_meta else None,
                        entry_regime=sig_meta.get('regime') if sig_meta else None,
                        entry_conviction=sig_meta.get('conviction') if sig_meta else None,
                        entry_strategy=sig_meta.get('strategy') if sig_meta else None,
                        quality_score=sig_meta.get('quality_score') if sig_meta else None,
                        is_whitelisted=bool(sig_meta.get('is_whitelisted', False)) if sig_meta else False,
                    )

                    # Append initial stack record for traceability
                    try:
                        stack_entry = {
                            'entry_price': actual_price,
                            'quantity': actual_qty,
                            'entry_time': time.time(),
                            'stack_id': 1
                        }
                        self.positions[vk].stacks.append(stack_entry)
                    except Exception:
                        pass

                    # --- LEGACY/INTEGRATION: Ensure held_assets and position_metadata
                    # reflect the actual filled quantity reported by the exchange.
                    try:
                        self.held_assets[symbol] = actual_qty if direction == 'BUY' else -actual_qty
                        # Update/insert position_metadata entry
                        pm = self.position_metadata.get(symbol, {})
                        pm.update({
                            'symbol': symbol,
                            'direction': direction,
                            'quantity': abs(actual_qty),
                            'entry_price': actual_price,
                            'entry_timestamp': datetime.now(timezone.utc).isoformat(),
                            'leverage': leverage,
                            'strategy': strategy
                        })
                        self.position_metadata[symbol] = pm
                    except Exception as _e:
                        logger.warning(f"[{self.name}] ⚠️ Failed to sync held_assets/position_metadata for {symbol}: {_e}")

                    # Log the stop loss and take profit for this position
                    logger.info(f"[{self.name}] 🛡️ ATR STOP LOSS SET: {symbol} {direction} {actual_qty} @ {actual_price:.4f} | SL: {stop_loss_price:.4f} ({stop_distance_pct:.2f}%) | TP: {take_profit_price:.4f} ({tp_distance_pct:.2f}%)")

                    # Place actual stop loss and take profit orders on the exchange
                    # Try to place them as a bracket (native bracket/reduce-only) when supported
                    bracket_res = self.place_bracket_orders(symbol, direction, actual_qty, stop_loss_price, take_profit_price, leverage)
                    if isinstance(bracket_res, dict) and ('sl' in bracket_res or 'tp' in bracket_res):
                        sl_res = bracket_res.get('sl')
                        tp_res = bracket_res.get('tp')
                    else:
                        # FIX 2026-03-19: Place TP before SL on Kraken Futures to avoid wouldNotReducePosition error
                        if 'kraken' in str(self.actuator.exchange.id).lower() if self.actuator and hasattr(self.actuator, 'exchange') else False:
                            tp_res = self.place_take_profit_order(symbol, direction, actual_qty, take_profit_price)
                            sl_res = self.place_stop_loss_order(symbol, direction, actual_qty, stop_loss_price)
                        else:
                            sl_res = self.place_stop_loss_order(symbol, direction, actual_qty, stop_loss_price)
                            tp_res = self.place_take_profit_order(symbol, direction, actual_qty, take_profit_price)

                    # Record SL/TP order ids for OCO management and reconciliation
                    try:
                        if isinstance(sl_res, dict):
                            sl_id = sl_res.get('order_id') or sl_res.get('id')
                            self.positions[vk].metadata['sl_order_id'] = sl_id
                            self.position_metadata[symbol]['sl_order_id'] = sl_id
                        if isinstance(tp_res, dict):
                            tp_id = tp_res.get('order_id') or tp_res.get('id')
                            self.positions[vk].metadata['tp_order_id'] = tp_id
                            self.position_metadata[symbol]['tp_order_id'] = tp_id
                    except Exception as _e:
                        logger.warning(f"[{self.name}] ⚠️ Failed to record SL/TP order ids for {symbol}: {_e}")
                    # Deduct entry fee from balance if present in fill
                    if actual_fee_usd and actual_fee_usd > 0:
                        self.balance_usd -= actual_fee_usd
                        logger.info(f"[{self.name}] 📈 ENTRY FEE SUBTRACTED: ${actual_fee_usd:.4f} | New Balance=${self.balance_usd:.2f}")
                else:
                    # Weighted average entry
                    old_qty_abs = abs(pos.quantity)
                    new_qty_abs = old_qty_abs + actual_qty
                    pos.entry_price = ((old_qty_abs * pos.entry_price) + (actual_qty * actual_price)) / new_qty_abs
                    pos.quantity += (actual_qty if direction == 'BUY' else -actual_qty)
                    pos.quantity = round(pos.quantity, 8)
                    pos.stack_count += 1

                    # Record new stack for history
                    try:
                        stack_entry = {
                            'entry_price': actual_price,
                            'quantity': actual_qty,
                            'entry_time': time.time(),
                            'stack_id': pos.stack_count
                        }
                        pos.stacks.append(stack_entry)
                    except Exception:
                        pass

                    # === ATR-BASED STOP-LOSS FOR STACKED POSITIONS ===
                    # Fetch ATR and recalculate stops based on new average entry
                    atr_value = self.get_atr_for_symbol(symbol)
                    pos.stop_loss = self.calculate_atr_stop_loss(symbol, pos.entry_price, direction, atr_value)
                    pos.take_profit = self.calculate_atr_take_profit(symbol, pos.entry_price, direction, atr_value)

                    stop_distance_pct = abs(pos.entry_price - pos.stop_loss) / pos.entry_price * 100
                    tp_distance_pct = abs(pos.take_profit - pos.entry_price) / pos.entry_price * 100

                    logger.info(f"[{self.name}] 🔄 STACK UPDATE: {symbol} {direction} stacked | New Avg Entry: {pos.entry_price:.4f} | SL: {pos.stop_loss:.4f} ({stop_distance_pct:.2f}%) | TP: {pos.take_profit:.4f} ({tp_distance_pct:.2f}%)")

                    # Update stop loss and take profit orders on the exchange for stacked position
                    # FIX 2026-03-19: Place TP before SL on Kraken Futures to avoid wouldNotReducePosition error
                    if 'kraken' in str(self.actuator.exchange.id).lower() if self.actuator and hasattr(self.actuator, 'exchange') else False:
                        tpu = self.place_take_profit_order(symbol, direction, abs(pos.quantity), pos.take_profit)
                        slu = self.place_stop_loss_order(symbol, direction, abs(pos.quantity), pos.stop_loss)
                    else:
                        slu = self.place_stop_loss_order(symbol, direction, abs(pos.quantity), pos.stop_loss)
                        tpu = self.place_take_profit_order(symbol, direction, abs(pos.quantity), pos.take_profit)
                    # Sync order ids for the stacked position
                    try:
                        if isinstance(slu, dict):
                            self.positions[vk].metadata['sl_order_id'] = slu.get('order_id') or slu.get('id')
                            self.position_metadata[symbol]['sl_order_id'] = self.positions[vk].metadata['sl_order_id']
                        if isinstance(tpu, dict):
                            self.positions[vk].metadata['tp_order_id'] = tpu.get('order_id') or tpu.get('id')
                            self.position_metadata[symbol]['tp_order_id'] = self.positions[vk].metadata['tp_order_id']
                    except Exception:
                        pass
                    # Sync legacy held_assets and metadata to reflect new aggregated quantity
                    try:
                        self.held_assets[symbol] = pos.quantity
                        if symbol in self.position_metadata:
                            self.position_metadata[symbol]['quantity'] = abs(pos.quantity)
                    except Exception as _e:
                        logger.warning(f"[{self.name}] ⚠️ Failed to sync held_assets after stacking for {symbol}: {_e}")
                    # === END ATR-BASED STOP-LOSS ===

                logger.info(f"[{self.name}] 📈 ENTRY {symbol}: {actual_qty} @ {actual_price}")
            
        self._persist_portfolio()
        if self.governor and hasattr(self.governor, 'reconcile_with_executor'):
            self.governor.reconcile_with_executor(self.get_positions_snapshot())
        return realized_pnl

    def _record_trade_result(self, symbol: str, direction: str, entry_price: float, exit_price: float,
                           quantity: float, pnl_usd: float, entry_time: str, exit_time: str,
                           strategy: str = 'DIRECTIONAL', mfe: float = 0.0, mae: float = 0.0,
                           exit_reason: str = None, entropy_score: float = None,
                           regime: str = None, conviction: float = None,
                           quality_score: float = None, is_whitelisted: bool = False):
        """
        Record trade result to database for win rate tracking.
        FIX 2026-03-21: Accept MFE/MAE from position tracking.
        FIX 2026-03-21: Accept entry context metadata for winning pattern analysis.
        """
        if not self.db_manager:
            return

        try:
            # Calculate PnL percentage
            pnl_pct = 0.0
            if direction.upper() in ['BUY', 'LONG']:
                pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
            else:  # SELL or SHORT
                pnl_pct = (entry_price - exit_price) / entry_price if entry_price > 0 else 0.0

            # Determine if trade was profitable
            is_profitable = pnl_usd > 0

            # Prepare trade record in the format expected by DatabaseManager
            trade_record = {
                'symbol': symbol,
                'direction': direction,
                'quantity': quantity,
                'price': exit_price,  # Use exit price as the reference price
                'cost_usd': quantity * entry_price,  # Cost basis
                'timestamp': exit_time,  # Use exit time as the timestamp
                'pnl': pnl_usd,  # PnL in USD
                'pnl_percent': pnl_pct * 100,  # PnL as percentage (Database expects % not decimal)
                'unrealized_pnl': 0.0,  # For closed trades, unrealized PnL is 0
                'unrealized_pnl_percent': 0.0,
                'mfe': mfe,   # Maximum Favorable Excursion (tracked during position lifetime)
                'mae': mae,    # Maximum Adverse Excursion (tracked during position lifetime)
                'exit_reason': exit_reason,
                'strategy_type': strategy,
                'entropy_score': entropy_score,
                'regime': regime,
                'conviction': conviction,
                'quality_score': quality_score,
                'is_whitelisted': 1 if is_whitelisted else 0,
            }

            # Use the proper DatabaseManager method
            if hasattr(self.db_manager, 'save_trade'):
                try:
                    self.db_manager.save_trade(trade_record)
                except Exception as e:
                    logger.error(f"[{self.name}] DatabaseManager.save_trade failed: {e}")
            else:
                logger.warning(f"[{self.name}] DatabaseManager missing save_trade method")

            # --- FIX: Dispatch Outcome to Governor Circuit Breaker ---
            # FIX 2026-03-15: Pass crisis score for crisis-aware loss threshold
            # FIX 2026-03-21: Pass entry_price, exit_price, pnl_usd for better logging
            # FIX 2026-03-22: MOVED OUTSIDE db_manager check - blacklist MUST work even if DB fails
            if self.governor and hasattr(self.governor, 'register_trade_outcome'):
                try:
                    # Get crisis score from sentiment holon if available
                    crisis_score = 0.0
                    if hasattr(self, 'sentiment') and self.sentiment:
                        # Sentiment holon may have crisis score
                        crisis_score = getattr(self.sentiment, 'crisis_score', 0.0)

                    # Get entry price from tracking dict
                    entry_price = getattr(self, 'entry_prices', {}).get(symbol, None)

                    self.governor.register_trade_outcome(
                        symbol,
                        pnl_pct * 100,
                        crisis_score,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl_usd=pnl_usd
                    )
                except Exception as e:
                    logger.error(f"[{self.name}] Failed to dispatch outcome to Governor: {e}")
            
            # === GENOME GUARDIAN INTEGRATION (2026-03-22) ===
            # Monitor live genome performance with tight thresholds
            try:
                from .genome_guardian import monitor_trade
                
                # Get current equity
                equity = getattr(self, 'balance', 100.0)
                
                # Monitor trade
                result = monitor_trade(
                    pnl_usd=pnl_usd,
                    pnl_percent=pnl_pct,
                    symbol=symbol,
                    equity=equity
                )
                
                # Log result
                if result.get('action') == 'SWITCH':
                    logger.warning(f"[{self.name}] 🛡️ GENOME GUARDIAN: SWITCH TRIGGERED - {result['reason']}")
                    logger.warning(f"[{self.name}] 🛡️ Switching to Genome #2 (11 trades, more reliable)")
                elif result.get('alerts'):
                    for alert in result['alerts']:
                        logger.warning(f"[{self.name}] 🛡️ GENOME GUARDIAN: {alert}")
                
            except Exception as e:
                logger.debug(f"[{self.name}] Genome Guardian error: {e}")
            # ============================================

        except Exception as e:
            logger.error(f"[{self.name}] Failed to record trade result to database: {e}")

    def _dispatch_to_market(self, symbol: str, direction: str, qty: float, price: float, leverage: float, is_exit: bool, urgent: bool = False):
        """
        Dispatch order to the market for execution.
        
        FIX 2026-03-04: Track slippage per asset for diagnostics.
        """
        if not self.market:
            logger.error(f"[{self.name}] ERROR No market linked for order dispatch: {symbol} {direction} {qty}@{price}")
            return None

        try:
            # Place the order via market
            order_result = self.market.place_order(
                symbol=symbol,
                direction=direction,
                quantity=qty,
                price=price,
                order_type='market',
                leverage=leverage,
                reduce_only=is_exit,  # Important: exits should be reduce-only
                urgent=urgent
            )

            # Format the fill result to match expected structure
            if order_result and isinstance(order_result, dict):
                # FIX 2026-02-28: Capture actual fee from order result if available
                fee_info = order_result.get('fee', {})
                fee_cost = fee_info.get('cost', 0.0) if isinstance(fee_info, dict) else 0.0
                
                # Extract actual fill price and quantity
                actual_fill_price = order_result.get('avg_fill_price', price)
                actual_filled_qty = order_result.get('filled', qty)

                # === FIX 2026-03-04: TRACK SLIPPAGE ===
                # Compare expected price (order price) vs actual fill price
                self.track_slippage(
                    symbol=symbol,
                    expected_price=price,
                    fill_price=actual_fill_price,
                    quantity=actual_filled_qty
                )

                fill = {
                    'symbol': symbol,
                    'direction': direction,
                    'filled_qty': actual_filled_qty,  # Use filled amount or expected
                    'price': actual_fill_price,  # Use actual fill price
                    'avg_fill_price': actual_fill_price,  # Normalized key for average fill price
                    'avgPrice': actual_fill_price,  # Alias for some market adapters
                    'timestamp': order_result.get('timestamp', datetime.now(timezone.utc).isoformat()),
                    'order_id': order_result.get('order_id', ''),
                    'status': order_result.get('status', 'filled'),
                    'fee_usd': fee_cost,  # FIX 2026-02-28: Actual fee from exchange
                    'fee_currency': fee_info.get('currency', 'USD') if isinstance(fee_info, dict) else 'USD'
                }
                return fill
            else:
                # If order placement failed or didn't return expected result
                logger.warning(f"[{self.name}] WARNING Order may not have filled properly: {symbol}")
                # Return a minimal fill structure indicating failure
                return None

        except Exception as e:
            logger.error(f"[{self.name}] ERROR Order dispatch failed: {e}")
            return None

    def place_stop_loss_order(self, symbol: str, direction: str, quantity: float, stop_price: float,
                             order_type: str = 'stop_market', leverage: float = 1.0):
        """
        Place a stop loss order for a position.
        FIX 2026-02-24: Use Actuator's dedicated place_stop_order method for Kraken Futures compatibility.
        FIX 2026-03-21: Retry 3x then emergency market close if all attempts fail.
        """
        if not self.market:
            logger.error(f"[{self.name}] ERROR No market linked for stop loss order: {symbol}")
            return None

        sl_direction = 'SELL' if direction == 'BUY' else 'BUY'
        max_retries = 3

        for attempt in range(1, max_retries + 1):
            try:
                if self.actuator and hasattr(self.actuator, 'place_stop_order'):
                    order_result = self.actuator.place_stop_order(
                        symbol=symbol,
                        direction=sl_direction,
                        quantity=quantity,
                        stop_price=stop_price
                    )
                    if order_result and isinstance(order_result, dict):
                        logger.info(f"[{self.name}] 🛡️ STOP LOSS ORDER PLACED: {symbol} {sl_direction} {quantity} @ {stop_price}")
                        return order_result
                    elif order_result:
                        logger.info(f"[{self.name}] 🛡️ STOP LOSS ORDER PLACED (legacy): {symbol} {sl_direction} {quantity} @ {stop_price}")
                        return {'status': 'placed', 'symbol': symbol}
                    else:
                        logger.warning(f"[{self.name}] ⚠️ Stop loss attempt {attempt}/{max_retries} failed: {symbol}")
                else:
                    order_result = self.market.place_order(
                        symbol=symbol, direction=sl_direction, quantity=quantity,
                        price=stop_price, order_type='limit', leverage=leverage,
                        reduce_only=True, stop_loss=True
                    )
                    if order_result and isinstance(order_result, dict):
                        logger.info(f"[{self.name}] 🛡️ STOP LOSS ORDER PLACED (FALLBACK): {symbol}")
                        return order_result
                    else:
                        logger.warning(f"[{self.name}] ⚠️ Stop loss fallback attempt {attempt}/{max_retries} failed: {symbol}")
            except Exception as e:
                logger.error(f"[{self.name}] ❌ Stop loss attempt {attempt}/{max_retries} exception: {e}")

        # All retries exhausted — emergency market close to prevent unprotected position
        logger.error(f"[{self.name}] 🚨 EMERGENCY_CLOSE: {symbol} — stop-loss failed {max_retries}x, closing position at market")
        try:
            if self.actuator and hasattr(self.actuator, 'close_position'):
                self.actuator.close_position(symbol, sl_direction, quantity)
                logger.error(f"[{self.name}] 🚨 EMERGENCY_CLOSE EXECUTED: {symbol} {sl_direction} {quantity}")
            elif self.market:
                self.market.place_order(
                    symbol=symbol, direction=sl_direction, quantity=quantity,
                    order_type='market', reduce_only=True
                )
                logger.error(f"[{self.name}] 🚨 EMERGENCY_CLOSE EXECUTED (market fallback): {symbol}")
        except Exception as ec:
            logger.critical(f"[{self.name}] 💀 EMERGENCY_CLOSE FAILED: {symbol} — POSITION UNPROTECTED: {ec}")
        return None

    def place_take_profit_order(self, symbol: str, direction: str, quantity: float, take_profit_price: float,
                               order_type: str = 'limit', leverage: float = 1.0):
        """
        Place a take profit order for a position.
        """
        if not self.market:
            logger.error(f"[{self.name}] ERROR No market linked for take profit order: {symbol}")
            return None

        try:
            # Convert direction for take profit (same as entry direction)
            tp_direction = 'SELL' if direction == 'BUY' else 'BUY'

            # Preferred path: use Actuator order flow for venue-specific safeguards/retries.
            # TP orders use limit (not urgent/market) so they sit at the target price.
            if self.actuator and hasattr(self.actuator, 'place_order'):
                order_result = self.actuator.place_order(
                    symbol=symbol,
                    direction=tp_direction,
                    quantity=quantity,
                    price=take_profit_price,
                    order_type=order_type,
                    leverage=leverage,
                    reduce_only=True,
                    urgent=False,
                    take_profit=True
                )
                if order_result:
                    logger.info(f"[{self.name}] 💰 TAKE PROFIT ORDER PLACED: {symbol} {tp_direction} {quantity} @ {take_profit_price}")
                    return order_result

            # Place the take profit order via market
            order_result = self.market.place_order(
                symbol=symbol,
                direction=tp_direction,
                quantity=quantity,
                price=take_profit_price,
                order_type=order_type,
                leverage=leverage,
                reduce_only=True,  # Take profits should only reduce position
                take_profit=True  # Indicate this is a take profit order
            )

            if order_result and isinstance(order_result, dict):
                logger.info(f"[{self.name}] 💰 TAKE PROFIT ORDER PLACED: {symbol} {tp_direction} {quantity} @ {take_profit_price}")
                return order_result
            else:
                logger.warning(f"[{self.name}] WARNING Take profit order may not have placed properly: {symbol}")
                return None

        except Exception as e:
            logger.error(f"[{self.name}] ERROR Take profit order dispatch failed: {e}")
            return None

    def place_bracket_orders(self, symbol: str, direction: str, quantity: float, stop_price: float, take_profit_price: float, leverage: float = 1.0):
        """
        Attempt to place a bracket (OCO) order via actuator or market when supported.
        Falls back to placing stop and take profit separately and returns a dict with both results.
        """
        try:
            # Preferred: Actuator-level bracket (exchange-native bracket/reduce-only)
            if self.actuator and hasattr(self.actuator, 'place_bracket_order'):
                res = self.actuator.place_bracket_order(
                    symbol=symbol,
                    direction=direction,
                    quantity=quantity,
                    stop_price=stop_price,
                    take_profit_price=take_profit_price,
                    leverage=leverage
                )
                if isinstance(res, dict):
                    logger.info(f"[{self.name}] 🧷 BRACKET ORDER PLACED (actuator): {symbol}")
                    return res

            # Secondary: Market-level bracket
            if self.market and hasattr(self.market, 'place_bracket_order'):
                res = self.market.place_bracket_order(
                    symbol=symbol,
                    direction=direction,
                    quantity=quantity,
                    stop_price=stop_price,
                    take_profit_price=take_profit_price,
                    leverage=leverage
                )
                if isinstance(res, dict):
                    logger.info(f"[{self.name}] 🧷 BRACKET ORDER PLACED (market): {symbol}")
                    return res

            # Fallback: return None to let caller place separate SL/TP
            return None
        except Exception as e:
            logger.warning(f"[{self.name}] ⚠️ Bracket placement failed for {symbol}: {e}")
            return None

    def close_worst_stack(self, symbol: str) -> Optional[dict]:
        """
        Close the worst-performing stack for a given symbol.
        Returns the fill dict or None.
        """
        vk = self._get_virt_key(symbol, 'DIRECTIONAL')
        with self.position_lock:
            pos = self.positions.get(vk)
            if not pos:
                logger.info(f"[{self.name}] No active position for {symbol} to close a worst stack.")
                return None

            worst = pos.get_worst_stack(pos.entry_price)
            if not worst or worst.get('quantity', 0) <= 0:
                logger.info(f"[{self.name}] No stack available to close for {symbol}.")
                return None

            qty = worst['quantity']
            exit_dir = 'SELL' if pos.is_long else 'BUY'

        # Get current market price for execution
        current_price = None
        try:
            if self.market and hasattr(self.market, 'fetch_ticker'):
                t = self.market.fetch_ticker(symbol)
                current_price = t.get('last') if isinstance(t, dict) else None
        except Exception:
            current_price = None

        # Fallback to entry price if we couldn't fetch current market price
        if not current_price:
            current_price = pos.entry_price

        fill = self._dispatch_to_market(symbol, exit_dir, qty, current_price, pos.leverage, is_exit=True, urgent=True)
        if fill:
            logger.info(f"[{self.name}] Closed worst stack for {symbol}: qty={qty} fill={fill.get('filled_qty')}")
            # Update state using existing flow
            self._update_state_post_fill(vk, fill, pos.leverage, pos.strategy)
            return fill

        return None

    def execute_transaction(self, decision: TradeDecision, current_price: float) -> Optional[float]:
        """
        Premium Unified Execution Engine.
        Returns: Optional[float]: Realized PnL percentage if an exit occurred, else None.
        """
        # FIX 2026-02-28: Block NEW ENTRY trading if fee halt is active.
        # CRITICAL: Exits, trailing stops, TP hits, emergency stops, and compliance reductions
        # must NEVER be blocked by the fee halt — failing to close a losing position because fees
        # look high is far more dangerous than the fee anomaly itself.
        if self._fee_halt_active:
            sig_meta = getattr(decision.original_signal, 'metadata', {}) or {}
            sig_reason = str(sig_meta.get('reason', '')).upper()
            is_exit_trade = (
                sig_meta.get('is_exit', False) or
                sig_meta.get('reduce_only', False) or
                sig_reason in ('STOP_LOSS', 'TAKE_PROFIT', 'EMERGENCY_STOP', 'COMPLIANCE_REDUCE',
                               'ADAPTIVE_STOP', 'VOL_TRAIL', 'VOL_STOP', 'MEAN_REV',
                               'THESIS', 'PROFIT_PRESERVE', 'TREND_EXPIRY', 'TIME_EXIT',
                               'RAPID_TP', 'NORMAL_TP', 'RUNNER_TP', 'SATELLITE_TP',
                               'SATELLITE_SL', 'MONTE_CARLO_CLOSURE', 'MANAGEMENT_MODE_LOSS_CUT')
            )
            if not is_exit_trade:
                logger.critical(f"[{self.name}] 🛑 FEE HALT ACTIVE: Blocking ENTRY for {decision.original_signal.symbol}. Manual intervention required.")
                self._last_execution_error = "Fee halt active - new entries suspended"
                return None
            else:
                logger.warning(f"[{self.name}] ⚠️ FEE HALT ACTIVE but allowing EXIT for {decision.original_signal.symbol} ({sig_reason})")

        # NEW: Track last execution error for retry diagnostics
        self._last_execution_error = "Unknown"

        if decision.action == 'HALT' or decision.adjusted_size <= 0:
            logger.warning(f"[{self.name}] ⚠️ EXECUTION BLOCKED: action={decision.action}, adjusted_size={decision.adjusted_size}")
            self._last_execution_error = "Decision HALT or invalid size"
            return None

        signal = decision.original_signal
        symbol = signal.symbol
        strategy = signal.metadata.get('strategy', 'DIRECTIONAL')
        vk = self._get_virt_key(symbol, strategy)

        # 1. Guards
        if not self._validate_trade_constraints(signal):
            logger.warning(f"[{self.name}] ⚠️ EXECUTION BLOCKED: Trade constraints failed for {symbol}")
            self._last_execution_error = "Trade constraints validation failed"
            return None
        
        logger.info(f"[{self.name}] 🐛 DEBUG: {symbol} passed constraints. Calculating qty...")

        # 2. Parameters (Sizing & Solvency)
        exec_qty, leverage = self._calculate_execution_qty(decision, current_price)
        logger.info(f"[{self.name}] 🐛 DEBUG: _calculate_execution_qty returned exec_qty={exec_qty}, lev={leverage}")

        if exec_qty <= 0:
            logger.warning(f"[{self.name}] ⚠️ EXECUTION BLOCKED: Invalid execution qty={exec_qty} for {symbol}")
            self._last_execution_error = "Invalid execution quantity"
            return None

        is_exit = (signal.direction == 'SELL' and vk in self.positions and self.positions[vk].is_long) or \
                  (signal.direction == 'BUY' and vk in self.positions and self.positions[vk].is_short)

        urgent = is_exit or signal.metadata.get('urgent', False)
        logger.info(f"[{self.name}] 🐛 DEBUG: Dispatching to market {symbol} {signal.direction} {exec_qty} {current_price}")
        
        # 2.5 Phase 3 Execution Check (TWAP/VWAP Intercept)
        # If order is large enough or strategy explicitly requests TWAP, hand off to Rust engine
        try:
            from phase3_execution import get_phase3, RUST_AVAILABLE
            phase3 = get_phase3()
        except ImportError:
            RUST_AVAILABLE = False
            phase3 = None
        notional_value = exec_qty * current_price
        
        is_twap_eligible = not is_exit and not urgent and RUST_AVAILABLE and (
            'TWAP' in strategy.upper() or 
            'VWAP' in strategy.upper() or 
            notional_value > getattr(config, 'TWAP_MIN_NOTIONAL', 500.0) # E.g., > $500 orders get sliced
        )
        
        if is_twap_eligible:
            execution_duration = signal.metadata.get('twap_duration', 30) # Default 30 min TWAP
            execution_slices = signal.metadata.get('twap_slices', 12)
            
            logger.info(f"[{self.name}] ⏳ Handing off {exec_qty} {symbol} to Phase 3 TWAP Engine (duration={execution_duration}m)")
            
            # Start TWAP execution
            phase3.start_twap(
                symbol=symbol,
                side=signal.direction,
                total_qty=exec_qty,
                duration_minutes=execution_duration,
                num_slices=execution_slices
            )
            
            # For TWAP, we don't immediately return a fill since it's asynchronous.
            # We return early. The actual ledger updates will happen as the slices fill 
            # (which would require a separate loop/callback, but for now we just start the tracker).
            return 0.0

        # 3. Dispatch (Direct Market Fill)

        fill = self._dispatch_to_market(
            symbol=symbol,
            direction=signal.direction,
            qty=exec_qty,
            price=current_price,
            leverage=leverage,
            is_exit=is_exit,
            urgent=urgent
        )

        # Check for quantity mismatch between requested qty and exchange-filled qty
        if fill:
            actual_filled_qty = fill.get('filled_qty', exec_qty)
            try:
                qty_diff = abs(actual_filled_qty - exec_qty)
            except Exception:
                qty_diff = 0.0
            tol = getattr(config, 'QTY_MISMATCH_TOLERANCE', 1e-6)
            if getattr(config, 'LOG_QTY_MISMATCH', True) and qty_diff > tol:
                logger.warning(f"[{self.name}] ⚠️ QTY MISMATCH: requested={exec_qty} filled={actual_filled_qty} symbol={symbol} (diff={qty_diff:.8f})")
                if getattr(config, 'ENFORCE_FILLED_QTY_FOR_SLTP', True):
                    logger.info(f"[{self.name}] 🔧 Enforcing filled qty for downstream SL/TP actions: {actual_filled_qty}")
                    fill['filled_qty'] = actual_filled_qty

        if not fill:
            logger.warning(f"[{self.name}] 💀 EXECUTION FAILED: Order failed or not confirmed for {symbol} (qty={exec_qty}, lev={leverage})")
            self._last_execution_error = f"Order dispatch failed for {symbol}"
            
            # FIX 2026-03-01 #9: Check if error is wouldNotReducePosition (position desync)
            # If so, clear the phantom position to prevent repeated failures
            if self._last_execution_error and 'wouldNotReducePosition' in self._last_execution_error:
                logger.warning(f"[{self.name}] 🧹 PHANTOM POSITION DETECTED: Clearing {vk} from tracking")
                with self.position_lock:
                    if vk in self.positions:
                        del self.positions[vk]
                # Also clear from held assets
                if symbol in self.held_assets:
                    del self.held_assets[symbol]
                logger.info(f"[{self.name}] ✅ PHANTOM CLEARED: {symbol} removed from internal tracking")
            
            return None

        # 4. State Update
        # Pass signal metadata for stop-loss override logic
        sig_meta = decision.original_signal.metadata if decision and hasattr(decision, 'original_signal') else {}
        realized_pnl = self._update_state_post_fill(vk, fill, leverage, strategy, sig_meta)

        # === FIX 2026-03-04: RECORD TRADE FOR FREQUENCY TRACKING ===
        self.record_trade_executed(symbol)

        return realized_pnl if is_exit else 0.0

    def get_portfolio_value(self, current_price_ref: float = 0.0) -> float:
        """
        Calculate total portfolio value in USD based on all held assets (Leveraged Equity).
        Equity = Free Balance + Sum(Margin Used + Unrealized PnL)
        Direction-aware (Long/Short).
        """
        # If we have a live market, trust its equity as the primary source
        # but cross-verify with our internal ledger for sanity.
        if self.market:
            try:
                live_equity = self.market.get_equity()
                if live_equity is not None and live_equity > 0:
                    internal_calc = self._calculate_internal_equity()
                    
                    # Throttle Divergence Logic to avoid spamming 
                    now = time.time()
                    if now - self.last_divergence_check > self.DIVERGENCE_CHECK_INTERVAL:
                        if abs(live_equity - internal_calc) > (internal_calc * self.DIVERGENCE_THRESHOLD):
                            logger.warning(f"[{self.name}] ⚠️ EQUITY DIVERGENCE: Live ${live_equity:.2f} vs Internal ${internal_calc:.2f}")
                            # Auto-Correction Pulse: Overwrite local cash baseline
                            real_cash = self.market.get_wallet_balance()
                            if real_cash > 0:
                                logger.info(f"[{self.name}] 📉 Ledger Correction: Syncing balance to ${real_cash:.2f} (Reality)")
                                self.balance_usd = real_cash
                                self._persist_portfolio()
                        self.last_divergence_check = now
                        
                    return live_equity
            except Exception as e:
                logger.error(f"[{self.name}] ⚠️ Market Equity Fetch Failed: {e}. Falling back to internal calculation.")

        return self._calculate_internal_equity()

    def _calculate_internal_equity(self) -> float:
        """Helper to compute equity from local ledger data."""
        equity = self.balance_usd

        for vk, pos in list(self.positions.items()):
            if abs(pos.quantity) < 1e-8: continue

            # 1. Price Source
            current_price = self.latest_prices.get(pos.symbol, 0.0)
            entry_price = pos.entry_price

            # 2. Fallback Logic: Never ignore a position just because price is missing
            if current_price <= 0:
                if entry_price > 0:
                     # Assume flat PnL if no live data
                     current_price = entry_price
                else:
                     continue

            if entry_price > 0:
                qty_abs = abs(pos.quantity)
                # Margin Impact
                margin_impact = (qty_abs * entry_price) / pos.leverage

                # Unrealized PnL
                unrealized_pnl = pos.get_pnl_usd(current_price)

                # --- SANITY CHECK: PNL CORRUPTION GUARD ---
                if abs(unrealized_pnl) > (margin_impact * 100):
                    logger.error(f"[{self.name}] 🚨 PNL ANOMALY DETECTED for {vk}: PnL ${unrealized_pnl:.2f} vs Margin ${margin_impact:.2f}")
                    unrealized_pnl = max(-margin_impact * 0.95, min(unrealized_pnl, margin_impact * 5.0))

                if config.TRADING_MODE == 'FUTURES':
                    equity += unrealized_pnl
                else:
                    equity += (margin_impact + unrealized_pnl)

        return equity



    def reset_ledger(self, target_balance: float = None):
        """
        Hard reset of internal ledger to match target balance and clear positions.
        Used for emergency recovery from PnL corruption.
        """
        logger.warning(f"[{self.name}] 🧪 LEDGER REPAIR INITIATED...")
        if target_balance is not None:
            self.balance_usd = target_balance

        self.positions.clear()
        self.held_assets.clear()
        self.entry_prices.clear()
        self.entry_timestamps.clear()
        self.position_metadata.clear()

        if self.db_manager:
            self.db_manager.save_portfolio(self.balance_usd, self.held_assets, self.position_metadata)

        logger.info(f"[{self.name}] ✅ LEDGER RESET to ${self.balance_usd:.2f}")

    def receive_message(self, sender: Any, content: Any) -> None:
        """
        Handle incoming messages.
        """
        if isinstance(content, Message):
            if content.type == 'EXECUTE':
                 pass # Logic to trigger execution via message
        else:
            pass

    def panic_close_all(self, current_prices: Dict[str, float]) -> List[str]:
        """
        🚨 PANIC BUTTON: Force close ALL positions immediately.
        Bypasses Governor, Risk Checks, and Disposition.
        Uses Actuator directly for maximum speed.
        """
        logger.warning(f"[{self.name}] 🚨🚨 PANIC PROTOCOL INITIATED 🚨🚨")
        results = []
        
        for vk, pos in list(self.positions.items()):
            if abs(pos.quantity) < 1e-8: continue
            
            price = current_prices.get(pos.symbol, self.latest_prices.get(pos.symbol, 0.0))
            logger.info(f"[{self.name}] 🚨 PANIC CLOSE: {vk} ({pos.quantity})")
            
            try:
                if self.market:
                    self.market.place_order(
                        symbol=pos.symbol,
                        direction='SELL' if pos.is_long else 'BUY',
                        quantity=abs(pos.quantity),
                        price=price,
                        order_type='market',
                        leverage=pos.leverage,
                        urgent=True,
                        reduce_only=True
                    )
                
                # Update realized PnL in ledger
                realized_pnl = pos.get_pnl_usd(price) if price > 0 else 0.0
                self.balance_usd += realized_pnl

                # Record trade result for win rate tracking
                self._record_trade_result(
                    symbol=pos.symbol,
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=price,
                    quantity=abs(pos.quantity),
                    pnl_usd=realized_pnl,
                    entry_time=pos.entry_timestamp,
                    exit_time=datetime.now(timezone.utc).isoformat(),
                    strategy=pos.strategy,
                    mfe=getattr(pos, 'mfe', 0.0),
                    mae=getattr(pos, 'mae', 0.0),
                    exit_reason='PANIC_CLOSE',
                    entropy_score=getattr(pos, 'entry_entropy', None),
                    regime=getattr(pos, 'entry_regime', None),
                    conviction=getattr(pos, 'entry_conviction', None),
                    quality_score=getattr(pos, 'quality_score', None),
                    is_whitelisted=getattr(pos, 'is_whitelisted', False),
                )

                self.ledger.add_block(
                    entropy_score=0.9,
                    regime='CHAOTIC',
                    action='HALT'
                )

                del self.positions[vk]
                results.append(f"SUCCESS: {vk}")
            except Exception as e:
                logger.error(f"[{self.name}] ❌ Panic Close Failed for {vk}: {e}")
                results.append(f"FAILED: {vk} ({e})")
                
        self._persist_portfolio()
        return results

    def update_position_metrics(self, latest_prices: Dict[str, float]):
        """
        Update position metrics like PnL, MFE, MAE based on latest prices.
        """
        for vk, pos in self.positions.items():
            if abs(pos.quantity) < 1e-8:  # Skip empty positions
                continue

            symbol = pos.symbol
            if symbol in latest_prices:
                current_price = latest_prices[symbol]

                # Update PnL related metrics
                pnl_pct = pos.get_pnl_pct(current_price)

                # Update MFE (Maximum Favorable Excursion) if positive movement
                if pnl_pct > 0 and pnl_pct > pos.mfe:
                    pos.mfe = pnl_pct
                # Update MAE (Maximum Adverse Excursion) if negative movement
                elif pnl_pct < 0 and abs(pnl_pct) > pos.mae:
                    pos.mae = abs(pnl_pct)

        # Also update the legacy structures if needed
        for symbol, price in latest_prices.items():
            if symbol in self.entry_prices and self.entry_prices[symbol] > 0:
                # Update any metrics in legacy structure if needed
                pass  # For now, the main logic is in the new structure

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

    def force_sync_with_governor(self):
        """
        Force a complete synchronization between executor and governor positions.
        This helps resolve phantom margin issues by ensuring both systems agree
        on the current position state.
        """
        print(f"[{self.name}] 🔄 FORCED SYNC: Executor -> Governor")

        # Sync current positions to governor
        if self.governor:
            self.governor.sync_positions(self.held_assets, self.position_metadata)
            print(f"[{self.name}] ✅ FORCED SYNC: Positions synchronized to Governor")

        # Trigger a margin recalculation in the governor
        if self.governor and hasattr(self.governor, '_calculate_portfolio_state'):
            try:
                portfolio_state = self.governor._calculate_portfolio_state()
                print(f"[{self.name}] 📊 SYNC CHECK: Equity=${portfolio_state['equity']:.2f}, Used Margin=${portfolio_state['used_margin']:.2f}")
            except Exception as e:
                print(f"[{self.name}] ⚠️ SYNC CHECK failed: {e}")

    def validate_position_consistency(self) -> dict:
        """
        Validate consistency between executor and governor positions.

        Returns a dictionary with validation results.
        """
        result = {
            'executor_positions': len(self.held_assets),
            'governor_positions': 0,
            'discrepancies': [],
            'needs_sync': False
        }

        if self.governor and hasattr(self.governor, 'positions'):
            gov_positions = getattr(self.governor, 'positions', {})
            result['governor_positions'] = len(gov_positions)

            # Check for discrepancies
            exec_symbols = set(self.held_assets.keys())
            gov_symbols = set(gov_positions.keys())

            if exec_symbols != gov_symbols:
                result['discrepancies'].append({
                    'type': 'symbol_mismatch',
                    'executor_only': exec_symbols - gov_symbols,
                    'governor_only': gov_symbols - exec_symbols
                })
                result['needs_sync'] = True

            # Check quantities for matching symbols
            for symbol in exec_symbols.intersection(gov_symbols):
                exec_qty = self.held_assets.get(symbol, 0)
                gov_qty = gov_positions.get(symbol, {}).get('quantity', 0)

                if abs(exec_qty - gov_qty) > 1e-6:  # Small tolerance for floating point
                    result['discrepancies'].append({
                        'type': 'quantity_mismatch',
                        'symbol': symbol,
                        'executor_qty': exec_qty,
                        'governor_qty': gov_qty
                    })
                    result['needs_sync'] = True

        return result
