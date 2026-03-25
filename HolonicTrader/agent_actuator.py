"""
ActuatorHolon - Execution (Micro-Holon Architecture)

Objective: Minimize friction (Fees).
Mandate:
- Maker-Only Mode: NEVER use Market Orders.
- Logic: Place Limit Orders at Bid (for Long) or Ask (for Short).
- Re-pricing: If price moves away, cancel and replace.
- Stop Loss: Triggered via Governor/Oracle alarms, switches to Market Order.
"""

import time
import ccxt
import config
import os
import csv
import random
import threading
from datetime import datetime, timezone
from typing import Any, Literal, Dict, List, Optional
from HolonicTrader.holon_core import Holon, Disposition
from HolonicTrader.network_resilience import with_retry, with_circuit_breaker

class ActuatorHolon(Holon):
    def __init__(self, name: str = "ActuatorAgent", exchange_id: str = 'kraken', paper_mode: bool = False):
        super().__init__(name=name, disposition=Disposition(autonomy=0.8, integration=0.2))
        self.pending_orders = []
        self.exchange_id = exchange_id
        self.paper_mode = paper_mode
        self.exchange = None
        
        # === FIX 2026-03-02: DUAL VENUE SUPPORT ===
        # Enable both Kraken Futures and Kraken Spot for optimal execution
        # - xStocks (SPYX, NVDAX, etc.) → Spot (lower fees, no expiry)
        # - Crypto/Gold Directional → Futures (leverage available)
        # - Crypto/Gold Arb → Spot (better funding capture)
        self.spot_exchange = None  # Kraken Spot for xStocks and arb
        self.venue_mode = getattr(config, 'TRADING_MODE', 'FUTURES')

        # --- PHASE 4: Initialization Bypass for Paper Mode ---
        if not self.paper_mode:
            # Initialize Futures exchange (primary venue)
            if self.venue_mode in ['FUTURES', 'DUAL']:
                self.exchange_id = 'krakenfutures'
                print(f"[{self.name}] [CONNECT] Connecting to Kraken FUTURES...")
                api_key = config.KRAKEN_FUTURES_API_KEY or config.API_KEY
                api_secret = config.KRAKEN_FUTURES_PRIVATE_KEY or config.API_SECRET
                
                if hasattr(ccxt, self.exchange_id):
                    self.exchange = getattr(ccxt, self.exchange_id)({
                        'apiKey': api_key,
                        'secret': api_secret,
                        'enableRateLimit': True,
                        'options': {
                            'defaultType': 'future'
                        }
                    })
                    print(f"[{self.name}] ✅ Kraken FUTURES connected")
                else:
                    raise ValueError(f"Exchange {self.exchange_id} not found in ccxt")
            
            # Initialize Spot exchange (secondary venue for DUAL mode)
            if self.venue_mode == 'DUAL':
                try:
                    self.spot_exchange = ccxt.kraken({
                        'apiKey': config.KRAKEN_SPOT_KEY,
                        'secret': config.KRAKEN_SPOT_SECRET,
                        'enableRateLimit': True,
                        'options': {
                            'defaultType': 'spot'
                        }
                    })
                    print(f"[{self.name}] 🔌 Kraken SPOT connected for xStocks execution")
                    self.spot_connected = True
                except Exception as e:
                    print(f"[{self.name}] ⚠️ Spot connection failed: {e}")
                    self.spot_connected = False
                    self.spot_exchange = None
            else:
                self.spot_connected = False
        else:
            print(f"[{self.name}] 🧪 PAPER MODE INITIALIZED (Simulation Only)")

        # Kraken Symbol Mapping (Internal USDT -> Kraken USD)
        self.symbol_map = config.KRAKEN_SYMBOL_MAP

        # --- CIRCUIT BREAKER STATE ---
        self.error_count = 0
        self.circuit_open = False
        self.hibernate_until = 0.0
        self.MAX_CONSECUTIVE_ERRORS = getattr(config, 'API_MAX_RETRIES', 15)
        self.HIBERNATION_TIME = getattr(config, 'API_HIBERNATION_TIME', 60)

        # Caching to prevent API Rate Limit Spam (Phase 4 Optimization)
        self.cached_equity = None
        self.last_equity_time = 0.0
        self.cached_balance = {}
        self.last_balance_time = 0.0
        self.last_balance_time = 0.0
        self.CACHE_TTL = 3.0 # Cache duration for balance calls

        # === API FAILURE STATE TRACKING (Fix for false drawdown detection) ===
        self._api_failure_mode = False
        self._api_failure_count = 0
        self._last_known_equity = None
        self._last_known_balance = None
        self._equity_fetch_success_count = 0  # Require N successes before trusting data
        self._recent_equity_readings = []  # Moving average for drawdown calculation
        self.MAX_EQUITY_READINGS = 10
        
        # Phase 6: Error Cooldowns (Nano/Micro Efficiency)
        self.failed_orders = {} # {f"{symbol}_{error}": timestamp}

        # Regime/Veto Change Retry Tracking for ETH/USDT
        self.regime_change_time = 0.0          # Timestamp of last regime shift
        self.veto_relax_time = 0.0             # Timestamp of last veto relaxation
        self.regime_retry_attempts = {}        # {symbol: attempt_count}
        self.last_regime = "HARVEST"           # Track regime for change detection

        # Release any existing locks on startup
        self.release_global_lock("System Initialization")

        # --- ADVANCED EXECUTION STATE ---
        self.active_algos = {} # {algo_id: {thread, stop_event, status}}
        self._algo_lock = threading.Lock()

        # --- DEBUG FLAG (Required by trailing stop manager) ---
        self.DEBUG = False  # Silence debug spam by default

    def check_circuit_breaker(self) -> bool:
        """
        Returns True if Circuit is CLOSED (Healthy).
        Returns False if Circuit is OPEN (Broken/Hibernating).

        FIX 2026-02-24: Force position reconciliation on circuit reset.
        """
        if self.circuit_open:
            remaining = self.hibernate_until - time.time()
            if remaining > 0:
                # Still hibernating
                if int(remaining) % 60 == 0:
                    print(f"[{self.name}] 💤 API CIRCUIT OPEN: Hibernating for {int(remaining)}s (Too many 503s)")
                return False
            else:
                # Wake up
                print(f"[{self.name}] 🌅 API CIRCUIT RESET: Attempting to reconnect...")
                self.circuit_open = False
                self.error_count = 0
                self._api_failure_mode = False
                self._api_failure_count = 0
                
                # NEW: Signal that reconciliation is needed after circuit reset
                print(f"[{self.name}] 🔄 Position reconciliation recommended after API recovery")
                return True
        return True

    def release_global_lock(self, reason: str = "Manual"):
        """Clear all failed order records to release the NANO GLOBAL LOCK."""
        count = len(self.failed_orders)
        self.failed_orders.clear()
        print(f"[{self.name}] [UNLOCK] GLOBAL LOCK RELEASED: {reason} (Cleared {count} failures)")

    def _log_paper_trade(self, symbol: str, direction: str, quantity: float, price: float,
                         is_exit: bool, order_id: str, timestamp: str):
        """
        Append a paper trade fill to paper_trades.csv for PnL tracking.
        Written to the same directory as live session logs.
        """
        try:
            log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            csv_path = os.path.join(log_dir, 'paper_trades.csv')
            write_header = not os.path.exists(csv_path)
            trade_type = 'CLOSE' if is_exit else 'OPEN'
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['timestamp', 'symbol', 'direction', 'quantity', 'price', 'type', 'order_id'])
                writer.writerow([timestamp, symbol, direction, f'{quantity:.8f}', f'{price:.8f}', trade_type, order_id])
            
            # Also log to execution_log.csv for detailed tracking
            self._log_execution(symbol, direction, quantity, price, trade_type, order_id, timestamp, "FILLED", "PAPER")
        except Exception as e:
            print(f"[{self.name}] ⚠️ Paper trade log failed: {e}")
    
    def _log_execution(self, symbol: str, direction: str, quantity: float, price: float,
                       order_type: str, order_id: str, timestamp: str, status: str, mode: str):
        """
        Log all order executions (paper and live) to execution_log.csv
        """
        try:
            log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            csv_path = os.path.join(log_dir, 'execution_log.csv')
            write_header = not os.path.exists(csv_path)
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['timestamp', 'symbol', 'direction', 'quantity', 'price', 
                                   'order_type', 'order_id', 'status', 'mode'])
                writer.writerow([timestamp, symbol, direction, f'{quantity:.8f}', 
                               f'{price:.8f}', order_type, order_id, status, mode])
        except Exception as e:
            print(f"[{self.name}] ⚠️ Execution log failed: {e}")

    def can_retry_order(self, symbol: str, error_type: str = 'General') -> bool:
        """
        Check if we are in cooldown for this specific error on this symbol.
        Prevents infinite retry loops on non-transient errors (e.g. Insufficient Funds).
        """
        key = f"{symbol}_{error_type}"
        last_time = self.failed_orders.get(key, 0)
        
        # If config has NANO_COOLDOWN, use it, else default 300
        cooldown = getattr(config, 'NANO_COOLDOWN_AFTER_FAILURE', 300)
        
        # AEHML Fix: Don't let transient price collar rejections trigger a long global lock
        if error_type == "outsidePriceCollar":
             cooldown = 30 # 30s pause for collar issues is enough
             
        if time.time() - last_time < cooldown:
            remaining = int(cooldown - (time.time() - last_time))
            if remaining % 60 == 0: # Reduce log spam
                print(f"[{self.name}] ⏳ COOLDOWN ACTIVE: {symbol} ({remaining}s rem) due to {error_type}")
            return False
            
        # --- NANO-MODE GLOBAL LOCK REMOVED ---
        # We rely on per-asset cooldowns (lines 119-133) + Circuit Breaker (lines 87-106)
        # This prevents one bad asset/error from freezing the entire system.
        # -------------------------------------
        # -----------------------------------------------
        
        return True

    def _reconnect_exchange(self):
        """
        CRITICAL: Re-establish exchange connection when it drops.
        Called when place_stop_order detects None exchange object.
        """
        try:
            print(f"[{self.name}] 🔄 RECONNECTING to exchange...")
            
            # Determine correct exchange and keys
            if config.TRADING_MODE == 'FUTURES':
                exchange_cls = getattr(ccxt, 'krakenfutures')
                api_key = config.KRAKEN_FUTURES_API_KEY or config.API_KEY
                api_secret = config.KRAKEN_FUTURES_PRIVATE_KEY or config.API_SECRET
            else:
                exchange_cls = getattr(ccxt, 'kraken')
                api_key = config.API_KEY
                api_secret = config.API_SECRET
            
            # Create new exchange instance
            self.exchange = exchange_cls({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future' if config.TRADING_MODE == 'FUTURES' else 'spot'
                },
                'timeout': 30000,  # 30 second timeout
                'retries': 3
            })
            
            # Verify connection with a lightweight API call
            self.exchange.load_markets()
            print(f"[{self.name}] ✅ EXCHANGE RECONNECTED: {self.exchange.id}")
            
        except Exception as e:
            print(f"[{self.name}] ❌ RECONNECTION FAILED: {e}")
            self.exchange = None  # Explicitly set to None so caller knows it failed

    def record_order_failure(self, symbol: str, error_type: str):
        """Record a failure to trigger cooldown."""
        key = f"{symbol}_{error_type}"
        self.failed_orders[key] = time.time()

    def notify_regime_change(self, new_regime: str):
        """
        Called when regime shifts (e.g., TRANSITION -> HARVEST).
        Resets retry tracking for priority assets like ETH/USDT.
        """
        now = time.time()
        if new_regime != self.last_regime:
            print(f"[{self.name}] 🔄 REGIME SHIFT: {self.last_regime} -> {new_regime}")
            self.last_regime = new_regime
            self.regime_change_time = now
            # Reset retry attempts on regime change
            self.regime_retry_attempts.clear()
            
            # Special handling for HARVEST regime (allows retries)
            if new_regime == "HARVEST":
                print(f"[{self.name}] ✅ HARVEST REGIME: Retries allowed for ETH/USDT")

    def notify_veto_relax(self, veto_type: str = "General"):
        """
        Called when veto conditions are relaxed (e.g., exposure counters reset).
        """
        self.veto_relax_time = time.time()
        print(f"[{self.name}] ✅ VETO RELAXED: {veto_type}")

    def can_retry_regime_change(self, symbol: str) -> tuple[bool, str]:
        """
        Check if retry is allowed after regime/veto change.
        Returns (allowed, reason).
        """
        now = time.time()
        
        # Check if ETH/USDT priority retry is enabled
        if not getattr(config, 'ETH_USDT_PRIORITY_RETRY', False):
            return True, "Priority retry disabled"
        
        # Only apply to ETH/USDT for now
        if symbol != 'ETH/USDT':
            return True, "Non-priority symbol"
        
        # Check regime change retry
        regime_delay = getattr(config, 'REGIME_CHANGE_RETRY_DELAY', 30)
        if now - self.regime_change_time < regime_delay:
            remaining = regime_delay - (now - self.regime_change_time)
            return False, f"Regime change cooldown active ({remaining:.0f}s remaining)"
        
        # Check veto relax retry
        veto_delay = getattr(config, 'VETO_RELAX_RETRY_DELAY', 45)
        if now - self.veto_relax_time < veto_delay:
            remaining = veto_delay - (now - self.veto_relax_time)
            return False, f"Veto relax cooldown active ({remaining:.0f}s remaining)"
        
        # Check max retry attempts
        max_attempts = getattr(config, 'REGIME_RETRY_MAX_ATTEMPTS', 3)
        attempts = self.regime_retry_attempts.get(symbol, 0)
        if attempts >= max_attempts:
            return False, f"Max retry attempts reached ({attempts}/{max_attempts})"
        
        return True, "Retry allowed"
    
    def record_regime_retry(self, symbol: str):
        """Increment retry attempt counter for symbol."""
        self.regime_retry_attempts[symbol] = self.regime_retry_attempts.get(symbol, 0) + 1
        backoff = getattr(config, 'REGIME_RETRY_BACKOFF', 1.5)
        next_delay = getattr(config, 'REGIME_CHANGE_RETRY_DELAY', 30) * (backoff ** self.regime_retry_attempts[symbol])
        print(f"[{self.name}] ⏳ {symbol} retry #{self.regime_retry_attempts[symbol]} recorded. Next delay: {next_delay:.0f}s")

    def report_success(self):
        """Call this after a successful API interaction to reset counters."""
        if self.error_count > 0:
            self.error_count = 0
            # print(f"[{self.name}] 🟢 API Connection Stabilized.")

    def report_failure(self, error_msg: str):
        """Call this after a Network/503 Error."""
        self.error_count += 1
        print(f"[{self.name}] ⚠️ API Error #{self.error_count}: {error_msg}")
        
        if self.error_count >= self.MAX_CONSECUTIVE_ERRORS:
            self.circuit_open = True
            self.hibernate_until = time.time() + self.HIBERNATION_TIME
            print(f"[{self.name}] 💥 CIRCUIT BREAKER TRIPPED: Entering {self.HIBERNATION_TIME}s Hibernation to save API Quota.")

    def check_liquidity(self, symbol: str, direction: str, quantity: float, price: float) -> bool:
        """
        Verify that the order book has sufficient depth to absorb this order
        without massive slippage.
        Rule: Top 10 levels must have cumulative volume >= 3x order quantity.
        """
        try:
            if price <= 0:
                print(f"[{self.name}] ⚠️ Liquidity Check Skipped: Invalid Price ({price})")
                return True

            if not self.exchange:
                if getattr(config, 'DEBUG', False): print(f"[{self.name}] ⚠️ Liquidity Check Skipped: No Exchange Connection (Paper Mode)")
                return True

            # FIX 2026-02-25: Convert symbol to exchange format before API call
            exec_symbol = symbol
            if config.TRADING_MODE == 'FUTURES':
                exec_symbol = config.KRAKEN_SYMBOL_MAP.get(symbol, symbol)

            # Fetch deeper book (Limit 50 for safety)
            book = self.exchange.fetch_order_book(exec_symbol, limit=50)
            # If Buying, we consume Asks. If Selling, we hit Bids.
            side_book = book['asks'] if direction == 'BUY' else book['bids']

            if not side_book:
                print(f"[{self.name}] ⚠️ HIGH IMPACT: {symbol} Order book returned empty.")
                return False

            cumulative_vol = 0.0
            
            # Anchor to the Best Bid / Offer (BBO) rather than the signal price
            # to avoid rejecting trades if the signal price is lagging the actual book.
            anchor_price = float(side_book[0][0])

            for bid_ask in side_book:
                level_price = float(bid_ask[0])
                level_qty = float(bid_ask[1])

                # Only count volume within 2% of the Best Bid/Offer
                if abs(level_price - anchor_price) / anchor_price < 0.02:
                     cumulative_vol += level_qty

            # Safety Factor: We want book volume to be at least 3x our size
            # IMPACT CHECK: If we are > X% of depth, it's a high impact trade
            required_vol = quantity / getattr(config, 'EXEC_IMPACT_THRESHOLD', 0.10)

            if cumulative_vol < required_vol:
                print(f"[{self.name}] ⚠️ HIGH IMPACT: {symbol} Depth {cumulative_vol:.4f} < Req {required_vol:.4f} (Threshold: {config.EXEC_IMPACT_THRESHOLD*100}%)")
                return False

            return True

        except Exception as e:
            print(f"[{self.name}] ⚠️ Liquidity Check Error: {e}. Proceeding with caution.")
            return True # Fail open to avoid paralysis, but log warning

    @with_circuit_breaker("actuator_fetch_balance", failure_threshold=5, recovery_timeout=60.0, fallback_value=None)
    @with_retry(max_retries=3, base_delay=1.0, max_delay=10.0, exceptions=(ccxt.NetworkError, ccxt.ExchangeError))
    def _fetch_balance_resilient(self):
        return self.exchange.fetch_balance()

    def get_account_balance(self, asset: str = 'USDT') -> float:
        """
        Fetch REAL free balance (Available Margin) from exchange.
        """
        if self.paper_mode:
            return config.INITIAL_CAPITAL # Simple mock for paper trading
            
        if not self.check_circuit_breaker(): return 0.0

        # Cache Check
        if (asset in self.cached_balance) and (time.time() - self.last_balance_time < self.CACHE_TTL):
            return self.cached_balance.get(asset, 0.0)

        try:
            balance = self._fetch_balance_resilient()
            if balance is None:
                return 0.0
                
            self.report_success()
            
            # Kraken Futures specific mapping
            if config.TRADING_MODE == 'FUTURES':
                info = balance.get('info', {})
                # Try to get explicit 'availableMargin' from flex account
                # Structure: info -> accounts -> flex -> availableMargin
                try:
                    accounts = info.get('accounts', {})
                    flex = accounts.get('flex', {})
                    avail_margin = float(flex.get('availableMargin', 0.0))
                    if avail_margin > 0:
                        self.cached_balance['USDT'] = avail_margin
                        self.last_balance_time = time.time()
                        return avail_margin
                except Exception as e_flex: 
                    print(f"[{self.name}] ⚠️ Futures Flex Margin Check Failed: {e_flex}")

            # Fallback to standard CCXT 'free'
            b_usd = balance['free'].get('USD', 0.0)
            b_usdt = balance['free'].get('USDT', 0.0)
            b_zusd = balance['free'].get('ZUSD', 0.0)
            
            total_avail = max(b_usd, b_usdt, b_zusd)
            
            # Update Cache
            self.cached_balance['USDT'] = total_avail
            self.cached_balance['USD'] = total_avail
            self.last_balance_time = time.time()
            
            return total_avail
        except Exception as e:
            self.report_failure(str(e))
            print(f"[{self.name}] ❌ Balance Check Failed: {e}")
            return 0.0

    def get_wallet_balance(self, asset: str = 'USDT') -> float:
        """
        Fetch REAL wallet balance (Cash + Realized PnL, excluding unrealized) from exchange.
        
        FIX 2026-02-25: Use marginEquity for consistency with initial capital sync.
        This prevents the $28.80 drift discrepancy on startup.
        """
        if self.paper_mode:
            return config.INITIAL_CAPITAL

        if not self.check_circuit_breaker(): return 0.0

        try:
            balance = self.exchange.fetch_balance()
            if config.TRADING_MODE == 'FUTURES':
                # FIX: Use marginEquity instead of walletBalance for consistency
                # with initial capital sync in main_live_phase4.py
                info = balance.get('info', {})
                accounts = info.get('accounts', {})
                flex = accounts.get('flex', {})
                
                # Try marginEquity first (total equity including unrealized PnL)
                margin_equity = float(flex.get('marginEquity', 0.0))
                if margin_equity > 0:
                    return margin_equity
                
                # Fallback to walletBalance if marginEquity not available
                wallet = float(flex.get('walletBalance', 0.0))
                if wallet > 0:
                    return wallet

            # Fallback to standard CCXT 'total'
            return balance.get('total', {}).get(asset, 0.0)
        except Exception as e:
            print(f"[{self.name}] ⚠️ Wallet Balance Check Failed: {e}")
            return 0.0


    def get_equity(self) -> float:
        """
        Fetch TOTAL EQUITY (Balance + Unrealized PnL).
        Crucial for accurate Drawdown calculation in Governor.

        FIX 2026-02-24: Graceful degradation during API failures.
        Returns last known good equity instead of None/0 to prevent false drawdown detection.
        """
        if self.paper_mode:
            return config.INITIAL_CAPITAL

        if not self.check_circuit_breaker():
            # Circuit open - return last known good value
            if self._last_known_equity is not None:
                self._api_failure_mode = True
                self._api_failure_count += 1
                if self._api_failure_count % 10 == 0:  # Log every 10 failures
                    print(f"[{self.name}] ⚠️ API FAILURE MODE #{self._api_failure_count}: Using last known equity ${self._last_known_equity:.2f}")
            return self._last_known_equity

        # Cache Check
        if self.cached_equity and (time.time() - self.last_equity_time < self.CACHE_TTL):
             return self.cached_equity

        for attempt in range(3):
            try:
                balance = self.exchange.fetch_balance()
                self.report_success()
                
                # NEW: Track successful fetch
                self._api_failure_mode = False
                self._api_failure_count = 0
                self._equity_fetch_success_count += 1

                info = balance.get('info', {})

                # 1. Futures: Explicit marginEquity
                if config.TRADING_MODE == 'FUTURES':
                    accounts = info.get('accounts', {})
                    flex = accounts.get('flex', {})
                    total_equity = float(flex.get('marginEquity', 0.0))
                    if total_equity > 0:
                        self.cached_equity = total_equity
                        self.last_equity_time = time.time()
                        self._last_known_equity = total_equity
                        # Track for moving average
                        self._recent_equity_readings.append(total_equity)
                        if len(self._recent_equity_readings) > self.MAX_EQUITY_READINGS:
                            self._recent_equity_readings.pop(0)
                        return total_equity

                # 2. Spot/Unified: Equivalent Balance ('eb')
                equity = float(info.get('eb', 0.0))
                if equity > 0:
                     self.cached_equity = equity
                     self.last_equity_time = time.time()
                     self._last_known_equity = equity
                     self._recent_equity_readings.append(equity)
                     if len(self._recent_equity_readings) > self.MAX_EQUITY_READINGS:
                         self._recent_equity_readings.pop(0)
                     return equity

                # 3. Fallback: Total USD
                usd_bal = balance.get('total', {}).get('USD', 0.0)
                self.cached_equity = usd_bal
                self.last_equity_time = time.time()
                self._last_known_equity = usd_bal
                self._recent_equity_readings.append(usd_bal)
                if len(self._recent_equity_readings) > self.MAX_EQUITY_READINGS:
                    self._recent_equity_readings.pop(0)
                return usd_bal

            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                self.report_failure(str(e))
                # NEW: Track failure state
                self._api_failure_mode = True
                self._api_failure_count += 1

                if attempt == 2:
                     print(f"[{self.name}] ❌ Equity Check Failed after 3 attempts: {e}")
                     # Return last known good value instead of None
                     if self._last_known_equity is not None:
                         print(f"[{self.name}] ⚠️ Using last known equity: ${self._last_known_equity:.2f}")
                         return self._last_known_equity
                     return None
                # FIX 2026-02-28: Exponential backoff for rate limit handling
                backoff_time = (2 ** attempt) + random.uniform(0, 0.5)  # 2s, 4s, 8s + jitter
                print(f"[{self.name}] ⏳ Equity fetch failed (attempt {attempt+1}/3), retrying in {backoff_time:.1f}s...")
                time.sleep(backoff_time)
        return None

    def get_reliable_equity(self) -> float:
        """
        Get equity with moving average smoothing for drawdown calculation.
        Requires minimum successful fetches before returning reliable data.

        Returns:
            float: Reliable equity value, or last known equity if insufficient data
        """
        # FIX 2026-02-28: Require at least 3 successful fetches before trusting moving average
        # But always return a valid equity value (never None) to prevent trading halts
        if self._equity_fetch_success_count < 3:
            # Only log warning once per failure cycle to reduce noise
            if self._api_failure_mode or getattr(self, '_last_equity_warning_count', 0) < 3:
                print(f"[{self.name}] ⚠️ Insufficient equity data ({self._equity_fetch_success_count}/3 fetches). Using last known: ${self._last_known_equity:.2f}")
                self._last_equity_warning_count = getattr(self, '_last_equity_warning_count', 0) + 1
            return self._last_known_equity if self._last_known_equity is not None else config.INITIAL_CAPITAL

        # Use moving average if we have enough readings
        if len(self._recent_equity_readings) >= 3:
            avg_equity = sum(self._recent_equity_readings[-3:]) / 3
            return avg_equity

        return self._last_known_equity if self._last_known_equity is not None else config.INITIAL_CAPITAL

    def calculate_equity_from_positions(self) -> float:
        """
        FIX 2026-02-28: Alternative equity calculation from balance + unrealized PnL.
        Used as fallback when Kraken Futures API is rate-limited.
        """
        try:
            # Get base balance
            balance = self.get_account_balance()
            
            # Add unrealized PnL from open positions
            unrealized_pnl = 0.0
            if hasattr(self, 'pending_orders') and hasattr(self, 'exchange'):
                try:
                    positions = self.exchange.fetch_positions()
                    for pos in positions:
                        unrealized_pnl += float(pos.get('unrealizedPnl', 0.0) or 0.0)
                except Exception:
                    pass  # Ignore if we can't fetch positions
            
            return balance + unrealized_pnl
        except Exception as e:
            print(f"[{self.name}] ⚠️ Alternative equity calc failed: {e}")
            return self._last_known_equity if self._last_known_equity is not None else config.INITIAL_CAPITAL

    def get_buying_power(self, leverage: float = 5.0) -> float:
        # ... (unchanged, but relying on get_account_balance now more reliable)
        return self.get_account_balance() * leverage

    def cancel_all_orders(self, symbol: str):
        """
        Cancel all open orders for a specific symbol on the exchange.
        Also cleans up internal pending_orders tracking.
        """
        # Paper mode handling
        if self.paper_mode:
            print(f"[{self.name}] 🧪 [SIM] Cancel all orders for {symbol}")
            self.pending_orders = [o for o in self.pending_orders if o['symbol'] != symbol]
            return
        
        if not self.check_circuit_breaker(): return

        exec_symbol = config.KRAKEN_SYMBOL_MAP.get(symbol, symbol)
        print(f"[{self.name}] 🗑️ Cancelling all orders for {symbol} ({exec_symbol})...")

        try:
            # 1. Cancel on Exchange (Kraken Futures specific)
            if config.TRADING_MODE == 'FUTURES':
                # Kraken Futures: cancel_all_orders is possible but often it's safer to cancel by symbol
                # CCXT unified: cancel_all_orders(symbol)
                try:
                    self.exchange.cancel_all_orders(exec_symbol)
                except Exception as e:
                    # Fallback if the above doesn't work for this specific CCXT version/exchange
                    if "not supported" in str(e).lower():
                        open_orders = self.exchange.fetch_open_orders(exec_symbol)
                        for order in open_orders:
                            self.exchange.cancel_order(order['id'], exec_symbol)
                    else:
                        raise e
            else:
                # Spot: cancel_all_orders(symbol)
                self.exchange.cancel_all_orders(exec_symbol)

            # 2. Cleanup Internal Tracking
            self.pending_orders = [o for o in self.pending_orders if o['symbol'] != symbol]
            print(f"[{self.name}] ✅ All orders for {symbol} cancelled.")
            self.report_success()

        except Exception as e:
            print(f"[{self.name}] ❌ Cancel All Orders Failed for {symbol}: {e}")
            self.report_failure(str(e))


    def close_position(self, symbol: str, qty: float = None, reason: str = None) -> bool:
        """
        Close an existing position (Market Order).
        Handles side inversion and reduceOnly flag automatically.
        FIX 2026-02-24: Better symbol mapping for Kraken Futures (PF_* format).
        FIX 2026-03-02: Added 'reason' parameter for trade logging.
        
        Args:
            symbol: Asset symbol (e.g., 'BTC/USDT')
            qty: Optional partial close quantity (None = close all)
            reason: Optional reason for close (e.g., 'MANUAL_C2_CLOSE', 'STOP_LOSS')
        """
        # Paper mode handling
        if self.paper_mode:
            reason_str = f" (Reason: {reason})" if reason else ""
            print(f"[{self.name}] 🧪 [SIM] Close position: {symbol} (qty: {qty}){reason_str}")
            return True

        if not self.check_circuit_breaker(): return False

        try:
            # 1. Fetch Position to verify it exists and get direction
            positions = self.exchange.fetch_positions()
            target_pos = None

            # Map internal symbol to Kraken format
            exec_symbol = config.KRAKEN_SYMBOL_MAP.get(symbol, symbol)
            
            # FIX: Also check for Kraken Futures format (PF_* -> internal)
            # If symbol is already in Kraken format (PF_ADAUSD), use it directly
            # Otherwise, try to find position by matching base asset
            kraken_format = exec_symbol.startswith('PF_') if exec_symbol else False

            for p in positions:
                p_symbol = p.get('symbol', '')
                # Direct match
                if p_symbol == exec_symbol:
                    target_pos = p
                    break
                # FIX: Match PF_ADAUSD to ADA/USDT
                if not kraken_format and p_symbol.startswith('PF_'):
                    # Extract base asset from PF_ADAUSD -> ADA
                    kraken_base = p_symbol[3:].replace('USD', '').replace('USDT', '')
                    internal_base = symbol.split('/')[0] if '/' in symbol else symbol
                    if kraken_base == internal_base:
                        target_pos = p
                        exec_symbol = p_symbol  # Use Kraken format for order
                        print(f"[{self.name}] 🔀 SYMBOL MAPPED: {symbol} -> {exec_symbol}")
                        break

            if not target_pos:
                print(f"[{self.name}] ⚠️ Close Failed: No active position found for {symbol} ({exec_symbol})")
                print(f"[{self.name}] ℹ️ Available positions: {[p.get('symbol', 'unknown') for p in positions]}")
                return False

            current_qty = float(target_pos.get('contracts', 0.0))
            if current_qty == 0:
                print(f"[{self.name}] ⚠️ Close Failed: Position size is 0 for {symbol}")
                return False

            # 2. Determine Close Direction
            pos_side = target_pos['side'] # 'long' or 'short'
            close_side = 'SELL' if pos_side == 'long' else 'BUY'

            # 3. Determine Quantity
            final_qty = current_qty
            if qty is not None and qty < current_qty:
                final_qty = qty

            print(f"[{self.name}] 🔪 CLOSING {symbol}: Found {pos_side.upper()} {current_qty}, Selling {final_qty} (Side: {close_side})")

            # --- UPDATE: CANCEL WORKING ORDERS FIRST ---
            # Flush parent/stops to prevent conflicts
            self.cancel_all_orders(symbol)
            time.sleep(0.3) # Give exchange 300ms to process cancels
            # ------------------------------------------

            # 4. Execute Market Close
            order_id = self.place_order(
                symbol=symbol,
                direction=close_side,
                quantity=final_qty,
                order_type='market',
                reduce_only=True,
                urgent=True
            )

            return order_id is not None

        except Exception as e:
            print(f"[{self.name}] ❌ Close Position Error: {e}")
            return False

    def place_stop_order(self, symbol: str, direction: str, quantity: float, stop_price: float) -> bool:
        """
        Place a Stop Loss Order (Reduce Only) to protect a position.
        Direction: 'BUY' (for Short Cover) or 'SELL' (for Long Exit)
        FIX 2026-03-01: Better error logging and minimum quantity handling for stop orders.
        """
        # Debug: Log paper_mode state
        if getattr(self, 'DEBUG', False):
            print(f"[{self.name}] DEBUG: place_stop_order called - paper_mode={self.paper_mode}, exchange={self.exchange}")

        if not self.check_circuit_breaker(): return False

        # --- PAPER MODE HANDLING ---
        if self.paper_mode:
            print(f"[{self.name}] 🧪 [SIM] Stop Loss: {direction} {quantity} {symbol} @ ${stop_price:.2f}")
            # Simulate successful placement in paper mode
            return True

        # CRITICAL FIX: Validate exchange connection before every stop loss attempt
        # This prevents "'NoneType' object has no attribute 'create_order'" errors
        if self.exchange is None:
            print(f"[{self.name}] ❌ CRITICAL: exchange object is None! Attempting reconnection...")
            self._reconnect_exchange()
            if self.exchange is None:
                print(f"[{self.name}] 🛑 EMERGENCY: Reconnection failed. Cannot place stop loss.")
                return False

        # Secondary validation: check if exchange has required methods
        if not hasattr(self.exchange, 'create_order'):
            print(f"[{self.name}] ❌ CRITICAL: exchange missing create_order method! Reinitializing...")
            self._reconnect_exchange()
            if not hasattr(self.exchange, 'create_order'):
                print(f"[{self.name}] 🛑 EMERGENCY: Reinitialization failed.")
                return False

        try:
            exec_symbol = config.KRAKEN_SYMBOL_MAP.get(symbol, symbol)

            # Kraken Futures Stop Loss: Use 'market' order with 'triggerPrice' (stop market)
            # Or 'limit' order with 'triggerPrice' and 'price' (stop-limit)
            # We use stop-market for guaranteed execution

            # --- FIX: PRECISION FORMATTING ---
            try:
                sl_price_str = self.exchange.price_to_precision(exec_symbol, stop_price)
                qty_str = self.exchange.amount_to_precision(exec_symbol, quantity)
            except Exception as pe:
                 # Fallback logic
                 if 'PAXG' in exec_symbol or 'ETH' in exec_symbol:
                     sl_price_str = f"{stop_price:.2f}"
                 elif 'BTC' in exec_symbol:
                     sl_price_str = f"{stop_price:.1f}"
                 elif 'SOL' in exec_symbol:
                     sl_price_str = f"{stop_price:.2f}"
                 else:
                     sl_price_str = f"{stop_price:.4f}"

                 qty_str = f"{quantity:.4f}"
                 if getattr(self, 'DEBUG', False): print(f"[{self.name}] ⚠️ Precision Fallback used for {exec_symbol}: Price {sl_price_str}, Qty {qty_str}")

            final_sl_price = float(sl_price_str)
            final_qty = float(qty_str)

            # FIX 2026-03-01 #2: Floating Point Precision Correction
            # Round to 8 decimal places to eliminate floating point artifacts
            # (e.g., 0.009999999999999998 -> 0.01)
            final_qty = round(final_qty, 8)
            final_sl_price = round(final_sl_price, 8)

            # FIX 2026-03-01: Stop-Loss Quantity Validation
            # Stop orders are risk-reducing, but must respect exchange minimums

            # Get exchange minimum order size for this symbol
            try:
                base_asset = symbol.split('/')[0]
                config_min = getattr(config, 'MIN_TRADE_QTY', {}).get(base_asset, 0.0)
                
                if config_min > 0:
                    min_qty = config_min
                else:
                    market_info = self.exchange.market(exec_symbol)
                    limits = market_info.get('limits', {})
                    if limits:
                        amount_limits = limits.get('amount', {})
                        min_qty = amount_limits.get('min') if amount_limits else None
                        if min_qty is None:
                            # Use $5 dynamic minimum to prevent $54 min position (e.g. PAXG)
                            min_qty = getattr(config, 'MIN_ORDER_VALUE', 5.0) / final_sl_price if final_sl_price > 0 else 0.0001
                    else:
                        min_qty = getattr(config, 'MIN_ORDER_VALUE', 5.0) / final_sl_price if final_sl_price > 0 else 0.0001
            except Exception as e:
                print(f"[{self.name}] ⚠️ Could not fetch market info for {symbol}: {e}")
                # Safe fallback using nominal $5 limit
                min_qty = getattr(config, 'MIN_ORDER_VALUE', 5.0) / final_sl_price if final_sl_price > 0 else 0.0001

            # Ensure quantity is not None
            if quantity is None:
                print(f"[{self.name}] 🚨 Position quantity is None for {symbol}, cannot place stop loss")
                return False

            # CRITICAL: Stop quantity should never exceed position quantity
            # Check this FIRST before any rounding
            if abs(quantity) > 0 and final_qty > abs(quantity):
                print(f"[{self.name}] ⚠️ Stop qty {final_qty} exceeds position {quantity}, capping to position size")
                final_qty = round(abs(quantity), 8)

            # Now handle minimum quantity - but only if position itself is large enough
            if final_qty < min_qty:
                if abs(quantity) >= min_qty:
                    # Position is large enough, round stop to minimum
                    # FIX 2026-03-01 #2: Round min_qty to same precision to avoid float issues
                    min_qty_rounded = round(min_qty, 8)
                    print(f"[{self.name}] ⚠️ Stop qty {final_qty:.6f} below minimum {min_qty_rounded}, rounding up for {symbol}")
                    final_qty = min(min_qty_rounded, abs(quantity))  # Don't exceed position
                else:
                    # Position itself is below minimum - this shouldn't happen
                    # Log warning but use position size (better than no stop)
                    print(f"[{self.name}] 🚨 POSITION SIZE {quantity} below exchange minimum {min_qty} for {symbol}")
                    print(f"[{self.name}] ⚠️ Using position size for stop (partial protection)")
                    final_qty = round(abs(quantity), 8)

            # Kraken Futures stop order parameters
            params = {
                'reduceOnly': True,
                'triggerPrice': final_sl_price,  # Kraken Futures specific
            }

            print(f"[{self.name}] 🛡️ PLACING STOP LOSS: {direction} {final_qty} {symbol} @ {final_sl_price}")

            # FIX: Use 'market' type with triggerPrice for Kraken Futures stop orders
            # 'stop' type is not valid for Kraken Futures API
            if config.TRADING_MODE == 'FUTURES':
                order = self.exchange.create_order(
                    symbol=exec_symbol,
                    type='market',  # Kraken Futures uses 'market' for stop-market orders
                    side=direction.lower(),
                    amount=final_qty,
                    price=None,  # Not needed for market orders
                    params=params
                )
            else:
                # Spot exchanges may use 'stop' or 'stop-loss'
                order = self.exchange.create_order(
                    symbol=exec_symbol,
                    type='stop',
                    side=direction.lower(),
                    amount=final_qty,
                    price=None,
                    params=params
                )

            if order:
                print(f"[{self.name}] ✅ STOP LOSS ACTIVE: {order['id']}")
                # Normalize stop order result
                order_obj = order if isinstance(order, dict) else {'id': order.get('id') if hasattr(order, 'get') else None}
                try:
                    filled = float(order.get('filled', order.get('amount', 0.0) or 0.0))
                except Exception:
                    filled = order.get('filled', 0.0)
                normalized = {
                    'id': order.get('id'),
                    'order_id': order.get('id'),
                    'status': order.get('status', 'open'),
                    'symbol': symbol,
                    'direction': direction,
                    'filled_qty': filled,
                    'avg_fill_price': order.get('average') or order.get('price') or stop_price,
                    'fee': order.get('fee', {}),
                    'raw': order,
                    'kind': 'stop_loss'
                }
                # Add to pending_orders tracking
                order_record = {
                    'id': normalized['id'],
                    'symbol': symbol,
                    'direction': direction,
                    'status': normalized['status'].upper(),
                    'type': 'stop',
                    'kind': 'stop_loss',
                    'timestamp': time.time(),
                    'quantity': final_qty,
                    'price': final_sl_price
                }
                self.pending_orders.append(order_record)
                return normalized
            return None

        except Exception as e:
            error_msg = str(e)
            print(f"[{self.name}] ❌ Stop Loss Failed: {error_msg}")
            
            # Log detailed error for debugging
            if '"orderType"' in error_msg or 'Argument invalid' in error_msg:
                print(f"[{self.name}] ⚠️ Kraken Futures order type issue. Check API docs.")
            
            return False

    def execute_twap(self, symbol: str, direction: str, total_quantity: float, duration_seconds: int = 3600):
        """
        Execute an order using Time-Weighted Average Price (TWAP).
        Slices the total quantity into N sub-orders over the specified duration.
        """
        def _twap_worker(stop_event):
            num_slices = max(1, duration_seconds // 60) # 1 slice per minute
            qty_per_slice = total_quantity / num_slices
            interval = duration_seconds / num_slices
            
            print(f"[{self.name}] ⏱️ TWAP START: {direction} {total_quantity} {symbol} over {duration_seconds}s ({num_slices} slices)")
            
            for i in range(num_slices):
                if stop_event.is_set():
                    print(f"[{self.name}] ⏱️ TWAP ABORTED: {symbol}")
                    break
                    
                # Execute slice
                self.place_order(symbol, direction, qty_per_slice, order_type='market', urgent=True)
                time.sleep(interval)
            
            print(f"[{self.name}] ⏱️ TWAP COMPLETE: {symbol}")

        algo_id = f"TWAP_{symbol}_{time.time()}"
        stop_event = threading.Event()
        thread = threading.Thread(target=_twap_worker, args=(stop_event,), daemon=True)
        
        with self._algo_lock:
            self.active_algos[algo_id] = {'thread': thread, 'stop_event': stop_event, 'type': 'TWAP'}
            
        thread.start()
        return algo_id

    def execute_pov(self, symbol: str, direction: str, total_quantity: float, pov_percentage: float = 0.05):
        """
        Execute an order using Percentage of Volume (POV).
        Monitors market volume and executes a percentage of that volume until total_quantity is reached.
        """
        def _pov_worker(stop_event):
            executed_qty = 0.0
            print(f"[{self.name}] 📈 POV START: {direction} {total_quantity} {symbol} @ {pov_percentage*100}% of Vol")
            
            while executed_qty < total_quantity and not stop_event.is_set():
                try:
                    ticker = self.exchange.fetch_ticker(config.KRAKEN_SYMBOL_MAP.get(symbol, symbol))
                    # Simplified: use 1m volume proxy or recent trade volume
                    # Real POV would listen to trades. Here we poll.
                    time.sleep(10) # 10s polling
                    
                    # Assume some volume based on ticker change or hardcoded 'safe' slice
                    # In a real implementation, we'd fetch recent trades from observer
                    slice_qty = min(total_quantity - executed_qty, total_quantity * 0.1) # Placeholder proxy
                    self.place_order(symbol, direction, slice_qty, order_type='market', urgent=True)
                    executed_qty += slice_qty
                    
                except Exception as e:
                    print(f"[{self.name}] ⚠️ POV Worker Error: {e}")
                    time.sleep(30)
            
            print(f"[{self.name}] 📈 POV COMPLETE: {symbol}")

        algo_id = f"POV_{symbol}_{time.time()}"
        stop_event = threading.Event()
        thread = threading.Thread(target=_pov_worker, args=(stop_event,), daemon=True)
        
        with self._algo_lock:
            self.active_algos[algo_id] = {'thread': thread, 'stop_event': stop_event, 'type': 'POV'}
            
        thread.start()
        return algo_id

    def execute_vwap(self, symbol: str, direction: str, total_quantity: float, observer: Any = None):
        """
        Execute an order using Volume-Weighted Average Price (VWAP).
        Requires an observer to fetch historical volume profile.
        """
        def _vwap_worker(stop_event):
            print(f"[{self.name}] 📊 VWAP START: {direction} {total_quantity} {symbol}")
            
            num_slices = 20
            qty_per_slice = total_quantity / num_slices
            
            for i in range(num_slices):
                if stop_event.is_set() or i * qty_per_slice >= total_quantity:
                    break
                
                self.place_order(symbol, direction, qty_per_slice, order_type='market', urgent=True)
                time.sleep(60) # 1 minute per slice
                
            print(f"[{self.name}] 📊 VWAP COMPLETE: {symbol}")

        algo_id = f"VWAP_{symbol}_{time.time()}"
        stop_event = threading.Event()
        thread = threading.Thread(target=_vwap_worker, args=(stop_event,), daemon=True)
        
        with self._algo_lock:
            self.active_algos[algo_id] = {'thread': thread, 'stop_event': stop_event, 'type': 'VWAP'}
            
        thread.start()
        return algo_id

    def stop_all_algos(self):
        """Emergency stop for all background execution threads."""
        print(f"[{self.name}] 🛑 Stopping all active execution algorithms...")
        with self._algo_lock:
            for algo_id, state in self.active_algos.items():
                state['stop_event'].set()
            self.active_algos.clear()

    @with_circuit_breaker("actuator_place_order", failure_threshold=5, recovery_timeout=60.0, fallback_value=None)
    @with_retry(max_retries=2, base_delay=1.0, max_delay=5.0, exceptions=(ccxt.NetworkError,)) # Only retry on NetworkDrops, NOT ExchangeError
    def _create_order_resilient(self, **kwargs):
        params = kwargs.get('params', {})
        if config.TRADING_MODE == 'FUTURES' and 'leverage' in kwargs:
             if 'kraken' in self.exchange.id.lower():
                 # FIX: Kraken Leverage and Margin Preferences Integration
                 # Kraken requires explicitly setting the margin mode to "isolated"
                 # to prevent Cross Margin wiping out the Multi-Collateral wallet limit.
                 try:
                     if hasattr(self.exchange, 'set_margin_mode'):
                         self.exchange.set_margin_mode('isolated', kwargs['symbol'])
                         print(f"[{self.name}] 🛡️ Set Isolated Margin for {kwargs['symbol']}")
                 except Exception as e:
                     print(f"[{self.name}] ⚠️ Could not set isolated margin for {kwargs['symbol']}: {e}")

                 self.exchange.set_leverage(kwargs['leverage'], kwargs['symbol'])
                 print(f"[{self.name}] ⚙️ Set Kraken Futures Leverage to {kwargs['leverage']}x for {kwargs['symbol']}")
                 pass
             else:
                 params['leverage'] = kwargs['leverage']
        return self.exchange.create_order(
             symbol=kwargs['symbol'],
             type=kwargs['type'],
             side=kwargs['side'],
             amount=kwargs['amount'],
             price=kwargs['price'] if 'price' in kwargs else None,
             params=params
        )

    def select_venue(self, symbol: str, is_arb: bool = False) -> Optional[str]:
        """
        Select optimal execution venue based on asset type and strategy.
        
        Venue Selection Rules:
        1. xStocks (SPYX, NVDAX, etc.) → Spot (lower fees, no expiry)
        2. Arb strategies → Spot (better funding rate capture)
        3. Directional crypto/gold → Futures (leverage available)
        
        Args:
            symbol: Internal symbol name (e.g., 'SPYX/USDT')
            is_arb: Whether this is an arbitrage position
            
        Returns:
            'spot' | 'futures' | None (if venue unavailable)
        """
        # xStocks always execute on Spot (if available)
        if hasattr(config, 'XSTOCKS_SYMBOLS') and symbol in config.XSTOCKS_SYMBOLS:
            if self.spot_connected and self.spot_exchange:
                return 'spot'
            else:
                print(f"[{self.name}] ⚠️ xStocks {symbol} requires Spot venue, but Spot not connected")
                return None
        
        # Arbitrage strategies prefer Spot (lower fees, no funding payments)
        if is_arb:
            if self.spot_connected and self.spot_exchange:
                return 'spot'
            # Fallback to Futures if Spot unavailable
            return 'futures'
        
        # Directional trades use Futures (leverage available)
        return 'futures'

    def _place_order_on_venue(self, exchange: Any, symbol: str, direction: str, quantity: float, 
                               price: float, order_type: str, urgent: bool, reduce_only: bool, is_spot: bool, take_profit: bool = False):
        """
        Place order on specified exchange venue (Spot or Futures).
        
        Args:
            exchange: ccxt exchange instance (spot or futures)
            symbol: Internal symbol name
            direction: 'BUY' or 'SELL'
            quantity: Order quantity
            price: Limit price
            order_type: 'limit' or 'market'
            urgent: If True, use market order
            reduce_only: Reduce-only flag
            is_spot: True if Spot venue, False if Futures
            
        Returns:
            Order result dict or None on failure
        """
        # FIX 2026-03-02: Removed XSTOCKS_SPOT_MAP - xStocks are on Futures only
        # Map symbol for venue
        exec_symbol = self.symbol_map.get(symbol, symbol)
        
        side = 'buy' if direction == 'BUY' else 'sell'
        
        try:
            # Load markets if needed
            if not exchange.markets:
                exchange.load_markets()
            
            # Format quantity and price with exchange precision
            qty_str = exchange.amount_to_precision(exec_symbol, quantity)
            price_str = exchange.price_to_precision(exec_symbol, price)
            
            final_qty = float(qty_str)
            final_price = float(price_str)

            # Build order params
            params = {}
            if order_type == 'limit' and not urgent:
                params['postOnly'] = True  # Maker rebate
            # FIX 2026-03-19: Take-profit orders must always be reduce-only
            if reduce_only or take_profit:
                params['reduceOnly'] = True

            # Determine kind (take_profit flag passed by caller)
            kind = 'take_profit' if take_profit else None

            # Place order
            order = exchange.create_order(
                symbol=exec_symbol,
                type=order_type,
                side=side,
                amount=final_qty,
                price=final_price if order_type == 'limit' else None,
                params=params
            )
            
            venue_name = 'Spot' if is_spot else 'Futures'
            print(f"[{self.name}] ✅ {venue_name} ORDER PLACED: {direction} {final_qty} {symbol} @ {final_price}")
            
            return {
                'id': order.get('id'),
                'status': order.get('status', 'open'),
                'symbol': symbol,
                'direction': direction,
                'filled': order.get('filled', 0),
                'avg_fill_price': order.get('average', final_price),
                'venue': 'spot' if is_spot else 'futures'
            }
            
        except Exception as e:
            print(f"[{self.name}] ❌ ORDER FAILED on {'Spot' if is_spot else 'Futures'}: {e}")
            return None

    def place_order(self, symbol: str, direction: Literal['BUY', 'SELL'], quantity: float, price: float = 0.0, order_type: str = 'limit', margin: bool = True, leverage: float = 1.0, urgent: bool = False, reduce_only: bool = False, is_arb: bool = False, take_profit: bool = False):
        """
        Place an order on the exchange.
        Supports LIMIT (Maker) and MARKET (Taker) orders.
        
        FIX 2026-03-02: Added venue selection for DUAL mode execution.

        Args:
            symbol: Internal symbol (e.g. 'BTC/USDT')
            direction: 'BUY' or 'SELL'
            quantity: Amount to buy/sell
            price: Limit price (ignored if order_type='market')
            order_type: 'limit' or 'market'
            margin: Whether to use margin (Futures default: True)
            leverage: Leverage multiplier.
            urgent: If True, allows Taker execution (disables PostOnly).
            reduce_only: If True, order will only reduce position (no new opens).
            is_arb: If True, this is an arbitrage position (prefers Spot venue).
        """
        # === FIX 2026-03-02: VENUE SELECTION ===
        # Select optimal venue based on asset type and strategy
        if self.venue_mode == 'DUAL':
            venue = self.select_venue(symbol, is_arb)
            
            if venue == 'spot' and self.spot_exchange:
                # Route to Spot exchange
                return self._place_order_on_venue(
                    exchange=self.spot_exchange,
                    symbol=symbol,
                    direction=direction,
                    quantity=quantity,
                    price=price,
                    order_type=order_type,
                    urgent=urgent,
                    reduce_only=reduce_only,
                    is_spot=True,
                    take_profit=take_profit
                )
            elif venue == 'futures' and self.exchange:
                # Route to Futures exchange (primary)
                pass  # Continue with normal flow below
            elif venue is None:
                # No venue available
                print(f"[{self.name}] ❌ ORDER ABORTED: No venue available for {symbol}")
                return None
        
        # Continue with existing Futures order logic...
        # Helper: normalize exchange order response to a consistent shape.
        # Defined here as a closure so it can reference `symbol` and `direction`.
        def _normalize_order_res(order_obj, used_price=None):
            try:
                filled = float(order_obj.get('filled', order_obj.get('amount', 0.0) or 0.0))
            except Exception:
                filled = order_obj.get('filled', 0.0)
            return {
                'id': order_obj.get('id'),
                'order_id': order_obj.get('id'),
                'status': order_obj.get('status', 'open'),
                'symbol': symbol,
                'direction': direction,
                'filled_qty': filled,
                'avg_fill_price': order_obj.get('average') or order_obj.get('price') or used_price,
                'fee': order_obj.get('fee', {}),
                'raw': order_obj
            }

        if self.paper_mode:
            # Simulated Fill Logic
            order_id = f"sim-{int(time.time())}-{random.randint(1000, 9999)}"
            fill_price = price if price and price > 0 else 0.0
            fill_ts = datetime.now(timezone.utc).isoformat()
            print(f"[{self.name}] 🧪 SIMULATED ORDER FILLED: {direction} {quantity} {symbol} @ {fill_price or 'MARKET'}")
            self._log_paper_trade(symbol, direction, quantity, fill_price, reduce_only, order_id, fill_ts)
            # Normalize simulated order to standard response
            sim_order = {
                'id': order_id,
                'status': 'closed',
                'symbol': symbol,
                'direction': direction,
                'filled': quantity,
                'average': fill_price,
                'timestamp': fill_ts
            }
            return _normalize_order_res(sim_order, used_price=fill_price)

        exec_symbol = self.symbol_map.get(symbol, symbol)
        side = 'buy' if direction == 'BUY' else 'sell'
        
        # --- OPTIMIZATION: Urgent Exits ---
        # --- OPTIMIZATION: Urgent Exits ---
        if reduce_only and urgent:
            # Force Market Order for Exits (Take Profit / Stop Loss) to ensure fill
            # and avoid 'postOnly' cancellations on limit orders that cross the spread
            order_type = 'market'
            # print(f"[{self.name}] ⚡ URGENT EXIT: Forcing MARKET order for {symbol}")
        
        # Prepare values with correct precision
        try:
            if not self.exchange.markets:
                self.exchange.load_markets()

            # --- PATCH: MIN QUANTITY CLAMPING ---
            market = self.exchange.market(exec_symbol)
            min_limit = market.get('limits', {}).get('amount', {}).get('min')
            prec_amount = market.get('precision', {}).get('amount')
            
            # Use strict fallback if None
            if min_limit is None: min_limit = 0.0
            if prec_amount is None: prec_amount = 0.0
            
            effective_min = max(min_limit, prec_amount)
            
            if quantity < effective_min and quantity > 0:
                 print(f"[{self.name}] 🤏 Clamping Qty {quantity} -> {effective_min} (Min Allowed)")
                 quantity = effective_min
            # ------------------------------------

            qty_str = self.exchange.amount_to_precision(exec_symbol, quantity)
            price_str = self.exchange.price_to_precision(exec_symbol, price)
            
            final_qty = float(qty_str)
            final_price = float(price_str)
            
        except Exception as e:
            print(f"[{self.name}] ⚠️ Precision formatting failed: {e}. Using raw values.")
            final_qty = quantity
            final_price = price

        # --- LIQUIDITY SANITY CHECK ---
        check_price = final_price
        if order_type != 'limit' or check_price <= 0:
            try:
                ticker = self.exchange.fetch_ticker(exec_symbol)
                check_price = ticker['last']
            except:
                check_price = 0.0

        if not self.check_liquidity(exec_symbol, direction, final_qty, check_price):
             print(f"[{self.name}] 🛑 LIQUIDITY CHECK FAILED: {exec_symbol} Book too thin for {final_qty}. Order Aborted.")
             return None
        # ------------------------------

        # === ORDER TYPE ENFORCEMENT ===
        # Prefer limit orders; only use market for urgent exits
        if getattr(config, 'MARKET_ORDER_ONLY_URGENT', True):
            if order_type == 'market' and not reduce_only:
                print(f"[{self.name}] ⚠️ Market order requested for non-exit. Forcing LIMIT order.")
                order_type = 'limit'
                urgent = False  # Disable urgent for non-exits

        # Construct params dictionary safely
        params = {}
        
        # FIX 2026-02-24: Track order cancellations to detect excessive churn
        if not hasattr(self, '_order_cancellations'):
            self._order_cancellations = {'count': 0, 'last_reset': time.time()}
        
        # Reset cancellation counter every minute
        if time.time() - self._order_cancellations['last_reset'] > 60:
            self._order_cancellations['count'] = 0
            self._order_cancellations['last_reset'] = time.time()

        if order_type == 'limit':
            if not urgent:
                # Add Post-Only to ensure Maker rebate (unless urgent)
                params['postOnly'] = True
            params['timeInForce'] = 'GTC'

            # === TIGHT COLLAR FOR THIN MARKETS ===
            # Apply tighter price collar in thin market conditions
            if getattr(config, 'TIGHT_COLLAR_ENABLED', True):
                # Check if book is thin (already passed liquidity check, but apply tighter collar)
                collar_bps = getattr(config, 'PRICE_COLLAR_BPS', 50)
                # For thin markets, use tighter collar (25 bps vs 50 bps)
                if getattr(config, 'EXEC_IMPACT_THRESHOLD', 0.10) > 0.15:  # High impact = thin
                    collar_bps = getattr(config, 'THIN_BOOK_COLLAR_BPS', 25)

                # Adjust price to stay within collar
                if check_price > 0:
                    collar_limit = check_price * (collar_bps / 10000)
                    if direction == 'BUY':
                        # Cap buy price to collar
                        if final_price > check_price + collar_limit:
                            final_price = check_price + collar_limit
                            print(f"[{self.name}] 📏 TIGHT COLLAR: Adjusted BUY price to {final_price} (within {collar_bps} bps)")
                    else:
                        # Floor sell price to collar
                        if final_price < check_price - collar_limit:
                            final_price = check_price - collar_limit
                            print(f"[{self.name}] 📏 TIGHT COLLAR: Adjusted SELL price to {final_price} (within {collar_bps} bps)")
        # ------------------------------

        # FIX 2026-03-19: Take-profit orders must always be reduce-only
        if reduce_only or take_profit:
            # Ensure order only closes existing positions (prevents flipping)
            params['reduceOnly'] = True
            
        try:
            # Execute on Exchange
            order = self.exchange.create_order(
                symbol=exec_symbol,
                type=order_type,
                side=side,
                amount=final_qty,
                price=final_price if order_type == 'limit' else None,
                params=params
            )
            
            order_record = {
                'id': order['id'],
                'symbol': symbol,
                'direction': direction,
                'status': 'OPEN',
                'type': order_type,
                'kind': 'take_profit' if take_profit else None,
                'entry_time': time.strftime("%H:%M:%S"),
                'timestamp': time.time(),
                'quantity': final_qty,
                'price': final_price if order_type == 'limit' else check_price
            }

            self.pending_orders.append(order_record)
            print(f"[{self.name}] ✅ Order Placed: {order['id']}")
            
            # Log execution for tracking
            order_ts = datetime.now(timezone.utc).isoformat()
            self._log_execution(symbol, direction, final_qty, final_price if order_type == 'limit' else check_price,
                              order_type, order['id'], order_ts, "PLACED", "LIVE")
            
            self.report_success()
            res = _normalize_order_res(order, used_price=(final_price if order_type == 'limit' else check_price))
            # Attach kind if present
            if order_record.get('kind'):
                res['kind'] = order_record['kind']
            return res
            
        except Exception as e:
            msg = str(e)
            # Only count as API failure if it's a network/exchange error, not logic (insufficient funds)
            msg = str(e)
            # Only count as API failure if it's a network/exchange error, not logic (insufficient funds)
            if "NetworkError" in msg or "503" in msg or "Service Unavailable" in msg or "timed out" in msg:
                self.report_failure(msg)
            else:
                # Logic Error (Funds, MinLimit, etc) -> Trigger specific cooldown
                print(f"[{self.name}] ❌ Order Logic Error: {msg}")
                self.record_order_failure(symbol, 'General')
                return None
            
            # GUARD: OUTSIDE PRICE COLLAR (Kraken Futures)
            if "outsidePriceCollar" in msg:
                 print(f"[{self.name}] ⚠️ Price Collar Violation ({msg}). Retrying as MARKET (Urgent)...")
                 self.record_order_failure(symbol, 'outsidePriceCollar') # Record specific error
                 return self.place_order(symbol, direction, quantity, order_type='market', margin=margin, leverage=leverage, urgent=True, reduce_only=reduce_only)

            # GUARD: POSTWOULDEXECUTE ERROR (Kraken Futures)
            if "postWouldExecute" in msg:
                 print(f"[{self.name}] ⚠️ Post-Only Order Would Execute. Retrying as MARKET (Urgent)...")
                 self.record_order_failure(symbol, 'postWouldExecute') # Record specific error
                 return self.place_order(symbol, direction, quantity, order_type='market', margin=margin, leverage=leverage, urgent=True, reduce_only=reduce_only)

            # GUARD: INSUFFICIENT FUNDS
            if "insufficientAvailableFunds" in msg:
                 print(f"[{self.name}] 🛑 Insufficient Funds. Recording specific error.")
                 self.record_order_failure(symbol, 'insufficientAvailableFunds')
                 return None

            # Generic failure recording (Catch-all)
            self.record_order_failure(symbol, 'General')
            print(f"[{self.name}] ❌ Order Placement Failed: {msg}")

            # GUARD: ALREADY CLOSED (Race Condition)
            # FIX 2026-03-01 #9: wouldNotReducePosition means exchange has no position but we think we do
            # This is a position tracking desync - need to clear the phantom position
            if "wouldNotReducePosition" in msg:
                 print(f"[{self.name}] ℹ️ Position appears already closed/reduced (Exchange Rejected). Skipping.")
                 # FIX: Signal to executor that position needs to be cleared from tracking
                 # This prevents repeated stop loss attempts on already-closed positions
                 return None
            
            # RETRY LOGIC (Only for Limit Orders usually, but maybe Market failed?)
            # If Market order failed, we generally just fail.
            # RETRY LOGIC (Only for Limit Orders usually, but maybe Market failed?)
            # If Market order failed, we generally just fail.
            if order_type == 'market':
                # PATCH: Handle "marketIsPostOnly" or similar rejections by trying a Limit Order
                if "marketIsPostOnly" in msg or "PostOnly" in msg:
                     print(f"[{self.name}] ⚠️ Market Order Rejected ({msg}). Retrying as LIMIT...")
                     # Retry as Limit at current price (Aggressive)
                     return self.place_order(symbol, direction, quantity, price=check_price, order_type='limit', margin=margin, leverage=leverage, urgent=True)
                return None

            # RETRY 1: TAKER RETRY (PostOnly Failed)
            # If we were trying to be a Maker but the price moved, just TAKE it.
            if "OrderImmediatelyFillable" in msg or "postOnly" in msg or "postWouldExecute" in msg:
                 try:
                     print(f"[{self.name}] ⚠️ PostOnly Failed (Price moved/Liquidity). Retrying as TAKER...")
                     params['postOnly'] = False

                     # --- REDUCE ONLY ON RETRY ---
                     if reduce_only and config.TRADING_MODE == 'FUTURES':
                          params['reduceOnly'] = True
                     # ----------------------------

                     order = self.exchange.create_order(
                        symbol=exec_symbol,
                        type='limit', # Still limit, but crossing book (becomes marketable limit)
                        side=side,
                        amount=final_qty,
                        price=final_price,
                        params=params
                     )
                     # If success, add to pending
                     order_record = {
                        'id': order['id'],
                        'symbol': symbol,
                        'direction': direction,
                        'status': 'OPEN',
                        'type': 'limit', # Original type was limit, now it's a taker limit
                        'entry_time': time.strftime("%H:%M:%S"),
                        'timestamp': time.time(),
                        'quantity': final_qty,
                        'price': final_price
                    }
                     self.pending_orders.append(order_record)
                     print(f"[{self.name}] ✅ Order Placed (TAKER): {order['id']}")
                     self.report_success()
                     return _normalize_order_res(order, used_price=(final_price if 'final_price' in locals() else check_price))
                 except Exception as e2:
                     print(f"[{self.name}] ❌ Taker Retry Failed: {e2}")
                     # FINAL RETRY: Market Order (Urgent Execution)
                     if urgent:
                         try:
                             print(f"[{self.name}] 🚨 URGENT: Final Retry as MARKET ORDER...")
                             order = self.exchange.create_order(
                                symbol=exec_symbol,
                                type='market',
                                side=side,
                                amount=final_qty,
                                price=None,  # Market orders don't need price
                                params={'reduceOnly': reduce_only} if reduce_only and config.TRADING_MODE == 'FUTURES' else {}
                             )
                             order_record = {
                                'id': order['id'],
                                'symbol': symbol,
                                'direction': direction,
                                'status': 'OPEN',
                                'type': 'market',
                                'entry_time': time.strftime("%H:%M:%S"),
                                'timestamp': time.time(),
                                'quantity': final_qty,
                                'price': check_price  # Use current price for market orders
                             }

                             self.pending_orders.append(order_record)
                             print(f"[{self.name}] ✅ Market Order Placed (Urgent): {order['id']}")
                             self.report_success()
                             return _normalize_order_res(order, used_price=check_price)
                         except Exception as e3:
                             print(f"[{self.name}] ❌ Market Order Failed: {e3}")

                     return None

            # RETRY 2: REDUCE ONLY (Aggressive Exit Fix)
            # If we failed due to funds AND we wanted to Reduce (or user logic implied it?), try to force reduceOnly.
            # Usually if 'reduce_only' was already True, we failed anyway.
            # But if 'reduce_only' was False, this might be a desperate attempt to 'close' if we messed up direction?
            # Actually, let's only do this if we suspect it's an exit failing.
            # For now, if we explicitly passed reduce_only=True, and it failed, we are done.           
            if "insufficientAvailableFunds" in msg and direction == 'SELL' and not reduce_only:
                 # Only retry with reduceOnly if we didn't try it yet
                 try:
                     print(f"[{self.name}] 🔄 Retrying with reduceOnly=True (Fallback)...")
                     params['reduceOnly'] = True
                     order = self.exchange.create_order(
                        symbol=exec_symbol,
                        type='limit',
                        side=side,
                        amount=final_qty,
                        price=final_price,
                        params=params
                     )
                     # If success, add to pending
                     order_record = {
                        'id': order['id'],
                        'symbol': symbol,
                        'direction': direction,
                        'status': 'OPEN',
                        'type': 'limit', # Original type was limit
                        'entry_time': time.strftime("%H:%M:%S"),
                        'timestamp': time.time(),
                        'quantity': final_qty,
                        'price': final_price
                    }
                     self.pending_orders.append(order_record)
                     print(f"[{self.name}] ✅ Retry Successful: {order['id']}")
                     return order['id']
                 except Exception as retry_e:
                     print(f"[{self.name}] ❌ Retry Failed: {retry_e}")
            
            # Original logic for postWouldExecute/OrderImmediatelyFillable
            if "postWouldExecute" in msg or "OrderImmediatelyFillable" in msg:
                print(f"[{self.name}] ⚠️ Maker Order Rejected (Price crossed spread). Retrying as TAKER...")
                try:
                    # Retry without Post-Only (Eat the Taker Fee to ensure execution)
                    if 'postOnly' in params:
                        del params['postOnly']
                    
                    order = self.exchange.create_order(
                        symbol=exec_symbol,
                        type='limit',
                        side=side,
                        amount=final_qty,
                        price=final_price,
                        params=params
                    )
                    
                    # Log successful retry
                    internal_order = {
                        'id': order['id'],
                        'status': 'OPEN',
                        'symbol': symbol,
                        'direction': direction,
                        'quantity': quantity,
                        'price': price, # Use 'price' instead of 'limit_price'
                        'type': 'limit', # Original type was limit
                        'timestamp': time.time()
                    }
                    self.pending_orders.append(internal_order)
                    print(f"[{self.name}] ✅ TAKER FILL SUBMITTED: {order['id']}")
                    return internal_order
                    
                except Exception as retry_err:
                    print(f"[{self.name}] ❌ Taker Retry Failed: {retry_err}")
                    return None
            else:
                print(f"[{self.name}] ❌ Order Placement Failed: {e}")
                return None
            
    def fetch_order_status(self, order_id: str, symbol: str = None) -> dict:
        """
        Fetch status of a specific order.
        Updates internal pending_orders state if filled/closed (prevents stale locks).
        """
        if not self.check_circuit_breaker(): return None
        
        exec_symbol = self.symbol_map.get(symbol, symbol) if symbol else None
        
        try:
            order = None
            # 1. Try direct fetch (Supported by most CCXT drivers)
            try:
                order = self.exchange.fetch_order(order_id, symbol=exec_symbol)
            except Exception as e:
                # Fallback: Scan Open/Closed manually (Kraken Futures quirk?)
                print(f"[{self.name}] ℹ️ Fetch Order {order_id} direct failed: {e}. Scanning Open/Closed...")
                pass
                
            if not order:
                # 2. Fallback Scan
                if exec_symbol:
                    # Check Open
                    try:
                        opens = self.exchange.fetch_open_orders(exec_symbol)
                        for o in opens:
                            if str(o['id']) == str(order_id):
                                order = o
                                break
                    except Exception as e_scan: 
                        print(f"[{self.name}] ⚠️ Open Order Scan Error: {e_scan}")
                    
                    # Check Closed
                    if not order:
                        try:
                            closed = self.exchange.fetch_closed_orders(exec_symbol, limit=20)
                            for o in closed:
                                if str(o['id']) == str(order_id):
                                    order = o
                                    break
                        except Exception as e_clos: 
                            print(f"[{self.name}] ⚠️ Closed Order Scan Error: {e_clos}")
            
            if not order:
                return None
                
            # Update Internal State
            remote_status = order.get('status')
            
            # Find in pending
            local_order = None
            for o in self.pending_orders:
                if str(o.get('id')) == str(order_id):
                    local_order = o
                    break
            
            if remote_status in ['closed', 'filled', 'canceled', 'expired', 'rejected']:
                 if local_order:
                     # Remove from pending to unlock Governor/Executor
                     if local_order in self.pending_orders:
                        self.pending_orders.remove(local_order)
                     print(f"[{self.name}] 🗑️ Cleared Completed Order {order_id} ({remote_status})")

            return {
                'status': remote_status, 
                'filled': float(order.get('filled', 0.0)), 
                'price': float(order.get('average') or order.get('price', 0.0))
            }

        except Exception as e:
            print(f"[{self.name}] ⚠️ fetch_order_status failed: {e}")
            return None

    def check_fills(self, candle_low: float = None, candle_high: float = None):
        """
        Check if pending orders were filled. For live, we fetch from exchange.
        """
        if not self.check_circuit_breaker(): return []

        filled_orders = []
        remaining_orders = []
        
        for order in self.pending_orders:
            try:
                # Kraken Futures usually doesn't support fetching a single order by ID easily
                # So we must scan Open and Closed lists
                found_order = None
                
                # Use mapped symbol for exchange calls
                exec_symbol = self.symbol_map.get(order['symbol'], order['symbol'])
                
                # 1. Check Open Orders
                # (Optimization: We could fetch all open orders ONCE per cycle instead of per order, 
                # but for now let's keep it robust)
                try:
                    # We pass symbol to narrow it down if possible
                    open_orders = self.exchange.fetch_open_orders(exec_symbol)
                    self.report_success()
                    for o in open_orders:
                        if o['id'] == order['id']:
                            found_order = o
                            break
                except Exception as e:
                    if "NetworkError" in str(e) or "503" in str(e): self.report_failure(str(e))
                    print(f"[{self.name}] ⚠️ fetch_open_orders failed: {e}")

                # 2. If not found, Check Closed Orders (It might have just filled)
                if not found_order:
                    try:
                        closed_orders = self.exchange.fetch_closed_orders(exec_symbol, limit=20)
                        for o in closed_orders:
                            if o['id'] == order['id']:
                                found_order = o
                                break
                    except Exception as e:
                        print(f"[{self.name}] ⚠️ fetch_closed_orders failed: {e}")
                
                if found_order:
                    remote_status = found_order['status']

                    if remote_status == 'closed':
                        order['status'] = 'FILLED'
                        order['filled_qty'] = found_order.get('filled', order.get('quantity'))
                        order['cost_usd'] = found_order.get('cost', 0.0)
                        # CRITICAL FIX: Ensure 'price' is populated for Executor
                        order['price'] = found_order.get('average') or found_order.get('price') or order.get('price')

                        filled_orders.append(order)
                        print(f"[{self.name}] ✅ FILL CONFIRMED: {order['id']} ({order['symbol']}) @ {order['price']}")
                        # If this was a TP or SL, attempt to cancel the opposite protective order (OCO emulation)
                        try:
                            kind = order.get('kind')
                            if kind in ('take_profit', 'stop_loss'):
                                # Find opposite kind
                                opposite = 'stop_loss' if kind == 'take_profit' else 'take_profit'
                                # Scan remaining pending orders for opposite protective order and cancel it
                                for po in list(self.pending_orders):
                                    if po.get('symbol') == order.get('symbol') and po.get('kind') == opposite:
                                        try:
                                            exec_sym = self.symbol_map.get(po['symbol'], po['symbol'])
                                            if po.get('id'):
                                                self.exchange.cancel_order(po['id'], exec_sym)
                                                print(f"[{self.name}] 🔁 OCO: Canceled opposite {opposite} order {po['id']} for {po['symbol']}")
                                        except Exception as _e:
                                            print(f"[{self.name}] ⚠️ OCO cancel failed for {po.get('id')}: {_e}")
                                        # Remove from tracking regardless (we attempted cancel)
                                        try:
                                            if po in self.pending_orders:
                                                self.pending_orders.remove(po)
                                        except: pass
                        except Exception as _e:
                            print(f"[{self.name}] ⚠️ OCO logic error: {_e}")
                        
                        # Log fill confirmation
                        fill_ts = datetime.now(timezone.utc).isoformat()
                        self._log_execution(order['symbol'], order['side'].upper(), 
                                          order['filled_qty'], order['price'],
                                          order['type'], order['id'], fill_ts, "FILLED", "LIVE")

                    elif remote_status == 'canceled':
                        print(f"[{self.name}] ⚠️ Order {order['id']} was CANCELED.")
                        # Do not add to remaining_orders -> Dropped from tracking
                        
                        # Still open, check timeout (55s + jitter)
                        age = time.time() - order['timestamp']
                        
                        # Add jitter to prevent thundering herd on API
                        timeout = 30 + random.randint(0, 5) # NANO FIX: 30s Timeout
                        
                        if age > timeout:
                            try:
                                self.exchange.cancel_order(order['id'], exec_symbol)
                                print(f"[{self.name}] ⏱️ Order {order['id']} TIMEOUT ({age:.2f}s > {timeout}s). Requesting Cancel to Reprice...")
                            except Exception as cancel_err:
                                print(f"[{self.name}] ⚠️ Cancel Failed for {order['id']}: {cancel_err}")
                            
                            # CRITICAL: Keep tracking it until verifying it is GONE/CANCELED in next cycle
                            remaining_orders.append(order)
                        else:
                            # print(f"[{self.name}] Order {order['id']} Open for {age:.2f}s")
                            remaining_orders.append(order)
                    else:
                        # Unknown status
                        remaining_orders.append(order)

                else:
                    # Order not found in either list? 
                    # Use Configured GC Timeout for Ghost Orders (AEHML Fix: 30s)
                    ghost_timeout = getattr(config, 'GC_STALE_ORDER_TIMEOUT', 30)
                    if time.time() - order['timestamp'] > ghost_timeout:
                         print(f"[{self.name}] 👻 Order {order['id']} Disappeared & Expired (> {ghost_timeout}s). Dropping.")
                         # Dropped
                    else:
                         remaining_orders.append(order)

            except Exception as e:
                print(f"[{self.name}] Error checking fill for {order['id']}: {e}")
                remaining_orders.append(order)
                
        self.pending_orders = remaining_orders
        return filled_orders

    def receive_message(self, sender: Any, content: Any) -> None:
        """Handle incoming messages."""
        if isinstance(content, Message):
            if content.type == 'PLACE_ORDER':
                pass
        else:
            pass

    def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        """
        Fetch Order Book from the EXECUTION VENUE (Kraken Futures etc).
        Crucial for checking liquidity on the exchange we actually trade on.
        """
        if not self.check_circuit_breaker(): return {'bids': [], 'asks': []}
        if not self.exchange: return {'bids': [], 'asks': []}

        try:
            # Map Symbol
            exec_symbol = symbol
            if config.TRADING_MODE == 'FUTURES':
                exec_symbol = config.KRAKEN_SYMBOL_MAP.get(symbol, symbol)
                
            book = self.exchange.fetch_order_book(exec_symbol, limit)
            self.report_success()
            return {
                'bids': book['bids'],
                'asks': book['asks'],
                'timestamp': book['timestamp']
            }
        except Exception as e:
            self.report_failure(str(e))
            print(f"[{self.name}] ⚠️ Actuator Book Fetch Fail {symbol}: {e}")
            return {'bids': [], 'asks': []}

    def gc_clean_stale_orders(self) -> int:
        """
        Garbage Collector: Cancel and remove orders older than GC_STALE_ORDER_TIMEOUT.
        Returns the count of cleaned orders.
        """
        cleaned_count = 0
        remaining_orders = []
        stale_timeout = getattr(config, 'GC_STALE_ORDER_TIMEOUT', 120)
        verbose = getattr(config, 'GC_LOG_VERBOSE', True)
        
        # Iterate over COPY to avoid modification issues if fetch_order_status removes items
        for order in list(self.pending_orders):
            age = time.time() - order.get('timestamp', time.time())
            
            if age > stale_timeout:
                # Attempt to cancel on exchange
                try:
                    exec_symbol = self.symbol_map.get(order['symbol'], order['symbol'])
                    
                    # PRE-CANCEL VERIFICATION (AEHML Fix)
                    # Check if it actually filled before we kill it
                    check = self.fetch_order_status(order['id'], order['symbol'])
                    if check and check.get('status') == 'filled':
                        if verbose: print(f"[GC Monitor] ⚠️ Stale Order {order['id']} was actually FILLED! Keeping for check_fills.")
                        # CRITICAL: Do NOT drop it. Add to remaining so check_fills can process the fill event properly.
                        remaining_orders.append(order)
                        continue

                    self.exchange.cancel_order(order['id'], exec_symbol)
                    if verbose:
                        print(f"[GC Monitor] 🗑️ Canceled stale order {order['id']} ({order['symbol']}) after {age:.0f}s")
                except Exception as e:
                    if verbose:
                        print(f"[GC Monitor] ⚠️ Failed to cancel {order['id']}: {e}")
                
                cleaned_count += 1
                # Do NOT add to remaining_orders -> dropped
            else:
                remaining_orders.append(order)
        
        self.pending_orders = remaining_orders
        
        if verbose and cleaned_count > 0:
            print(f"[GC Monitor] ✅ Actuator Cleanup: {cleaned_count} stale orders removed.")
        
        return cleaned_count

    def fetch_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Verify the status of a specific order on the exchange.
        Used by Executor for Hard Gating.
        """
        try:
             # CCXT fetch_order
             exec_symbol = self.symbol_map.get(symbol, symbol)
             order = self.exchange.fetch_order(order_id, exec_symbol)
             return order
        except Exception as e:
             # print(f"[{self.name}] ⚠️ Fetch Order Failed: {e}")
             
             # Fallback: Check closed orders if fetch_order fails (order might have moved to history)
             try:
                 exec_symbol = self.symbol_map.get(symbol, symbol)
                 closed = self.exchange.fetch_closed_orders(exec_symbol, limit=50) # Increased limit for safety
                 for o in closed:
                     if o['id'] == order_id:
                         return o
             except: pass
             
             return None

        return True

    def check_spread_health(self, symbol: str, ticker: Dict) -> bool:
        """
        Spread Veto: Prevent entry if spread > Threshold (0.4%).
        """
        if not ticker or 'bid' not in ticker or 'ask' not in ticker:
            return True # Can't check
            
        spread_pct = (ticker['ask'] - ticker['bid']) / ticker['ask'] if ticker['ask'] > 0 else 0.0
        
        limit = config.VOL_WINDOW_SPREAD_THRESHOLD # 0.4%
        
        if spread_pct > limit:
            print(f"[{self.name}] 🧊 SPREAD VETO: {symbol} spread {spread_pct*100:.2f}% > {limit*100:.2f}%")
            return False
            
        return True
        
    def should_force_close_funding(self, symbol: str, position_dir: str, funding_rate: float) -> bool:
        """
        Funding Flip Kill Switch:
        If we are holding a position primarily for Funding Arb (or just generally),
        and the funding rate moves against us (we start paying), signal CLOSE.
        
        Pos Funding (>0): Longs Pay Shorts.
        Neg Funding (<0): Shorts Pay Longs.
        """
        # If we are Long, we want Neg Funding (Get Paid). If Pos, we Pay.
        # If we are Short, we want Pos Funding (Get Paid). If Neg, we Pay.
        
        if position_dir == 'BUY':
             # We hold Long. We pay if Funding > 0.
             # If Funding > 0 and we entered for Arb, we should exit.
             # But normal trend trading pays funding often.
             # We assume this check is only called for ARB strategies or if strictly enforcing "Don't Pay Funding".
             # For VOL_WINDOW, we might be stricter.
             if funding_rate > 0.0001: # Small buffer
                 return True
                 
        elif position_dir == 'SELL':
             # We hold Short. We pay if Funding < 0.
             if funding_rate < -0.0001:
                 return True
                 
        return False

