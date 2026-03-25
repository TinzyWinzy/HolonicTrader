import os
import ccxt
import time
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone
from typing import Literal, Any, List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from HolonicTrader.holon_core import Holon, Disposition, Message

import requests
import random
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import config

import threading
import asyncio
import ccxt.pro as ccxtpro

from HolonicTrader.network_resilience import with_retry, with_circuit_breaker

# AEGIS QUANTSEC: WebSocket Health Monitoring
from HolonicTrader.websocket_health import (
    get_global_health_monitor,
    WebSocketHealthMonitor
)

class ObserverHolon(Holon):
    """
    ObserverHolon is responsible for acquiring market data from exchanges
    and processing it for other agents (like the Entropy Agent).
    """

    # Class-Level Shared Cache to prevent redundant Disk I/O across instances
    _shared_cache = {}
    _shared_cache_lock = threading.Lock() # Ensure thread safety
    _xstock_bases_cache = None  # Cache for xStock base symbols

    def _is_xstock_symbol(self, symbol: str) -> bool:
        """
        FIX 2026-03-02: Check if a symbol is an xStock variant.
        Handles formats like 'SPYX/USDT', 'SPYX/USD', 'SPYX/USD:USD', etc.
        """
        if not hasattr(config, 'XSTOCKS_SYMBOLS'):
            return False
        
        # Build cache of xStock base names on first call
        if ObserverHolon._xstock_bases_cache is None:
            with ObserverHolon._shared_cache_lock:
                if ObserverHolon._xstock_bases_cache is None:
                    ObserverHolon._xstock_bases_cache = set()
                    for xs_sym in config.XSTOCKS_SYMBOLS:
                        base = xs_sym.split('/')[0] if '/' in xs_sym else xs_sym
                        ObserverHolon._xstock_bases_cache.add(base)
        
        # Check exact match first
        if symbol in config.XSTOCKS_SYMBOLS:
            return True
        
        # Check base symbol match
        base = symbol.split('/')[0] if '/' in symbol else symbol
        return base in ObserverHolon._xstock_bases_cache

    def __init__(self, exchange_id: str = 'kucoin', symbol: str = 'BTC/USDT'):
        # Initialize with default highly autonomous and integrated disposition for now
        # or minimal, depending on system design. Using balanced values here.
        default_disposition = Disposition(autonomy=0.5, integration=0.5)
        super().__init__(name=f"Observer_{exchange_id}_{symbol}", disposition=default_disposition)
        
        self.symbol = symbol
        self.exchange_id = exchange_id
        
        # Initialize exchange with rate limiting and larger pool size
        if hasattr(ccxt, exchange_id):
            # Create a custom session with a larger connection pool
            session = requests.Session()
            adapter = HTTPAdapter(
                pool_connections=config.CCXT_POOL_SIZE, 
                pool_maxsize=config.CCXT_POOL_SIZE
            )
            session.mount('https://', adapter)
            session.mount('http://', adapter)

            self.exchange = getattr(ccxt, exchange_id)({
                'enableRateLimit': config.CCXT_RATE_LIMIT,
                'timeout': 30000, # 30s Timeout to prevent sticky threads
                'session': session
            })
        else:
            raise ValueError(f"Exchange {exchange_id} not found in ccxt")

        # Map for local history files
        self.data_dir = os.path.join(os.getcwd(), 'market_data')
        
        # Data Cache (Instance level alias to shared?)
        # self._cache = {} # Legacy instance cache
        # WS Cache
        self._ticker_cache = {}
        self._last_ticker_fetch = 0.0
        
        # --- PHASE 46.2: WEBSOCKET INFRASTRUCTURE ---
        self._ws_enabled = False
        self._ws_symbols = []
        self._ws_thread = None
        self._ws_loop = None
        self._ws_exchange = None
        self._ws_trades_log = {} # symbol -> [trades]
        
        # Smart OHLCV Cache (Phase 1 WS Integration)
        self._smart_ohlcv_cache = {} # symbol_timeframe -> {'df': DataFrame, 'last_fetch': float, 'candle_open_time': float}

        # Determine if we should start WS (only for primary exchange if needed)
        # For now, we allow any ObserverHolon to start WS if symbols are provided
        if config.TRADING_MODE == 'FUTURES' or exchange_id == 'krakenfutures':
            self._ws_enabled = True
            # Start in a separate method to avoid blocking init

        # --- WS UPDATE DEBOUNCING (Prevent Kraken subscription spam) ---
        self._ws_last_update = 0.0
        self._WS_UPDATE_MIN_INTERVAL = 300  # 5 minutes between WS subscription updates

        # --- SHARED EXECUTOR (Prevent Thread/Socket Leak) ---
        self.executor = ThreadPoolExecutor(
            max_workers=config.CCXT_POOL_SIZE,
            thread_name_prefix=f"{self.name}_Worker"
        )

        # === AEGIS QUANTSEC: WebSocket Health Monitor ===
        # CHRONOS FIX: Respect REST-only mode configuration
        self._ws_health_monitor: Optional[WebSocketHealthMonitor] = None
        self._ws_health_enabled = not getattr(config, 'WS_FORCE_REST_ONLY', False)
        self._ws_fallback_to_rest = getattr(config, 'WS_FORCE_REST_ONLY', False)  # Start in REST mode if configured
        self._ws_last_rest_fetch = 0.0
        self._ws_rest_cooldown = 5.0  # Seconds between REST fallback fetches
        self._ws_unhealthy_threshold = getattr(config, 'WS_UNHEALTHY_THRESHOLD', 0.5)

    def start_ws(self, symbols: List[str] = None):
        """Starts the background WebSocket thread."""
        if not self._ws_enabled or self._ws_thread is not None:
            return

        # Use provided symbols or default university
        watch_list = symbols if symbols else [self.symbol]
        # Map symbols for Kraken Futures if needed
        if self.exchange_id == 'krakenfutures':
            watch_list = [config.KRAKEN_SYMBOL_MAP.get(s, s) for s in watch_list]

        self._ws_symbols = watch_list
        print(f"[{self.name}] 📡 Starting WebSocket Stream for {len(self._ws_symbols)} assets...")

        # === AEGIS QUANTSEC: Initialize Health Monitor ===
        if self._ws_health_enabled:
            self._ws_health_monitor = get_global_health_monitor()

            # Register all symbols for health monitoring
            for sym in self._ws_symbols:
                self._ws_health_monitor.register_connection('tickers', sym)

            # Register callbacks for health status changes
            self._ws_health_monitor.register_unhealthy_callback(self._on_ws_unhealthy)
            self._ws_health_monitor.register_recovered_callback(self._on_ws_recovered)

            print(f"[{self.name}] 🛡️ AEGIS WebSocket Health Monitor enabled")

        self._ws_thread = threading.Thread(target=self._run_ws_loop, daemon=True)
        self._ws_thread.start()

    def _run_ws_loop(self):
        """Entry point for the WS thread."""
        self._ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._ws_loop)
        try:
            self._ws_loop.run_until_complete(self._ws_main_loop())
        except Exception as e:
            print(f"[{self.name}] ❌ WebSocket Thread Crashed: {e}")

    async def _ws_main_loop(self):
        """The actual async loop using ccxt.pro with enhanced keepalive."""
        # === FIX 2026-03-15: Aggressive WebSocket Keepalive ===
        # Previous 15s ping was insufficient for unstable networks
        # New config: 5s ping interval, 3s timeout tolerance
        ws_config = {
            'enableRateLimit': True,
            'pingInterval': 5000,  # 5s ping interval (was 10s/15s)
            'pingTimeout': 3000,   # 3s timeout before reconnect
            # Enhanced keepalive settings to prevent timeout issues
            'options': {
                'defaultType': 'future',  # For Kraken Futures
                'keepAlive': True,
                'heartbeat': True,
                'heartbeatDelay': 5000,  # 5s heartbeat (was 15s)
            }
        }

        # Exchange-specific tuning
        if self.exchange_id == 'krakenfutures':
            # Kraken Futures needs more aggressive keepalive
            ws_config['pingInterval'] = 3000  # 3s ping
            ws_config['pingTimeout'] = 2000   # 2s timeout
            ws_config['options']['heartbeatDelay'] = 3000
        elif self.exchange_id == 'kucoin':
            # Kucoin has strict rate limits but needs frequent pings
            ws_config['pingInterval'] = 5000
            ws_config['options']['heartbeatDelay'] = 5000

        # Add API keys if available for private streams (though we mostly use public here)
        if self.exchange_id == 'krakenfutures':
            if config.KRAKEN_FUTURES_API_KEY:
                ws_config['apiKey'] = config.KRAKEN_FUTURES_API_KEY
                ws_config['secret'] = config.KRAKEN_FUTURES_PRIVATE_KEY

        self._ws_exchange = getattr(ccxtpro, self.exchange_id)(ws_config)

        print(f"[{self.name}] 🛡️ WebSocket configured with AGGRESSIVE keepalive ({ws_config['pingInterval']}ms ping, {ws_config['pingTimeout']}ms timeout)")

        try:
            tasks = [
                self._watch_tickers_loop(),
                self._watch_trades_loop()
            ]
            await asyncio.gather(*tasks)
        finally:
            await self._ws_exchange.close()

    async def _watch_tickers_loop(self):
        """Background loop to update ticker cache via WS with health monitoring."""
        while True:
            try:
                # === AEGIS QUANTSEC: Check if fallback to REST needed ===
                if self._ws_fallback_to_rest:
                    await asyncio.sleep(5)
                    # Try to recover WS after cooldown
                    if self._ws_health_monitor:
                        all_healthy = all(
                            self._ws_health_monitor.get_health_status('tickers', sym).status == 'HEALTHY'
                            for sym in self._ws_symbols[:5]  # Check first 5
                        )
                        if all_healthy:
                            print(f"[{self.name}] ✅ WebSocket health recovered, switching from REST fallback")
                            self._ws_fallback_to_rest = False
                    continue

                # CCXT.pro unified watch_tickers
                # If we have a lot of symbols, some exchanges prefer a list
                tickers = await self._ws_exchange.watch_tickers(self._ws_symbols)

                # Update synchronous cache
                self._ticker_cache.update(tickers)
                self._last_ticker_fetch = time.time()

                # === AEGIS QUANTSEC: Record successful message for health tracking ===
                if self._ws_health_monitor:
                    now_ms = time.time() * 1000.0
                    for symbol in self._ws_symbols:
                        if symbol in tickers:
                            # Calculate proper latency from CCXT ticker timestamp (in ms)
                            ticker_ts = tickers[symbol].get('timestamp')
                            latency = 0.0
                            if ticker_ts:
                                latency = now_ms - float(ticker_ts)
                            self._ws_health_monitor.record_message('tickers', symbol, latency_ms=max(0.0, latency))

            except Exception as e:
                error_msg = str(e)

                # === AEGIS QUANTSEC: Record errors for health tracking ===
                if self._ws_health_monitor:
                    for symbol in self._ws_symbols:
                        self._ws_health_monitor.record_error('tickers', symbol, error_type=str(type(e).__name__))

                # FIX 2026-03-14: Suppress harmless "Already subscribed" errors from Kraken Futures
                # These occur when WS tries to re-subscribe to already-active feeds
                if 'Already subscribed' in error_msg or 're-requesting' in error_msg:
                    # Silent skip - this is normal behavior for Kraken Futures
                    pass
                elif 'timed out' in error_msg.lower() or 'timeout' in error_msg.lower():
                    # === AEGIS: Timeout detected - check health ===
                    print(f"[{self.name}] ⚠️ WebSocket timeout detected")

                    if self._ws_health_monitor:
                        # Check if we should fallback to REST
                        unhealthy_count = sum(
                            1 for sym in self._ws_symbols
                            if self._ws_health_monitor.get_health_status('tickers', sym).status in ('UNHEALTHY', 'CRITICAL')
                        )

                        if unhealthy_count > len(self._ws_symbols) * self._ws_unhealthy_threshold:  # Configurable threshold
                            print(f"[{self.name}] 🚨 Fallback to REST API (>{unhealthy_count} unhealthy connections)")
                            self._ws_fallback_to_rest = True

                            # Fetch via REST as fallback
                            await self._fetch_tickers_fallback()

                    await asyncio.sleep(5)
                else:
                    # Log other errors
                    print(f"[{self.name}] WS Ticker Loop Error: {e}")

                await asyncio.sleep(5)

    async def _fetch_tickers_fallback(self):
        """
        AEGIS QUANTSEC: REST API fallback during WebSocket outages.

        Fetches tickers via REST when WebSocket is unhealthy.
        Implements rate limiting and cooldown to prevent API spam.
        """
        now = time.time()

        # Check cooldown
        if now - self._ws_last_rest_fetch < self._ws_rest_cooldown:
            return

        self._ws_last_rest_fetch = now

        try:
            print(f"[{self.name}] 🔄 Fetching tickers via REST fallback...")

            # Use resilient fetch with retry
            tickers = self._fetch_tickers_resilient(self._ws_symbols[:20])  # Cap at 20 for rate limit

            if tickers:
                self._ticker_cache.update(tickers)
                self._last_ticker_fetch = time.time()
                print(f"[{self.name}] ✅ REST fallback successful: {len(tickers)} tickers fetched")

                # Record successful messages for health recovery tracking
                if self._ws_health_monitor:
                    for symbol in tickers:
                        self._ws_health_monitor.record_message('tickers', symbol, latency_ms=0.0)
            else:
                print(f"[{self.name}] ⚠️ REST fallback returned no data")

        except Exception as e:
            print(f"[{self.name}] ❌ REST fallback failed: {e}")

    def _on_ws_unhealthy(self, channel: str, symbol: str, status):
        """AEGIS: Callback for unhealthy WebSocket connections."""
        print(f"[{self.name}] 🚨 UNHEALTHY: {channel}/{symbol} (score: {status.health_score:.2f})")
        print(f"   Issues: {', '.join(status.issues)}")

        if status.recommendations:
            print(f"   Recommendations: {', '.join(status.recommendations)}")

    def _on_ws_recovered(self, channel: str, symbol: str, status):
        """AEGIS: Callback for recovered WebSocket connections."""
        print(f"[{self.name}] ✅ RECOVERED: {channel}/{symbol} (score: {status.health_score:.2f})")

        # Disable REST fallback if all connections recovered
        if self._ws_fallback_to_rest and self._ws_health_monitor:
            healthy_count = sum(
                1 for sym in self._ws_symbols
                if self._ws_health_monitor.get_health_status('tickers', sym).status == 'HEALTHY'
            )
            if healthy_count >= len(self._ws_symbols) * 0.8:  # 80% healthy
                self._ws_fallback_to_rest = False
                print(f"[{self.name}] ✅ WebSocket health restored, disabled REST fallback")

    def update_ws_symbols(self, symbols: List[str]):
        """
        Dynamically updates the symbols being watched by WS.

        FIX 2026-03-14: Added debouncing to prevent Kraken Futures subscription spam.
        - Only updates every 5 minutes (prevents 'Already subscribed' errors)
        - Only updates if symbol list actually changed
        """
        if not self._ws_enabled or not symbols:
            return

        # === DEBOUNCING CHECK ===
        now = time.time()
        time_since_update = now - self._ws_last_update

        if time_since_update < self._WS_UPDATE_MIN_INTERVAL:
            # Too soon - skip update (prevents subscription spam)
            return

        new_list = symbols
        if self.exchange_id == 'krakenfutures':
            new_list = [config.KRAKEN_SYMBOL_MAP.get(s, s) for s in symbols]

        # Filter duplicates and check if changed
        new_set = set(new_list)
        old_set = set(self._ws_symbols)

        # Only update if symbols actually changed
        if new_set != old_set:
            self._ws_symbols = list(new_set)
            self._ws_last_update = now  # Update timestamp
            print(f"[{self.name}] 🔄 WebSocket Subscriptions Updated: {len(self._ws_symbols)} assets.")
            # CCXT.pro handles the new symbols on the next watch_tickers call usually,
            # but some exchanges might need a reconnect or specific logic.
            # For Kraken Futures, watch_tickers(symbols) works well.
        else:
            # Symbols unchanged - no update needed (silent skip)
            pass

    async def _watch_trades_loop(self):
        """Background loop to update trade log via WS with health monitoring."""
        while True:
            try:
                # Optimized approach for multiple symbols in CCXT.pro:
                # Some exchanges support watchTrades(None) for all. Kraken Futures might not.
                # Let's just watch the symbols in a loop or concurrently.

                async def watch_single(symbol):
                    while True:
                        try:
                            trades = await self._ws_exchange.watch_trades(symbol)
                            if symbol not in self._ws_trades_log:
                                self._ws_trades_log[symbol] = []
                            # Keep only last 100 trades for OMI/Physics
                            self._ws_trades_log[symbol].extend(trades)
                            if len(self._ws_trades_log[symbol]) > 100:
                                self._ws_trades_log[symbol] = self._ws_trades_log[symbol][-100:]

                            # === AEGIS QUANTSEC: Record trade message for health ===
                            if self._ws_health_monitor:
                                latency = 0.0
                                if trades and len(trades) > 0 and trades[-1].get('timestamp'):
                                    latency = (time.time() * 1000.0) - float(trades[-1]['timestamp'])
                                self._ws_health_monitor.record_message('trades', symbol, latency_ms=max(0.0, latency))

                        except Exception as e:
                            # === AEGIS: Record error for health tracking ===
                            if self._ws_health_monitor:
                                self._ws_health_monitor.record_error('trades', symbol, error_type=str(type(e).__name__))
                            await asyncio.sleep(1)

                sub_tasks = [watch_single(s) for s in self._ws_symbols[:10]] # Cap at 10 for safety
                await asyncio.gather(*sub_tasks)

            except Exception as e:
                # === AEGIS: Record error for health tracking ===
                if self._ws_health_monitor:
                    for symbol in self._ws_symbols[:10]:
                        self._ws_health_monitor.record_error('trades', symbol, error_type=str(type(e).__name__))

                # print(f"[{self.name}] WS Trade Loop Error: {e}")
                await asyncio.sleep(5)
                break

    # ========================================================================
    # AEGIS QUANTSEC: WebSocket Health Monitoring Methods
    # ========================================================================

    def get_ws_health_status(self, symbol: str = None) -> Dict[str, Any]:
        """
        Get WebSocket health status for dashboard monitoring.

        Args:
            symbol: Optional specific symbol to check. If None, returns all.

        Returns:
            Dictionary with health status information
        """
        if not self._ws_health_monitor:
            return {
                'enabled': False,
                'status': 'NOT_INITIALIZED',
                'message': 'WebSocket health monitor not initialized'
            }

        if symbol:
            status = self._ws_health_monitor.get_health_status('tickers', symbol)
            return {
                'enabled': True,
                'symbol': symbol,
                'status': status.status,
                'health_score': status.health_score,
                'issues': status.issues,
                'recommendations': status.recommendations
            }
        else:
            report = self._ws_health_monitor.get_summary_report()
            return {
                'enabled': True,
                'exchange': self.exchange_id,
                'total_connections': report['total_connections'],
                'healthy': report['healthy'],
                'degraded': report['degraded'],
                'unhealthy': report['unhealthy'],
                'fallback_active': self._ws_fallback_to_rest,
                'connections': report['connections']
            }

    def is_ws_healthy(self, symbol: str = None) -> bool:
        """
        Check if WebSocket connection is healthy.

        Args:
            symbol: Optional specific symbol. If None, checks all.

        Returns:
            True if healthy, False otherwise
        """
        if not self._ws_health_monitor:
            return True  # Assume healthy if monitoring disabled

        if symbol:
            status = self._ws_health_monitor.get_health_status('tickers', symbol)
            return status.status == 'HEALTHY'
        else:
            # Check if any connection is unhealthy
            statuses = self._ws_health_monitor.get_all_health_statuses()
            return all(s.status == 'HEALTHY' for s in statuses.values())

    # time imported at module level

    @with_circuit_breaker("observer_fetch_tickers", failure_threshold=5, recovery_timeout=60.0, fallback_value={})
    @with_retry(max_retries=3, base_delay=1.0, max_delay=10.0, exceptions=(ccxt.NetworkError, ccxt.ExchangeError))
    def _fetch_tickers_resilient(self, symbols: List[str]) -> Dict[str, Any]:
        if self.exchange.has['fetchTickers']:
            return self.exchange.fetch_tickers(symbols)
        else:
            print(f"[{self.name}] ⚠️ Exchange does not support fetchTickers!")
            return {}

    def fetch_tickers_batch(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Optimized Scout Fetch: Gets 24hr stats for MULTIPLE symbols in ONE API call.
        Implements TTL Cache to prevent rate limit bans.
        
        FIX 2026-02-24: Sanitize symbols for exchange compatibility.
        - KuCoin: USDT pairs only (no xStocks, no PF_*)
        - Kraken Futures: USD:USD pairs + xStocks
        """
        now = time.time()
        
        # === SYMBOL SANITIZATION ===
        # Filter symbols to only those available on this exchange
        if self.exchange_id == 'kucoin':
            # KuCoin doesn't have xStocks or PF_* symbols
            # Filter to USDT pairs only, and exclude xStocks
            sanitized = []
            for s in symbols:
                # Skip PF_ prefix symbols (Kraken Futures only)
                if s.startswith('PF_'):
                    continue
                # Skip xStocks (any variant: SPYX/USDT, SPYX/USD, etc.)
                if self._is_xstock_symbol(s):
                    continue
                # Keep only USDT pairs
                if '/USDT' in s:
                    sanitized.append(s)
            # Ensure USDT format (convert any /USD:USD to /USDT)
            sanitized = [s.replace('/USD:USD', '/USDT') for s in sanitized]
        elif self.exchange_id == 'krakenfutures':
            # Kraken Futures needs USD:USD format
            # Map internal symbols to Kraken Futures format
            sanitized = [config.KRAKEN_SYMBOL_MAP.get(s, s) for s in symbols]
            # Filter out any that didn't map properly
            sanitized = [s for s in sanitized if s and 'USD:USD' in s]
        elif self.exchange_id == 'kraken':
            # FIX 2026-03-02: Kraken Spot doesn't have xStocks
            # Just pass symbols through (they should be in Spot format already)
            sanitized = symbols
        else:
            sanitized = symbols
        # ===========================
        
        # 1. Check Cache
        if self._ticker_cache and (now - self._last_ticker_fetch < config.SCOUT_CACHE_TTL):
            return self._ticker_cache

        # 2. Fetch Live with Resilience
        if sanitized:
            tickers = self._fetch_tickers_resilient(sanitized)
        else:
            tickers = {}
            print(f"[{self.name}] ⚠️ No valid symbols for {self.exchange_id} (filtered from {len(symbols)})")

        # 3. Update Cache
        if tickers:
            self._ticker_cache = tickers
            self._last_ticker_fetch = now

        return tickers

    def fetch_ticker(self, symbol: str) -> dict:
        """
        Fetches the current ticker for a given symbol.
        Used by ArbitrageHolon for real-time price discovery.
        """
        # FIX 2026-03-02: KuCoin should not fetch xStock tickers
        if self.exchange_id == 'kucoin' and self._is_xstock_symbol(symbol):
            return {}

        target_symbol = symbol
        if self.exchange_id == 'krakenfutures':
            target_symbol = config.KRAKEN_SYMBOL_MAP.get(symbol, symbol)
        # FIX 2026-03-02: Removed XSTOCKS_SPOT_MAP - Kraken Spot doesn't have xStocks
            
        # 1. Check Cache
        if target_symbol in self._ticker_cache:
            return self._ticker_cache[target_symbol]
            
        # 2. Fetch Live
        try:
            ticker = self.exchange.fetch_ticker(target_symbol)
            # Update Cache
            self._ticker_cache[target_symbol] = ticker
            self._last_ticker_fetch = time.time()
            return ticker
        except Exception as e:
            # print(f"[{self.name}] ⚠️ Ticker Fetch Failed for {target_symbol}: {e}")
            return {}

    def _get_local_filename(self, symbol: str, timeframe: str) -> str:
        """Constructs the standard filename for local data."""
        # Sanitize symbol
        safe_symbol = str(symbol).replace('/', '').replace(':', '')
        filename = f"{safe_symbol}_{timeframe}.csv"
        return os.path.join(self.data_dir, filename)

    def load_local_history(self, symbol: str, timeframe: str = '1h') -> pd.DataFrame:
        """Load historical data from market_data directory (Cached)."""
        cache_key = f"{symbol}_{timeframe}"
        
        # 1. Check Class-Level Shared Cache (Process Memory)
        with self._shared_cache_lock:
             if cache_key in self._shared_cache:
                 # Check simple expiry? For local history, we assume it's static for the session unless explicitly reloaded.
                 # Optimization: Return copy? No, read-only is fine for speed.
                 return self._shared_cache[cache_key].copy() # Copy to prevent mutation issues downstream

        filepath = self._get_local_filename(symbol, timeframe)
        
        if not os.path.exists(filepath):
            # print(f"[{self.name}] No local history for {symbol} at {filepath}") # Reduce noise
            return pd.DataFrame()

        # === OPTIMIZATION: Pickle Cache ===
        pickle_path = filepath.replace('.csv', '.pkl')
        use_pickle = False
        
        if os.path.exists(pickle_path):
            csv_mtime = os.path.getmtime(filepath)
            pkl_mtime = os.path.getmtime(pickle_path)
            if pkl_mtime >= csv_mtime:
                use_pickle = True
        
        if use_pickle:
            try:
                # Fast Path
                # print(f"[{self.name}] ⚡ Loading cached history for {symbol}") 
                df = pd.read_pickle(pickle_path)
            except Exception:
                # Fallback if pickle corrupt
                print(f"[{self.name}] ⚠️ Pickle corrupt, falling back to CSV for {symbol}")
                df = self.load_data_from_csv(filepath)
        else:
            # Slow Path
            print(f"[{self.name}] Loading local history for {symbol} from {filepath} (DISK READ)")
            df = self.load_data_from_csv(filepath)
            # Save Pickle for next time
            if not df.empty:
                try:
                    df.to_pickle(pickle_path)
                    print(f"[{self.name}] 💾 Cached {symbol} to Pickle.")
                except Exception as e:
                    print(f"[{self.name}] ⚠️ Failed to save pickle: {e}")
        
        # 2. Populate Shared Cache
        with self._shared_cache_lock:
             self._shared_cache[cache_key] = df
        
        return df

    @with_circuit_breaker("observer_fetch_ohlcv", failure_threshold=5, recovery_timeout=60.0, fallback_value=[])
    @with_retry(max_retries=3, base_delay=1.0, max_delay=10.0, exceptions=(ccxt.NetworkError, ccxt.ExchangeError))
    def _fetch_ohlcv_resilient(self, target_symbol: str, target_timeframe: str, since: Optional[int], limit: int):
        if since is not None:
            return self.exchange.fetch_ohlcv(target_symbol, target_timeframe, since=since, limit=limit)
        return self.exchange.fetch_ohlcv(target_symbol, target_timeframe, limit=limit)

    def fetch_market_data(self, timeframe: str = None, limit: int = 500, symbol: str = None) -> pd.DataFrame:
        """
        Fetches Hybrid Market Data: Local History + Smart WS Cache + CCXT Live Sync.
        Uses WS Ticker Feed to synthesize the live candle, bypassing REST latency.
        """
        target_timeframe = timeframe if timeframe else config.TIMEFRAME
        target_symbol = symbol if symbol else self.symbol

        # PATCH: Apply Kraken Symbol Map for Futures
        if config.TRADING_MODE in ['FUTURES', 'DUAL'] and 'kucoin' not in self.exchange.id:
             if self.exchange_id == 'krakenfutures':
                 target_symbol = config.KRAKEN_SYMBOL_MAP.get(target_symbol, target_symbol)
             # FIX 2026-03-02: Removed XSTOCKS_SPOT_MAP - Kraken Spot doesn't have xStocks
        
        # --- PHASE 1: SMART OHLCV CACHE (ZERO LATENCY) ---
        cache_key = f"{target_symbol}_{target_timeframe}"
        now_ts = time.time()
        
        # Determine the current candle boundary (e.g., 15m = 900 seconds)
        tf_mins = 15
        if 'm' in target_timeframe: tf_mins = int(target_timeframe.replace('m', ''))
        elif 'h' in target_timeframe: tf_mins = int(target_timeframe.replace('h', '')) * 60
        tf_secs = tf_mins * 60
        
        current_candle_open = (int(now_ts) // tf_secs) * tf_secs
        
        has_valid_cache = False
        df_cached = pd.DataFrame()
        
        if cache_key in self._smart_ohlcv_cache:
            entry = self._smart_ohlcv_cache[cache_key]
            # Must be from the current candle epoch, AND we must actively have WS connection
            if entry['candle_open_time'] == current_candle_open and getattr(config, 'USE_WEBSOCKETS', False):
                has_valid_cache = True
                df_cached = entry['df'].copy()
        
        if has_valid_cache and not df_cached.empty:
            # FAST PATH: 0ms execution
            # Synthesize the live active candle using the WS Ticker Cache
            live_price = self.get_latest_price(target_symbol)
            if live_price > 0:
                # Update the very last row (the active, unclosed candle)
                last_idx = df_cached.index[-1]
                df_cached.at[last_idx, 'close'] = live_price
                if live_price > df_cached.at[last_idx, 'high']:
                    df_cached.at[last_idx, 'high'] = live_price
                if live_price < df_cached.at[last_idx, 'low']:
                    df_cached.at[last_idx, 'low'] = live_price
                
                # Update returns quickly
                df_cached['returns'] = np.log(df_cached['close'] / df_cached['close'].shift(1))
                return df_cached.dropna()

        # === SLOW PATH: REST API FETCH (ONLY HAPPENS ONCE PER CANDLE EPOCH) ===
        # 1. Load Local History
        df_local = self.load_local_history(target_symbol, target_timeframe)
        
        # 2. Fetch Live Sync (CCXT)
        if not self.exchange.has['fetchOHLCV']:
             return df_local

        df_live = pd.DataFrame()
        last_ts = None
        fetch_limit = limit
        
        if not df_local.empty:
            last_ts = int(df_local['timestamp'].iloc[-1].timestamp() * 1000)
            current_ts = int(time.time() * 1000)
            
            gap_ms = current_ts - last_ts
            gap_hours = gap_ms / (1000 * 3600)
            
            if gap_ms < 0: return df_local

            fetch_limit = 5 if gap_hours < 5 else 1000
            
        try:
            ohlcv_live = self._fetch_ohlcv_resilient(target_symbol, target_timeframe, since=last_ts, limit=fetch_limit)
            
            if ohlcv_live:
                df_temp = pd.DataFrame(ohlcv_live, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'], unit='ms')
                df_live = df_temp
        except Exception as e:
            # print(f"[{self.name}] Sync failed for {target_symbol}: {e}")
            pass
        
        try:
            if not df_live.empty:
                if not df_local.empty:
                    df = pd.concat([df_local, df_live]).drop_duplicates(subset='timestamp').reset_index(drop=True)
                else:
                    df = df_live
            else:
                 df = df_local # Fallback
        except Exception as e:
            # print(f"[{self.name}] Data Merge error for {target_symbol}: {e}")
            df = df_local 

        # 3. Cache it back into Memory
        if not df.empty:
            self._smart_ohlcv_cache[cache_key] = {
                'df': df.copy(),
                'last_fetch': now_ts,
                'candle_open_time': current_candle_open
            }
            
            df['returns'] = np.log(df['close'] / df['close'].shift(1))
            df.dropna(inplace=True)
        
        return df

    def fetch_market_data_batch(self, symbols: List[str], timeframe: str = None, limit: int = 100) -> Dict[str, pd.DataFrame]:
        """
        Parallelized Batch Fetch for OHLCV data.
        Distributes requests across threads to minimize latency.
        """
        target_timeframe = timeframe if timeframe else config.TIMEFRAME
        results = {}
        
        # Use Shared Executor
        future_to_symbol = {}
        try:
            for symbol in symbols:
                # Optimized submissions: Reduced from 0.2s to 0.05s to maximize pool utilization
                time.sleep(0.05) 
                future = self.executor.submit(self.fetch_market_data, target_timeframe, limit, symbol)
                future_to_symbol[future] = symbol
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if not df.empty:
                        results[symbol] = df
                except Exception as e:
                    print(f"[{self.name}] ⚠️ Batch Fetch Failed for {symbol}: {e}")
        except Exception as e:
            print(f"[{self.name}] ❌ Executor Error in Validating Batch: {e}")
                    
        return results

    def fetch_matrix_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Unified High-Speed Data Sink:
        Fetches 15m OHLCV, 1h OHLCV, Books, and Funding in a single parallel burst.
        Eliminates redundant loop overhead in the Trader cycles.
        """
        results = {}
        target_15m = config.TIMEFRAME # usually 15m
        target_1h = '1h'

        def fetch_asset_unit(symbol):
            try:
                # 1. Fetch 15m (Entry/Risk)
                df_15m = self.fetch_market_data(timeframe=target_15m, limit=100, symbol=symbol)
                # 2. Fetch 1h (Regime/Structure)
                df_1h = self.fetch_market_data(timeframe=target_1h, limit=100, symbol=symbol)
                # 3. Order Book
                book = self.fetch_order_book(symbol, limit=20)
                # 4. Funding & Open Interest (Futures Only)
                funding, oi = self.fetch_funding_and_oi(symbol)
                
                return symbol, {
                    'df_15m': df_15m,
                    'df_1h': df_1h,
                    'book': book,
                    'funding': funding,
                    'oi': oi
                }
            except Exception as e:
                print(f"[{self.name}] ⚠️ Matrix Unit Fetch Failed for {symbol}: {e}")
                return symbol, None

        # Use Shared Executor
        try:
            futures = {self.executor.submit(fetch_asset_unit, s): s for s in symbols}
            for future in as_completed(futures):
                sym, data = future.result()
                if data:
                    results[sym] = data
        except Exception as e:
            print(f"[{self.name}] ❌ Matrix Executor Error: {e}")
        
        return results

    def get_latest_price(self, symbol: str = None) -> float:
        """
        Returns the current market price (last close).
        Prioritizes WebSocket Cache for Warp Velocity.
        
        FIX 2: Circuit Breaker - Zero-price detection and deviation alerts.
        """
        target_symbol = symbol if symbol else self.symbol
        
        # 1. Map for Kraken Futures
        if self.exchange_id == 'krakenfutures':
            target_symbol = config.KRAKEN_SYMBOL_MAP.get(target_symbol, target_symbol)
        # FIX 2026-03-02: Removed XSTOCKS_SPOT_MAP - Kraken Spot doesn't have xStocks
        elif self.exchange_id == 'kucoin':
            # FIX 2026-03-02: KuCoin should not fetch xStock prices
            if self._is_xstock_symbol(target_symbol):
                return 0.0
        
        price = 0.0

        # 2. Check WS Cache
        if target_symbol in self._ticker_cache:
            ticker = self._ticker_cache[target_symbol]
            if ticker:
                # FIX: Support multiple price sources for futures/xStocks
                price = float(ticker.get('last', 0) or 
                             ticker.get('markPrice', 0) or
                             ticker.get('bid', 0) or
                             ticker.get('ask', 0) or 0.0)

        # 3. Fallback to REST if WS failed
        if price == 0.0:
            for attempt in range(3):
                try:
                    ticker = self.exchange.fetch_ticker(target_symbol)
                    # FIX: Support multiple price sources for futures/xStocks
                    price = float(ticker.get('last', 0) or 
                                 ticker.get('markPrice', 0) or
                                 ticker.get('bid', 0) or
                                 ticker.get('ask', 0) or 0.0)
                    if price > 0:
                        break
                except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                    if attempt == 2:
                        print(f"[{self.name}] ⚠️ Price Fetch Error {target_symbol}: {e}")
                    time.sleep(1 * (attempt + 1))
        
        # --- FIX 2: CIRCUIT BREAKER ---
        # Maintain last valid price cache
        if not hasattr(self, '_last_valid_prices'):
            self._last_valid_prices = {}
        
        last_valid = self._last_valid_prices.get(target_symbol, 0.0)
        
        # A. Zero-Price Detection
        if price == 0.0 or price is None:
            if last_valid > 0:
                print(f"[{self.name}] 🚨 CIRCUIT BREAKER: {target_symbol} returned $0.00! Using last valid ${self.normalize_price(last_valid)}")
                return last_valid
            else:
                print(f"[{self.name}] 🚨 CIRCUIT BREAKER: {target_symbol} has NO valid price data!")
                return 0.0
        
        # B. Deviation Alert (>50% spike/drop)
        if last_valid > 0:
            deviation = abs(price - last_valid) / last_valid
            if deviation > 0.50:
                print(f"[{self.name}] ⚠️ PRICE DEVIATION ALERT: {target_symbol} moved {deviation:.0%}! (${self.normalize_price(last_valid)} -> ${self.normalize_price(price)})")
        
        # C. Update valid price cache
        self._last_valid_prices[target_symbol] = price
        # -----------------------------
        
        return price


    @staticmethod
    def normalize_price(price: float) -> str:
        """
        Smart formatter for sub-penny assets.
        0.00001234 -> "0.00001234"
        10.50 -> "10.50"
        """
        if price < 0.01:
            return f"{price:.8f}".rstrip('0').rstrip('.')
        elif price < 1.0:
            return f"{price:.4f}"
        else:
            return f"{price:.2f}"

    def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        """
        Fetch current order book depth (Bids/Asks).
        Returns {'bids': [[price, qty], ...], 'asks': [[price, qty], ...]}
        """
        for attempt in range(3):
            try:
                # Map symbol if needed
                req_symbol = symbol
                # FIX: Check if we are incorrectly mapping Kraken Futures symbols for KuCoin
                if config.TRADING_MODE == 'FUTURES':
                    # If this observer is KuCoin, we CANNOT use Kraken symbols (e.g. BTC/USD:USD)
                    if 'kucoin' in self.exchange.id:
                        # KuCoin Futures often uses XBTUSDTM or similar, but for Spot/Simulated checking
                        # we likely want the standard symbol 'BTC/USDT' or 'BTC-USDT'
                        # Assuming 'symbol' passed in IS 'BTC/USDT' (internal format).
                        # We just rely on CCXT's unified symbol handling for KuCoin which usually works with 'BTC/USDT'.
                        req_symbol = symbol 
                    else:
                        # If this IS Kraken (e.g. Executor referencing it), use the map.
                        req_symbol = config.KRAKEN_SYMBOL_MAP.get(symbol, symbol)
                    
                book = self.exchange.fetch_order_book(req_symbol, limit)
                return {
                    'bids': book['bids'],
                    'asks': book['asks'],
                    'timestamp': book['timestamp']
                }
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                if attempt == 2:
                    print(f"[{self.name}] ⚠️ OrderBook Fetch Fail {symbol}: {e}")
                    return {'bids': [], 'asks': []}
                time.sleep(1 * (attempt + 1))
        return {'bids': [], 'asks': []}

    def fetch_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        """
        Fetch recent executions (Tick Data) for Order Flow Analysis.
        Includes a 15s TTL Cache to protect API limits.
        Returns: [{'price': float, 'amount': float, 'side': 'buy'/'sell', 'timestamp': int}, ...]
        """
        # 1. Check Cache
        now = time.time()
        if not hasattr(self, '_trades_cache'): self._trades_cache = {}
        
        cache_key = f"{symbol}_{limit}"
        if cache_key in self._trades_cache:
            entry = self._trades_cache[cache_key]
            if now - entry['ts'] < 15.0: # 15s TTL
                return entry['data']
                
        # 2. Fetch Live
        for attempt in range(3):
            try:
                # Map symbol if needed
                req_symbol = symbol
                if config.TRADING_MODE == 'FUTURES' and 'kucoin' not in self.exchange.id:
                     req_symbol = config.KRAKEN_SYMBOL_MAP.get(symbol, symbol)
                
                if self.exchange.has['fetchTrades']:
                    trades = self.exchange.fetch_trades(req_symbol, limit=limit)
                    # Cache result
                    self._trades_cache[cache_key] = {'data': trades, 'ts': now}
                    return trades
                else:
                    return []
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                if attempt == 2:
                    print(f"[{self.name}] ⚠️ Trade Fetch Fail {symbol}: {e}")
                    return []
                time.sleep(1 * (attempt + 1))
        return []

    def fetch_funding_and_oi(self, symbol: str) -> tuple[float, float]:
        """
        Fetch Current Funding Rate and Open Interest for a symbol.
        Used for Liquidity Stress and Funding Reversion modeling.
        Returns: (Funding Rate Decimal, Open Interest Notional)
        """
        if config.TRADING_MODE != 'FUTURES':
            return 0.0, 0.0

        # Simple Cache (Funding rates strictly change every 1-4-8h depending on exchange)
        # We can cache for ~15 mins safely.
        cache_key = f"funding_oi_{symbol}"
        now = time.time()
        
        # Initialize specialized cache if missing
        if not hasattr(self, '_funding_cache'):
             self._funding_cache = {}
        
        if cache_key in self._funding_cache:
             entry = self._funding_cache[cache_key]
             if now - entry['ts'] < 900: # 15 min TTL
                 return entry['rate'], entry.get('oi', 0.0)

        for attempt in range(3):
            try:
                # CCXT Unified
                # Check if exchange supports it
                if self.exchange.has['fetchTicker']:
                     exec_symbol = config.KRAKEN_SYMBOL_MAP.get(symbol, symbol)
                     
                     # FIX 2026-01-28: Use Ticker 'info' for valid Funding Rate
                     ticker = self.exchange.fetch_ticker(exec_symbol)
                     
                     rate = 0.0
                     oi = 0.0
                     
                     # Extract OI
                     if 'info' in ticker and 'openInterest' in ticker['info']:
                         oi = float(ticker['info']['openInterest'] or 0.0)
                         
                     # Extract Rate
                     if 'info' in ticker and 'fundingRate' in ticker['info']:
                         rate = float(ticker['info']['fundingRate'] or 0.0)
                     elif 'info' in ticker and 'lastFundingRate' in ticker['info']:
                          rate = float(ticker['info']['lastFundingRate'] or 0.0)
                          
                     if rate != 0.0 or oi != 0.0:
                          self._funding_cache[cache_key] = {'rate': rate, 'oi': oi, 'ts': now}
                          return rate, oi
                     
                     # Fallback
                     return 0.0, 0.0
                else:
                     return 0.0, 0.0
                     
            except Exception as e:
                # print(f"[{self.name}] ⚠️ Funding/OI Fetch Fail: {e}")
                time.sleep(1)
        
        return 0.0, 0.0

    def receive_message(self, sender: Any, content: Any) -> Any:
        """
        Handle incoming messages for health checks or data requests.
        """
        # Unwrap Holon Message if needed
        if isinstance(content, Message):
            msg_type = content.type
            payload = content.payload
        elif isinstance(content, dict):
             msg_type = content.get('type')
             payload = content
        else:
             return None

        if msg_type == 'GET_STATUS':
            # Report health status
            return {
                'status': 'OK',
                'last_fetch': datetime.now().isoformat(),
                'primary_symbol': self.symbol
            }
            
        elif msg_type == 'FORCE_FETCH':
            symbol = payload.get('symbol') if isinstance(payload, dict) else None
            print(f"[{self.name}] Received FORCE_FETCH for {symbol or 'ALL'}")
            return True
            
        return None

    def load_data_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load historical data from a CSV file.
        Expects columns: timestamp, open, high, low, close, volume
        Calculates returns automatically.
        """
        try:
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calculate Log Returns
            df['returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Drop NaNs
            df.dropna(inplace=True)
            
            return df
        except Exception as e:
            print(f"[{self.name}] Error loading CSV: {e}")
            return pd.DataFrame()

    # === VOL-WINDOW HELPERS ===
    def fetch_realized_vol(self, symbol: str, window_hours: int = 24) -> float:
        """
        Calculate realized volatility (annualized) for the given window.
        """
        try:
            # Fetch 15m candles enough to cover window
            limit = int((window_hours * 60) / 15) + 20 # Buffer
            df = self.fetch_market_data(timeframe='15m', limit=limit, symbol=symbol)
            if df.empty or 'returns' not in df.columns:
                return 0.0
            
            # Std Dev of log returns
            std_dev = df['returns'].iloc[-limit:].std()
            
            # Annualize (assuming 15m candles)
            # Crypto trades 24/7/365. 
            # 15m periods per year = 4 * 24 * 365 = 35040
            annualized_vol = std_dev * np.sqrt(35040)
            
            return float(annualized_vol)
        except Exception as e:
            print(f"[{self.name}] ⚠️ Vol Calc Failed for {symbol}: {e}")
            return 0.0

    def fetch_listing_age(self, symbol: str) -> float:
        """
        Estimate listing age in days. 
        Uses first available candle timestamp from exchange or local history.
        """
        try:
            # Try getting earliest candle via CCXT (if supported)
            # or rely on local history start
             # 1. Check Local
            df_local = self.load_local_history(symbol, '1h')
            local_start = df_local['timestamp'].iloc[0] if not df_local.empty else datetime.now()
            
            # 2. If we really need accuracy, we'd query exchange "since 2010" limit 1
            # For now, we return a heuristic or assume older if local history is deep.
            
            # Simple heuristic: If we have > 14 days of local history, it's > 14 days old.
            age = (datetime.now() - local_start).total_seconds() / 86400.0
            return age
        except Exception:
            # Return 999.0 (Old) on failure to avoid falsely triggering "New Listing" logic
            return 999.0

    def shutdown(self):
        """
        Gracefully shut down the Observer, closing threads and connections.
        """
        print(f"[{self.name}] 🛑 Shutting down Observer...")
        
        # 1. Stop WS Loop
        if self._ws_loop and self._ws_loop.is_running():
            self._ws_loop.stop()
        
        # 2. Shutdown Executor (Wait=False to force kill pending if needed, but best to let finish)
        if self.executor:
            self.executor.shutdown(wait=False)
            print(f"[{self.name}] 🛑 Shared Executor Shutdown.")


