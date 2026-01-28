import os
import ccxt
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Literal, Any, List, Dict
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

class ObserverHolon(Holon):
    """
    ObserverHolon is responsible for acquiring market data from exchanges
    and processing it for other agents (like the Entropy Agent).
    """

    # Class-Level Shared Cache to prevent redundant Disk I/O across instances
    _shared_cache = {} 
    _shared_cache_lock = threading.Lock() # Ensure thread safety

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
        
        # Determine if we should start WS (only for primary exchange if needed)
        # For now, we allow any ObserverHolon to start WS if symbols are provided
        if config.TRADING_MODE == 'FUTURES' or exchange_id == 'krakenfutures':
            self._ws_enabled = True
            # Start in a separate method to avoid blocking init

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
        """The actual async loop using ccxt.pro."""
        ws_config = {'enableRateLimit': True}
        
        # Add API keys if available for private streams (though we mostly use public here)
        if self.exchange_id == 'krakenfutures':
            if config.KRAKEN_FUTURES_API_KEY:
                ws_config['apiKey'] = config.KRAKEN_FUTURES_API_KEY
                ws_config['secret'] = config.KRAKEN_FUTURES_PRIVATE_KEY
        
        self._ws_exchange = getattr(ccxtpro, self.exchange_id)(ws_config)
        
        try:
            tasks = [
                self._watch_tickers_loop(),
                self._watch_trades_loop()
            ]
            await asyncio.gather(*tasks)
        finally:
            await self._ws_exchange.close()

    async def _watch_tickers_loop(self):
        """Background loop to update ticker cache via WS."""
        while True:
            try:
                # CCXT.pro unified watch_tickers
                # If we have a lot of symbols, some exchanges prefer a list
                tickers = await self._ws_exchange.watch_tickers(self._ws_symbols)
                
                # Update synchronous cache
                self._ticker_cache.update(tickers)
                self._last_ticker_fetch = time.time()
                
            except Exception as e:
                # print(f"[{self.name}] WS Ticker Loop Error: {e}")
                await asyncio.sleep(5)

    def update_ws_symbols(self, symbols: List[str]):
        """Dynamically updates the symbols being watched by WS."""
        if not self._ws_enabled or not symbols:
            return
            
        new_list = symbols
        if self.exchange_id == 'krakenfutures':
            new_list = [config.KRAKEN_SYMBOL_MAP.get(s, s) for s in symbols]
            
        # Filter duplicates and check if changed
        new_set = set(new_list)
        if new_set != set(self._ws_symbols):
            self._ws_symbols = list(new_set)
            print(f"[{self.name}] 🔄 WebSocket Subscriptions Updated: {len(self._ws_symbols)} assets.")
            # CCXT.pro handles the new symbols on the next watch_tickers call usually,
            # but some exchanges might need a reconnect or specific logic.
            # For Kraken Futures, watch_tickers(symbols) works well.

    async def _watch_trades_loop(self):
        """Background loop to update trade log via WS."""
        while True:
            try:
                # watch_trades usually returns one symbol at a time or uses a shared stream
                # In CCXT.pro, watchTrades(symbol) blocks until a trade for THAT symbol arrives.
                # To watch multiple symbols, we might need a separate task per symbol or use an exchange-specific multi-symbol method.
                # For now, let's watch the primary symbol and a few others if the list is small.
                
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
                        except Exception:
                            await asyncio.sleep(1)

                sub_tasks = [watch_single(s) for s in self._ws_symbols[:10]] # Cap at 10 for safety
                await asyncio.gather(*sub_tasks)
                
            except Exception as e:
                # print(f"[{self.name}] WS Trade Loop Error: {e}")
                await asyncio.sleep(5)
                break

    import time

    def fetch_tickers_batch(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Optimized Scout Fetch: Gets 24hr stats for MULTIPLE symbols in ONE API call.
        Implements TTL Cache to prevent rate limit bans.
        """
        now = time.time()
        # 1. Check Cache
        if self._ticker_cache and (now - self._last_ticker_fetch < config.SCOUT_CACHE_TTL):
            return self._ticker_cache

        # 2. Fetch Live
        # 2. Fetch Live with Retry
        for attempt in range(3):
            try:
                if self.exchange.has['fetchTickers']:
                    # Filter strictly for requested symbols to save bandwidth (if exchange supports partial)
                    tickers = self.exchange.fetch_tickers(symbols)
                    
                    # 3. Update Cache
                    self._ticker_cache = tickers
                    self._last_ticker_fetch = now
                    return tickers
                else:
                    print(f"[{self.name}] ⚠️ Exchange does not support fetchTickers!")
                    return {}
            except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                if attempt == 2:
                    print(f"[{self.name}] ⚠️ Ticker Batch Fetch Error after 3 attempts: {e}")
                    return {}
                sleep_time = 2 ** attempt # 1s, 2s, 4s
                time.sleep(sleep_time)
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

    def fetch_market_data(self, timeframe: str = None, limit: int = 500, symbol: str = None) -> pd.DataFrame:
        """
        Fetches Hybrid Market Data: Local History + CCXT Live Sync.
        """
        target_timeframe = timeframe if timeframe else config.TIMEFRAME
        target_symbol = symbol if symbol else self.symbol

        # PATCH: Apply Kraken Symbol Map for Futures/Spot
        if config.TRADING_MODE == 'FUTURES' and 'kucoin' not in self.exchange.id:
             target_symbol = config.KRAKEN_SYMBOL_MAP.get(target_symbol, target_symbol)
        
        # 1. Load Local History
        df_local = self.load_local_history(target_symbol, target_timeframe)
        
        # 2. Fetch Live Sync (CCXT)
        if not self.exchange.has['fetchOHLCV']:
             # If no CCXT support, return local or empty
             return df_local

        df_live = pd.DataFrame()
        
        for attempt in range(3):
            try:
                # If we have local data, we fetch since last timestamp
                if not df_local.empty:
                    last_ts = int(df_local['timestamp'].iloc[-1].timestamp() * 1000)
                    current_ts = int(time.time() * 1000)
                    
                    # Gap Analysis
                    gap_ms = current_ts - last_ts
                    gap_hours = gap_ms / (1000 * 3600)
                    
                    if gap_ms < 0: # Future data check
                         # print(f"[{self.name}] Local history is in future. Skipping live sync.")
                         return df_local

                    # DELTA FETCHING LOGIC
                    # If gap is small (< 5 hours), just fetch the tip.
                    # Otherwise, fetch full history (up to 1000).
                    fetch_limit = 5 if gap_hours < 5 else 1000
                    
                    ohlcv_live = self.exchange.fetch_ohlcv(target_symbol, target_timeframe, since=last_ts, limit=fetch_limit)
                else:
                    # Startup / Fresh: Fetch full history
                    ohlcv_live = self.exchange.fetch_ohlcv(target_symbol, target_timeframe, limit=limit)
                
                if ohlcv_live:
                    df_temp = pd.DataFrame(ohlcv_live, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'], unit='ms')
                    df_live = df_temp
                break # Success

            except (ccxt.RateLimitExceeded, ccxt.DDoSProtection) as e:
                # print(f"[{self.name}] ⏳ RATE LIMIT HIT for {target_symbol}. Cooling down...")
                cool = getattr(config, 'API_RATE_LIMIT_COOL', 10.0)
                time.sleep(random.uniform(cool * 0.5, cool)) # Randomized Backoff
                if attempt == 4: print(f"[{self.name}] ❌ Max Retries (Rate Limit) for {target_symbol}")

            except Exception as e:
                print(f"[{self.name}] Sync Attempt {attempt+1}/5 failed for {target_symbol}: {e}")
                j_min = getattr(config, 'API_RETRY_JITTER_MIN', 1.0)
                j_max = getattr(config, 'API_RETRY_JITTER_MAX', 5.0)
                time.sleep(random.uniform(j_min, j_max)) # Jittered Backoff
        
        try:
            # Combine
            if not df_live.empty:
                if not df_local.empty:
                    df = pd.concat([df_local, df_live]).drop_duplicates(subset='timestamp').reset_index(drop=True)
                else:
                    df = df_live
            else:
                 df = df_local # Fallback
                
        except Exception as e:
            print(f"[{self.name}] Data Merge error for {target_symbol}: {e}")
            df = df_local # Fallback to local only

        # 3. Process Returns
        if not df.empty:
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
        with ThreadPoolExecutor(max_workers=config.CCXT_POOL_SIZE) as executor:
            future_to_symbol = {}
            for symbol in symbols:
                # Optimized submissions: Reduced from 0.2s to 0.05s to maximize pool utilization
                time.sleep(0.05) 
                future = executor.submit(self.fetch_market_data, target_timeframe, limit, symbol)
                future_to_symbol[future] = symbol
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if not df.empty:
                        results[symbol] = df
                except Exception as e:
                    print(f"[{self.name}] ⚠️ Batch Fetch Failed for {symbol}: {e}")
                    
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
                df_1h = self.fetch_market_data(timeframe=target_1h, limit=50, symbol=symbol)
                # 3. Order Book
                book = self.fetch_order_book(symbol, limit=20)
                # 4. Funding (Futures Only)
                funding = self.fetch_funding_rate(symbol)
                
                return symbol, {
                    'df_15m': df_15m,
                    'df_1h': df_1h,
                    'book': book,
                    'funding': funding
                }
            except Exception as e:
                print(f"[{self.name}] ⚠️ Matrix Unit Fetch Failed for {symbol}: {e}")
                return symbol, None

        with ThreadPoolExecutor(max_workers=config.CCXT_POOL_SIZE) as executor:
            futures = {executor.submit(fetch_asset_unit, s): s for s in symbols}
            for future in as_completed(futures):
                sym, data = future.result()
                if data:
                    results[sym] = data
        
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
        
        price = 0.0
        
        # 2. Check WS Cache
        if target_symbol in self._ticker_cache:
            ticker = self._ticker_cache[target_symbol]
            if ticker and 'last' in ticker:
                price = float(ticker['last'])
                
        # 3. Fallback to REST if WS failed
        if price == 0.0:
            for attempt in range(3):
                try:
                    ticker = self.exchange.fetch_ticker(target_symbol)
                    price = float(ticker['last'])
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

    def fetch_funding_rate(self, symbol: str) -> float:
        """
        Fetch Current Funding Rate for a symbol.
        Used for Short Squeeze detection (Negative Funding = Shorts paying Longs).
        Returns the funding rate as a decimal (e.g., 0.0001 = 0.01%).
        """
        if config.TRADING_MODE != 'FUTURES':
            return 0.0

        # Simple Cache (Funding rates strictly change every 1-4-8h depending on exchange)
        # We can cache for ~15 mins safely.
        cache_key = f"funding_{symbol}"
        now = time.time()
        
        # Initialize specialized cache if missing
        if not hasattr(self, '_funding_cache'):
             self._funding_cache = {}
        
        if cache_key in self._funding_cache:
             entry = self._funding_cache[cache_key]
             if now - entry['ts'] < 900: # 15 min TTL
                 return entry['rate']

        for attempt in range(3):
            try:
                # CCXT Unified
                # Check if exchange supports it
                if self.exchange.has['fetchTicker']:
                     exec_symbol = config.KRAKEN_SYMBOL_MAP.get(symbol, symbol)
                     
                     # FIX 2026-01-28: Use Ticker 'info' for valid Funding Rate
                     # Standard fetch_funding_rate returns garbage (-0.25)
                     ticker = self.exchange.fetch_ticker(exec_symbol)
                     
                     # Kraken Futures extracts to 'info' -> 'fundingRate'
                     if 'info' in ticker and 'fundingRate' in ticker['info']:
                         rate = float(ticker['info']['fundingRate'])
                         
                         # Update Cache
                         self._funding_cache[cache_key] = {'rate': rate, 'ts': now}
                         return rate
                     elif 'info' in ticker and 'lastFundingRate' in ticker['info']:
                          rate = float(ticker['info']['lastFundingRate'])
                          self._funding_cache[cache_key] = {'rate': rate, 'ts': now}
                          return rate
                     
                     # Fallback
                     return 0.0
                else:
                     return 0.0
                     
            except Exception as e:
                # print(f"[{self.name}] ⚠️ Funding Rate Fetch Fail: {e}")
                time.sleep(1)
        
        return 0.0

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
            return 0.0 # Safer to assume brand new? Or old? 
                       # Strategy needs < 14 days for meme listing.
                       # If we return 0.0, we might falsely trigger "New Listing".
                       # Let's return 999.0 (Old) on failure to be safe.
            return 999.0

