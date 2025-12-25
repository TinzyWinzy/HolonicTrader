import os
import ccxt
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Literal, Any
from HolonicTrader.holon_core import Holon, Disposition, Message

import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import config

class ObserverHolon(Holon):
    """
    ObserverHolon is responsible for acquiring market data from exchanges
    and processing it for other agents (like the Entropy Agent).
    """

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
        
        # Data Cache
        self._cache = {} # symbol -> DataFrame

    def _get_local_filename(self, symbol: str) -> str:
        """Map symbol to its local CSV filename."""
        # Map exchange symbols to CSV filenames
        # Files: ADAUSDT_1h.csv, BTCUSDT_1h.csv, DOGEUSDT_1h.csv, SUIUSDT_1h.csv, XRPUSDT_1h.csv
        symbol_map = {
            'ADA/USDT': 'ADAUSDT_1h.csv',
            'BTC/USD': 'BTCUSD_1h.csv',
            'BTC/USDT': 'BTCUSD_1h.csv',
            'DOGE/USDT': 'DOGEUSDT_1h.csv',
            'SUI/USDT': 'SUIUSDT_1h.csv',
            'XRP/USDT': 'XRPUSDT_1h.csv',
            'SHIB/USDT': 'SHIBUSDT_1h.csv',
            'LTC/USDT': 'LTCUSDT_1h.csv',
            'LINK/USDT': 'LINKUSDT_1h.csv',
            'ALGO/USDT': 'ALGOUSDT_1h.csv'
        }
        
        filename = symbol_map.get(symbol)
        if not filename:
            # Fallback: try to construct from symbol
            clean_symbol = symbol.replace('/', '').replace(':', '')
            filename = f"{clean_symbol}_1h.csv"
            
        return os.path.join(self.data_dir, filename)

    def load_local_history(self, symbol: str, timeframe: str = '1h') -> pd.DataFrame:
        """Load historical data from market_data directory (Cached)."""
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        filepath = self._get_local_filename(symbol) # Filename logic might need update if we had multiplle files, but currently only 1h.
        # CRITICAL: We only have 1h CSVs locally. We cannot load 15m from disk yet.
        # We will rely on CCXT for 15m if not local. 
        # But for '1h', we use local.
        
        if timeframe != '1h':
             return pd.DataFrame() 

        if not os.path.exists(filepath):
            # print(f"[{self.name}] No local history for {symbol} at {filepath}") # Reduce noise
            return pd.DataFrame()
            
        print(f"[{self.name}] Loading local history for {symbol} from {filepath} (DISK READ)")
        df = self.load_data_from_csv(filepath)
        self._cache[cache_key] = df
        return df

    def fetch_market_data(self, timeframe: str = '1h', limit: int = 500, symbol: str = None) -> pd.DataFrame:
        """
        Fetches Hybrid Market Data: Local History + CCXT Live Sync.
        """
        target_symbol = symbol if symbol else self.symbol
        
        # 1. Load Local History
        df_local = self.load_local_history(target_symbol, timeframe)
        
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
                    
                    # FUTURE PROTECTION:
                    # If local history is in the future (e.g. simulation/backtest), don't sync with live exchange.
                    import time
                    current_ts = int(time.time() * 1000)
                    if last_ts > current_ts + 60000: # 1 minute buffer
                         # print(f"[{self.name}] Local history is in future ({last_ts} > {current_ts}). Skipping live sync.")
                         return df_local

                    # We fetch with a larger limit to bridge gaps, or multiple fetches
                    # For simple robustness: fetch up to 1000 candles since last_ts
                    ohlcv_live = self.exchange.fetch_ohlcv(target_symbol, timeframe, since=last_ts, limit=1000)
                else:
                    ohlcv_live = self.exchange.fetch_ohlcv(target_symbol, timeframe, limit=limit)
                
                if ohlcv_live:
                    df_temp = pd.DataFrame(ohlcv_live, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'], unit='ms')
                    df_live = df_temp
                break # Success
            except Exception as e:
                print(f"[{self.name}] Sync Attempt {attempt+1}/3 failed for {target_symbol}: {e}")
                time.sleep(2) # Backoff
        
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

    def get_latest_price(self, symbol: str = None) -> float:
        """
        Returns the current market price (last close).
        """
        target_symbol = symbol if symbol else self.symbol
        ticker = self.exchange.fetch_ticker(target_symbol)
        return float(ticker['last'])

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
