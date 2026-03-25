"""
PaxgBtcHolon - PAXG/BTC Macro Arbitrage Engine

Trades the Gold/Bitcoin ratio using PAXG as gold proxy.
Strategy: Mean reversion on 90-day z-score.

Signals:
- Long PAXG/BTC when z-score < -2.0 (BTC undervalued vs gold)
- Short PAXG/BTC when z-score > +2.0 (BTC overvalued vs gold)
"""

import time
import logging
import numpy as np
from collections import deque
from typing import Dict, Optional, Any
from HolonicTrader.holon_core import Holon, Disposition

logger = logging.getLogger("PaxgBtc")

class PaxgBtcHolon(Holon):
    """
    PAXG/BTC Macro Arbitrage Engine
    
    Exploits mean reversion in the gold/bitcoin ratio.
    Uses PAXG (Paxos Gold) as gold price proxy.
    """
    
    def __init__(self, name: str = "PaxgBtc"):
        super().__init__(name=name, disposition=Disposition(autonomy=0.75, integration=0.85))
        
        # Price tracking
        self.paxg_usdt_price = 0.0
        self.btc_usdt_price = 0.0
        self.paxg_btc_ratio = 0.0
        self.last_update_ts = 0.0
        
        # Z-score calculation
        self.ratio_history = deque(maxlen=129600)  # 90 days * 24 hours * 60 minutes
        self.zscore = 0.0
        self.zscore_mean = 0.0
        self.zscore_std = 0.0
        
        # Strategy parameters
        self.zscore_entry_threshold = 2.0    # Enter at ±2.0 std dev
        self.zscore_exit_threshold = 0.5     # Exit at ±0.5 std dev (mean reversion)
        self.lookback_days = 90              # 90-day rolling window
        self.min_data_points = 1000          # Minimum data before trading
        
        # Cooldowns
        self.last_trade_ts = 0.0
        self.trade_cooldown = 3600  # 1 hour between trades (macro strategy)
        
        # Exchange config
        self.exchange = 'kucoin'  # KuCoin has direct PAXG/BTC pair
        
    def fetch_prices(self, observer) -> bool:
        """
        Fetch PAXG/USDT and BTC/USDT prices from observer
        """
        try:
            # Fetch PAXG/USDT
            paxg_data = observer.fetch_market_data('PAXG/USDT', limit=5)
            if paxg_data is None or paxg_data.empty:
                logger.warning(f"[{self.name}] No PAXG/USDT data")
                return False
            
            self.paxg_usdt_price = float(paxg_data['close'].iloc[-1])
            
            # Fetch BTC/USDT
            btc_data = observer.fetch_market_data('BTC/USDT', limit=5)
            if btc_data is None or btc_data.empty:
                logger.warning(f"[{self.name}] No BTC/USDT data")
                return False
            
            self.btc_usdt_price = float(btc_data['close'].iloc[-1])
            
            # Calculate synthetic PAXG/BTC ratio
            if self.btc_usdt_price > 0:
                self.paxg_btc_ratio = self.paxg_usdt_price / self.btc_usdt_price
                self.ratio_history.append(self.paxg_btc_ratio)
                self.last_update_ts = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"[{self.name}] Price fetch failed: {e}")
            return False
    
    def calculate_zscore(self) -> Optional[float]:
        """
        Calculate 90-day rolling z-score of PAXG/BTC ratio
        """
        if len(self.ratio_history) < self.min_data_points:
            return None
        
        arr = np.array(self.ratio_history)
        self.zscore_mean = np.mean(arr)
        self.zscore_std = np.std(arr)
        current = arr[-1]
        
        if self.zscore_std == 0:
            return 0.0
        
        self.zscore = (current - self.zscore_mean) / self.zscore_std
        return self.zscore
    
    def detect_opportunity(self, observer) -> Optional[Dict]:
        """
        Main detection logic for PAXG/BTC mean reversion
        
        Returns signal if:
        1. Z-score > entry_threshold (extreme deviation)
        2. Not in cooldown
        3. Sufficient historical data
        """
        # 1. Fetch latest prices
        if not self.fetch_prices(observer):
            return None
        
        # 2. Calculate z-score
        zscore = self.calculate_zscore()
        
        if zscore is None:
            # Not enough data yet
            return None
        
        # 3. Check cooldown
        if time.time() - self.last_trade_ts < self.trade_cooldown:
            return None
        
        # 4. Check entry conditions
        if abs(zscore) < self.zscore_entry_threshold:
            # Not extreme enough
            return None
        
        # 5. Generate signal
        if zscore < -self.zscore_entry_threshold:
            # Bitcoin undervalued vs gold → LONG PAXG/BTC
            # (Buy PAXG, Short BTC equivalent)
            confidence = min(0.95, 0.5 + abs(zscore) * 0.15)
            
            return {
                'symbol': 'PAXG/BTC',
                'direction': 'BUY',  # Long PAXG/BTC = Long gold, Short bitcoin
                'conviction': confidence,
                'reason': f'PAXG/BTC Z-Score {zscore:.2f} (BTC undervalued vs gold)',
                'metadata': {
                    'strategy': 'MACRO_MEAN_REVERSION',
                    'zscore': zscore,
                    'zscore_mean': self.zscore_mean,
                    'zscore_std': self.zscore_std,
                    'paxg_usdt': self.paxg_usdt_price,
                    'btc_usdt': self.btc_usdt_price,
                    'paxg_btc_ratio': self.paxg_btc_ratio,
                    'target_zscore': self.zscore_exit_threshold,
                    'stop_zscore': -3.0,  # Stop if z-score goes against us
                },
                'entry_ratio': self.paxg_btc_ratio,
                'target_ratio': self.zscore_mean,  # Mean reversion target
                'stop_loss_ratio': self.zscore_mean - (3.0 * self.zscore_std),
            }
            
        elif zscore > self.zscore_entry_threshold:
            # Bitcoin overvalued vs gold → SHORT PAXG/BTC
            # (Sell PAXG, Long BTC equivalent)
            confidence = min(0.95, 0.5 + abs(zscore) * 0.15)
            
            return {
                'symbol': 'PAXG/BTC',
                'direction': 'SELL',  # Short PAXG/BTC = Short gold, Long bitcoin
                'conviction': confidence,
                'reason': f'PAXG/BTC Z-Score {zscore:.2f} (BTC overvalued vs gold)',
                'metadata': {
                    'strategy': 'MACRO_MEAN_REVERSION',
                    'zscore': zscore,
                    'zscore_mean': self.zscore_mean,
                    'zscore_std': self.zscore_std,
                    'paxg_usdt': self.paxg_usdt_price,
                    'btc_usdt': self.btc_usdt_price,
                    'paxg_btc_ratio': self.paxg_btc_ratio,
                    'target_zscore': -self.zscore_exit_threshold,
                    'stop_zscore': 3.0,
                },
                'entry_ratio': self.paxg_btc_ratio,
                'target_ratio': self.zscore_mean,
                'stop_loss_ratio': self.zscore_mean + (3.0 * self.zscore_std),
            }
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Return current statistics for dashboard"""
        return {
            'paxg_usdt': self.paxg_usdt_price,
            'btc_usdt': self.btc_usdt_price,
            'paxg_btc_ratio': self.paxg_btc_ratio,
            'ratio_oz_per_btc': 1.0 / self.paxg_btc_ratio if self.paxg_btc_ratio > 0 else 0,  # oz gold per BTC
            'zscore': self.zscore,
            'zscore_mean': self.zscore_mean,
            'zscore_std': self.zscore_std,
            'data_points': len(self.ratio_history),
            'min_required': self.min_data_points,
            'ready_to_trade': len(self.ratio_history) >= self.min_data_points,
            'last_update': self.last_update_ts,
        }
    
    def receive_message(self, sender: Any, content: Any) -> None:
        """Process price updates"""
        pass
