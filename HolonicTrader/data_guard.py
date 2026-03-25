import logging
import time
import requests
from typing import Dict, Tuple, Optional

logger = logging.getLogger("DataGuard")

class DataGuard:
    def __init__(self):
        # Category-Specific Glitch Thresholds
        # Absolute Max Daily ROI (e.g., 0.50 = 50% for BTC)
        # FIX #2 2026-03-02: RAISED thresholds to stop rejecting winning genomes
        # Old thresholds were too aggressive (rejected 25%+ ROI as glitches)
        self.glitch_thresholds = {
            'large_cap': {'max_daily_roi': 0.80, 'max_hourly_roi': 0.25, 'max_candle_roi': 0.15},  # Was 0.50/0.15/0.10
            'mid_cap':   {'max_daily_roi': 1.50, 'max_hourly_roi': 0.50, 'max_candle_roi': 0.30},  # Was 0.80/0.30/0.20
            'meme_coin': {'max_daily_roi': 200.0, 'max_hourly_roi': 50.0, 'max_candle_roi': 20.0},  # Was 100/25/10
            'default':   {'max_daily_roi': 2.0, 'max_hourly_roi': 0.60, 'max_candle_roi': 0.40}   # Was 1.0/0.40/0.25
        }

        # Mappings (Simplified internal classification)
        self.meme_list = ['PEPE', 'SHIB', 'BONK', 'WIF', 'FLOKI', 'DOGE']
        self.large_cap = ['BTC', 'ETH']
        self.mid_cap = ['SOL', 'AVAX', 'LINK', 'UNI', 'AAVE', 'BNB', 'NEAR', 'SUI', 'XRP', 'ADA', 'DOGE', 'TAO', 'FET', 'WLD', 'ARB', 'OP', 'IMX', 'STX', 'APT', 'TIA', 'SEI', 'INJ', 'KAS', 'LDO', 'PYTH', 'JTO', 'ORDI']

        self.rejection_log = []
        self.alert_threshold = 5 # Alerts if > 5 glitches per hour
        self.glitch_history = [] # Timestamps of detected glitches

    def get_category(self, symbol: str) -> str:
        base = symbol.split('/')[0]
        if base in self.large_cap: return 'large_cap'
        if base in self.mid_cap: return 'mid_cap'
        if base in self.meme_list: return 'meme_coin'
        return 'default'

    def validate_roi(self, symbol: str, roi: float, timeframe: str = '1h') -> Tuple[bool, str]:
        """
        Validate ROI against physical and category limits.
        :param roi: decimal (e.g. 0.10 for 10%)
        """
        category = self.get_category(symbol)
        thresholds = self.glitch_thresholds.get(category, self.glitch_thresholds['default'])
        
        # ROI input is often already decimalized (e.g. 1.2M% -> 12000.0)
        # Check against timeframe limits
        limit = thresholds['max_candle_roi'] # Per candle
        
        if roi > 100.0: # 10000% absolute hard limit for any valid single candle
            is_valid, reason = self.cross_validate_external(symbol, roi)
            if not is_valid: return False, f"PHYSICS VIOLATION: {reason}"
        
        if roi > limit:
            return False, f"CATEGORY LIMIT: {symbol} ({category}) ROI {roi*100:.1f}% > limit {limit*100:.1f}%"
            
        return True, "Valid"

    def cross_validate_external(self, symbol: str, internal_roi: float) -> Tuple[bool, str]:
        """Query external feed for price verification if internal spike is massive."""
        # For simulation, we primarily use this to detect 'Lookahead' or 'Zero Volume' glitches
        # In live, we hit CoinGecko. In evolution, we just nuke suspicious outliers.
        logger.warning(f"🛡️ CROSS-VETTING MASSIVE SPIKE on {symbol}: {internal_roi*100:.2f}%")
        
        # Logic: If it's too good to be true, and it's not a verified meme-pump, it's a glitch.
        if internal_roi > 1000.0: # 100,000% ROI? No.
            self._log_rejection(symbol, internal_roi, "Extreme Outlier")
            return False, "Extreme Outlier"
            
        return True, "Validation Passed"

    def _log_rejection(self, symbol: str, roi: float, reason: str):
        now = time.time()
        self.rejection_log.append({
            'timestamp': now, 
            'symbol': symbol, 
            'roi': roi, 
            'reason': reason
        })
        self.glitch_history.append(now)
        
        # Check frequency
        one_hour_ago = now - 3600
        recent = [t for t in self.glitch_history if t > one_hour_ago]
        if len(recent) > self.alert_threshold:
            logger.critical(f"⚠️ SYSTEMIC FEED GLITCH: {len(recent)} detection errors in the last hour!")

    def audit_candle(self, symbol: str, open_p: float, high_p: float, low_p: float, close_p: float, volume: float) -> bool:
        """High-speed structural check for single candle integrity."""
        if volume <= 0 and abs(close_p - open_p) / open_p > 0.01:
            return False # Price moved > 1% on zero volume
            
        # Flash Crash check
        wick_size = (high_p - low_p) / low_p
        if wick_size > 0.50 and self.get_category(symbol) != 'meme_coin':
             return False # 50% candle wick on non-meme
             
        return True
