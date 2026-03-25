"""
Signal Quality Gate - Early Filtering to Prevent Churn

Filters signals BEFORE expensive analysis (Structure, Orion, ML, etc.)

Checks:
1. Minimum contract size (prevent "qty too small" rejections)
2. ML confidence threshold (filter very low confidence early)
3. Spread/Liquidity (filter untradable symbols)
4. Blacklist (filter banned symbols)

Usage:
    from HolonicTrader.signal_quality_gate import SignalQualityGate
    
    gate = SignalQualityGate()
    if gate.passes_quality_check(signal_data):
        # Proceed with full analysis
    else:
        # Reject early, save compute
"""

import time
from typing import Dict, Any, Optional
from pathlib import Path

class SignalQualityGate:
    """
    Early filtering to prevent signal churn
    Reject low-quality signals BEFORE expensive analysis
    """
    
    def __init__(self):
        # Minimum contract sizes (from exchange)
        self.min_contract_sizes = {
            'BNB/USDT': 0.01,
            'BTC/USDT': 0.001,
            'ETH/USDT': 0.01,
            'SOL/USDT': 0.1,
            'TAO/USDT': 0.01,
            'LDO/USDT': 0.1,
            'AAVE/USDT': 0.01,
            'XRP/USDT': 1.0,
            'DOGE/USDT': 1.0,
            'PEPE/USDT': 1000.0,
            'WIF/USDT': 0.1,
            # Default fallback
            'DEFAULT': 0.01
        }
        
        # ML confidence threshold (reject below this)
        self.ml_min_confidence = 0.35  # Reject <35% win probability
        
        # Spread threshold (reject if spread too wide)
        self.max_spread_pct = 0.005  # 0.5% max spread
        
        # Liquidity threshold
        self.min_liquidity_score = 0.3
        
        # Cache for ML predictions (prevent duplicate calls)
        self._ml_cache = {}
        self._cache_ttl = 60  # seconds
        
        # Blacklist (from Governor)
        self.blacklist = {}
        
        print("🚪 Signal Quality Gate initialized")
        print(f"   ML min confidence: {self.ml_min_confidence:.0%}")
        print(f"   Max spread: {self.max_spread_pct:.1%}")
        print(f"   Cache TTL: {self._cache_ttl}s")
    
    def passes_quality_check(self, signal: Dict[str, Any], market_data: Dict[str, Any] = None) -> tuple:
        """
        Early quality check BEFORE expensive analysis
        
        Returns: (passed: bool, reason: str)
        """
        symbol = signal.get('symbol', 'UNKNOWN')
        
        # Check 1: Blacklist (fastest check first)
        if symbol in self.blacklist:
            if time.time() < self.blacklist[symbol]:
                return False, f'BLACKLISTED ({self._get_blacklist_remaining(symbol)})'
            else:
                del self.blacklist[symbol]  # Expired
        
        # Check 2: Minimum contract size (prevent "qty too small" rejections)
        qty = signal.get('quantity', 0)
        min_qty = self._get_min_contract_size(symbol)
        
        if qty > 0 and qty < min_qty:
            return False, f'QTY_TOO_SMALL ({qty:.4f} < {min_qty})'
        
        # Check 3: Spread (if market data available)
        if market_data:
            spread = market_data.get('spread_pct', 0)
            if spread > self.max_spread_pct:
                return False, f'SPREAD_TOO_WIDE ({spread:.2%} > {self.max_spread_pct:.1%})'
            
            liquidity = market_data.get('liquidity_score', 1.0)
            if liquidity < self.min_liquidity_score:
                return False, f'LOW_LIQUIDITY ({liquidity:.2f} < {self.min_liquidity_score:.1f})'
        
        # Check 4: ML confidence (cached to prevent duplicate calls)
        ml_result = self._get_ml_confidence_cached(signal)
        if ml_result:
            win_prob = ml_result['win_probability']
            
            if win_prob < self.ml_min_confidence:
                return False, f'ML_LOW_CONFIDENCE ({win_prob:.1%} < {self.ml_min_confidence:.0%})'
        
        # All checks passed
        return True, 'PASSED'
    
    def _get_min_contract_size(self, symbol: str) -> float:
        """Get minimum contract size for symbol"""
        base = symbol.split('/')[0]
        
        # Check full symbol first
        if symbol in self.min_contract_sizes:
            return self.min_contract_sizes[symbol]
        
        # Check base asset
        if base in self.min_contract_sizes:
            return self.min_contract_sizes[base]
        
        # Default
        return self.min_contract_sizes['DEFAULT']
    
    def _get_ml_confidence_cached(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get ML confidence with caching (prevent duplicate calls)
        
        Cache key: symbol + direction + timestamp (minute granularity)
        """
        symbol = signal.get('symbol', '')
        direction = signal.get('direction', '')
        price = signal.get('price', 0)
        
        # Create cache key (minute granularity)
        minute = int(time.time() / 60)
        cache_key = f"{symbol}_{direction}_{minute}"
        
        # Check cache
        if cache_key in self._ml_cache:
            cached = self._ml_cache[cache_key]
            if time.time() - cached['time'] < self._cache_ttl:
                return cached['result']
            else:
                del self._ml_cache[cache_key]  # Expired
        
        # Not cached - caller should fetch and cache
        # Return None to indicate "fetch fresh"
        return None
    
    def cache_ml_result(self, signal: Dict[str, Any], result: Dict[str, Any]):
        """Cache ML prediction result"""
        symbol = signal.get('symbol', '')
        direction = signal.get('direction', '')
        
        minute = int(time.time() / 60)
        cache_key = f"{symbol}_{direction}_{minute}"
        
        self._ml_cache[cache_key] = {
            'time': time.time(),
            'result': result
        }
    
    def update_blacklist(self, symbol: str, expiry_timestamp: float):
        """Update symbol blacklist"""
        self.blacklist[symbol] = expiry_timestamp
    
    def _get_blacklist_remaining(self, symbol: str) -> str:
        """Get remaining blacklist time"""
        if symbol not in self.blacklist:
            return '0h'
        
        remaining = self.blacklist[symbol] - time.time()
        if remaining <= 0:
            return '0h'
        
        hours = remaining / 3600
        return f'{hours:.1f}h'
    
    def get_stats(self) -> Dict[str, Any]:
        """Get gate statistics"""
        return {
            'cache_size': len(self._ml_cache),
            'blacklist_size': len(self.blacklist),
            'ml_min_confidence': self.ml_min_confidence,
            'min_contract_sizes': len(self.min_contract_sizes)
        }


# Singleton
_gate_instance = None

def get_signal_quality_gate() -> SignalQualityGate:
    """Get gate singleton"""
    global _gate_instance
    if _gate_instance is None:
        _gate_instance = SignalQualityGate()
    return _gate_instance


# Convenience function
def check_signal_quality(signal: Dict[str, Any], market_data: Dict[str, Any] = None) -> tuple:
    """Check signal quality"""
    gate = get_signal_quality_gate()
    return gate.passes_quality_check(signal, market_data)


print("🚪 Signal Quality Gate loaded - Early filtering active")
