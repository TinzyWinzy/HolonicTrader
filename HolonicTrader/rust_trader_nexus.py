"""
Phase 4: Rust TraderNexus Python Wrapper

Provides a clean Python interface to the Rust TraderNexus core.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger("RustTraderNexus")

try:
    import holonic_speed
    RUST_AVAILABLE = True
except ImportError as e:
    RUST_AVAILABLE = False
    logger.warning(f"holonic_speed not available: {e}")


class RustTraderNexus:
    """
    Python wrapper for Rust TraderNexus.
    
    This class provides a high-level interface to the Rust trading core,
    handling market data formatting, indicator preparation, and signal execution.
    """
    
    def __init__(
        self,
        initial_capital: float = 100.0,
        max_positions: int = 8,
        leverage: float = 5.0,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        cycle_interval_ms: int = 60000,
    ):
        """
        Initialize Rust TraderNexus.
        
        Args:
            initial_capital: Starting capital in USD
            max_positions: Maximum concurrent positions
            leverage: Default leverage for positions
            stop_loss_pct: Default stop loss percentage
            take_profit_pct: Default take profit percentage
            cycle_interval_ms: Trading cycle interval in milliseconds
        """
        if not RUST_AVAILABLE:
            raise ImportError("holonic_speed module not available")
        
        # Create Rust config
        self.config = holonic_speed.NexusConfig(
            initial_capital=initial_capital,
            fee_rate=0.002,
            leverage=leverage,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            trailing_stop_active=0.02,
            trailing_stop_dist=0.01,
            max_positions=max_positions,
            risk_per_trade=0.02,
            cycle_interval_ms=cycle_interval_ms,
        )
        
        # Create Rust TraderNexus instance
        self.nexus = holonic_speed.TraderNexus(self.config)
        
        self.is_running = False
        self.cycle_count = 0
        
        logger.info(f"Rust TraderNexus initialized: ${initial_capital} capital, {max_positions} max positions")
    
    def start(self):
        """Start the trading loop"""
        self.nexus.start_py()
        self.is_running = True
        logger.info("Rust TraderNexus started")
    
    def stop(self):
        """Stop the trading loop"""
        self.nexus.stop_py()
        self.is_running = False
        logger.info("Rust TraderNexus stopped")
    
    def run_cycle(
        self,
        market_data: Dict[str, Dict[str, Any]],
        indicators: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Run single trading cycle.
        
        Args:
            market_data: Dict of symbol -> {
                'closes': [price1, price2, ...],
                'highs': [...],
                'lows': [...],
                'opens': [...],
                'volumes': [...]
            }
            indicators: Dict of symbol -> {
                'rsi': [values],
                'bb_lower': [values],
                'bb_upper': [values],
                'atr': [values],
                'entropy': [values],
                'obv': [values]
            }
        
        Returns:
            List of trading signals generated
        """
        if not self.is_running:
            logger.warning("TraderNexus not running, skipping cycle")
            return []
        
        # Format market data for Rust (symbol -> closes)
        rust_market_data = {}
        for symbol, data in market_data.items():
            if 'closes' in data and data['closes']:
                rust_market_data[symbol] = data['closes']
        
        # Format indicators for Rust (symbol_indicator -> values)
        rust_indicators = {}
        for symbol, ind in indicators.items():
            for ind_name, values in ind.items():
                if values:
                    key = f"{symbol}_{ind_name}"
                    rust_indicators[key] = values
        
        # Run Rust cycle
        try:
            signals = self.nexus.run_cycle(rust_market_data, rust_indicators)
            self.cycle_count = self.nexus.cycle_count()
            
            # Convert Rust signals to Python dicts
            py_signals = []
            for sig in signals:
                py_signals.append({
                    'symbol': sig.symbol if hasattr(sig, 'symbol') else 'UNKNOWN',
                    'action': sig.action if hasattr(sig, 'action') else 'HOLD',
                    'confidence': sig.confidence if hasattr(sig, 'confidence') else 0.5,
                    'size': sig.size if hasattr(sig, 'size') else 0.0,
                })
            
            return py_signals
            
        except Exception as e:
            logger.error(f"Rust cycle failed: {e}")
            return []
    
    def get_equity(self) -> float:
        """Get current equity"""
        return self.nexus.get_equity()
    
    def get_status(self) -> Dict[str, Any]:
        """Get nexus status"""
        status = self.nexus.status()
        status['is_running'] = self.is_running
        status['cycle_count'] = self.cycle_count
        return status
    
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        return self.nexus.get_metrics()
    
    def start_twap(
        self,
        symbol: str,
        side: str,
        total_qty: float,
        duration_minutes: int = 30,
        num_slices: int = 6,
    ):
        """
        Start TWAP execution.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            side: "BUY" or "SELL"
            total_qty: Total quantity to execute
            duration_minutes: Duration over which to execute
            num_slices: Number of execution slices
        """
        self.nexus.start_twap(symbol, side, total_qty, duration_minutes, num_slices)
        logger.info(f"TWAP started: {symbol} {side} {total_qty} over {duration_minutes}min")
    
    def start_vwap(
        self,
        symbol: str,
        side: str,
        total_qty: float,
        volume_profile: Optional[List[float]] = None,
    ):
        """
        Start VWAP execution.
        
        Args:
            symbol: Trading pair
            side: "BUY" or "SELL"
            total_qty: Total quantity to execute
            volume_profile: Optional volume profile weights
        """
        if volume_profile is None:
            # Use default U-shaped profile
            try:
                volume_profile = holonic_speed.vwap_generate_volume_profile(12)
            except:
                volume_profile = [1.0/12] * 12
        
        self.nexus.start_vwap(symbol, side, total_qty, volume_profile)
        logger.info(f"VWAP started: {symbol} {side} {total_qty}")
    
    def scan_arbitrage(self) -> List[Dict[str, Any]]:
        """Scan for arbitrage opportunities"""
        try:
            opportunities = self.nexus.scan_arbitrage()
            return [
                {
                    'base_asset': opp.base_asset,
                    'quote_asset': opp.quote_asset,
                    'buy_exchange': opp.buy_exchange,
                    'sell_exchange': opp.sell_exchange,
                    'buy_price': opp.buy_price,
                    'sell_price': opp.sell_price,
                    'spread_pct': opp.spread_pct,
                    'expected_profit_pct': opp.expected_profit_pct,
                    'max_quantity': opp.max_quantity,
                }
                for opp in opportunities
            ]
        except Exception as e:
            logger.error(f"Arbitrage scan failed: {e}")
            return []


def create_rust_nexus(
    initial_capital: float = 100.0,
    max_positions: int = 8,
    **kwargs
) -> Optional[RustTraderNexus]:
    """
    Factory function to create Rust TraderNexus.
    
    Returns None if Rust engine not available.
    """
    if not RUST_AVAILABLE:
        logger.warning("Rust engine not available, falling back to Python TraderHolon")
        return None
    
    try:
        nexus = RustTraderNexus(
            initial_capital=initial_capital,
            max_positions=max_positions,
            **kwargs
        )
        logger.info("Rust TraderNexus created successfully")
        return nexus
    except Exception as e:
        logger.error(f"Failed to create Rust TraderNexus: {e}")
        return None
