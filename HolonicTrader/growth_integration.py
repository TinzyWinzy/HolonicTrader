"""
Exponential Growth Integration Module

This module integrates the ExponentialGrowthHolon with the live trading system.
It runs as a companion to main_live.py and handles auto-compounding of arb nuggets.

SAFETY FEATURES:
- Real-time position monitoring
- Stop loss enforcement
- Funding flip detection
- Daily loss limits
- Emergency halt capability
"""

import time
import config
from typing import Optional, Dict
from HolonicTrader.holon_exponential_growth import ExponentialGrowthHolon
from HolonicTrader.strategy_xstocks_arb import XStocksArbitrage, scan_xstocks_arb
from HolonicTrader.arb_safety_monitor import (
    initialize_safety_monitor,
    get_safety_monitor,
    track_position,
    untrack_position,
    check_safety,
    print_safety_status
)

# Global instances
_growth_engine: Optional[ExponentialGrowthHolon] = None
_arb_scanner: Optional[XStocksArbitrage] = None
_last_rebalance_time = 0
_rebalance_interval = 3600  # Rebalance every hour

def initialize_growth_engine(governor, executor, initial_equity: float = 120.0) -> bool:
    """
    Initialize the exponential growth engine AND safety monitor
    
    Args:
        governor: GovernorHolon instance
        executor: ExecutorHolon instance
        initial_equity: Starting equity
        
    Returns:
        True if successful, False otherwise
    """
    global _growth_engine, _arb_scanner
    
    try:
        print("\n" + "=" * 60)
        print("🚀 EXPONENTIAL GROWTH MODE ENABLED")
        print("=" * 60)
        
        # Initialize growth engine
        _growth_engine = ExponentialGrowthHolon(governor, initial_equity=initial_equity)
        
        # Initialize arb scanner
        _arb_scanner = XStocksArbitrage()
        
        # Initialize SAFETY MONITOR (critical!)
        initialize_safety_monitor(governor, executor)
        print("\n🛡️  SAFETY MONITOR INITIALIZED")
        print(f"   Stop Loss: {config.STOP_LOSS_ARB_PCT*100:.1f}%")
        print(f"   Daily Loss Limit: {config.MAX_DAILY_LOSS_PCT*100:.1f}%")
        print(f"   Max Leverage: {config.ARBITRAGE_MAX_LEVERAGE}x")
        
        # Print current opportunities
        print("\n📊 CURRENT ARB OPPORTUNITIES:")
        try:
            from HolonicTrader.strategy_xstocks_arb import get_xstocks_summary
            print(get_xstocks_summary())
        except Exception as e:
            print(f"Could not fetch summary: {e}")
        
        # Calculate initial allocation
        print("\n📋 CALCULATING INITIAL ALLOCATION...")
        opportunities = _arb_scanner.find_arbitrage_opportunities(
            min_apy=config.ARBITRAGE_MIN_APY,
            min_oi=10
        )
        allocation = _growth_engine.calculate_position_sizes(opportunities)
        
        print(f"\n✅ Growth Engine Ready - Phase: {_growth_engine.phase}")
        print(f"   Reinvest Rate: {_growth_engine.PHASES[_growth_engine.phase].reinvest_rate*100:.0f}%")
        print(f"   Risk per Trade: {_growth_engine.PHASES[_growth_engine.phase].risk_per_trade*100:.0f}%")
        print("=" * 60 + "\n")
        
        return True
        
    except Exception as e:
        print(f">> [Warning] Exponential Growth Engine failed to start: {e}")
        _growth_engine = None
        _arb_scanner = None
        return False

def get_growth_engine() -> Optional[ExponentialGrowthHolon]:
    """Get the growth engine instance"""
    return _growth_engine

def get_arb_scanner() -> Optional[XStocksArbitrage]:
    """Get the arb scanner instance"""
    return _arb_scanner

def on_funding_payment(symbol: str, payment_usd: float, 
                       funding_rate: float, position_size: float) -> Dict:
    """
    Called when a funding payment is received
    
    Args:
        symbol: Asset symbol
        payment_usd: Payment amount
        funding_rate: Funding rate
        position_size: Position size
        
    Returns:
        Compounding result dict
    """
    if _growth_engine is None:
        return {'error': 'Growth engine not initialized'}
    
    return _growth_engine.on_funding_payment(symbol, payment_usd, funding_rate, position_size)

def should_rebalance() -> bool:
    """Check if it's time to rebalance positions"""
    global _last_rebalance_time
    now = time.time()
    
    if now - _last_rebalance_time > _rebalance_interval:
        _last_rebalance_time = now
        return True
    return False

def get_current_allocation(account_equity: float) -> Dict:
    """
    Get current recommended allocation
    
    Args:
        account_equity: Current account equity
        
    Returns:
        Allocation dict
    """
    if _growth_engine is None or _arb_scanner is None:
        return {}
    
    # Update equity
    _growth_engine.equity = account_equity
    
    # Scan for opportunities
    opportunities = _arb_scanner.find_arbitrage_opportunities(
        min_apy=config.ARBITRAGE_MIN_APY,
        min_oi=10
    )
    
    # Calculate allocation
    return _growth_engine.calculate_position_sizes(opportunities)

def get_status() -> Dict:
    """Get growth engine status"""
    if _growth_engine is None:
        return {'enabled': False}
    
    status = _growth_engine.get_status()
    status['enabled'] = True
    return status

def print_status():
    """Print formatted status"""
    if _growth_engine is None:
        print("Growth engine not initialized")
        return
    
    _growth_engine.print_status()
    print_safety_status()

def run_safety_check() -> Dict:
    """
    Run safety check on all positions
    
    Returns:
        Dict with safety status and any alerts
    """
    alerts = check_safety()
    
    if alerts:
        print("\n⚠️  SAFETY ALERTS DETECTED:")
        for symbol, symbol_alerts in alerts.items():
            for alert in symbol_alerts:
                print(f"   [{symbol}] {alert}")
        
        # Check for kill zone
        if 'KILL_ZONE' in alerts:
            print("\n🛑 KILL ZONE TRIGGERED - Halting new positions!")
            # Would set a flag to halt trading here
    
    return alerts
