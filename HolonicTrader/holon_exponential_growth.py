"""
ExponentialGrowthHolon - Auto-Compounding for Arb Nuggets

This module automatically compounds arbitrage profits to achieve exponential account growth.
It manages position sizing, reinvestment rates, and phase transitions based on account equity.
"""

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class GrowthPhase:
    """Configuration for a growth phase"""
    name: str
    reinvest_rate: float      # 0.0-1.0 (100% = full compounding)
    risk_per_trade: float     # % of equity to deploy
    min_apy_threshold: float  # Minimum APY to consider
    max_positions: int        # Max concurrent positions
    profit_take_rate: float   # % to withdraw (1 - reinvest_rate)

@dataclass
class ArbNugget:
    """Represents a collected funding payment"""
    symbol: str
    timestamp: str
    payment_usd: float
    funding_rate: float
    position_size: float
    reinvested: float
    reserved: float

class ExponentialGrowthHolon:
    """
    Automatically compounds arbitrage profits for exponential growth.
    
    Phases:
    1. AGGRESSIVE (<$1k): 100% reinvest, concentrated bets
    2. SCALED ($1k-$10k): 80% reinvest, more diversification
    3. CAPACITY ($10k-$30k): 60% reinvest, full diversification
    4. DIVERSIFIED (>$30k): 50% reinvest, multi-asset
    """
    
    # Phase configurations
    # FIX 2026-03-02: Lowered APY thresholds to enable xStocks execution
    # xStocks typical APY: 50-450% annualized (was blocked at 500%)
    PHASES = {
        'AGGRESSIVE': GrowthPhase(
            name='AGGRESSIVE',
            reinvest_rate=1.0,
            risk_per_trade=0.95,
            min_apy_threshold=50,    # Was 500 - enables xStocks arb
            max_positions=5,         # Was 3 - more concurrent arb positions
            profit_take_rate=0.0,
        ),
        'SCALED': GrowthPhase(
            name='SCALED',
            reinvest_rate=0.80,
            risk_per_trade=0.70,
            min_apy_threshold=30,    # Was 200 - captures more opportunities
            max_positions=8,         # Was 5
            profit_take_rate=0.20,
        ),
        'CAPACITY': GrowthPhase(
            name='CAPACITY',
            reinvest_rate=0.60,
            risk_per_trade=0.50,
            min_apy_threshold=20,    # Was 100
            max_positions=12,        # Was 8
            profit_take_rate=0.40,
        ),
        'DIVERSIFIED': GrowthPhase(
            name='DIVERSIFIED',
            reinvest_rate=0.50,
            risk_per_trade=0.40,
            min_apy_threshold=10,    # Was 50
            max_positions=20,        # Was 15
            profit_take_rate=0.50,
        ),
    }
    
    def __init__(self, governor=None, initial_equity: float = 120.0):
        self.governor = governor
        self.equity = initial_equity
        self.initial_equity = initial_equity
        self.phase = self._determine_phase()
        self.nuggets_collected: List[ArbNugget] = []
        self.total_nugget_value = 0.0
        self.total_reinvested = 0.0
        self.total_reserved = 0.0
        
        # Tracking
        self._last_phase_transition = time.time()
        self._peak_equity = initial_equity
        self._drawdown = 0.0
        
        # Kill zone limits
        self.max_daily_loss = 0.15
        self.max_weekly_loss = 0.25
        self.max_drawdown = 0.30
        
        print(f"[ExponentialGrowth] Initialized with ${initial_equity:.2f}")
        print(f"[ExponentialGrowth] Phase: {self.phase}")
    
    def _determine_phase(self) -> str:
        """Determine growth phase based on equity"""
        if self.equity < 1000:
            return 'AGGRESSIVE'
        elif self.equity < 10000:
            return 'SCALED'
        elif self.equity < 30000:
            return 'CAPACITY'
        else:
            return 'DIVERSIFIED'
    
    def calculate_position_sizes(self, opportunities: List[Dict]) -> Dict[str, Dict]:
        """
        Calculate optimal position sizes for exponential growth
        
        Args:
            opportunities: List of arb opportunities from scanner
                          [{symbol, apy, funding_rate, max_position_usd, ...}]
        
        Returns:
            Dict of {symbol: {size_usd, direction, expected_apy, nugget_per_8h}}
        """
        config = self.PHASES[self.phase]
        
        # Filter by APY threshold
        qualified = [
            o for o in opportunities 
            if abs(o.get('apy', 0)) >= config.min_apy_threshold
        ]
        
        if not qualified:
            print(f"[ExponentialGrowth] No opportunities meet {config.min_apy_threshold}% APY threshold")
            return {}
        
        # Sort by APY (highest first)
        qualified.sort(key=lambda x: abs(x.get('apy', 0)), reverse=True)
        
        # Take top N positions
        top_opps = qualified[:config.max_positions]
        
        # Calculate total risk capital
        total_risk_capital = self.equity * config.risk_per_trade
        
        # Allocate capital (equal weight among selected)
        per_position = total_risk_capital / len(top_opps) if top_opps else 0
        
        allocation = {}
        for opp in top_opps:
            symbol = opp.get('symbol', '')
            
            # Respect max position (OI limit)
            max_pos = opp.get('max_position_usd', float('inf'))
            size = min(per_position, max_pos)
            
            # Determine direction based on funding sign
            funding_rate = opp.get('funding_rate', 0)
            direction = 'SELL' if funding_rate < 0 else 'BUY'
            
            # Calculate expected nugget per 8h
            nugget_per_8h = size * abs(funding_rate)
            
            allocation[symbol] = {
                'size_usd': size,
                'direction': direction,
                'expected_apy': opp.get('apy', 0),
                'funding_rate': funding_rate,
                'nugget_per_8h': nugget_per_8h,
                'nugget_daily': nugget_per_8h * 3,  # 3 payments per day
                'max_position_usd': max_pos,
            }
        
        # Log allocation summary
        total_allocated = sum(a['size_usd'] for a in allocation.values())
        expected_daily = sum(a['nugget_daily'] for a in allocation.values())
        
        print(f"\n[ALLOCATION] {self.phase}")
        print(f"   Equity: ${self.equity:,.2f}")
        print(f"   Risk Capital: ${total_risk_capital:,.2f} ({config.risk_per_trade*100:.0f}%)")
        print(f"   Allocated: ${total_allocated:,.2f}")
        print(f"   Expected Daily Nuggets: ${expected_daily:,.2f}")
        print(f"   Positions: {len(allocation)}")
        
        for symbol, data in allocation.items():
            print(f"   - {symbol}: ${data['size_usd']:,.2f} -> ${data['nugget_daily']:.2f}/day")
        
        return allocation
    
    def on_funding_payment(self, symbol: str, payment_usd: float, 
                           funding_rate: float, position_size: float) -> Dict:
        """
        Called when funding payment is received
        
        Automatically compounds according to phase rules
        
        Args:
            symbol: Asset symbol
            payment_usd: Payment amount (positive = received, negative = paid)
            funding_rate: Funding rate that generated this payment
            position_size: Position size that generated payment
            
        Returns:
            Dict with compounding details
        """
        config = self.PHASES[self.phase]
        
        # Only compound positive payments
        if payment_usd <= 0:
            return {'payment': payment_usd, 'action': 'LOSS', 'reinvested': 0}
        
        # Calculate reinvestment
        reinvest = payment_usd * config.reinvest_rate
        reserve = payment_usd * (1 - config.reinvest_rate)
        
        # Compound
        self.equity += reinvest
        self.total_reinvested += reinvest
        self.total_reserved += reserve
        self.total_nugget_value += payment_usd
        
        # Track peak and drawdown
        if self.equity > self._peak_equity:
            self._peak_equity = self.equity
        self._drawdown = (self._peak_equity - self.equity) / self._peak_equity
        
        # Check for phase transition
        new_phase = self._determine_phase()
        phase_changed = new_phase != self.phase
        
        if phase_changed:
            old_phase = self.phase
            self.phase = new_phase
            self._last_phase_transition = time.time()
            print(f"\n[PHASE] TRANSITION: {old_phase} -> {new_phase}")
            print(f"   Equity: ${self.equity:,.2f}")
            print(f"   New Reinvest Rate: {config.reinvest_rate*100:.0f}%")
        
        # Record nugget
        nugget = ArbNugget(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payment_usd=payment_usd,
            funding_rate=funding_rate,
            position_size=position_size,
            reinvested=reinvest,
            reserved=reserve,
        )
        self.nuggets_collected.append(nugget)
        
        # Log
        print(f"\n[NUGGET] COLLECTED: {symbol}")
        print(f"   Payment: ${payment_usd:.2f}")
        print(f"   Reinvested: ${reinvest:.2f} ({config.reinvest_rate*100:.0f}%)")
        print(f"   Reserved: ${reserve:.2f} ({(1-config.reinvest_rate)*100:.0f}%)")
        print(f"   New Equity: ${self.equity:,.2f}")
        print(f"   Total Nuggets: ${self.total_nugget_value:,.2f}")
        
        return {
            'payment': payment_usd,
            'reinvested': reinvest,
            'reserved': reserve,
            'new_equity': self.equity,
            'phase': self.phase,
            'phase_changed': phase_changed,
            'nugget_count': len(self.nuggets_collected),
        }
    
    def check_kill_zone(self) -> Tuple[bool, str]:
        """
        Check if kill zone limits are breached
        
        Returns:
            (is_triggered, reason)
        """
        # Drawdown check
        if self._drawdown > self.max_drawdown:
            return True, f"Max drawdown breached: {self._drawdown*100:.1f}% > {self.max_drawdown*100:.0f}%"
        
        # Daily loss check (simplified - would track daily PnL)
        # Weekly loss check (simplified - would track weekly PnL)
        
        return False, ""
    
    def get_growth_projection(self, days: int = 30, apy_decay: float = 0.5) -> List[Dict]:
        """
        Project account growth with realistic APY decay
        
        Args:
            days: Projection period
            apy_decay: Weekly APY decay factor (0.5 = 50% reduction per week)
            
        Returns:
            List of daily projections
        """
        config = self.PHASES[self.phase]
        current_apy = 3000  # Start at 3,000%
        
        projections = []
        equity = self.equity
        
        for day in range(days):
            # Weekly decay
            if day % 7 == 0 and day > 0:
                current_apy *= apy_decay
            
            # Daily growth
            daily_rate = (current_apy / 100) / 365
            daily_growth = equity * daily_rate * config.reinvest_rate
            equity += daily_growth
            
            projections.append({
                'day': day + 1,
                'equity': equity,
                'apy': current_apy,
                'daily_growth': daily_growth,
                'cumulative_growth': (equity / self.equity - 1) * 100,
            })
        
        return projections
    
    def get_status(self) -> Dict:
        """Get current status summary"""
        config = self.PHASES[self.phase]
        
        return {
            'equity': self.equity,
            'phase': self.phase,
            'reinvest_rate': config.reinvest_rate,
            'risk_per_trade': config.risk_per_trade,
            'total_nuggets': len(self.nuggets_collected),
            'total_nugget_value': self.total_nugget_value,
            'total_reinvested': self.total_reinvested,
            'total_reserved': self.total_reserved,
            'peak_equity': self._peak_equity,
            'current_drawdown': self._drawdown,
            'growth_multiple': self.equity / self.initial_equity,
        }
    
    def print_status(self):
        """Print formatted status"""
        status = self.get_status()
        
        print("\n" + "=" * 60)
        print("EXPONENTIAL GROWTH STATUS")
        print("=" * 60)
        print(f"Equity: ${status['equity']:,.2f} ({status['growth_multiple']:.1f}x from start)")
        print(f"Phase: {status['phase']}")
        print(f"Reinvest Rate: {status['reinvest_rate']*100:.0f}%")
        print(f"Risk per Trade: {status['risk_per_trade']*100:.0f}%")
        print(f"Peak Equity: ${status['peak_equity']:,.2f}")
        print(f"Drawdown: {status['current_drawdown']*100:.1f}%")
        print(f"Total Nuggets: {status['total_nuggets']} (${status['total_nugget_value']:,.2f})")
        print(f"Total Reinvested: ${status['total_reinvested']:,.2f}")
        print(f"Total Reserved: ${status['total_reserved']:,.2f}")
        print("=" * 60)


# Convenience functions for integration

def create_growth_engine(governor, initial_equity: float = 120.0) -> ExponentialGrowthHolon:
    """Create and initialize growth engine"""
    return ExponentialGrowthHolon(governor, initial_equity)


def scan_and_allocate(growth_engine: ExponentialGrowthHolon, 
                      arb_scanner,
                      account_equity: float) -> Dict:
    """
    Scan for opportunities and calculate allocation
    
    Args:
        growth_engine: ExponentialGrowthHolon instance
        arb_scanner: XStocksArbitrage instance
        account_equity: Current account equity
        
    Returns:
        Allocation dict
    """
    # Update equity
    growth_engine.equity = account_equity
    
    # Scan for opportunities
    opportunities = arb_scanner.find_arbitrage_opportunities(
        min_apy=100,
        min_oi=10
    )
    
    # Calculate allocation
    allocation = growth_engine.calculate_position_sizes(opportunities)
    
    return allocation


if __name__ == '__main__':
    # Test run
    print("Testing ExponentialGrowthHolon\n")
    
    # Initialize
    engine = ExponentialGrowthHolon(initial_equity=120.0)
    
    # Simulate funding payments
    print("\n--- Simulating Funding Payments ---\n")
    
    payments = [
        ('SPYX/USD:USD', 106.30, -0.1063, 1000),  # $1000 position, -10.63% funding
        ('MSTRX/USD:USD', 61.40, -0.0614, 1000),   # $1000 position, -6.14% funding
        ('GOOGLX/USD:USD', 18.30, -0.0183, 1000),  # $1000 position, -1.83% funding
    ]
    
    for symbol, payment, rate, size in payments:
        engine.on_funding_payment(symbol, payment, rate, size)
    
    # Print status
    engine.print_status()
    
    # Show projection
    print("\n--- 30-Day Projection (50% weekly APY decay) ---\n")
    projections = engine.get_growth_projection(days=30, apy_decay=0.5)
    
    print(f"{'Day':<5} {'Equity':>12} {'APY':>10} {'Daily':>12} {'Growth':>10}")
    print("-" * 55)
    for p in projections:
        print(f"{p['day']:<5} ${p['equity']:>10,.2f} {p['apy']:>8.1f}% ${p['daily_growth']:>10,.2f} {p['cumulative_growth']:>+8.1f}%")
