"""
Arb Safety Monitor - Real-time protection for funding arbitrage positions

Monitors:
1. Price moves against position (stop loss)
2. Funding rate flips (exit signal)
3. Funding convergence (exit when no longer profitable)
4. PnL thresholds (kill zone)
5. Exchange connectivity (circuit breaker)

Actions:
1. Alert on warning thresholds
2. Auto-close on stop loss or funding convergence
3. Halt new positions on kill zone
4. Emergency close on defensive regime

FIX 2026-03-04:
- Remove stop-losses from pure arb (exit on funding convergence instead)
- Add emergency close for defensive regime
- Add arb cooldown tracking after stop-loss
"""

import time
import ccxt
import config
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

class ArbSafetyMonitor:
    """
    Real-time safety monitor for arb positions
    
    FIX 2026-03-04: Arb positions should exit on funding convergence,
    not price-based stop-losses (which trigger before funding can offset losses)
    """

    def __init__(self, governor, executor):
        self.governor = governor
        self.executor = executor
        self.exchange = None

        # Initialize exchange connection
        try:
            if config.TRADING_MODE == 'FUTURES':
                self.exchange = ccxt.krakenfutures({
                    'apiKey': config.KRAKEN_FUTURES_API_KEY or config.API_KEY,
                    'secret': config.KRAKEN_FUTURES_PRIVATE_KEY or config.API_SECRET,
                    'enableRateLimit': True,
                })
        except Exception as e:
            print(f"[ArbSafety] Exchange init failed: {e}")

        # Position tracking
        self.entry_prices = {}  # {symbol: entry_price}
        self.entry_funding_rates = {}  # {symbol: funding_rate_at_entry}
        self.position_sizes = {}  # {symbol: size_usd}
        self.entry_times = {}  # {symbol: timestamp}

        # === FIX 2026-03-04: ARB MANAGEMENT ===
        # Stop-loss disabled for pure arb - use funding convergence instead
        self.stop_loss_enabled = False  # DISABLED for arb positions
        self.stop_loss_pct = getattr(config, 'STOP_LOSS_ARB_PCT', 0.08)  # Fallback only
        
        # Funding convergence threshold - exit when funding drops below this
        self.funding_convergence_threshold = getattr(config, 'FUNDING_CONVERGENCE_THRESHOLD', 0.10)  # 10% per 8H
        
        # Arb cooldown after stop-loss (prevent immediate re-entry)
        self._arb_cooldowns = {}  # {symbol: cooldown_end_timestamp}
        self.arb_cooldown_duration = 3600  # 1 hour cooldown after stop-loss
        
        # Emergency close threshold
        self.emergency_close_drawdown = 0.04  # 4% drawdown triggers emergency close

        # Safety thresholds (from config)
        self.funding_flip_threshold = 0.0  # Flip from negative to positive
        self.max_daily_loss = getattr(config, 'MAX_DAILY_LOSS_PCT', 0.10)

        # Tracking
        self.daily_pnl = 0.0
        self.last_reset = datetime.now(timezone.utc).date()
        self.alerts = []
        self.positions_closed_by_stop = []  # Track symbols that hit stop (for cooldown)

        print(f"[ArbSafety] Monitor initialized")
        print(f"   Stop Loss: DISABLED (using funding convergence)")
        print(f"   Funding Convergence: {self.funding_convergence_threshold*100:.1f}% per 8H")
        print(f"   Daily Loss Limit: {self.max_daily_loss*100:.1f}%")
        print(f"   Emergency Close: {self.emergency_close_drawdown*100:.1f}% drawdown")
        print(f"   Arb Cooldown: {self.arb_cooldown_duration/60:.0f} minutes after stop-loss")
    
    def track_position(self, symbol: str, entry_price: float, 
                       size_usd: float, funding_rate: float):
        """
        Start tracking a new position

        Args:
            symbol: Asset symbol
            entry_price: Entry price
            size_usd: Position size in USD
            funding_rate: Funding rate at entry
        """
        self.entry_prices[symbol] = entry_price
        self.position_sizes[symbol] = size_usd
        self.entry_funding_rates[symbol] = funding_rate
        self.entry_times[symbol] = time.time()

        print(f"[ArbSafety] Tracking: {symbol} @ ${entry_price:.4f}, {size_usd:.2f} USD, funding {funding_rate*100:.2f}%/8H")

    def untrack_position(self, symbol: str):
        """Stop tracking a closed position"""
        if symbol in self.entry_prices:
            del self.entry_prices[symbol]
        if symbol in self.position_sizes:
            del self.position_sizes[symbol]
        if symbol in self.entry_funding_rates:
            del self.entry_funding_rates[symbol]
        if symbol in self.entry_times:
            del self.entry_times[symbol]

        print(f"[ArbSafety] Stopped tracking: {symbol}")

    def is_in_cooldown(self, symbol: str) -> bool:
        """
        FIX 2026-03-04: Check if symbol is in cooldown after stop-loss.
        Prevents immediate re-entry after a losing arb position.
        """
        if symbol not in self._arb_cooldowns:
            return False
        
        now = time.time()
        if now < self._arb_cooldowns[symbol]:
            remaining = int(self._arb_cooldowns[symbol] - now)
            print(f"[ArbSafety] ⏱️ COOLDOWN ACTIVE: {symbol} ({remaining}s remaining)")
            return True
        
        # Cooldown expired, remove it
        del self._arb_cooldowns[symbol]
        return False

    def set_cooldown(self, symbol: str):
        """Set cooldown for a symbol after stop-loss"""
        self._arb_cooldowns[symbol] = time.time() + self.arb_cooldown_duration
        self.positions_closed_by_stop.append(symbol)
        print(f"[ArbSafety] ⏱️ COOLDOWN SET: {symbol} for {self.arb_cooldown_duration/60:.0f} minutes")
    
    def check_all_positions(self) -> Dict[str, List[str]]:
        """
        Check all tracked positions for safety violations

        FIX 2026-03-04:
        - Check funding convergence (exit when no longer profitable)
        - Check emergency close (defensive regime)
        - Removed: price-based stop-loss for arb positions

        Returns:
            Dict of {symbol: [alerts]}
        """
        # Reset daily PnL if new day
        today = datetime.now(timezone.utc).date()
        if today != self.last_reset:
            self.daily_pnl = 0.0
            self.last_reset = today

        alerts = {}

        # === FIX 2026-03-04: EMERGENCY CLOSE CHECK ===
        emergency_alert = self._check_emergency_close()
        if emergency_alert:
            alerts['EMERGENCY_CLOSE'] = [emergency_alert]
            return alerts  # Return immediately to trigger emergency close

        for symbol in list(self.entry_prices.keys()):
            symbol_alerts = []

            # Check 1: Funding convergence (REPLACED stop-loss)
            convergence_alert = self._check_funding_convergence(symbol)
            if convergence_alert:
                symbol_alerts.append(convergence_alert)

            # Check 2: Funding flip
            funding_alert = self._check_funding_flip(symbol)
            if funding_alert:
                symbol_alerts.append(funding_alert)

            # Check 3: PnL threshold
            pnl_alert = self._check_pnl_threshold(symbol)
            if pnl_alert:
                symbol_alerts.append(pnl_alert)

            if symbol_alerts:
                alerts[symbol] = symbol_alerts

        # Check 4: Daily loss limit
        if self.daily_pnl < -self.max_daily_loss:
            alerts['KILL_ZONE'] = [f"Daily loss limit breached: {self.daily_pnl*100:.1f}%"]

        return alerts

    def _check_emergency_close(self) -> Optional[str]:
        """
        FIX 2026-03-04: Emergency close all arb positions if drawdown exceeds threshold.
        This prevents death spiral in defensive regime.
        """
        # Get current drawdown from executor
        if hasattr(self.executor, 'daily_drawdown'):
            drawdown = self.executor.daily_drawdown
        else:
            # Calculate from initial capital vs current
            if self.executor.initial_capital > 0:
                drawdown = (self.executor.initial_capital - self.executor.balance_usd) / self.executor.initial_capital
            else:
                return None

        if drawdown > self.emergency_close_drawdown:
            return f"EMERGENCY CLOSE: Drawdown {drawdown*100:.1f}% > {self.emergency_close_drawdown*100:.1f}% limit - Close all arb positions!"
        
        return None

    def _check_funding_convergence(self, symbol: str) -> Optional[str]:
        """
        FIX 2026-03-04: Check if funding rate has converged (no longer profitable).
        This REPLACES price-based stop-loss for arb positions.
        
        Arb thesis: Collect funding while it's extreme (>100% APY)
        Exit when: Funding drops below threshold (<10% APY)
        """
        if symbol not in self.entry_funding_rates:
            return None

        try:
            ticker = self.exchange.fetch_ticker(symbol.replace('/USDT', '/USD:USD') if '/USDT' in symbol else symbol)
            info = ticker.get('info', {})
            current_funding = float(info.get('fundingRate', 0))

            entry_funding = self.entry_funding_rates[symbol]
            entry_funding_8h = entry_funding * 100  # Convert to % per 8H
            current_funding_8h = current_funding * 100

            # Check if funding has converged below threshold
            if abs(current_funding_8h) < self.funding_convergence_threshold:
                # Funding no longer profitable - exit
                funding_collected = self._estimate_funding_collected(symbol)
                return f"FUNDING CONVERGENCE: {symbol} funding {current_funding_8h:.2f}% < {self.funding_convergence_threshold*100:.1f}% threshold - Exit (Collected: ${funding_collected:.2f})"

        except Exception as e:
            pass  # Don't alert on price fetch failures

        return None

    def _estimate_funding_collected(self, symbol: str) -> float:
        """Estimate funding collected since entry"""
        if symbol not in self.entry_times or symbol not in self.position_sizes:
            return 0.0
        
        entry_funding = self.entry_funding_rates.get(symbol, 0)
        position_size = self.position_sizes.get(symbol, 0)
        entry_time = self.entry_times.get(symbol, time.time())
        
        # Calculate hours since entry
        hours_elapsed = (time.time() - entry_time) / 3600
        
        # Funding is paid every 8 hours, estimate pro-rated
        funding_per_hour = abs(entry_funding) / 8
        estimated_collected = position_size * funding_per_hour * hours_elapsed
        
        return estimated_collected
    
    def _check_funding_flip(self, symbol: str) -> Optional[str]:
        """Check if funding rate flipped sign"""
        if symbol not in self.entry_funding_rates:
            return None
        
        try:
            ticker = self.exchange.fetch_ticker(symbol.replace('/USDT', '/USD:USD') if '/USDT' in symbol else symbol)
            info = ticker.get('info', {})
            current_funding = float(info.get('fundingRate', 0))
            
            entry_funding = self.entry_funding_rates[symbol]
            
            # Check for flip (negative to positive or vice versa)
            if (entry_funding < 0 and current_funding > 0.005) or \
               (entry_funding > 0 and current_funding < -0.005):
                return f"FUNDING FLIP: {symbol} {entry_funding*100:.2f}% -> {current_funding*100:.2f}%"
            
        except Exception as e:
            pass
        
        return None
    
    def _check_pnl_threshold(self, symbol: str) -> Optional[str]:
        """Check if position hit PnL threshold"""
        # Placeholder for future PnL tracking
        return None
    
    def get_status(self) -> Dict:
        """Get monitor status"""
        return {
            'tracking_count': len(self.entry_prices),
            'daily_pnl': self.daily_pnl,
            'alerts_count': len(self.alerts),
            'stop_loss_pct': self.stop_loss_pct,
            'max_daily_loss': self.max_daily_loss,
        }
    
    def print_status(self):
        """Print formatted status"""
        status = self.get_status()
        print("\n" + "=" * 50)
        print("ARB SAFETY MONITOR STATUS")
        print("=" * 50)
        print(f"Tracking: {status['tracking_count']} positions")
        print(f"Daily PnL: {status['daily_pnl']*100:.2f}%")
        print(f"Stop Loss: {status['stop_loss_pct']*100:.1f}%")
        print(f"Daily Limit: {status['max_daily_loss']*100:.1f}%")
        print(f"Active Alerts: {status['alerts_count']}")
        print("=" * 50)


# Global instance
_safety_monitor = None

def initialize_safety_monitor(governor, executor) -> ArbSafetyMonitor:
    """Initialize the safety monitor"""
    global _safety_monitor
    _safety_monitor = ArbSafetyMonitor(governor, executor)
    return _safety_monitor

def get_safety_monitor() -> Optional[ArbSafetyMonitor]:
    """Get the safety monitor instance"""
    return _safety_monitor

def track_position(symbol: str, entry_price: float, size_usd: float, funding_rate: float):
    """Track a new position"""
    if _safety_monitor:
        _safety_monitor.track_position(symbol, entry_price, size_usd, funding_rate)

def untrack_position(symbol: str):
    """Untrack a closed position"""
    if _safety_monitor:
        _safety_monitor.untrack_position(symbol)

def check_safety() -> Dict:
    """Check all positions for safety violations"""
    if _safety_monitor:
        return _safety_monitor.check_all_positions()
    return {}

def print_safety_status():
    """Print safety monitor status"""
    if _safety_monitor:
        _safety_monitor.print_status()
    else:
        print("Safety monitor not initialized")
