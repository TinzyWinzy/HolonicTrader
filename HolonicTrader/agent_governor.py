"""
GovernorHolon - NEXUS Risk & Homeostasis (Phase 15)

Implements:
1. Dual Metabolic State (SCAVENGER / PREDATOR)
2. Volatility Targeting (ATR-based position sizing)
3. Principal Protection (Never risk the $10 base)
"""

from typing import Any, Tuple, Literal
from HolonicTrader.holon_core import Holon, Disposition
import config

import time

class GovernorHolon(Holon):
    def __init__(self, name: str = "GovernorAgent", initial_balance: float = 10.0):
        super().__init__(name=name, disposition=Disposition(autonomy=0.9, integration=0.9))
        
        self.balance = initial_balance
        self.hard_stop_threshold = 5.0
        self.DEBUG = False # Silence rejection spam
        
        # Reference ATR for volatility targeting (set during first cycle)
        self.reference_atr = None
        
        # Position Tracking (Multi-Asset)
        self.positions = {} # symbol -> {entry_price, quantity, direction}
        self.last_trade_time = {} # symbol -> timestamp
        self.last_specific_entry = {} # symbol -> price (for stacking distance)
        
    def sync_positions(self, held_assets: dict, metadata: dict):
        """
        Sync positions from Executor/DB on startup to cure Amnesia.
        """
        print(f"[{self.name}] Syncing positions from DB...")
        count = 0
        for symbol, qty in held_assets.items():
            if qty > 0:
                meta = metadata.get(symbol, {})
                entry_price = meta.get('entry_price', 0.0)
                # Reconstruct position entry
                self.positions[symbol] = {
                    'direction': 'LONG', # Assuming LONG for now
                    'entry_price': entry_price,
                    'quantity': qty,
                    'stack_count': 1 # Assume initial entry for synced positions
                }
                # Sync stacking tracker
                self.last_specific_entry[symbol] = entry_price
                
                count += 1
                print(f"[{self.name}] Restricted: {symbol} (Qty: {qty:.4f})")
                
        if count == 0:
            print(f"[{self.name}] No active positions found to sync.")
        
    def update_balance(self, new_balance: float):
        """Update the internal balance knowledge."""
        self.balance = new_balance
        self._check_homeostasis()

    def _check_homeostasis(self):
        """Check if the system is viable."""
        if self.balance < self.hard_stop_threshold:
            self.state = 'HIBERNATE'
            print(f"[{self.name}] CRITICAL: Balance ${self.balance:.2f} < ${self.hard_stop_threshold}. HIBERNATING.")
        else:
            if self.state == 'HIBERNATE':
                self.state = 'ACTIVE'

    def get_metabolism_state(self) -> Literal['SCAVENGER', 'PREDATOR']:
        """
        Determine current metabolic state based on balance.
        """
        if self.balance <= config.SCAVENGER_THRESHOLD:
            return 'SCAVENGER'
        else:
            return 'PREDATOR'

    def is_trade_allowed(self, symbol: str, asset_price: float) -> bool:
        """
        Lightweight check to see if a trade would be allowed.
        Prevents Strategy from wasting compute on blocked trades.
        """
        # 1. Cooldown Check
        last_time = self.last_trade_time.get(symbol, 0)
        if time.time() - last_time < config.GOVERNOR_COOLDOWN_SECONDS:
            return False
            
        # 2. Price Distance Check
        last_entry = self.last_specific_entry.get(symbol, 0)
        if last_entry > 0 and symbol in self.positions:
            dist = abs(asset_price - last_entry) / last_entry
            if dist < config.GOVERNOR_MIN_STACK_DIST:
                return False
                
        return True

    def calc_position_size(self, symbol: str, asset_price: float, current_atr: float = None) -> Tuple[bool, float, float]:
        """
        Calculate position size based on metabolic state and volatility.
        
        Returns:
            (is_approved: bool, quantity: float, leverage: float)
        """
        if self.state == 'HIBERNATE':
            print(f"[{self.name}] Trade REJECTED: System in HIBERNATION.")
            return False, 0.0, 0.0

        if asset_price <= 0:
            print(f"[{self.name}] Trade REJECTED: Invalid Asset Price.")
            return False, 0.0, 0.0
            
        # WARP SPEED 3.0: Smart Stacking & Cooldowns
        
        # 1. Cooldown Check
        last_time = self.last_trade_time.get(symbol, 0)
        if time.time() - last_time < config.GOVERNOR_COOLDOWN_SECONDS:
            if self.DEBUG:
                print(f"[{self.name}] REJECTED: Cooldown active for {symbol} ({int(config.GOVERNOR_COOLDOWN_SECONDS - (time.time() - last_time))}s rem).")
            return False, 0.0, 0.0
            
        # 2. Price Distance Check
        # Only if we already have a position
        last_entry = self.last_specific_entry.get(symbol, 0)
        if last_entry > 0 and symbol in self.positions:
            dist = abs(asset_price - last_entry) / last_entry
            if dist < config.GOVERNOR_MIN_STACK_DIST:
                if self.DEBUG:
                    print(f"[{self.name}] REJECTED: Price {asset_price} too close to last entry {last_entry} (Dist: {dist*100:.2f}% < {config.GOVERNOR_MIN_STACK_DIST*100}%).")
                return False, 0.0, 0.0
        
        state = self.get_metabolism_state()
        
        if state == 'SCAVENGER':
            # 10-Bullet Rule: Max margin %
            margin = min(config.SCAVENGER_MAX_MARGIN, self.balance * config.GOVERNOR_MAX_MARGIN_PCT)
            leverage = config.SCAVENGER_LEVERAGE
            effective_size = margin * leverage
            quantity = effective_size / asset_price
            
            print(f"[{self.name}] SCAVENGER: Margin ${margin:.2f}, Lev {leverage}x, Qty {quantity:.4f}")
            return True, quantity, leverage
            
        else:  # PREDATOR
            # Surplus = Balance - Principal
            surplus = self.balance - config.INITIAL_CAPITAL
            
            if surplus <= 0:
                # Edge case: exactly at threshold, use scavenger logic
                return self.calc_position_size(symbol, asset_price, current_atr)
            
            # Base Size = Surplus * Leverage
            base_leverage = config.PREDATOR_LEVERAGE
            
            # Volatility Targeting (BlackRock Rule)
            if current_atr and self.reference_atr:
                vol_adj = self.reference_atr / current_atr
                vol_adj = max(0.5, min(2.0, vol_adj))  # Clamp between 0.5x and 2x
            else:
                vol_adj = 1.0
                
            # Step-Down Sizing (Pyramiding Optimization)
            current_stack = self.positions.get(symbol, {}).get('stack_count', 0)
            stack_decay = 0.85 ** current_stack
            
            # Combine adjustments
            adjusted_leverage = base_leverage * vol_adj * stack_decay
            effective_size = surplus * adjusted_leverage
            quantity = effective_size / asset_price
            
            print(f"[{self.name}] PREDATOR: Surplus ${surplus:.2f}, Lev {adjusted_leverage:.1f}x (Vol: {vol_adj:.2f}, Stack[{current_stack}]: {stack_decay:.2f}), Qty {quantity:.4f}")
            return True, quantity, adjusted_leverage
    
    def open_position(self, symbol: str, direction: str, entry_price: float, quantity: float):
        """Track that a position has been opened or added to (Weighted Average)."""
        
        # Update State Trackers
        self.last_trade_time[symbol] = time.time()
        self.last_specific_entry[symbol] = entry_price
        
        existing = self.positions.get(symbol)
        
        if existing:
            old_qty = existing['quantity']
            old_price = existing['entry_price']
            new_qty = old_qty + quantity
            
            # Weighted Average Price
            avg_price = ((old_qty * old_price) + (quantity * entry_price)) / new_qty
            
            self.positions[symbol] = {
                'direction': direction,
                'entry_price': avg_price,
                'quantity': new_qty,
                'stack_count': existing.get('stack_count', 1) + 1
            }
            print(f"[{self.name}] Position STACKED: {symbol} (New Avg: {avg_price:.4f}, Total Qty: {new_qty:.4f}, Stacks: {existing.get('stack_count', 1) + 1})")
        else:
            self.positions[symbol] = {
                'direction': direction,
                'entry_price': entry_price,
                'quantity': quantity,
                'stack_count': 1
            }
            print(f"[{self.name}] Position OPENED: {symbol} {direction} @ {entry_price}")
        
    def close_position(self, symbol: str):
        """Clear position tracking."""
        if symbol in self.positions:
            del self.positions[symbol]
            print(f"[{self.name}] Position CLOSED: {symbol}")

    def set_reference_atr(self, atr: float):
        """Set the reference ATR for volatility targeting."""
        if self.reference_atr is None:
            self.reference_atr = atr
            print(f"[{self.name}] Reference ATR set: {atr:.6f}")

    def receive_message(self, sender: Any, content: Any) -> Any:
        """Handle incoming messages."""
        msg_type = content.get('type')
        if msg_type == 'VALIDATE_TRADE':
            symbol = content.get('symbol')
            price = content.get('price')
            atr = content.get('atr')
            return self.calc_position_size(symbol, price, atr)
            
        elif msg_type == 'POSITION_FILLED':
            self.open_position(
                content.get('symbol'),
                content.get('direction'),
                content.get('price'),
                content.get('quantity')
            )
            return True
            
        elif msg_type == 'POSITION_CLOSED':
            self.close_position(content.get('symbol'))
            return True
            
        elif msg_type == 'GET_STATE':
            return self.get_metabolism_state()
            
        elif msg_type == 'WAKE_UP':
            print(f"[{self.name}] Received WAKE_UP signal from Immune System.")
            self.state = 'ACTIVE'
            return True
            
        return None
