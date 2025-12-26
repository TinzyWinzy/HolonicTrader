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
    def __init__(self, name: str = "GovernorAgent", initial_balance: float = 10.0, db_manager: Any = None):
        super().__init__(name=name, disposition=Disposition(autonomy=0.9, integration=0.9))
        
        self.balance = initial_balance
        self.hard_stop_threshold = 5.0
        self.DEBUG = False # Silence rejection spam
        self.db_manager = db_manager  # For win rate tracking
        
        # Phase 22: Portfolio Health Tracking
        self.max_balance = initial_balance
        self.drawdown_pct = 0.0
        self.margin_utilization = 0.0
        
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
                direction = meta.get('direction', 'BUY') # Proper sync from metadata
                
                # Reconstruct position entry
                self.positions[symbol] = {
                    'direction': direction,
                    'entry_price': entry_price,
                    'quantity': qty,
                    'stack_count': 1, # Assume initial entry for synced positions
                    'first_entry_time': time.time() # Sync time as approx start
                }
                # Sync stacking tracker
                self.last_specific_entry[symbol] = entry_price
                
                count += 1
                print(f"[{self.name}] Synchronized: {symbol} ({direction}, Qty: {qty:.4f})")
                
        if count == 0:
            print(f"[{self.name}] No active positions found to sync.")
        
    def update_balance(self, new_balance: float):
        """Update the internal balance knowledge and health metrics."""
        self.balance = new_balance
        
        # Track Drawdown
        if self.balance > self.max_balance:
            self.max_balance = self.balance
            
        if self.max_balance > 0:
            self.drawdown_pct = (self.max_balance - self.balance) / self.max_balance
        
        # Calculate Margin Utilization
        total_exposure = 0.0
        for sym, pos in self.positions.items():
            total_exposure += abs(pos['quantity']) * pos['entry_price']
            
        if self.balance > 0:
            # We normalize margin utilization based on the config limit
            # If we use all allowed margin, util = 1.0
            allowed_exposure = self.balance * config.GOVERNOR_MAX_MARGIN_PCT * config.PREDATOR_LEVERAGE
            self.margin_utilization = total_exposure / allowed_exposure if allowed_exposure > 0 else 0.0

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

    def get_portfolio_health(self) -> dict:
        """Expose health metrics for PPO Brain."""
        return {
            'drawdown_pct': self.drawdown_pct,
            'margin_utilization': self.margin_utilization,
            'balance': self.balance,
            'max_balance': self.max_balance
        }

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

    def calc_position_size(self, symbol: str, asset_price: float, current_atr: float = None, atr_ref: float = None, conviction: float = 0.5) -> Tuple[bool, float, float]:
        """
        Calculate position size with Phase 12 institutional risk management.
        
        Integrates:
        1. Minimax Constraint (protect principal)
        2. Volatility Scalar (ATR-based sizing)
        4. Conviction Scalar (LSTM-based scaling)
        
        Returns:
            (is_approved: bool, quantity: float, leverage: float)
        """
        # --- PHASE 25: SATELLITE OVERRIDE ---
        # The executor passes 'conviction' which we used as a carrier for metadata in previous versions,
        # but here we might need a clearer signal. 
        # Actually, let's inspect the 'conviction' arg. If it's a dict or special object? 
        # No, 'conviction' is float.
        # We need to rely on the `symbol` being in `SATELLITE_ASSETS`.
        if symbol in config.SATELLITE_ASSETS:
             # Fixed Sizing: $10 Margin * 10x Lev = $100 Position
             notional = config.SATELLITE_MARGIN * config.SATELLITE_LEVERAGE
             quantity = notional / asset_price
             return True, quantity, config.SATELLITE_LEVERAGE
        # ------------------------------------

        if self.state == 'HIBERNATE':
            print(f"[{self.name}] Trade REJECTED: System in HIBERNATION.")
            return False, 0.0, 0.0

        # --- PATCH 2: THE STACKING CAP (Stop the Martingale) ---
        MAX_STACKS = 3
        existing_pos = self.positions.get(symbol)
        if existing_pos:
            current_stacks = existing_pos.get('stack_count', 1)
            if current_stacks >= MAX_STACKS:
                if self.DEBUG:
                     print(f"[{self.name}] ⚠️ MAX STACKS REACHED ({current_stacks}). REJECTING ORDER.")
                return False, 0.0, 0.0
        # -------------------------------------------------------

        # --- PHASE 35: IMMUNE SYSTEM CHECKS ---
        if not self.check_cluster_risk(symbol):
            return False, 0.0, 0.0
            
        # --------------------------------------

        # 1. Minimax Constraint (The "House Money" Rule)
        max_loss_usd = self.calculate_max_risk(self.balance)
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
        last_entry = self.last_specific_entry.get(symbol, 0)
        if last_entry > 0 and symbol in self.positions:
            dist = abs(asset_price - last_entry) / last_entry
            if dist < config.GOVERNOR_MIN_STACK_DIST:
                if self.DEBUG:
                    print(f"[{self.name}] REJECTED: Price {asset_price} too close to last entry {last_entry} (Dist: {dist*100:.2f}% < {config.GOVERNOR_MIN_STACK_DIST*100}%).")
                return False, 0.0, 0.0
        
        state = self.get_metabolism_state()
        
        # === PHASE 12: INSTITUTIONAL RISK MANAGEMENT ===
        
        # Conviction Scalar (0.5 to 1.5)
        # conviction here is LSTM prob (0-1). We transform it.
        # For BUYS: prob > 0.5 is good. For SELLS: prob < 0.5 is good.
        # Wait, the EntryOracle already chooses direction. 
        # Let's assume passed conviction is 'strength' (0.5 to 1.0).
        conv_scalar = 0.5 + (max(0.0, conviction - 0.5) * 2.0)
        conv_scalar = max(0.5, min(1.5, conv_scalar))

        # Base position sizing
        if state == 'SCAVENGER':
            # 10-Bullet Rule: Max margin %
            margin = min(config.SCAVENGER_MAX_MARGIN, self.balance * config.GOVERNOR_MAX_MARGIN_PCT)
            leverage = config.SCAVENGER_LEVERAGE
            base_notional = margin * leverage * conv_scalar
            
        else:  # PREDATOR
            leverage = config.PREDATOR_LEVERAGE
            
            # Use Modified Kelly for PREDATOR
            kelly_size_usd = self.calculate_kelly_size(self.balance)
            
            # Trend Age Decay
            current_pos = self.positions.get(symbol)
            decay_mult = 1.0
            if current_pos:
                # 1. Age-based Decay
                age_hours = (time.time() - current_pos.get('first_entry_time', time.time())) / 3600.0
                if age_hours > config.GOVERNOR_TREND_DECAY_START:
                    overtime = age_hours - config.GOVERNOR_TREND_DECAY_START
                    window = config.GOVERNOR_MAX_TREND_AGE_HOURS - config.GOVERNOR_TREND_DECAY_START
                    decay_mult *= max(0.0, 1.0 - (overtime / window))
                    print(f"[{self.name}] ⏳ Trend Age {age_hours:.1f}h. Decaying by {decay_mult:.2f}x")
                
                # 2. Stack-based Decay (Phase 18)
                stacks = current_pos.get('stack_count', 0)
                stack_decay = (config.GOVERNOR_STACK_DECAY ** stacks)
                decay_mult *= stack_decay
                
                if decay_mult < 1.0:
                    print(f"[{self.name}] 🥞 Stack {stacks} Decay: {stack_decay:.2f}x (Total Decay: {decay_mult:.2f}x)")
                    kelly_size_usd *= decay_mult
                    
                if age_hours > config.GOVERNOR_MAX_TREND_AGE_HOURS:
                    print(f"[{self.name}] 🛑 Trend Exhausted (>24h). Rejecting Stack.")
                    return False, 0.0, 0.0
            
            base_notional = kelly_size_usd * leverage * conv_scalar
        
        # Apply Volatility Scalar (if ATR provided)
        if current_atr and atr_ref:
            vol_scalar = self.calculate_volatility_scalar(current_atr, atr_ref)
            vol_adjusted_notional = base_notional * vol_scalar
            print(f"[{self.name}] 📊 Volatility Scalar: {vol_scalar:.2f}x, Conviction: {conv_scalar:.2f}x")
        else:
            vol_adjusted_notional = base_notional
            vol_scalar = 1.0
        
        # Apply Minimax Constraint (CRITICAL)
        max_risk_usd = self.calculate_max_risk(self.balance)
        
        # Assume mode-specific stop loss distance for risk calculation
        sl_dist = config.SCAVENGER_STOP_LOSS if state == 'SCAVENGER' else config.PREDATOR_STOP_LOSS
        max_notional_from_risk = max_risk_usd / sl_dist
        
        # Take minimum of volatility-adjusted and risk-constrained
        final_notional = min(vol_adjusted_notional, max_notional_from_risk)
        
        # Convert to quantity
        quantity = final_notional / asset_price
        
        # Leverage Cap (Dynamic based on Conviction?)
        # For now, we stick to Config limits per asset class
        max_leverage = config.SCAVENGER_LEVERAGE if state == 'SCAVENGER' else config.PREDATOR_LEVERAGE
        
        # --- PHASE 35: LEVERAGE CHECK ---
        notional_value = quantity * asset_price
        if not self.check_leverage_risk(notional_value):
            return False, 0.0, 0.0
        # -------------------------------
        
        # Log decision
        if state == 'SCAVENGER':
            print(f"[{self.name}] SCAVENGER: Margin ${margin:.2f}, Lev {leverage}x, Vol Scalar {vol_scalar:.2f}x, Conv Scalar {conv_scalar:.2f}x, Qty {quantity:.4f}")
        else:
            print(f"[{self.name}] PREDATOR (Kelly): Kelly ${kelly_size_usd:.2f}, Lev {leverage}x, Vol Scalar {vol_scalar:.2f}x, Conv Scalar {conv_scalar:.2f}x, Qty {quantity:.4f}")
        
        return True, quantity, leverage

            
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
            if abs(new_qty) > 1e-9:
                avg_price = ((old_qty * old_price) + (quantity * entry_price)) / new_qty
            else:
                avg_price = 0.0
            
            self.positions[symbol] = {
                'direction': direction,
                'entry_price': avg_price,
                'quantity': new_qty,
                'stack_count': existing.get('stack_count', 1) + 1,
                'first_entry_time': existing.get('first_entry_time', time.time())
            }
            print(f"[{self.name}] Position STACKED: {symbol} (New Avg: {avg_price:.4f}, Total Qty: {new_qty:.4f}, Stacks: {existing.get('stack_count', 1) + 1})")
        else:
            self.positions[symbol] = {
                'direction': direction,
                'entry_price': entry_price,
                'quantity': quantity,
                'stack_count': 1,
                'first_entry_time': time.time()
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

    # === PHASE 12: INSTITUTIONAL RISK MANAGEMENT ===
    
    def calculate_max_risk(self, balance: float) -> float:
        """
        Minimax Constraint (Game Theory):
        Never risk the principal ($10). Only risk house money OR 1% of total.
        
        Args:
            balance: Current account balance
            
        Returns:
            Maximum USD that can be risked on a single trade
        """
        house_money = max(0, balance - config.PRINCIPAL)
        pct_risk = balance * config.MAX_RISK_PCT
        
        # Whichever is lower: house money or 1% of total
        max_risk_usd = min(house_money, pct_risk)
        
        return max_risk_usd
    
    def calculate_volatility_scalar(self, atr_current: float, atr_ref: float) -> float:
        """
        Volatility Scalar (Inverse Variance Weighting):
        Normalize position size based on current volatility.
        
        Formula: Size_adj = Size_base × (ATR_ref / ATR_current)
        
        Args:
            atr_current: Current ATR value
            atr_ref: Reference ATR (14-period average)
            
        Returns:
            Scalar multiplier (clamped to 0.5-2.0)
        """
        if atr_current <= 0 or atr_ref <= 0:
            return 1.0
        
        # Inverse relationship: high volatility = smaller size
        scalar = atr_ref / atr_current
        
        # Clamp to reasonable range
        return max(config.VOL_SCALAR_MIN, min(config.VOL_SCALAR_MAX, scalar))
    
    def calculate_recent_win_rate(self, lookback: int = None) -> float:
        """
        Calculate win rate from recent trades.
        
        Args:
            lookback: Number of recent trades to analyze
            
        Returns:
            Win rate (0.0 to 1.0)
        """
        if lookback is None:
            lookback = config.KELLY_LOOKBACK
        
        # Integrate with database to get actual win rate
        if hasattr(self, 'db_manager') and self.db_manager:
            try:
                # Get recent trades from database
                trades = self.db_manager.get_recent_trades(lookback)
                if trades and len(trades) > 0:
                    # Calculate actual win rate
                    wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
                    actual_wr = wins / len(trades)
                    
                    # BLENDING: If we have few trades, blend with a neutral baseline (0.40)
                    # to prevent "Cold Start" rejection (e.g. 0% WR after 1 loss).
                    sample_size = len(trades)
                    min_sample = 10
                    if sample_size < min_sample:
                        baseline = 0.40
                        weight = sample_size / min_sample
                        win_rate = (actual_wr * weight) + (baseline * (1 - weight))
                    else:
                        win_rate = actual_wr
                        
                    print(f"[{self.name}] 📊 Win Rate: {win_rate*100:.1f}% (Actual: {actual_wr*100:.1f}%, n={sample_size})")
                    return win_rate
            except Exception as e:
                print(f"[{self.name}] ⚠️ Win rate calculation failed: {e}")
        
        return 0.40

    def check_cluster_risk(self, symbol: str) -> bool:
        """
        Refuse trade if we already hold an asset from the same family.
        Returns: False if RISK DETECTED (Reject), True if SAFE.
        """
        family = None
        if symbol in config.FAMILY_L1: family = config.FAMILY_L1
        elif symbol in config.FAMILY_PAYMENT: family = config.FAMILY_PAYMENT
        elif symbol in config.FAMILY_MEME: family = config.FAMILY_MEME
        
        if not family: return True # No family, no risk
        
        # Check holdings
        for asset, data in self.positions.items():
            if abs(data['quantity']) > 0 and asset in family and asset != symbol:
                print(f"[{self.name}] ⚠️ CLUSTER RISK: Rejecting {symbol} (Already hold {asset})")
                return False
        return True

    def check_leverage_risk(self, new_notional_value: float) -> bool:
        """
        Refuse trade if Total Notional Exposure > 10x Balance.
        """
        current_exposure = 0.0
        # Sum absolute notional value of all positions
        for asset, data in self.positions.items():
            # We need current price for accurate notional, but entry_price is a decent proxy for risk check
            # preventing API calls here.
            qty = abs(data['quantity'])
            price = data['entry_price']
            current_exposure += (qty * price)
            
        total_exposure = current_exposure + new_notional_value
        max_allowed = self.balance * config.IMMUNE_MAX_LEVERAGE_RATIO
        
        if total_exposure > max_allowed:
            print(f"[{self.name}] ⚠️ OVER-LEVERAGE: Exposure ${total_exposure:.0f} > Limit ${max_allowed:.0f}")
            return False
        return True

    def calculate_kelly_size(self, balance: float, win_rate: float = None, risk_reward: float = None) -> float:
        """
        Modified Kelly Criterion (Half-Kelly):
        Calculate optimal position size for PREDATOR mode.
        
        Formula: f* = [(p(b+1) - 1) / b] × 0.5
        
        Args:
            balance: Current account balance
            win_rate: Recent win rate (0.0 to 1.0)
            risk_reward: Expected reward/risk ratio
            
        Returns:
            Maximum position size in USD
        """
        # Only use surplus, not principal
        surplus = max(0, balance - config.PRINCIPAL)
        
        if surplus <= 0:
            # Emergency Unit: If we are at the edge, allow a $1.00 margin unit 
            # to prevent total paralysis if conviction is high.
            if balance > (config.PRINCIPAL * 0.9):
                return 1.0 / config.PREDATOR_LEVERAGE 
            return 0.0
        
        # Use smoothed win rate if not provided
        if win_rate is None:
            win_rate = self.calculate_recent_win_rate()
        if risk_reward is None:
            risk_reward = config.KELLY_RISK_REWARD
        
        # Kelly formula: f* = [p(b+1) - 1] / b
        b = risk_reward
        kelly_fraction = ((win_rate * (b + 1)) - 1) / b
        
        # Half-Kelly for safety
        half_kelly = kelly_fraction * 0.5
        
        # Clamp to reasonable range (Floor prevents 0% WR from killing all trades)
        safe_fraction = max(config.KELLY_MIN_FRACTION, min(config.KELLY_MAX_FRACTION, half_kelly))
        
        return surplus * safe_fraction

    def receive_message(self, sender: Any, content: Any) -> Any:
        """Handle incoming messages."""
        msg_type = content.get('type')
        if msg_type == 'VALIDATE_TRADE':
            symbol = content.get('symbol')
            price = content.get('price')
            atr = content.get('atr')
            conviction = content.get('conviction', 0.5)
            # Check if conviction is None (if key exists but value is None)
            if conviction is None: conviction = 0.5
            
            return self.calc_position_size(symbol, price, atr, conviction=conviction)
            
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
