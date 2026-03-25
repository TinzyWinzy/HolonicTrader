"""
governor_risk.py - Risk Management Module
Phase 2: Extracted from GovernorHolon for modularity

Implements:
1. Position sizing (Kelly, Minimax, Volatility)
2. Solvency checks (Margin, Leverage)
3. Risk calculations (Ruin probability, Cluster risk)
"""

from typing import Dict, Any, Optional
import config


class RiskManager:
    """
    Handles all risk calculations and position sizing for GovernorHolon.
    Delegates back to parent for position state.
    """
    
    def __init__(self, governor):
        """
        Initialize RiskManager with reference to parent GovernorHolon.
        
        Args:
            governor: Parent GovernorHolon instance for state access
        """
        self.governor = governor
        self.name = f"{governor.name}.RiskManager"
        self.DEBUG = getattr(governor, 'DEBUG', False)
        
    # =========================================================================
    # CORE RISK CALCULATIONS
    # =========================================================================
    
    def calculate_max_risk(self, balance: float) -> float:
        """
        Minimax Constraint (Game Theory):
        Never risk the principal ($10). Only risk house money OR 1% of total.
        Uses Rust Engine (holonic_speed) if available.
        """
        try:
            import holonic_speed
            return holonic_speed.governor_calculate_max_risk(
                balance, config.PRINCIPAL, balance
            )
        except ImportError:
            # Fallback to Python
            house_money = max(0, balance - config.PRINCIPAL)
            pct_risk = balance * config.MAX_RISK_PCT
            return min(house_money, pct_risk)
    
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

    def calculate_sde_physics_scalar(self, metadata: Dict[str, Any], direction: str = 'BUY') -> float:
        """
        Physics-Based Position Scaling (SDE Layer).
        Dynamically adjusts size based on SDE drift and diffusion.
        """
        if not metadata or 'sde_physics' not in metadata:
            return 1.0
            
        try:
            sde = metadata['sde_physics']
            mu = sde.get('mu', 0.0)      # Drift
            sigma = sde.get('sigma', 0.1)  # Diffusion
            
            # Sharpe-like ratio for direction
            if sigma <= 0:
                return 1.0
                
            # If LONG, want positive drift; if SHORT, want negative drift
            direction_sign = 1.0 if direction == 'BUY' else -1.0
            alignment = (direction_sign * mu) / sigma
            
            # Scale: alignment of 0.5+ = 1.5x, alignment of -0.5 = 0.5x
            scalar = 1.0 + (alignment * 0.5)
            
            # Clamp to safe range
            return max(0.5, min(2.0, scalar))
            
        except Exception as e:
            if self.DEBUG:
                print(f"[{self.name}] SDE Physics Error: {e}")
            return 1.0
    
    def calculate_ruin_probability(
        self, 
        symbol: str, 
        entry_price: float, 
        direction: str, 
        stop_loss: float, 
        take_profit: float, 
        metadata: Dict[str, Any]
    ) -> float:
        """
        Monte Carlo Ruin Guard:
        Uses optimized SDEEngine (Rust accelerated) to estimate 
        the probability of hitting Stop Loss before Take Profit/Horizon.
        """
        if not metadata or 'sde_physics' not in metadata:
            return 0.5 
            
        try:
            from HolonicTrader.sde_engine import SDEEngine
            sde = metadata['sde_physics']
            # Parameters from Oracle
            params = {
                'mu': sde.get('mu', 0.0),
                'sigma': sde.get('sigma', 0.1),
                'lambda': sde.get('lambda', 0.1)
            }
            
            # Use Rust-accelerated calculation
            return SDEEngine.calculate_ruin_probability(
                'GBM',  # Default model
                params, 
                entry_price, 
                stop_loss, 
                take_profit, 
                horizon=100, 
                paths=500
            )
            
        except Exception as e:
            if self.DEBUG:
                print(f"[{self.name}] Ruin Guard Error: {e}")
            return 0.5

    def calculate_kelly_size(
        self, 
        balance: float, 
        win_rate: float = None, 
        risk_reward: float = None
    ) -> float:
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
        # Get win rate from Governor if not provided
        if win_rate is None:
            win_rate = self.governor.calculate_recent_win_rate()
            
        if risk_reward is None:
            risk_reward = getattr(config, 'DEFAULT_RISK_REWARD', 2.0)
            
        # Edge case handling
        if win_rate <= 0 or risk_reward <= 0:
            return balance * 0.01  # Minimum 1% position
            
        # Full Kelly = (p(b+1) - 1) / b
        # where p = win_rate, b = risk_reward
        p = win_rate
        b = risk_reward
        
        kelly_fraction = ((p * (b + 1)) - 1) / b
        
        # Half-Kelly for safety
        half_kelly = kelly_fraction * 0.5
        
        # Clamp to reasonable range (1% - 25% of balance)
        half_kelly = max(0.01, min(0.25, half_kelly))
        
        return balance * half_kelly
    
    # =========================================================================
    # RISK CHECKS
    # =========================================================================
    
    def check_cluster_risk(self, symbol: str) -> bool:
        """
        Refuse trade if we already hold an asset from the same family.
        Returns: False if RISK DETECTED (Reject), True if SAFE.
        Uses Rust Engine (holonic_speed) if available.
        """
        try:
            import holonic_speed
            # Get currently held symbols
            held_symbols = [s for s, d in self.governor.positions.items() 
                          if abs(d.get('quantity', 0)) > 0]
            result = holonic_speed.governor_check_cluster_risk(held_symbols, symbol)
            if not result:
                print(f"[{self.name}] CLUSTER RISK: Rejecting {symbol} (Same family as held)")
            return result
        except ImportError:
            # Fallback to Python
            
            # --- USER HEDGE OVERRIDE ---
            hedge_assets = getattr(config, 'BTC_HEDGE_ASSETS', [])
            if symbol in hedge_assets:
                return True
            # ---------------------------

            family = None
            if symbol in config.FAMILY_L1:
                family = config.FAMILY_L1
            elif symbol in config.FAMILY_PAYMENT:
                family = config.FAMILY_PAYMENT
            elif symbol in config.FAMILY_MEME:
                family = config.FAMILY_MEME
            
            if not family:
                return True
            
            for asset, data in self.governor.positions.items():
                if abs(data['quantity']) > 0 and asset in family and asset != symbol:
                    print(f"[{self.name}] CLUSTER RISK: Rejecting {symbol} (Already hold {asset})")
                    return False
        return True

    def check_leverage_risk(self, new_notional_value: float) -> bool:
        """
        Refuse trade if Total Notional Exposure > 10x Balance.
        """
        current_exposure = 0.0

        for sym, pos in self.governor.positions.items():
            # Duck-typed accessor for both dict and Position objects
            if isinstance(pos, dict):
                qty = abs(pos.get('quantity', 0))
                price = self.governor.latest_prices.get(sym, pos.get('entry_price', 0))
            else:
                qty = abs(getattr(pos, 'quantity', 0))
                price = self.governor.latest_prices.get(sym, getattr(pos, 'entry_price', 0))
            current_exposure += qty * price

        total_exposure = current_exposure + new_notional_value
        max_leverage = getattr(config, 'MAX_TOTAL_LEVERAGE', 10.0)
        max_exposure = self.governor.balance * max_leverage

        if total_exposure > max_exposure:
            print(f"[{self.name}] LEVERAGE RISK: Total ${total_exposure:.2f} > Max ${max_exposure:.2f}")
            return False

        return True
    
    # =========================================================================
    # SOLVENCY CHECKS
    # =========================================================================
    
    def calculate_portfolio_state(self) -> Dict[str, float]:
        """
        CROSS-MARGIN CALCULATOR (Safe Mode):
        Aggregates portfolio margin usage and available equity.
        Equity = Balance + Unrealized PnL
        Used Initial Margin = Sum(Position Value / Leverage)
        """
        total_upnl = 0.0
        used_margin = 0.0

        for sym, pos in self.governor.positions.items():
            # Duck-typed accessor for both dict and Position objects
            if isinstance(pos, dict):
                qty = abs(pos.get('quantity', 0))
                entry = pos.get('entry_price', 0)
                direction = pos.get('direction', 'BUY')
                leverage = pos.get('leverage', 1.0)
            else:
                qty = abs(getattr(pos, 'quantity', 0))
                entry = getattr(pos, 'entry_price', 0)
                direction = getattr(pos, 'direction', 'BUY')
                leverage = getattr(pos, 'leverage', 1.0)

            # Get current price
            current_price = self.governor.latest_prices.get(sym, entry)

            if entry > 0 and current_price > 0:
                # Calculate Unrealized PnL
                if direction == 'BUY':
                    upnl = (current_price - entry) * qty
                else:
                    upnl = (entry - current_price) * qty
                total_upnl += upnl
                
                # Calculate Used Margin (Initial Margin)
                notional = qty * current_price
                margin = notional / leverage
                used_margin += margin
                
        # Calculate Equity and Free Margin
        equity = self.governor.balance + total_upnl
        free_margin = equity - used_margin
        
        # Margin Level (> 1.0 is safe, < 1.0 is approaching liquidation)
        margin_level = equity / used_margin if used_margin > 0 else 999.0
        
        return {
            'equity': equity,
            'used_margin': used_margin,
            'free_margin': free_margin,
            'margin_level': margin_level,
            'unrealized_pnl': total_upnl
        }
    
    def check_solvency(self, trade_metadata: dict) -> bool:
        """
        PRE-FLIGHT CHECK: Simulate trade to ensure it doesn't break margin rules.
        Called by Executor immediately before locking the ledger.
        
        Returns True if trade is safe, False if it would break solvency.
        """
        symbol = trade_metadata.get('symbol', '')
        direction = trade_metadata.get('direction', 'BUY')
        quantity = abs(trade_metadata.get('quantity', 0))
        price = trade_metadata.get('price', 0)
        leverage = trade_metadata.get('leverage', 1.0)
        
        if quantity <= 0 or price <= 0:
            return False
            
        # 1. Calculate new margin requirement
        notional = quantity * price
        new_margin = notional / leverage
        
        # 2. Get current portfolio state
        state = self.calculate_portfolio_state()
        
        # 3. Check if we have enough free margin
        margin_buffer = getattr(config, 'MIN_MARGIN_BUFFER', 1.2)
        required_free = new_margin * margin_buffer
        
        if state['free_margin'] < required_free:
            print(f"[{self.name}] SOLVENCY FAIL: Need ${required_free:.2f}, have ${state['free_margin']:.2f}")
            return False
            
        # 4. Check post-trade margin level
        post_trade_margin = state['used_margin'] + new_margin
        post_trade_level = state['equity'] / post_trade_margin if post_trade_margin > 0 else 0
        
        min_margin_level = getattr(config, 'MIN_MARGIN_LEVEL', 1.5)
        if post_trade_level < min_margin_level:
            print(f"[{self.name}] SOLVENCY FAIL: Post-trade margin level {post_trade_level:.2f} < {min_margin_level}")
            return False
            
        return True
