"""
ExitGuardianHolon - The "Defense" Brain (Phase 16)

Specialized in:
1. Unrealized PnL Protection
2. Time-at-Risk Management
3. Volatility-based Trailing Stops
4. Monte Carlo-based Position Health Evaluation
"""

from typing import Any, Optional, Literal
from HolonicTrader.holon_core import Holon, Disposition
import config
import os
import json
from datetime import datetime, timezone
import random # For ML placeholder noise

# Import Monte Carlo Position Manager
try:
    from .monte_carlo_position_manager import MonteCarloPositionManager
    MONTE_CARLO_AVAILABLE = True
except ImportError:
    MONTE_CARLO_AVAILABLE = False
    print("[ExitGuardian] Monte Carlo Position Manager not available")

class ProfitOptimizer:
    """
    Phase 2: Heuristic Profit Optimizer (Pre-ML).
    Determines optimal exit profile (Aggressive/Balanced/Conservative) based on market state.
    """
    def __init__(self):
        self.history = []

    def predict_exit_profile(self, rsi: float, atr: float, pnl_pct: float) -> str:
        """
        Predict best exit profile.
        High RSI + High Profit -> Aggressive (Capture now)
        Low RSI + Profit -> Conservative (Let it run)
        """
        # score: 0-10 (10 = Most Aggressive)
        score = 5
        
        # PSI Pressure
        if rsi > 75: score += 3 # Overbought
        elif rsi > 65: score += 1
        elif rsi < 40: score -= 2 # Potential for more upside? or trend continuation
        
        # PnL Satisfaction
        if pnl_pct > 0.05: score += 2 # Deep profit, lock it in
        
        if score >= 8: return 'aggressive' # [30% @ rapid, 40% @ normal, 30% @ runner]
        if score <= 4: return 'conservative' # [10% @ rapid, 60% @ normal, 30% @ runner]
        return 'balanced'

class ExitGuardianHolon(Holon):
    def __init__(self, name: str = "ExitGuardian"):
        super().__init__(name=name, disposition=Disposition(autonomy=0.8, integration=0.4))
        self.last_exit_times = {} # {symbol: timestamp}
        self.trade_watermarks = {} # {symbol: {high: float, low: float}}
        self.exit_stages = {} # {symbol: int} 0=None, 1=Rapid, 2=Normal, 3=Runner
        self.exit_pending = set() # {symbol} — guard against duplicate trail-hit signals
        self.partial_exits = {} # {symbol: bool} — track partial exit state per symbol
        self.optimizer = ProfitOptimizer() # Phase 2 integration

        # Initialize Monte Carlo Position Manager
        self.monte_carlo_manager = None
        if MONTE_CARLO_AVAILABLE:
            try:
                self.monte_carlo_manager = MonteCarloPositionManager()
                print(f"[{self.name}] Monte Carlo Position Manager initialized")
            except Exception as e:
                print(f"[{self.name}] Failed to initialize Monte Carlo Position Manager: {e}")

        # Initialize trader reference to None (will be set by Trader)
        self.trader_ref = None

    def get_dynamic_trail_mult(self, pnl_pct: float, reason: str = 'DEFAULT') -> float:
        """
        Calculate ATR multiplier for trailing stop based on PnL depth and Strategy.
        'Scalp the Whales' Logic: Tighten stops as we go deeper into profit.

        ARB/CARRY/BASIS/FUNDING positions: Wide fixed trail (no progressive tightening).
        These positions are held for funding yield — not directional momentum — so they
        MUST survive intraday spikes without being swept by noise. XMR can move 5%+
        intraday while the funding thesis remains 100% valid.
        """
        # Normalize reason to detect strategy type
        reason_upper = (reason or '').upper()

        # ARB / Carry / Basis / Funding: Wide, FIXED trail — no progressive tightening
        # Using 5.0x ATR: on XMR ATR ~$6.4, this gives ~$32 trail buffer
        # which handles normal intraday volatility without false exits.
        IS_ARB_CARRY = (
            'ARB' in reason_upper or 'CARRY' in reason_upper or
            'BASIS' in reason_upper or 'FUNDING' in reason_upper
        )
        if IS_ARB_CARRY:
            return 5.0  # Wide fixed trail — no tightening for yield positions

        # Base Multipliers (ATR) for directional strategies
        base_mults = {
            'WHALE_BID_WALL': 1.5,     # Aggressive
            'WHALE_ACCUMULATION': 3.0, # Loose (Accumulation noise)
            'PACK_HUNT': 1.0,          # Tight (Momentum)
            'DIP': 2.0,                # Medium
            'DEFAULT': config.PREDATOR_TRAILING_STOP_ATR_MULT # Fallback (3.5)
        }
        
        # Normalize Reason key
        key = 'DEFAULT'
        if reason:
            if 'WHALE' in reason_upper and 'BID' in reason_upper: key = 'WHALE_BID_WALL'
            elif 'WHALE' in reason_upper: key = 'WHALE_ACCUMULATION'
            elif 'PACK' in reason_upper: key = 'PACK_HUNT'
            elif 'DIP' in reason_upper or 'REVERSION' in reason_upper: key = 'DIP'
            
        base = base_mults.get(key, 3.0)
        
        # Progressive Tightening (directional only — lock in profits as they grow)
        # FIX 2026-03-19 (Helix/Chronos): Old values (0.3x/0.5x/0.7x) strangled R:R.
        # Crypto noise wicks are 1-2 ATR. Trail < 1.0 ATR = guaranteed stop-out on noise.
        # New: minimum 1.0x ATR even at deep profit. Let winners reach TP.
        # FIX 2026-03-20: Widen multipliers further to reduce premature exits
        # Issue: Positions being stopped out before reaching TP due to noise
        if pnl_pct >= 0.05:    # > 5% Profit → moderately tight (was 0.3 = death sentence)
            return max(base * 0.8, 1.5)  # Widened: was 0.6x, now 0.8x with 1.5x floor
        elif pnl_pct >= 0.03:  # > 3% Profit
            return max(base * 0.9, 1.8)  # Widened: was 0.75x, now 0.9x with 1.8x floor
        elif pnl_pct >= 0.015: # > 1.5% Profit
            return max(base * 0.95, 2.0) # Widened: was 0.85x, now 0.95x with 2.0x floor
        else:
            # Deep loss or small profit - use base (already wide enough)
            return base

    def get_market_condition_multiplier(self, sentiment_score: float = 0.0, crisis_score: float = 0.0) -> float:
        """
        Adjust profit targets based on Crisis/Sentiment.
        Crisis -> Lower targets (Get out fast)
        Bullish -> Higher targets (Let it run)
        """
        mult = 1.0
        
        # Crisis Logic
        if crisis_score >= config.MARKET_ADJUSTMENTS['crisis_score']['critical']:
            mult *= config.MARKET_ADJUSTMENTS['crisis_score']['reduce_targets'] # e.g. 0.7
            
        # Sentiment Logic
        if sentiment_score < config.MARKET_ADJUSTMENTS['sentiment']['bearish']:
            mult *= 0.9 # Slightly easier targets in bear
        elif sentiment_score > config.MARKET_ADJUSTMENTS['sentiment']['bullish']:
            mult *= 1.1 # Extend targets in bull
            
        return mult

    def get_time_adjusted_targets(self):
        """
        Adjust profit targets based on Global Trading Session.
        Asian (Low Vol) -> Tighter Targets
        London/NY (High Vol) -> Wider Targets
        Weekend -> Ultra Tight
        """
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        hour = now.hour
        weekday = now.weekday() # 0=Mon, 6=Sun
        
        if weekday >= 5:
            return config.TIME_BASED_PROFIT_TARGETS['weekend']
        
        # Asian: 00-08 UTC (Approx)
        if 0 <= hour < 8:
            return config.TIME_BASED_PROFIT_TARGETS['asian_session']
        # London (Overlap): 08-16 UTC
        elif 8 <= hour < 16:
             return config.TIME_BASED_PROFIT_TARGETS['london_session']
        # NY: 16-24 UTC
        else:
             return config.TIME_BASED_PROFIT_TARGETS['ny_session']

    def check_profit_taking(self, symbol: str, pnl_pct: float, current_price: float, direction: str, reason: str = 'DEFAULT', rsi: float = 50.0, meta: dict = None) -> Optional['TradeSignal']:
        """
        Tiered Profit Taking Logic (Pyramid Exit).
        """
        # 1. Determine Strategy Key
        key = 'DEFAULT'
        if reason:
            if 'WHALE' in reason and 'BID' in reason: key = 'WHALE_BID_WALL'
            elif 'WHALE' in reason: key = 'WHALE_ACCUMULATION'
            elif 'PACK' in reason: key = 'PACK_HUNT'
            elif 'DIP' in reason: key = 'DIP'
            
        base_targets = config.PROFIT_TARGETS.get(key, config.PROFIT_TARGETS['DEFAULT'])
        
        # 2. Apply Time-Based Adjustments (Overlap Logic)
        time_targets = self.get_time_adjusted_targets()
        
        # 3. Apply Market Condition Multiplier (Phase 3)
        # We need sentiment/crisis. For now, assume neutral or fetch?
        # Ideally passed in `indicators` or `meta`.
        # Taking from `meta` if available, else neutral
        sent = meta.get('sentiment_score', 0.0) if meta else 0.0
        crisis = meta.get('crisis_score', 0.0) if meta else 0.0
        
        mkt_mult = self.get_market_condition_multiplier(sent, crisis)

        targets = {
            'rapid': ((base_targets['rapid'] + time_targets['rapid']) / 2) * mkt_mult,
            'normal': ((base_targets['normal'] + time_targets['normal']) / 2) * mkt_mult,
            'runner': ((base_targets['runner'] + time_targets['runner']) / 2) * mkt_mult
        }
        
        # 3. Determine Exit Profile via Optimizer
        # Uses RSI/ATR to decide if we should be Aggressive or Conservative
        atr_val = 0.0 # Context passed? Need to add ATR to arg list or context?
        # check_profit_taking arg list doesn't have ATR. Adding it.
        # But wait, I can just rely on defaults or pass it.
        # Let's use 'rsi' which is passed.
        
        profile = self.optimizer.predict_exit_profile(rsi, 0.0, pnl_pct)
        exits = config.EXIT_PYRAMID.get(profile, config.EXIT_PYRAMID['balanced'])
        
        current_stage = self.exit_stages.get(symbol, 0)
        signal = None
        new_stage = current_stage
        
        # STAGE 1: RAPID SCALP
        if current_stage < 1:
            # Trigger: PnL > Rapid Target OR (RSI > 80 and PnL > 0.3%)
            rapid_trigger = pnl_pct >= targets['rapid']
            
            # Immediate Scalp Trigger (RSI)
            if rsi >= config.IMMEDIATE_SCALP_CONFIG['rsi_overbought'] and pnl_pct >= config.IMMEDIATE_SCALP_CONFIG['profit_target']:
                 print(f"[{self.name}] ⚡ RSI SUPER-SCALP: {symbol} (RSI {rsi:.1f}, PnL {pnl_pct*100:.2f}%)")
                 rapid_trigger = True

            if rapid_trigger:
                print(f"[{self.name}] 💰 RAPID PROFIT (Stage 1): {symbol} (+{pnl_pct*100:.2f}%) -> Closing {exits[0]*100}%")
                from .agent_executor import TradeSignal
                # Close % using exits[0] (e.g. 0.3 for 30%)
                signal = TradeSignal(symbol, 'SELL' if direction == 'BUY' else 'BUY', size=exits[0], price=current_price, 
                                     metadata={'reason': 'RAPID_TP', 'is_percent': True})
                new_stage = 1

        # STAGE 2: NORMAL TARGET
        elif current_stage < 2:
            if pnl_pct >= targets['normal']:
                print(f"[{self.name}] 💰 NORMAL PROFIT (Stage 2): {symbol} (+{pnl_pct*100:.2f}%) -> Closing {exits[1]*100}%")
                from .agent_executor import TradeSignal
                signal = TradeSignal(symbol, 'SELL' if direction == 'BUY' else 'BUY', size=exits[1], price=current_price, 
                                     metadata={'reason': 'NORMAL_TP', 'is_percent': True})
                new_stage = 2
                
        # STAGE 3: RUNNER (Managed by Trail, but we can have a Hard Target too)
        elif current_stage < 3:
            if pnl_pct >= targets['runner']:
                print(f"[{self.name}] 🚀 RUNNER TARGET (Stage 3): {symbol} (+{pnl_pct*100:.2f}%) -> Closing Remainder")
                from .agent_executor import TradeSignal
                signal = TradeSignal(symbol, 'SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price, 
                                     metadata={'reason': 'RUNNER_TP', 'is_percent': True})
                new_stage = 3 # Done

        if new_stage != current_stage:
            self.exit_stages[symbol] = new_stage
            
        return signal

    def update_watermark(self, symbol: str, current_price: float, entry_price: float = None):
        """Update the High/Low watermark for true trailing."""
        if symbol not in self.trade_watermarks:
            # Initialize with Entry Price comparison to ensure we capture the full range
            high = max(current_price, entry_price) if entry_price else current_price
            low = min(current_price, entry_price) if entry_price else current_price
            self.trade_watermarks[symbol] = {'high': high, 'low': low}
        else:
            if current_price > self.trade_watermarks[symbol]['high']:
                self.trade_watermarks[symbol]['high'] = current_price
            if current_price < self.trade_watermarks[symbol]['low']:
                self.trade_watermarks[symbol]['low'] = current_price

    def reset_watermark(self, symbol: str):
         if symbol in self.trade_watermarks:
             del self.trade_watermarks[symbol]
         if symbol in self.exit_stages:
             del self.exit_stages[symbol]
         self.exit_pending.discard(symbol)  # FIX: clear pending on full reset

    def adjust_exit_strategy_for_management_mode(self, symbol: str, current_price: float, entry_price: float, direction: str, pnl_pct: float, position_age_hours: float):
        """
        Adjust exit strategy when in management mode to optimize existing positions.
        """
        # This method is called from analyze_for_exit, which has access to the trader context
        # We'll need to receive the governor reference as a parameter or access it differently
        # For now, we'll implement a simpler version that can be enhanced later

        # In management mode, adjust exit strategy based on position profitability
        # FIX 2026-03-19 (Helix): Raised from 2%→4%. Old 2% was within noise range and
        # killed R:R by banking profits on trades that should run to 6% TP.
        if pnl_pct > 0.04:  # If position is profitable (>4%, was 2%)
            # Be more aggressive with profit taking to secure gains
            print(f"[{self.name}] 🛠️ MANAGEMENT MODE: Taking profit on {symbol} ({pnl_pct*100:.2f}%)")
            from .agent_executor import TradeSignal
            return TradeSignal(
                symbol=symbol,
                direction='SELL' if direction == 'BUY' else 'BUY',
                size=0.3,  # Take partial profit (30%)
                price=current_price,
                metadata={
                    'reason': 'MANAGEMENT_MODE_PROFIT_TAKE',
                    'is_percent': True
                }
            )
        elif pnl_pct < -0.03 and position_age_hours > 2:  # Losing >3% for >2 hours
            # Close losing positions more quickly to prevent further losses
            print(f"[{self.name}] 🛠️ MANAGEMENT MODE: Closing losing position {symbol} ({pnl_pct*100:.2f}%)")
            from .agent_executor import TradeSignal
            return TradeSignal(
                symbol=symbol,
                direction='SELL' if direction == 'BUY' else 'BUY',
                size=1.0,  # Close all
                price=current_price,
                metadata={
                    'reason': 'MANAGEMENT_MODE_LOSS_CUT',
                    'is_percent': True
                }
            )

        return None  # Use normal exit logic

    def evaluate_losing_position_monte_carlo(self,
                                           symbol: str,
                                           current_price: float,
                                           entry_price: float,
                                           direction: Literal['BUY', 'SELL'],
                                           position_age_hours: float,
                                           sde_params: dict = None,
                                           pnl_pct: float = 0.0) -> Optional['TradeSignal']:
        """
        Use Monte Carlo simulation to evaluate if a losing position should be closed.
        """
        if not self.monte_carlo_manager:
            return None

        try:
            # Prepare SDE parameters
            if not sde_params:
                sde_params = {
                    'mu': 0.0,
                    'sigma': 0.1,
                    'lambda': 0.1
                }

            # Evaluate position using Monte Carlo
            result = self.monte_carlo_manager.evaluate_position_for_closure(
                symbol=symbol,
                current_price=current_price,
                entry_price=entry_price,
                direction=direction,
                position_age_hours=position_age_hours,
                sde_params=sde_params,
                pnl_pct=pnl_pct
            )

            # Check if result is valid
            if result is None:
                return None

            # Unpack the result with safety checks
            if isinstance(result, tuple) and len(result) >= 3:
                should_close, confidence, reason = result[0], result[1], result[2]
            else:
                return None

            if should_close and confidence and confidence > 0.6:  # Only act on high confidence signals
                print(f"[{self.name}] Monte Carlo Closure Signal: {symbol} - {reason} (Conf: {confidence:.2%})")
                from .agent_executor import TradeSignal
                return TradeSignal(
                    symbol=symbol,
                    direction='SELL' if direction == 'BUY' else 'BUY',
                    size=1.0,
                    price=current_price,
                    metadata={
                        'reason': 'MONTE_CARLO_CLOSURE',
                        'is_percent': True,
                        'confidence': confidence,
                        'monte_carlo_reason': reason
                    }
                )

        except Exception as e:
            print(f"[{self.name}] Monte Carlo evaluation error for {symbol}: {e}")
            import traceback
            traceback.print_exc()

        return None

    def _load_genome(self):
        """Load the latest evolved parameters from disk."""
        try:
            path = os.path.join(os.getcwd(), 'live_genome.json')
            if os.path.exists(path):
                with open(path, 'r') as f:
                     return json.load(f)
        except Exception:
            pass # Silent fail is fine, use defaults
        return None

    def manage_satellite_positions(self, symbol: str, current_price: float, entry_price: float, direction: Literal['BUY', 'SELL']):
        """
        Hit & Run Management for Satellite Assets.
        Breakeven at +1.5%, Take Profit 50% at +3%.
        """
        from .agent_executor import TradeSignal
        
        if entry_price <= 0: return None
        
        # Update Watermark
        self.update_watermark(symbol, current_price, entry_price)
        
        # PnL Calculation
        if direction == 'BUY':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
            
        # 1. Breakeven Trigger (Move SL to Entry + 0.1%)
        # Logic: If we are > 1.5% profit, we assume the actuator moves the SL.
        # But here, we simulate the "Close" if price drops back.
        # Ideally, actuator handles hard stops. Guardian handles 'decisions'.

        pnl_pct = (current_price - entry_price) / entry_price if direction == 'BUY' else (entry_price - current_price) / entry_price
        
        # Load Config
        genome = self._load_genome()
        if genome:
             stop_limit = genome.get('satellite_stop_loss', config.SATELLITE_STOP_LOSS)
             tp_target = genome.get('satellite_take_profit', config.SATELLITE_TAKE_PROFIT_1)
        else:
             stop_limit = config.SATELLITE_STOP_LOSS
             tp_target = config.SATELLITE_TAKE_PROFIT_1
             
        # 1. Take Profit (Sniper Moonshot)
        if pnl_pct >= tp_target:
             print(f"[{self.name}] 🎯 SATELLITE SNIPER HIT: {symbol} (+{pnl_pct*100:.2f}%)")
             self.reset_watermark(symbol)
             return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price, metadata={'reason': 'SATELLITE_TP', 'is_percent': True})

        # 2. Breakeven (Safety) - Static for now, or could evolve
        # be_trigger = config.SATELLITE_BREAKEVEN_TRIGGER
        # ... logic for BE is complex state, keeping simple for now ...
             
        # 3. Hard Stop (Genome Evolved)
        # Note: stop_limit is positive number (e.g. 0.05), so we check <= -0.05
        if pnl_pct <= -stop_limit:
             print(f"[{self.name}] 💥 SATELLITE STOP LOSS (EVO): {symbol} ({pnl_pct*100:.2f}%)")
             self.reset_watermark(symbol)
             return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price, metadata={'reason': 'SATELLITE_SL', 'is_percent': True})
             
        # 4. RSI Exit? (Optional, if in genome)
        # genome['rsi_sell'] check would require passing Indicators to Guardian.
        # For now, we rely on PnL exits mainly for robustness.
        
        return None

    def analyze_for_exit(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        bb: dict,
        atr: float,
        metabolism_state: Literal['SCAVENGER', 'PREDATOR'],
        position_age_hours: float = 0.0,
        direction: Literal['BUY', 'SELL'] = 'BUY',
        regime: str = 'MICRO',
        indicators: dict = None,
        meta: dict = None
    ):
        
        if symbol in config.SATELLITE_ASSETS:
            return self.manage_satellite_positions(symbol, current_price, entry_price, direction)
        # -----------------------------------

        from .agent_executor import TradeSignal
        
        if entry_price <= 0:
            return None
            
        # Update Watermark (Initialize with Entry if needed)
        self.update_watermark(symbol, current_price, entry_price)
            
        # PnL Calculation
        if direction == 'BUY':
            pnl_pct = (current_price - entry_price) / entry_price
        else: # SELL (Short)
            pnl_pct = (entry_price - current_price) / entry_price
            
        # --- VOL-WINDOW EXIT LOGIC ---
        if regime == 'VOL_WINDOW':
            # Tighter trailing stop: 1.5 ATR (High Turnover)
            if atr > 0:
                watermark = self.trade_watermarks.get(symbol, {'high': current_price, 'low': current_price})
                trail_dist = atr * 1.5 
                
                if direction == 'BUY':
                    stop_price = watermark['high'] - trail_dist
                    if current_price <= stop_price:
                         print(f"[{self.name}] ⚡ VOL_WINDOW TRAIL HIT: {symbol} @ {current_price:.4f} (High {watermark['high']:.4f})")
                         self.reset_watermark(symbol)
                         return TradeSignal(symbol=symbol, direction='SELL', size=1.0, price=current_price, metadata={'reason': 'VOL_TRAIL', 'is_percent': True})
                else: # SELL
                    stop_price = watermark['low'] + trail_dist
                    if current_price >= stop_price:
                         print(f"[{self.name}] ⚡ VOL_WINDOW TRAIL HIT: {symbol} @ {current_price:.4f} (Low {watermark['low']:.4f})")
                         self.reset_watermark(symbol)
                         return TradeSignal(symbol=symbol, direction='BUY', size=1.0, price=current_price, metadata={'reason': 'VOL_TRAIL', 'is_percent': True})
            
            # Hard Stop (1%)
            if pnl_pct <= -0.01:
                 print(f"[{self.name}] ⚡ VOL_WINDOW HARD STOP: {symbol} ({pnl_pct*100:.2f}%)")
                 self.reset_watermark(symbol)
                 return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price, metadata={'reason': 'VOL_STOP', 'is_percent': True})
                 
            return None # Skip standard logic
        # -----------------------------

        
        # 1. HARD STOP LOSS (Circuit Breaker)
        # FIX 2026-02-28: ARB / Carry / Basis positions held for funding yield need a
        # WIDER emergency stop. XMR, PAXG, BNB, TAO can see 5-8% intraday spikes while
        # the funding thesis (we collect 0.07%/8h) remains completely valid. Using the
        # standard PREDATOR_STOP_LOSS (~2%) causes constant false emergency exits.
        # ARB positions use 8% hard stop — beyond this, something is truly wrong
        # (e.g. exchange crisis, black swan, or delisting) and we should exit.
        _meta_reason = (meta or {}).get('reason', '') if meta else ''
        _meta_strat = (meta or {}).get('strategy', '') if meta else ''
        _is_arb_carry = (
            'ARB' in str(_meta_reason).upper() or 'CARRY' in str(_meta_reason).upper() or
            'BASIS' in str(_meta_reason).upper() or 'FUNDING' in str(_meta_reason).upper() or
            'ARB' in str(_meta_strat).upper() or 'CARRY' in str(_meta_strat).upper()
        )
        if _is_arb_carry:
            sl_target = 0.08  # 8% wide stop for funding-carry positions (intraday noise buffer)
        else:
            sl_target = config.SCAVENGER_STOP_LOSS if metabolism_state == 'SCAVENGER' else config.PREDATOR_STOP_LOSS
        if pnl_pct <= -sl_target:
            print(f"[{self.name}] 🚨 EMERGENCY EXIT: {symbol} ({direction}) PnL {pnl_pct*100:.2f}%")
            self.reset_watermark(symbol)
            return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price, metadata={'reason': 'EMERGENCY_STOP', 'is_percent': True})

        _projected_yield_apy = meta.get('projected_yield_apy', 0.0) if meta else 0.0

        # FIX 2026-03-20: Regime-aware time exits to improve capital efficiency
        # Prevents positions from being held for hours with no realization
        REGIME_TIME_LIMITS = {
            'VOLATILE': 0.25,      # 15 minutes - very fast exit
            'MEAN_REVERTING': 0.5, # 30 minutes - quick scalps
            'TRANSITION': 1.0,     # 60 minutes - moderate
            'TRENDING': 2.0,       # 120 minutes - let winners run
        }
        
        if metabolism_state == 'SCAVENGER':
            # Scavenger Exits: Quick Mean Reversion
            # FIX 2026-03-20: Reduced from 12h to regime-based limits
            time_limit = REGIME_TIME_LIMITS.get(regime, 1.0)
            
            # Adjust for projected yield (funding rate considerations)
            if _projected_yield_apy > 50.0: 
                time_limit = min(time_limit * 2, 4.0)  # Paid to hold, but cap at 4h
            elif _projected_yield_apy < -50.0: 
                time_limit = max(time_limit * 0.5, 0.15)  # Toxic: GET OUT faster

            if position_age_hours >= time_limit:
                print(f"[{self.name}] ⏳ TIME EXIT: {symbol} ({time_limit}h reached. Yield: {_projected_yield_apy:.1f}%, Regime: {regime})")
                self.reset_watermark(symbol)
                return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price, metadata={'reason': 'TIME_EXIT', 'is_percent': True})
            
            if pnl_pct >= config.SCAVENGER_SCALP_TP:
                if not self.partial_exits.get(symbol, False):
                    print(f"[{self.name}] 💰 SCALP-TO-PYRAMID: {symbol} (+{pnl_pct*100:.2f}%) -> Banking 50% House Money")
                    self.partial_exits[symbol] = True
                    # Do NOT reset watermark - let the runner trail
                    return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=0.5, price=current_price, metadata={'reason': 'PARTIAL_TP', 'is_percent': True})
                else:
                    # Already scalped - Let it ride! (Handled by Trailing Stop or Time Exit)
                    pass
                
            # Mean Reversion: Hit BB Middle
            if direction == 'BUY' and current_price >= bb['middle']:
                print(f"[{self.name}] 🔄 MEAN REVERSION (Long): {symbol} hit BB Middle")
                self.reset_watermark(symbol)
                return TradeSignal(symbol=symbol, direction='SELL', size=1.0, price=current_price, metadata={'reason': 'MEAN_REV', 'is_percent': True})
            elif direction == 'SELL' and current_price <= bb['middle']:
                print(f"[{self.name}] 🔄 MEAN REVERSION (Short): {symbol} hit BB Middle")
                self.reset_watermark(symbol)
                return TradeSignal(symbol=symbol, direction='BUY', size=1.0, price=current_price, metadata={'reason': 'MEAN_REV', 'is_percent': True})

        else: # PREDATOR
            # FIX 2026-03-20: Regime-aware time exits for Predator mode
            # Prevents positions from being held for 24+ hours
            time_limit = REGIME_TIME_LIMITS.get(regime, 1.0)
            
            # Adjust for projected yield (funding rate considerations)
            if _projected_yield_apy > 50.0: 
                time_limit = min(time_limit * 2, 6.0)  # Paid to hold (Extend Trend), cap 6h
            elif _projected_yield_apy < -50.0: 
                time_limit = max(time_limit * 0.5, 0.15)  # Toxic: GET OUT faster

            if position_age_hours >= time_limit:
                print(f"[{self.name}] ⏳ TREND EXPIRY: {symbol} ({time_limit}h reached. Yield: {_projected_yield_apy:.1f}%, Regime: {regime})")
                self.reset_watermark(symbol)
                return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price, metadata={'reason': 'TREND_EXPIRY', 'is_percent': True})

            # FIX 2026-03-20: Partial profit target at +0.5% (bank some profit, let rest run)
            # This improves expectancy by realizing some gains early while keeping upside exposure
            PARTIAL_TP_PCT = 0.005  # +0.5%
            if pnl_pct >= PARTIAL_TP_PCT:
                if not self.partial_exits.get(symbol, False):
                    print(f"[{self.name}] 💰 PARTIAL PROFIT: {symbol} (+{pnl_pct*100:.2f}%) -> Banking 50%, trailing rest")
                    self.partial_exits[symbol] = True
                    # Do NOT reset watermark - let the runner trail
                    return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=0.5, price=current_price, metadata={'reason': 'PARTIAL_TP', 'is_percent': True})
                # else: Already scalped - let the runner ride (handled by trailing stop or full TP)

            if pnl_pct >= config.PREDATOR_TAKE_PROFIT:
                print(f"[{self.name}] 💰 PREDATOR TP: {symbol} (+{pnl_pct*100:.2f}%)")
                self.reset_watermark(symbol)
                return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price, metadata={'reason': 'TAKE_PROFIT', 'is_percent': True})
            
            # --- RAPID PROFIT TAKING (SCALP THE WHALES) ---
            reason = meta.get('reason') if meta else 'DEFAULT'
            rsi_val = indicators.get('rsi', 50.0) if indicators else 50.0
            
            profit_signal = self.check_profit_taking(symbol, pnl_pct, current_price, direction, reason, rsi_val, meta=meta)
            if profit_signal:
                return profit_signal
            # ----------------------------------------------

            # --- PROFIT PRESERVATION (Half-Breach Rule) ---
            # If we reached meaningful profit, do not let it collapse back to noise.
            # Tuned to avoid strangling expectancy on small +0.5% swings.
            watermark = self.trade_watermarks.get(symbol, {'high': current_price, 'low': current_price})
            
            if direction == 'BUY':
                max_price = watermark['high']
                max_pnl_pct = (max_price - entry_price) / entry_price
            else:
                min_price = watermark['low']
                max_pnl_pct = (entry_price - min_price) / entry_price
                
            min_peak = float(getattr(config, 'HALF_BREACH_MIN_PEAK_PCT', 0.02))   # default 2% peak before activating
            retain_frac = float(getattr(config, 'HALF_BREACH_RETAIN_FRACTION', 0.35))  # keep 35% of peak, allow 65% giveback
            min_lock = float(getattr(config, 'HALF_BREACH_MIN_LOCK_PNL_PCT', 0.005))  # don't fire if we're basically flat

            if max_pnl_pct >= min_peak:  # Qualified for protection
                breach_threshold = max_pnl_pct * retain_frac
                if pnl_pct >= 0 and pnl_pct <= breach_threshold and pnl_pct >= min_lock:
                    print(f"[{self.name}] 🛡️ HALF-BREACH PROTECTION: {symbol} Dropped from +{max_pnl_pct*100:.2f}% to +{pnl_pct*100:.2f}% (Peak giveback trigger)")
                    # Reset stages so we don't double signal? No, this is a full close.
                    self.reset_watermark(symbol)
                    return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price, metadata={'reason': 'PROFIT_PRESERVE', 'is_percent': True})
            # ----------------------------------------------

            # --- ADAPTIVE TRAILING STOP (Dynamic Multiplier) ---
            # FIX 2026-02-28: Use .get() to prevent KeyError if watermark not initialized
            if atr > 0:
                watermark = self.trade_watermarks.get(symbol, {'high': current_price, 'low': current_price})

                # Dynamic Multiplier based on PnL depth
                atr_mult = self.get_dynamic_trail_mult(pnl_pct, reason)
                trail_dist = atr * atr_mult

                if direction == 'BUY':
                    # Trail from Highest High
                    highest = watermark['high']
                    trailing_stop = highest - trail_dist

                    if current_price <= trailing_stop:
                        # FIX-1: Guard against duplicate trail-hit signals while executor fills the order
                        if symbol in self.exit_pending:
                            pass  # Already signalled — executor hasn't cleared it yet, skip
                        else:
                            print(f"[{self.name}] 🛡️ DYNAMIC TRAIL HIT (Long): {symbol} @ {current_price:.8f} (High: {highest:.8f}, Trail: {trail_dist:.8f}, Mult: {atr_mult:.1f}x)")
                            self.exit_pending.add(symbol)
                            # FIX 2026-03-06: Do NOT reset_watermark here — record_exit() handles
                            # cleanup after the order confirms. Premature reset caused the watermark
                            # to re-initialize next cycle, firing a new trail signal every ~60s.
                            return TradeSignal(symbol=symbol, direction='SELL', size=1.0, price=current_price, metadata={'reason': 'ADAPTIVE_STOP', 'is_percent': True})

                else: # SELL
                    # Trail from Lowest Low
                    lowest = watermark['low']
                    trailing_stop = lowest + trail_dist

                    if current_price >= trailing_stop:
                        # FIX-1: Guard against duplicate trail-hit signals while executor fills the order
                        if symbol in self.exit_pending:
                            pass  # Already signalled — executor hasn't cleared it yet, skip
                        else:
                            print(f"[{self.name}] 🛡️ DYNAMIC TRAIL HIT (Short): {symbol} @ {current_price:.4f} (Low: {lowest:.4f}, Trail: {trail_dist:.4f}, Mult: {atr_mult:.1f}x)")
                            self.exit_pending.add(symbol)
                            # FIX 2026-03-06: Do NOT reset_watermark here — record_exit() handles
                            # cleanup after the order confirms. Premature reset caused the watermark
                            # to re-initialize next cycle, firing a new trail signal every ~60s.
                            return TradeSignal(symbol=symbol, direction='BUY', size=1.0, price=current_price, metadata={'reason': 'ADAPTIVE_STOP', 'is_percent': True})

        # --- MANAGEMENT MODE ADJUSTMENT ---
        # If in management mode, adjust exit strategy for existing positions
        # Check if we're in management mode by accessing the governor through the trader context
        governor = None
        if hasattr(self, 'trader_ref') and self.trader_ref:
            governor = self.trader_ref.sub_holons.get('governor')

        # If we have a governor and it's in management mode, apply adjustments
        if governor:
            try:
                if governor.is_in_management_mode():
                    management_signal = self.adjust_exit_strategy_for_management_mode(
                        symbol, current_price, entry_price, direction, pnl_pct, position_age_hours
                    )
                    if management_signal:
                        return management_signal
            except AttributeError:
                # Governor might not have management mode methods if not properly initialized
                pass
        # ---------------------------------

        # --- MONTE CARLO EVALUATION FOR LOSING POSITIONS ---
        # If we reach here and the position is losing, run Monte Carlo evaluation
        # WHALE OVERRIDE: Whale setups get more breathing room before MC panic
        reason_upper = (reason or '').upper()
        is_whale = 'WHALE' in reason_upper
        # FIX 2026-03-15 (Chronos v2): Raised MC threshold from -0.5% to -1.5% for normal signals.
        # Old -0.5% fired within normal noise range, crystallizing recoverable drawdowns.
        # Whale threshold raised 1.5%→2.5% to give thesis-driven positions more room.
        mc_threshold = -0.025 if is_whale else -0.015
        
        if pnl_pct < mc_threshold:  # Position is losing more than threshold (Whale = 1.5%, Normal = 0.5%)
            # Try to get SDE parameters from meta if available
            sde_params = meta.get('sde_physics', {}) if meta else {}

            monte_carlo_signal = self.evaluate_losing_position_monte_carlo(
                symbol=symbol,
                current_price=current_price,
                entry_price=entry_price,
                direction=direction,
                position_age_hours=position_age_hours,
                sde_params=sde_params,
                pnl_pct=pnl_pct
            )

            if monte_carlo_signal:
                print(f"[{self.name}] 🎲 MONTE CARLO CLOSURE: {symbol} (PnL: {pnl_pct*100:.2f}%)")
                return monte_carlo_signal
        # -----------------------------------------------

        return None


    def check_liquidity_health(self, symbol: str, direction: Literal['BUY', 'SELL'], quantity: float, order_book: dict) -> str:
        """
        Analyze order book depth to ensure safe exit.
        direction: 'SELL' to exit Long, 'BUY' to cover Short.
        """
        if quantity <= 0 or not order_book: return "UNKNOWN"

        # If exiting Long (SELL), we look at BIDS (buyers)
        # If covering Short (BUY), we look at ASKS (sellers)
        side = 'bids' if direction == 'SELL' else 'asks'
        book_side = order_book.get(side, [])

        if not book_side:
            # Log more detail about what went wrong
            logger.warning(f"[{self.name}] Liquidity check: Empty {side} for {symbol} (qty={quantity})")
            return "NO_LIQUIDITY"

        # Calculate cumulative liquidity within ~2% slippage (relaxed from 0.5%)
        best_price = book_side[0][0]
        slippage_limit = best_price * 0.02  # 2% tolerance for thin books

        available_vol = 0.0
        weighted_price_sum = 0.0

        for price, vol in book_side:
            if direction == 'SELL':
                if price < (best_price - slippage_limit): break
            else:
                if price > (best_price + slippage_limit): break

            available_vol += vol
            weighted_price_sum += (price * vol)

            if available_vol >= quantity * 1.5: # Found enough + 50% buffer
                break

        if available_vol < quantity:
            return "CRITICAL_ILLIQUIDITY" # Can't fill without massive slippage
        elif available_vol < quantity * 1.5:
            return "WARNING_THIN_BOOK"

        return "HEALTHY"

    def record_exit(self, symbol: str, timestamp: Any):
        self.last_exit_times[symbol] = timestamp
        self.exit_pending.discard(symbol)  # FIX-1: Clear pending guard on confirmed exit
        # Clean up watermarks
        if symbol in self.trade_watermarks:
            del self.trade_watermarks[symbol]

    def get_health(self) -> dict:
        return {
            'status': 'OK',
            'exits_tracked': len(self.last_exit_times)
        }

    def receive_message(self, sender: Any, content: Any) -> None:
        pass
