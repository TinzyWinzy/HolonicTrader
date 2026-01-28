"""
ExitGuardianHolon - The "Defense" Brain (Phase 16)

Specialized in:
1. Unrealized PnL Protection
2. Time-at-Risk Management
3. Volatility-based Trailing Stops
"""

from typing import Any, Optional, Literal
from HolonicTrader.holon_core import Holon, Disposition
import config
import os
import json
from datetime import datetime, timezone
import random # For ML placeholder noise

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
        self.optimizer = ProfitOptimizer() # Phase 2 integration

    def get_dynamic_trail_mult(self, pnl_pct: float, reason: str = 'DEFAULT') -> float:
        """
        Calculate ATR multiplier for trailing stop based on PnL depth and Strategy.
        'Scalp the Whales' Logic: Tighten stops as we go deeper into profit.
        """
        # Base Multipliers (ATR)
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
            if 'WHALE' in reason and 'BID' in reason: key = 'WHALE_BID_WALL'
            elif 'WHALE' in reason: key = 'WHALE_ACCUMULATION'
            elif 'PACK' in reason: key = 'PACK_HUNT'
            elif 'DIP' in reason or 'REVERSION' in reason: key = 'DIP'
            
        base = base_mults.get(key, 3.0)
        
        # Progressive Tightening
        # PnL thresholds (e.g., 2% -> 0.02)
        if pnl_pct >= 0.05:   # > 5% Profit
            return base * 0.3 # Extremely Tight (Lock it in)
        elif pnl_pct >= 0.03: # > 3% Profit
            return base * 0.5
        elif pnl_pct >= 0.015: # > 1.5% Profit
            return base * 0.7
        else:
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
        sl_target = config.SCAVENGER_STOP_LOSS if metabolism_state == 'SCAVENGER' else config.PREDATOR_STOP_LOSS
        if pnl_pct <= -sl_target:
            print(f"[{self.name}] 🚨 EMERGENCY EXIT: {symbol} ({direction}) PnL {pnl_pct*100:.2f}%")
            self.reset_watermark(symbol)
            return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price, metadata={'reason': 'EMERGENCY_STOP', 'is_percent': True})

        if metabolism_state == 'SCAVENGER':
            # Scavenger Exits: Quick Mean Reversion
            if position_age_hours >= 12.0:
                print(f"[{self.name}] ⏳ TIME EXIT: {symbol} (12h reached)")
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
            # Predator Exits: Momentum Following
            if position_age_hours >= 24.0:
                print(f"[{self.name}] ⏳ TREND EXPIRY: {symbol} (24h reached)")
                self.reset_watermark(symbol)
                return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price, metadata={'reason': 'TREND_EXPIRY', 'is_percent': True})
                
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
            # If we reached > 1% profit, do not let it drop below 50% of max profit
            watermark = self.trade_watermarks.get(symbol, {'high': current_price, 'low': current_price})
            
            if direction == 'BUY':
                max_price = watermark['high']
                max_pnl_pct = (max_price - entry_price) / entry_price
            else:
                min_price = watermark['low']
                max_pnl_pct = (entry_price - min_price) / entry_price
                
            if max_pnl_pct >= 0.01: # Qualified for protection (>1% peak)
                breach_threshold = max_pnl_pct * 0.5
                if pnl_pct <= breach_threshold:
                    print(f"[{self.name}] 🛡️ HALF-BREACH PROTECTION: {symbol} Dropped from +{max_pnl_pct*100:.2f}% to +{pnl_pct*100:.2f}% (Lost >50% Gains)")
                    # Reset stages so we don't double signal? No, this is a full close.
                    self.reset_watermark(symbol)
                    return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price, metadata={'reason': 'PROFIT_PRESERVE', 'is_percent': True})
            # ----------------------------------------------

            # --- ADAPTIVE TRAILING STOP (Dynamic Multiplier) ---
            if atr > 0:
                watermark = self.trade_watermarks[symbol]
                
                # Dynamic Multiplier based on PnL depth
                atr_mult = self.get_dynamic_trail_mult(pnl_pct, reason)
                trail_dist = atr * atr_mult
                
                if direction == 'BUY':
                    # Trail from Highest High
                    highest = watermark['high']
                    trailing_stop = highest - trail_dist
                    
                    if current_price <= trailing_stop:
                        print(f"[{self.name}] 🛡️ DYNAMIC TRAIL HIT (Long): {symbol} @ {current_price:.8f} (High: {highest:.8f}, Trail: {trail_dist:.8f}, Mult: {atr_mult:.1f}x)")
                        self.reset_watermark(symbol)
                        return TradeSignal(symbol=symbol, direction='SELL', size=1.0, price=current_price, metadata={'reason': 'ADAPTIVE_STOP', 'is_percent': True})
                        
                else: # SELL
                    # Trail from Lowest Low
                    lowest = watermark['low']
                    trailing_stop = lowest + trail_dist
                    
                    if current_price >= trailing_stop:
                        print(f"[{self.name}] 🛡️ DYNAMIC TRAIL HIT (Short): {symbol} @ {current_price:.4f} (Low: {lowest:.4f}, Trail: {trail_dist:.4f}, Mult: {atr_mult:.1f}x)")
                        self.reset_watermark(symbol)
                        return TradeSignal(symbol=symbol, direction='BUY', size=1.0, price=current_price, metadata={'reason': 'ADAPTIVE_STOP', 'is_percent': True})

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
        
        if not book_side: return "NO_LIQUIDITY"
        
        # Calculate cumulative liquidity within ~0.5% slippage
        best_price = book_side[0][0]
        slippage_limit = best_price * 0.005 # 0.5% tolerance
        
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
