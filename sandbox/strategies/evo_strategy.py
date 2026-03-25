from .base import Strategy, Signal
import pandas as pd
import numpy as np
import config

class EvoStrategy(Strategy):
    """
    Evolutionary Strategy.
    Behavior is defined by the 'genome' dictionary.
    """
    def __init__(self, genome: dict):
        unique_name = f"Evo-{str(hash(frozenset(genome.items())))[:6]}"
        super().__init__(name=unique_name)
        self.genome = genome
        
        # Unpack Genome
        self.rsi_buy = genome.get('rsi_buy', 30)
        self.rsi_sell = genome.get('rsi_sell', 70)
        self.bb_dev = genome.get('bb_dev', 2.0) # Not used directly if indicators pre-calced with 2.0
        self.stop_loss = genome.get('stop_loss', 0.02)
        self.take_profit = genome.get('take_profit', 0.05)
        
        # Satellite Genes
        self.use_satellite = genome.get('use_satellite', False)
        self.sat_rvol = genome.get('sat_rvol', 1.2)
        self.sat_bb_expand = genome.get('sat_bb_expand', 0.0)

        # Trailing Stop Genes
        self.trailing_activation = genome.get('trailing_activation', 0.5) # Activation R-multiple
        self.trailing_distance = genome.get('trailing_distance', 0.02)    # Distance from peak
        
        self.peak_price = 0.0
        self.trail_active = False

    def on_candle(self, slice_df, indicators, portfolio_state, secondary_slice_df=None) -> Signal:
        inventory = portfolio_state['inventory']
        price = indicators['price']
        rsi = indicators['rsi']
        
        # EXIT LOGIC
        if inventory > 0:
            avg_entry = portfolio_state['avg_entry']
            
            # SL
            if price < avg_entry * (1 - self.stop_loss):
                return Signal('SELL', reason="Stop Loss (Gene)")
            
            # TP
            if price > avg_entry * (1 + self.take_profit):
                return Signal('SELL', reason="Take Profit (Gene)")
            
            # --- TRAILING STOP LOGIC (SYNC WITH RUST) ---
            # 1. Activation
            activation_price = avg_entry * (1 + (self.stop_loss * self.trailing_activation))
            if not self.trail_active and price >= activation_price:
                self.trail_active = True
                self.peak_price = price
            
            # 2. Tracking
            if self.trail_active:
                if price > self.peak_price:
                    self.peak_price = price
                
                trail_stop_price = self.peak_price * (1 - self.trailing_distance)
                if price <= trail_stop_price:
                    return Signal('SELL', reason="Trailing Stop (Gene)")
            # --------------------------------------------

            # Indicator Exit
            if rsi > self.rsi_sell:
                self.trail_active = False # Reset for next
                return Signal('SELL', reason="RSI Overbought")
                
            return Signal('HOLD')
        else:
            # Entry / Reset
            self.peak_price = 0.0
            self.trail_active = False

        # ENTRY LOGIC
        if inventory == 0:
            # 1. Base Filter (RSI)
            if rsi >= self.rsi_buy: return Signal('HOLD')
            
            # 2. Satellite Check (Multi-Timeframe)
            if self.use_satellite and secondary_slice_df is not None and not secondary_slice_df.empty:
                # 15m Slice
                last_15m = secondary_slice_df.iloc[-1]
                
                # Check 15m RVOL
                # vol_sma = secondary_slice_df['volume'].rolling(20).mean().iloc[-1] # Already computed in slice? No, calc now
                # We need safe calculation, slice might be small
                if len(secondary_slice_df) >= 20:
                     avg_vol = secondary_slice_df['volume'].rolling(20).mean().iloc[-1]
                     cur_vol = secondary_slice_df['volume'].iloc[-1]
                     rvol = cur_vol / avg_vol if avg_vol > 0 else 0
                     
                     if rvol < self.sat_rvol: return Signal('HOLD') # Veto
                
                # Check 15m BB Expansion
                if len(secondary_slice_df) >= 2:
                    current_bbw = (last_15m['bb_upper'] - last_15m['bb_lower'])
                    prev_bbw = (secondary_slice_df['bb_upper'].iloc[-2] - secondary_slice_df['bb_lower'].iloc[-2])
                    
                    expansion = (current_bbw - prev_bbw) / prev_bbw if prev_bbw > 0 else 0
                    
                    if expansion < self.sat_bb_expand: return Signal('HOLD') # Veto
            
            # If we survived checks -> BUY
            # If we survived checks -> BUY
            # If we survived checks -> BUY
            # UNLEASHED MODE: Use config size (25%) instead of hardcoded 20%
            return Signal('BUY', size=config.SIZE_MAX_ALLOCATION, reason="Evo Entry", stop_loss=self.stop_loss, take_profit=self.take_profit)
            
        return Signal('HOLD')
