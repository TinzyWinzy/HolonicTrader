"""
StrategyHolon - NEXUS Cognition (Phase 15)

Implements:
1. SCAVENGER Logic: Bollinger Band Mean Reversion + RSI
2. PREDATOR Logic: Liquidity Sniper (FVG/Breakout) + OBV Confirmation
3. Clear Entry/Exit conditions (no stacking)
"""

import pandas as pd
import numpy as np
import os
try:
    import joblib
except ImportError:
    joblib = None

from typing import Any, Optional, Literal
from scipy.stats import linregress

# TensorFlow is heavy, only import if needed or handle carefully
try:
    import tensorflow as tf
except ImportError:
    tf = None

from HolonicTrader.holon_core import Holon, Disposition
from HolonicTrader.holon_core import Message
from .agent_executor import TradeSignal
from .kalman import KalmanFilter1D
import config

class StrategyHolon(Holon):
    def __init__(self, name: str = "StrategyAgent", stop_loss: float = None):
        default_disposition = Disposition(autonomy=0.8, integration=0.4)
        super().__init__(name=name, disposition=default_disposition)
        
        # RSI Parameters
        self.rsi_period = 14
        self.rsi_oversold = config.STRATEGY_RSI_OVERSOLD 
        self.rsi_overbought = config.STRATEGY_RSI_OVERBOUGHT
        self.stop_loss = stop_loss if stop_loss is not None else config.SCAVENGER_STOP_LOSS

        # LSTM Brain
        self.model = None
        self.scaler = None
        self.load_brain()
        
        # State Memory
        self.last_exit_times = {} # {symbol: timestamp_index}
        self.kalman_filters = {} # {symbol: KalmanFilter1D}

    def load_brain(self):
        """Load the LSTM model and scaler."""
        model_path = 'lstm_model.keras'
        scaler_path = 'scaler.pkl'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path) and tf is not None and joblib is not None:
            try:
                self.model = tf.keras.models.load_model(model_path)
                self.scaler = joblib.load(scaler_path)
                print(f"[{self.name}] Brain loaded successfully.")
            except Exception as e:
                print(f"[{self.name}] Error loading brain: {e}")
        else:
            print(f"[{self.name}] Brain missing or deps (TensorFlow/Joblib) not found. Running in heuristic mode.")

    def record_exit(self, symbol: str, timestamp) -> None:
        """Record the timestamp of a strategy exit for cooldown enforcement."""
        self.last_exit_times[symbol] = timestamp
        print(f"[{self.name}] Recorded exit for {symbol} at {timestamp}. Cooldown active for {config.STRATEGY_POST_EXIT_COOLDOWN_CANDLES} candles.")

    def predict_trend_lstm(self, prices: pd.Series) -> float:
        """
        Predict probability of price UP using LSTM.
        Returns: 0.0 to 1.0 (Probability)
        """
        if self.model is None or self.scaler is None or len(prices) < 60:
            return 0.53 # Heuristic: Slight Bullish Bias to allow OBV/RSI to drive
            
        try:
            # Prepare data
            data = prices.values[-60:].reshape(-1, 1)
            scaled_data = self.scaler.transform(data)
            x_input = scaled_data.reshape(1, 60, 1)
            
            # Predict
            prob = self.model.predict(x_input, verbose=0)[0][0]
            return float(prob)
        except Exception as e:
            print(f"[{self.name}] LSTM Prediction Error: {e}")
            return 0.5

    def calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume (OBV)."""
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=df.index)

    def calculate_obv_slope(self, obv_series: pd.Series, window: int = 14) -> float:
        """Calculate the slope of OBV over a window using linear regression."""
        if len(obv_series) < window:
            return 0.0
            
        y = obv_series.iloc[-window:].values
        x = np.arange(window)
        
        slope, _, _, _, _ = linregress(x, y)
        return slope

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        last_rsi = rsi_series.iloc[-1]

        if pd.isna(last_rsi):
            return 50.0
        return float(last_rsi)

    def get_kalman_estimate(self, symbol: str, prices: pd.Series) -> float:
        """
        Get the latest Kalman estimate for a symbol.
        Initializes filter if new, or updates it with latest price.
        Ideally should run sequentially on stream, but here we might just re-run on window end
        or persist properly. For now, we update with the latest price.
        """
        if symbol not in self.kalman_filters:
            # Initialize with decent noise params
            self.kalman_filters[symbol] = KalmanFilter1D(process_noise=0.01, measurement_noise=0.1)
            # Warm up with last 10 prices to converge
            for p in prices.values[-10:]:
                self.kalman_filters[symbol].update(p)
        else:
            # Update with just the latest price
            self.kalman_filters[symbol].update(prices.iloc[-1])
            
        return self.kalman_filters[symbol].x

    def check_scavenger_entry(self, price: float, bb: dict, rsi: float) -> bool:
        """
        SCAVENGER Entry: Ultra-Aggressive Mean Reversion
        """
        if bb['lower'] <= 0:
            return False
            
        # WARP SPEED 2.0: Buy lower half of channel
        # If Price < Middle BB AND RSI < ENTRY_MAX (60)
        is_below_middle = price <= bb['middle']
        
        # Or Panic Buy if RSI is truly low
        is_panic_level = rsi < config.STRATEGY_RSI_PANIC_BUY
        
        should_enter = (is_below_middle and rsi < config.STRATEGY_RSI_ENTRY_MAX) or is_panic_level
        
        return should_enter
    
    def check_scavenger_exit(self, price: float, bb: dict) -> bool:
        """
        SCAVENGER Exit: Price touches Middle BB (SMA 20)
        """
        if bb['middle'] <= 0:
            return False
        return price >= bb['middle']
    
    def check_predator_entry(self, df: pd.DataFrame, obv_slope: float) -> bool:
        """
        PREDATOR Entry: Liquidity Sniper
        - Price sweeps 1H Low but closes back inside range (Liquidity Grab)
        - OBV supports reversal (slope > 0)
        """
        if len(df) < 3:
            return False
            
        # Get recent candles
        prev_low = df['low'].iloc[-2]
        current_low = df['low'].iloc[-1]
        current_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        
        # Liquidity Grab: Swept low but closed above
        swept_low = current_low < prev_low
        closed_inside = current_close > prev_low
        
        # OBV Confirmation (Ignored in Warp Speed if needed, but let's keep it for sniper quality)
        obv_bullish = obv_slope > 0
        
        return swept_low and closed_inside and obv_bullish
    
    def check_momentum_entry(self, price: float, bb: dict, rsi: float) -> bool:
        """
        PREDATOR Entry: Momentum / Trend Following (Warp Speed)
        # - Price is ABOVE Middle BB (Trending Up)
        # - Price is BELOW Upper BB (Not Overextended / Mean Reversion Zone)
        # - RSI is Healthy (50-75), not overbought
        """
        is_above_middle = price > bb['middle']
        is_not_overextended = price < bb['upper']
        is_healthy_rsi = 50.0 <= rsi < 75.0
        
        # Shadow Log for "Missed" Signals
        if is_above_middle and is_healthy_rsi and not is_not_overextended:
             print(f"[{self.name}] FILTERED MOMENTUM: {price:.4f} > UpperBB {bb['upper']:.4f} (Overextended)")

        return is_above_middle and is_not_overextended and is_healthy_rsi

    def analyze_for_entry(
        self, 
        symbol: str,
        window_data: pd.DataFrame, 
        bb: dict, 
        obv_slope: float,
        metabolism_state: Literal['SCAVENGER', 'PREDATOR']
    ) -> Optional[TradeSignal]:
        """
        Generate entry signal based on metabolic state.
        Now confirmed by LSTM Brain.
        """
        prices = window_data['close']
        current_price = float(prices.iloc[-1])
        rsi = self.calculate_rsi(prices, self.rsi_period)
        
        # LSTM Bullish Confirmation
        # WARP SPEED: Lowered to 0.45 to be very permissive
        lstm_prob = self.predict_trend_lstm(prices)
        lstm_prob = self.predict_trend_lstm(prices)
        is_bullish = lstm_prob > config.STRATEGY_LSTM_THRESHOLD
        
        # KALMAN TREND FILTER
        kalman_price = self.get_kalman_estimate(symbol, prices)
        is_kalman_bullish = current_price > kalman_price
        
        # Combine filters?
        # For now, let's say Kalman confirms Scavenger (Mean Rev) if price < Kalman (Undervalued)?
        # Or Price > Kalman (Trend)?
        # Scavenger is Buy Low -> Price should be BELOW Kalman (Reverting up to it)
        # Predator is Buy High -> Price should be ABOVE Kalman (Momentum)
        
        
        # POST-EXIT COOLDOWN CHECK
        if symbol in self.last_exit_times:
            last_exit = self.last_exit_times[symbol]
            # Check if last_exit is within the last X candles of the window
            # Assuming window_data index is timestamps
            try:
                # Get the location of the last exit in the current window index
                # If it's present and close to the end, we are in cooldown
                exit_loc = window_data.index.get_loc(last_exit)
                current_loc = len(window_data) - 1
                distance = current_loc - exit_loc
                
                if distance < config.STRATEGY_POST_EXIT_COOLDOWN_CANDLES:
                    # Shadow Log
                    print(f"[{self.name}] {symbol} FILTERED COOLDOWN: Re-entry blocked (Dist {distance})")
                    return None
            except KeyError:
                # Last exit timestamp not in current window (long time ago)
                pass
            except Exception as e:
                print(f"[{self.name}] Cooldown Check Error: {e}")
        
        if metabolism_state == 'SCAVENGER':
            # Scavenger wants to buy dips. Kalman as Fair Value.
            # If Price < Kalman, it is "undervalued" relative to state.
            if self.check_scavenger_entry(current_price, bb, rsi):
                if is_bullish:
                    # Optional Kalman Check: Only buy if "deep" enough below Kalman?
                    # or just use as logger for now.
                    # Let's enforce Price < Kalman for Scavenger validity
                    if current_price < kalman_price:
                         print(f"[{self.name}] {symbol} SCAVENGER ENTRY (LSTM+KALMAN): P={current_price:.2f} < K={kalman_price:.2f}")
                         return TradeSignal(symbol=symbol, direction='BUY', size=1.0, price=current_price)
                    else:
                         pass # Price > Kalman, not a deep value buy?
                else:
                    print(f"[{self.name}] {symbol} SCAVENGER BLOCKED BY LSTM: Prob {lstm_prob:.2f} <= 0.45")
                    
        else:  # PREDATOR
            # 1. Sniper (Reversal)
            is_sniper = self.check_predator_entry(window_data, obv_slope)
            
            # 2. Relaxed Scavenger (Dip Buy)
            is_scavenger_valid = self.check_scavenger_entry(current_price, bb, rsi)
            
            # 3. Momentum (Trend Follow) - NEW
            is_momentum = self.check_momentum_entry(current_price, bb, rsi)
            
            if is_sniper or is_scavenger_valid or is_momentum:
                if is_bullish:
                    # Predator wants Momentum. Price > Kalman.
                    if current_price > kalman_price:
                        # Determine reason
                        if is_sniper: reason = "SNIPER"
                        elif is_scavenger_valid: reason = "DIP_BUY"
                        else: reason = "MOMENTUM"
                        
                        print(f"[{self.name}] {symbol} PREDATOR ENTRY ({reason}+KALMAN): P={current_price:.2f} > K={kalman_price:.2f}")
                        return TradeSignal(symbol=symbol, direction='BUY', size=1.0, price=current_price)
                    else:
                         print(f"[{self.name}] {symbol} PREDATOR BLOCKED BY KALMAN: P={current_price:.2f} < K={kalman_price:.2f}")
                else:
                    print(f"[{self.name}] {symbol} PREDATOR BLOCKED BY LSTM: Prob {lstm_prob:.2f} <= 0.45")
            else:
                # Log why it failed
                # print(f"[{self.name}] {symbol} PREDATOR FAIL: Sniper={is_sniper}, Dip={is_scavenger_valid}, Mom={is_momentum}")
                pass
        
        return None
    
    def get_health(self) -> dict:
        """Report agent health status."""
        return {
            'status': 'OK',
            'model': 'LSTM' if self.model else 'HEURISTIC'
        }

    def analyze_for_exit(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        bb: dict,
        atr: float,
        metabolism_state: Literal['SCAVENGER', 'PREDATOR']
    ) -> Optional[TradeSignal]:
        """
        Generate exit signal based on metabolic state.
        Now includes Proximity Reporting and Scalp TP.
        """
        if entry_price <= 0:
            return None
            
        pnl_pct = (current_price - entry_price) / entry_price
        
        if metabolism_state == 'SCAVENGER':
            # 1. BB Middle (Mean Reversion)
            if self.check_scavenger_exit(current_price, bb):
                print(f"[{self.name}] {symbol} MEAN REVERSION EXIT: Price {current_price:.4f} >= BB_Middle {bb['middle']:.4f}")
                return TradeSignal(symbol=symbol, direction='SELL', size=1.0, price=current_price)
            
            # 2. Scalp TP (Warp Speed Velocity)
            if pnl_pct >= config.SCAVENGER_SCALP_TP:
                print(f"[{self.name}] {symbol} SCALP TP REACHED: PnL {pnl_pct*100:.2f}% (Target: {config.SCAVENGER_SCALP_TP*100}%)")
                return TradeSignal(symbol=symbol, direction='SELL', size=1.0, price=current_price)
            
            # 3. Stop Loss
            if pnl_pct <= -self.stop_loss:
                print(f"[{self.name}] {symbol} SCAVENGER STOP-LOSS: PnL {pnl_pct*100:.2f}%")
                return TradeSignal(symbol=symbol, direction='SELL', size=1.0, price=current_price)
                    
        else:  # PREDATOR
            # 1. Take Profit (Big Move)
            if pnl_pct >= config.PREDATOR_TAKE_PROFIT:
                print(f"[{self.name}] {symbol} PREDATOR profit target reached: {pnl_pct*100:.2f}%")
                return TradeSignal(symbol=symbol, direction='SELL', size=1.0, price=current_price)

            # 2. Trailing Stop (ATR * 2)
            if atr > 0:
                trailing_stop = entry_price - (atr * config.PREDATOR_TRAILING_STOP_ATR_MULT)
                if current_price <= trailing_stop:
                    print(f"[{self.name}] {symbol} PREDATOR TRAILING STOP: Price {current_price:.4f} <= Stop {trailing_stop:.4f}")
                    return TradeSignal(symbol=symbol, direction='SELL', size=1.0, price=current_price)
        
        return None

    def receive_message(self, sender: Any, content: Any) -> None:
        """Handle incoming messages."""
        if isinstance(content, Message):
             pass
        else:
             pass
