"""
EntryOracleHolon - The "Offense" Brain (Phase 16)

Specialized in:
1. Pattern Recognition (LSTM)
2. Global Market Bias (GMB) Calculation
3. Entry Signal Generation (Scavenger/Predator)
"""

import pandas as pd
import numpy as np
import os
try:
    import joblib
except ImportError:
    joblib = None

try:
    import tensorflow
    import tensorflow as tf
except ImportError:
    tensorflow = None
    tf = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import openvino as ov
except ImportError:
    ov = None

from typing import Any, Optional, Literal
from HolonicTrader.holon_core import Holon, Disposition
from .kalman import KalmanFilter1D
import config

class EntryOracleHolon(Holon):
    def __init__(self, name: str = "EntryOracle"):
        super().__init__(name=name, disposition=Disposition(autonomy=0.9, integration=0.6))
        
        # Parameters
        self.rsi_period = 14
        
        # AI Brains
        self.model = None       # LSTM
        self.scaler = None      # Scaler for LSTM
        self.xgb_model = None   # XGBoost
        self.ov_compiled_model = None # OpenVINO
        self.load_brains()
        
        # State Memory
        self.kalman_filters = {} # {symbol: KalmanFilter1D}
        self.kalman_last_ts = {} # {symbol: timestamp}
        self.symbol_trends = {}  # {symbol: bool (is_bullish)}
        self.last_probes = {}    # {symbol: {'lstm': prob, 'xgb': prob}}

    def load_brains(self):
        """Load AI brains (LSTM and XGBoost)."""
        # LSTM Paths
        model_path = 'lstm_model.keras'
        scaler_path = 'scaler.pkl'
        # XGBoost Path
        xgb_path = 'xgboost_model.json'
        
        # 1. Load LSTM
        if os.path.exists(model_path) and os.path.exists(scaler_path) and tf is not None and joblib is not None:
            try:
                self.model = tf.keras.models.load_model(model_path)
                self.scaler = joblib.load(scaler_path)
                print(f"[{self.name}] LSTM Brain loaded successfully.")
            except Exception as e:
                print(f"[{self.name}] Error loading LSTM: {e}")
        
        # 2. Load XGBoost
        if os.path.exists(xgb_path) and xgb is not None:
            try:
                self.xgb_model = xgb.Booster()
                self.xgb_model.load_model(xgb_path)
                print(f"[{self.name}] XGBoost Brain loaded successfully.")
            except Exception as e:
                print(f"[{self.name}] Error loading XGBoost: {e}")

        if self.model is None and self.xgb_model is None:
            print(f"[{self.name}] All brains missing or deps failed. Running heuristic mode.")

        # 3. OpenVINO Integration (Speed Optimization)
        if self.model is not None and ov is not None and config.USE_OPENVINO:
            try:
                core = ov.Core()
                # Convert Keras model to OpenVINO IR
                ov_model = ov.convert_model(self.model)
                device = "GPU" if config.USE_INTEL_GPU else "CPU"
                self.ov_compiled_model = core.compile_model(ov_model, device)
                print(f"[{self.name}] OpenVINO LSTM Backend initialized on {device}.")
            except Exception as e:
                print(f"[{self.name}] OpenVINO Setup failed: {e}. Falling back to native TensorFlow.")

    def predict_trend_lstm(self, prices: pd.Series) -> float:
        if self.model is None or self.scaler is None or len(prices) < 60 or tf is None:
            return 0.53
        try:
            data = prices.values[-60:].reshape(-1, 1)
            scaled_data = self.scaler.transform(data)
            x_input = scaled_data.reshape(1, 60, 1)
            
            # High-Performance OpenVINO Inference if available
            if self.ov_compiled_model:
                res = self.ov_compiled_model(x_input)[0]
                return float(res[0][0])

            # High-Performance Functional Call (Avoids Retracing)
            x_tensor = tf.convert_to_tensor(x_input, dtype=tf.float32)
            prob_tensor = self.model(x_tensor, training=False)
            prob = prob_tensor.numpy()[0][0]
            
            return float(prob)
        except Exception as e:
            print(f"[{self.name}] Prediction Error: {e}")
            return 0.5

    def predict_trend_xgboost(self, features: dict) -> float:
        """Predict trend probability using XGBoost."""
        if self.xgb_model is None or xgb is None:
            return 0.55
        try:
            # Prepare DMatrix
            df_feat = pd.DataFrame([features])
            dmatrix = xgb.DMatrix(df_feat)
            prob = self.xgb_model.predict(dmatrix)[0]
            return float(prob)
        except Exception as e:
            print(f"[{self.name}] XGBoost Prediction Error: {e}")
            return 0.5
    def get_kalman_estimate(self, symbol: str, window_data: pd.DataFrame) -> float:
        prices = window_data['close']
        log_prices = np.log(prices)
        current_ts = window_data['timestamp'].iloc[-1]
        
        if symbol not in self.kalman_filters:
            self.kalman_filters[symbol] = KalmanFilter1D(process_noise=0.0001, measurement_noise=0.001)
            self.kalman_last_ts[symbol] = None
            for i in range(len(log_prices)):
                self.kalman_filters[symbol].update(log_prices.iloc[i])
                self.kalman_last_ts[symbol] = window_data['timestamp'].iloc[i]
        else:
            if current_ts != self.kalman_last_ts.get(symbol):
                self.kalman_filters[symbol].update(log_prices.iloc[-1])
                self.kalman_last_ts[symbol] = current_ts
                
        kalman_price = float(np.exp(self.kalman_filters[symbol].x))
        self.symbol_trends[symbol] = prices.iloc[-1] > kalman_price
        return kalman_price

    def get_market_bias(self) -> float:
        if not self.symbol_trends:
            return 0.5
            
        # Hardening: If we only have a few symbols, return neutral 0.5
        # This prevents the first 1-2 bearish symbols from setting GMB to 0.0
        if len(self.symbol_trends) < (len(config.ALLOWED_ASSETS) / 4):
            return 0.5
            
        bullish_count = sum(1 for trend in self.symbol_trends.values() if trend)
        return bullish_count / len(self.symbol_trends)

    def analyze_for_entry(
        self, 
        symbol: str,
        window_data: pd.DataFrame, 
        bb_vals: dict, 
        obv_slope: float,
        metabolism_state: Literal['SCAVENGER', 'PREDATOR']
    ):
        from .agent_executor import TradeSignal
        
        prices = window_data['close']
        current_price = float(prices.iloc[-1])
        
        # 1. Feature Engineering
        # RSI (14)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1]
        
        # BB %B
        bb_pct_b = (current_price - bb_vals['lower']) / (bb_vals['upper'] - bb_vals['lower']) if (bb_vals['upper'] - bb_vals['lower']) != 0 else 0.5
        
        # MACD (12, 26, 9)
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = (macd - macd_signal).iloc[-1]
        
        # Volatility & Momentum
        returns = prices.pct_change()
        volatility = returns.rolling(14).std().iloc[-1]
        
        # 2. Hierarchical Inference (Monolith-V4)
        lstm_prob = self.predict_trend_lstm(prices)
        
        xgb_features = {
            'lstm_prob': lstm_prob,
            'rsi': rsi,
            'bb_pct_b': bb_pct_b,
            'macd_hist': macd_hist,
            'volatility': volatility
        }
        xgb_prob = self.predict_trend_xgboost(xgb_features)
        
        # Store for GUI/Logging
        self.last_probes[symbol] = {
            'lstm': lstm_prob,
            'xgb': xgb_prob
        }
        
        # Final Decision from Master Brain (XGBoost)
        is_bullish = xgb_prob > config.STRATEGY_XGB_THRESHOLD
        high_conv_bullish = xgb_prob > 0.7
        high_conv_bearish = xgb_prob < 0.3
        
        kalman_price = self.get_kalman_estimate(symbol, window_data)
        market_bias = self.get_market_bias()
        is_market_bullish = market_bias >= config.GMB_THRESHOLD
        
        # Logging Consensus (Internal Diagnostic)
        if lstm_prob > 0.6 or xgb_prob > 0.6:
            print(f"[{self.name}] Ensemble Check {symbol}: LSTM({lstm_prob:.2f}) XGB({xgb_prob:.2f})")

        if metabolism_state == 'SCAVENGER':
            # --- BULLISH SCAVENGER ---
            is_below_middle = current_price <= bb_vals['middle']
            is_panic_buy = rsi < config.STRATEGY_RSI_PANIC_BUY
            should_buy = (is_below_middle and rsi < config.STRATEGY_RSI_ENTRY_MAX) or is_panic_buy
            
            # RECALIBRATION: Relax Kalman filter if high-conviction
            if should_buy and (is_bullish or rsi < 30) and (current_price < kalman_price or high_conv_bullish):
                if is_market_bullish:
                    print(f"[{self.name}] {symbol} ENSEMBLE BUY (GMB {market_bias:.2f})")
                    return TradeSignal(symbol=symbol, direction='BUY', size=1.0, price=current_price)
            
            # --- BEARISH SCAVENGER ---
            is_above_middle = current_price >= bb_vals['middle']
            is_panic_sell = rsi > 75 
            should_short = (is_above_middle and rsi > 65) or is_panic_sell
            
            # RECALIBRATION: Relax Kalman filter if high-conviction
            if should_short and (not is_bullish or rsi > 80) and (current_price > kalman_price or high_conv_bearish):
                if not is_market_bullish:
                    print(f"[{self.name}] {symbol} ENSEMBLE SHORT (GMB {market_bias:.2f})")
                    return TradeSignal(symbol=symbol, direction='SELL', size=1.0, price=current_price)
                    
        else: # PREDATOR
            # --- BULLISH MOMENTUM ---
            is_above_middle = current_price > bb_vals['middle']
            is_not_overextended_up = current_price < bb_vals['upper']
            # RECALIBRATION: Widened RSI window (45-85)
            is_healthy_rsi_up = 45.0 <= rsi < 85.0
            is_momentum_up = is_above_middle and is_not_overextended_up and is_healthy_rsi_up
            
            # RECALIBRATION: Relax Kalman filter if high-conviction
            if is_momentum_up and is_bullish and (current_price > kalman_price or high_conv_bullish):
                 if is_market_bullish:
                    print(f"[{self.name}] {symbol} ENSEMBLE MOMENTUM BUY (GMB {market_bias:.2f})")
                    return TradeSignal(symbol=symbol, direction='BUY', size=1.0, price=current_price)
            
            # --- BEARISH MOMENTUM (Shorting) ---
            is_below_middle = current_price < bb_vals['middle']
            is_not_overextended_down = current_price > bb_vals['lower']
            # RECALIBRATION: Widened RSI window (15-55)
            is_healthy_rsi_down = 15.0 < rsi <= 55.0
            is_momentum_down = is_below_middle and is_not_overextended_down and is_healthy_rsi_down
            
            # RECALIBRATION: Relax Kalman filter if high-conviction
            if is_momentum_down and (not is_bullish) and (current_price < kalman_price or high_conv_bearish):
                if not is_market_bullish:
                    print(f"[{self.name}] {symbol} ENSEMBLE MOMENTUM SHORT (GMB {market_bias:.2f})")
                    return TradeSignal(symbol=symbol, direction='SELL', size=1.0, price=current_price)
        
        return None

    def get_health(self) -> dict:
        last_lstm = 0.5
        last_xgb = 0.5
        if self.last_probes:
            # Get the last symbol analyzed
            last_sym = list(self.last_probes.keys())[-1]
            last_lstm = self.last_probes[last_sym]['lstm']
            last_xgb = self.last_probes[last_sym]['xgb']

        return {
            'status': 'OK' if (self.model or self.xgb_model) else 'HEURISTIC',
            'lstm_loaded': self.model is not None,
            'xgb_loaded': self.xgb_model is not None,
            'last_lstm': last_lstm,
            'last_xgb': last_xgb
        }

    def receive_message(self, sender: Any, content: Any) -> None:
        pass
