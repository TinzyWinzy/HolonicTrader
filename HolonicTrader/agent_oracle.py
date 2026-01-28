"""
EntryOracleHolon - The "Offense" Brain (Phase 16)

Specialized in:
1. Pattern Recognition (LSTM)
2. Global Market Bias (GMB) Calculation
3. Entry Signal Generation (Scavenger/Predator)
"""

import pandas as pd
import numpy as np
import traceback
from typing import Any, Dict, List, Optional, Literal, Tuple
from .agent_executor import TradeSignal as GlobalTradeSignal # Fix: Renamed to avoid scope collision
import os
import json
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
from HolonicTrader.sde_engine import SDEEngine
import config
import threading
import sys
# Path Hacking to reach sandbox
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path: sys.path.append(parent_dir)
    from sandbox.strategies.ensemble import EnsembleStrategy
except ImportError as e:
    print(f"⚠️ Ensemble Import Failed: {e}")
    EnsembleStrategy = None

class EntryOracleHolon(Holon):
    def __init__(self, name: str = "EntryOracle", xgb_model=None):
        super().__init__(name=name, disposition=Disposition(autonomy=0.9, integration=0.6))
        
        # Parameters
        self.rsi_period = 14
        self._lock = threading.Lock()
        self.DEBUG = getattr(config, 'DEBUG', False) # Fix for AttributeError
        
        # AI Brains
        self.model = None       # LSTM
        self.scaler = None      # Scaler for LSTM
        self.xgb_model = xgb_model   # XGBoost - ALLOW INJECTION
        self.ov_compiled_model = None # OpenVINO
        
        # State Memory
        self.kalman_filters = {} # {symbol: KalmanFilter1D}
        self.kalman_last_ts = {} # {symbol: timestamp}
        self.symbol_trends = {}  # {symbol: bool (is_bullish)}
        self.last_probes = {}    # {symbol: {'lstm': prob, 'xgb': prob}}
        self.last_macro_state = {} # {symbol: 'BULLISH'/'BEARISH'} for log damping
        self.current_metabolism = 'SCAVENGER' # Default Tracking
        self.market_state = {}   # {correlation_matrix: df, entropy: float}
        self.crisis_score = 0.0  # Macro Crisis Score (0.0 - 1.0)
        
        # Optimization: Inference Cache
        self._inference_cache = {} # {f"{symbol}_{ts}": prob}

        # Emergence: Emotional State
        self.emotional_state = {'fear': 0.0, 'greed': 1.0}
        
        # Ensemble Strategy (Hall of Fame)
        self.ensemble = None

        
        # Load Brains
        self.load_brains()
        
    def set_market_state(self, correlation_matrix: pd.DataFrame = None, entropy: float = None):
        """Receive live market physics data from Trader/Observer."""
        with self._lock:
            if correlation_matrix is not None:
                self.market_state['correlation'] = correlation_matrix
                self.market_state['entropy'] = entropy
        
    def set_crisis_score(self, score: float):
        """Update macro crisis score from SentimentHolon."""
        self.crisis_score = score
        
    def set_expert_model(self, model):
        """Inject a specific XGBoost model (for Walk-Forward Optimization)."""
        with self._lock:
            self.xgb_model = model
            print(f"[{self.name}] 🧠 New XGBoost Brain Injected.")

    def load_ensemble(self, hall_of_fame_path: str):
        """Hot-Swap the Ensemble Strategy from disk."""
        if EnsembleStrategy is None:
            print(f"[{self.name}] ⚠️ Cannot load Ensemble: Class not imported.")
            return

        try:
            with self._lock:
                print(f"[{self.name}] 🎭 Loading Ensemble Strategy from {hall_of_fame_path}...")
                self.ensemble = EnsembleStrategy(hof_path=hall_of_fame_path)
                print(f"[{self.name}] ✅ Ensemble Loaded: {len(self.ensemble.strategies)} Kings are Active.")
        except Exception as e:
            print(f"[{self.name}] ❌ Ensemble Load Failed: {e}")


    def set_emotional_bias(self, fear: float, greed: float):
        """
        Receive Emotional Feedback from the Neural Network (Trader).
        Fear = Drawdown (0.0 - 1.0)
        Greed = Risk Multiplier (0.5 - 2.0)
        """
        with self._lock:
            self.emotional_state = {'fear': fear, 'greed': greed}
            
    def apply_asset_personality(self, symbol: str, signal: Any) -> Any:
        """
        Apply Asset-Specific Rules (The Physics Layer).
        Modifies or Vetos signals based on asset class.
        """
        if not signal: return None
        
        # --- EMOTIONAL OVERRIDE (Amygdala) ---
        fear = self.emotional_state.get('fear', 0.0)
        greed = self.emotional_state.get('greed', 1.0)
        
        # FEAR: If Drawdown > 10% (Fear > 0.1), Block weak signals
        # FEAR: If Drawdown > 15% (Fear > 0.15), Block weak signals
        # UNLEASHED: Relaxed scaling. Max inhibition is +0.2 (Req 0.7)
        if fear > 0.15:
             # Old: 0.5 + (fear * 0.4)
             # New: Base 0.5 + (fear * 0.2). Milder penalty.
             required_conviction = min(0.8, 0.5 + (fear * 0.2))
             
             if signal.conviction < required_conviction:
                  print(f"[{self.name}] 😨 FEAR VETO: {symbol} {signal.direction} Conviction {signal.conviction:.2f} < Req {required_conviction:.2f} (Fear {fear:.2f})")
                  return None
                 
        # GREED: If Risk Multiplier > 1.2, Boost conviction (Confidence)
        if greed > 1.2:
            signal.conviction = min(1.0, signal.conviction * 1.1)
            
        # 1. BTC: Dead Market Filter
        if symbol == 'BTC/USDT':
            meta = signal.metadata
            atr = meta.get('atr', 0)
            avg_atr = meta.get('avg_atr', atr) # Fallback
            if avg_atr > 0 and atr < (avg_atr * config.PERSONALITY_BTC_ATR_FILTER):
                print(f"[{self.name}] ☠️ BTC FILTER: Market Dead (ATR {atr:.2f} < 50% Avg). Signal IGNORED.")
                return None
                
        # 2. DOGE: Fakeout Filter (RVOL)
        # 2. DYNAMIC ASSET PROFILES (Evolved Personalty)
        profiles = getattr(config, 'ASSET_PROFILES', {})
        if symbol in profiles:
            profile = profiles[symbol]
            meta = signal.metadata
            
            # RVOL Check
            if 'rvol_threshold' in profile:
                rvol = meta.get('rvol', 1.0)
                limit = profile['rvol_threshold']
                if rvol < limit:
                    print(f"[{self.name}] 🧬 {symbol} FILTER: Low Volume (RVOL {rvol:.1f} < {limit}). IGNORED.")
                    return None
            
            # RSI Check (Contextual)
            # If 'rsi_buy' is defined, it usually implies a max-rsi for entry (value/pullback)
            # But for SOL (Trend), 49 means "Buy early". 
            # We treat it as a "Don't Buy if Overbought" cap relative to the profile?
            # Or simplified: Veto if RSI is SIGNIFICANTLY higher than optimal.
            if 'rsi_buy' in profile and signal.direction == 'BUY':
                rsi = meta.get('rsi', 50.0)
                optimal = profile['rsi_buy']
                
                # --- PATCH: RSI FLEX & STRUCTURAL BYPASS ---
                # 1. Relaxed Flex (+25 instead of +15)
                # 2. Structural Bypass: If Trigger D (Resonance) and Bullish Regime, 
                #    we allow entry if RSI < 55 (Neutral-Bullish).
                reason = meta.get('reason', '')
                is_resonance = (reason == 'STRUCTURAL_RESONANCE')
                
                flex = 25.0
                
                # --- PROFIT BOOST: RSI FLEXING ---
                # If in PREDATOR mode, we care more about the trend than the dip.
                # If conviction is high (>0.65) and metabolism is PREDATOR, we add extra flex.
                if self.current_metabolism == 'PREDATOR' and signal.conviction > 0.65:
                    flex += 15.0 # Total 40.0 Flex (e.g. 23 + 40 = 63 RSI allowed)
                
                # --- RSI FIX (CRITICAL) ---
                # User Request: "Only block if RSI > 70"
                # Removed complex optimal+flex logic which was blocking at 50.0
                if rsi > 70.0:
                     if is_resonance and rsi < 75.0: # Allow slightly higher for Resonance
                          pass 
                     else:
                          print(f"[{self.name}] 🧬 {symbol} FILTER: Overbought (RSI {rsi:.1f} > 70.0). IGNORED.")
                          return None
                
        # 4. XRP: Whole Number Front-running
        elif symbol == 'XRP/USDT':
            # Add TP instruction to metadata
            # For Phase 4 simple execution, we just log it. Real execution needs smarter order types.
            signal.metadata['special_instruction'] = 'FRONT_RUN_WHOLE_NUMBERS'
            
        # 5. FAIR WEATHER PROTOCOL (Global Bias Veto)
        # Block ALL Satellite Longs if Global Bias is weak (< GMB_THRESHOLD - 0.15)
        # Core assets (BTC/ETH) are strong enough to buck the trend.
        if signal.direction == 'BUY' and symbol in config.SATELLITE_ASSETS:
            # WHALE BYPASS: Whales swim in any weather
            if signal.metadata.get('is_whale', False):
                pass 
            else:
                gmb = self.get_market_bias()
                if gmb < (config.GMB_THRESHOLD - 0.05):
                    print(f"[{self.name}] ☁️ FAIR WEATHER VETO: {symbol} Long blocked (Bias {gmb:.2f} < {config.GMB_THRESHOLD - 0.05:.2f})")
                    return None
                
        # 6. CRISIS PROTOCOL (Macro Strategy)
        # Assuming self.crisis_score is updated by TraderHolon
        # 6. CRISIS PROTOCOL (Macro Strategy)
        # Assuming self.crisis_score is updated by TraderHolon
        # Adjusted threshold to 0.75 (User Request - Actual Conditions)
        if self.crisis_score > 0.75:
            # A. FLIGHT TO SAFETY (Boost Gold/BTC)
            if symbol in config.CRISIS_SAFE_HAVENS and signal.direction == 'BUY':
                signal.conviction = min(1.0, signal.conviction * 1.2) # +20% Boost
                signal.metadata['crisis_boost'] = True
                print(f"[{self.name}] ☢️ CRISIS BOOST: {symbol} Conviction increased to {signal.conviction:.2f}")
            
            # B. RISK OFF (Block Meme Longs)
            elif symbol in config.CRISIS_RISK_ASSETS and signal.direction == 'BUY':
                print(f"[{self.name}] ☢️ CRISIS VETO: {symbol} Long blocked (Crisis Score {self.crisis_score:.2f})")
                return None
        
        # 6b. WHALE TRACKING (Volume Physics)
        if signal.metadata.get('is_whale', False):
            signal.conviction = min(1.0, signal.conviction * 1.25) # +25% Boost
            print(f"[{self.name}] 🐋 WHALE BOOST: {symbol} Riding the wave! Conviction -> {signal.conviction:.2f}")

        # 7. PHYSICS LAYER (Global Validation)
        return self.apply_market_physics(symbol, signal)

    def apply_market_physics(self, symbol: str, signal: Any) -> Any:
        """
        The 'Physics Layer': Validates signals against laws of Correlation, Entropy, and Energy (Volume).
        """
        if not signal: return None
        
        # A. ENTROPY PROOF (Proof of Order)
        # We need the Entropy Agent's assessment. 
        # Ideally, we query it. For now, we assume if we are in this method, 
        # the market is 'tradeable' or we calculate it locally if crucial.
        
        # --- PATCH: PROBABILISTIC WEIGHTING (Global Bias) ---
        # Adjust Conviction based on Macro Trend (Bayesian Update)
        global_bias = self.get_market_bias()
        
        # User Directive: "Enable more short signals given bearish structures"
        if signal.direction == 'SELL':
            # Counter-Trend Short? (Bullish Bias > 0.6)
            if global_bias > 0.6:
                penalty = (global_bias - 0.6) * 1.5 # Stricter Penalty (Max 0.6)
                original_conv = signal.conviction
                signal.conviction -= penalty
                if self.DEBUG: 
                    print(f"[{self.name}] 📉 PROBABILITY ADJUST: {symbol} Short Conviction {original_conv:.2f}->{signal.conviction:.2f} (Bull Bias {global_bias:.2f})")
            
            # Trend-Following Short (Bear Bias < 0.4) -> BOOST
            elif global_bias < 0.4:
                bonus = (0.4 - global_bias) * 0.5 # Moderate Boost
                signal.conviction = min(1.0, signal.conviction + bonus)
                if self.DEBUG:
                     print(f"[{self.name}] 📈 BEAR REGIME BOOST: {symbol} Short Conviction -> {signal.conviction:.2f}")

        elif signal.direction == 'BUY':
             # Counter-Trend Long? (Bearish Bias < 0.4)
             if global_bias < 0.4:
                 penalty = (0.4 - global_bias) * 2.0 # SEVERE Penalty for fighting the bear
                 original_conv = signal.conviction
                 signal.conviction -= penalty
                 if self.DEBUG:
                     print(f"[{self.name}] 📉 PROBABILITY ADJUST: {symbol} Long Conviction {original_conv:.2f}->{signal.conviction:.2f} (Bear Bias {global_bias:.2f})")
                     
        # If conviction drops <= 0, signal is dead.
        if signal.conviction <= 0.0:
            return None
        # ----------------------------------------------------
        
        # A.2 PIVOT POINT REGIME (Structure Filter)
        # "Respect the Floor" - Don't buy below value unless conviction is high.
        structure = signal.metadata.get('structure', {})
        pivots = structure.get('pivots', {}) if structure else {}
        
        # FIX 4: Dynamic Pivot Veto Thresholds based on Market Regime
        global_bias = self.get_market_bias()
        is_extreme_bear = global_bias < 0.30
        is_extreme_bull = global_bias > 0.70
        
        # Lower thresholds when regime is extreme (more permissive in trending markets)
        long_pivot_thresh = 0.05 if is_extreme_bear else 0.45  # Was always 0.45
        short_pivot_thresh = 0.05 if is_extreme_bull else 0.50  # Was always 0.50
        
        if pivots and signal.direction == 'BUY':
            pivot_p = pivots.get('P', 0)
            current_price = signal.price
            
            # If Price is BELOW Daily Pivot (Bearish Zone)
            # If Price is BELOW Daily Pivot (Bearish Zone)
            # User Request: Allow 5% leeway with High Conviction
            # Super Signal Override
            if signal.metadata.get('is_whale', False):
                 pass
            elif current_price < (pivot_p * 0.95): # Deep below pivot (>5%)
                # Require Higher Conviction to buck the trend
                if signal.conviction < 0.7: # Raised req for deep underwater
                    print(f"[{self.name}] 🛡️ PIVOT VETO: {symbol} Long Deep below Pivot ({current_price:.2f} < {pivot_p:.2f}). Conviction {signal.conviction:.2f} too weak for deep dive.")
                    return None
            elif current_price < pivot_p: # Just below pivot (0-5%)
                 if signal.conviction < long_pivot_thresh: # FIX 4: Dynamic threshold
                    print(f"[{self.name}] 🛡️ PIVOT VETO: {symbol} Long below Daily Pivot. Conviction {signal.conviction:.2f} too weak.")
                    return None
        
        elif pivots and signal.direction == 'SELL':
            # "Don't short the Floor"
            pivot_p = pivots.get('P', 0)
            current_price = signal.price
            
            # If Price is significantly ABOVE Daily Pivot (Bullish Zone), shorting is risky
            if current_price > (pivot_p * 1.05): # Deep above pivot (>5%)
                if signal.conviction < 0.7:
                    print(f"[{self.name}] 🛡️ PIVOT VETO: {symbol} Short Deep above Pivot ({current_price:.2f} > {pivot_p:.2f}). Conviction {signal.conviction:.2f} too weak for top fishing.")
                    return None
            elif current_price > pivot_p: # Just above pivot (0-5%)
                 if signal.conviction < short_pivot_thresh: # FIX 4: Dynamic threshold
                    print(f"[{self.name}] 🛡️ PIVOT VETO: {symbol} Short above Daily Pivot. Conviction {signal.conviction:.2f} too weak.")
                    return None

        
        # B. VOLUME TRUTH (Energy)
        rvol = signal.metadata.get('rvol', 1.0)
        if rvol < config.PHYSICS_MIN_RVOL:
            # WHALE BYPASS: Whales create their own energy, L2 order book signals might precede volume
            if signal.metadata.get('is_whale', False):
                 pass 
            else:
                # Degrade confidence or Veto
                # Relaxed Soft Veto: Only blocks if GMB is VERY weak (< GMB_THRESHOLD - 0.05)
                gmb = self.get_market_bias()
                if gmb < (config.GMB_THRESHOLD - 0.05):  # Relaxed from 0.6 - only veto in very weak markets
                    print(f"[{self.name}] 🔋 LOW ENERGY VETO: {symbol} RVOL {rvol:.1f} < {config.PHYSICS_MIN_RVOL} & Very Weak Bias ({gmb:.2f})")
                    return None
                
        # C. PACK LOGIC (Correlation)
        # "Don't fight the Alpha."
        corr_matrix = self.market_state.get('correlation')
        if corr_matrix is not None and not corr_matrix.empty and 'BTC/USDT' in corr_matrix.columns:
            # Check Correlation to BTC
            btc_corr = corr_matrix['BTC/USDT'].get(symbol, 0.5)
            
            # Check BTC Trend (Proxy via Market Bias or explicit check. Using Bias < 0.5 as Bearish)
            gmb = self.get_market_bias()
            
            # RULE: If Correlated (>0.75) AND Leader is Weak (<0.5) -> VETO LONG
            # (Configurable Thresholds)
            if signal.direction == 'BUY':
                # Use Global Threshold (0.40) instead of Hard 0.5
                if btc_corr > config.PHYSICS_CORRELATION_THRESHOLD and gmb < config.GMB_THRESHOLD:
                    print(f"[{self.name}] 🐺 PACK VETO: {symbol} Correlated ({btc_corr:.2f}) & Market Weak ({gmb:.2f})")
                    return None
            
            # RULE: If Inverse Correlated (<-0.75) AND Leader is Strong -> VETO LONG? 
            # (Usually we want to buy inverse assets when market is weak, so this is fine.)
        
        return signal

    def predict_position_health(self, symbol: str, current_price: float, entry_price: float, direction: str) -> Dict[str, Any]:
        """
        Oracle Diagnosis: Check if a held position is growing or decaying.
        Uses: Price Action, Ensemble Vote (if avail), and Global Bias.
        """
        health = {
            'status': 'STABLE',
            'decay_score': 0.0, # 0.0 = Healthy, 1.0 = Critical
            'action': 'HOLD'
        }
        
        # 1. Price Momentum Check
        if direction == 'BUY':
            pnl = (current_price - entry_price) / entry_price
        else:
            pnl = (entry_price - current_price) / entry_price
            
        # 2. Ensemble AI Check (Forecast)
        # If we have ensemble loaded, ask it: "Is the trend still valid?"
        ensemble_vote = 0.5 # Neutral
        if self.ensemble:
            # We would need recent data frame here. Assuming we can fetch or it's cached.
            # Simplified: Use Global Bias as proxy if deep ensemble check is expensive.
            pass
            
        gmb = self.get_market_bias()
        
        # 3. Decay Logic
        # A. Early Profit Decay
        # If we are in profit (>1%) but Global Bias turning against us?
        if pnl > 0.01:
            if direction == 'BUY' and gmb < 0.4:
                health['status'] = 'DECAYING'
                health['decay_score'] = 0.7
                health['action'] = 'TIGHTEN_SL'
            elif direction == 'SELL' and gmb > 0.6:
                health['status'] = 'DECAYING'
                health['decay_score'] = 0.7
                health['action'] = 'TIGHTEN_SL'
                
        # B. Growth (Let it Run)
        # If trend is super strong, relax TP
        if direction == 'BUY' and gmb > 0.65:
            health['status'] = 'GROWTH'
            health['action'] = 'RELAX_TP'
        elif direction == 'SELL' and gmb < 0.35:
            health['status'] = 'GROWTH'
            health['action'] = 'RELAX_TP'
            
        return health

    def verify_holding_physics(self, symbol: str, direction: str, current_price: float = None, entry_price: float = None) -> Dict[str, Any]:
        """
        Proof of Holding: Re-verify the thesis for an open position.
        Returns detailed health dict (replacing simple boolean).
        """
        result = {'valid': True, 'reason': '', 'recommendation': 'HOLD'}
        
        # 1. ENTROPY CHECK (Chaos Veto)
        #Ideally we query Entropy Agent, but assuming we have access or use a proxy
        # For now, we return True as placeholder or implement local check if needed.
        # *Optimization*: In Phase 5 we link Agent State. 
        # Here we check Global Bias as a proxy for "Market Environment".
        
        # 2. PACK LOGIC (Correlation Veto)
        # If we are Long, and Global Bias drops below buffered threshold, and we are a Satellite...
        if direction == 'BUY' and symbol in config.SATELLITE_ASSETS:
             gmb = self.get_market_bias()
             threshold = config.GMB_THRESHOLD
             buffer = getattr(config, 'GMB_EXIT_HYSTERESIS', 0.10)
             
             if gmb < (threshold - buffer): # Buffered Exit
                 print(f"[{self.name}] 📉 THESIS FAILED: {symbol} Long held but Global Bias collapsed to {gmb:.2f} (Threshold {threshold - buffer:.2f})")
                 result['valid'] = False
                 result['reason'] = 'BIAS_COLLAPSE'
                 result['recommendation'] = 'EXIT_MARKET'
                 return result

        # 3. AI HEALTH CHECK (New)
        if current_price and entry_price:
            health = self.predict_position_health(symbol, current_price, entry_price, direction)
            if health['action'] != 'HOLD':
                result['recommendation'] = health['action']
                result['reason'] = f"AI_{health['status']}"
                if health['status'] == 'DECAYING':
                     print(f"[{self.name}] 🤒 POS DECAY: {symbol} {direction} (GMB mismatch). Rec: {health['action']}")
                elif health['status'] == 'GROWTH':
                     print(f"[{self.name}] 🌱 POS GROWTH: {symbol} {direction} (Trend Strong). Rec: {health['action']}")
                 
        return result

    def audit_asset_profile(self, symbol: str, data: Any) -> Dict[str, Any]:
        """
        Diagnostic: Check asset health against personality rules WITHOUT needing a signal.
        """
        status = "HEALTHY"
        details = []
        
        # Calculate Metrics
        atr_period = 14
        if len(data) < atr_period + 1:
            return {'status': 'INSUFFICIENT_DATA', 'metrics': {}}
            
        # ATR / Volatility
        tr = pd.concat([(data['high']-data['low']), (data['high']-data['close'].shift()).abs(), (data['low']-data['close'].shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean().iloc[-1]
        avg_atr = tr.rolling(30).mean().iloc[-1] # 30-period baseline
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # RVOL
        vol_sma = data['volume'].rolling(20).mean().iloc[-1]
        rvol = data['volume'].iloc[-1] / vol_sma if vol_sma > 0 else 0.0
        
        metrics = {
            'ATR': f"{atr:.4f}",
            'AvgATR': f"{avg_atr:.4f}",
            'RSI': f"{rsi:.1f}",
            'RVOL': f"{rvol:.2f}",
            'Price': f"{data['close'].iloc[-1]:.4f}"
        }
        
        # Check Rules
        if symbol == 'BTC/USDT':
            if avg_atr > 0 and atr < (avg_atr * config.PERSONALITY_BTC_ATR_FILTER):
                status = "DEAD_MARKET"
                details.append(f"ATR {atr:.4f} < {config.PERSONALITY_BTC_ATR_FILTER*100}% of Avg")
    

                
        elif symbol == 'DOGE/USDT':
            if rvol < config.PERSONALITY_DOGE_RVOL:
                status = "FAKEOUT_RISK"
                details.append(f"RVOL {rvol:.1f} < {config.PERSONALITY_DOGE_RVOL}")
                
        elif symbol == 'SOL/USDT':
            if rsi < config.PERSONALITY_SOL_RSI_LONG:
                details.append(f"Weak Momentum (RSI {rsi:.1f} < {config.PERSONALITY_SOL_RSI_LONG})") # Just a note, not necessarily unhealthy if Shorting
            if rsi > config.PERSONALITY_SOL_RSI_SHORT:
                 details.append(f"Strong Momentum (RSI {rsi:.1f} > {config.PERSONALITY_SOL_RSI_SHORT})")

        # Global Bias Check
        gmb = self.get_market_bias()
        if symbol in config.SATELLITE_ASSETS and gmb < 0.45:
             status = "VETOED (Fair Weather)"
             details.append(f"GMB {gmb:.2f} < 0.45")
             
        return {
            'symbol': symbol,
            'status': status,
            'details': ", ".join(details) if details else "None",
            'metrics': metrics
        }

    def analyze_satellite_entry(self, symbol: str, df_1h: pd.DataFrame, observer: Any) -> Any:
        from .agent_executor import TradeSignal
        
        # 🔑 KEY 1: TIMEFRAME ALIGNMENT (Trend)
        # 1H Check
        ema200_1h = df_1h['close'].ewm(span=200, adjust=False).mean().iloc[-1]
        price = df_1h['close'].iloc[-1]
        
        trend_1h = 'BULL' if price > ema200_1h else 'BEAR'
        
        # 15m Check (Fetch fresh data)
        df_15m = observer.fetch_market_data(timeframe='15m', limit=100, symbol=symbol)
        if df_15m.empty or len(df_15m) < 50: return None
        
        ema50_15m = df_15m['close'].ewm(span=50, adjust=False).mean().iloc[-1]
        price_15m = df_15m['close'].iloc[-1]
        
        trend_15m = 'BULL' if price_15m > ema50_15m else 'BEAR'
        
        # Alignment Veto
        if trend_1h != trend_15m: return None
        
        direction = 'BUY' if trend_1h == 'BULL' else 'SELL'
        
        # 🔑 KEY 2: VOLATILITY SQUEEZE (Timing)
        # Bollinger Bands (20, 2) on 15m
        sma20 = df_15m['close'].rolling(20).mean()
        std20 = df_15m['close'].rolling(20).std()
        upper = sma20 + (std20 * 2)
        lower = sma20 - (std20 * 2)
        
        # Using BB Width Expansion
        bb_middle = sma20.iloc[-1] # SMA20 is the middle band
        bb_upper = upper.iloc[-1]
        bb_lower = lower.iloc[-1]
        
        bb_width = (bb_upper - bb_lower) / bb_middle
        # rolling_avg_width = ((df_15m['bb_upper'] - df_15m['bb_lower']) / df_15m['bb_middle']).rolling(20).mean().iloc[-1]
        
        # Expansion Check: Is width > Threshold?
        # Note: Genome 'sat_bb_expand' is likely absolute width requirement or expansion factor.
        # Implemented as absolute width requirement for simplicity in Playground, so matching here.
        bbw_thresh = config.SATELLITE_BBW_EXPANSION_THRESHOLD
        if bb_width < bbw_thresh: return None
        
        # Breakout Check - DISABLED: Conficts with RSI Cap (Early Entry logic)
        # if direction == 'BUY' and price_15m <= upper.iloc[-1]: return None
        # if direction == 'SELL' and price_15m >= lower.iloc[-1]: return None
        
        # 🔑 KEY 3: VOLUME CONFIRMATION (Truth)
        # RVOL Calculation
        current_vol = df_15m['volume'].iloc[-1]
        avg_vol = df_15m['volume'].rolling(20).mean().iloc[-2] # Preceding 20 avg
        
        # New RVOL calculation from snippet
        volume_ema = df_15m['volume'].ewm(span=20).mean().iloc[-1]
        rvol = current_vol / volume_ema if volume_ema > 0 else 0
        
        # rvol_thresh set above
        rvol_thresh = config.SATELLITE_RVOL_THRESHOLD
        if rvol < rvol_thresh: return None
        
        # 🔑 KEY 4: RSI CEILING (Genome: Buy Early/Dipper)
        # We need RSI for this check. Re-using df_1h or df_15m?
        # Genome logic likely on execution timeframe (15m or 1H).
        # Let's use 15m RSI for precision.
        delta = df_15m['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi_15m = 100 - (100 / (1 + rs)).iloc[-1]
        
        rsi_cap = config.SATELLITE_ENTRY_RSI_CAP
        if rsi_15m >= rsi_cap:
             return None # Veto: Logic prefers buying dips/starts, not chased extensions
        
        # 🚀 ALL KEYS TURNED - FIRE
        
        # 🧬 KEY 5: ENSEMBLE VOTE (Evolutionary Intelligence)
        if self.ensemble:
             # Evaluate all active strategies in the Hall of Fame
             vote_score = 0.0
             try:
                 # Pass recent data to ensemble (mocking OHLCV dict or DF)
                 # Ensemble expects dict of {strategy_name: signal} or aggregated? 
                 # Let's assume evaluate_all returns list of signals or aggregate score.
                 # Checking sandbox interface: self.ensemble.evaluate_all(symbol, data) -> returns dict of signals
                 
                 # We need to adapt data to what Ensemble expects (likely DataFrame)
                 strategies = self.ensemble.strategies
                 vote_count = 0
                 affirmative = 0
                 
                 for strat in strategies:
                     # Simple check: Does strat agree?
                     # Requires strategy.analyze(df) -> Signal
                     # For Phase 46, we skip deep integration and assume if Ensemble is Loaded, 
                     # we give a small boost if it exists, or fully veto if 0% agree?
                     # Let's do a "Soft Boost" for now to avoid crashes.
                     pass
                     
                 # Placeholder: If Ensemble is alive, we trust it.
                 # In future: vote_score = self.ensemble.get_consensus(df_15m)
                 pass
             except Exception as e:
                 print(f"[{self.name}] ⚠️ Ensemble Vote Error: {e}")

        self._safe_print(f"[{self.name}] 🚀 SATELLITE ENTRY: {symbol} {direction} (1H/15m Align, BBW {bb_width:.2f} > {bbw_thresh:.2f}, RVOL {rvol:.1f} > {rvol_thresh:.1f})")
        
        sig = TradeSignal(symbol=symbol, direction=direction, size=1.0, price=price_15m)
        sig.metadata = {
            'strategy': 'SATELLITE', 
            'atr': 0.0,
            'structure': structure_ctx # Pass context for downstream filtering
        }
        return self.apply_asset_personality(symbol, sig)

    def _safe_print(self, msg: str):
        """Thread-safe printing to avoid log corruption."""
        with self._lock:
            print(msg)

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
                self._safe_print(f"[{self.name}] LSTM Brain loaded successfully.")
            except Exception as e:
                self._safe_print(f"[{self.name}] Error loading LSTM: {e}")
        
        # 2. Load XGBoost
        if os.path.exists(xgb_path) and xgb is not None:
            try:
                self.xgb_model = xgb.Booster()
                self.xgb_model.load_model(xgb_path)
                self._safe_print(f"[{self.name}] XGBoost Brain loaded successfully.")
            except Exception as e:
                self._safe_print(f"[{self.name}] Error loading XGBoost: {e}")

        if self.model is None and self.xgb_model is None:
            self._safe_print(f"[{self.name}] All brains missing or deps failed. Running heuristic mode.")

        # 3. OpenVINO Integration (Speed Optimization)
        if self.model is not None and ov is not None and config.USE_OPENVINO:
            try:
                core = ov.Core()
                # Convert Keras model to OpenVINO IR
                ov_model = ov.convert_model(self.model)
                device = "GPU" if config.USE_INTEL_GPU else "CPU"
                self.ov_compiled_model = core.compile_model(ov_model, device)
                self._safe_print(f"[{self.name}] OpenVINO LSTM Backend initialized on {device}.")
            except Exception as e:
                self._safe_print(f"[{self.name}] OpenVINO Setup failed: {e}. Falling back to native TensorFlow.")


    def _extract_ml_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Gathers standard features for XGBoost inference.
        Returns a dictionary of features for the last candle.
        """
        try:
            closes = df['close']
            volumes = df['volume']
            
            # Simple Features
            rsi = 50.0 # Default
            delta = closes.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            if loss.iloc[-1] != 0:
                rs = gain.iloc[-1] / loss.iloc[-1]
                rsi = 100 - (100 / (1 + rs))
                
            # Volatility
            returns = closes.pct_change()
            vol = returns.rolling(20).std().iloc[-1]
            
            # Momentum
            mom_10 = closes.iloc[-1] / closes.iloc[-10] - 1 if len(closes) > 10 else 0
            
            # Volume
            vol_sma = volumes.rolling(20).mean().iloc[-1]
            rvol = volumes.iloc[-1] / vol_sma if vol_sma > 0 else 1.0
            
            return {
                'rsi': rsi,
                'volatility': vol,
                'momentum_10': mom_10,
                'rvol': rvol,
                'close': closes.iloc[-1]
            }
        except Exception:
            return {}

    def predict_trend_lstm(self, prices: pd.Series, symbol: str, entropy_context: float = None) -> float:
        """
        Predict Trend with LSTM + EntroPE.
        """
        if self.model is None or self.scaler is None or len(prices) < 60 or tf is None:
            return 0.53

        # AEHML 2.0: EntroPE (Compute Optimization)
        # If market is Pure Noise (Entropy ~ Max), LSTM finds nothing but hallucinations.
        # Use Simple Heuristic instead to save GPU/CPU cycles.
        if entropy_context and entropy_context > 1.5: 
             # Extreme Chaos -> Random Walk -> Neutral
             # self._safe_print(f"[{self.name}] EntroPE: Skipping LSTM for {symbol} (Entropy {entropy_context:.2f} > 1.5)")
             return 0.5

        # 1. Check Cache
        last_ts = prices.index[-1]
        cache_key = f"{symbol}_{last_ts}" # Using Last Price Timestamp
        if cache_key in self._inference_cache:
            return self._inference_cache[cache_key]

        try:
            data = prices.values[-60:].reshape(-1, 1)
            scaled_data = self.scaler.transform(data)
            x_input = scaled_data.reshape(1, 60, 1)
            
            res = 0.5
            
            # High-Performance OpenVINO Inference if available
            if self.ov_compiled_model:
                out = self.ov_compiled_model(x_input)[0]
                res = float(out[0][0])
            else:
                # High-Performance Functional Call (Avoids Retracing)
                x_tensor = tf.convert_to_tensor(x_input, dtype=tf.float32)
                prob_tensor = self.model(x_tensor, training=False)
                res = prob_tensor.numpy()[0][0]
                
            result = float(res)
            
            # EntroPE Modulation (Post-Process)
            if entropy_context:
                if entropy_context < 0.9: # Highly Ordered
                    # Neural Net is likely very accurate here. Boost conviction.
                    result = 0.5 + (result - 0.5) * 1.1
                    result = min(0.99, max(0.01, result))

            # 2. Update Cache
            # Simple eviction rule
            if len(self._inference_cache) > 2000:
                self._inference_cache.clear()
            self._inference_cache[cache_key] = result
            
            return result
        except Exception as e:
            self._safe_print(f"[{self.name}] Prediction Error: {e}")
            return 0.5

    def predict_trend_xgboost(self, features: dict, entropy_context: float = None) -> float:
        """
        Predict trend using XGBoost with EntroPE (Entropy-Guided Attention).
        """
        if self.xgb_model is None or xgb is None:
            return 0.55
            
        # AEHML 2.0: EntroPE
        # If Entropy is High (Chaotic > 1.3), XGBoost (Tree Logic) often fails or overfits specific noise.
        # We can Dampen confidence.
        if entropy_context and entropy_context > 1.3:
            # High Entropy: Dampen towards 0.5 (Safety)
            # Or skip expensive compute if we were doing feature engineering here.
            pass

        try:
            # Prepare DMatrix
            df_feat = pd.DataFrame([features])
            dmatrix = xgb.DMatrix(df_feat)
            prob = self.xgb_model.predict(dmatrix)[0]
            
            # EntroPE Modulation
            if entropy_context:
                if entropy_context > 1.35: # Chaotic
                     # Dampen: Move prob towards 0.5
                     prob = 0.5 + (prob - 0.5) * 0.5 
                     # Check if we should log this
                     # self._safe_print(f"[{self.name}] EntroPE: Dampened XGB check due to Chaos ({entropy_context:.2f})")
                elif entropy_context < 1.0: # Ordered
                     # Boost: Slightly amplify confidence as structure is reliable
                     prob = 0.5 + (prob - 0.5) * 1.1

            return float(prob)
        except Exception as e:
            self._safe_print(f"[{self.name}] XGBoost Prediction Error: {e}")
            return 0.5
    def get_kalman_estimate(self, symbol: str, window_data: pd.DataFrame) -> float:
        prices = window_data['close']
        log_prices = np.log(prices)
        current_ts = window_data['timestamp'].iloc[-1]
        
        # Try Rust Kalman (faster for batch initialization)
        try:
            import holonic_speed
            
            if symbol not in self.kalman_filters:
                # Batch initialize with Rust (much faster for 100+ data points)
                estimates = holonic_speed.kalman_filter_batch(
                    log_prices.tolist(), 0.0001, 0.001
                )
                # Store last estimate as current state
                self.kalman_filters[symbol] = {
                    'x': estimates[-1],
                    'p': 0.001,
                    'use_rust': True
                }
                self.kalman_last_ts[symbol] = current_ts
            else:
                if current_ts != self.kalman_last_ts.get(symbol):
                    state = self.kalman_filters[symbol]
                    est, x, p, _ = holonic_speed.kalman_filter_single(
                        log_prices.iloc[-1],
                        (state['x'], state['p'], True),
                        0.0001, 0.001
                    )
                    self.kalman_filters[symbol] = {'x': x, 'p': p, 'use_rust': True}
                    self.kalman_last_ts[symbol] = current_ts
            
            kalman_price = float(np.exp(self.kalman_filters[symbol]['x']))
            
        except ImportError:
            # Fallback to Python Kalman
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

    def _analyze_ou_physics(self, symbol: str, prices: np.ndarray) -> Dict[str, float]:
        """
        Quantum Oracle: Analyze mean-reversion physics using OU process logic.
        """
        try:
            # We use log-prices for better stability in financial time series
            log_prices = np.log(prices)
            params = SDEEngine.estimate_ou_parameters(log_prices)
            
            # Convert mu back to price space for easier interpretation
            params['mu_price'] = float(np.exp(params['mu']))
            
            # Calculate current distance from mean in units of sigma
            current_log_p = log_prices.iloc[-1] if hasattr(log_prices, 'iloc') else log_prices[-1]
            dist_sigma = (current_log_p - params['mu']) / params['sigma'] if params['sigma'] > 0 else 0.0
            params['dist_sigma'] = float(dist_sigma)
            
            return params
        except Exception as e:
            if self.DEBUG: self._safe_print(f"[{self.name}] OU Physics Error: {e}")
            return {}

    def get_market_bias(self, sentiment_score: float = 0.0) -> float:
        if not self.symbol_trends:
            return 0.5
            
        # Allow GMB with just 2 symbols (was 25% of assets = ~4)
        if len(self.symbol_trends) < 2:
            return 0.5
            
        bullish_count = sum(1 for trend in self.symbol_trends.values() if trend)
        technical_bias = bullish_count / len(self.symbol_trends)
        
        # Blend with Sentiment (Configurable Weight)
        # Scale Sentiment (-1 to 1) to Bias (0 to 1) -> (S + 1) / 2
        sentiment_bias_norm = (sentiment_score + 1.0) / 2.0
        
        if sentiment_score != 0.0:
            final_bias = (technical_bias * (1 - config.SENTIMENT_WEIGHT)) + (sentiment_bias_norm * config.SENTIMENT_WEIGHT)
            # Log significant divergence
            if abs(final_bias - technical_bias) > 0.1:
                self._safe_print(f"[{self.name}] 📰 Sentiment Modified Bias: {technical_bias:.2f} -> {final_bias:.2f} (Score: {sentiment_score:.2f})")
            return final_bias
            
        return technical_bias

    # === PROJECT AHAB HELPER METHODS ===
    def detect_accumulation(self, window_data: pd.DataFrame, atr: float, avg_atr: float) -> bool:
        """
        Stealth Accumulation: High Volume + Low Volatility.
        Whales absorbing supply without moving price.
        """
        if len(window_data) < 20: return False
        
        # 1. Volume Spike (Quietly High)
        vol_avg = window_data['volume'].rolling(20).mean().iloc[-1]
        rvol = window_data['volume'].iloc[-1] / vol_avg if vol_avg > 0 else 1.0
        
        # 2. Volatility Compression
        # If ATR is decreasing or below average
        is_compressed = atr < (avg_atr * config.WHALE_ACCUMULATION_ATR_FACTOR) if avg_atr > 0 else False
        
        # 3. Price Contained (Tight Range)
        # Check last 3 candles range < 1%
        last_3 = window_data['close'].iloc[-3:]
        range_pct = (last_3.max() - last_3.min()) / last_3.mean()
        is_tight = range_pct < 0.01
        
        return rvol > config.WHALE_ACCUMULATION_RVOL and (is_compressed or is_tight)

    def detect_whale_defense(self, structure_ctx: Dict[str, Any], window_data: pd.DataFrame) -> bool:
        """
        Whale Defense: Price hits Support + Volume Spike + Rejection Wick.
        """
        if not structure_ctx: return False
        
        # 1. Context: At Support
        dist_sup = structure_ctx.get('dist_to_sup_pct', 0.0)
        # Allow slight undercut (bear trap) or near miss
        at_support = -0.005 <= dist_sup <= 0.005 
        
        if not at_support: return False
        
        # 2. Volume Spike
        vol_avg = window_data['volume'].rolling(20).mean().iloc[-1]
        rvol = window_data['volume'].iloc[-1] / vol_avg if vol_avg > 0 else 1.0
        
        if rvol < config.WHALE_DEFENSE_RVOL: return False
        
        # 3. Rejection Wick (Bullish Hammer / Pinbar)
        # Check last candle
        row = window_data.iloc[-1]
        body = abs(row['close'] - row['open'])
        lower_wick = min(row['close'], row['open']) - row['low']
        
        # Wick must be significant (e.g., > body)
        return lower_wick > body

    def calculate_book_pressure(self, book_data: Dict[str, Any]) -> float:
        """
        Order Book Imbalance: Ratio of Bid Volume to Ask Volume.
        Returns: Ratio (e.g. 2.0 = 2x Bids vs Asks)
        """
        if not book_data or not book_data.get('bids') or not book_data.get('asks'):
            return 1.0
            
        # Sum top 20 levels volume
        # Bids: [[price, qty], ...]
        bid_vol = sum([b[1] for b in book_data['bids'][:20]])
        ask_vol = sum([a[1] for a in book_data['asks'][:20]])
        
        if ask_vol == 0: return 99.0 # Infinite support
        
        return bid_vol / ask_vol

    def detect_short_squeeze(self, funding_rate: float, trend: str) -> bool:
        """
        Short Squeeze: Negative Funding (Shorts Paying) + Bullish/Neutral Trend.
        """
        if funding_rate >= 0: return False # Positive funding = Longs paying
        
        # Check if significantly negative
        is_negative = funding_rate < config.WHALE_FUNDING_SQUEEZE_THRESHOLD
        
        # Squeeze usually happens when shorts are trapped in a non-bearish market
        is_trapped = trend != 'BEARISH'
        
        return is_negative and is_trapped

    # === VOL-WINDOW SPECIAL SETUPS ===
        # 2. Confusion Check
        is_confused = 0.45 <= current_xgb_prob <= 0.55
        
        return is_chaotic and is_confused

    def detect_scavenger_trap(self, symbol: str, window_data: pd.DataFrame, structure_ctx: Dict[str, Any]) -> Tuple[bool, str]:
        """
        The Scavenger Trap: "Liquidity Reclaim".
        Detects if price dipped below a Support Level but CLOSED above it.
        (Bear Trap / Spring Pattern).
        """
        if not structure_ctx or len(window_data) < 2: return False, ""
        
        # Get Pivots
        pivots = structure_ctx.get('pivots', {})
        if not pivots: return False, ""
        
        # Current Candle (or just closed)
        row = window_data.iloc[-1]
        close = row['close']
        low = row['low']
        
        # Check Standard Supports
        for level_name in ['S1', 'S2', 'S3']:
            level = pivots.get(level_name)
            if not level: continue
            
            # Logic: 
            # 1. Wick went below level (Liquidity Grab)
            # 2. Body closed above level (Reclaim)
            # 3. Validation: Close is not miles above (e.g. < 0.5% away) to catch it fresh
            if low < level and close > level:
                # Calculate trap magnitude
                trap_depth = (level - low) / level
                reclaim_height = (close - level) / level
                
                # Filter: Significant Wick (>0.1%) but close proximity
                if trap_depth > 0.001 and reclaim_height < 0.005:
                    return True, level_name
                    
        return False, ""

    def detect_pack_laggard(self, symbol: str, ticker_data: Dict[str, Any], pack_stats: Dict[str, float]) -> bool:
        """
        The Pack Hunt: "Lagging Alpha".
        If Market Bias > 0.7 (Strong Bull) AND Assset is lagging behind the Pack (Z-Score < -1.0),
        Signal a "Catch-up" Buy.
        """
        if not pack_stats or not ticker_data: return False
        
        # 1. Check Global Market Bias
        gmb = self.get_market_bias()
        if gmb < 0.70: return False # Only hunt in Strong Bull markets
        
        # 2. Check Z-Score
        # Z = (Asset% - PackMean) / PackStd
        try:
            asset_pct = float(ticker_data.get('percentage', 0.0))
            pack_mean = pack_stats.get('mean', 0.0)
            pack_std = pack_stats.get('std', 1.0)
            
            if pack_std == 0: return False
            
            z_score = (asset_pct - pack_mean) / pack_std
            
            # Laggard Threshold: -1.0 sigmas
            if z_score < -1.0:
                self._safe_print(f"[{self.name}] 🐺 PACK LAGGARD: {symbol} Z-Score {z_score:.2f} (Pct {asset_pct:.2f}% vs Mean {pack_mean:.2f}%)")
                return True
                
        except Exception:
            return False
            
        return False

    def detect_whale_shadow(self, symbol: str, window_data: pd.DataFrame) -> bool:
        """
        The Whale Shadow: "CVD/OBV Divergence".
        Detects BULLISH DIVERGENCE (Absorption).
        Logic: Price makes LOWER LOW, but OBV makes HIGHER LOW.
        """
        if len(window_data) < 30: return False
        
        # 1. Calculate OBV (Proxy for CVD)
        # We calculate it fresh to ensure alignment
        obv = (np.sign(window_data['close'].diff()).fillna(0) * window_data['volume']).cumsum()
        
        # 2. Find Fractals (Lows)
        # We work on a copy to calculate fractals without mutating the main df if it's not present
        df_calc = window_data.copy()
        df_calc['obv'] = obv
        df_calc = self._calculate_fractals(df_calc)
        
        # Get purely the rows that are Fractal Lows
        lows = df_calc[df_calc['fractal_low']]
        
        if len(lows) < 2: return False
        
        # 3. Check Divergence on LAST 2 Lows
        # Note: Fractal at index T is only confirmed at T+2. 
        # So 'last_low' is the most recent confirmed valley.
        last_low = lows.iloc[-1]
        prev_low = lows.iloc[-2]
        
        # Condition A: Price Lower Low (The Bear Trend)
        price_lower_low = last_low['low'] < prev_low['low']
        
        # Condition B: OBV Higher Low (The Hidden Bull)
        obv_higher_low = last_low['obv'] > prev_low['obv']
        
        if price_lower_low and obv_higher_low:
            self._safe_print(f"[{self.name}] 🐋 WHALE SHADOW: {symbol} Divergence Detected! Price LL ({last_low['low']:.2f} < {prev_low['low']:.2f}) vs OBV HL.")
            return True
            
        return False
        
    def check_funding_arb(self, funding_rate: float) -> bool:
        """
        Funding-Arb: High Postive Funding + Strong Market Bias.
        """
        # 1. Funding Check
        is_high_funding = funding_rate > config.VOL_WINDOW_FUNDING_THRESHOLD
        
        # 2. Bias Check
        bias = self.get_market_bias()
        is_supported = bias >= 0.45
        
        return is_high_funding and is_supported


    # === ORDER FLOW PHYSICS (Whale Radar) ===
    def analyze_order_flow(self, symbol: str, observer: Any) -> Dict[str, Any]:
        """
        Analyze TICKS to find Whale Absorption or Exhaustion.
        Returns: {'delta': float, 'signal': 'BULL_ABSORPTION' | 'BEAR_EXHAUSTION' | 'NEUTRAL', 'buy_ratio': float}
        """
        # 1. Fetch Ticks (Sniper Mode - only fetches if cache expired)
        trades = observer.fetch_recent_trades(symbol, limit=500)
        if not trades: return {'delta': 0.0, 'signal': 'NEUTRAL', 'buy_ratio': 0.5}
        
        # 2. Calculate Cumulative Volume Delta
        buy_vol = 0.0
        sell_vol = 0.0
        
        for t in trades:
            if t['side'] == 'buy':
                buy_vol += t['amount']
            else:
                sell_vol += t['amount']
                
        total_vol = buy_vol + sell_vol
        if total_vol == 0: return {'delta': 0.0, 'signal': 'NEUTRAL', 'buy_ratio': 0.5}
        
        net_delta = buy_vol - sell_vol
        buy_ratio = buy_vol / total_vol
        
        # 3. Detect Reversal Signatures
        # We need Price Context. Is Price making Lows?
        # Ideally we compare Delta Trend vs Price Trend.
        # Simple heuristic for single-snapshot:
        
        signal = 'NEUTRAL'
        
        # BULLISH ABSORPTION: 
        # Price is DOWN in last 15m (we assume caller checks context), 
        # BUT Buying Pressure is dominant (> 55%).
        # This means sellers are hitting the bid, but buyers are reloading (Passive Buying).
        # WAIT: Taker Buys > Taker Sells usually means aggressive buying.
        # Absorption is usually: Price Flat/Down + High Buying Volume.
        # Or: Price Hitting Support + Negative Delta (Sellers selling) but Price Stalls.
        
        # Let's use CVD Divergence logic:
        # If Buy Ratio > 0.60 (Aggressive Buying)
        if buy_ratio > 0.60:
            signal = 'AGGRESSIVE_BUYING'
        elif buy_ratio < 0.40:
            signal = 'AGGRESSIVE_SELLING'
            
        return {
            'delta': net_delta,
            'buy_ratio': buy_ratio,
            'signal': signal,
            'vol_processed': total_vol
        }


    def analyze_for_entry(
        self, 
        symbol: str,
        window_data: pd.DataFrame, 
        bb_vals: dict, 
        obv_slope: float,
        metabolism_state: Literal['SCAVENGER', 'PREDATOR'],
        structure_ctx: Dict[str, Any] = None,
        book_data: Dict[str, Any] = None,
        ticker_data: Dict[str, Any] = None,
        pack_stats: Dict[str, float] = None, # New: Mean/Std of Universe
        funding_rate: float = 0.0,
        observer: Any = None
    ):
        self.current_metabolism = metabolism_state
        from .agent_executor import TradeSignal # Ensure import
        is_whale = False # Default initialization
        prices = window_data['close']
        
        # Default Initializations (Fix for UnboundLocalError)
        rsi = 50.0
        rvol = 1.0
        is_whale = False
        whale_reason = []

        # 🔑 KEY 0: SCAVENGER TRAP (Pattern Override)
        # Does this asset look like it just trapped bears at support?
        # Note: window_data usually passed from Trader is the active timeframe (e.g. 15m)
        is_trap, trap_level = self.detect_scavenger_trap(symbol, window_data, structure_ctx)
        if is_trap:
            self._safe_print(f"[{self.name}] 🪤 SCAVENGER TRAP: {symbol} Reclaimed {trap_level}. Triggering Long.")
            price = window_data['close'].iloc[-1]
            sig = TradeSignal(symbol=symbol, direction='BUY', size=1.0, price=price)
            sig.conviction = 0.85 # High Conviction for Structural Reclaims
            sig.metadata = {
                'strategy': 'SCAVENGER_TRAP',
                'trap_level': trap_level,
                'structure': structure_ctx
            }
            return self.apply_asset_personality(symbol, sig)

        # 🔑 KEY 0.5: PACK HUNT (Laggard Alpha)
        if self.detect_pack_laggard(symbol, ticker_data, pack_stats):
            self._safe_print(f"[{self.name}] 🐺 PACK HUNT: {symbol} Catch-up Play Triggered.")
            sig = TradeSignal(symbol=symbol, direction='BUY', size=1.0, price=window_data['close'].iloc[-1])
            sig.conviction = 0.75 # Good conviction, but relies on market beta
            sig.metadata = {
                'strategy': 'PACK_HUNT',
                'structure': structure_ctx
            }
            return self.apply_asset_personality(symbol, sig)

        # 🔑 KEY 0.8: WHALE SHADOW (CVD Divergence)
        if self.detect_whale_shadow(symbol, window_data):
            self._safe_print(f"[{self.name}] 🐋 WHALE SHADOW: {symbol} Absorption Detected. Triggering Long.")
            sig = TradeSignal(symbol=symbol, direction='BUY', size=1.0, price=window_data['close'].iloc[-1])
            sig.conviction = 0.80 # High Conviction for Absorption
            sig.metadata = {
                'strategy': 'WHALE_SHADOW',
                'structure': structure_ctx
            }
            return self.apply_asset_personality(symbol, sig)

        # --- PATCH 4: STRUCTURAL TARGETING (Fractal Flows) ---
        if structure_ctx:
            # 1. Broken Support Check (Falling Knife)
            # Only veto if significantly below support (> 0.2%) to allow for "Reclaiming Support" plays.
            dist_sup = structure_ctx.get('dist_to_sup_pct', 0.0) # usually negative if below
            if structure_ctx.get('structure_mode') == 'BREAKDOWN_DOWN':
                if dist_sup < -0.002: # More than 0.2% below support
                    can_recover = False
                    
                    # === ORDER FLOW INTERVENTION (Reversal Catch) ===
                    if observer:
                        flow = self.analyze_order_flow(symbol, observer)
                        if flow['signal'] == 'AGGRESSIVE_BUYING':
                            self._safe_print(f"[{self.name}] 🌊 FLOW REVERSAL: Catching Knife on {symbol}! (Buy Ratio {flow['buy_ratio']:.2f})")
                            can_recover = True
                            is_whale = True # Treat as Whale Signal
                            whale_reason.append("FLOW_REVERSAL")
                    
                    if not can_recover:
                        # self._safe_print(f"[{self.name}] 🧱 STRUCTURAL VETO {symbol}: Price < Support ({dist_sup*100:.2f}%). (Falling Knife)")
                        # can_long = False # DISABLED: User Requested "Take All Opportunities"
                        pass
                
            # 2. Key Level Resistance Check (Buying the Ceiling)
            # Only allow if 'Whale' is present or if we have at least 0.15% room (Scalpable)
            dist_res = structure_ctx.get('dist_to_res_pct', 1.0)
            if 0.0 < dist_res < 0.0015 and not is_whale: # Reduced from 0.3% to 0.15%
                # self._safe_print(f"[{self.name}] 🧱 STRUCTURAL VETO {symbol}: Too close to Resistance ({dist_res*100:.2f}%) without Whale backup.")
                # can_long = False # DISABLED: User Requested "Take All Opportunities"
                pass
        
        # Optimization: Early exit if trapped (though we need analysis to know direction...
        # unless we pass 'allowed_directions' down? Or just filter at the end.)
        
        # --- PATCH 5: MULTI-TIMEFRAME ALIGNMENT (1H RIVER) ---
        # The 'Macro Trend' (1h) must support the 'Micro Entry' (15m).
        # We expect 'macro_trend' to be passed in 'structure_ctx' or derived.
        macro_trend = structure_ctx.get('macro_trend', 'NEUTRAL') if structure_ctx else 'NEUTRAL'
        
        # --- SECTOR PHYSICS OVERRIDE (Holistic 5b) ---
        # Allow energetic decoupling for memecoins
        sector_override = False
        if symbol in config.MEMECOIN_ASSETS:
             # Calculate local rvol for check
             vol_avg_chk = window_data['volume'].rolling(window=20).mean().iloc[-1]
             vol_curr_chk = window_data['volume'].iloc[-1]
             rvol_chk = vol_curr_chk / vol_avg_chk if vol_avg_chk > 0 else 1.0
             
             if rvol_chk > config.MEMECOIN_PUMP_RVOL:
                 self._safe_print(f"[{self.name}] 🚀 SECTOR PHYSICS: {symbol} Decoupling from Macro (RVOL {rvol_chk:.1f})")
                 sector_override = True
                 is_whale = True
                 whale_reason.append(f"SECTOR_FORCE_RVOL_{rvol_chk:.1f}")
        
        if not sector_override:
            last_state = self.last_macro_state.get(symbol, 'UNKNOWN')
            
            # Volatility & Momentum (Moved UP for Whale Logic)
            returns = prices.pct_change()
            volatility = returns.rolling(14).std().iloc[-1]
            
            # 1. Feature Engineering (Moved UP for Filter Logic)
            # RSI (14)
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1]
            
            # Whale Volume
            vol_avg = window_data['volume'].rolling(window=20).mean().iloc[-1]
            vol_curr = window_data['volume'].iloc[-1]
            rvol = vol_curr / vol_avg if vol_avg > 0 else 1.0
            
            # --- PROJECT AHAB: WHALE DETECTION ---
            # 1. Base Whale (Rocket)
            is_rocket = rvol > config.WHALE_RVOL_THRESHOLD
            
            # 2. Stealth Accumulation
            atr = volatility # approximated by std dev of returns * price? 
            # Better re-calc precise ATR or use passed logic. 
            # Let's use the local volatility as a proxy or calculate ATR properly if needed.
            # We will use the 'volatility' calculated later (std dev of returns) for now.
            # Actually, let's calc simple range-based volatility here for 'detect_accumulation'
            tr = (window_data['high'] - window_data['low']).iloc[-1]
            avg_tr = (window_data['high'] - window_data['low']).rolling(20).mean().iloc[-1]
            
            is_accumulation = self.detect_accumulation(window_data, tr, avg_tr)
            
            # 3. Whale Defense
            is_defense = self.detect_whale_defense(structure_ctx, window_data)
            
            # 4. Front-Running (Book & Funding)
            book_ratio = self.calculate_book_pressure(book_data)
            is_book_skewed = book_ratio > config.WHALE_ORDER_IMBALANCE_RATIO
            
            is_squeeze = self.detect_short_squeeze(funding_rate, macro_trend)
            
            # Combine Signals
            whale_reason = []
            if is_rocket: whale_reason.append("ROCKET")
            if is_accumulation: whale_reason.append("ACCUMULATION")
            if is_defense: whale_reason.append("DEFENSE")
            if is_book_skewed: whale_reason.append(f"BID_WALL({book_ratio:.1f})")
            if is_squeeze: whale_reason.append("SQUEEZE")
            
            is_whale = bool(whale_reason) # True if any whale signal found


            # Default Permissions (Allow All unless restricted)
            can_long = True
            can_short = True

            if macro_trend == 'BULLISH':
                if last_state != 'BULLISH':
                    self._safe_print(f"[{self.name}] 🌊 MACRO FLOW (1H): {symbol} Turned BULLISH.")
                    self.last_macro_state[symbol] = 'BULLISH'
                
                # RECALIBRATION: Enforce Trend congruence (Restrict Shorts in Bull markets)
                can_short = True if is_whale else False
                
            elif macro_trend == 'BEARISH':
                if last_state != 'BEARISH':
                    self._safe_print(f"[{self.name}] 🌊 MACRO FLOW (1H): {symbol} Turned BEARISH.")
                    self.last_macro_state[symbol] = 'BEARISH'
                
                # RECALIBRATION: Enforce Trend congruence (Restrict Longs in Bear markets)
                can_long = True if is_whale else False
        else:
             # No Observer = Safe Defaults
             can_long = True
             can_short = True
             
        # --- PATCH 6: POLYMARKET PATIENCE (The Wait) ---
        # If we are in the first 3 minutes of a 15m candle, we BLOCK standard signals.
        # This filters out the initial "Fakeout" noise.
        # We assume 'minutes_into_candle' is passed in structure_ctx
        minutes_in = structure_ctx.get('minutes_into_candle', 10) if structure_ctx else 10
        if minutes_in < config.POLYMARKET_PATIENCE_MINUTES and self.crisis_score < 0.8:
             # self._safe_print(f"[{self.name}] ⏳ PATIENCE: Waiting for candle settlement ({minutes_in}m < {config.POLYMARKET_PATIENCE_MINUTES}m).")
             # return None # STAND ASIDE - DISABLED PER USER REQUEST
             pass
             
        # --------------------------------------------------
        
        # 🔑 KEY 0.9: EVOLUTIONARY ENSEMBLE (The Three Kings)
        if hasattr(self, 'ensemble') and self.ensemble:
             # Construct minimal indicators required by EvoStrategy
             # Note: EvoStrategy expects a slice (window_data), indicators dict, and portfolio state
             current_p = float(prices.iloc[-1])
             evo_indicators = {'price': current_p, 'rsi': float(rsi)}
             # Oracle assumes no position (Entries only)
             evo_port = {'inventory': 0, 'avg_entry': 0}
             
             try:
                 evo_sig = self.ensemble.on_candle(window_data, evo_indicators, evo_port)
                 if evo_sig.action == 'BUY':
                      self._safe_print(f"[{self.name}] 🧬 ENSEMBLE VOTE: {symbol} Buy Signal ({evo_sig.reason})")
                      sig = GlobalTradeSignal(symbol=symbol, direction='BUY', size=1.0, price=current_p)
                      sig.conviction = 0.65 # Conservative for Evolved Logic
                      sig.metadata = {
                          'strategy': 'ENSEMBLE_EVO', 
                          'reason': evo_sig.reason,
                          'structure': structure_ctx
                      }
                      return self.apply_asset_personality(symbol, sig)
             except Exception as e:
                 # self._safe_print(f"[{self.name}] Evo Error: {e}")
                 pass 

                  # self._safe_print(f"[{self.name}] Evo Error: {e}")
                 pass 

        # 🔑 KEY 1.1: AI DEEP CONFIRMATION (LSTM + XGBoost)
        ai_signal_reason = []
        lstm_prob = 0.5
        xgb_prob = 0.5
        
        # 1. LSTM (Time Series)
        if hasattr(self, 'model') and self.model: 
             lstm_prob = self.predict_trend_lstm(prices, symbol)
             if lstm_prob > 0.75: ai_signal_reason.append(f"LSTM({lstm_prob:.2f})")
             
        # 2. XGBoost (Tabular)
        if hasattr(self, 'xgb_model') and self.xgb_model: 
             feats = self._extract_ml_features(window_data)
             if feats:
                 xgb_prob = self.predict_trend_xgboost(feats)
                 if xgb_prob > 0.75: ai_signal_reason.append(f"XGB({xgb_prob:.2f})")
        
        # 3. Consensus Trigger
        if ai_signal_reason and can_long:
             reason_str = "+".join(ai_signal_reason)
             self._safe_print(f"[{self.name}] 🧠 NEURAL ALERT: {symbol} AI Bullish [{reason_str}]")
             
             # If BOTH agree, High Conviction
             conviction = 0.65
             if lstm_prob > 0.75 and xgb_prob > 0.75: conviction = 0.80
             
             sig = GlobalTradeSignal(symbol=symbol, direction='BUY', size=1.0, price=prices.iloc[-1])
             sig.conviction = conviction
             sig.metadata = {
                 'strategy': 'NEURAL_HYBRID', 
                 'reason': reason_str,
                 'structure': structure_ctx
             }
             return self.apply_asset_personality(symbol, sig)
        
        # --- PATCH 6: TRIGGER D - STRUCTURAL RESONANCE (The Paralysis Breaker) ---
        # Direct override: If Structure is perfect (Bullish Trend + Support Zone), we FIRE.
        # This bypasses the ML/Ensemble hesitation logic.
        sls_zone = structure_ctx.get('sls_zone', 'NEUTRAL') if structure_ctx else 'NEUTRAL'
        tda_critical = structure_ctx.get('tda_critical', False) if structure_ctx else False
        
        if can_long and macro_trend == 'BULLISH' and sls_zone == 'SUPPORT' and not tda_critical:
             self._safe_print(f"[{self.name}] 🏛️ TRIGGER D: Structural Resonance for {symbol} (Bullish + Support). Firing Entry.")
             meta = {'reason': 'STRUCTURAL_RESONANCE', 'structure': structure_ctx, 'is_whale': False}
             # Use current price
             current_price = float(window_data['close'].iloc[-1])
             # Construct signal (GlobalTradeSignal matches usage in Trigger A/B/C)
             sig = GlobalTradeSignal(symbol=symbol, direction='BUY', size=1.0, price=current_price, metadata=meta)
             return self.apply_asset_personality(symbol, sig)
        # ------------------------------------------------------------------------
        
        # --------------------------------------------------
        current_price = float(prices.iloc[-1])
        
        # Volatility & Momentum (Already calculated above)
        returns = prices.pct_change()
        volatility = returns.rolling(14).std().iloc[-1]
        
        # 1. Feature Engineering (Remaining)
        # RSI (14) - Already calculated above
        # delta = prices.diff()
        # gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        # loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1]
        
        # Ensure rsi variable is ALWAYS available (FIX: was only in sector_override path)
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
        # returns = prices.pct_change()
        # volatility = returns.rolling(14).std().iloc[-1]
        
        # 2. Hierarchical Inference (Monolith-V4)
        # 2. Hierarchical Inference (Monolith-V4)
        # === WHALE TRACKING (Volume Physics) ===
        # Already calculated above or need calc
        if sector_override:
             vol_avg = window_data['volume'].rolling(window=20).mean().iloc[-1]
             vol_curr = window_data['volume'].iloc[-1]
             rvol = vol_curr / vol_avg if vol_avg > 0 else 1.0
             is_whale = rvol > config.WHALE_RVOL_THRESHOLD

        lstm_prob = self.predict_trend_lstm(prices, symbol)
        
        xgb_features = {
            'lstm_prob': lstm_prob,
            'rsi': rsi,
            'bb_pct_b': bb_pct_b,
            'macd_hist': macd_hist,
            'volatility': volatility
        }
        xgb_prob = self.predict_trend_xgboost(xgb_features)
        
        # --- ENTROPY INTEGRATION (Chaotic Dampening) ---
        # If the market is CHAOTIC (High Entropy), we reduce our confidence in the ML models.
        entropy_regime = structure_ctx.get('entropy_regime', 'TRANSITION') if structure_ctx else 'TRANSITION'
        
        if entropy_regime == 'CHAOTIC':
            # Dampen probabilities towards 0.5 (Uncertainty) by 50%
            original_xgb = xgb_prob
            xgb_prob = 0.5 + (xgb_prob - 0.5) * 0.5
            lstm_prob = 0.5 + (lstm_prob - 0.5) * 0.5
            # self._safe_print(f"[{self.name}] 🌪️ CHAOS DETECTED ({symbol}): Dampening Confidence (XGB {original_xgb:.2f}->{xgb_prob:.2f})")
        # -----------------------------------------------

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
            self._safe_print(f"[{self.name}] Ensemble Check {symbol}: LSTM({lstm_prob:.2f}) XGB({xgb_prob:.2f})")

        # 7.5 QUANTUM PHYSICS LAYER (SDE)
        ou_params = self._analyze_ou_physics(symbol, window_data['close'].values)
        quantum_conviction = 0.0
        is_quantum_reversion = False
        
        if ou_params:
            # If we are > 2 sigma from mean, we are "Stretched"
            dist_sigma = ou_params.get('dist_sigma', 0.0)
            is_quantum_reversion = abs(dist_sigma) > config.PHYSICS_OU_STRETCH_THRESHOLD
            
            if is_quantum_reversion:
                # Conviction scales with distance and mean-reversion speed (lambda)
                quantum_conviction = min(1.0, abs(dist_sigma) / 4.0)
                # Boost if lambda is high
                if ou_params.get('lambda', 0) > config.PHYSICS_OU_LAMBDA_THRESHOLD:
                    quantum_conviction = min(1.0, quantum_conviction * 1.25)
                
                self._safe_print(f"[{self.name}] ⚛️ QUANTUM STATE: {symbol} dist={dist_sigma:.2f}s, lambda={ou_params.get('lambda'):.2f}")

        # 8. LOG & EXECUTE
        # 8. LOG & EXECUTE (UNIFIED STRATEGY - "UNLEASHED")
        # Merging Scavenger and Predator Logic to allow High Value Trades regardless of mode.
        
        # --- BULLISH TRIGGERS ---
        is_below_middle = current_price <= bb_vals['middle']
        is_oversold = rsi < config.STRATEGY_RSI_ENTRY_MAX
        is_panic_buy = rsi < config.STRATEGY_RSI_PANIC_BUY # e.g. < 40
        
        # Trigger A: Trend Following (High Conviction)
        # Buy if Model is Bullish AND Market Bias is Supportive (or we have Physics Override)
        # Note: We relax GMB check if Whale is present
        trigger_trend_buy = is_bullish and (is_market_bullish or is_whale) and can_long
        
        # Trigger B: Mean Reversion (Dip Buying)
        # Buy if Price is low (Below Middle) AND RSI is safe
        trigger_dip_buy = is_below_middle and is_oversold and can_long
        
        # Trigger C: Panic Buy (Falling Knife Catch)
        # Buy if RSI is extreme (ignore other filters)
        trigger_panic_buy = is_panic_buy and can_long

        # Trigger D: Quantum Reversion (SDE Mean Reversion)
        # Buy if price is significantly below OU Mean
        trigger_quantum_buy = is_quantum_reversion and (ou_params.get('dist_sigma', 0) < 0) and can_long

        if (trigger_trend_buy or trigger_dip_buy or trigger_panic_buy or trigger_quantum_buy):
             # VALIDATION: Kalman Check (Relaxed)
             # Allow if Price < Kalman (Value) OR if High Conviction Bullish (Momentum)
             # OR if it's a Panic Buy (Value is extreme)
             if current_price < kalman_price or high_conv_bullish or trigger_panic_buy or trigger_quantum_buy:
                 reason = "TREND" if trigger_trend_buy else ("DIP" if trigger_dip_buy else "PANIC")
                 if is_whale:
                     main_whale_reason = whale_reason[0] if whale_reason else "FORCE_OVERRIDE"
                     reason = f"WHALE_{main_whale_reason}" # Override reason with primary whale driver
                     self._safe_print(f"[{self.name}] 🐋 WHALE SIGHTING: {symbol} - {', '.join(whale_reason) if whale_reason else 'Manual/Sector Force'}")

                 self._safe_print(f"[{self.name}] 🚀 {symbol} BUY SIGNAL ({reason}) | XGB:{xgb_prob:.2f} GMB:{market_bias:.2f}")
                 
                 
                 meta = {
                      'is_whale': is_whale, 
                      'whale_factors': whale_reason, 
                      'structure': structure_ctx, 
                      'reason': reason,
                      'sde_physics': ou_params,
                      'quantum_conviction': quantum_conviction
                  }
                 sig = GlobalTradeSignal(symbol=symbol, direction='BUY', size=1.0, price=current_price, metadata=meta)
                 return self.apply_asset_personality(symbol, sig)

        # --- BEARISH TRIGGERS ---
        is_above_middle = current_price >= bb_vals['middle']
        is_overbought = rsi > 65
        is_panic_sell = rsi > 75
        
        # Trigger A: Trend Shorting
        trigger_trend_sell = (not is_bullish) and (not is_market_bullish or is_whale) and can_short
        
        # Trigger B: Mean Reversion Short (Top Selling)
        trigger_top_sell = is_above_middle and is_overbought and can_short
        
        # Trigger C: Panic Sell (Blow-off Top)
        trigger_panic_sell = is_panic_sell and can_short
        
        # Trigger D: Quantum Reversion (OU Mean Reversion)
        trigger_quantum_sell = is_quantum_reversion and (ou_params.get('dist_sigma', 0) > 0) and can_short
        
        if (trigger_trend_sell or trigger_top_sell or trigger_panic_sell or trigger_quantum_sell):
            # VALIDATION: Kalman Check (Relaxed)
            # Allow if Price > Kalman (Premium) OR if High Conviction Bearish (Momentum) OR Quantum
            if current_price > kalman_price or high_conv_bearish or trigger_panic_sell or trigger_quantum_sell:
                 reason = "TREND_SELL" if trigger_trend_sell else ("TOP_SELL" if trigger_top_sell else ("PANIC_SELL" if trigger_panic_sell else "QUANTUM_SELL"))
                 meta = {
                     'is_whale': False, 
                     'structure': structure_ctx, 
                     'reason': reason,
                     'sde_physics': ou_params,
                     'quantum_conviction': quantum_conviction
                 }
                 sig = GlobalTradeSignal(symbol=symbol, direction='SELL', size=1.0, price=current_price, metadata=meta)
                 return self.apply_asset_personality(symbol, sig)

        # 9. ENSEMBLE STRATEGY CHECK (The Triumvirate)
        # If the manual/ML triggers above didn't fire, ask the Ancient Kings.
        if self.ensemble:
             # Construct minimal context for Ensemble (Price, Indicators, State)
             ens_indicators = {
                 'rsi': rsi,
                 'atr': volatility, # approximating ATR with vol? Or pass real ATR if available
                 'bb_upper': bb_vals['upper'],
                 'bb_lower': bb_vals['lower'],
                 'bb_middle': bb_vals['middle'],
                 'adx': 25.0 # Placeholder if needed
             }
             
             # Need real ATR if possible. We calculated volatility above.
             # Ideally re-use bb_vals etc.
             
             # Portfolio State: Assume Inventory 0 (We are Entry Oracle)
             # Note: Ensemble handles Exits too, but Oracle is usually Entry-Only.
             # However, if we return a SELL signal here, Executor might process it if we hold the asset?
             # But analyze_for_entry is usually called when lookin' for buys or explicit sells.
             ens_state = {'inventory': 0, 'avg_entry': 0.0} 
             
             # Pass the raw window data (Slice)
             # Ensemble expects DataFrame with columns [close, high, low, open, volume]
             try:
                 ens_sig = self.ensemble.on_candle(window_data, ens_indicators, ens_state)
                 
                 if ens_sig.direction == 'BUY':
                     reason = f"ENSEMBLE_{ens_sig.reason}"
                     self._safe_print(f"[{self.name}] 🎭 ENSEMBLE VOTE: {symbol} BUY ({ens_sig.reason})")
                     
                     meta = {
                         'is_whale': False, 
                         'structure': structure_ctx, 
                         'reason': reason,
                         'ensemble_sl': ens_sig.stop_loss,
                         'ensemble_tp': ens_sig.take_profit
                     }
                     # Map parameters
                     # Note: Ensemble returns size 0.0-1.0. Executor handles sizing logic usually.
                     # We pass size=1.0 and let Governor scale it, OR pass ensemble suggestion?
                     # Let's pass ensemble suggestion in metadata or rely on Governor.
                     # Usually Oracle sends size=1.0 (Full Signal) and Governor reduces.
                     
                     sig = GlobalTradeSignal(
                         symbol=symbol, 
                         direction='BUY', 
                         size=ens_sig.size if ens_sig.size else 1.0, 
                         price=current_price, 
                         metadata=meta
                     )
                     return self.apply_asset_personality(symbol, sig)
             except Exception as e:
                 if self.DEBUG: print(f"[{self.name}] Ensemble Error: {e}")

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

    def _calculate_fractals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bill Williams 5-Bar Fractal Generation.
        Fractal Up: High[i] > High[i-2, i-1, i+1, i+2]
        Fractal Down: Low[i] < Low[i-2, i-1, i+1, i+2]
        """
        if len(df) < 5: return df
        
        # We need a copy to avoid SettingWithCopy warnings on the main DF
        # Actually, we usually want to modify the DF being analyzed.
        
        # Using verifying logic with shifting (Assuming future data is not available, we trigger these on lag)
        # But for historical analysis (df provided), we can look ahead.
        # Note: In live trading, the fractal at T is only confirmed at T+2.
        
        # Highs
        highs = df['high']
        df['fractal_high'] = (highs > highs.shift(1)) & \
                             (highs > highs.shift(2)) & \
                             (highs > highs.shift(-1)) & \
                             (highs > highs.shift(-2))

        # Lows
        lows = df['low']
        df['fractal_low'] = (lows < lows.shift(1)) & \
                            (lows < lows.shift(2)) & \
                            (lows < lows.shift(-1)) & \
                            (lows < lows.shift(-2))
                            
        return df

    def get_structural_context(self, symbol: str, df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        Finds the nearest Structural Support (Lower Fractal) and Resistance (Upper Fractal).
        Returns distances and levels.
        """
        if df.empty or 'fractal_high' not in df.columns:
            df = self._calculate_fractals(df)
            
        # Scan backwards for the last CONFIRMED fractals
        # Note: shift(-2) means we lose the last 2 bars of fractal data.
        # So valid fractals are from index 0 to -3.
        
        valid_df = df.iloc[:-2] # Exclude unconfirmed
        
        last_resistance = valid_df[valid_df['fractal_high']]['high'].iloc[-1] if valid_df['fractal_high'].any() else current_price * 2
        last_support = valid_df[valid_df['fractal_low']]['low'].iloc[-1] if valid_df['fractal_low'].any() else current_price * 0.5
        
        dist_to_res_pct = (last_resistance - current_price) / current_price
        dist_to_sup_pct = (current_price - last_support) / current_price
        
        structure = "RANGE"
        if current_price > last_resistance: structure = "BREAKOUT_UP"
        elif current_price < last_support: structure = "BREAKDOWN_DOWN"
        
        return {
            'resistance_price': last_resistance,
            'support_price': last_support,
            'dist_to_res_pct': dist_to_res_pct,
            'dist_to_sup_pct': dist_to_sup_pct,
            'structure_mode': structure
        }

    def profile_asset_class(self, symbol: str, ticker_data: dict) -> str:
        """
        The Scout's Eye: Classifies an asset based on lightweight Ticker Data.
        Returns: 'ANCHOR', 'ROCKET', or 'DEAD'
        """
        if not ticker_data: return 'DEAD'
        
        # Extract Metrics
        try:
            pct_change = float(ticker_data.get('percentage', 0.0))
            quote_vol = float(ticker_data.get('quoteVolume', 0.0)) # USDT Volume
            
            # 1. ROCKET CHECK (High Energy)
            # Must be moving fast (>3%) with decent liquidity (>$25M)
            if quote_vol > 25_000_000 and abs(pct_change) > 3.0:
                 return 'ROCKET'
                 
            # 2. ANCHOR CHECK (Deep Liquidity)
            # Only promote boring assets if they are MASSIVE (>$500M) or Core (BTC/ETH/SOL)
            is_core = symbol in ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']
            if (quote_vol > 500_000_000) or (is_core and quote_vol > 50_000_000):
                return 'ANCHOR'
                
            return 'DEAD' # Not interesting enough for the Active List
        except:
            return 'DEAD'
