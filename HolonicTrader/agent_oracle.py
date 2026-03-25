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
from collections import deque
from datetime import datetime, timezone, timedelta
from .agent_executor import TradeSignal as GlobalTradeSignal # Fix: Renamed to avoid scope collision
from .sde_engine import SDEEngine # Import Physics Engine
import os
import json
import time
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

# ── Unified Regime Engine Availability Check ────────────────────────────────
try:
    from .unified_regime_engine import get_unified_regime_engine, BehavioralRegime, OperationalRegime
    UNIFIED_REGIME_AVAILABLE = True
except ImportError as _unified_err:
    UNIFIED_REGIME_AVAILABLE = False
    print(f"[{__name__}] Unified Regime Engine not available: {_unified_err}")
# ─────────────────────────────────────────────────────────────────────────────
# Path Hacking to reach sandbox
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path: sys.path.append(parent_dir)
    from sandbox.strategies.ensemble import EnsembleStrategy
except ImportError as e:
    print(f"⚠️ Ensemble Import Failed: {e}")
except ImportError as e:
    print(f"⚠️ Ensemble Import Failed: {e}")
    EnsembleStrategy = None

try:
    import yfinance as yf
except ImportError:
    yf = None

class MacroOracle:
    """
    Detailed Macro-Economic Oracle (Phase 47).
    Fetches Traditional Finance (TradFi) Indices and USDT Dominance to providing 'Stacking Probabilities'.
    Also maintains Gold Arbitrage monitoring.
    """
    def __init__(self):
        self.last_update = 0
        # Indices: S&P 500, Nasdaq, Russell 2000, USDT Dominance (Proxy), VIX, Gold Futures
        # + Orion intermarket: DXY, US10Y, US2Y
        orion_tickers = getattr(config, 'ORION_INTERMARKET_TICKERS', [])
        self.tickers = getattr(config, 'MACRO_TICKERS', ['^GSPC', '^IXIC', '^RUT', 'USDT-USD', '^VIX']) + ['GC=F'] + orion_tickers
        self.macro_state = {
            'risk_on': False,
            'bias_score': 0.0, # -1.0 (Risk Off) to 1.0 (Risk On)
            'details': {},
            'vix': 0.0
        }
        self.last_gold_price = 0.0
        self.last_vix = 0.0  # Cached VIX reading

    def fetch_macro_context(self) -> Dict[str, Any]:
        """
        Fetches global market context from Yahoo Finance.
        Returns a dict with 'bias_score' and 'risk_on' boolean.
        """
        if not yf:
            return self.macro_state

        # Cache for 60s (Macro doesn't move that fast)
        if time.time() - self.last_update < 60:
            return self.macro_state

        try:
            # Multi-fetch
            data = yf.download(self.tickers, period="5d", interval="1d", progress=False)
            
            # Calculate 24h Change %
            # Note: yfinance multi-index df structure can be tricky
            # We look at 'Close' column
            closes = data['Close']
            
            score = 0.0
            details = {}
            
            # 1. INDICES (Equities) - Positive Correlation with Crypto
            equity_tickers = ['^GSPC', '^IXIC', '^RUT']
            for t in equity_tickers:
                if t in closes:
                    series = closes[t].dropna()
                    if len(series) >= 2:
                        change = (series.iloc[-1] - series.iloc[-2]) / series.iloc[-2]
                        details[t] = change
                        
                        if change > 0.002: score += 1.0 # Green > 0.2%
                        elif change < -0.002: score -= 1.0 # Red < -0.2%
            
            # 2. USDT DOMINANCE (Liquidity) - Inversely Correlated
            # If USDT-USD is UP, it means money is flowing INTO stablecoins (Risk Off)
            # If USDT-USD is DOWN (or Volume drops?), actually USDT price is usually stable $1.
            # User suggested "iusdtd" (USDT.D). YF doesn't have USDT.D directly.
            # Proxy: We check 'USDT-USD' Volume or just skip if unreliable.
            # Better Proxy: BTC-USD (if we consider it the leader)
            # For now, we rely on Equities as the primary Risk-On gauge.
            
            # 3. VIX (Fear Index) — Inversely correlated with crypto/equities
            # VIX zones: <15=calm (risk-on), 15-20=normal, 20-25=mild fear, 25-30=fear, 30+=panic (risk-off)
            if '^VIX' in closes:
                vix_series = closes['^VIX'].dropna()
                if len(vix_series) >= 1:
                    vix_level = float(vix_series.iloc[-1])
                    self.last_vix = vix_level
                    vix_panic  = getattr(config, 'VIX_PANIC_THRESHOLD', 30.0)
                    vix_fear   = getattr(config, 'VIX_FEAR_THRESHOLD', 20.0)
                    vix_calm   = getattr(config, 'VIX_CALM_THRESHOLD', 15.0)
                    if vix_level >= vix_panic:  score -= 2.0  # Panic: strong risk-off
                    elif vix_level >= 25:        score -= 1.0  # Elevated fear
                    elif vix_level >= vix_fear:  score -= 0.5  # Mild fear
                    elif vix_level < vix_calm:   score += 0.5  # Calm: risk-on bonus
                    details['^VIX'] = vix_level

            # 4. ORION INTERMARKET: DXY (Dollar Index) & Yields
            # DXY rising → pressure on risk assets (crypto, equities)
            orion_tickers = getattr(config, 'ORION_INTERMARKET_TICKERS', [])
            for t in orion_tickers:
                if t in closes:
                    series = closes[t].dropna()
                    if len(series) >= 2:
                        change = (series.iloc[-1] - series.iloc[-2]) / series.iloc[-2]
                        details[t] = change

            # Normalize Score
            # Max score = 5 (All indices green + calm VIX).
            # We map 5 -> 1.0, -5 -> -1.0
            bias = max(-1.0, min(1.0, score / 5.0))

            self.macro_state = {
                'risk_on': bias > 0.2,
                'bias_score': bias,
                'details': details,
                'vix': self.last_vix
            }
            
            # Update Gold for Arb
            if 'GC=F' in closes:
                self.last_gold_price = float(closes['GC=F'].iloc[-1])
                
            self.last_update = time.time()
            # print(f"[MacroOracle] 🌍 Global Bias: {bias:.2f} ({details})")
            
        except Exception as e:
            print(f"[MacroOracle] ⚠️ Fetch Failed: {e}")
            
        return self.macro_state

    def get_gold_price(self) -> float:
        self.fetch_macro_context()  # Ensure fresh-ish
        return self.last_gold_price

    def get_vix_level(self) -> float:
        """Return the most recently fetched VIX level (cached, updates every 60s)."""
        return self.last_vix


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
        self.bias_history = deque(maxlen=5) # History for Smoothing (Hysteresis)
        
        # Ensemble Strategy (Hall of Fame)
        self.ensemble = None
        
        # Directive REC-2026-01-31: Partial Persistence for Thesis Failures
        self.thesis_failures = {} # {symbol: delta_cycles}

        # P2 FIX 2026-03-05: Signal Cooldown Mechanism
        # Prevents repetitive signals on same asset (PEPE triggered 20+ times without execution)
        self.signal_cooldowns = {}  # {f"{symbol}_{strategy}": expiry_timestamp}
        self.SIGNAL_COOLDOWN = {
            'WHALE_BID_WALL': 300,    # 5 min
            'WHALE_SHADOW': 180,      # 3 min
            'WHALE_SECTOR_FORCE': 300, # 5 min
            'MARKET_OPEN_FVG': 300,   # 5 min
            'TREND': 600,             # 10 min
            'DIP': 300,               # 5 min
            'PACK_HUNT': 180,         # 3 min
            'SCAVENGER_TRAP': 300,    # 5 min
            'FLOW_REVERSAL': 180,     # 3 min
            'DEFAULT': 120            # 2 min
        }

        # === FIX 2026-03-15: Walk-Forward Signal Validation ===
        # Track recent signal performance per symbol and strategy
        self.signal_performance = {}  # {f"{symbol}_{strategy}": {'wins': int, 'losses': int, 'last_5': []}}
        self.SIGNAL_HISTORY_WINDOW = 10  # Track last 10 signals per strategy
        self.MIN_SIGNAL_HISTORY = 3  # Minimum history before validation kicks in
        self.MAX_ACCEPTABLE_LOSS_RATE = 0.60  # Disable signal if >60% losses in window

        # Conviction calibration - map historical win rate to appropriate conviction
        self.conviction_calibration = {
            'STRUCTURAL_RESONANCE': {'win_rate': 0.10, 'target_conviction': 0.35},  # Was 0.8, should be 0.35
            'VOLATILITY_SQUEEZE': {'win_rate': 0.15, 'target_conviction': 0.40},   # Was ~0.7, should be 0.40
            'WHALE_SHADOW': {'win_rate': 0.45, 'target_conviction': 0.55},
            'WHALE_SQUEEZE': {'win_rate': 0.50, 'target_conviction': 0.60},
            'NEURAL_HYBRID': {'win_rate': 0.40, 'target_conviction': 0.50},
            'DEFAULT': {'win_rate': 0.35, 'target_conviction': 0.45}
        }

        # Load Brains
        self.load_brains()

        # Initialize Macro Oracle
        self.macro_oracle = MacroOracle()

        # === FIX 2026-03-14: SMCE Regime for Macro Stack Weight Adaptation ===
        self._current_smce_regime = 'HARVEST'  # Default, will be synced by TraderNexus

        # ── Unified Regime Engine (Phase 52) ─────────────────────────────────
        # Replaces HolonicAdaptor with unified regime system
        self.unified_regime = None
        if UNIFIED_REGIME_AVAILABLE:
            try:
                from .unified_regime_engine import get_unified_regime_engine
                self.unified_regime = get_unified_regime_engine()
                print(f"[{self.name}] 🌐 Unified Regime Engine initialized (Phase 52)")
            except Exception as e:
                print(f"[{self.name}] ⚠️ Unified Regime init failed: {e}")
                self.unified_regime = None
        else:
            # Fallback to old HolonicAdaptor if unified not available
            try:
                from .holonic_adaptor import get_holonic_adaptor
                self.holonic_adaptor = get_holonic_adaptor()
                print(f"[{self.name}] 🌊 Holonic Adaptor initialized (Market Resonance Layer)")
            except Exception as e:
                print(f"[{self.name}] ⚠️ Holonic Adaptor init failed: {e}")
                self.holonic_adaptor = None
        # ─────────────────────────────────────────────────────────────────────

    def process_holonic_signal(self, symbol: str, signal_type: str,
                               signal_data: Any, market_data: Dict) -> Any:
        """
        FIX 2026-03-14: Unified Signal Routing through Regime Engine

        All signal generators (whale, arb, etc.) pass through here for:
        1. Regime-adaptive parameter adjustment
        2. Entropy-based filtering
        3. Unified permission check
        4. Learning feedback collection

        Returns: Processed signal data (may be None if filtered out)
        """
        # Try unified regime engine first
        if self.unified_regime:
            return self._process_unified_signal(symbol, signal_type, signal_data, market_data)
        # Fallback to old holonic adaptor
        elif hasattr(self, 'holonic_adaptor') and self.holonic_adaptor:
            return self._process_holonic_signal_legacy(symbol, signal_type, signal_data, market_data)
        else:
            return signal_data  # Pass through if no regime engine available

    def _process_unified_signal(self, symbol: str, signal_type: str,
                                signal_data: Any, market_data: Dict) -> Any:
        """
        Process signal through Unified Regime Engine.
        """
        try:
            # 1. Update regime with market data
            prices = market_data.get('prices', np.array([100.0]))
            state = self.unified_regime.update(
                prices=prices,
                volumes=market_data.get('volumes'),
                atr=market_data.get('atr'),
                structure=market_data.get('structure', 'NEUTRAL'),
                liquidity_status=market_data.get('liquidity_status', 'healthy'),
                correlation_idx=market_data.get('correlation_idx', 0.3),
                drawdown_breach=False,
            )

            # 2. Calculate signal conviction (use existing or default)
            if isinstance(signal_data, dict):
                conviction = signal_data.get('conviction', 0.65)
            else:
                conviction = 0.65  # Default

            # 3. Unified permission check
            allowed, reason = self.unified_regime.should_allow_entry(
                conviction=conviction,
                symbol=symbol,
            )

            if not allowed:
                print(f"[{self.name}] 🌐 UNIFIED VETO: {symbol} {signal_type} - {reason}")
                return None

            # 4. Apply regime-adaptive adjustments to signal
            if isinstance(signal_data, dict):
                # Adjust conviction based on regime confidence
                regime_boost = state.confidence * 0.2
                signal_data['conviction'] = min(1.0, conviction + regime_boost)

                # Store regime info for later use
                signal_data['regime'] = state.behavioral.value
                signal_data['operational_regime'] = state.operational.value
                signal_data['adaptive_params'] = {
                    'trailing_stop_mult': state.trailing_stop_mult,
                    'profit_targets': state.profit_targets,
                    'size_modifier': state.size_modifier,
                    'max_leverage': state.max_leverage,
                    'min_conviction': state.min_conviction,
                }

                print(f"[{self.name}] 🌐 UNIFIED SIGNAL: {symbol} {signal_type} approved "
                      f"({state.behavioral.value} + {state.operational.value}, conv={state.min_conviction:.2f})")

            return signal_data

        except Exception as e:
            print(f"[{self.name}] ⚠️ Unified signal processing failed: {e}")
            return signal_data  # Pass through on error

    def _process_holonic_signal_legacy(self, symbol: str, signal_type: str,
                                       signal_data: Any, market_data: Dict) -> Any:
        """
        Legacy Holonic Adaptor signal processing (fallback).
        """
        if self.holonic_adaptor is None:
            return signal_data

        try:
            # 1. Get adaptive parameters for current regime
            adaptive_params = self.holonic_adaptor.get_adaptive_parameters(market_data)

            # 2. Calculate signal conviction (use existing or default)
            if isinstance(signal_data, dict):
                conviction = signal_data.get('conviction', 0.65)
            else:
                conviction = 0.65  # Default

            # 3. Holonic permission check
            allowed, reason = self.holonic_adaptor.should_allow_trade(
                signal_type=signal_type,
                current_conviction=conviction,
                metadata={'symbol': symbol, 'signal_data': signal_data}
            )

            if not allowed:
                print(f"[{self.name}] 🌊 HOLONIC VETO: {symbol} {signal_type} - {reason}")
                return None

            # 4. Apply regime-adaptive adjustments to signal
            if isinstance(signal_data, dict):
                # Adjust conviction based on regime
                regime_boost = adaptive_params.get('regime_confidence', 0.5) * 0.2
                signal_data['conviction'] = min(1.0, conviction + regime_boost)

                # Store regime info for later use
                signal_data['regime'] = adaptive_params['regime']
                signal_data['adaptive_params'] = adaptive_params

                print(f"[{self.name}] 🌊 HOLONIC SIGNAL: {symbol} {signal_type} approved ({adaptive_params['regime']})")

            # 5. Record for learning (will be updated with trade outcome later)
            # This builds the holonic memory over time

            return signal_data

        except Exception as e:
            print(f"[{self.name}] ⚠️ Holonic signal processing failed: {e}")
            return signal_data  # Pass through on error

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
            
            # --- SESSION 3 FIX: BRAIN PERSISTENCE ---
            # Save to disk so it survives restarts/crashes
            try:
                # Assuming model is an xgb.Booster or compatible
                if hasattr(model, 'save_model'):
                    model.save_model('xgboost_model.json')
                    print(f"[{self.name}] 💾 Brain Saved to Disk (xgboost_model.json).")
                else:
                    print(f"[{self.name}] ⚠️ Brain not savable: No save_model method.")
            except Exception as e:
                print(f"[{self.name}] ❌ Brain Save Failed: {e}")
            # ----------------------------------------

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

    def _check_signal_cooldown(self, symbol: str, strategy: str) -> bool:
        """
        P2 FIX 2026-03-05: Check if signal type is on cooldown for this symbol.
        Returns True if signal is ALLOWED, False if on cooldown.
        
        FIX 2026-03-22: Tightened cooldown to reduce signal spam (6,745 signals → target 50-100)
        """
        key = f"{symbol}_{strategy}"
        now = time.time()

        # Clean up expired cooldowns
        expired = [k for k, v in self.signal_cooldowns.items() if v <= now]
        for k in expired:
            del self.signal_cooldowns[k]

        # Check if this signal is on cooldown
        if key in self.signal_cooldowns:
            remaining = self.signal_cooldowns[key] - now
            return False  # On cooldown

        return True  # Allowed
    
    def _set_signal_cooldown(self, symbol: str, strategy: str):
        """
        Set cooldown for signal type on symbol.
        
        FIX 2026-03-22: Extended cooldown to reduce spam
        - Was: 300 seconds (5 minutes)
        - Now: 900 seconds (15 minutes) for normal strategies
        - 1800 seconds (30 minutes) for VOLATILITY_SQUEEZE (too trigger-happy)
        """
        key = f"{symbol}_{strategy}"
        
        # Extended cooldowns to reduce signal spam
        if strategy == 'VOLATILITY_SQUEEZE' or strategy == 'VOLATILITY_COMPRESSION':
            cooldown_seconds = 1800  # 30 minutes for volatility strategies
        elif strategy == 'SATELLITE':
            cooldown_seconds = 1200  # 20 minutes for satellite
        else:
            cooldown_seconds = 900  # 15 minutes for others (was 5 min)
        
        self.signal_cooldowns[key] = time.time() + cooldown_seconds

    # === FIX 2026-03-15: Walk-Forward Validation Methods ===

    def _record_signal_outcome(self, symbol: str, strategy: str, outcome: str):
        """
        FIX 2026-03-15: Record signal outcome for walk-forward validation.
        outcome: 'WIN' or 'LOSS'
        """
        key = f"{symbol}_{strategy}"

        if key not in self.signal_performance:
            self.signal_performance[key] = {
                'wins': 0,
                'losses': 0,
                'last_5': [],
                'total': 0
            }

        perf = self.signal_performance[key]
        perf['total'] += 1

        if outcome == 'WIN':
            perf['wins'] += 1
            perf['last_5'].append(1)
        else:
            perf['losses'] += 1
            perf['last_5'].append(0)

        # Keep only last N results
        if len(perf['last_5']) > self.SIGNAL_HISTORY_WINDOW:
            perf['last_5'].pop(0)

    def _validate_signal_walk_forward(self, symbol: str, strategy: str) -> tuple[bool, float]:
        """
        FIX 2026-03-15: Walk-forward validation for signal generation.

        Returns:
            tuple: (is_valid, recent_win_rate)
            - is_valid: False if signal should be disabled (>60% loss rate)
            - recent_win_rate: Win rate over last N signals
        """
        key = f"{symbol}_{strategy}"

        if key not in self.signal_performance:
            return True, 0.5  # No history yet, allow signal

        perf = self.signal_performance[key]

        # Need minimum history before validation kicks in
        if len(perf['last_5']) < self.MIN_SIGNAL_HISTORY:
            return True, 0.5

        # Calculate recent win rate
        recent_win_rate = sum(perf['last_5']) / len(perf['last_5'])
        recent_loss_rate = 1.0 - recent_win_rate

        # Block signal if loss rate exceeds threshold
        if recent_loss_rate > self.MAX_ACCEPTABLE_LOSS_RATE:
            return False, recent_win_rate

        return True, recent_win_rate

    def _get_calibrated_conviction(self, strategy: str, recent_win_rate: float = None) -> float:
        """
        FIX 2026-03-15: Get conviction calibrated to actual historical performance.

        Instead of hardcoded 0.8 conviction, use actual win rate to set conviction.
        """
        if strategy in self.conviction_calibration:
            calib = self.conviction_calibration[strategy]

            # If we have recent win rate, use it directly
            if recent_win_rate is not None and recent_win_rate > 0:
                # Map win rate to conviction (0.3-0.9 range)
                calibrated = 0.3 + (recent_win_rate * 0.6)
                return min(0.9, max(0.3, calibrated))

            # Otherwise use target conviction from calibration table
            return calib.get('target_conviction', 0.45)

        # Default conviction for unknown strategies
        return self.conviction_calibration['DEFAULT']['target_conviction']

    def _get_adaptive_conviction_floor(self, symbol: str, strategy: str,
                                        confluence_count: int = 1,
                                        recent_win_rate: float = None,
                                        current_regime: str = None) -> float:
        """
        FIX 2026-03-24: Dynamic Conviction Engine - Adaptive conviction floors.

        Instead of static conviction floors, dynamically adjust based on:
        1. Confluence score (more signals = lower floor)
        2. Recent performance (hot streak = lower floor)
        3. Regime alignment (strategy matches regime = lower floor)

        Args:
            symbol: Trading symbol (e.g., 'SHIB/USDT')
            strategy: Signal strategy (e.g., 'DIP', 'TREND', 'WHALE')
            confluence_count: Number of confirming signals (1 = single, 3+ = strong)
            recent_win_rate: Recent win rate for this symbol/strategy (0.0-1.0)
            current_regime: Current market regime (e.g., 'ORDERED', 'CHAOTIC')

        Returns:
            float: Adaptive conviction floor (0.35-0.70 range)
        """
        # Start with base floor or symbol-specific floor
        symbol_floors = getattr(config, 'SYMBOL_CONVICTION_FLOORS', {})
        base_floor = symbol_floors.get(symbol, getattr(config, 'CONVICTION_FLOOR_BASE', 0.50))

        # === ADJUSTMENT 1: Confluence Bonus ===
        # More confirming signals = lower conviction floor required
        confluence_bonus = 0.0
        if confluence_count > 1:
            extra_signals = confluence_count - 1
            max_bonus = getattr(config, 'CONFLUENCE_MAX_BONUS', 0.15)
            per_signal = getattr(config, 'CONFLUENCE_BONUS_PER_SIGNAL', 0.05)
            confluence_bonus = min(extra_signals * per_signal, max_bonus)
            base_floor -= confluence_bonus

        # === ADJUSTMENT 2: Performance Adjustment ===
        # Hot streak (WR > 60%) = lower floor, Cold streak (WR < 40%) = higher floor
        if recent_win_rate is not None:
            bonus_threshold = getattr(config, 'PERFORMANCE_BONUS_THRESHOLD', 0.60)
            penalty_threshold = getattr(config, 'PERFORMANCE_PENALTY_THRESHOLD', 0.40)
            adjustment_pct = getattr(config, 'PERFORMANCE_ADJUSTMENT_PCT', 0.10)

            if recent_win_rate >= bonus_threshold:
                # Hot streak: 10% discount on floor
                base_floor *= (1.0 - adjustment_pct)
            elif recent_win_rate <= penalty_threshold:
                # Cold streak: 10% premium on floor
                base_floor *= (1.0 + adjustment_pct)

        # === ADJUSTMENT 3: Regime Alignment ===
        # Strategy matching current regime = lower floor
        if current_regime:
            alignment_map = getattr(config, 'STRATEGY_REGIME_ALIGNMENT', {})
            aligned_regimes = alignment_map.get(strategy, [])
            alignment_bonus = getattr(config, 'REGIME_ALIGNMENT_BONUS', 0.05)
            misalignment_penalty = getattr(config, 'REGIME_MISALIGNMENT_PENALTY', 0.10)

            if current_regime in aligned_regimes:
                # Strategy aligned with regime = 5% discount
                base_floor *= (1.0 - alignment_bonus)
            else:
                # Strategy misaligned = 10% premium
                base_floor *= (1.0 + misalignment_penalty)

        # === SAFETY NETS: Hard floor and soft ceiling ===
        hard_min = getattr(config, 'CONVICTION_FLOOR_HARD_MINIMUM', 0.35)
        soft_max = getattr(config, 'CONVICTION_FLOOR_SOFT_MAXIMUM', 0.70)

        adaptive_floor = max(hard_min, min(soft_max, base_floor))

        return adaptive_floor

    def _set_signal_cooldown(self, symbol: str, strategy: str):
        """Set cooldown for a signal type after it's generated."""
        key = f"{symbol}_{strategy}"
        cooldown_sec = self.SIGNAL_COOLDOWN.get(strategy, self.SIGNAL_COOLDOWN['DEFAULT'])
        self.signal_cooldowns[key] = time.time() + cooldown_sec
        if self.DEBUG:
            self._safe_print(f"[{self.name}] ⏱️ COOLDOWN SET: {symbol} {strategy} for {cooldown_sec}s")

    def set_emotional_bias(self, fear: float, greed: float):
        """
        Receive Emotional Feedback from the Neural Network (Trader).
        Fear = Drawdown (0.0 - 1.0)
        Greed = Risk Multiplier (0.5 - 2.0)
        """
        with self._lock:
            self.emotional_state = {'fear': fear, 'greed': greed}
            
    def apply_asset_personality(self, symbol: str, signal: Any, prices: list = None) -> Any:
        """
        Apply Asset-Specific Rules (The Physics Layer).
        Modifies or Vetos signals based on asset class.
        Also Injects Dynamic Take Profit Targets (PIPs).
        """
        if not signal: return None
        
        # --- DYNAMIC TAKE PROFIT (PIPs) ---
        # User Request: "How do we make our system take profits through pips"
        # We find the nearest Structural Pivot to use as a Target.
        if prices and len(prices) > 20:
            try:
                # 1. Extract ZigZag Pivots
                pips_idx = self.extract_pips(prices, n_pips=10)
                pip_vals = [prices[i] for i in pips_idx]
                current_p = signal.price
                
                target_pip = None
                
                if signal.direction == 'BUY':
                    # Find lowest PIP that is ABOVE current price (Resistance)
                    # We sort ascending to find the nearest one.
                    candidates = [p for p in pip_vals if p > (current_p * 1.01)] # >1% away
                    candidates.sort()
                    if candidates:
                        target_pip = candidates[0]
                        
                elif signal.direction == 'SELL':
                    # Find highest PIP that is BELOW current price (Support)
                    # We sort descending to find the nearest one.
                    candidates = [p for p in pip_vals if p < (current_p * 0.99)] # >1% away
                    candidates.sort(reverse=True)
                    if candidates:
                        target_pip = candidates[0]
                        
                # 2. Inject Target
                if target_pip:
                    # Don't overwrite if Ensemble already provided a specific plan
                    if not signal.take_profit_price:
                        signal.take_profit_price = target_pip
                        signal.metadata['take_profit_type'] = 'STRUCTURAL_PIP'
                        # print(f"[{self.name}] 🎯 PIP TARGET: {symbol} TP set to ${target_pip:.2f} (Structure)")
                        
            except Exception as e:
                pass # Non-critical
                
        # --- NEW: STRUCTURE DRIVES TARGETS ---
        # User Request: "structure drives targets"
        # Enforce Structural TP if enabled and available
        if getattr(config, 'STRUCTURE_DRIVES_TARGETS', True):
             structure = signal.metadata.get('structure') or {}
             pivots = structure.get('pivots', {}) if structure else {}
             current_p = signal.price
             
             struct_tp = None
             
             # Find nearest structure level offering > 1.5R? 
             # For now, just find the next logical level.
             
             if signal.direction == 'BUY':
                 # R1, R2, R3...
                 levels = [v for k,v in pivots.items() if k.startswith('R') and v > current_p * 1.01]
                 levels.sort() # Lowest valid resistance first
                 if levels: 
                     struct_tp = levels[0]
                     
             elif signal.direction == 'SELL':
                 # S1, S2, S3...
                 levels = [v for k,v in pivots.items() if k.startswith('S') and v < current_p * 0.99]
                 levels.sort(reverse=True) # Highest valid support first
                 if levels:
                     struct_tp = levels[0]
                     
             if struct_tp:
                 # Override
                 signal.take_profit_price = struct_tp
                 signal.metadata['take_profit_type'] = 'STRUCTURE_PIVOT'
                 # print(f"[{self.name}] 🏗️ STRUCTURE TARGET: {symbol} TP -> ${struct_tp:.2f}")

        
        # --- EMOTIONAL OVERRIDE (Amygdala) ---
        fear = self.emotional_state.get('fear', 0.0)
        greed = self.emotional_state.get('greed', 1.0)
        
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

        # === FIX 2026-03-24: DYNAMIC CONVICTION ENGINE ===
        # Replace static conviction floors with adaptive floors based on:
        # 1. Confluence count (more signals = lower floor)
        # 2. Recent win rate (hot streak = lower floor)
        # 3. Regime alignment (strategy matches regime = lower floor)

        # Get signal metadata for confluence count
        meta = signal.metadata
        confluence_count = meta.get('confirmation_score', 1)
        strategy = meta.get('strategy', 'DIP')  # Default to DIP if not specified
        recent_wr = meta.get('recent_win_rate', None)
        current_regime = meta.get('regime', None)

        # Calculate adaptive floor
        min_conviction = self._get_adaptive_conviction_floor(
            symbol=symbol,
            strategy=strategy,
            confluence_count=confluence_count,
            recent_win_rate=recent_wr,
            current_regime=current_regime
        )

        # Log the adaptive floor calculation for debugging
        floor_reason = f"Base={getattr(config, 'CONVICTION_FLOOR_BASE', 0.50):.2f}"
        if confluence_count > 1:
            floor_reason += f", Confluence={confluence_count}"
        if recent_wr is not None:
            floor_reason += f", WR={recent_wr:.0%}"
        if current_regime:
            floor_reason += f", Regime={current_regime}"

        if signal.conviction < min_conviction:
            self._safe_print(f"[{self.name}] 🚫 CONVICTION FLOOR: {symbol} {signal.direction} Conviction {signal.conviction:.2f} < Min {min_conviction:.2f} ({floor_reason})")
            return None
        else:
            # Log approval with adaptive floor details
            self._safe_print(f"[{self.name}] ✅ CONVICTION OK: {symbol} Conviction {signal.conviction:.2f} >= Floor {min_conviction:.2f} ({floor_reason})")
        # === END DYNAMIC CONVICTION ENGINE ===
            
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

            # RVOL Check - BYPASS for whale signals
            # Whale detection implies institutional volume, so RVOL filter is redundant
            is_whale = meta.get('is_whale', False) or 'WHALE' in meta.get('reason', '').upper()
            if 'rvol_threshold' in profile and not is_whale:
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
                # FIX 2026-03-20: Use config.STRATEGY_RSI_OVERBOUGHT (set by evolution)
                # instead of hardcoded 70.0
                rsi_ob = getattr(config, 'STRATEGY_RSI_OVERBOUGHT', 70.0)
                if rsi > rsi_ob:
                     if is_resonance and rsi < (rsi_ob + 5.0): # Allow slightly higher for Resonance
                          pass 
                     elif signal.metadata.get('is_whale', False):
                          print(f"[{self.name}] 🐋 RSI OVERRIDE: {symbol} Whale ignoring overbought RSI ({rsi:.1f})")
                     elif self.current_metabolism == 'PREDATOR' and signal.conviction > 0.7:
                          print(f"[{self.name}] 🦖 RSI OVERRIDE: {symbol} PREDATOR momentum ignoring overbought RSI ({rsi:.1f})")
                     else:
                          print(f"[{self.name}] 🧬 {symbol} FILTER: Overbought (RSI {rsi:.1f} > {rsi_ob:.0f}). IGNORED.")
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
                limit = getattr(config, 'FAIR_WEATHER_MIN_BIAS', config.GMB_THRESHOLD - 0.05)
                if gmb < limit:
                    print(f"[{self.name}] ☁️ FAIR WEATHER VETO: {symbol} Long blocked (Bias {gmb:.2f} < {limit:.2f})")
                    return None
                
        # 6. CRISIS PROTOCOL (Macro Strategy)
        # Assuming self.crisis_score is updated by TraderHolon
        # Adjusted threshold to 0.75 (User Request - Actual Conditions)
        # FIX: Dynamic threshold based on account tier - SMALL accounts need more risk tolerance
        # P3 FIX 2026-03-05: Regime-aware crisis thresholds for AGGRESSIVE mode alignment
        # Current crisis score 1.00 was blocking all longs despite AGGRESSIVE mode (95% risk allocation)
        
        # Get current growth mode for regime-aware thresholds
        _growth_mode = getattr(config, 'GROWTH_PHASE', 'AGGRESSIVE')
        _equity = getattr(self, '_last_known_equity', 100.0)
        _is_small_tier = _equity < 500.0  # SMALL tier ceiling
        
        # Regime-aware crisis thresholds (AGGRESSIVE mode allows more risk)
        CRISIS_THRESHOLDS = {
            'AGGRESSIVE': {'block_longs': 0.85, 'reduce_size': 0.60},  # Only block at extreme
            'BALANCED':   {'block_longs': 0.75, 'reduce_size': 0.50},
            'DEFENSIVE':  {'block_longs': 0.50, 'reduce_size': 0.30},
        }
        crisis_config = CRISIS_THRESHOLDS.get(_growth_mode, CRISIS_THRESHOLDS['BALANCED'])
        
        if self.crisis_score > crisis_config['block_longs']:
            # A. FLIGHT TO SAFETY (Boost Gold/BTC)
            if symbol in getattr(config, 'CRISIS_SAFE_HAVENS', []) and signal.direction == 'BUY':
                signal.conviction = min(1.0, signal.conviction * 1.2) # +20% Boost
                signal.metadata['crisis_boost'] = True
                print(f"[{self.name}] ☢️ CRISIS BOOST: {symbol} Conviction increased to {signal.conviction:.2f}")

            # B. RISK OFF (Block Meme Longs) - RELAXED for SMALL tier
            elif symbol in getattr(config, 'CRISIS_RISK_ASSETS', []) and signal.direction == 'BUY':
                # SMALL tier: Allow trades with higher crisis threshold (0.90 instead of 0.85)
                # This allows small accounts to take calculated risks for growth
                if _is_small_tier and self.crisis_score < 0.90:
                    print(f"[{self.name}] ☢️ CRISIS OVERRIDE: {symbol} Small Tier allowed (Score {self.crisis_score:.2f} < 0.90)")
                else:
                    print(f"[{self.name}] ☢️ CRISIS VETO: {symbol} Long blocked (Crisis Score {self.crisis_score:.2f} > {crisis_config['block_longs']:.2f} [{_growth_mode}])")
                    return None
        elif self.crisis_score > crisis_config['reduce_size']:
            # Size reduction zone - reduce conviction but don't block
            if signal.direction == 'BUY' and symbol not in getattr(config, 'CRISIS_SAFE_HAVENS', []):
                signal.conviction *= 0.7  # -30% conviction
                print(f"[{self.name}] ☢️ CRISIS CAUTION: {symbol} conviction reduced (Score {self.crisis_score:.2f})")
        
        # 6b. WHALE TRACKING (Volume Physics)
        if signal.metadata.get('is_whale', False):
            # === FIX 2026-03-12: Whale Structure Gate ===
            # Whales can no longer force entries against structure
            structure = signal.metadata.get('structure', {})
            sls_zone = structure.get('sls_zone', 'NEUTRAL')
            macro_trend = structure.get('macro_trend', 'NEUTRAL')
            signal_direction = 'BUY' if 'Long' in signal.metadata.get('reason', '') or 'BID' in signal.metadata.get('reason', '') else 'SELL'

            # LONG Whale: Must be at SUPPORT (not NEUTRAL, not RESISTANCE)
            if signal_direction == 'BUY':
                if sls_zone == 'SUPPORT':
                    pass  # Valid whale entry
                elif sls_zone == 'NEUTRAL':
                    # Allow ONLY if macro trend is BULLISH
                    if macro_trend != 'BULLISH':
                        print(f"[{self.name}] 🐋🚫 WHALE STRUCTURE GATE: {symbol} Long rejected (Zone: {sls_zone} != SUPPORT, Macro: {macro_trend})")
                        return None
                else:
                    # RESISTANCE -> Block
                    print(f"[{self.name}] 🐋🚫 WHALE STRUCTURE GATE: {symbol} Long rejected (Zone: {sls_zone} != SUPPORT)")
                    return None

            # SHORT Whale: Must be at RESISTANCE
            elif signal_direction == 'SELL':
                if sls_zone == 'RESISTANCE':
                    pass  # Valid whale entry
                elif sls_zone == 'NEUTRAL':
                    if macro_trend != 'BEARISH':
                        print(f"[{self.name}] 🐋🚫 WHALE STRUCTURE GATE: {symbol} Short rejected (Zone: {sls_zone} != RESISTANCE, Macro: {macro_trend})")
                        return None
                else:
                    # SUPPORT -> Block
                    print(f"[{self.name}] 🐋🚫 WHALE STRUCTURE GATE: {symbol} Short rejected (Zone: {sls_zone} != RESISTANCE)")
                    return None
            # === END WHALE STRUCTURE GATE ===

            if getattr(config, 'WHALE_REQUIRES_STRUCTURE_SUPPORT', True):
                structure = signal.metadata.get('structure', {})
                sls_zone = structure.get('sls_zone', 'NEUTRAL')
                
                # Check Overrides
                allow_neutral = getattr(config, 'WHALE_STRUCTURE_GATE_ALLOW_NEUTRAL', False)
                allow_bid_wall = getattr(config, 'WHALE_ALLOW_NEUTRAL_WITH_BID_WALL', False)
                
                # Only enforce for LONGs (Whales buying support)
                if signal.direction == 'BUY':
                    if sls_zone == 'SUPPORT':
                         pass # Valid
                    elif sls_zone == 'NEUTRAL':
                         # If NEUTRAL, check overrides
                         bid_wall_score = signal.metadata.get('bid_wall', 0.0)
                         is_wall_valid = allow_bid_wall and bid_wall_score > 5.0
                         
                         if allow_neutral or is_wall_valid or signal.conviction > 0.9:
                             pass # Allowed via Override
                         else:
                             print(f"[{self.name}] 🐋🚫 WHALE STRUCTURE GATE: {symbol} Long rejected (Zone: {sls_zone} != SUPPORT)")
                             return None
                    else:
                         # RESISTANCE etc -> Block
                         print(f"[{self.name}] 🐋🚫 WHALE STRUCTURE GATE: {symbol} Long rejected (Zone: {sls_zone})")
                         return None
            
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
                 # FIX 2026-03-08: Protect whale convictions from excessive penalties
                 is_whale = signal.metadata.get('is_whale', False)
                 penalty = (0.4 - global_bias) * 2.0 # SEVERE Penalty for fighting the bear

                 if is_whale:
                     # Whales get 50% penalty reduction (they have their own research)
                     penalty *= 0.5
                     original_conv = signal.conviction
                     signal.conviction = max(0.30, signal.conviction - penalty) # Floor at 0.30 for whales
                 else:
                     original_conv = signal.conviction
                     signal.conviction -= penalty

                 if self.DEBUG:
                     print(f"[{self.name}] 📉 PROBABILITY ADJUST: {symbol} Long Conviction {original_conv:.2f}->{signal.conviction:.2f} (Bear Bias {global_bias:.2f})")
                     
        # If conviction drops <= 0, signal is dead.
        if signal.conviction <= 0.0:
            return None
            
        # --- NEW: MACRO PROBABILITY STACKING ---
        # "Stacking Probabilities" - User Request

        # FIX 2026-03-14: Macro Stack Weight Adaptation based on SMCE Regime
        # Different regimes should weight macro signals differently
        smce_regime = getattr(self, '_current_smce_regime', 'HARVEST')

        # Regime-adaptive stack weights
        # HARVEST/EXPANSION: Trust macro more (ordered markets)
        # TRANSITION: Reduce macro weight (uncertain regime)
        # DEFENSIVE: Macro irrelevant (no entries anyway)
        MACRO_STACK_WEIGHTS = {
            'HARVEST':    0.20,  # 20% influence (ordered, macro reliable)
            'EXPANSION':  0.25,  # 25% influence (trending, macro very reliable)
            'TRANSITION': 0.10,  # 10% influence (uncertain, reduce macro trust)
            'DEFENSIVE':  0.00,  # 0% influence (no entries in defensive)
        }

        macro = self.macro_oracle.fetch_macro_context()
        macro_bias = macro.get('bias_score', 0.0)
        base_stack_weight = getattr(config, 'MACRO_STACK_WEIGHT', 0.15)

        # Use regime-adaptive weight if available, otherwise fallback to config
        stack_weight = MACRO_STACK_WEIGHTS.get(smce_regime, base_stack_weight)

        # Logic:
        # Long + Positive Macro -> Boost
        # Short + Negative Macro -> Boost
        # Long + Negative Macro -> Drag
        # Short + Positive Macro -> Drag

        # FIX 2026-03-08: Protect whale convictions from macro headwinds
        is_whale = signal.metadata.get('is_whale', False)

        if signal.direction == 'BUY':
            stack_boost = macro_bias * stack_weight
        else:
            stack_boost = -macro_bias * stack_weight # Invert for shorts

        original_conv = signal.conviction

        # Whales get reduced macro penalty (50% reduction) and a floor
        if is_whale and stack_boost < 0:
            stack_boost *= 0.5  # Reduce penalty
            signal.conviction = max(0.25, min(1.0, signal.conviction + stack_boost))  # Floor at 0.25
        else:
            signal.conviction = max(0.0, min(1.0, signal.conviction + stack_boost))

        if abs(stack_boost) > 0.03:  # Lower threshold for logging with adaptive weights
            print(f"[{self.name}] 🌍 MACRO STACK ({smce_regime}): {symbol} Conviction {original_conv:.2f} -> {signal.conviction:.2f} (Bias {macro_bias:.2f}, Weight {stack_weight:.0%})")
            signal.metadata['macro_stack_score'] = stack_boost
            signal.metadata['macro_regime'] = smce_regime  # Track which regime was used
        # ---------------------------------------

        # --- PRE-PHYSICS: STRUCTURE DRIVES TARGETS ---
        # User Request: "structure drives targets"
        # enforced here to ensure all signals (even Arb) respect structure.
        if getattr(config, 'STRUCTURE_DRIVES_TARGETS', True):
             structure = signal.metadata.get('structure', {})
             pivots = structure.get('pivots', {})
             current_p = signal.price
             
             struct_tp = None
             
             if signal.direction == 'BUY':
                 # Find nearest R level > Entry * 1.005 (0.5% min separation)
                 levels = [v for k,v in pivots.items() if k.startswith('R') and v > current_p * 1.005]
                 levels.sort()
                 if levels: struct_tp = levels[0]
                     
             elif signal.direction == 'SELL':
                 # Find nearest S level < Entry * 0.995
                 levels = [v for k,v in pivots.items() if k.startswith('S') and v < current_p * 0.995]
                 levels.sort(reverse=True)
                 if levels: struct_tp = levels[0]
                     
             if struct_tp:
                 signal.take_profit_price = struct_tp
                 signal.metadata['take_profit_type'] = 'STRUCTURE_PIVOT'
        # ---------------------------------------------
        
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
            is_whale = signal.metadata.get('is_whale', False)
            
            # --- ☁️ FAIR WEATHER VETO (Recalibration Hotfix) ---
            # If Global Market Bias < 0.05 (Neutral/Bearish), only allow WHALES.
            gmb = self.get_market_bias()
            if gmb < 0.05 and not is_whale:
                print(f"[{self.name}] ☁️ FAIR WEATHER VETO: {symbol} Bias {gmb:.2f} < 0.05. Only Whales allowed.")
                return None
            # ----------------------------------------------------

            if is_whale:
                 # --- WHALE QUALITY FILTER (Recalibration) ---
                 if signal.conviction < 0.30:
                      print(f"[{self.name}] 🐋 WHALE VETO: Weak Signal ({signal.conviction:.2f} < 0.30).")
                      return None
                 # Whale completely bypasses structural pivot checks below
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
        # FIX 2026-03-05: Gold (XAUT/PAXG) is a safe-haven with structurally lower RVOL.
        # In a bearish crypto market, gold should be MORE attractive — not vetoed.
        _is_gold_asset = any(g in symbol for g in ('XAUT', 'PAXG'))
        if rvol < config.PHYSICS_MIN_RVOL and not _is_gold_asset:
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
        # FIX 2026-03-05: Gold (XAUT/PAXG) is a SAFE-HAVEN with INVERSE correlation to crypto.
        # In a bearish market (low GMB), gold should be MORE tradeable, not less.
        # Exempting gold from BTC-based pack veto logic.
        corr_matrix = self.market_state.get('correlation')
        if corr_matrix is not None and not corr_matrix.empty and 'BTC/USDT' in corr_matrix.columns and not _is_gold_asset:
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

        # D. ORDER FLOW PHYSICS (Micro-Structure)
        # Using CVD and Buy Ratio to detect Absorption or Exhaustion
        order_flow = signal.metadata.get('order_flow', {})
        if order_flow:
            of_signal = order_flow.get('signal', 'NEUTRAL')
            buy_ratio = order_flow.get('buy_ratio', 0.5)
            
            # 1. ABSORPTION BOOST (Dip Buying)
            # If we are BUYING and there is AGGRESSIVE BUYING (Absorption)
            if signal.direction == 'BUY' and of_signal == 'AGGRESSIVE_BUYING':
                signal.conviction = min(1.0, signal.conviction * 1.15) # +15% Boost
                signal.metadata['absorption_boost'] = True
                print(f"[{self.name}] 🌊 ORDER FLOW BOOST: {symbol} Absorption Detected (Buy Ratio {buy_ratio:.2%}). Conviction -> {signal.conviction:.2f}")

            # 2. EXHAUSTION VETO (Buying into weakness)
            # If we are BUYING but Sellers are Aggressive (Knife Catching without Absorption)
            elif signal.direction == 'BUY' and of_signal == 'AGGRESSIVE_SELLING':
                 # Only Veto if structure is weak too
                 if signal.conviction < 0.6:
                     print(f"[{self.name}] 🛑 ORDER FLOW VETO: {symbol} Long into Aggressive Selling (Ratio {buy_ratio:.2%}).")
                     return None
        
        # E. EXPERIENCE MEMORY (Hippocampus Veto)
        # "Have I seen this before?"
        memory = getattr(self, 'sub_holons', {}).get('memory') if hasattr(self, 'sub_holons') else None
        if memory:
             # Construct Context Vector: [RSI, BB_Width, GMB, Entropy, Volatility_Score]
             # We need to approximate these from signal metadata or fetch them
             try:
                 meta = signal.metadata
                 vec = [
                     float(meta.get('rsi', 50.0)) / 100.0,         # Norm 0-1
                     float(meta.get('bb_width', 0.05)) * config.RISK_MIN_BASE_NOTIONAL,     # Norm approx
                     self.get_market_bias(),                       # 0-1
                     self.market_state.get('entropy', 0.5),        # 0.5 approx
                     float(meta.get('rvol', 1.0)) / 5.0            # Norm approx
                 ]
                 
                 recall = memory.query_memory(vec)
                 score = recall['deja_vu_score']
                 
                 # 1. PTSD VETO (Strong Negative Memory)
                 if score < -0.6:
                     print(f"[{self.name}] ⛔ MEMORY BLOCK: {symbol} Déjà Vu Score {score:.2f} (Bad Outcome). {recall['details']}")
                     return None
                     
                 # 2. CONFIDENCE BOOST (Strong Positive Memory)
                 elif score > 0.6:
                     print(f"[{self.name}] ⚡ MEMORY BOOST: {symbol} Déjà Vu Score {score:.2f} (Good Outcome). {recall['details']}")
                     signal.conviction = min(1.0, signal.conviction * 1.25)
                     
             except Exception as e:
                 # print(f"Memory Query Error: {e}")
                 pass

        return signal

    def predict_position_health(self, symbol: str, current_price: float, entry_price: float, direction: str, metadata: dict = None) -> Dict[str, Any]:
        """
        Oracle Diagnosis: Check if a held position is growing or decaying.
        Uses: Price Action, Ensemble Vote (if avail), Global Bias, and MOMENTUM (metadata).
        """
        health = {
            'status': 'STABLE',
            'decay_score': 0.0, # 0.0 = Healthy, 1.0 = Critical
            'action': 'HOLD'
        }

        # FIX 2026-02-28: Null check MUST come before first use
        if metadata is None:
            metadata = {}

        # AEHML Fix: Strategy Exemption (Carry/Arb Protection)
        # Funding trades should not be subject to directional health decay.
        strat = metadata.get('strategy', '')
        reason = metadata.get('reason', '')
        is_carry = any(x in strat or x in reason for x in ['BASIS_CARRY', 'ARBITRAGE', 'FUNDING', 'CARRY'])

        if is_carry:
             return health # Bypass further logic

        # --- PATCH: MINIMUM HOLD IMMUNITY ---
        # If position is less than 30 minutes old, it's immune to early thesis exit
        pos_entry_time = metadata.get('first_entry_time', time.time())
        age_seconds = time.time() - pos_entry_time
        if age_seconds < 1800: # 30 minutes
             if self.DEBUG: print(f"[{self.name}] ⏳ HOLD IMMUNITY: {symbol} ({int(age_seconds)}s < 1800s)")
             return health
        rsi = metadata.get('rsi')
        rvol = metadata.get('rvol')
        bb_width = metadata.get('bb_width')
        
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

        # 4. Global Bias Collapse (Thesis Invalidation)
        # Hysteresis Added: Don't exit just because it dipped below entry threshold.
        # Wait for a clearer invalidation (e.g. < 0.20 if Entry was 0.30)
        invalidation_buffer = 0.20 # Significant drop required
        
        # --- PATCH 2.3: SOL BIAS OVERRIDE ---
        if symbol == 'SOL/USDT':
             invalidation_buffer = 0.30 # Effective Threshold 0.10
             # Reason: "SOL funding rate model outperforms directional bias"
        # ------------------------------------
        
        if direction == 'BUY':
             # If market turns significantly Bearish (below 0.20 usually)
             invalidation_level = max(0.10, config.GMB_THRESHOLD - invalidation_buffer) 
             
             # AEHML Fix: Skip check if we are in a carry/arb trade (directional bias doesn't invalidate yield)
             if is_carry:
                  self.thesis_failures[symbol] = 0
             elif gmb < invalidation_level:
                 # DIRECTIVE REC-2026-01-31: Decaying Counter
                 # DIRECTIVE REC-2026-01-31: Decaying Counter
                 count = self.thesis_failures.get(symbol, 0) + 1
                 self.thesis_failures[symbol] = count
                 
                 # NEW: Extended TTL for Whole Trades (Phase 1)
                 is_whale_pos = metadata.get('is_whale', False) if metadata else False
                 if is_whale_pos:
                     tolerance = getattr(config, 'WHALE_THESIS_TTL', 8)
                 else:
                     tolerance = getattr(config, 'STANDARD_THESIS_TTL', 2)
                 
                 if count >= tolerance:
                     health['status'] = 'THESIS_INVALID'
                     health['decay_score'] = 1.0
                     health['action'] = 'CLOSE'
                     print(f"[{self.name}] 📉 THESIS FAILED: {symbol} after {count} cycles (Limit {tolerance}) (Bias {gmb:.2f} < {invalidation_level:.2f})")
             else:
                 self.thesis_failures[symbol] = 0

        elif direction == 'SELL':
             # If market turns significantly Bullish (above 0.80 usually)
             # Short entry threshold is typically around 0.5 or 0.4
             # We exit if it rallies past GMB_THRESHOLD + Buffer
             invalidation_level = min(0.90, config.GMB_THRESHOLD + invalidation_buffer)
             
             # AEHML Fix: Skip check if we are in a carry/arb trade
             if is_carry:
                  self.thesis_failures[symbol] = 0
             elif gmb > invalidation_level:
                  count = self.thesis_failures.get(symbol, 0) + 1
                  self.thesis_failures[symbol] = count
                  tolerance = getattr(config, 'THESIS_FAILURE_TOLERANCE', 2)
                  
                  if count >= tolerance:
                      health['status'] = 'THESIS_INVALID'
                      health['decay_score'] = 1.0
                      health['action'] = 'CLOSE'
                      print(f"[{self.name}] 📈 THESIS FAILED: {symbol} Short after {count} cycles (Bias {gmb:.2f} > {invalidation_level:.2f})")
             else:
                  self.thesis_failures[symbol] = 0
        
        # 3. Decay & Growth Logic (Refined for "Peddling Faster")
        # A. Early Profit Decay
        if pnl > 0.01:
            if direction == 'BUY' and gmb < 0.4:
                health['status'] = 'DECAYING'
                health['decay_score'] = 0.7
                health['action'] = 'TIGHTEN_SL'
            elif direction == 'SELL' and gmb > 0.6:
                health['status'] = 'DECAYING'
                health['decay_score'] = 0.7
                health['action'] = 'TIGHTEN_SL'

        # B. Growth Potential (Refined)
        is_growth = False
        details = []
        
        # Factor 1: Market Bias Support
        if direction == 'BUY' and gmb > 0.55:
            is_growth = True
            details.append(f"GMB({gmb:.2f})")
        elif direction == 'SELL' and gmb < 0.45:
            is_growth = True
            details.append(f"GMB({gmb:.2f})")
            
        # Factor 2: MOMENTUM (RSI) - The "Peddling" Metric
        if rsi is not None:
            if direction == 'BUY' and rsi > 55 and rsi < 85: # Rising, not topped
                is_growth = True
                details.append(f"RSI({rsi:.1f})")
            elif direction == 'SELL' and rsi < 45 and rsi > 15: # Falling, not floored
                is_growth = True
                details.append(f"RSI({rsi:.1f})")
                
        # Factor 3: ENERGY (Volume) - The "Effort" Metric
        if rvol is not None:
            if rvol > 1.5: # 50% above average volume
                is_growth = True
                details.append(f"RVOL({rvol:.1f})")

        # Factor 4: ORDER FLOW (Micro-Structure)
        order_flow = metadata.get('order_flow', {})
        if order_flow:
            of_signal = order_flow.get('signal', 'NEUTRAL')
            buy_ratio = order_flow.get('buy_ratio', 0.5)
            
            if direction == 'BUY':
                if of_signal == 'AGGRESSIVE_BUYING':
                    is_growth = True
                    details.append(f"FLOW_ABSORB({buy_ratio:.0%})")
                elif of_signal == 'AGGRESSIVE_SELLING':
                    # Warning Sign
                    if pnl > 0.01: # Secure profits if sellers take over
                         health['action'] = 'TIGHTEN_TP' 
            
            elif direction == 'SELL':
                if of_signal == 'AGGRESSIVE_SELLING':
                    is_growth = True
                    details.append(f"FLOW_DUMP({1-buy_ratio:.0%})")
                    
        # Factor 5: YIELD FARMING (Ecological Mode)
        yield_apy = metadata.get('yield_apy', 0.0)
        # We FARM if Yield is High (>20%) and we are on the earning side
        is_farming = False
        if direction == 'BUY':
             # Longs earn if Rate is Negative (Yield > 0 in our logic? No, Yield is generally APY)
             # Arbitrage agent stores 'Net Yield'. 
             # If Yield > 20% and we are Long, it implies Funding was Negative.
             # Logic check: arbitrage agent calculates 'yield_apy' based on direction?
             # No, it stores 'funding_yields' as simple annualized rate.
             # Wait, `analyze_funding_yield` results: 
             # It stores the RAW APY. If Rate is Negative, APY is Negative.
             # But `get_arb_conviction` treats Negative Rate as "Yield for Long".
             # So if APY < -20.0 (Negative 20%), Longs are earning >20%.
             if yield_apy < -20.0:
                 is_farming = True
                 details.append(f"YIELD_FARM({abs(yield_apy):.0f}%)")
        elif direction == 'SELL':
             # Shorts earn if Rate is Positive.
             if yield_apy > 20.0:
                 is_farming = True
                 details.append(f"YIELD_FARM({yield_apy:.0f}%)")
        
        if is_farming:
             health['status'] = 'FARMING'
             health['action'] = 'RELAX_TP' # Let it grow
             # If PnL is also good, maybe PLANT_SEED (Pyramid)?
             if pnl > 0.02 and is_growth:
                 health['action'] = 'PLANT_SEED' # Signal to increase size? (Not fully impl yet)
                 
        if is_growth:
            # Boost confidence if PnL is already strong (>2%)
            if pnl > 0.02:
                health['status'] = 'GROWTH'
                health['action'] = 'RELAX_TP'
            
            # Or if Structure is great (Multiple confirmations)
            elif len(details) >= 2 and pnl > 0.005:
                health['status'] = 'GROWTH'
                health['action'] = 'RELAX_TP'
                # print(f"[{self.name}] 🌱 GROWTH MODE: {symbol} (+{pnl*100:.1f}%) Factors: {', '.join(details)}")
                
        return health

    def verify_holding_physics(self, symbol: str, direction: str, current_price: float = None, entry_price: float = None, metadata: dict = None) -> Dict[str, Any]:
        """
        Proof of Holding: Re-verify the thesis for an open position.
        Returns detailed health dict (replacing simple boolean).
        """
        result = {'valid': True, 'reason': '', 'recommendation': 'HOLD'}

        # FIX 2026-02-28: Null check MUST come before first use
        if metadata is None:
            metadata = {}

        # 0. STRATEGY EXEMPTION (AEHML Fix: Carry/Arb Protection)
        # Prevent funding trades from closing just because directional bias changed.
        strat = metadata.get('strategy', '')
        reason = metadata.get('reason', '')
        is_carry = any(x in strat or x in reason for x in ['BASIS_CARRY', 'ARBITRAGE', 'FUNDING', 'CARRY'])

        if is_carry:
             # Funding arb positions are held for yield, not directional bias.
             if self.DEBUG: print(f"[{self.name}] 🛡️ THESIS IMMUNITY: {symbol} (funding arb)")
             return result

        # --- PATCH: MINIMUM HOLD IMMUNITY ---
        # 30 minute minimum hold before thesis validation kicks in
        
        pos_entry_time = metadata.get('first_entry_time', time.time())
        age_seconds = time.time() - pos_entry_time
        if age_seconds < 1800:
             if self.DEBUG: print(f"[{self.name}] ⏳ HOLD IMMUNITY: {symbol} ({int(age_seconds)}s < 1800s)")
             return result

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
            health = self.predict_position_health(symbol, current_price, entry_price, direction, metadata=metadata)
            if health['action'] != 'HOLD':
                result['recommendation'] = health['action']
                result['reason'] = f"AI_{health['status']}"
                if health['status'] == 'DECAYING':
                     print(f"[{self.name}] 🤒 POS DECAY: {symbol} {direction} (GMB mismatch). Rec: {health['action']}")
                elif health['status'] == 'GROWTH':
                     print(f"[{self.name}] 🌱 POS GROWTH: {symbol} {direction} (Trend Strong). Rec: {health['action']}")
                 
        return result

    def generate_forecast(self, symbol: str, target_price: float, days: int = 30, prices: list = None) -> Dict[str, Any]:
        """
        Generates a Monte Carlo forecast for a specific target price.
        """
        if prices is None or len(prices) < 20: 
            return {'probability': 0.0, 'error': 'Insufficient Data'}
            
        current_price = prices[-1]
        
        # 1. Estimate Physics Parameters (GBM)
        # We use a daily dt for parameter estimation if prices are daily, 
        # but if prices are 15m (likely), we need to adjust.
        # However, SDEEngine expects raw series.
        # Let's assume input prices are 1h or Daily for meaningful long-term check? 
        # If 15m, 30 days = 2880 candles.
        
        # Determine dt based on input frequency implication? 
        # For simplicity, we assume prices are "Conceptually Daily" or we normalize.
        # Actually SDEEngine params (sigma) depend on dt.
        # Let's use standard GBM estimation where dt is implicit in the series interval.
        # If we want 'days' horizon, we need to know how many 'steps' that is.
        # If we pass 1h candles:
        params = SDEEngine.estimate_gbm_parameters(np.array(prices), dt=1/24) # Assuming Hourly
        
        # 2. Simulate
        steps = days * 24 # Hourly steps for N days
        
        prob = SDEEngine.calculate_hit_probability(
            'GBM', params, current_price, target_price, horizon=steps, paths=1000, dt=1/24
        )
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'target_price': target_price,
            'horizon_days': days,
            'drift': params.get('drift', 0.0),
            'volatility': params.get('diffusion', 0.0),
            'probability': prob,
            'model': 'Geometric Brownian Motion (Monte Carlo)'
        }
        
    def calculate_trade_expectancy(self, symbol: str, stop_loss: float, prices: list = None) -> Dict[str, Any]:
        """
        Estimates the median time before a trade hits its stop loss.
        """
        if prices is None or len(prices) < 20: 
            return {'error': 'Insufficient Data'}
            
        current_price = prices[-1]
        params = SDEEngine.estimate_gbm_parameters(np.array(prices), dt=1/24) # Hourly
        
        # Simulate up to 90 days (2160 hours)
        max_hours = 24 * 90 
        
        median_hours = SDEEngine.estimate_survival_horizon(
            'GBM', params, current_price, stop_loss, max_horizon=max_hours, paths=1000, dt=1/24
        )
        
        median_days = median_hours / 24.0
        
        return {
            'symbol': symbol,
            'stop_loss': stop_loss,
            'current_price': current_price,
            'expected_duration_days': median_days,
            'drift': params.get('drift', 0.0),
            'volatility': params.get('diffusion', 0.0)
        }
        
    def generate_term_structure(self, symbol: str, prices: list = None) -> Dict[str, Any]:
        """
        Generates a 3, 7, and 21-day Probabilistic Price Cone (Term Structure).
        Returns P5 (Bear Case), P50 (Base Case), P95 (Bull Case) for each horizon.
        """
        if prices is None or len(prices) < 20: 
            return {'error': 'Insufficient Data'}
            
        current_price = prices[-1]
        # 1. Estimate
        params = SDEEngine.estimate_gbm_parameters(np.array(prices), dt=1/24)
        
        # 2. Simulate Max Horizon (21 Days)
        max_days = 21
        steps = max_days * 24
        sim_paths = SDEEngine.simulate_paths('GBM', params, current_price, steps, paths=2000, dt=1/24)
        
        # 3. Slice & Dice
        horizons = [3, 7, 21]
        structure = {}
        
        for d in horizons:
            step_idx = (d * 24) - 1
            if step_idx >= steps: step_idx = steps - 1
            
            # Get all path values at this specific time step
            slice_prices = sim_paths[:, step_idx]
            
            structure[f'{d}d'] = {
                'p05': float(np.percentile(slice_prices, 5)),  # 95% worst case
                'p50': float(np.percentile(slice_prices, 50)), # Median
                'p95': float(np.percentile(slice_prices, 95)), # 95% best case
                'mean': float(np.mean(slice_prices))
            }
            
        return {
            'symbol': symbol,
            'current_price': current_price,
            'drift': params.get('drift', 0.0),
            'vol': params.get('diffusion', 0.0),
            'structure': structure
        }

    def extract_pips(self, prices: list, n_pips: int = 10) -> List[int]:
        """
        Identify Perceptually Important Points (PIPs) using Vector Distance.
        Returns indices of the PIPs sorted ascending.
        """
        if not prices or len(prices) < n_pips: return []
        
        # Need at least start and end
        pips = [0, len(prices) - 1]
        data = np.array(prices)
        
        while len(pips) < n_pips:
            max_dist = -1
            max_idx = -1
            
            pips.sort()
            
            # Check deviation for each segment
            for i in range(len(pips) - 1):
                start = pips[i]
                end = pips[i+1]
                
                if end - start < 2: continue
                
                # Line Points
                x1, y1 = start, data[start]
                x2, y2 = end, data[end]
                
                # Precompute constants for distance formula
                # Distance = |(y1-y2)x0 + (x2-x1)y0 + (x1y2 - x2y1)| / sqrt(...)
                A = y1 - y2
                B = x2 - x1
                C = x1*y2 - x2*y1
                denom = np.sqrt(A*A + B*B)
                if denom == 0: denom = 1.0 
                
                # Scan internal points
                segment_indices = np.arange(start + 1, end)
                segment_vals = data[segment_indices]
                
                # Vectorized Distance Check
                dists = np.abs(A * segment_indices + B * segment_vals + C) / denom
                
                curr_max_dist = np.max(dists)
                curr_max_loc = segment_indices[np.argmax(dists)]
                
                if curr_max_dist > max_dist:
                    max_dist = curr_max_dist
                    max_idx = curr_max_loc
            
            if max_idx != -1 and max_idx not in pips:
                pips.append(max_idx)
            else:
                break
                
        pips.sort()
        return pips

    def analyze_structure(self, extract: Dict[str, Any]) -> str:
        """
        Interprets Market Structure from PIPs (HH/HL Analysis).
        Input is result format from a helper or just filtered prices.
        Actually for now, input 'pips' is just indices, we need price/time context.
        Let's refactor: this method takes (prices, pip_indices).
        """
        return "UNKNOWN" # Placeholder, will update signature in next step for clarity or inline it.

    def get_structure_status(self, symbol: str, prices: list) -> Dict[str, Any]:
        """
        Full pipeline: Prices -> PIPs -> Structure Label.
        """
        if not prices or len(prices) < 20: 
            return {'status': 'INSUFFICIENT_DATA', 'pips': []}
            
        pips_idx = self.extract_pips(prices, n_pips=7) # ZigZag 7
        pip_prices = [prices[i] for i in pips_idx]
        
        # Analyze Peaks and Valleys
        # We need to distinguish Highs vs Lows.
        # Simple heuristic: If index i > i-1 and i > i+1 it's a High? No PIPs are far apart.
        # We assume alternating High/Low usually if n_pips is small and trend significant.
        # Better: Classify each PIP as Peak or Trough based on neighbors.
        
        structure = "NEUTRAL"
        
        # Last 4 Pivots (High/Low sequence) determine immediate structure
        if len(pip_prices) >= 4:
            # Check for Higher Highs / Higher Lows
            # This is complex to do robustly in one pass.
            # Simplified: Compare last 2 legs slope.
             
            last_price = pip_prices[-1]
            prev_pip = pip_prices[-2]
            
            delta = last_price - prev_pip
            if delta > 0:
                structure = "BULLISH_LEG"
            else:
                structure = "BEARISH_LEG"
                
        return {
            'symbol': symbol,
            'status': structure,
            'pips_count': len(pips_idx),
            'last_price': pip_prices[-1],
            'pips_indices': pips_idx,
            'pips_values': pip_prices
        }

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

    def analyze_volatility_compression(self, symbol: str, data: pd.DataFrame) -> Any:
        """
        User Strategy: Volatility Compression (Squeeze)
        Logic: ATR(20) < ATR(30) -> Compression
        Entry: Open + 2 * ATR(20) (Breakout)
        Stop: 0.5 * ATR(20) (Tight)
        
        FIX 2026-03-22: Added stricter filtering to reduce signal spam
        """
        from .agent_executor import TradeSignal

        if len(data) < 35: return None

        # 1. Calculate TR
        high = data['high']
        low = data['low']
        close = data['close']
        prev_close = close.shift(1)

        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

        # 2. Calculate ATRs
        atr20 = tr.rolling(20).mean().iloc[-1]
        atr30 = tr.rolling(30).mean().iloc[-1]

        if atr20 <= 0 or atr30 <= 0: return None

        # 3. Check Compression Condition
        if atr20 < atr30:
            # === FIX 2026-03-22: Additional Filtering to Reduce Spam ===
            # Require significant compression (not just any ATR cross)
            compression_ratio = atr20 / atr30
            if compression_ratio > 0.85:  # Must be at least 15% compression
                return None  # Not enough compression

            # Require volume confirmation (filter low-volume noise)
            current_vol = data['volume'].iloc[-1]
            avg_vol = data['volume'].rolling(20).mean().iloc[-1]
            if current_vol < avg_vol * 0.8:  # Volume must be at least 80% of average
                return None  # Low volume = likely false signal

            # === FIX 2026-03-23: ADX Directional Confirmation ===
            # Volatility squeeze alone doesn't tell direction - need ADX to confirm trend strength
            # Calculate ADX(14) to filter counter-trend breakouts
            plus_dm = high.diff()
            minus_dm = low.diff()

            plus_dm[plus_dm < 0] = 0  # +DM: only positive high changes
            minus_dm[minus_dm > 0] = 0  # -DM: only negative low changes

            tr_series = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
            atr14 = tr_series.rolling(14).mean()

            plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
            minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)

            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
            adx = dx.rolling(14).mean().iloc[-1]

            # ADX must be > 20 for trend confirmation (ADX < 20 = ranging/noise)
            if pd.isna(adx) or adx < 20:
                return None  # No strong trend - squeeze breakout likely false

            # Direction must align with DI crossover
            plus_di_val = plus_di.iloc[-1]
            minus_di_val = minus_di.iloc[-1]
            # === END ADX FILTER ===
            
            # === FIX 2026-03-15: Walk-Forward Validation for Volatility Squeeze ===
            wf_valid, recent_wr = self._validate_signal_walk_forward(symbol, 'VOLATILITY_SQUEEZE')
            if not wf_valid:
                self._safe_print(f"[{self.name}] 🚫 WALK-FORWARD VETO: {symbol} VOLATILITY_SQUEEZE (Recent WR: {recent_wr*100:.0f}%)")
                return None  # Block signal
            # === END WALK-FORWARD VALIDATION ===

            # SQUEEZE DETECTED
            current_open = data['open'].iloc[-1]
            current_close = data['close'].iloc[-1]

            # Entry Logic: "Entry = PrevClose + 1.0 * ATR(20)"
            trigger_buy = prev_close.iloc[-1] + (1.0 * atr20)
            trigger_sell = prev_close.iloc[-1] - (1.0 * atr20)

            direction = None
            if current_close > trigger_buy:
                direction = 'BUY'
            elif current_close < trigger_sell:
                direction = 'SELL'

            # === FIX 2026-03-23: ADX Directional Alignment ===
            # Direction must align with DI crossover for higher probability
            if direction:
                if direction == 'BUY' and plus_di_val < minus_di_val:
                    # BUY signal but -DI > +DI = counter-trend, skip
                    return None
                elif direction == 'SELL' and plus_di_val > minus_di_val:
                    # SELL signal but +DI > -DI = counter-trend, skip
                    return None
            # === END ADX ALIGNMENT ===

            if direction:
                # Dynamic Stop
                stop_dist = 0.5 * atr20
                if direction == 'BUY':
                    sl_price = current_close - stop_dist
                else:
                    sl_price = current_close + stop_dist

                # Get calibrated conviction
                calibrated_conviction = self._get_calibrated_conviction('VOLATILITY_SQUEEZE', recent_wr)

                # === FIX 2026-03-22: Raise minimum conviction threshold ===
                if calibrated_conviction < 0.50:
                    return None  # Filter low conviction signals
                # === END FIX ===

                self._safe_print(f"[{self.name}] ⚡ VOLATILITY SQUEEZE: {symbol} {direction} (ATR20 {atr20:.4f} < ATR30 {atr30:.4f}) | ADX:{adx:.1f} DI+:{plus_di_val:.1f} DI-:{minus_di_val:.1f} | Conv: {calibrated_conviction:.2f}")

                sig = TradeSignal(symbol=symbol, direction=direction, size=1.0, price=current_close, conviction=calibrated_conviction)
                sig.metadata = {
                    'strategy': 'VOLATILITY_COMPRESSION',
                    'stop_type': 'VOLATILITY', # Handled by Governor Patch
                    'sl_price': sl_price,      # Specific Stop Level
                    'atr_value': atr20,
                    'stop_width_atr': 0.5,
                    'walk_forward_validated': True,
                    'recent_win_rate': recent_wr
                }
                return self.apply_asset_personality(symbol, sig)

        return None

    def _safe_print(self, msg: str):
        """Thread-safe printing to avoid log corruption."""
        with self._lock:
            print(msg)

    def load_brains(self):
        """Load AI brains (LSTM and XGBoost)."""
        # LSTM Paths - Search multiple locations
        model_paths = [
            'lstm_model.keras',  # Current directory
            os.path.join(os.path.dirname(__file__), 'lstm_model.keras'),  # Same dir as this file
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lstm_model.keras'),  # Parent dir
            os.path.join(os.getcwd(), 'lstm_model.keras'),  # Working directory
        ]
        scaler_paths = [
            'scaler.pkl',
            os.path.join(os.path.dirname(__file__), 'scaler.pkl'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scaler.pkl'),
            os.path.join(os.getcwd(), 'scaler.pkl'),  # Working directory
        ]
        # XGBoost Path
        xgb_paths = [
            'xgboost_model.json',
            os.path.join(os.path.dirname(__file__), 'xgboost_model.json'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'xgboost_model.json'),
            os.path.join(os.getcwd(), 'xgboost_model.json'),  # Working directory
        ]

        # Find first existing path
        model_path = next((p for p in model_paths if os.path.exists(p)), None)
        scaler_path = next((p for p in scaler_paths if os.path.exists(p)), None)
        xgb_path = next((p for p in xgb_paths if os.path.exists(p)), None)

        # Debug: Print found paths
        self._safe_print(f"[{self.name}] LSTM path search: Found={model_path}")
        self._safe_print(f"[{self.name}] Scaler path search: Found={scaler_path}")
        self._safe_print(f"[{self.name}] XGB path search: Found={xgb_path}")
        self._safe_print(f"[{self.name}] TensorFlow available: {tf is not None}")
        self._safe_print(f"[{self.name}] Joblib available: {joblib is not None}")

        # 1. Load LSTM
        if model_path and scaler_path and tf is not None and joblib is not None:
            try:
                self._safe_print(f"[{self.name}] Loading LSTM from {model_path}...")
                self.model = tf.keras.models.load_model(model_path)
                self._safe_print(f"[{self.name}] Loading scaler from {scaler_path}...")
                self.scaler = joblib.load(scaler_path)
                self._safe_print(f"[{self.name}] ✅ LSTM Brain loaded successfully from {model_path}.")
            except Exception as e:
                self._safe_print(f"[{self.name}] ❌ Error loading LSTM: {type(e).__name__}: {e}")
                import traceback
                self._safe_print(traceback.format_exc())
        else:
            missing = []
            if not model_path: missing.append('lstm_model.keras')
            if not scaler_path: missing.append('scaler.pkl')
            if tf is None: missing.append('tensorflow')
            if joblib is None: missing.append('joblib')
            self._safe_print(f"[{self.name}] ℹ️  Optional Models absent (heuristic fallback active): {', '.join(missing)}")

        # 2. Load XGBoost
        if xgb_path and xgb is not None:
            try:
                self._safe_print(f"[{self.name}] Loading XGBoost from {xgb_path}...")
                self.xgb_model = xgb.Booster()
                self.xgb_model.load_model(xgb_path)
                self._safe_print(f"[{self.name}] ✅ XGBoost Brain loaded successfully.")
            except Exception as e:
                self._safe_print(f"[{self.name}] ❌ Error loading XGBoost: {type(e).__name__}: {e}")

        if self.model is None and self.xgb_model is None:
            self._safe_print(f"[{self.name}] ⚠️ All brains missing or deps failed. Running heuristic mode.")

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


    def _extract_ml_features(self, df: pd.DataFrame, symbol: str = None) -> Dict[str, float]:
        """
        Gathers standard features for XGBoost inference.
        Returns a dictionary of features for the last candle.
        Aligned with research/train_xgboost.py
        """
        try:
            closes = df['close']
            volumes = df['volume']
            current_price = closes.iloc[-1]
            
            # 1. RSI (14)
            rsi = 50.0
            delta = closes.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            if not loss.empty and loss.iloc[-1] != 0:
                rs = gain.iloc[-1] / loss.iloc[-1]
                rsi = 100 - (100 / (1 + rs))
                
            # 2. BB %B
            rolling_mean = closes.rolling(window=20).mean()
            rolling_std = closes.rolling(window=20).std()
            bb_high = rolling_mean + (2 * rolling_std)
            bb_low = rolling_mean - (2 * rolling_std)
            
            bb_pct_b = 0.5
            if not bb_high.empty and (bb_high.iloc[-1] - bb_low.iloc[-1]) != 0:
                bb_pct_b = (current_price - bb_low.iloc[-1]) / (bb_high.iloc[-1] - bb_low.iloc[-1])
                
            # 3. Volatility
            returns = closes.pct_change()
            vol = returns.rolling(14).std().iloc[-1]
            
            # 4. MACD Hist
            exp1 = closes.ewm(span=12, adjust=False).mean()
            exp2 = closes.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            macd_hist = (macd - macd_signal).iloc[-1]
            
            # 5. RVOL
            vol_sma = volumes.rolling(20).mean().iloc[-1]
            rvol = volumes.iloc[-1] / vol_sma if vol_sma > 0 else 1.0
            
            # 6. LSTM Prob
            lstm_prob = self.predict_trend_lstm(closes, symbol)
            
            return {
                'rsi': float(rsi),
                'bb_pct_b': float(bb_pct_b),
                'volatility': float(vol),
                'macd_hist': float(macd_hist),
                'rvol': float(rvol),
                'lstm_prob': float(lstm_prob)
            }
        except Exception as e:
            # self._safe_print(f"[{self.name}] Feature Extraction Error: {e}")
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
        if entropy_context and entropy_context > config.RISK_MAX_ENTROPY_VETO: 
             # Extreme Chaos -> Random Walk -> Neutral
             # self._safe_print(f"[{self.name}] EntroPE: Skipping LSTM for {symbol} (Entropy {entropy_context:.2f} > {config.RISK_MAX_ENTROPY_VETO})")
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
            
            # EntroPE Modulation (Post-Process) — Shannon nats scale: ORDERED<1.20, CHAOTIC>1.85
            if entropy_context:
                if entropy_context < 1.20: # Highly Ordered
                    # Structure is reliable — boost LSTM conviction slightly
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
        # If Entropy is High (Chaotic > 1.85), XGBoost (Tree Logic) often fails or overfits specific noise.
        # Shannon nats scale: ORDERED<1.20, TRANSITION 1.20-1.85, CHAOTIC>1.85
        if entropy_context and entropy_context > 1.85:
            # Near-uniform distribution — skip expensive compute, return neutral
            pass

        try:
            # Prepare DMatrix
            df_feat = pd.DataFrame([features])
            dmatrix = xgb.DMatrix(df_feat)
            prob = self.xgb_model.predict(dmatrix)[0]

            # EntroPE Modulation — Shannon nats thresholds
            if entropy_context:
                if entropy_context > 1.85: # Chaotic (near-uniform returns)
                     # Dampen: Move prob towards 0.5
                     prob = 0.5 + (prob - 0.5) * 0.5
                elif entropy_context < 1.20: # Ordered (structured/trending)
                     # Boost: Structure is reliable, amplify confidence
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
        """
        Calculate Global Market Bias (GMB) from multiple sources.

        FIX 2026-03-23: Use Structure Boss macro trends instead of Kalman noise.
        The previous implementation (price > Kalman) was too noisy and created
        false bias signals that contradicted actual market structure.
        """
        # Priority 1: Use Structure Boss macro trends if available
        structure_bias = self._get_structure_macro_bias()
        if structure_bias is not None:
            # Blend structure bias with sentiment
            sentiment_bias_norm = (sentiment_score + 1.0) / 2.0
            if sentiment_score != 0.0:
                final_bias = (structure_bias * (1 - config.SENTIMENT_WEIGHT)) + (sentiment_bias_norm * config.SENTIMENT_WEIGHT)
            else:
                final_bias = structure_bias

            self.bias_history.append(final_bias)
            if len(self.bias_history) > 10:
                self.bias_history = self.bias_history[-10:]
            return sum(self.bias_history) / len(self.bias_history)

        # Fallback: Use symbol_trends (Kalman-based) if Structure Boss unavailable
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

            # --- DIRECTIVE REC-2026-01-31: ADAPTIVE MOMENTUM BIAS ---
            # 1. Participation Boost
            base_bias = max(0.05, final_bias + 0.15)

            # 2. Velocity Component
            velocity = 0.0
            if len(self.bias_history) >= 2:
                 # Simple Delta or EMA of Delta
                 delta = self.bias_history[-1] - self.bias_history[0] # Total change over history
                 velocity = delta * 0.30

            # 3. Cap and Floor
            adaptive_bias = min(0.85, base_bias + velocity)

            self.bias_history.append(adaptive_bias)
            smoothed_bias = sum(self.bias_history) / len(self.bias_history)
            return smoothed_bias

        # Even purely technical bias gets smoothed
        self.bias_history.append(technical_bias)
        smoothed_bias = sum(self.bias_history) / len(self.bias_history)
        return smoothed_bias

    def _get_structure_macro_bias(self) -> Optional[float]:
        """
        FIX 2026-03-23: Get macro trend bias from Structure Boss.
        Returns: 1.0 if mostly BULLISH, 0.0 if mostly BEARISH, 0.5 if mixed/NEUTRAL
        """
        try:
            # Check if we have structure data in recent signals
            if not hasattr(self, 'recent_structure_biases'):
                self.recent_structure_biases = {}

            if not self.recent_structure_biases:
                return None

            # Count macro trends
            bullish = sum(1 for trend in self.recent_structure_biases.values() if trend == 'BULLISH')
            bearish = sum(1 for trend in self.recent_structure_biases.values() if trend == 'BEARISH')
            neutral = sum(1 for trend in self.recent_structure_biases.values() if trend == 'NEUTRAL')
            total = len(self.recent_structure_biases)

            if total < 2:
                return None

            # Calculate bias: BULLISH=1.0, NEUTRAL=0.5, BEARISH=0.0
            bias_score = (bullish * 1.0 + neutral * 0.5 + bearish * 0.0) / total
            return bias_score
        except Exception:
            return None

    def update_structure_bias(self, symbol: str, macro_trend: str):
        """
        FIX 2026-03-23: Update structure bias for GMB calculation.
        Called by Structure Boss when macro trend is determined.
        """
        if not hasattr(self, 'recent_structure_biases'):
            self.recent_structure_biases = {}

        self.recent_structure_biases[symbol] = macro_trend

        # Keep only recent (last 11 symbols)
        if len(self.recent_structure_biases) > 11:
            oldest_key = next(iter(self.recent_structure_biases))
            del self.recent_structure_biases[oldest_key]

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

    def detect_ict_judas_swing(self, symbol: str, observer: Any, current_time: datetime = None) -> Tuple[bool, str, float]:
        """
        ICT AMD (Accumulation, Manipulation, Distribution) Strategy.
        Identifies Asian Range (00:00 - 08:00 UTC).
        Detects Judas Swing liquidation (Sweep of High/Low).
        Returns: Tuple(is_detected, direction ('BUY'/'SELL'), sl_price)
        """
        if observer is None:
            return False, 'NEUTRAL', 0.0
            
        if current_time is None:
            current_time = datetime.now(timezone.utc)
            
        # Parse Session Windows
        def in_window(time_str: str, dt: datetime):
            start_str, end_str = time_str.split('-')
            sh, sm = map(int, start_str.split(':'))
            eh, em = map(int, end_str.split(':'))
            start = dt.replace(hour=sh, minute=sm, second=0, microsecond=0)
            end = dt.replace(hour=eh, minute=em, second=0, microsecond=0)
            return start <= dt <= end

        # 1. Check if we are active in a Manipulation Window (London or NY Open)
        london_active = in_window(getattr(config, 'ICT_LONDON_OPEN', '08:00-10:00'), current_time)
        ny_active = in_window(getattr(config, 'ICT_NY_OPEN', '13:30-15:30'), current_time)
        
        if not (london_active or ny_active):
            return False, 'NEUTRAL', 0.0
            
        # 2. Extract Asian Range (00:00 - 08:00 UTC today)
        asian_window = getattr(config, 'ICT_ASIAN_SESSION', '00:00-08:00')
        ash, asm = map(int, asian_window.split('-')[0].split(':'))
        aeh, aem = map(int, asian_window.split('-')[1].split(':'))
        
        asian_start = current_time.replace(hour=ash, minute=asm, second=0, microsecond=0)
        asian_end = current_time.replace(hour=aeh, minute=aem, second=0, microsecond=0)
        
        # Get historical data (15m is best for this)
        try:
            df_15m = observer.fetch_market_data(limit=100, timeframe='15m', symbol=symbol)
        except Exception as e:
            self._safe_print(f"[{self.name}] 🪟 ICT JUDAS ERROR (15m fetch): {e}")
            return False, 'NEUTRAL', 0.0
            
        if df_15m is None or df_15m.empty:
            return False, 'NEUTRAL', 0.0
            
        df_15m['datetime'] = pd.to_datetime(df_15m['timestamp'], unit='ms', utc=True)
        
        # Filter for the Asian session candles
        asian_filter = df_15m[(df_15m['datetime'] >= asian_start) & (df_15m['datetime'] < asian_end)]
        if asian_filter.empty:
            return False, 'NEUTRAL', 0.0
            
        asian_high = asian_filter['high'].max()
        asian_low = asian_filter['low'].min()
        
        if asian_high == asian_low: return False, 'NEUTRAL', 0.0 # Error or stale data
        range_size = (asian_high - asian_low) / asian_low
        
        # If Asian range is too massive (e.g. > 5%), it's not accumulation, it's already a trend.
        if range_size > 0.05: return False, 'NEUTRAL', 0.0 
        
        # 3. Detect the Judas Sweep and Reversal
        current_close = df_15m['close'].iloc[-1]
        tolerance = getattr(config, 'ICT_SWEEP_TOLERANCE_PCT', 0.005)
        
        # Get recent candles during the active manipulation window
        session_filter = df_15m[df_15m['datetime'] >= asian_end]
        if session_filter.empty: return False, 'NEUTRAL', 0.0
        
        session_high = session_filter['high'].max()
        session_low = session_filter['low'].min()
        
        # BULLISH REVERSAL (Trapped Shorts)
        # Price swept below Asian Low, but has now reversed back above it
        sweep_low_target = asian_low * (1 - tolerance)
        if session_low < sweep_low_target and current_close > asian_low:
             self._safe_print(f"[{self.name}] 🏛️ ICT JUDAS SWING DETECTED: {symbol} Bullish Reversal at {'London' if london_active else 'NY'} Open")
             return True, 'BUY', session_low
             
        # BEARISH REVERSAL (Trapped Longs)
        # Price swept above Asian High, but has now reversed back below it
        sweep_high_target = asian_high * (1 + tolerance)
        if session_high > sweep_high_target and current_close < asian_high:
             self._safe_print(f"[{self.name}] 🏛️ ICT JUDAS SWING DETECTED: {symbol} Bearish Reversal at {'London' if london_active else 'NY'} Open")
             return True, 'SELL', session_high
             
        return False, 'NEUTRAL', 0.0

    def detect_market_open_fvg(self, symbol: str, observer: Any, current_time: datetime = None, funding_rate: float = 0.0) -> Tuple[bool, str, float]:
        """
        Market Open FVG Strategy: Detects if there's a Fair Value Gap and strong volume dominance
        during the highest volume 5m candle within the market open window.
        Returns: Tuple(is_detected, direction ('BUY'/'SELL'), sl_price)
        """
        if not getattr(config, 'MARKET_OPEN_FVG_ENABLED', False) or observer is None:
            return False, 'NEUTRAL', 0.0
            
        if current_time is None:
            current_time = datetime.now(timezone.utc)
            
        windows = getattr(config, 'MARKET_OPEN_FVG_WINDOWS', [(0, 0), (14, 30)])
        
        # 1. Check if we are within 60 minutes of any market open window
        is_in_window = False
        start_ms = 0
        end_ms = 0
        
        for w_hour, w_min in windows:
            # Create a localized time for the window today
            window_time = current_time.replace(hour=w_hour, minute=w_min, second=0, microsecond=0)
            
            # If current time is past midnight but the window was yesterday (e.g. 14:30), check yesterday too
            window_time_yesterday = window_time - timedelta(days=1)
            
            for wt in [window_time, window_time_yesterday]:
                diff_minutes = (current_time - wt).total_seconds() / 60.0
                if 0 <= diff_minutes <= 60:
                    is_in_window = True
                    start_ms = int(wt.timestamp() * 1000)
                    end_ms = start_ms + 3600 * 1000 # 1 hour window
                    break
            
            if is_in_window: break
            
        if not is_in_window:
            return False, 'NEUTRAL', 0.0
            
        # 2. Fetch 5m data to find highest volume candle
        try:
            df_5m = observer.fetch_market_data(limit=18, timeframe='5m', symbol=symbol)
            if df_5m is None or df_5m.empty: return False, 'NEUTRAL', 0.0
            
            # Filter within the window
            start_dt = pd.to_datetime(start_ms, unit='ms')
            end_dt = pd.to_datetime(end_ms, unit='ms')
            df_5m_window = df_5m[(df_5m['timestamp'] >= start_dt) & (df_5m['timestamp'] < end_dt)]
            if df_5m_window.empty: return False, 'NEUTRAL', 0.0
            
            # Find max volume candle
            max_vol_idx = df_5m_window['volume'].idxmax()
            best_5m = df_5m_window.loc[max_vol_idx]
            best_5m_ts = best_5m['timestamp']
            
        except Exception as e:
            self._safe_print(f"[{self.name}] 🪟 MARKET OPEN ERROR (5m fetch): {e}")
            return False, 'NEUTRAL', 0.0
            
        # 3. Fetch 1m data for the specific 5m candle
        try:
            df_1m = observer.fetch_market_data(limit=90, timeframe='1m', symbol=symbol)
            if df_1m is None or df_1m.empty: return False, 'NEUTRAL', 0.0
            
            # Filter exactly the 5 minutes belonging to that 5m candle
            # best_5m_ts is already a pandas Timestamp
            best_5m_end = best_5m_ts + pd.Timedelta(minutes=5)
            df_1m_target = df_1m[(df_1m['timestamp'] >= best_5m_ts) & (df_1m['timestamp'] < best_5m_end)]
            if len(df_1m_target) < 3: return False, 'NEUTRAL', 0.0 # Need at least 3 candles for FVG
            
            # 4. Detect FVG
            fvg_found = False
            fvg_dir = 'NEUTRAL'
            fvg_sl = 0.0
            
            for i in range(len(df_1m_target) - 2):
                c1 = df_1m_target.iloc[i]
                c3 = df_1m_target.iloc[i+2]
                
                if c1['high'] < c3['low']:
                    fvg_dir = 'BUY'
                    fvg_sl = float(c1['high']) # Stop loss right below the gap's bottom edge (c3 low to c1 high gap -> invalidation if closes below gap)
                    fvg_found = True
                    break
                elif c1['low'] > c3['high']:
                    fvg_dir = 'SELL'
                    fvg_sl = float(c1['low'])
                    fvg_found = True
                    break
                    
            if not fvg_found: return False, 'NEUTRAL', 0.0
            
            # 5. Volume Control Checks
            buying_vol = 0.0
            selling_vol = 0.0
            
            for _, c in df_1m_target.iterrows():
                if c['close'] > c['open']: buying_vol += c['volume']
                elif c['close'] < c['open']: selling_vol += c['volume']
                else: 
                    buying_vol += c['volume']/2
                    selling_vol += c['volume']/2
                    
            total_vol = buying_vol + selling_vol
            if total_vol == 0: return False, 'NEUTRAL', 0.0
            
            threshold = getattr(config, 'MARKET_OPEN_FVG_DOMINANCE_THRESHOLD', 0.55)
            
            buy_dominance = buying_vol / total_vol
            sell_dominance = selling_vol / total_vol
            
            control = 'NEUTRAL'
            if buy_dominance > threshold: control = 'BUY'
            elif sell_dominance > threshold: control = 'SELL'
            
            # 6. Final Alignment Check
            if fvg_dir == control and control != 'NEUTRAL':
                # --- FUNDING YIELD TOXICITY CHECK ---
                # If we are going Long into a massive negative yield (e.g. paying > 150% APR to hold)
                # Or Short into a massive positive yield
                toxic_threshold = getattr(config, 'TOXIC_FUNDING_RATE_THRESHOLD', 0.001) # ~109% APY
                
                is_toxic = False
                if control == 'BUY' and funding_rate > toxic_threshold:
                     is_toxic = True
                elif control == 'SELL' and funding_rate < -toxic_threshold:
                     is_toxic = True
                     
                if is_toxic:
                     self._safe_print(f"[{self.name}] ☣️ TOXIC FVG VETO: {symbol} {control} invalidated by extremely toxic funding rate ({funding_rate*100*3*365:.0f}% APY)")
                     return False, 'NEUTRAL', 0.0
                
                self._safe_print(f"[{self.name}] 🪟 MARKET OPEN FVG DETECTED: {symbol} Direction: {control} (Dominance: {max(buy_dominance, sell_dominance)*100:.1f}%)")
                return True, control, fvg_sl
                
            return False, 'NEUTRAL', 0.0
            
        except Exception as e:
            self._safe_print(f"[{self.name}] 🪟 MARKET OPEN ERROR (1m FVG): {e}")
            return False, 'NEUTRAL', 0.0

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
        
        # Condition A: Price Lower Low (The Bear Trend) — require minimum 0.5% divergence
        price_diff_pct = (prev_low['low'] - last_low['low']) / prev_low['low'] if prev_low['low'] > 0 else 0
        price_lower_low = last_low['low'] < prev_low['low'] and price_diff_pct > 0.005
        
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



    def analyze_active_position(
        self, 
        symbol: str, 
        position_data: Dict[str, Any], 
        account_health: Dict[str, Any], 
        window_data: pd.DataFrame,
        structure_ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        AI Position Management & Risk Guard.
        Decides: HOLD, REDUCE, CLOSE, or STACK (Add).
        """
        action = {'type': 'HOLD', 'reason': 'Conviction Stable', 'urgency': 0.0}
        
        try:
            qty = float(position_data.get('qty', 0.0))
            if qty == 0: return action
            
            side = 'LONG' if qty > 0 else 'SHORT'
            current_price = window_data['close'].iloc[-1]
            entry_price = float(position_data.get('entry_price', current_price)) # Default if missing
            
            # --- 1. LIQUIDATION GUARD (CRITICAL) ---
            # If liquidation distance < 5%, we are in danger.
            liq_dist = float(account_health.get('liquidation_distance', 1.0))
            margin_level = float(account_health.get('margin_level', 10.0))
            
            if liq_dist < 0.05: # 5% Buffer
                return {
                    'type': 'URGENT_CLOSE',
                    'reason': f"CRITICAL: Liquidation Risk! Dist {liq_dist*100:.1f}%",
                    'urgency': 1.0
                }
                
            if margin_level < config.RISK_MIN_MARGIN_LEVEL: # Margin Call Proximity
                return {
                    'type': 'REDUCE',
                    'reason': f"Risk: Low Margin Level ({margin_level:.2f})",
                    'urgency': 0.8
                }

            # --- 2. MONTE CARLO STRESS TEST ---
            # "Can we survive the next 24 hours of volatility?"
            # Estimate GBM parameters
            sde_params = SDEEngine.estimate_gbm_parameters(window_data['close'].values)
            
            # Simulate 1000 paths for 24h (approx 1440 mins)
            horizon = 1440 
            paths = SDEEngine.simulate_paths('GBM', sde_params, current_price, horizon, 1000)
            
            # Bear Case: 5th Percentile outcome
            p05_path = np.percentile(paths, 5, axis=0) # Worst 5% case
            min_projected = np.min(p05_path)
            
            # Check if worst case hits invalidation
            # For structure, invalidation is below S2 (Long) or above R2 (Short)
            # Or if it hits Liquidation Price (approx)
            # Simplify: If Worst Case Drop > 10% and we are leveraged > 5x
            drop_pct = (current_price - min_projected) / current_price
            leverage = self.calculate_leverage(qty, current_price, float(account_health.get('equity', 1.0)))
            
            if drop_pct * leverage > 0.8: # Logic: If drop wipes out > 80% of equity allocated
                 return {
                    'type': 'REDUCE',
                    'reason': f"Monte Carlo: 95% Risk of -{drop_pct*100:.1f}% drop wipes equity.",
                    'urgency': 0.7
                }

            # --- 3. STRUCTURAL VALIDATION ---
            # Are we holding a Long below Support?
            if structure_ctx:
                pivots = structure_ctx.get('pivots', {})
                if side == 'LONG':
                    # Invalidation: Close below S2
                    s2 = pivots.get('S2')
                    if s2 and current_price < s2:
                         return {
                            'type': 'CLOSE',
                            'reason': "Technical Invalidation: Price broken S2 Support",
                            'urgency': 0.6
                        }
                elif side == 'SHORT':
                    r2 = pivots.get('R2')
                    if r2 and current_price > r2:
                        return {
                            'type': 'CLOSE',
                            'reason': "Technical Invalidation: Price broken R2 Resistance",
                            'urgency': 0.6
                        }

            # --- 4. OPPORTUNITY: STACKING ---
            # If Median Path > Target AND Margin Available
            available_margin = float(account_health.get('available', 0.0))
            if available_margin > 100.0: # Minimum $100 to stack
                # Bull Case: Median outcome
                p50_end = np.median(paths[:, -1])
                potential_pnl = (p50_end - current_price) / current_price if side == 'LONG' else (current_price - p50_end) / current_price
                
                # If projected PnL > 3% in 24h and Conviction High
                if potential_pnl > 0.03:
                    xgb_prob = self.predict_trend_xgboost(self._extract_ml_features(window_data, symbol))
                    
                    if (side == 'LONG' and xgb_prob > 0.75) or (side == 'SHORT' and xgb_prob < 0.25):
                        limit_safe = self._verify_usable_balance(qty, current_price, available_margin)
                        if limit_safe:
                            return {
                                'type': 'STACK',
                                'reason': f"High Confidence: MC Projects +{potential_pnl*100:.1f}% & AI confirms.",
                                'urgency': 0.4
                            }

        except Exception as e:
            self._safe_print(f"[{self.name}] Analysis Error: {e}")
            
        return action

    def calculate_leverage(self, qty: float, price: float, equity: float) -> float:
        if equity <= 0: return 100.0
        notional = abs(qty * price)
        return notional / equity

    def _verify_usable_balance(self, qty: float, price: float, available: float) -> bool:
        # Check if adding 25% to position consumes < 50% of available
        add_cost = (abs(qty) * 0.25 * price) / config.RISK_MIN_BASE_NOTIONAL # Approx 10x leverage cost
        return add_cost < (available * 0.5)

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
        observer: Any = None,
        is_whale: bool = False
    ):
        self.current_metabolism = metabolism_state
        from .agent_executor import TradeSignal # Ensure import
        # is_whale = False # Default initialization -> REMOVED
        prices = window_data['close']

        # 2026-03-20 Audit: Asset Blacklist — block negative-expectancy assets early
        asset_blacklist = getattr(config, 'ASSET_BLACKLIST', set())
        if symbol in asset_blacklist:
            return None

        # Normalize entropy regime once and reuse everywhere in this function.
        raw_entropy_regime = (structure_ctx.get('entropy_regime') if structure_ctx else None) or 'TRANSITION'
        entropy_regime = str(raw_entropy_regime).upper()
        regime_aliases = getattr(config, 'ENTRY_REGIME_ALIASES', {
            'BULLISH': 'ORDERED',
            'NEUTRAL': 'TRANSITION',
            'BEARISH': 'CHAOTIC'
        })
        normalized_entropy_regime = regime_aliases.get(entropy_regime, entropy_regime)

        # 2026-03-21 FIX: Redundant entropy ceiling removed.
        # The single authoritative entropy ceiling is in trader_entry_handler.py (ENTROPY GATE).
        # Keeping it here caused double-blocking and confusing log messages.
        
        # Default Initializations (Fix for UnboundLocalError)
        rsi = 50.0
        rvol = 1.0
        # is_whale = False -> REMOVED
        whale_reason = []

        # 🔑 KEY 0: SCAVENGER TRAP (Pattern Override)
        # Does this asset look like it just trapped bears at support?
        # Note: window_data usually passed from Trader is the active timeframe (e.g. 15m)
        is_trap, trap_level = self.detect_scavenger_trap(symbol, window_data, structure_ctx)
        if is_trap:
            # P2 FIX 2026-03-05: Check cooldown before generating signal
            if not self._check_signal_cooldown(symbol, 'SCAVENGER_TRAP'):
                return None
            self._safe_print(f"[{self.name}] 🪤 SCAVENGER TRAP: {symbol} Reclaimed {trap_level}. Triggering Long.")
            price = window_data['close'].iloc[-1]
            sig = TradeSignal(symbol=symbol, direction='BUY', size=1.0, price=price)
            sig.conviction = 0.85 # High Conviction for Structural Reclaims
            sig.metadata = {
                'strategy': 'SCAVENGER_TRAP',
                'trap_level': trap_level,
                'structure': structure_ctx
            }
            self._set_signal_cooldown(symbol, 'SCAVENGER_TRAP')
            return self.apply_asset_personality(symbol, sig)

        # 🔑 KEY 0.5: PACK HUNT (Laggard Alpha)
        if self.detect_pack_laggard(symbol, ticker_data, pack_stats):
            # P2 FIX 2026-03-05: Check cooldown before generating signal
            if not self._check_signal_cooldown(symbol, 'PACK_HUNT'):
                return None
            self._safe_print(f"[{self.name}] 🐺 PACK HUNT: {symbol} Catch-up Play Triggered.")
            sig = TradeSignal(symbol=symbol, direction='BUY', size=1.0, price=window_data['close'].iloc[-1])
            sig.conviction = 0.75 # Good conviction, but relies on market beta
            sig.metadata = {
                'strategy': 'PACK_HUNT',
                'structure': structure_ctx
            }
            self._set_signal_cooldown(symbol, 'PACK_HUNT')
            return self.apply_asset_personality(symbol, sig)

        # 🔑 KEY 0.6: ICT JUDAS SWING (London / NY Reversal)
        is_judas, judas_dir, judas_sl = self.detect_ict_judas_swing(symbol, observer)
        if is_judas:
            if not self._check_signal_cooldown(symbol, 'ICT_JUDAS_SWING'):
                pass
            else:
                self._safe_print(f"[{self.name}] 🏛️ ICT JUDAS SWING: {symbol} {judas_dir} trigger.")
                sig = TradeSignal(symbol=symbol, direction=judas_dir, size=1.0, price=window_data['close'].iloc[-1])
                sig.conviction = 0.85 # High conviction
                sig.metadata = {
                    'strategy': 'ICT_JUDAS_SWING',
                    'structure': structure_ctx,
                    'sl_price': judas_sl,
                    'is_whale': True # Tag as whale to survive strict RSI filters
                }
                self._set_signal_cooldown(symbol, 'ICT_JUDAS_SWING')
                return self.apply_asset_personality(symbol, sig)

        # 🔑 KEY 0.7: MARKET OPEN FVG Strategy
        # --- MACRO ALIGNMENT PATCH ---
        # Ensure we know the macro trend before firing FVG signals (Fixes Signal Contradiction P0)
        macro_trend = structure_ctx.get('macro_trend', 'NEUTRAL') if structure_ctx else 'NEUTRAL'
        
        is_fvg, fvg_dir, fvg_sl = self.detect_market_open_fvg(symbol, observer, funding_rate=funding_rate)
        if is_fvg:
            # P0 FIX: Prevent FVG from fighting the 1H Macro Flow
            is_congruent = True
            if macro_trend == 'BULLISH' and fvg_dir == 'SELL':
                is_congruent = False
                # self._safe_print(f"[{self.name}] ⛔ VETO: FVG SELL contradicts Macro BULLISH flow.")
            elif macro_trend == 'BEARISH' and fvg_dir == 'BUY':
                is_congruent = False
                # self._safe_print(f"[{self.name}] ⛔ VETO: FVG BUY contradicts Macro BEARISH flow.")

            if is_congruent:
                if not self._check_signal_cooldown(symbol, 'MARKET_OPEN_FVG'):
                    pass # Return None is too aggressive, just skip FVG signal if cooled down
                else:
                    self._safe_print(f"[{self.name}] 🪟 MARKET OPEN FVG: {symbol} {fvg_dir} trigger.")
                    sig = TradeSignal(symbol=symbol, direction=fvg_dir, size=1.0, price=window_data['close'].iloc[-1])
                    sig.conviction = 0.85 # High conviction
                    sig.metadata = {
                        'strategy': 'MARKET_OPEN_FVG',
                        'structure': structure_ctx,
                        'sl_price': fvg_sl
                    }
                    self._set_signal_cooldown(symbol, 'MARKET_OPEN_FVG')
                    return self.apply_asset_personality(symbol, sig)

        # 🔑 KEY 0.8: WHALE SHADOW (CVD Divergence)
        if self.detect_whale_shadow(symbol, window_data):
            # P2 FIX 2026-03-05: Check cooldown before generating signal
            if not self._check_signal_cooldown(symbol, 'WHALE_SHADOW'):
                return None
            self._safe_print(f"[{self.name}] 🐋 WHALE SHADOW: {symbol} Absorption Detected. Triggering Long.")
            sig = TradeSignal(symbol=symbol, direction='BUY', size=1.0, price=window_data['close'].iloc[-1])
            sig.conviction = 0.80 # High Conviction for Absorption
            sig.metadata = {
                'strategy': 'WHALE_SHADOW',
                'structure': structure_ctx,
                'is_whale': True
            }
            self._set_signal_cooldown(symbol, 'WHALE_SHADOW')
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
        # (Moved up to line 2668 for FVG macro alignment)
        
        # Volatility & Momentum (Moved UP for Whale Logic)
        returns = prices.pct_change()
        volatility = returns.rolling(14).std().iloc[-1]
        
        # 1. Feature Engineering (Moved UP for Filter Logic)
        # RSI (14)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        # Avoid division by zero
        if loss.iloc[-1] == 0:
            rsi = 100.0
        else:
            rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1]
        
        # Whale Volume
        vol_avg = window_data['volume'].rolling(window=20).mean().iloc[-1]
        vol_curr = window_data['volume'].iloc[-1]
        rvol = vol_curr / vol_avg if vol_avg > 0 else 1.0

        # --- SECTOR PHYSICS OVERRIDE (Holistic 5b) ---
        # Allow energetic decoupling for memecoins
        sector_override = False
        if symbol in config.MEMECOIN_ASSETS:
             if rvol > config.MEMECOIN_PUMP_RVOL:
                 self._safe_print(f"[{self.name}] 🚀 SECTOR PHYSICS: {symbol} Decoupling from Macro (RVOL {rvol:.1f})")
                 sector_override = True
                 is_whale = True
                 whale_reason.append(f"SECTOR_FORCE_RVOL_{rvol:.1f}")
        
        if not sector_override:
            last_state = self.last_macro_state.get(symbol, 'UNKNOWN')
            
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
            
            # 2026-03-20 Audit: Require multi-factor whale confirmation
            # Single BID_WALL signal fired 5,334 times as sole trigger — too noisy
            min_factors = getattr(config, 'WHALE_MIN_FACTORS', 1)
            is_whale = len(whale_reason) >= min_factors


            # Default Permissions (Allow All unless restricted)
            can_long = True
            can_short = not getattr(config, 'BUY_ONLY_MODE', False)  # ATLAS: Disable SELL if BUY_ONLY_MODE
            
            # --- PHASE 3: REGIME-ALIGNED TRADING ---
            # Strict Gate: If SMALL Regime + BEARISH Macro, Block Longs unless at SUPPORT
            if getattr(config, 'SMALL_REGIME_BEARISH_MODE', False):
                # How do we know we are in SMALL regime? We need to ask EntropyAgent or check balance?
                # Simpler: Assume we are in SMALL if balance < Config threshold, OR check macro_trend which implies 'BEARISH'
                # Actually, the requirement says "In SMALL regime + BEARISH macro".
                # We can approximate SMALL regime by ensuring we are not in BULL_MARKET global bias.
                
                # Check Global Bias from self.get_market_bias() or utilize existing macro_trend variable
                # macro_trend is from 1H data (local trend).
                
                if macro_trend == 'BEARISH':
                    # Only allow LONG if:
                    # 1. At Support (Buy the dip)
                    # 2. Whale Signal (Override)
                    # 3. Arb (Passed via is_whale flag usually, or needs explicit check?) - Arb has special bypass elsewhere.
                    
                    sls_zone = structure_ctx.get('sls_zone', 'NEUTRAL')
                    is_support = sls_zone == 'SUPPORT'
                    
                    if not is_support and not is_whale:
                        # RELAXED for SMALL Regime: Allow NEUTRAL if not in a critical crash
                        if structure_ctx.get('tda_critical', False):
                            can_long = False
                            # self._safe_print(f"[{self.name}] 🛡️ REGIME: Blocking {symbol} Long due to TDA_CRITICAL.")
                        else:
                            # self._safe_print(f"[{self.name}] 🔓 REGIME RELAXED: Allowing {symbol} Long in Neutral Zone (SMALL Regime).")
                            pass
                    elif is_whale and not is_support:
                         if getattr(config, 'WHALE_ALLOW_NEUTRAL_WITH_BID_WALL', True):
                             # self._safe_print(f"[{self.name}] 🐋 WHALE OVERRIDE: {symbol} Long Allowed in {sls_zone} due to Whale Signal.")
                             pass

            if macro_trend == 'BULLISH':
                if last_state != 'BULLISH':
                    self._safe_print(f"[{self.name}] 🌊 MACRO FLOW (1H): {symbol} Turned BULLISH.")
                    self.last_macro_state[symbol] = 'BULLISH'
                
                # RECALIBRATION: Enforce Trend congruence (Restrict Shorts in Bull markets)
                can_short = (True if is_whale else False) and (not getattr(config, 'BUY_ONLY_MODE', False))
                
            elif macro_trend == 'BEARISH':
                if last_state != 'BEARISH':
                    self._safe_print(f"[{self.name}] 🌊 MACRO FLOW (1H): {symbol} Turned BEARISH.")
                    self.last_macro_state[symbol] = 'BEARISH'
                
                # RECALIBRATION: Enforce Trend congruence (Restrict Longs in Bear markets)
                can_long = True if is_whale else False
        else:
             # No Observer = Safe Defaults
             can_long = True
             can_short = not getattr(config, 'BUY_ONLY_MODE', False)
             
        # Hard guard: BUY-only mode always disables shorting, regardless of upstream overrides
        if getattr(config, 'BUY_ONLY_MODE', False):
            can_short = False

        # ATLAS: Regime Filter - Only trade in favorable market conditions
        regime_filter = getattr(config, 'ENTRY_REGIME_FILTER', None)
        if regime_filter:
            normalized_allowed = {regime_aliases.get(str(r).upper(), str(r).upper()) for r in regime_filter}
            if normalized_entropy_regime not in normalized_allowed:
                self._safe_print(f"[{self.name}] 🚫 REGIME FILTER: {symbol} blocked (regime: {normalized_entropy_regime}, allowed: {sorted(normalized_allowed)})")
                return None
             
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
                      return self.apply_asset_personality(symbol, sig, prices=prices.values.tolist())
                 elif evo_sig.action == 'SELL' and can_short:
                      self._safe_print(f"[{self.name}] 🧬 ENSEMBLE VOTE: {symbol} Sell Signal ({evo_sig.reason})")
                      sig = GlobalTradeSignal(symbol=symbol, direction='SELL', size=1.0, price=current_p)
                      sig.conviction = 0.65
                      sig.metadata = {
                          'strategy': 'ENSEMBLE_EVO', 
                          'reason': evo_sig.reason,
                          'structure': structure_ctx
                      }
                      return self.apply_asset_personality(symbol, sig, prices=prices.values.tolist())
             except Exception as e:
                 # self._safe_print(f"[{self.name}] Evo Error: {e}")
                 pass 

        # 🔑 KEY 1.1: AI DEEP CONFIRMATION (LSTM + XGBoost)
        ai_bull_reason = []
        ai_bear_reason = []
        lstm_prob = 0.5
        xgb_prob = 0.5
        
        _entropy_ctx = structure_ctx.get('entropy_val') if structure_ctx else None
        # 1. LSTM (Time Series)
        if hasattr(self, 'model') and self.model:
             lstm_prob = self.predict_trend_lstm(prices, symbol, entropy_context=_entropy_ctx)
             if lstm_prob > 0.75: ai_bull_reason.append(f"LSTM({lstm_prob:.2f})")
             if lstm_prob < 0.25: ai_bear_reason.append(f"LSTM({lstm_prob:.2f})")

        # 2. XGBoost (Tabular)
        if hasattr(self, 'xgb_model') and self.xgb_model:
             feats = self._extract_ml_features(window_data, symbol) if hasattr(self, '_extract_ml_features') else None
             if feats is not None:
                 xgb_prob = self.predict_trend_xgboost(feats, entropy_context=_entropy_ctx)
                 if xgb_prob > 0.75: ai_bull_reason.append(f"XGB({xgb_prob:.2f})")
                 if xgb_prob < 0.25: ai_bear_reason.append(f"XGB({xgb_prob:.2f})")
        
        # 3. Consensus Trigger
        if ai_bull_reason and can_long:
             reason_str = "+".join(ai_bull_reason)
             self._safe_print(f"[{self.name}] 🧠 NEURAL ALERT: {symbol} AI Bullish [{reason_str}]")
             
             conviction = 0.80 if (lstm_prob > 0.75 and xgb_prob > 0.75) else 0.65
             sig = GlobalTradeSignal(symbol=symbol, direction='BUY', size=1.0, conviction=conviction, price=prices.iloc[-1])
             sig.conviction = conviction
             sig.metadata = {'strategy': 'NEURAL_HYBRID', 'reason': reason_str, 'structure': structure_ctx}
             if conviction >= 0.8:
                 sig.metadata['stack_conviction_bypass'] = True
             return self.apply_asset_personality(symbol, sig)
             
        elif ai_bear_reason and can_short:
             reason_str = "+".join(ai_bear_reason)
             self._safe_print(f"[{self.name}] 🧠 NEURAL ALERT: {symbol} AI Bearish [{reason_str}]")
             
             conviction = 0.80 if (lstm_prob < 0.25 and xgb_prob < 0.25) else 0.65
             sig = GlobalTradeSignal(symbol=symbol, direction='SELL', size=1.0, conviction=conviction, price=prices.iloc[-1])
             sig.conviction = conviction
             sig.metadata = {'strategy': 'NEURAL_HYBRID', 'reason': reason_str, 'structure': structure_ctx}
             if conviction >= 0.8:
                 sig.metadata['stack_conviction_bypass'] = True
             return self.apply_asset_personality(symbol, sig)
        
        # --- PATCH 6: TRIGGER D - STRUCTURAL RESONANCE (The Paralysis Breaker) ---
        # Direct override: If Structure is perfect (Bullish Trend + Support Zone), we FIRE.
        # This bypasses the ML/Ensemble hesitation logic.
        sls_zone = structure_ctx.get('sls_zone', 'NEUTRAL') if structure_ctx else 'NEUTRAL'
        tda_critical = structure_ctx.get('tda_critical', False) if structure_ctx else False

        # === FIX 2026-03-15: Walk-Forward Validation for Structural Resonance ===
        # Check if this signal has been failing recently
        wf_valid, recent_wr = self._validate_signal_walk_forward(symbol, 'STRUCTURAL_RESONANCE')
        if not wf_valid:
            self._safe_print(f"[{self.name}] 🚫 WALK-FORWARD VETO: {symbol} STRUCTURAL_RESONANCE (Recent WR: {recent_wr*100:.0f}%, Max Allowed: {100*(1-self.MAX_ACCEPTABLE_LOSS_RATE):.0f}%)")
            # Block signal - it's been failing recently
        elif can_long and macro_trend == 'BULLISH' and sls_zone == 'SUPPORT' and not tda_critical:
             # Get calibrated conviction based on actual performance
             calibrated_conviction = self._get_calibrated_conviction('STRUCTURAL_RESONANCE', recent_wr)

             # Low-conviction resonance trades are only allowed in ORDERED regime.
             # The Trader injects `entropy_regime` into structure_ctx.
             low_conv = calibrated_conviction <= getattr(config, 'STRUCTURAL_RESONANCE_LOW_CONV_THRESHOLD', 0.60)
             if low_conv:
                 sr_regime = normalized_entropy_regime
                 if sr_regime != 'ORDERED':
                     self._safe_print(f"[{self.name}] 🚫 RESONANCE VETO: {symbol} conviction {calibrated_conviction:.2f} requires ORDERED regime (got {sr_regime})")
                     return None

             self._safe_print(f"[{self.name}] 🏛️ TRIGGER D: Structural Resonance for {symbol} (Bullish + Support). Conviction: {calibrated_conviction:.2f} (was 0.80)")
             meta = {
                 'reason': 'STRUCTURAL_RESONANCE',
                 'structure': structure_ctx,
                 'is_whale': False,
                 'stack_conviction_bypass': calibrated_conviction >= 0.7,  # Only bypass if conviction >= 0.7
                 'walk_forward_validated': True,
                 'recent_win_rate': recent_wr
             }
             current_price = float(window_data['close'].iloc[-1])
             sig = GlobalTradeSignal(symbol=symbol, direction='BUY', size=1.0, price=current_price, metadata=meta, conviction=calibrated_conviction)
             return self.apply_asset_personality(symbol, sig, prices=prices.values.tolist())

        elif can_short and macro_trend == 'BEARISH' and sls_zone == 'RESISTANCE' and not tda_critical:
             calibrated_conviction = self._get_calibrated_conviction('STRUCTURAL_RESONANCE', recent_wr)

             low_conv = calibrated_conviction <= getattr(config, 'STRUCTURAL_RESONANCE_LOW_CONV_THRESHOLD', 0.60)
             if low_conv:
                 sr_regime = normalized_entropy_regime
                 if sr_regime != 'ORDERED':
                     self._safe_print(f"[{self.name}] 🚫 RESONANCE VETO: {symbol} conviction {calibrated_conviction:.2f} requires ORDERED regime (got {sr_regime})")
                     return None

             self._safe_print(f"[{self.name}] 🏛️ TRIGGER D: Structural Resonance for {symbol} (Bearish + Resistance). Conviction: {calibrated_conviction:.2f} (was 0.80)")
             meta = {
                 'reason': 'STRUCTURAL_RESONANCE',
                 'structure': structure_ctx,
                 'is_whale': False,
                 'stack_conviction_bypass': calibrated_conviction >= 0.7,
                 'walk_forward_validated': True,
                 'recent_win_rate': recent_wr
             }
             current_price = float(window_data['close'].iloc[-1])
             sig = GlobalTradeSignal(symbol=symbol, direction='SELL', size=1.0, price=current_price, metadata=meta, conviction=calibrated_conviction)
             return self.apply_asset_personality(symbol, sig, prices=prices.values.tolist())
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

        _entropy_ctx = structure_ctx.get('entropy_val') if structure_ctx else None
        lstm_prob = self.predict_trend_lstm(prices, symbol, entropy_context=_entropy_ctx)

        xgb_features = self._extract_ml_features(window_data, symbol)
        xgb_prob = self.predict_trend_xgboost(xgb_features, entropy_context=_entropy_ctx)
        
        # --- ENTROPY INTEGRATION (Chaotic Dampening) ---
        # If the market is CHAOTIC (High Entropy), we reduce our confidence in the ML models.
        if normalized_entropy_regime == 'CHAOTIC':
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

        # === 2026-03-20 AUDIT: MULTI-SIGNAL CONFIRMATION SYSTEM ===
        # Diversify beyond single-signal dependency. Score independent confirming signals.
        # Each TRUE trigger adds to the confirmation score. Require min score for entry.
        # Whale adds +1, but alone is not enough (requires at least one technical confirmation).
        buy_confirmation_score = 0
        buy_signals_active = []

        if trigger_trend_buy:
            buy_confirmation_score += 1
            buy_signals_active.append('TREND')
        if trigger_dip_buy:
            buy_confirmation_score += 1
            buy_signals_active.append('DIP')
        if trigger_panic_buy:
            buy_confirmation_score += 1
            buy_signals_active.append('PANIC')
        if trigger_quantum_buy:
            buy_confirmation_score += 1
            buy_signals_active.append('QUANTUM')
        if is_whale:
            buy_confirmation_score += 1
            buy_signals_active.append('WHALE')

        # Additional confirming signals (technical confluence)
        # Kalman value: price below Kalman estimate = undervalued
        if current_price < kalman_price:
            buy_confirmation_score += 1
            buy_signals_active.append('KALMAN_VALUE')
        # LSTM agrees with direction
        if lstm_prob > 0.55:
            buy_confirmation_score += 1
            buy_signals_active.append('LSTM_AGREE')

        # Minimum confirmation threshold (configurable)
        min_buy_confirmations = getattr(config, 'MIN_BUY_CONFIRMATIONS', 2)

        if (trigger_trend_buy or trigger_dip_buy or trigger_panic_buy or trigger_quantum_buy):
             # NEW: Require minimum signal confluence
             if buy_confirmation_score < min_buy_confirmations:
                 self._safe_print(f"[{self.name}] 🚫 CONFLUENCE VETO: {symbol} BUY — only {buy_confirmation_score} signals ({buy_signals_active}), need {min_buy_confirmations}")
                 return None

             # VALIDATION: Kalman Check (Relaxed)
             # Allow if Price < Kalman (Value) OR if High Conviction Bullish (Momentum)
             # OR if it's a Panic Buy (Value is extreme)
             if current_price < kalman_price or high_conv_bullish or trigger_panic_buy or trigger_quantum_buy:
                 reason = "TREND" if trigger_trend_buy else ("DIP" if trigger_dip_buy else "PANIC")
                 if is_whale:
                     main_whale_reason = whale_reason[0] if whale_reason else "FORCE_OVERRIDE"
                     reason = f"WHALE_{main_whale_reason}" # Override reason with primary whale driver
                     self._safe_print(f"[{self.name}] 🐋 WHALE SIGHTING: {symbol} - {', '.join(whale_reason) if whale_reason else 'Manual/Sector Force'}")
                 
                 # P2 FIX 2026-03-05: Check cooldown before generating whale/trend signal
                 cooldown_strategy = reason if is_whale else ("TREND" if trigger_trend_buy else "DIP")
                 if not self._check_signal_cooldown(symbol, cooldown_strategy):
                     return None

                 self._safe_print(f"[{self.name}] 🚀 {symbol} BUY SIGNAL ({reason}) | XGB:{xgb_prob:.2f} GMB:{market_bias:.2f} | Confluence: {buy_confirmation_score} ({', '.join(buy_signals_active)})")


                 meta = {
                      'is_whale': is_whale,
                      'whale_factors': whale_reason,
                      'structure': structure_ctx,
                      'reason': reason,
                      'sde_physics': ou_params,
                      'quantum_conviction': quantum_conviction,
                      'confirmation_score': buy_confirmation_score,
                      'confirming_signals': buy_signals_active,
                  }
                 sig = GlobalTradeSignal(symbol=symbol, direction='BUY', size=1.0, price=current_price, metadata=meta)
                 self._set_signal_cooldown(symbol, cooldown_strategy)
                 return self.apply_asset_personality(symbol, sig, prices=prices.values.tolist())

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
                 return self.apply_asset_personality(symbol, sig, prices=prices.values.tolist())

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
                     return self.apply_asset_personality(symbol, sig, prices=prices.values.tolist())
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
