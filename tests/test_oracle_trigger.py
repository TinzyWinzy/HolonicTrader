
import sys
import os
import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock


# Add project root (Outer HolonicTrader) to path so we can resolve 'HolonicTrader' package
# test file is at Outer/tests/test_oracle_trigger.py
# .. is Outer/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock config
sys.modules['config'] = MagicMock()
import config
# ... (config setups remain same, just ensuring correct indent/flow if I replace block)
config.SATELLITE_BBW_EXPANSION_THRESHOLD = 0.05
config.SATELLITE_RVOL_THRESHOLD = 5.0
config.SATELLITE_ENTRY_RSI_CAP = 70
config.GMB_THRESHOLD = 0.40
config.STRATEGY_RSI_ENTRY_MAX = 50
config.STRATEGY_RSI_PANIC_BUY = 30
config.STRATEGY_XGB_THRESHOLD = 0.75
config.MEMECOIN_ASSETS = []
config.MEMECOIN_PUMP_RVOL = 10.0
config.SATELLITE_ASSETS = []
config.SENTIMENT_WEIGHT = 0.5
config.WHALE_ACCUMULATION_ATR_FACTOR = 0.5
config.WHALE_ACCUMULATION_RVOL = 3.0
config.WHALE_DEFENSE_RVOL = 3.0
config.WHALE_FUNDING_SQUEEZE_THRESHOLD = -0.01
config.VOL_WINDOW_FUNDING_THRESHOLD = 0.01
config.WHALE_RVOL_THRESHOLD = 5.0
config.WHALE_ORDER_IMBALANCE_RATIO = 2.0
config.USE_OPENVINO = False
config.USE_INTEL_GPU = False
config.POLYMARKET_PATIENCE_MINUTES = 3 
config.PERSONALITY_BTC_ATR_FILTER = 0.8
config.PERSONALITY_DOGE_RVOL = 2.0
config.PERSONALITY_SOL_RSI_LONG = 40
config.PERSONALITY_SOL_RSI_SHORT = 60
config.PHYSICS_MIN_RVOL = 1.0
config.PHYSICS_OU_STRETCH_THRESHOLD = 2.0
config.MACRO_STACK_WEIGHT = 0.15
config.STRUCTURE_DRIVES_TARGETS = True

# Mock PPO Agent and Monitors that might be imported
sys.modules['HolonicTrader.agent_ppo'] = MagicMock()
sys.modules['performance_tracker'] = MagicMock()

from HolonicTrader.agent_oracle import EntryOracleHolon as OracleHolon
from HolonicTrader.agent_executor import TradeSignal
# GlobalTradeSignal is just an alias inside agent_oracle. 
# We can check against TradeSignal as it's the underlying type.


class TestOracleTriggerD(unittest.TestCase):
    def setUp(self):
        self.oracle = OracleHolon(name="TestOracle")
        # Ensure models are None so we bypass ML
        self.oracle.model = None
        self.oracle.xgb_model = None
        self.oracle.ensemble = None
        self.oracle.DEBUG = True
        
        self.oracle.provider = MagicMock()
        mock_signal = TradeSignal("BTC/USDT", "BUY", 1.0, 0.5, {})
        self.oracle.provider.generate_signal_from_trigger.return_value = mock_signal
        
        self.oracle.macro_oracle = MagicMock()
        self.oracle.macro_oracle.fetch_macro_context.return_value = {'bias_score': 0.5}

    def test_trigger_d_bullish_support(self):
        """
        Test that Trigger D fires valid BUY signal when:
        - Macro Trend is BULLISH
        - Zone is SUPPORT
        - TDA Critical is False
        - Minutes into candle < Patience (Should FAIL if patience check is strict, but user might have relaxed it. 
          Actually, code says if minutes < patience -> pass (wait). Correction: If minutes < patience it might return None.
          We should set minutes_into_candle > patience.)
        """
        symbol = "BTC/USDT"
        
        # 1. Create Dummy Market Data
        # 100 periods of closing prices
        dates = pd.date_range(start="2025-01-01", periods=100, freq="15min")
        data = pd.DataFrame({
            'timestamp': dates,
            'close': np.linspace(50000, 51000, 100), # Slight uptrend
            'high': np.linspace(50050, 51050, 100),
            'low': np.linspace(49950, 50950, 100),
            'open': np.linspace(50000, 50900, 100),
            'volume': np.random.normal(100, 10, 100)
        })
        
        # 2. Indicators (BB, OBV)
        close = data['close'].iloc[-1]
        bb_vals = {'upper': close * 1.02, 'middle': close, 'lower': close * 0.98}
        obv_slope = 0.5
        metabolism = 'PREDATOR'
        
        # 3. Structure Context (THE KEY)
        structure_ctx = {
            'macro_trend': 'BULLISH',
            'sls_zone': 'SUPPORT',
            'tda_critical': False,
            'minutes_into_candle': 10 # Past patience threshold
        }
        
        # 4. Mock dependencies
        observer = MagicMock()
        
        # 5. Run Analysis
        # We expect a TradeSignal because of Trigger D Logic override
        # even with no ML models.
        
        # Mocking get_market_bias to avoid NoneType errors during logging if called
        self.oracle.get_market_bias = MagicMock(return_value=0.5)
        # Mock get_kalman_estimate
        self.oracle.get_kalman_estimate = MagicMock(return_value=close * 1.01) # Kalman above price usually bullish but check logic
        
        # The function signature:
        # analyze_for_entry(self, symbol, window_data, bb_vals, obv_slope, metabolism_state, ...)
        
        signal = self.oracle.analyze_for_entry(
            symbol=symbol,
            window_data=data,
            bb_vals=bb_vals,
            obv_slope=obv_slope,
            metabolism_state=metabolism,
            structure_ctx=structure_ctx,
            book_data={},
            funding_rate=0.0,
            observer=observer
        )
        
        # 6. Assertions
        self.assertIsNotNone(signal, "Trigger D failed to generate a signal! Paralysis Persists.")
        if signal:
             self.assertEqual(signal.direction, 'BUY', "Signal should be BUY")
             self.assertEqual(signal.metadata.get('reason'), 'STRUCTURAL_RESONANCE', "Reason should be STRUCTURAL_RESONANCE")
             print("\n✅ Trigger D Test Passed: Signal Generated Successfully.")

    def test_trigger_d_veto_if_tda_critical(self):
        """
        Test that Trigger D does NOT fire if TDA is CRITICAL.
        """
        symbol = "BTC/USDT"
        dates = pd.date_range(start="2025-01-01", periods=100, freq="15min")
        data = pd.DataFrame({
            'timestamp': dates, 
            'close': [50000]*100, 'high': [50100]*100, 'low': [49900]*100, 'open': [50000]*100, 'volume': [100]*100
        })
        
        structure_ctx = {
            'macro_trend': 'BULLISH',
            'sls_zone': 'SUPPORT',
            'tda_critical': True, # <--- CRITICAL VETO
            'minutes_into_candle': 10
        }
        
        self.oracle.get_market_bias = MagicMock(return_value=0.5)
        self.oracle.get_kalman_estimate = MagicMock(return_value=50000)
        
        signal = self.oracle.analyze_for_entry(
            symbol=symbol, window_data=data, bb_vals={'upper': 51000, 'middle': 50000, 'lower': 49000},
            obv_slope=0, metabolism_state='PREDATOR', structure_ctx=structure_ctx
        )
        
        self.assertIsNone(signal, "Trigger D should be vetoed by TDA Critical status")
        print("\n✅ TDA Veto Test Passed: Signal suppressed.")

if __name__ == '__main__':
    unittest.main()
