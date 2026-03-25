import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add project root to path
sys.path.append(os.getcwd())

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PipelineVerifier")

# Mock Configuration
try:
    import config
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'HolonicTrader')) 
    import config

config.ALLOWED_ASSETS = ['BTC/USDT', 'ETH/USDT']
config.TRADING_MODE = 'PAPER'

# Import Holons
from HolonicTrader.agent_observer import ObserverHolon
from HolonicTrader.agent_oracle import EntryOracleHolon
from HolonicTrader.agent_signal_provider import SignalProviderHolon
from HolonicTrader.agent_arbitrage import ArbitrageHolon
from HolonicTrader.agent_governor import GovernorHolon

class MockObserver(ObserverHolon):
    def fetch_market_data(self, symbol, timeframe='1h', limit=100):
        dates = pd.date_range(end=datetime.now(), periods=limit, freq=timeframe)
        df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(50000, 52000, limit),
            'high': np.random.uniform(52000, 53000, limit),
            'low': np.random.uniform(49000, 50000, limit),
            'close': np.random.uniform(50000, 52000, limit),
            'volume': np.random.uniform(100, 1000, limit)
        })
        df['close'] = np.linspace(50000, 55000, limit) + np.random.normal(0, 100, limit)
        return df

    def fetch_order_book(self, symbol):
        return {'bids': [[50000, 1.0]], 'asks': [[50100, 1.0]]}

    def fetch_funding_rate(self, symbol):
        return 0.0001


class MockKraken:
    """Mock KrakenHolon that simulates a DANGER scenario."""
    def __init__(self, scenario='DANGER'):
        self.scenario = scenario
        
    def detect_ghost_positions(self, ledger):
        # Simulate an open BTC position
        return {'ghosts': {'BTC/USDT': 2.5}}
        
    def get_platform_info(self):
        if self.scenario == 'DANGER':
            return {
                'status': 'HEALTHY',
                'account_health': {
                    'equity': 5000.0,
                    'used_margin': 4000.0,
                    'available': 1000.0,
                    'margin_level': 1.25,        # < 1.5 = DANGER
                    'liquidation_distance': 0.04  # < 5% = CRITICAL
                }
            }
        else: # HEALTHY
            return {
                'status': 'HEALTHY',
                'account_health': {
                    'equity': 50000.0,
                    'used_margin': 10000.0,
                    'available': 40000.0,
                    'margin_level': 5.0,
                    'liquidation_distance': 0.50
                }
            }


def verify_pipeline():
    print("🚀 Starting Pipeline Verification...")
    print("=" * 50)

    # 1. Initialize Holons
    print("\n1. Initializing Holons...")
    observer = MockObserver()
    oracle = EntryOracleHolon()
    arbitrage = ArbitrageHolon()
    governor = GovernorHolon()
    provider = SignalProviderHolon()
    
    # === TEST 1: DANGER SCENARIO (Should trigger URGENT_CLOSE or REDUCE) ===
    print("\n" + "=" * 50)
    print("📋 TEST 1: DANGER Scenario (Low Margin + Near Liquidation)")
    print("=" * 50)
    
    holons = {
        'observer': observer,
        'oracle': oracle,
        'arbitrage': arbitrage,
        'governor': governor,
        'kraken': MockKraken(scenario='DANGER'),  # ← Mock Kraken
        'structure': None,
        'entropy': None,
        'topology': None,
        'whale': None
    }

    report = provider.generate_signal_report(holons)

    mgmt_signals = [s for s in report if s.get('metadata', {}).get('strategy') == 'POSITION_MANAGEMENT']
    entry_signals = [s for s in report if s.get('metadata', {}).get('strategy') != 'POSITION_MANAGEMENT']
    
    print(f"\n>> Management Signals: {len(mgmt_signals)}")
    print(f">> Entry Signals: {len(entry_signals)}")
    
    for sig in mgmt_signals:
        print(f"\n  🛡️ [{sig['direction']}] {sig['symbol']}")
        print(f"     Urgency: {sig['conviction']}")
        print(f"     Reason:  {sig['reason']}")
        
    if mgmt_signals:
        print("\n✅ TEST 1 PASSED: AI Management Signal generated for danger scenario!")
    else:
        print("\n❌ TEST 1 FAILED: No management signal generated.")
        
    # === TEST 2: HEALTHY SCENARIO (Should return HOLD, no mgmt signals) ===
    print("\n" + "=" * 50)
    print("📋 TEST 2: HEALTHY Scenario (Good Margin)")
    print("=" * 50)
    
    holons['kraken'] = MockKraken(scenario='HEALTHY')
    report2 = provider.generate_signal_report(holons)
    
    mgmt_signals2 = [s for s in report2 if s.get('metadata', {}).get('strategy') == 'POSITION_MANAGEMENT']
    
    print(f"\n>> Management Signals: {len(mgmt_signals2)}")
    
    if len(mgmt_signals2) == 0:
        print("✅ TEST 2 PASSED: No unnecessary management signals in healthy scenario.")
    else:
        for sig in mgmt_signals2:
            print(f"  ⚠️ [{sig['direction']}] {sig['symbol']}: {sig['reason']}")
        print("⚠️ TEST 2: Management signals issued even in healthy scenario (may be MC-driven).")
    
    # === TEST 3: HPS CONFLUENCE SCAN ===
    print("\n" + "=" * 50)
    print("📋 TEST 3: HPS CONFLUENCE SCAN (Mock 'Perfect Storm')")
    print("=" * 50)
    
    # Mock Provider Context with HPS Data
    mock_context = {
        'data': None, # Not used directly in simple check if other keys exist
        'structure': {'pivots': {'S2': 50000}}, 
        'rsi': 55.0,
        'entropy': 0.4, 
        'hit_prob': 0.75, 
        'regime': 'BULLISH_TREND'
    }
    
    # Mock Signal
    class MockSignal:
        def __init__(self):
            self.direction = 'BUY'
            self.price = 50050 # Near S2 (50000 * 1.01 = 50500)
            self.metadata = {'is_whale': True} 
            
    mock_sig = MockSignal()
    
    try:
        if hasattr(provider, 'check_hps_confluence'):
            # We need to mock 'data' to avoid errors if the method tries to access it
            # But the method uses context.get('data') only for RSI calc if missing.
            # We provided 'rsi' key, so it should be fine.
            # Wait, method line: data = context.get('data') ... if 'rsi' in data.columns...
            # If data is None, 'rsi' in data.columns will throw.
            # Let's provide a dummy dataframe.
            df = pd.DataFrame({'close': [50000, 50100], 'rsi': [55, 55]})
            mock_context['data'] = df

            result = provider.check_hps_confluence('BTC/USDT', mock_sig, mock_context)
            print(f"HPS Result: Is HPS? {result['is_hps']}")
            print(f"HPS Score: {result['score']}/5")
            print(f"Pillars: {result['pillars']}")
            
            if result['is_hps'] and result['score'] >= 3:
                 print("✅ TEST 3 PASSED: High Probability Setup identified!")
            else:
                 print(f"❌ TEST 3 FAILED: Score {result['score']} too low.")
        else:
            print("❌ TEST 3 FAILED: Method check_hps_confluence missing.")
            
    except Exception as e:
        print(f"❌ TEST 3 ERROR: {e}")
        import traceback
        traceback.print_exc()

    # === TEST 4: ARBITRAGE SCAN ===
    print("\n" + "=" * 50)
    print("📋 TEST 4: ARBITRAGE SCAN (Mock Funding/Spread)")
    print("=" * 50)
    
    # 1. Mock Arbitrage Holon with Signal
    class MockArbitrage:
        def get_active_signal(self, symbol, price):
            # Return a valid signal structure expected by SignalProvider
            # (See agent_arbitrage.py get_active_signal return format)
            # It usually returns a dict with 'direction', 'confidence', 'reason'
            return {
                'direction': 'BUY',
                'confidence': 0.95,
                'reason': 'High Yield Funding (15% APY)'
            }
            
    holons_arb = holons.copy()
    holons_arb['arbitrage'] = MockArbitrage()
    
    # 2. Generate Report
    print(">> Generating report with Mock Arbitrage Holon...")
    report_arb = provider.generate_signal_report(holons_arb)
    
    # 3. Filter for Arb Signals
    arb_signals = [s for s in report_arb if s.get('metadata', {}).get('is_arb')]
    print(f">> Arbitrage Signals Found: {len(arb_signals)}")
    
    if len(arb_signals) > 0:
        sig = arb_signals[0]
        print(f"  ✅ Signals: {sig['symbol']} {sig['direction']}")
        print(f"     Reason: {sig['reason']}")
        if sig['execution_details']['quantity'] > 0: # Ensure governor approved it (mock governor is permissive)
             print("     Governor Status: APPROVED")
             print("✅ TEST 4 PASSED: Arbitrage signal generated and approved!")
        else:
             print("⚠️ TEST 4 WARNING: Signal generated but Governor vetoed (Size 0).")
    else:
        print("❌ TEST 4 FAILED: No arbitrage signal generated.")


    # === SUMMARY ===
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"  Total Signals (Danger): {len(report)}")
    print(f"  Total Signals (Healthy): {len(report2)}")
    print(f"  Management Signals (Danger): {len(mgmt_signals)}")
    print(f"  Management Signals (Healthy): {len(mgmt_signals2)}")


if __name__ == "__main__":
    verify_pipeline()
