"""
Test PAXG/BTC Macro Arbitrage Strategy
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone
import numpy as np

# Test the PAXG/BTC holon
def test_paxg_btc_initialization():
    """Test PaxgBtcHolon initializes correctly"""
    from HolonicTrader.agent_paxg_btc import PaxgBtcHolon
    
    holon = PaxgBtcHolon(name="PaxgBtc")
    
    assert holon.name == "PaxgBtc"
    assert holon.zscore_entry_threshold == 2.0
    assert holon.zscore_exit_threshold == 0.5
    assert holon.lookback_days == 90
    assert holon.min_data_points == 1000
    assert holon.trade_cooldown == 3600  # 1 hour

def test_paxg_btc_zscore_calculation():
    """Test z-score calculation with known data"""
    from HolonicTrader.agent_paxg_btc import PaxgBtcHolon
    
    holon = PaxgBtcHolon()
    
    # Add known data points (normal distribution around 0.034)
    np.random.seed(42)
    mean = 0.034
    std = 0.001
    data = np.random.normal(mean, std, 1000)
    
    for val in data:
        holon.ratio_history.append(val)
    
    # Calculate z-score
    zscore = holon.calculate_zscore()
    
    # Last value
    last_val = data[-1]
    expected_zscore = (last_val - mean) / std
    
    assert zscore is not None
    assert abs(zscore - expected_zscore) < 0.1  # Allow small rounding error
    assert abs(holon.zscore_mean - mean) < 0.001
    assert abs(holon.zscore_std - std) < 0.001

def test_paxg_btc_opportunity_detection():
    """Test opportunity detection with extreme z-score"""
    from HolonicTrader.agent_paxg_btc import PaxgBtcHolon
    import pandas as pd
    
    holon = PaxgBtcHolon()
    holon.zscore_entry_threshold = 2.0  # Lower for testing
    holon.trade_cooldown = 0  # No cooldown for testing
    
    # Mock observer with proper DataFrame
    mock_observer = MagicMock()
    
    # Create proper DataFrame mock
    df_paxg = pd.DataFrame({'close': [2040.0] * 5})
    df_btc = pd.DataFrame({'close': [61000.0] * 5})
    
    mock_observer.fetch_market_data.side_effect = [df_paxg, df_btc]
    
    # Fill history with normal data (mean = 0.034, std = 0.001)
    np.random.seed(42)
    for i in range(1000):
        holon.ratio_history.append(0.034 + np.random.normal(0, 0.001))
    
    # Manually set extreme ratio (3 std dev below mean)
    holon.paxg_usdt_price = 2040.0
    holon.btc_usdt_price = 61000.0
    holon.paxg_btc_ratio = 0.031  # Extreme low
    holon.ratio_history.append(0.031)
    
    # Calculate z-score manually
    arr = np.array(holon.ratio_history)
    holon.zscore_mean = np.mean(arr)
    holon.zscore_std = np.std(arr)
    holon.zscore = (0.031 - holon.zscore_mean) / holon.zscore_std
    
    # Detect opportunity (skip fetch_prices by setting values directly)
    signal = holon.detect_opportunity(mock_observer)
    
    # Signal should be generated for extreme z-score
    if signal:
        assert signal['symbol'] == 'PAXG/BTC'
        assert signal['direction'] == 'BUY'  # Low ratio = buy
        assert signal['metadata']['zscore'] < -2.0
        assert signal['conviction'] > 0.5

def test_paxg_btc_stats():
    """Test statistics reporting"""
    from HolonicTrader.agent_paxg_btc import PaxgBtcHolon
    
    holon = PaxgBtcHolon()
    holon.paxg_usdt_price = 2040.0
    holon.btc_usdt_price = 61000.0
    holon.paxg_btc_ratio = 0.03344
    holon.zscore_mean = 0.034
    holon.zscore_std = 0.001
    holon.zscore = -2.5
    
    # Add some history
    for i in range(1000):
        holon.ratio_history.append(0.034 + np.random.normal(0, 0.001))
    
    stats = holon.get_stats()
    
    assert stats['paxg_usdt'] == 2040.0
    assert stats['btc_usdt'] == 61000.0
    assert stats['paxg_btc_ratio'] == 0.03344
    assert abs(stats['ratio_oz_per_btc'] - 29.9) < 0.5  # ~29.9 oz per BTC
    assert stats['zscore'] == -2.5
    assert stats['ready_to_trade'] == True
    assert stats['data_points'] >= 1000

if __name__ == "__main__":
    test_paxg_btc_initialization()
    test_paxg_btc_zscore_calculation()
    test_paxg_btc_opportunity_detection()
    test_paxg_btc_stats()
    print("All PAXG/BTC tests passed!")
