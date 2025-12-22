"""
HolonicTrader Backtest Simulation
Runs strategy over historical data and generates performance report
"""

import pandas as pd
import numpy as np
from datetime import datetime
from HolonicTrader.agent_observer import ObserverHolon
from HolonicTrader.agent_entropy import EntropyHolon
from HolonicTrader.agent_strategy import StrategyHolon
from HolonicTrader.agent_governor import GovernorHolon
from HolonicTrader.agent_executor import ExecutorHolon
from HolonicTrader.holon_core import Disposition
import config
import sys

# Hack: If QueueLogger is needed, importing from main might be circular. 
# For now, we just accept the arg to prevent crash.

def calculate_bollinger_bands(df, period=20, std=2):
    """Calculate Bollinger Bands."""
    df['sma'] = df['close'].rolling(window=period).mean()
    df['std'] = df['close'].rolling(window=period).std()
    df['bb_upper'] = df['sma'] + (std * df['std'])
    df['bb_middle'] = df['sma']
    df['bb_lower'] = df['sma'] - (std * df['std'])
    return df

def calculate_atr(df, period=14):
    """Calculate Average True Range."""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period).mean()
    
    return atr

def run_backtest(status_queue=None, symbol='XRP/USDT', start_date=None, end_date=None):
    """
    Run backtest simulation for a single asset.
    Accepts status_queue for GUI compatibility.
    """
    print(f"\n{'='*60}")
    print(f"BACKTEST SIMULATION: {symbol}")
    print(f"{'='*60}\n")
    
    # 1. Load Data
    print("[1/6] Loading historical data...")
    observer = ObserverHolon(exchange_id='kucoin')
    df = observer.load_local_history(symbol)
    
    if df.empty:
        print(f"❌ No data available for {symbol}")
        return None
    
    # Filter by date if specified
    if start_date:
        df = df[df['timestamp'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['timestamp'] <= pd.to_datetime(end_date)]
    
    print(f"   Loaded {len(df)} candles")
    print(f"   Period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    
    # 2. Initialize Agents
    print("\n[2/6] Initializing Holonic Agents...")
    entropy = EntropyHolon()
    strategy = StrategyHolon()
    governor = GovernorHolon(initial_balance=config.INITIAL_CAPITAL)
    executor = ExecutorHolon(
        initial_capital=config.INITIAL_CAPITAL,
        governor=governor,
        use_compounding=False,
        fixed_stake=1.0
    )
    
    # 3. Calculate Indicators
    print("\n[3/6] Computing indicators...")
    df = calculate_bollinger_bands(df, config.BB_PERIOD, config.BB_STD)
    df['atr'] = calculate_atr(df, config.ATR_PERIOD)
    
    # Calculate OBV
    obv_series = strategy.calculate_obv(df)
    df['obv'] = obv_series
    df['obv_slope'] = obv_series.rolling(14).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 14 else 0
    )
    
    df.dropna(inplace=True)
    print(f"   Indicators ready for {len(df)} candles")
    
    # 4. Simulate Trading
    print("\n[4/6] Running simulation...")
    trades = []
    positions = []
    
    for idx, row in df.iterrows():
        # Calculate regime
        returns_window = df.loc[:idx, 'returns'].tail(50)
        if len(returns_window) < 20:
            continue
            
        entropy_val = entropy.calculate_shannon_entropy(returns_window)
        regime = entropy.determine_regime(entropy_val)
        
        # Get metabolism state
        metabolism = governor.get_metabolism_state()
        
        # Build BB dict
        bb = {
            'upper': row['bb_upper'],
            'middle': row['bb_middle'],
            'lower': row['bb_lower']
        }
        
        current_price = row['close']
        obv_slope = row['obv_slope']
        
        # Check for entry
        if executor.balance_asset == 0:  # Not in position
            signal = strategy.analyze_for_entry(
                window_data=df.loc[:idx].tail(100),
                bb=bb,
                obv_slope=obv_slope,
                metabolism_state=metabolism
            )
            
            if signal and signal.direction == 'BUY':
                # Execute buy
                executor.balance_usd -= 1.0
                executor.balance_asset = 1.0 / current_price
                executor.entry_price = current_price
                
                trades.append({
                    'timestamp': row['timestamp'],
                    'type': 'BUY',
                    'price': current_price,
                    'regime': regime,
                    'metabolism': metabolism,
                    'obv_slope': obv_slope
                })
                positions.append(idx)
                
        else:  # In position
            signal = strategy.analyze_for_exit(
                current_price=current_price,
                entry_price=executor.entry_price,
                bb=bb,
                atr=row['atr'],
                metabolism_state=metabolism
            )
            
            if signal and signal.direction == 'SELL':
                # Execute sell
                usd_value = executor.balance_asset * current_price
                pnl = usd_value - 1.0
                
                executor.balance_usd += usd_value
                executor.balance_asset = 0
                executor.entry_price = None
                
                trades.append({
                    'timestamp': row['timestamp'],
                    'type': 'SELL',
                    'price': current_price,
                    'pnl': pnl,
                    'regime': regime,
                    'metabolism': metabolism
                })
    
    # 5. Generate Report
    print("\n[5/6] Generating performance report...")
    
    total_trades = len([t for t in trades if t['type'] == 'BUY'])
    winning_trades = len([t for t in trades if t['type'] == 'SELL' and t.get('pnl', 0) > 0])
    losing_trades = len([t for t in trades if t['type'] == 'SELL' and t.get('pnl', 0) <= 0])
    
    total_pnl = sum([t.get('pnl', 0) for t in trades if t['type'] == 'SELL'])
    final_balance = executor.balance_usd + (executor.balance_asset * df['close'].iloc[-1] if executor.balance_asset > 0 else 0)
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # 6. Print Results
    print(f"\n{'='*60}")
    print("BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Symbol:          {symbol}")
    print(f"Period:          {df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[-1].date()}")
    print(f"Candles:         {len(df)}")
    print(f"\nTRADE STATS:")
    print(f"Total Trades:    {total_trades}")
    print(f"Winning Trades:  {winning_trades}")
    print(f"Losing Trades:   {losing_trades}")
    print(f"Win Rate:        {win_rate:.1f}%")
    print(f"\nPERFORMANCE:")
    print(f"Initial Capital: ${config.INITIAL_CAPITAL:.2f}")
    print(f"Final Balance:   ${final_balance:.2f}")
    print(f"Total PnL:       ${total_pnl:.2f}")
    print(f"ROI:             {(total_pnl / config.INITIAL_CAPITAL * 100):.1f}%")
    print(f"{'='*60}\n")
    
    return {
        'symbol': symbol,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'final_balance': final_balance,
        'trades': trades
    }

def run_multi_asset_backtest():
    """Run backtest across all ALLOWED_ASSETS."""
    print("\n🚀 MULTI-ASSET BACKTEST SIMULATION")
    print("=" * 60)
    
    results = []
    for symbol in config.ALLOWED_ASSETS:
        if symbol == 'BTC/USD':  # Skip if no live sync
            symbol_test = 'BTC/USDT'  # Test with USDT pair if needed
        else:
            symbol_test = symbol
            
        result = run_backtest(symbol_test)
        if result:
            results.append(result)
    
    # Aggregate Results
    if results:
        print("\n" + "="*60)
        print("AGGREGATE MULTI-ASSET RESULTS")
        print("="*60)
        
        total_pnl = sum([r['total_pnl'] for r in results])
        avg_win_rate = np.mean([r['win_rate'] for r in results])
        total_trades = sum([r['total_trades'] for r in results])
        
        print(f"Assets Tested:     {len(results)}")
        print(f"Total Trades:      {total_trades}")
        print(f"Average Win Rate:  {avg_win_rate:.1f}%")
        print(f"Combined PnL:      ${total_pnl:.2f}")
        print(f"Combined ROI:      {(total_pnl / (config.INITIAL_CAPITAL * len(results)) * 100):.1f}%")
        print("="*60)

if __name__ == "__main__":
    # Run single asset test
    # run_backtest('XRP/USDT', start_date='2025-01-01')
    
    # Run multi-asset test
    run_multi_asset_backtest()
