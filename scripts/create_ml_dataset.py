"""
Create ML training dataset from extracted trade events.
Joins ENTRY and EXIT events, matches with features, and creates training samples.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 70)
print("ML TRAINING DATASET CREATION")
print("=" * 70)

# Load extracted trade events
raw_trades = pd.read_parquet('datasets/raw_trades_snapshot.parquet')
print(f"Loaded raw_trades_snapshot.parquet: {len(raw_trades)} events")

# Event distribution
print("\nEvent distribution:")
print(raw_trades['event'].value_counts())

# Separate entries and exits
entries = raw_trades[raw_trades['event'] == 'ENTRY'].copy()
exits = raw_trades[raw_trades['event'] == 'EXIT'].copy()
stops = raw_trades[raw_trades['event'] == 'STOP_PLACED'].copy()

print(f"\nEntries: {len(entries)}, Exits: {len(exits)}, Stops: {len(stops)}")

# Parse timestamps
entries['ts'] = pd.to_datetime(entries['timestamp'])
exits['ts'] = pd.to_datetime(exits['timestamp'])
stops['ts'] = pd.to_datetime(stops['timestamp'])

# Sort by timestamp
entries = entries.sort_values('ts').reset_index(drop=True)
exits = exits.sort_values('ts').reset_index(drop=True)

print("\n" + "=" * 70)
print("MATCHING ENTRY → EXIT PAIRS")
print("=" * 70)

# Match entries to exits by symbol and timestamp proximity
matched_trades = []
used_exits = set()

for idx, entry in entries.iterrows():
    symbol = entry['symbol']
    entry_time = entry['ts']
    session = entry['session_file']
    
    # Find matching exit (same symbol, after entry, within 24 hours, same session preferred)
    best_match = None
    best_exit_idx = None
    
    for exit_idx, exit_row in exits.iterrows():
        if exit_idx in used_exits:
            continue
        if exit_row['symbol'] != symbol:
            continue
        if exit_row['ts'] <= entry_time:
            continue
        
        # Calculate time difference
        time_diff_sec = (exit_row['ts'] - entry_time).total_seconds()
        
        # Must be within 24 hours
        if time_diff_sec > 86400:
            continue
        
        # Prefer same session
        same_session = (exit_row['session_file'] == session)
        
        # Score: lower is better (time diff + penalty for different session)
        score = time_diff_sec
        if not same_session:
            score += 3600  # 1 hour penalty
        
        if best_match is None or score < best_match['score']:
            best_match = {
                'entry_idx': idx,
                'exit_idx': exit_idx,
                'symbol': symbol,
                'entry_time': entry_time,
                'exit_time': exit_row['ts'],
                'entry_qty': entry['qty'],
                'entry_leverage': entry.get('leverage'),
                'exit_pnl_percent': exit_row['pnl_percent'],
                'time_diff_sec': time_diff_sec,
                'score': score,
                'entry_session': session,
                'exit_session': exit_row['session_file'],
            }
            best_exit_idx = exit_idx
    
    if best_match:
        matched_trades.append(best_match)
        used_exits.add(best_exit_idx)

print(f"Matched {len(matched_trades)} complete trades")

if matched_trades:
    trades_df = pd.DataFrame(matched_trades)
    
    # Add derived features
    trades_df['time_diff_minutes'] = trades_df['time_diff_sec'] / 60
    trades_df['pnl_direction'] = (trades_df['exit_pnl_percent'] > 0).astype(int)
    trades_df['is_loss'] = (trades_df['exit_pnl_percent'] < 0).astype(int)
    
    print("\nMatched trades summary:")
    print(trades_df[['symbol', 'entry_time', 'exit_time', 'exit_pnl_percent', 'time_diff_minutes']].to_string())
    
    print("\nSymbol distribution:")
    print(trades_df['symbol'].value_counts())
    
    print("\nPnL statistics:")
    print(trades_df['exit_pnl_percent'].describe())
    
    win_rate = (trades_df['exit_pnl_percent'] > 0).mean() * 100
    print(f"\nWin rate: {win_rate:.1f}%")
    
    # Save matched trades
    trades_df.to_parquet('datasets/complete_trades_v2.parquet', index=False)
    trades_df.to_csv('datasets/complete_trades_v2.csv', index=False)
    print(f"\nSaved {len(trades_df)} complete trades to datasets/complete_trades_v2.parquet")

# Now match with features
print("\n" + "=" * 70)
print("MATCHING WITH FEATURES")
print("=" * 70)

# Load rich features
if os.path.exists('datasets/rich_features.parquet'):
    rich = pd.read_parquet('datasets/rich_features.parquet')
    rich['ts'] = pd.to_datetime(rich['timestamp'])
    print(f"Loaded rich_features.parquet: {len(rich)} rows")
    
    # For each matched trade, find the closest feature row BEFORE entry
    feature_rows = []
    
    if matched_trades:
        for idx, trade in trades_df.iterrows():
            symbol = trade['symbol']
            entry_time = trade['entry_time']
            
            # Find closest feature row before entry time for same symbol
            mask = (rich['symbol'] == symbol) & (rich['ts'] <= entry_time)
            candidates = rich[mask]
            
            if len(candidates) > 0:
                closest = candidates.loc[candidates['ts'].idxmax()]
                feature_row = closest.to_dict()
                
                # Add trade outcome as target
                feature_row['trade_exit_pnl'] = trade['exit_pnl_percent']
                feature_row['trade_duration_min'] = trade['time_diff_minutes']
                feature_row['trade_is_win'] = 1 if trade['exit_pnl_percent'] > 0 else 0
                feature_row['trade_entry_time'] = entry_time
                feature_row['trade_exit_time'] = trade['exit_time']
                feature_row['trade_entry_qty'] = trade['entry_qty']
                feature_row['trade_entry_leverage'] = trade['entry_leverage']
                
                feature_rows.append(feature_row)
    
    if feature_rows:
        ml_df = pd.DataFrame(feature_rows)
        ml_df.to_parquet('datasets/ml_training_features_v2.parquet', index=False)
        ml_df.to_csv('datasets/ml_training_features_v2.csv', index=False)
        print(f"\nCreated ML dataset with {len(ml_df)} samples")
        print(f"Saved to datasets/ml_training_features_v2.parquet")
        
        print("\nFeature columns (numeric):")
        numeric_cols = ml_df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"  {len(numeric_cols)} columns: {numeric_cols[:15]}...")
        
        print("\nTarget distribution (exit_pnl):")
        print(ml_df['trade_exit_pnl'].describe())
        
        print("\nTarget distribution (is_win):")
        print(ml_df['trade_is_win'].value_counts())
    else:
        print("Could not match trades to feature rows - check timestamp alignment")
else:
    print("rich_features.parquet not found")

print("\n" + "=" * 70)
print("DATASET CREATION COMPLETE")
print("=" * 70)
