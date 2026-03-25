"""
Join ENTRY and EXIT events to create complete trade records for ML training.
Matches by symbol and timestamp proximity within the same session.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Load entry and exit events
entry_df = pd.read_parquet('datasets/all_entry_events.parquet')
exit_df = pd.read_parquet('datasets/all_exit_events.parquet')

print("=" * 70)
print("JOINING ENTRY AND EXIT EVENTS")
print("=" * 70)
print(f"Entry events: {len(entry_df)}")
print(f"Exit events: {len(exit_df)}")

# Parse timestamps
entry_df['ts'] = pd.to_datetime(entry_df['timestamp'])
exit_df['ts'] = pd.to_datetime(exit_df['timestamp'])

# Sort by timestamp
entry_df = entry_df.sort_values('ts').reset_index(drop=True)
exit_df = exit_df.sort_values('ts').reset_index(drop=True)

# Match entries to exits by symbol and timestamp proximity
# An exit matches an entry if:
# 1. Same symbol
# 2. Exit is after entry
# 3. Exit is within 24 hours of entry
# 4. Entry hasn't been matched yet

matched_trades = []
used_exits = set()

for idx, entry in entry_df.iterrows():
    symbol = entry['symbol']
    entry_time = entry['ts']
    session = entry['session_file']
    
    # Find matching exit
    best_match = None
    best_exit_idx = None
    
    for exit_idx, exit_row in exit_df.iterrows():
        if exit_idx in used_exits:
            continue
        if exit_row['symbol'] != symbol:
            continue
        if exit_row['ts'] <= entry_time:
            continue
        if exit_row['session_file'] != session:
            # Allow cross-session if within 1 hour
            time_diff = (exit_row['ts'] - entry_time).total_seconds()
            if time_diff > 3600:
                continue
        
        # Found a match
        time_diff = (exit_row['ts'] - entry_time).total_seconds()
        if best_match is None or time_diff < best_match['time_diff_sec']:
            best_match = {
                'entry_idx': idx,
                'exit_idx': exit_idx,
                'symbol': symbol,
                'entry_time': entry_time,
                'exit_time': exit_row['ts'],
                'entry_qty': entry['qty'],
                'entry_leverage': entry['leverage'],
                'exit_pnl_percent': exit_row['pnl_percent'],
                'time_diff_sec': time_diff,
                'entry_session': session,
                'exit_session': exit_row['session_file'],
            }
            best_exit_idx = exit_idx
    
    if best_match:
        matched_trades.append(best_match)
        used_exits.add(best_exit_idx)

print(f"\nMatched trades: {len(matched_trades)}")

# Create DataFrame
if matched_trades:
    trades_df = pd.DataFrame(matched_trades)
    
    # Add derived features
    trades_df['time_diff_minutes'] = trades_df['time_diff_sec'] / 60
    trades_df['pnl_direction'] = (trades_df['exit_pnl_percent'] > 0).astype(int)
    
    print("\nMatched trades summary:")
    print(trades_df[['symbol', 'entry_time', 'exit_time', 'exit_pnl_percent', 'time_diff_minutes']].to_string())
    
    print("\nSymbol distribution:")
    print(trades_df['symbol'].value_counts())
    
    print("\nPnL statistics:")
    print(trades_df['exit_pnl_percent'].describe())
    
    print(f"\nWin rate: {(trades_df['exit_pnl_percent'] > 0).mean()*100:.1f}%")
    
    # Save to parquet
    trades_df.to_parquet('datasets/complete_trades.parquet', index=False)
    print(f"\nSaved {len(trades_df)} complete trades to datasets/complete_trades.parquet")
    
    # Also save as CSV for inspection
    trades_df.to_csv('datasets/complete_trades.csv', index=False)
    print(f"Saved datasets/complete_trades.csv")
else:
    print("No matched trades found!")

# Now let's also create a feature dataset by combining with existing features
print("\n" + "=" * 70)
print("CREATING ML TRAINING DATASET")
print("=" * 70)

# Load existing features
if os.path.exists('datasets/rich_features.parquet'):
    rich = pd.read_parquet('datasets/rich_features.parquet')
    rich['ts'] = pd.to_datetime(rich['timestamp'])
    
    # For each matched trade, find the corresponding feature row
    feature_rows = []
    for idx, trade in trades_df.iterrows():
        symbol = trade['symbol']
        entry_time = trade['entry_time']
        
        # Find closest feature row before entry time
        mask = (rich['symbol'] == symbol) & (rich['ts'] <= entry_time)
        candidates = rich[mask]
        
        if len(candidates) > 0:
            closest = candidates.loc[candidates['ts'].idxmax()]
            feature_row = closest.to_dict()
            feature_row['trade_exit_pnl'] = trade['exit_pnl_percent']
            feature_row['trade_duration_min'] = trade['time_diff_minutes']
            feature_row['trade_is_win'] = 1 if trade['exit_pnl_percent'] > 0 else 0
            feature_rows.append(feature_row)
    
    if feature_rows:
        ml_df = pd.DataFrame(feature_rows)
        ml_df.to_parquet('datasets/ml_training_features.parquet', index=False)
        print(f"Created ML dataset with {len(ml_df)} samples")
        print(f"Saved to datasets/ml_training_features.parquet")
        
        print("\nFeature columns available:")
        numeric_cols = ml_df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"  {len(numeric_cols)} numeric columns: {numeric_cols[:10]}...")
        
        print("\nTarget distribution (exit_pnl):")
        print(ml_df['trade_exit_pnl'].describe())
    else:
        print("Could not match trades to feature rows")
else:
    print("rich_features.parquet not found")
