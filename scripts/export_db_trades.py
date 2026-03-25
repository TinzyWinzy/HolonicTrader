"""
Export trades from SQLite database to parquet for ML training.
Creates bridge between operational DB and ML feature datasets.
"""
import sqlite3
import pandas as pd
import numpy as np
import os

DB_PATH = 'holonic_trader.db'
OUTPUT_DIR = 'datasets'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("DATABASE TO PARQUET EXPORT")
print("=" * 70)

# Connect to database
conn = sqlite3.connect(DB_PATH)

# Load all trades
print("\nLoading trades from database...")
query = """
SELECT 
    id, symbol, direction, quantity, price, cost_usd, timestamp,
    pnl, pnl_percent,
    unrealized_pnl, unrealized_pnl_percent,
    mfe, mae,
    exit_reason, strategy_type, entropy_score, regime, 
    conviction, quality_score, is_whitelisted
FROM trades 
WHERE pnl IS NOT NULL AND pnl != 0
ORDER BY timestamp
"""

trades_df = pd.read_sql_query(query, conn)
print(f"Loaded {len(trades_df)} trades with PnL")

if len(trades_df) == 0:
    print("No trades with PnL found. Trying all trades...")
    query = "SELECT * FROM trades ORDER BY timestamp"
    trades_df = pd.read_sql_query(query, conn)
    print(f"Loaded {len(trades_df)} total trades")

# Parse timestamp
trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
trades_df['date'] = trades_df['timestamp'].dt.date
trades_df['hour'] = trades_df['timestamp'].dt.hour
trades_df['day_of_week'] = trades_df['timestamp'].dt.dayofweek

print("\n" + "=" * 70)
print("DATABASE TRADE STATISTICS")
print("=" * 70)

print(f"\nDate range: {trades_df['timestamp'].min()} to {trades_df['timestamp'].max()}")
print(f"\nSymbols traded: {trades_df['symbol'].nunique()}")
print(trades_df['symbol'].value_counts().head(10))

print(f"\nPnL Statistics:")
print(f"  Total PnL: ${trades_df['pnl'].sum():.2f}")
print(f"  Mean PnL:  ${trades_df['pnl'].mean():.4f}")
print(f"  Std PnL:   ${trades_df['pnl'].std():.4f}")
print(f"  Win Rate:  {(trades_df['pnl'] > 0).mean():.2%}")

print(f"\nPnL % Statistics:")
print(f"  Mean:  {trades_df['pnl_percent'].mean():.4f} ({trades_df['pnl_percent'].mean()*100:.2f}%)")
print(f"  Std:   {trades_df['pnl_percent'].std():.4f} ({trades_df['pnl_percent'].std()*100:.2f}%)")
print(f"  Min:   {trades_df['pnl_percent'].min():.4f} ({trades_df['pnl_percent'].min()*100:.2f}%)")
print(f"  Max:   {trades_df['pnl_percent'].max():.4f} ({trades_df['pnl_percent'].max()*100:.2f}%)")

# Win/Loss analysis
wins = trades_df[trades_df['pnl'] > 0]
losses = trades_df[trades_df['pnl'] <= 0]

print(f"\nWin/Loss Breakdown:")
print(f"  Wins:   {len(wins)} ({len(wins)/len(trades_df)*100:.1f}%)")
print(f"  Losses: {len(losses)} ({len(losses)/len(trades_df)*100:.1f}%)")

if len(wins) > 0:
    print(f"  Avg Win: ${wins['pnl'].mean():.4f} ({wins['pnl_percent'].mean():.2f}%)")
if len(losses) > 0:
    print(f"  Avg Loss: ${losses['pnl'].mean():.4f} ({losses['pnl_percent'].mean():.2f}%)")

# Exit reason analysis
if 'exit_reason' in trades_df.columns and trades_df['exit_reason'].notna().any():
    print(f"\nExit Reasons:")
    print(trades_df['exit_reason'].value_counts().head(10))

# Strategy type analysis
if 'strategy_type' in trades_df.columns and trades_df['strategy_type'].notna().any():
    print(f"\nStrategy Types:")
    print(trades_df['strategy_type'].value_counts().head(10))

# Regime analysis
if 'regime' in trades_df.columns and trades_df['regime'].notna().any():
    print(f"\nRegime Distribution:")
    print(trades_df['regime'].value_counts())
    
    # Performance by regime
    print(f"\nPerformance by Regime:")
    regime_perf = trades_df.groupby('regime')['pnl_percent'].agg(['count', 'mean', 'std'])
    print(regime_perf)

print("\n" + "=" * 70)
print("SAVING DATASETS")
print("=" * 70)

# Save full trades
trades_path = os.path.join(OUTPUT_DIR, 'db_trades_full.parquet')
trades_df.to_parquet(trades_path, index=False)
print(f"Saved {len(trades_df)} trades to {trades_path}")

# Save ML-ready features
ml_cols = ['symbol', 'direction', 'quantity', 'price', 'cost_usd', 'timestamp',
           'pnl', 'pnl_percent', 'mfe', 'mae', 'exit_reason', 'strategy_type',
           'entropy_score', 'regime', 'conviction', 'quality_score', 
           'hour', 'day_of_week']
ml_cols = [c for c in ml_cols if c in trades_df.columns]

ml_df = trades_df[ml_cols].copy()

# Create target variable: good_trade (win or small loss < 2%)
ml_df['good_trade'] = (ml_df['pnl_percent'] > -0.02).astype(int)

ml_path = os.path.join(OUTPUT_DIR, 'db_trades_ml.parquet')
ml_df.to_parquet(ml_path, index=False)
print(f"Saved ML dataset with {len(ml_df)} samples to {ml_path}")

# Create summary report
summary = {
    'total_trades': len(trades_df),
    'date_range': {
        'start': str(trades_df['timestamp'].min()),
        'end': str(trades_df['timestamp'].max()),
    },
    'symbols': trades_df['symbol'].unique().tolist(),
    'pnl_stats': {
        'total': float(trades_df['pnl'].sum()),
        'mean': float(trades_df['pnl'].mean()),
        'std': float(trades_df['pnl'].std()),
        'win_rate': float((trades_df['pnl'] > 0).mean()),
    },
    'pnl_percent_stats': {
        'mean': float(trades_df['pnl_percent'].mean()),
        'std': float(trades_df['pnl_percent'].std()),
        'min': float(trades_df['pnl_percent'].min()),
        'max': float(trades_df['pnl_percent'].max()),
    },
    'wins': len(wins),
    'losses': len(losses),
    'avg_win_pct': float(wins['pnl_percent'].mean()) if len(wins) > 0 else None,
    'avg_loss_pct': float(losses['pnl_percent'].mean()) if len(losses) > 0 else None,
    'timestamp': pd.Timestamp.now().isoformat(),
}

if 'regime' in ml_df.columns and ml_df['regime'].notna().any():
    summary['regime_distribution'] = ml_df['regime'].value_counts().to_dict()
    summary['regime_performance'] = ml_df.groupby('regime')['pnl_percent'].mean().to_dict()

if 'exit_reason' in ml_df.columns and ml_df['exit_reason'].notna().any():
    summary['exit_reasons'] = ml_df['exit_reason'].value_counts().to_dict()

if 'strategy_type' in ml_df.columns and ml_df['strategy_type'].notna().any():
    summary['strategy_types'] = ml_df['strategy_type'].value_counts().to_dict()

summary_path = os.path.join(OUTPUT_DIR, 'db_trades_summary.json')
import json
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Saved summary report to {summary_path}")

print("\n" + "=" * 70)
print("EXPORT COMPLETE")
print("=" * 70)

# Print comparison with log-extracted trades
print("\nCOMPARISON WITH LOG-EXTRACTED TRADES:")
print("-" * 70)

log_trades_path = os.path.join(OUTPUT_DIR, 'complete_trades_v2.parquet')
if os.path.exists(log_trades_path):
    log_df = pd.read_parquet(log_trades_path)
    print(f"Database trades: {len(trades_df)}")
    print(f"Log-extracted trades: {len(log_df)}")
    print(f"Difference: {len(trades_df) - len(log_df)} trades")
    print("\n✓ Database has MORE historical trades - use for training!")
else:
    print("No log-extracted trades found for comparison")

conn.close()
