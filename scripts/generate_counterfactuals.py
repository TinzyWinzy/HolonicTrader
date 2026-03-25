"""
Generate Counterfactual Training Data

Creates synthetic training examples to:
1. Simulate slippage effects
2. Generate opposite position scenarios
3. Augment limited dataset
4. Reduce selection bias

Usage:
    python scripts/generate_counterfactuals.py
"""
import pandas as pd
import numpy as np
import os

BASE = os.path.join(os.path.dirname(__file__), '..')
DB_TRADES = os.path.join(BASE, 'datasets', 'db_trades_ml.parquet')
OUTPUT_DIR = os.path.join(BASE, 'datasets')

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("COUNTERFACTUAL DATA GENERATION")
print("=" * 70)

# Load original trades
if not os.path.exists(DB_TRADES):
    print(f'Missing: {DB_TRADES}')
    print('Run: python scripts/export_db_trades.py first')
    raise SystemExit(1)

df = pd.read_parquet(DB_TRADES)
print(f"Loaded {len(df)} original trades")

# ============================================================================
# Counterfactual 1: Opposite Position
# ============================================================================
print("\n1. Generating opposite position counterfactuals...")

opposite = df.copy()
opposite['direction'] = opposite['direction'].apply(lambda x: 'SELL' if x == 'BUY' else 'BUY')
opposite['pnl_percent'] = -opposite['pnl_percent']  # Opposite outcome
opposite['pnl'] = -opposite['pnl']
opposite['is_counterfactual'] = True
opposite['counterfactual_type'] = 'OPPOSITE_POSITION'

# Adjust for direction-encoded feature
opposite['direction_encoded'] = (opposite['direction'] == 'BUY').astype(int)

print(f"   Generated {len(opposite)} opposite position samples")

# ============================================================================
# Counterfactual 2: Slippage Simulation
# ============================================================================
print("\n2. Generating slippage counterfactuals...")

slippage_scenarios = []
for slippage in [-0.005, -0.002, 0.002, 0.005]:  # -0.5% to +0.5%
    slipped = df.copy()
    
    # Adjust entry price for slippage
    slipped['price'] = slipped['price'] * (1 + slippage)
    
    # Adjust PnL for slippage impact
    # For BUY: higher entry price = lower PnL
    # For SELL: higher entry price = higher PnL
    slippage_impact = slippage * 100  # Convert to percentage
    
    slipped['pnl_percent'] = slipped.apply(
        lambda row: row['pnl_percent'] - slippage_impact if row['direction'] == 'BUY' 
                    else row['pnl_percent'] + slippage_impact,
        axis=1
    )
    
    slipped['pnl'] = slipped['pnl_percent'] * slipped['cost_usd'] / 100
    slipped['is_counterfactual'] = True
    slipped['counterfactual_type'] = f'SLIPPAGE_{slippage*100:+.1f}%'
    
    slippage_scenarios.append(slipped)

slippage_df = pd.concat(slippage_scenarios, ignore_index=True)
print(f"   Generated {len(slippage_df)} slippage samples (4 scenarios)")

# ============================================================================
# Counterfactual 3: Time Perturbation
# ============================================================================
print("\n3. Generating time perturbation counterfactuals...")

time_shifted = []
for hour_shift in [-1, 1]:
    shifted = df.copy()
    shifted['hour'] = (shifted['hour'] + hour_shift) % 24
    shifted['is_counterfactual'] = True
    shifted['counterfactual_type'] = f'HOUR_SHIFT_{hour_shift:+d}'
    time_shifted.append(shifted)

time_df = pd.concat(time_shifted, ignore_index=True)
print(f"   Generated {len(time_df)} time-shifted samples")

# ============================================================================
# Combine All Data
# ============================================================================
print("\n4. Combining all datasets...")

# Mark original data
df['is_counterfactual'] = False
df['counterfactual_type'] = 'ORIGINAL'

# Combine
all_data = pd.concat([df, opposite, slippage_df, time_df], ignore_index=True)

print(f"\nDataset Statistics:")
print(f"  Original trades: {len(df)}")
print(f"  Opposite positions: {len(opposite)}")
print(f"  Slippage scenarios: {len(slippage_df)}")
print(f"  Time shifts: {len(time_df)}")
print(f"  TOTAL: {len(all_data)}")
print(f"  Augmentation ratio: {len(all_data) / len(df):.1f}x")

# ============================================================================
# Save Augmented Dataset
# ============================================================================
print("\n5. Saving augmented dataset...")

output_path = os.path.join(OUTPUT_DIR, 'db_trades_augmented.parquet')
all_data.to_parquet(output_path, index=False)
print(f"Saved to {output_path}")

# Also save ML-ready version
ml_cols = [c for c in all_data.columns if c in df.columns]
ml_path = os.path.join(OUTPUT_DIR, 'db_trades_augmented_ml.parquet')
all_data[ml_cols].to_parquet(ml_path, index=False)
print(f"Saved ML-ready version to {ml_path}")

# ============================================================================
# Analysis
# ============================================================================
print("\n" + "=" * 70)
print("AUGMENTATION ANALYSIS")
print("=" * 70)

# Win rate comparison
print("\nWin Rate by Counterfactual Type:")
for cf_type in all_data['counterfactual_type'].unique():
    subset = all_data[all_data['counterfactual_type'] == cf_type]
    win_rate = (subset['pnl_percent'] > 0).mean()
    print(f"  {cf_type}: {len(subset)} samples, {win_rate:.1%} win rate")

# PnL distribution
print("\nPnL Distribution:")
print(f"  Original - Mean: {df['pnl_percent'].mean():.2f}%, Std: {df['pnl_percent'].std():.2f}%")
print(f"  Augmented - Mean: {all_data['pnl_percent'].mean():.2f}%, Std: {all_data['pnl_percent'].std():.2f}%")

# Feature distribution comparison
print("\nFeature Distribution Comparison (Original vs Augmented):")
for feature in ['price', 'quantity', 'cost_usd', 'hour']:
    if feature in all_data.columns:
        orig_mean = df[feature].mean()
        aug_mean = all_data[feature].mean()
        diff_pct = (aug_mean - orig_mean) / orig_mean * 100 if orig_mean != 0 else 0
        print(f"  {feature}: Original={orig_mean:.2f}, Augmented={aug_mean:.2f} ({diff_pct:+.1f}%)")

# Save summary
summary = {
    'original_count': len(df),
    'augmented_count': len(all_data),
    'augmentation_ratio': len(all_data) / len(df),
    'by_type': {
        'original': len(df),
        'opposite_position': len(opposite),
        'slippage': len(slippage_df),
        'time_shift': len(time_df),
    },
    'win_rates': {
        cf_type: float((all_data[all_data['counterfactual_type'] == cf_type]['pnl_percent'] > 0).mean())
        for cf_type in all_data['counterfactual_type'].unique()
    },
    'timestamp': pd.Timestamp.now().isoformat(),
}

import json
summary_path = os.path.join(OUTPUT_DIR, 'counterfactual_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved summary to {summary_path}")

print("\n" + "=" * 70)
print("COUNTERFACTUAL GENERATION COMPLETE")
print("=" * 70)

print("""
NEXT STEPS:

1. Train on augmented dataset:
   python scripts/train_on_augmented.py

2. Compare with original model:
   python scripts/compare_models.py

3. Validate calibration:
   python scripts/validate_calibration.py

EXPECTED IMPROVEMENTS:
- Better slippage robustness
- More realistic win rate estimates
- Reduced selection bias
- Improved model calibration
""")
