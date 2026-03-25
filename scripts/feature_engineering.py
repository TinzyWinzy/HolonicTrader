"""
Feature Engineering Pipeline for HolonicTrader

Implements the following steps:
1. Audit and drop constant/zero features
2. Analyze feature distributions and correlations
3. Engineer new features (rolling stats, regime, time-based, lagged)
4. Feature importance analysis with LightGBM
5. Output refined feature set for training
"""
import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# --- Config ---
RICH_FEATURES = 'datasets/rich_features.parquet'
ENGINEERED_FEATURES = 'datasets/engineered_features.parquet'
OUTPUT_FEATURES = 'datasets/engineered_features_v2.parquet'
OUTPUT_REPORT = 'reports/feature_engineering_report.json'
PLOTS_DIR = 'plots/feature_analysis'

os.makedirs('reports', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Step 1: Load and Audit Data ---
print("=" * 70)
print("STEP 1: AUDIT AND DROP CONSTANT/ZERO FEATURES")
print("=" * 70)

# Prefer rich features if available
if os.path.exists(RICH_FEATURES):
    df = pd.read_parquet(RICH_FEATURES)
    print(f"Loaded rich_features.parquet: {df.shape}")
else:
    df = pd.read_parquet(ENGINEERED_FEATURES)
    print(f"Loaded engineered_features.parquet: {df.shape}")

# Identify columns to drop
constant_cols = []
zero_cols = []
high_null_cols = []

for col in df.columns:
    if df[col].dtype in ['object', 'string']:
        continue  # Skip non-numeric for now
    if df[col].nunique() == 1:
        constant_cols.append(col)
    if (df[col] == 0).all():
        zero_cols.append(col)
    if df[col].isnull().mean() > 0.9:
        high_null_cols.append(col)

print(f"\nConstant columns (1 unique value): {constant_cols}")
print(f"All-zero columns: {zero_cols}")
print(f"High-null columns (>90%): {high_null_cols}")

# Drop these columns
cols_to_drop = list(set(constant_cols + zero_cols + high_null_cols))
# Also drop non-informative columns
cols_to_drop.extend(['session_file', 'order_id', 'raw', 'event'])
cols_to_drop = list(set(cols_to_drop))
cols_to_drop = [c for c in cols_to_drop if c in df.columns]

print(f"\nDropping {len(cols_to_drop)} columns: {cols_to_drop}")
df_clean = df.drop(columns=cols_to_drop).copy()
print(f"Remaining columns: {df_clean.columns.tolist()}")

# --- Step 2: Feature Distributions and Correlations ---
print("\n" + "=" * 70)
print("STEP 2: FEATURE DISTRIBUTIONS AND CORRELATIONS")
print("=" * 70)

# Create price_next target
df_clean = df_clean.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
df_clean['price_next'] = df_clean.groupby('symbol')['price'].shift(-1)
df_clean = df_clean.dropna(subset=['price_next'])

# Target: log return
df_clean['target'] = np.log(df_clean['price_next'] / df_clean['price'])

# Compute correlations with target
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
correlations = {}
print("\nCorrelation with target (price log-return):")
for col in numeric_cols:
    if col not in ['price_next', 'target'] and df_clean[col].nunique() > 1:
        corr = df_clean['target'].corr(df_clean[col])
        correlations[col] = corr if not np.isnan(corr) else 0.0
        print(f"  {col}: {corr:.4f}")

# Sort by absolute correlation
correlations_sorted = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
print("\nFeatures ranked by |correlation| with target:")
for feat, corr in correlations_sorted:
    print(f"  {feat}: {corr:.4f}")

# Save correlation plot
if len(correlations) > 0:
    plt.figure(figsize=(10, 6))
    feats = [f[0] for f in correlations_sorted]
    corrs = [f[1] for f in correlations_sorted]
    plt.barh(feats, corrs)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=0.8)
    plt.xlabel('Correlation with Target')
    plt.title('Feature Correlations with Price Log-Return')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_correlations.png'), dpi=150)
    plt.close()
    print(f"\nSaved correlation plot to {PLOTS_DIR}/feature_correlations.png")

# --- Step 3: Engineer New Features ---
print("\n" + "=" * 70)
print("STEP 3: ENGINEER NEW FEATURES")
print("=" * 70)

# Parse timestamp for time-based features
df_clean['timestamp_dt'] = pd.to_datetime(df_clean['timestamp'])
df_clean['hour'] = df_clean['timestamp_dt'].dt.hour
df_clean['day_of_week'] = df_clean['timestamp_dt'].dt.dayofweek
df_clean['is_weekend'] = (df_clean['day_of_week'] >= 5).astype(int)

# Rolling statistics per symbol (need to sort first)
df_clean = df_clean.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

# Rolling returns and volatility
for symbol in df_clean['symbol'].unique():
    mask = df_clean['symbol'] == symbol
    idx = df_clean.loc[mask].index
    
    # Rolling mean and std of returns (if ret exists)
    if 'ret' in df_clean.columns:
        df_clean.loc[idx, 'ret_roll3'] = df_clean.loc[idx, 'ret'].rolling(3, min_periods=1).mean()
        df_clean.loc[idx, 'ret_std3'] = df_clean.loc[idx, 'ret'].rolling(3, min_periods=1).std()
    
    # Rolling price stats
    df_clean.loc[idx, 'price_ma3'] = df_clean.loc[idx, 'price'].rolling(3, min_periods=1).mean()
    df_clean.loc[idx, 'price_std3'] = df_clean.loc[idx, 'price'].rolling(3, min_periods=1).std()
    df_clean.loc[idx, 'price_ma5'] = df_clean.loc[idx, 'price'].rolling(5, min_periods=1).mean()
    
    # Lagged features
    df_clean.loc[idx, 'ret_lag1'] = df_clean.loc[idx, 'ret'].shift(1)
    df_clean.loc[idx, 'ret_lag2'] = df_clean.loc[idx, 'ret'].shift(2)
    df_clean.loc[idx, 'vol_lag1'] = df_clean.loc[idx, 'rv_10_ann'].shift(1) if 'rv_10_ann' in df_clean.columns else 0

# Price momentum
df_clean['price_momentum'] = df_clean['price'] / df_clean.groupby('symbol')['price'].shift(1) - 1

# Volatility regime
if 'rv_10_ann' in df_clean.columns:
    df_clean['vol_regime'] = (df_clean['rv_10_ann'] > df_clean['rv_10_ann'].rolling(10, min_periods=1).mean()).astype(int)

# Fill NaN from rolling/lag operations
df_clean = df_clean.fillna(0)

# New engineered features list
NEW_FEATURES = [
    'hour', 'day_of_week', 'is_weekend',
    'ret_roll3', 'ret_std3', 'price_ma3', 'price_std3', 'price_ma5',
    'ret_lag1', 'ret_lag2', 'vol_lag1',
    'price_momentum', 'vol_regime'
]

# Add existing informative features
BASE_FEATURES = ['qty', 'price', 'ret', 'rv_10', 'rv_10_ann', 'atr', 'vol_spike']
BASE_FEATURES = [f for f in BASE_FEATURES if f in df_clean.columns]

ALL_FEATURES = BASE_FEATURES + NEW_FEATURES
ALL_FEATURES = [f for f in ALL_FEATURES if f in df_clean.columns]

print(f"\nEngineered {len(NEW_FEATURES)} new features")
print(f"Total features: {len(ALL_FEATURES)}")
print(f"Feature list: {ALL_FEATURES}")

# --- Step 4: Feature Importance with LightGBM ---
print("\n" + "=" * 70)
print("STEP 4: FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

X = df_clean[ALL_FEATURES].replace([np.inf, -np.inf], 0).fillna(0)
y = df_clean['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
}

bst = lgb.train(
    params,
    train_data,
    num_boost_round=200,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(False)]
)

# Get feature importance
importance = bst.feature_importance(importance_type='gain')
feature_importance = pd.DataFrame({
    'feature': ALL_FEATURES,
    'importance': importance
}).sort_values('importance', ascending=False)

print("\nFeature Importance (Gain):")
for _, row in feature_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.2f}")

# Save importance plot
plt.figure(figsize=(10, 8))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.gca().invert_yaxis()
plt.xlabel('Importance (Gain)')
plt.title('LightGBM Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'), dpi=150)
plt.close()
print(f"\nSaved importance plot to {PLOTS_DIR}/feature_importance.png")

# Select top features (cumulative importance > 95%)
total_imp = feature_importance['importance'].sum()
feature_importance['cumulative_pct'] = feature_importance['importance'].cumsum() / total_imp * 100
top_features = feature_importance[feature_importance['cumulative_pct'] <= 95]['feature'].tolist()
# Always include price and qty if they exist
for f in ['price', 'qty', 'ret']:
    if f in df_clean.columns and f not in top_features:
        top_features.append(f)

print(f"\nTop features (95% cumulative importance): {top_features}")
print(f"Reduced from {len(ALL_FEATURES)} to {len(top_features)} features")

# --- Step 5: Save Refined Features ---
print("\n" + "=" * 70)
print("STEP 5: SAVE REFINED FEATURE SET")
print("=" * 70)

# Select final columns (avoid duplicates)
base_cols = ['symbol', 'timestamp', 'price', 'price_next', 'target']
base_cols = [c for c in base_cols if c in df_clean.columns]
feature_cols = [c for c in top_features if c not in base_cols]
final_cols = base_cols + feature_cols

df_final = df_clean[final_cols].copy()
df_final.to_parquet(OUTPUT_FEATURES, index=False)
print(f"\nSaved refined features to {OUTPUT_FEATURES}")
print(f"Shape: {df_final.shape}")
print(f"Columns: {df_final.columns.tolist()}")

# --- Save Report ---
report = {
    'audit': {
        'original_shape': list(df.shape),
        'cleaned_shape': list(df_clean.shape),
        'final_shape': list(df_final.shape),
        'dropped_columns': cols_to_drop,
    },
    'correlations': correlations_sorted,
    'feature_importance': feature_importance.to_dict('records'),
    'selected_features': top_features,
    'new_features_added': NEW_FEATURES,
    'model_metrics': {
        'train_mse': float(mean_squared_error(y_train, bst.predict(X_train))),
        'test_mse': float(mean_squared_error(y_test, bst.predict(X_test))),
    }
}

with open(OUTPUT_REPORT, 'w') as f:
    json.dump(report, f, indent=2, default=str)
print(f"\nSaved feature engineering report to {OUTPUT_REPORT}")

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)
print(f"\nNext steps:")
print(f"1. Update training scripts to use features: {top_features}")
print(f"2. Retrain models with python scripts/train_save_rich.py")
print(f"3. Run backtest with python scripts/ab_backtest.py")
