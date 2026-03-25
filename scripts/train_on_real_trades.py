"""
Train LightGBM models on real trade data (ml_training_features_v2.parquet).
Creates both baseline and rich models.
"""
import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, accuracy_score
import json

BASE = os.path.join(os.path.dirname(__file__), '..')
ML_FEATURES = os.path.join(BASE, 'datasets', 'ml_training_features_v2.parquet')
MODEL_DIR = os.path.join(BASE, 'models')
LOG_DIR = os.path.join(BASE, 'logs')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("=" * 70)
print("MODEL TRAINING ON REAL TRADE DATA")
print("=" * 70)

# Load ML training features
if not os.path.exists(ML_FEATURES):
    print(f'Missing features: {ML_FEATURES}')
    print('Run: python scripts/create_ml_dataset.py first')
    raise SystemExit(1)

df = pd.read_parquet(ML_FEATURES)
print(f"Loaded {len(df)} training samples from ml_training_features_v2.parquet")

if len(df) < 5:
    print(f"\n⚠️  WARNING: Only {len(df)} samples - model will be unreliable")
    print("Consider collecting more trading data before training")

# Define features
BASE_FEATURES = ['qty', 'price', 'ret', 'rv_10', 'rv_10_ann', 'atr', 'vol_spike']
ALL_FEATURES = BASE_FEATURES + ['hour', 'day_of_week', 'ret_roll3', 'ret_std3', 'price_ma3']

# Filter to available features
BASE_FEATURES = [f for f in BASE_FEATURES if f in df.columns]
ALL_FEATURES = [f for f in ALL_FEATURES if f in df.columns]

print(f"\nUsing {len(BASE_FEATURES)} base features: {BASE_FEATURES}")
print(f"Using {len(ALL_FEATURES)} rich features: {ALL_FEATURES}")

# Target: trade_exit_pnl (regression) or trade_is_win (classification)
TARGET_REG = 'trade_exit_pnl'
TARGET_CLF = 'trade_is_win'

# Prepare data
X_base = df[BASE_FEATURES].fillna(0).replace([np.inf, -np.inf], 0)
X_rich = df[ALL_FEATURES].fillna(0).replace([np.inf, -np.inf], 0)
y_reg = df[TARGET_REG].astype(float)
y_clf = df[TARGET_CLF].astype(int)

print(f"\nTarget (regression) stats:")
print(y_reg.describe())
print(f"\nTarget (classification) distribution:")
print(y_clf.value_counts())

# For small datasets, use cross-validation
n_samples = len(df)
n_splits = min(3, n_samples)  # At least 3 folds or fewer if not enough samples

print(f"\nUsing {n_splits}-fold cross-validation (dataset has {n_samples} samples)")

# --- Train Baseline Model (Regression) ---
print("\n" + "=" * 70)
print("TRAINING BASELINE MODEL (Regression)")
print("=" * 70)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'num_leaves': 8,  # Small for limited data
    'learning_rate': 0.1,
    'min_data_in_leaf': 2,  # Minimum for small data
    'feature_fraction': 0.8,
}

# Cross-validation for baseline
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_base)):
    X_train, X_val = X_base.iloc[train_idx], X_base.iloc[val_idx]
    y_train, y_val = y_reg.iloc[train_idx], y_reg.iloc[val_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    bst = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(False)]
    )
    
    pred = bst.predict(X_val, num_iteration=bst.best_iteration)
    score = mean_squared_error(y_val, pred)
    cv_scores.append(score)
    print(f"Fold {fold+1}: RMSE = {np.sqrt(score):.6f}")

print(f"\nBaseline CV RMSE: {np.sqrt(np.mean(cv_scores)):.6f} (+/- {np.std(cv_scores):.6f})")

# Train final baseline model on all data
train_data = lgb.Dataset(X_base, label=y_reg)
bst_base = lgb.train(
    params,
    train_data,
    num_boost_round=200,
    valid_sets=[train_data],
    callbacks=[lgb.log_evaluation(False)]
)

# Save baseline model
model_base_path = os.path.join(MODEL_DIR, 'lgbm_return_v1.pkl')
joblib.dump(bst_base, model_base_path)
print(f"Saved baseline model to {model_base_path}")

# --- Train Rich Model ---
print("\n" + "=" * 70)
print("TRAINING RICH MODEL (Regression)")
print("=" * 70)

params_rich = params.copy()
params_rich['num_leaves'] = 12  # Slightly more complex

# Cross-validation for rich model
cv_scores_rich = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_rich)):
    X_train, X_val = X_rich.iloc[train_idx], X_rich.iloc[val_idx]
    y_train, y_val = y_reg.iloc[train_idx], y_reg.iloc[val_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    bst = lgb.train(
        params_rich,
        train_data,
        num_boost_round=200,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(False)]
    )
    
    pred = bst.predict(X_val, num_iteration=bst.best_iteration)
    score = mean_squared_error(y_val, pred)
    cv_scores_rich.append(score)
    print(f"Fold {fold+1}: RMSE = {np.sqrt(score):.6f}")

print(f"\nRich Model CV RMSE: {np.sqrt(np.mean(cv_scores_rich)):.6f} (+/- {np.std(cv_scores_rich):.6f})")

# Train final rich model on all data
train_data = lgb.Dataset(X_rich, label=y_reg)
bst_rich = lgb.train(
    params_rich,
    train_data,
    num_boost_round=200,
    valid_sets=[train_data],
    callbacks=[lgb.log_evaluation(False)]
)

# Save rich model
model_rich_path = os.path.join(MODEL_DIR, 'lgbm_return_rich.pkl')
joblib.dump(bst_rich, model_rich_path)
print(f"Saved rich model to {model_rich_path}")

# --- Feature Importance ---
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importance_base = pd.DataFrame({
    'feature': BASE_FEATURES,
    'importance': bst_base.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print("\nBaseline Model:")
for _, row in importance_base.iterrows():
    print(f"  {row['feature']}: {row['importance']:.2f}")

importance_rich = pd.DataFrame({
    'feature': ALL_FEATURES,
    'importance': bst_rich.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print("\nRich Model:")
for _, row in importance_rich.iterrows():
    print(f"  {row['feature']}: {row['importance']:.2f}")

# --- Save Training Report ---
report = {
    'n_samples': n_samples,
    'n_base_features': len(BASE_FEATURES),
    'n_rich_features': len(ALL_FEATURES),
    'base_features': BASE_FEATURES,
    'rich_features': ALL_FEATURES,
    'baseline_cv_rmse': float(np.sqrt(np.mean(cv_scores))),
    'baseline_cv_std': float(np.sqrt(np.std(cv_scores))),
    'rich_cv_rmse': float(np.sqrt(np.mean(cv_scores_rich))),
    'rich_cv_std': float(np.sqrt(np.std(cv_scores_rich))),
    'target_mean': float(y_reg.mean()),
    'target_std': float(y_reg.std()),
    'win_rate': float(y_clf.mean()),
    'timestamp': pd.Timestamp.now().isoformat(),
}

log_path = os.path.join(LOG_DIR, 'train_trade_data.json')
with open(log_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\nSaved training report to {log_path}")
print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
