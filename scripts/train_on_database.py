"""
Train ML models on historical database trades (872 samples).
Much larger dataset than log-extracted trades (9 samples).

Trains:
1. PnL regression model (predict trade outcome)
2. Directional classifier (predict win/loss)
3. Feature importance analysis
"""
import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import json

BASE = os.path.join(os.path.dirname(__file__), '..')
DB_TRADES = os.path.join(BASE, 'datasets', 'db_trades_ml.parquet')
MODEL_DIR = os.path.join(BASE, 'models')
LOG_DIR = os.path.join(BASE, 'logs')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("=" * 70)
print("DATABASE-BASED MODEL TRAINING (872 trades)")
print("=" * 70)

# Load database trades
if not os.path.exists(DB_TRADES):
    print(f'Missing: {DB_TRADES}')
    print('Run: python scripts/export_db_trades.py first')
    raise SystemExit(1)

df = pd.read_parquet(DB_TRADES)
print(f"Loaded {len(df)} trades from database")

# Data quality check
print(f"\nData quality:")
print(f"  Null pnl_percent: {df['pnl_percent'].isnull().sum()}")
print(f"  Null direction: {df['direction'].isnull().sum()}")
print(f"  Null regime: {df['regime'].notna().sum()} ({df['regime'].notna().mean():.1%})")
print(f"  Null conviction: {df['conviction'].notna().sum()} ({df['conviction'].notna().mean():.1%})")

# Prepare features
FEATURES = ['quantity', 'price', 'cost_usd', 'hour', 'day_of_week']

# Add optional features if available
optional_features = ['entropy_score', 'conviction', 'quality_score', 'mfe', 'mae']
for f in optional_features:
    if f in df.columns and df[f].notna().mean() > 0.3:  # At least 30% non-null
        FEATURES.append(f)

# Encode direction
df['direction_encoded'] = (df['direction'] == 'BUY').astype(int)
FEATURES.append('direction_encoded')

# Encode regime if available
if 'regime' in df.columns and df['regime'].notna().mean() > 0.3:
    regime_dummies = pd.get_dummies(df['regime'], prefix='regime', dummy_na=True)
    df = pd.concat([df, regime_dummies], axis=1)
    FEATURES.extend(regime_dummies.columns.tolist())

print(f"\nUsing {len(FEATURES)} features: {FEATURES}")

# Prepare data
X = df[FEATURES].fillna(0).replace([np.inf, -np.inf], 0)
y_reg = df['pnl_percent']  # Regression target
y_clf = (df['pnl_percent'] > 0).astype(int)  # Classification target (win/loss)
y_good = (df['pnl_percent'] > -0.02).astype(int)  # Good trade (loss < 2%)

print(f"\nTarget distributions:")
print(f"  pnl_percent: mean={y_reg.mean():.2f}, std={y_reg.std():.2f}")
print(f"  win (pnl>0): {y_clf.sum()} ({y_clf.mean():.1%})")
print(f"  good_trade (loss<2%): {y_good.sum()} ({y_good.mean():.1%})")

# Split data
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

print(f"\nTrain/Test split: {len(X_train)}/{len(X_test)}")

# ============================================================================
# MODEL 1: PnL REGRESSION
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 1: PnL REGRESSION")
print("=" * 70)

params_reg = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'min_data_in_leaf': 10,
}

train_data = lgb.Dataset(X_train, label=y_train_reg)
valid_data = lgb.Dataset(X_test, label=y_test_reg, reference=train_data)

bst_reg = lgb.train(
    params_reg,
    train_data,
    num_boost_round=500,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# Evaluate regression
pred_reg = bst_reg.predict(X_test, num_iteration=bst_reg.best_iteration)
mse_reg = mean_squared_error(y_test_reg, pred_reg)
rmse_reg = np.sqrt(mse_reg)

print(f"\nRegression Results:")
print(f"  RMSE: {rmse_reg:.4f} ({rmse_reg*100:.2f}%)")
print(f"  Best iteration: {bst_reg.best_iteration}")

# Save regression model
model_reg_path = os.path.join(MODEL_DIR, 'lgbm_pnl_regression.pkl')
joblib.dump(bst_reg, model_reg_path)
print(f"Saved to {model_reg_path}")

# ============================================================================
# MODEL 2: WIN/LOSS CLASSIFIER
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 2: WIN/LOSS CLASSIFIER")
print("=" * 70)

params_clf = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'min_data_in_leaf': 10,
    'class_weight': 'balanced',  # Handle any class imbalance
}

train_data_clf = lgb.Dataset(X_train, label=y_train_clf)
valid_data_clf = lgb.Dataset(X_test, label=y_test_clf, reference=train_data_clf)

bst_clf = lgb.train(
    params_clf,
    train_data_clf,
    num_boost_round=500,
    valid_sets=[valid_data_clf],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# Evaluate classification
pred_clf_prob = bst_clf.predict(X_test, num_iteration=bst_clf.best_iteration)
pred_clf = (pred_clf_prob > 0.5).astype(int)
acc_clf = accuracy_score(y_test_clf, pred_clf)

print(f"\nClassification Results:")
print(f"  Accuracy: {acc_clf:.2%}")
print(f"  Best iteration: {bst_clf.best_iteration}")
print(f"\nClassification Report:")
print(classification_report(y_test_clf, pred_clf, target_names=['Loss', 'Win']))

# Save classification model
model_clf_path = os.path.join(MODEL_DIR, 'lgbm_win_classifier.pkl')
joblib.dump(bst_clf, model_clf_path)
print(f"Saved to {model_clf_path}")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

# Regression importance
importance_reg = pd.DataFrame({
    'feature': FEATURES,
    'importance_reg': bst_reg.feature_importance(importance_type='gain'),
}).sort_values('importance_reg', ascending=False)

# Classification importance
importance_clf = pd.DataFrame({
    'feature': FEATURES,
    'importance_clf': bst_clf.feature_importance(importance_type='gain'),
}).sort_values('importance_clf', ascending=False)

# Merge
importance = importance_reg.merge(importance_clf, on='feature', how='outer').fillna(0)
importance['avg_importance'] = (importance['importance_reg'] + importance['importance_clf']) / 2
importance = importance.sort_values('avg_importance', ascending=False)

print("\nTop 15 Features:")
for _, row in importance.head(15).iterrows():
    print(f"  {row['feature']:<25} Reg: {row['importance_reg']:8.0f}  Clf: {row['importance_clf']:8.0f}")

# ============================================================================
# SAVE TRAINING REPORT
# ============================================================================
report = {
    'dataset': {
        'source': 'holonic_trader.db',
        'n_trades': len(df),
        'n_features': len(FEATURES),
        'features': FEATURES,
        'train_size': len(X_train),
        'test_size': len(X_test),
    },
    'regression_model': {
        'rmse': float(rmse_reg),
        'best_iteration': int(bst_reg.best_iteration),
        'top_features': importance.head(5)['feature'].tolist(),
    },
    'classification_model': {
        'accuracy': float(acc_clf),
        'best_iteration': int(bst_clf.best_iteration),
        'top_features': importance_clf.head(5)['feature'].tolist(),
    },
    'data_statistics': {
        'win_rate': float(y_clf.mean()),
        'avg_pnl_percent': float(y_reg.mean()),
        'std_pnl_percent': float(y_reg.std()),
    },
    'timestamp': pd.Timestamp.now().isoformat(),
}

report_path = os.path.join(LOG_DIR, 'train_db_trades.json')
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"\nSaved training report to {report_path}")

print("\n" + "=" * 70)
print("DATABASE MODEL TRAINING COMPLETE")
print("=" * 70)

# Usage example
print("""
USAGE EXAMPLE:

# Load models
import joblib
reg_model = joblib.load('models/lgbm_pnl_regression.pkl')
clf_model = joblib.load('models/lgbm_win_classifier.pkl')

# Predict
features = {
    'quantity': 10.0,
    'price': 100.0,
    'cost_usd': 1000.0,
    'hour': 14,
    'day_of_week': 3,
    'direction_encoded': 1,  # BUY
    # ... add other features
}
X = pd.DataFrame([features])[FEATURES]

pnl_pred = reg_model.predict(X)[0]
win_prob = clf_model.predict(X)[0]

print(f"Predicted PnL: {pnl_pred:.2f}%")
print(f"Win Probability: {win_prob:.1%}")
""")
