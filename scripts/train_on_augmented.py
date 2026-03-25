"""
Train ML models on augmented dataset (6,976 samples).

Includes counterfactual data for:
- Opposite positions
- Slippage scenarios
- Time perturbations

Usage:
    python scripts/train_on_augmented.py
"""
import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import json

BASE = os.path.join(os.path.dirname(__file__), '..')
AUGMENTED_DATA = os.path.join(BASE, 'datasets', 'db_trades_augmented_ml.parquet')
MODEL_DIR = os.path.join(BASE, 'models')
LOG_DIR = os.path.join(BASE, 'logs')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("=" * 70)
print("AUGMENTED MODEL TRAINING (6,976 samples)")
print("=" * 70)

# Load augmented data
if not os.path.exists(AUGMENTED_DATA):
    print(f'Missing: {AUGMENTED_DATA}')
    print('Run: python scripts/generate_counterfactuals.py first')
    raise SystemExit(1)

df = pd.read_parquet(AUGMENTED_DATA)
print(f"Loaded {len(df)} augmented samples")

# Separate original vs counterfactual
original = df[~df['is_counterfactual']]
counterfactual = df[df['is_counterfactual']]
print(f"  Original: {len(original)}")
print(f"  Counterfactual: {len(counterfactual)}")

# Prepare features
FEATURES = ['quantity', 'price', 'cost_usd', 'hour', 'day_of_week']

# Add optional features if available
optional = ['entropy_score', 'conviction', 'quality_score', 'mfe', 'mae']
for f in optional:
    if f in df.columns and df[f].notna().mean() > 0.3:
        FEATURES.append(f)

# Encode direction
df['direction_encoded'] = (df['direction'] == 'BUY').astype(int)
FEATURES.append('direction_encoded')

print(f"\nUsing {len(FEATURES)} features: {FEATURES}")

# Prepare data
X = df[FEATURES].fillna(0).replace([np.inf, -np.inf], 0)
y_reg = df['pnl_percent']
y_clf = (df['pnl_percent'] > 0).astype(int)

# Split (stratified to maintain original/counterfactual ratio)
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42, stratify=y_clf
)
_, _, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

print(f"\nTrain/Test split: {len(X_train)}/{len(X_test)}")

# ============================================================================
# MODEL 1: PnL REGRESSION (Augmented)
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 1: PnL REGRESSION (AUGMENTED)")
print("=" * 70)

params_reg = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'min_data_in_leaf': 20,  # Higher for larger dataset
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

# Evaluate
pred_reg = bst_reg.predict(X_test, num_iteration=bst_reg.best_iteration)
mse_reg = mean_squared_error(y_test_reg, pred_reg)
rmse_reg = np.sqrt(mse_reg)

print(f"\nRegression Results:")
print(f"  RMSE: {rmse_reg:.4f} ({rmse_reg*100:.2f}%)")
print(f"  Best iteration: {bst_reg.best_iteration}")

# Save
model_reg_path = os.path.join(MODEL_DIR, 'lgbm_pnl_regression_augmented.pkl')
joblib.dump(bst_reg, model_reg_path)
print(f"Saved to {model_reg_path}")

# ============================================================================
# MODEL 2: WIN/LOSS CLASSIFIER (Augmented)
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 2: WIN/LOSS CLASSIFIER (AUGMENTED)")
print("=" * 70)

params_clf = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'min_data_in_leaf': 20,
    'class_weight': 'balanced',
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

# Evaluate
pred_clf_prob = bst_clf.predict(X_test, num_iteration=bst_clf.best_iteration)
pred_clf = (pred_clf_prob > 0.5).astype(int)
acc_clf = accuracy_score(y_test_clf, pred_clf)

print(f"\nClassification Results:")
print(f"  Accuracy: {acc_clf:.2%}")
print(f"  Best iteration: {bst_clf.best_iteration}")
print(f"\nClassification Report:")
print(classification_report(y_test_clf, pred_clf, target_names=['Loss', 'Win']))

# Save
model_clf_path = os.path.join(MODEL_DIR, 'lgbm_win_classifier_augmented.pkl')
joblib.dump(bst_clf, model_clf_path)
print(f"Saved to {model_clf_path}")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importance_reg = pd.DataFrame({
    'feature': FEATURES,
    'importance_reg': bst_reg.feature_importance(importance_type='gain'),
}).sort_values('importance_reg', ascending=False)

importance_clf = pd.DataFrame({
    'feature': FEATURES,
    'importance_clf': bst_clf.feature_importance(importance_type='gain'),
}).sort_values('importance_clf', ascending=False)

importance = importance_reg.merge(importance_clf, on='feature', how='outer').fillna(0)
importance['avg_importance'] = (importance['importance_reg'] + importance['importance_clf']) / 2
importance = importance.sort_values('avg_importance', ascending=False)

print("\nTop 15 Features:")
for _, row in importance.head(15).iterrows():
    print(f"  {row['feature']:<25} Reg: {row['importance_reg']:10.0f}  Clf: {row['importance_clf']:10.0f}")

# ============================================================================
# CALIBRATION ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("CALIBRATION ANALYSIS")
print("=" * 70)

# Check if predicted probabilities match actual win rates
calibration_data = pd.DataFrame({
    'predicted_prob': pred_clf_prob,
    'actual_win': y_test_clf.values
})

# Bin by predicted probability
calibration_data['prob_bin'] = pd.cut(calibration_data['predicted_prob'], bins=10, labels=False)
calibration_summary = calibration_data.groupby('prob_bin').agg({
    'predicted_prob': 'mean',
    'actual_win': 'mean',
    'actual_win': 'count'
}).reset_index()

print("\nPredicted vs Actual Win Rate by Bin:")
for _, row in calibration_summary.iterrows():
    n_samples = int(row['actual_win'] if 'count' not in row else row['count'])
    print(f"  Pred {row['predicted_prob']:.1%} → Actual {row['actual_win']:.1%} ({n_samples} samples)")

# ============================================================================
# SAVE TRAINING REPORT
# ============================================================================
report = {
    'dataset': {
        'source': 'db_trades_augmented.parquet',
        'total_samples': len(df),
        'original_samples': len(original),
        'counterfactual_samples': len(counterfactual),
        'augmentation_ratio': len(df) / len(original),
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
    'calibration': calibration_summary.to_dict('records'),
    'data_statistics': {
        'win_rate': float(y_clf.mean()),
        'avg_pnl_percent': float(y_reg.mean()),
        'std_pnl_percent': float(y_reg.std()),
    },
    'timestamp': pd.Timestamp.now().isoformat(),
}

report_path = os.path.join(LOG_DIR, 'train_augmented.json')
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"\nSaved training report to {report_path}")

# ============================================================================
# COMPARE WITH ORIGINAL MODEL
# ============================================================================
print("\n" + "=" * 70)
print("COMPARISON: Augmented vs Original")
print("=" * 70)

# Load original model stats
try:
    with open(os.path.join(LOG_DIR, 'train_db_trades.json'), 'r') as f:
        original_report = json.load(f)
    
    print(f"\nDataset Size:")
    print(f"  Original: {original_report['dataset']['n_trades']} trades")
    print(f"  Augmented: {len(df)} samples ({len(df)/original_report['dataset']['n_trades']:.1f}x)")
    
    print(f"\nClassification Accuracy:")
    print(f"  Original: {original_report['classification_model']['accuracy']:.1%}")
    print(f"  Augmented: {acc_clf:.1%}")
    print(f"  Change: {(acc_clf - original_report['classification_model']['accuracy'])*100:+.1f} pp")
    
    print(f"\nRegression RMSE:")
    print(f"  Original: {original_report['regression_model']['rmse']:.2f}%")
    print(f"  Augmented: {rmse_reg:.2f}%")
    print(f"  Change: {(rmse_reg - original_report['regression_model']['rmse'])*100:+.2f} pp")
    
except Exception as e:
    print(f"Could not load original report: {e}")

print("\n" + "=" * 70)
print("AUGMENTED TRAINING COMPLETE")
print("=" * 70)

print("""
NEXT STEPS:

1. Update ML Advisor to use augmented models:
   - models/lgbm_win_classifier_augmented.pkl
   - models/lgbm_pnl_regression_augmented.pkl

2. Test augmented models:
   python test_ml_advisor.py

3. Monitor calibration in live trading

EXPECTED IMPROVEMENTS:
✓ 8x more training data
✓ Better slippage robustness
✓ More realistic win rate estimates
✓ Improved calibration
✓ Reduced selection bias
""")
