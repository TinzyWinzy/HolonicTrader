"""
Train directional models for BUY/SELL signal prediction.
Uses real trade data to predict entry direction and timing.
"""
import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

BASE = os.path.join(os.path.dirname(__file__), '..')
ML_FEATURES = os.path.join(BASE, 'datasets', 'ml_training_features_v2.parquet')
COMPLETE_TRADES = os.path.join(BASE, 'datasets', 'complete_trades_v2.parquet')
MODEL_DIR = os.path.join(BASE, 'models')
LOG_DIR = os.path.join(BASE, 'logs')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("=" * 70)
print("DIRECTIONAL MODEL TRAINING (BUY/SELL PREDICTION)")
print("=" * 70)

# Load complete trades for direction analysis
if not os.path.exists(COMPLETE_TRADES):
    print(f'Missing: {COMPLETE_TRADES}')
    print('Run: python scripts/create_ml_dataset.py first')
    raise SystemExit(1)

trades_df = pd.read_parquet(COMPLETE_TRADES)
print(f"Loaded {len(trades_df)} complete trades")

# Analyze direction patterns
print("\nTrade Direction Analysis:")
print(f"Symbols traded: {trades_df['symbol'].unique().tolist()}")

# Load ML features if available
if os.path.exists(ML_FEATURES):
    ml_df = pd.read_parquet(ML_FEATURES)
    print(f"Loaded {len(ml_df)} ML samples with features")
    
    # Create classification target: was this a good entry?
    # Good entry = trade eventually went in predicted direction
    ml_df['good_entry'] = (ml_df['trade_exit_pnl'] > -0.05).astype(int)  # Loss < 5% = acceptable
    
    print(f"\nTarget distribution (good_entry):")
    print(ml_df['good_entry'].value_counts())
    
    # Features for directional model
    FEATURES = ['qty', 'price', 'ret', 'rv_10', 'rv_10_ann', 'atr', 'vol_spike', 
                'trade_entry_qty', 'trade_entry_leverage']
    FEATURES = [f for f in FEATURES if f in ml_df.columns]
    
    X = ml_df[FEATURES].fillna(0).replace([np.inf, -np.inf], 0)
    y = ml_df['good_entry']
    
    print(f"\nUsing {len(FEATURES)} features: {FEATURES}")
    
    # Cross-validation for classification
    n_samples = len(ml_df)
    n_splits = min(3, n_samples)
    
    print(f"\nUsing {n_splits}-fold cross-validation")
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': 8,
        'learning_rate': 0.1,
        'min_data_in_leaf': 2,
        'feature_fraction': 0.8,
    }
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        bst = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(False)]
        )
        
        pred = (bst.predict(X_val, num_iteration=bst.best_iteration) > 0.5).astype(int)
        acc = accuracy_score(y_val, pred)
        cv_scores.append(acc)
        print(f"Fold {fold+1}: Accuracy = {acc:.2%}")
    
    print(f"\nDirectional Model CV Accuracy: {np.mean(cv_scores):.2%} (+/- {np.std(cv_scores):.2%})")
    
    # Train final model
    train_data = lgb.Dataset(X, label=y)
    bst_directional = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        callbacks=[lgb.log_evaluation(False)]
    )
    
    # Save model
    model_path = os.path.join(MODEL_DIR, 'lgbm_directional.pkl')
    joblib.dump(bst_directional, model_path)
    print(f"Saved directional model to {model_path}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': FEATURES,
        'importance': bst_directional.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    for _, row in importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.2f}")
    
    # Save training report
    report = {
        'n_samples': n_samples,
        'n_features': len(FEATURES),
        'features': FEATURES,
        'cv_accuracy': float(np.mean(cv_scores)),
        'cv_std': float(np.std(cv_scores)),
        'target_distribution': {
            'good_entry': int((y == 1).sum()),
            'bad_entry': int((y == 0).sum()),
        },
        'timestamp': pd.Timestamp.now().isoformat(),
    }
    
    log_path = os.path.join(LOG_DIR, 'train_directional.json')
    with open(log_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved training report to {log_path}")

else:
    print(f"Missing: {ML_FEATURES}")
    print("Cannot train directional model without features")

print("\n" + "=" * 70)
print("DIRECTIONAL TRAINING COMPLETE")
print("=" * 70)
