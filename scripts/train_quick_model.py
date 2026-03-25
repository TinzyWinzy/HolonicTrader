"""
Train a simple LightGBM regressor on `datasets/engineered_features.parquet`.
Saves model to `models/lgbm_return_v1.pkl` and logs to `logs/train_quick.json`.
Updated with refined feature set and improved target calculation.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

BASE = os.path.join(os.path.dirname(__file__), '..')
FEAT_PARQ = os.path.join(BASE, 'datasets', 'engineered_features.parquet')
MODEL_DIR = os.path.join(BASE, 'models')
LOG_DIR = os.path.join(BASE, 'logs')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

if not os.path.exists(FEAT_PARQ):
    print('Feature file missing:', FEAT_PARQ)
    raise SystemExit(1)

df = pd.read_parquet(FEAT_PARQ)
# Simple label: next-event price change for same symbol
df = df.sort_values(['symbol','timestamp'])

df['price_next'] = df.groupby('symbol')['price'].shift(-1)
# target: log return (more stable for ML)
df['target'] = np.log(df['price_next'] / df['price']).replace([np.inf, -np.inf], np.nan)

# Drop rows without target
df = df.dropna(subset=['target'])

# Refined features: drop constant/zero features (event_id, fee_usd)
# Use informative features only
FEATURES = ['qty', 'price', 'price_change', 'price_roll3']
for c in FEATURES:
    if c not in df.columns:
        df[c] = 0.0

X = df[FEATURES].fillna(0)
y = df['target'].astype(float)

# small train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'num_leaves': 15,  # Reduced for small dataset
    'learning_rate': 0.1,  # Higher for small data
    'min_data_in_leaf': 5,  # Lower for small dataset
}

# LightGBM newer versions use callbacks for early stopping
bst = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(False)]
)

preds = bst.predict(X_test, num_iteration=bst.best_iteration)
mse = mean_squared_error(y_test, preds)

# Save model and logs
model_path = os.path.join(MODEL_DIR, 'lgbm_return_v1.pkl')
joblib.dump(bst, model_path)

log = {
    'model_path': model_path,
    'features': FEATURES,
    'n_train': len(X_train),
    'n_test': len(X_test),
    'mse': float(mse),
    'best_iteration': int(bst.best_iteration)
}
with open(os.path.join(LOG_DIR, 'train_quick.json'), 'w') as fh:
    json.dump(log, fh, indent=2)

# sample preds saved
sample_out = df.loc[X_test.index].copy()
sample_out = sample_out.reset_index(drop=True)
sample_out['pred'] = preds
sample_csv = os.path.join(BASE, 'reports', 'sample_preds.csv')
os.makedirs(os.path.dirname(sample_csv), exist_ok=True)
sample_out.to_csv(sample_csv, index=False)

print('Training complete. MSE:', mse)
print('Model saved to', model_path)
