"""
Train a LightGBM on refined features and save model to `models/lgbm_return_rich.pkl`.
Uses engineered features from feature_engineering.py pipeline.
"""
import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

BASE = os.path.join(os.path.dirname(__file__), '..')
# Prefer engineered v2 features, fallback to rich_features
FEAT_V2 = os.path.join(BASE, 'datasets', 'engineered_features_v2.parquet')
FEAT = os.path.join(BASE, 'datasets', 'rich_features.parquet')
FEAT_PATH = FEAT_V2 if os.path.exists(FEAT_V2) else FEAT
OUT = os.path.join(BASE, 'models', 'lgbm_return_rich.pkl')
LOG = os.path.join(BASE, 'logs', 'train_rich.json')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
os.makedirs(os.path.dirname(LOG), exist_ok=True)

if not os.path.exists(FEAT_PATH):
    print('Missing features:', FEAT_PATH)
    raise SystemExit(1)

print(f"Loading features from: {FEAT_PATH}")
df = pd.read_parquet(FEAT_PATH)

# Ensure price_next and target exist
if 'price_next' not in df.columns:
    df = df.sort_values(['symbol','timestamp']).reset_index(drop=True)
    df['price_next'] = df.groupby('symbol')['price'].shift(-1)

df = df.dropna(subset=['price_next'])

if 'target' not in df.columns:
    df['target'] = np.log(df['price_next'] / df['price'])

# Refined feature set from feature engineering pipeline
FEATURES = [
    'qty', 'price', 'ret', 'rv_10', 'rv_10_ann', 'atr', 'vol_spike',
    'hour', 'day_of_week', 'is_weekend',
    'ret_roll3', 'ret_std3', 'price_ma3', 'price_std3', 'price_ma5',
    'ret_lag1', 'ret_lag2', 'vol_lag1',
    'price_momentum', 'vol_regime'
]
# Filter to available features
FEATURES = [f for f in FEATURES if f in df.columns]
print(f"Using {len(FEATURES)} features: {FEATURES}")

for c in FEATURES:
    if c not in df.columns:
        df[c] = 0.0

X = df[FEATURES].replace([np.inf, -np.inf], 0).fillna(0)
y = df['target'].astype(float)

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
    'feature_fraction': 0.8,
}

bst = lgb.train(
    params,
    train_data,
    num_boost_round=500,  # More rounds
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(False)]  # Longer patience
)

preds = bst.predict(X_test, num_iteration=bst.best_iteration)
mse = mean_squared_error(y_test, preds)
joblib.dump(bst, OUT)

with open(LOG, 'w') as fh:
    import json
    json.dump({'model_path':OUT,'mse':float(mse),'n_train':len(X_train),'n_test':len(X_test),'best_iter':int(bst.best_iteration)}, fh, indent=2)

print('Trained rich model saved to', OUT)
print('MSE:', mse)
