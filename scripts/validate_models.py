"""
Validate trained models on real trade data.
Tests prediction quality and simulates trading performance.
"""
import os
import joblib
import pandas as pd
import numpy as np
import json

BASE = os.path.join(os.path.dirname(__file__), '..')
ML_FEATURES = os.path.join(BASE, 'datasets', 'ml_training_features_v2.parquet')
MODEL_BASE = os.path.join(BASE, 'models', 'lgbm_return_v1.pkl')
MODEL_RICH = os.path.join(BASE, 'models', 'lgbm_return_rich.pkl')

print("=" * 70)
print("MODEL VALIDATION ON REAL TRADE DATA")
print("=" * 70)

# Load data
if not os.path.exists(ML_FEATURES):
    print(f'Missing: {ML_FEATURES}')
    raise SystemExit(1)

df = pd.read_parquet(ML_FEATURES)
print(f"Loaded {len(df)} samples")

# Load models
if not os.path.exists(MODEL_BASE):
    print(f'Missing: {MODEL_BASE}')
    raise SystemExit(1)

model_base = joblib.load(MODEL_BASE)
model_rich = joblib.load(MODEL_RICH)
print("Loaded both models")

# Features
BASE_FEATURES = ['qty', 'price', 'ret', 'rv_10', 'rv_10_ann', 'atr', 'vol_spike']
BASE_FEATURES = [f for f in BASE_FEATURES if f in df.columns]

# Generate predictions
X = df[BASE_FEATURES].fillna(0).replace([np.inf, -np.inf], 0)
y_true = df['trade_exit_pnl'].values

pred_base = model_base.predict(X)
pred_rich = model_rich.predict(X)

print("\n" + "=" * 70)
print("PREDICTION QUALITY")
print("=" * 70)

# Calculate metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse_base = mean_squared_error(y_true, pred_base)
mae_base = mean_absolute_error(y_true, pred_base)
r2_base = r2_score(y_true, pred_base)

mse_rich = mean_squared_error(y_true, pred_rich)
mae_rich = mean_absolute_error(y_true, pred_rich)
r2_rich = r2_score(y_true, pred_rich)

print(f"\nBaseline Model:")
print(f"  MSE: {mse_base:.6f}")
print(f"  MAE: {mae_base:.6f} ({mae_base*100:.2f}%)")
print(f"  R²:  {r2_base:.4f}")

print(f"\nRich Model:")
print(f"  MSE: {mse_rich:.6f}")
print(f"  MAE: {mae_rich:.6f} ({mae_rich*100:.2f}%)")
print(f"  R²:  {r2_rich:.4f}")

# Prediction distribution
print("\n" + "=" * 70)
print("PREDICTION DISTRIBUTION")
print("=" * 70)

print(f"\nTrue PnL:     mean={y_true.mean():.4f}, std={y_true.std():.4f}, min={y_true.min():.4f}, max={y_true.max():.4f}")
print(f"Pred Base:    mean={pred_base.mean():.4f}, std={pred_base.std():.4f}, min={pred_base.min():.4f}, max={pred_base.max():.4f}")
print(f"Pred Rich:    mean={pred_rich.mean():.4f}, std={pred_rich.std():.4f}, min={pred_rich.min():.4f}, max={pred_rich.max():.4f}")

# Sample predictions
print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

sample_df = pd.DataFrame({
    'symbol': df.get('symbol', ['N/A'] * len(y_true)),
    'true_pnl': y_true,
    'pred_base': pred_base,
    'pred_rich': pred_rich,
    'error_base': pred_base - y_true,
    'error_rich': pred_rich - y_true,
})

print(sample_df.to_string())

# Trading simulation
print("\n" + "=" * 70)
print("TRADING SIMULATION")
print("=" * 70)

# Simulate: Only trade when model predicts positive (win)
threshold = 0.0  # Predict win if > 0

# Baseline strategy
signals_base = pred_base > threshold
pnl_base = y_true[signals_base]
win_rate_base = (pnl_base > 0).mean() if len(pnl_base) > 0 else 0
total_pnl_base = pnl_base.sum() if len(pnl_base) > 0 else 0

# Rich strategy
signals_rich = pred_rich > threshold
pnl_rich = y_true[signals_rich]
win_rate_rich = (pnl_rich > 0).mean() if len(pnl_rich) > 0 else 0
total_pnl_rich = pnl_rich.sum() if len(pnl_rich) > 0 else 0

# Buy & hold (all trades)
win_rate_hold = (y_true > 0).mean()
total_pnl_hold = y_true.sum()

print(f"\nStrategy              Trades   Win Rate   Total PnL")
print(f"------------------------------------------------------")
print(f"Buy & Hold            {len(y_true):6d}   {win_rate_hold*100:7.1f}%   {total_pnl_hold:7.4f} ({total_pnl_hold*100:.2f}%)")
print(f"Baseline (pred>0)     {len(pnl_base):6d}   {win_rate_base*100:7.1f}%   {total_pnl_base:7.4f} ({total_pnl_base*100:.2f}%)")
print(f"Rich (pred>0)         {len(pnl_rich):6d}   {win_rate_rich*100:7.1f}%   {total_pnl_rich:7.4f} ({total_pnl_rich*100:.2f}%)")

# Save validation report
report = {
    'n_samples': len(df),
    'baseline': {
        'mse': float(mse_base),
        'mae': float(mae_base),
        'r2': float(r2_base),
    },
    'rich': {
        'mse': float(mse_rich),
        'mae': float(mae_rich),
        'r2': float(r2_rich),
    },
    'trading_simulation': {
        'buy_hold': {
            'trades': int(len(y_true)),
            'win_rate': float(win_rate_hold),
            'total_pnl': float(total_pnl_hold),
        },
        'baseline': {
            'trades': int(len(pnl_base)),
            'win_rate': float(win_rate_base),
            'total_pnl': float(total_pnl_base),
        },
        'rich': {
            'trades': int(len(pnl_rich)),
            'win_rate': float(win_rate_rich),
            'total_pnl': float(total_pnl_rich),
        },
    },
    'timestamp': pd.Timestamp.now().isoformat(),
}

report_path = os.path.join(BASE, 'reports', 'model_validation.json')
os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\nSaved validation report to {report_path}")

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

# Recommendations
print("\nRECOMMENDATIONS:")
if len(df) < 20:
    print("⚠️  Dataset too small (< 20 samples)")
    print("   → Collect more live trading data")
    print("   → Run export_trades.py after each trading session")

if r2_base < 0:
    print("⚠️  Negative R² - model worse than predicting mean")
    print("   → Need more diverse training data")
    print("   → Consider different features or model architecture")

if win_rate_base <= win_rate_hold:
    print("⚠️  Model strategy not better than buy & hold")
    print("   → Adjust prediction threshold")
    print("   → Add filtering based on prediction confidence")
