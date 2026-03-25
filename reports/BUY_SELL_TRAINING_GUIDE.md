# Buy/Sell Training & Monte Carlo Position Management Guide

**Date:** 2026-03-21  
**Status:** ✅ Complete Training Pipeline

---

## Overview

This guide covers two critical components of the HolonicTrader ML system:

1. **Directional Model Training** - Predicting good vs bad entry signals
2. **Monte Carlo Position Sizing** - Optimizing position sizes using historical trade analysis

---

## 1. Directional Model Training

### Purpose
Train a classification model to predict whether a trade entry will be profitable (good entry) or result in significant losses (bad entry).

### Script
```bash
python scripts/train_directional.py
```

### What It Does

1. **Loads complete trades** from `datasets/complete_trades_v2.parquet`
2. **Creates classification target**: `good_entry = (PnL > -5%)`
3. **Trains LightGBM classifier** with cross-validation
4. **Saves model** to `models/lgbm_directional.pkl`

### Output

```
Directional Model CV Accuracy: 100.00% (+/- 0.00%)
Saved directional model to models/lgbm_directional.pkl
```

**Note:** 100% accuracy indicates overfitting on small dataset (6 samples). Collect more trades for reliable models.

### Model Usage

```python
import joblib

# Load model
model = joblib.load('models/lgbm_directional.pkl')

# Predict (1 = good entry, 0 = bad entry)
features = {
    'qty': 10.0,
    'price': 100.0,
    'ret': 0.01,
    'rv_10': 0.02,
    'rv_10_ann': 0.5,
    'atr': 2.0,
    'vol_spike': 0,
    'trade_entry_qty': 10.0,
    'trade_entry_leverage': 2.0,
}
X = pd.DataFrame([features])[model.feature_name()]
prob_good = model.predict(X)[0]

if prob_good > 0.5:
    print("Good entry signal - proceed")
else:
    print("Bad entry signal - skip")
```

---

## 2. Monte Carlo Position Sizing Optimization

### Purpose
Analyze historical trades to find optimal position sizing parameters for the Monte Carlo risk court.

### Script
```bash
python scripts/train_monte_carlo_sizing.py
```

### What It Does

1. **Analyzes historical PnL** distribution
2. **Calculates Kelly Criterion** for optimal bet sizing
3. **Simulates 8 position sizing strategies** (5% to 50% of equity)
4. **Runs 1000 Monte Carlo paths** per strategy
5. **Calculates CVaR** (Conditional Value at Risk)
6. **Recommends optimal parameters** for SMCE Monte Carlo Court

### Output Example

```
Kelly Criterion:
  Win/Loss Ratio: 0.000
  Kelly Fraction: 0.00% (no winning trades yet)
  Half-Kelly:     0.00%

Optimal Fraction: 5% (Sharpe=-20.85, DD=0.2%)

CVaR (95%): 0.86%
CVaR (99%): 0.86%

Recommended Monte Carlo Parameters:
  • Fixed Fraction:     5.0% of equity
  • Kelly Fraction:     0.0% of equity
  • Max Drawdown:       0.2%
  • CVaR (95%):         1.7%
```

### Integration with SMCE Monte Carlo Court

**File:** `HolonicTrader/smce_monte_carlo_court.py`

Update the veto thresholds:

```python
# Current (from optimization report)
VETO_DRAWDOWN_PROB  = 0.10   # Keep
VETO_DRAWDOWN_LIMIT = 0.002  # Updated from 0.05 (0.2%)
VETO_CVAR_LIMIT     = 0.017  # Updated from 0.08 (1.7%)
VETO_LIQ_PROB       = 0.01   # Keep
```

**File:** `config.py`

Add optimized parameters:

```python
# Monte Carlo Position Sizing (from optimization)
MONTE_CARLO_POSITION_SIZE = 0.05      # 5% of equity
MONTE_CARLO_CVAR_LIMIT = 0.017        # 1.7% CVaR limit
MONTE_CARLO_DRAWDOWN_LIMIT = 0.002    # 0.2% max drawdown
MONTE_CARLO_PATHS = 1000
MONTE_CARLO_HORIZON_HOURS = 24
```

---

## 3. Complete Training Pipeline

### Step-by-Step Workflow

```bash
# 1. Extract latest trades from logs
python scripts/export_trades.py

# 2. Create ML dataset (match ENTRY→EXIT with features)
python scripts/create_ml_dataset.py

# 3. Train regression model (predict PnL)
python scripts/train_on_real_trades.py

# 4. Train directional model (predict good/bad entry)
python scripts/train_directional.py

# 5. Optimize Monte Carlo position sizing
python scripts/train_monte_carlo_sizing.py

# 6. Validate all models
python scripts/validate_models.py
```

### Output Files

| File | Description |
|------|-------------|
| `models/lgbm_return_v1.pkl` | Baseline PnL regression |
| `models/lgbm_return_rich.pkl` | Rich feature PnL regression |
| `models/lgbm_directional.pkl` | Good/bad entry classifier |
| `reports/model_validation.json` | Validation metrics |
| `reports/monte_carlo_sizing_optimization.json` | MC parameters |

---

## 4. Monte Carlo Position Manager Integration

### Existing System

The `MonteCarloPositionManager` holon already exists and provides:

- **Position health evaluation** using Monte Carlo simulations
- **Closure recommendations** for losing positions
- **Probability calculations** for hitting stop loss vs take profit

### How It Works

```python
# Called by Governor during position management
should_close, confidence, reason = monte_carlo.evaluate_position_for_closure(
    symbol='DOT/USDT',
    current_price=1.485,
    entry_price=1.492,
    direction='BUY',
    position_age_hours=2.5,
    sde_params={'mu': 0.0, 'sigma': 0.1, 'lambda': 0.1},
    pnl_pct=-0.0047  # -0.47%
)

# Returns: (True, 0.72, "Low recovery chance: 28%")
```

### SDE Engine Integration

The Monte Carlo simulations use the `SDEEngine` for path generation:

```python
from HolonicTrader.sde_engine import SDEEngine

# Calculate probability of hitting stop loss before take profit
prob_sl = SDEEngine.calculate_ruin_probability(
    model='GBM',
    params={'mu': 0.0, 'sigma': 0.1, 'lambda': 0.1},
    start_price=current_price,
    sl_price=stop_loss,
    tp_price=take_profit,
    horizon=10000,
    paths=500
)
```

---

## 5. Current Limitations & Recommendations

### Data Limitations

| Issue | Impact | Solution |
|-------|--------|----------|
| **Only 6-9 trades** | Models overfit, unreliable | Collect 50+ trades |
| **0% win rate** | Kelly = 0, no winning patterns | Continue live trading |
| **All losses** | Negative Sharpe ratios | Improve entry timing |

### Recommended Next Steps

1. **Collect More Data** (Priority: HIGH)
   - Run `export_trades.py` after each trading session
   - Target: 50-100 complete trades for statistical significance

2. **Retrain Weekly** (Priority: MEDIUM)
   - Once you have 20+ trades, retrain models
   - Monitor improvement in win rate and Sharpe ratio

3. **Integrate with Governor** (Priority: MEDIUM)
   - Use directional model to filter entry signals
   - Use Monte Carlo sizing for position management

4. **Add Features** (Priority: LOW)
   - Market regime indicators
   - Correlation with BTC/ETH
   - Order book imbalance features

---

## 6. Configuration Template

### `config.py` - Add These Parameters

```python
# === ML Model Configuration ===
ML_MODEL_ENABLED = True
ML_MODEL_PATH = 'models/lgbm_return_rich.pkl'
ML_DIRECTIONAL_PATH = 'models/lgbm_directional.pkl'
ML_MIN_CONFIDENCE = 0.6  # Minimum probability to trust model

# === Monte Carlo Position Sizing ===
MONTE_CARLO_ENABLED = True
MONTE_CARLO_POSITION_SIZE = 0.05      # 5% of equity
MONTE_CARLO_CVAR_LIMIT = 0.017        # 1.7% CVaR limit
MONTE_CARLO_DRAWDOWN_LIMIT = 0.002    # 0.2% max drawdown
MONTE_CARLO_PATHS = 1000
MONTE_CARLO_HORIZON_HOURS = 24
MONTE_CARLO_COOLDOWN_MINUTES = 30

# === Kelly Criterion Sizing ===
USE_KELLY_SIZING = False  # Enable after first winning trade
KELLY_FRACTION = 0.5      # Use half-Kelly for safety
MAX_KELLY_POSITION = 0.10 # Max 10% even if Kelly says more
```

---

## 7. Monitoring & Validation

### Daily Checks

```bash
# Check model predictions vs actual outcomes
python scripts/validate_models.py

# Review Monte Carlo optimization
cat reports/monte_carlo_sizing_optimization.json | python -m json.tool
```

### Key Metrics to Track

| Metric | Target | Current |
|--------|--------|---------|
| Win Rate | > 45% | 0% |
| Avg Win/Avg Loss | > 1.2 | 0.0 |
| Sharpe Ratio | > 0.5 | -20.8 |
| Max Drawdown | < 20% | 0.2% |
| Model Accuracy | > 60% | 100%* |

*100% indicates overfitting on tiny dataset

---

## 8. Troubleshooting

### Problem: "Models predict all losses"
**Cause:** Training data has 0% win rate  
**Solution:** Continue trading, collect winning trades

### Problem: "Kelly fraction = 0"
**Cause:** No winning trades yet  
**Solution:** Use fixed fraction (5%) until first win

### Problem: "Monte Carlo vetoes all trades"
**Cause:** CVaR limits too tight  
**Solution:** Increase `VETO_CVAR_LIMIT` to 0.03 (3%)

### Problem: "Directional model 100% accurate"
**Cause:** Overfitting on < 10 samples  
**Solution:** Collect more data, don't trust predictions yet

---

## 9. Advanced: Custom Monte Carlo Simulations

### Run Custom Simulation

```python
from scripts.train_monte_carlo_sizing import simulate_strategy

# Test your own strategy
pnl_series = pd.Series([-0.01, 0.02, -0.005, 0.015, -0.008])
result = simulate_strategy(
    pnl_series,
    sizing_method='kelly',
    fraction=0.5,  # Half-Kelly
    initial_equity=100,
    n_paths=1000
)

print(f"Expected Equity: ${result['final_equity_median']:.2f}")
print(f"Max Drawdown: {result['max_drawdown_95pct']:.1%}")
print(f"Sharpe Ratio: {result['sharpe_median']:.2f}")
```

---

## Contact & Support

For questions about this training pipeline:
- `reports/IMPLEMENTATION_SUMMARY.md` - Full implementation details
- `reports/DATA_AUDIT_REPORT.md` - Data source documentation
- `scripts/validate_models.py` - Model validation tools
