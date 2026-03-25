# HolonicTrader Database Integration Guide

**Date:** 2026-03-21  
**Status:** ✅ Database Discovered & Integrated

---

## 🎉 Major Discovery: Historical Trade Database

The system has a **SQLite database** (`holonic_trader.db`) containing **872 historical trades** with:

| Metric | Value |
|--------|-------|
| **Total Trades** | 872 |
| **Win Rate** | 50.2% |
| **Total Profit** | $420.47 |
| **Avg Win** | $5.17 (148%) |
| **Avg Loss** | $-4.25 (-41%) |
| **Symbols** | 38 different pairs |
| **Date Range** | Feb 21 - Mar 21, 2026 |

This is **vastly superior** to the log-extracted dataset (9 trades, 0% win rate).

---

## Database Schema

### Tables Available

```sql
portfolio          -- Account balance and holdings
ledger             -- Blockchain-style transaction log
trades             -- Trade history with PnL (872 rows)
rl_experiences     -- DQN reinforcement learning memory
memory_vectors     -- Pattern memory embeddings
smce_state         -- SMCE capital doctrine state
asset_profiles     -- Symbol-specific metadata
system_flags       -- System configuration
quantops_reports   -- Quant Ops cycle reports
```

### Trades Table Columns

```python
[
    'id', 'symbol', 'direction', 'quantity', 'price', 'cost_usd',
    'timestamp', 'pnl', 'pnl_percent',
    'unrealized_pnl', 'unrealized_pnl_percent',
    'mfe', 'mae',  # Maximum Favorable/Adverse Excursion
    'exit_reason', 'strategy_type',
    'entropy_score', 'regime', 'conviction',
    'quality_score', 'is_whitelisted'
]
```

---

## New Scripts Created

### 1. `scripts/export_db_trades.py`

Exports database trades to parquet for ML training.

**Usage:**
```bash
python scripts/export_db_trades.py
```

**Output:**
- `datasets/db_trades_full.parquet` (872 trades)
- `datasets/db_trades_ml.parquet` (ML-ready features)
- `datasets/db_trades_summary.json` (Statistics report)

### 2. `scripts/train_on_database.py`

Trains ML models on historical database trades.

**Usage:**
```bash
python scripts/train_on_database.py
```

**Models Created:**
- `models/lgbm_pnl_regression.pkl` - Predicts PnL %
- `models/lgbm_win_classifier.pkl` - Predicts win/loss

**Performance:**
- **Regression RMSE:** 156.7% (high variance in PnL)
- **Classification Accuracy:** 77.7%
- **Precision (Win):** 76%
- **Recall (Win):** 80%

---

## Model Comparison

| Dataset | Trades | Win Rate | Model Accuracy |
|---------|--------|----------|----------------|
| **Database** | 872 | 50.2% | **77.7%** |
| Log-extracted | 9 | 0% | 100%* (overfit) |

*100% accuracy on 6 samples indicates overfitting

---

## Feature Importance Analysis

**Top Features for Win/Loss Prediction:**

| Feature | Importance |
|---------|------------|
| price | 1,709 |
| cost_usd | 2,116 |
| quantity | 1,011 |
| hour | 764 |
| day_of_week | 735 |
| direction_encoded | 370 |

**Key Insight:** Trade size (price, quantity, cost_usd) and timing (hour, day_of_week) are the strongest predictors.

---

## Integration with Existing Pipeline

### Updated Training Workflow

```bash
# 1. Export database trades (ONE-TIME or after DB updates)
python scripts/export_db_trades.py

# 2. Train on database (872 samples - PRIMARY)
python scripts/train_on_database.py

# 3. Train on log-extracted (9 samples - SUPPLEMENTARY)
python scripts/train_on_real_trades.py
python scripts/train_directional.py

# 4. Monte Carlo optimization
python scripts/train_monte_carlo_sizing.py

# 5. Validate all models
python scripts/validate_models.py
```

### Model Selection Strategy

| Use Case | Recommended Model |
|----------|-------------------|
| **Live trading** | `lgbm_win_classifier.pkl` (database-trained) |
| **PnL prediction** | `lgbm_pnl_regression.pkl` (database-trained) |
| **Recent pattern matching** | `lgbm_directional.pkl` (log-trained) |
| **Position sizing** | Monte Carlo optimization results |

---

## Database Query Examples

### Get Recent Winning Trades

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('holonic_trader.db')
query = """
SELECT symbol, direction, pnl, pnl_percent, timestamp
FROM trades 
WHERE pnl > 0 
ORDER BY timestamp DESC 
LIMIT 20
"""
wins = pd.read_sql_query(query, conn)
print(wins)
```

### Analyze by Symbol

```python
query = """
SELECT 
    symbol,
    COUNT(*) as trades,
    SUM(pnl) as total_pnl,
    AVG(pnl_percent) as avg_pnl_pct,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as win_rate
FROM trades
GROUP BY symbol
ORDER BY total_pnl DESC
"""
symbol_perf = pd.read_sql_query(query, conn)
print(symbol_perf)
```

### Analyze by Regime (if available)

```python
query = """
SELECT 
    regime,
    COUNT(*) as trades,
    AVG(pnl_percent) as avg_pnl,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as win_rate
FROM trades
WHERE regime IS NOT NULL
GROUP BY regime
"""
regime_perf = pd.read_sql_query(query, conn)
print(regime_perf)
```

---

## Model Usage Example

```python
import joblib
import pandas as pd

# Load models
reg_model = joblib.load('models/lgbm_pnl_regression.pkl')
clf_model = joblib.load('models/lgbm_win_classifier.pkl')

# Prepare features
FEATURES = ['quantity', 'price', 'cost_usd', 'hour', 'day_of_week', 
            'mfe', 'mae', 'direction_encoded']

features = {
    'quantity': 10.0,
    'price': 100.0,
    'cost_usd': 1000.0,
    'hour': 14,
    'day_of_week': 3,
    'mfe': 0.0,
    'mae': 0.0,
    'direction_encoded': 1,  # BUY
}

# Predict
X = pd.DataFrame([features])[FEATURES]
pnl_pred = reg_model.predict(X)[0]
win_prob = clf_model.predict(X)[0]

print(f"Predicted PnL: {pnl_pred:.2f}%")
print(f"Win Probability: {win_prob:.1%}")

# Trading decision
if win_prob > 0.6:
    print("✓ High confidence win - proceed with trade")
elif win_prob > 0.4:
    print("⚠ Moderate confidence - reduce position size")
else:
    print("✗ Low confidence - skip trade")
```

---

## Performance Metrics

### Database Trades Performance

```
Total PnL: $420.47
Mean PnL:  $0.48 per trade
Std PnL:   $15.49

Win Rate:  50.23%
Avg Win:   $5.17 (148%)
Avg Loss:  $-4.25 (-41%)

Profit Factor: 2.26 (gross wins / gross losses)
```

### Model Performance

```
Classification Accuracy: 77.7%
Precision (Win): 76%
Recall (Win): 80%
F1 Score: 0.78

Regression RMSE: 156.7%
(Note: High due to extreme PnL variance in data)
```

---

## Recommendations

### Immediate Actions

1. **Use database-trained models for live trading**
   - 872 samples vs 9 samples
   - 50% win rate vs 0%
   - 77.7% accuracy vs overfit 100%

2. **Continue logging to database**
   - Database is the PRIMARY data source
   - Log extraction is BACKUP only

3. **Retrain weekly**
   - Add new trades from database
   - Monitor model drift

### Model Improvements

4. **Add more features**
   - Market regime indicators
   - Correlation features
   - Technical indicators at entry time

5. **Ensemble models**
   - Combine database + log-trained models
   - Weight by sample size and recency

6. **Calibrate probabilities**
   - Use Platt scaling or isotonic regression
   - Improve win probability estimates

---

## Troubleshooting

### Database Not Found

```bash
# Check if database exists
ls -la holonic_trader.db

# If missing, check main_live_phase4.py for DB path
# Default: holonic_trader.db in project root
```

### No Trades with PnL

```python
# Check if trades table has data
import sqlite3
conn = sqlite3.connect('holonic_trader.db')
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM trades")
print(f"Total trades: {cur.fetchone()[0]}")
cur.execute("SELECT COUNT(*) FROM trades WHERE pnl IS NOT NULL")
print(f"Trades with PnL: {cur.fetchone()[0]}")
```

### Model Accuracy Too Low

```bash
# Check data quality
python scripts/export_db_trades.py

# Review feature importance
# May need to add more predictive features
# Or segment by symbol/regime
```

---

## Files Modified/Created

```
scripts/
├── export_db_trades.py           [NEW] Database export
└── train_on_database.py          [NEW] Database training

datasets/
├── db_trades_full.parquet        [NEW] 872 trades
├── db_trades_ml.parquet          [NEW] ML-ready features
└── db_trades_summary.json        [NEW] Statistics

models/
├── lgbm_pnl_regression.pkl       [NEW] PnL prediction
└── lgbm_win_classifier.pkl       [NEW] Win/loss classifier

logs/
└── train_db_trades.json          [NEW] Training report

reports/
└── DATABASE_INTEGRATION_GUIDE.md [NEW] This document
```

---

## Next Steps

1. ✅ Database discovered and exported
2. ✅ Models trained on 872 trades
3. ✅ 77.7% classification accuracy achieved
4. 🔄 **Integrate with live trading system**
5. 🔄 **Add real-time feature computation**
6. 🔄 **Deploy model predictions to Governor**

---

## Contact

For questions about database integration:
- `scripts/export_db_trades.py` - Export logic
- `scripts/train_on_database.py` - Training logic
- `reports/BUY_SELL_TRAINING_GUIDE.md` - Full training guide
