# Feature Engineering & Model Retraining Report

**Date:** 2026-03-20  
**Status:** ✅ Pipeline Complete, ⚠️ Data Quality Issues Identified

---

## Executive Summary

The feature engineering pipeline has been successfully implemented and executed. However, **critical data quality issues** were discovered that prevent effective model learning.

---

## Completed Work

### 1. Feature Engineering Pipeline (`scripts/feature_engineering.py`)

Implemented all 7 steps from the plan:

| Step | Status | Output |
|------|--------|--------|
| 1. Audit & drop constant/zero features | ✅ | Dropped: `fee_usd`, `order_id`, `session_file`, `raw`, `event` |
| 2. Feature distributions & correlations | ✅ | Plot: `plots/feature_analysis/feature_correlations.png` |
| 3. Engineer new features | ✅ | Added 13 features (rolling, lagged, time-based, regime) |
| 4. Feature importance analysis | ✅ | Plot: `plots/feature_analysis/feature_importance.png` |
| 5. Update training pipelines | ✅ | Features aligned across scripts |
| 6. Retrain models | ✅ | Both models retrained with optimized hyperparameters |
| 7. Validate & backtest | ✅ | Backtest executed |

### 2. Updated Scripts

| Script | Changes |
|--------|---------|
| `train_save_rich.py` | Uses `engineered_features_v2.parquet`, refined features, optimized hyperparams |
| `train_quick_model.py` | Dropped constant features (`event_id`, `fee_usd`), improved target calculation |
| `ab_backtest.py` | Prefers v2 features, consistent feature sets, better error handling |

### 3. Generated Artifacts

- `datasets/engineered_features_v2.parquet` - Refined feature set (131 rows, 7 columns)
- `reports/feature_engineering_report.json` - Full analysis report
- `plots/feature_analysis/feature_correlations.png` - Feature correlation heatmap
- `plots/feature_analysis/feature_importance.png` - LightGBM feature importance
- `models/lgbm_return_v1.pkl` - Retrained baseline model
- `models/lgbm_return_rich.pkl` - Retrained rich model
- `backtests/ab_backtest.csv` - A/B backtest results

---

## Critical Findings

### 🚨 Data Quality Issues

| Issue | Impact | Evidence |
|-------|--------|----------|
| **Small dataset** | Model can't learn patterns | Only 131 rows → 105 train / 26 test |
| **64% zero price movement** | Model predicts constant | 84/131 rows have `target == 0` |
| **Class imbalance** | Biased predictions | 23 positive vs 24 negative vs 84 zero |
| **Early stopping @ iteration 1** | No learning occurs | Model defaults to predicting mean |

### Feature Analysis Results

**Correlation with Target (price log-return):**
```
ret:           -0.2594  (highest)
atr:            0.0913
rv_10_ann:      0.0612
rv_10:          0.0612
price:         -0.0473
qty:            0.0443
vol_spike:     -0.0418
```

**Feature Importance (all zero - no learning):**
```
All 20 features: 0.00 importance
```

**Final Selected Features:** `['qty', 'price', 'ret']`

---

## Model Performance

| Model | Train MSE | Test MSE | Best Iteration |
|-------|-----------|----------|----------------|
| Baseline (v1) | 0.000462 | 0.000462 | 1 |
| Rich (v2) | 0.000463 | 0.000463 | 1 |

**⚠️ Both models stop at iteration 1** - indicating no learnable patterns beyond the mean.

### Prediction Distribution (Rich Model)
```
min:    -0.003247
max:    -0.002487
mean:   -0.002652
std:     0.000296
```

**All predictions are negative** - model learned the slight negative bias in the data.

### Backtest Results
```
Baseline:  0 trades, $0.00 PnL
Rich:      0 trades, $0.00 PnL
```

No trades triggered because all predictions < threshold (0.0005).

---

## Root Cause Analysis

The dataset consists primarily of **stop-loss placement events** where:
1. Price often doesn't move between consecutive events (`price_next == price`)
2. The target variable is mostly zero (64% of samples)
3. The model learns to predict the mean (-0.002) rather than patterns

**This is fundamentally a data collection issue, not a modeling issue.**

---

## Recommendations

### Immediate Actions

1. **Collect More Diverse Data**
   - Need data from actual trades (entry/exit), not just stop placements
   - Include more price movement scenarios
   - Target: minimum 1,000+ samples with varied price movements

2. **Change Target Definition**
   - Current: `log(price_next / price)` between consecutive events
   - Suggested: `log(exit_price / entry_price)` for complete trades
   - Or: Multi-step forward returns (e.g., 5-event horizon)

3. **Filter Zero-Movement Samples**
   ```python
   # In feature_engineering.py, add:
   df = df[df['price_next'] != df['price']].copy()
   ```

4. **Alternative: Use Classification**
   - Predict direction (up/down) instead of magnitude
   - More robust to class imbalance with proper weighting

### Pipeline Improvements

5. **Add Cross-Validation**
   - K-fold CV for small datasets
   - More robust performance estimation

6. **Feature Scaling**
   - Add StandardScaler/RobustScaler for numeric features
   - Especially important for `qty` and `price` (large magnitude differences)

7. **Regularization**
   - Add `lambda_l1`, `lambda_l2` to prevent overfitting on small data

8. **Ensemble Methods**
   - Bagging multiple models trained on bootstrap samples
   - More stable predictions

---

## Next Steps

1. **Decision Required:** Should we:
   - [ ] Filter out zero-movement samples and retrain?
   - [ ] Collect more diverse trading data first?
   - [ ] Switch to a classification approach?
   - [ ] Use synthetic data augmentation?

2. **If proceeding with current data:**
   ```bash
   # Lower the prediction threshold for backtest
   python scripts/ab_backtest.py 0.00001
   ```

3. **For production:**
   - Implement real-time feature computation in trading loop
   - Add model monitoring for drift detection
   - Set up automated retraining triggers

---

## Files Modified

```
scripts/
├── feature_engineering.py          [NEW] Complete pipeline
├── train_save_rich.py              [UPDATED] Refined features
├── train_quick_model.py            [UPDATED] Refined features
└── ab_backtest.py                  [UPDATED] Consistent features

datasets/
└── engineered_features_v2.parquet  [NEW] Refined feature set

reports/
└── feature_engineering_report.json [NEW] Analysis report

plots/feature_analysis/
├── feature_correlations.png        [NEW]
└── feature_importance.png          [NEW]

models/
├── lgbm_return_v1.pkl              [RETRAINED]
└── lgbm_return_rich.pkl            [RETRAINED]
```

---

## Contact

For questions about this analysis, review:
- `reports/feature_engineering_report.json` - Full metrics
- `plots/feature_analysis/` - Visualizations
- `backtests/ab_backtest.csv` - Detailed trade log
