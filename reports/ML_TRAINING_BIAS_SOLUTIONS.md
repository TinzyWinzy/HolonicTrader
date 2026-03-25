# ML Training Bias Analysis & Solutions

**Date:** 2026-03-21  
**Issue:** Paper trading and live environment training introduces selection biases

---

## 🎯 Problem Statement

### Current Training Data Sources

| Source | Trades | Bias Type | Severity |
|--------|--------|-----------|----------|
| **Database (historical)** | 872 | Survivorship bias | Medium |
| **Paper trading** | 0 (future) | Execution bias, No slippage | High |
| **Live trading** | 9 (so far) | Selection bias, Filtered signals | High |

### Why Paper Trading Generates Biases

1. **No Execution Slippage**
   - Paper: Fill at exact signal price
   - Live: Slippage of 0.1-0.5% typical
   - **Impact:** Overstates win rate by 5-15%

2. **Perfect Fill Assumption**
   - Paper: All signals get filled
   - Live: Some orders rejected/partial
   - **Impact:** Misses worst-case scenarios

3. **No Market Impact**
   - Paper: Trades don't move market
   - Live: Large orders affect price
   - **Impact:** Overstates capacity

4. **Selection Bias**
   - Only captures trades that pass ALL filters
   - Missing: Rejected signals, vetoed trades
   - **Impact:** Model only sees "approved" trades

---

## 📊 Bias Breakdown

### Selection Bias in Current Dataset

```
Total Signals Generated: ~5,000 (estimated)
Signals Passing Filters: ~2,000
Signals Approved by Governor: ~1,000
Executed Trades: ~872 (in database)
Trades with PnL: 872

→ Model trained on 17% of original signals
→ Only sees "survivors" of multiple filters
→ Cannot learn from rejected signals
```

### Consequences

1. **Overconfidence**
   - Model learns from pre-filtered, high-quality signals
   - Will overestimate win rate on raw signals

2. **Missing Rejection Patterns**
   - Cannot learn characteristics of bad trades
   - Only knows what winning trades look like

3. **Regime Bias**
   - Different regimes have different filter pass-rates
   - Model may be biased toward certain regimes

---

## ✅ Solutions

### Solution 1: Train on ALL Signals (Recommended)

**Implementation:**
```python
# In signal generation layer
def generate_signal(symbol, direction, price, metadata):
    signal = Signal(symbol, direction, price, metadata)
    
    # Record ALL signals, not just executed ones
    record_signal_for_ml(
        symbol=symbol,
        direction=direction,
        price=price,
        timestamp=datetime.now(),
        passed_filters=False,  # Initially false
        executed=False,
        outcome=None  # Unknown yet
    )
    
    # Pass through filters
    if passes_all_filters(signal):
        # Update record
        update_signal_record(signal.id, passed_filters=True)
        
        # Execute trade
        result = execute_signal(signal)
        
        # Update with outcome
        update_signal_record(signal.id, 
                            executed=True,
                            pnl=result.pnl,
                            pnl_percent=result.pnl_percent)
    
    return signal
```

**Benefits:**
- Model sees full signal distribution
- Can learn rejection patterns
- Calibrated probabilities

**Implementation Effort:** Medium

---

### Solution 2: Counterfactual Data Augmentation

**Implementation:**
```python
# For each executed trade, create counterfactuals
def create_counterfactuals(trade):
    """
    Generate what-if scenarios for training
    """
    counterfactuals = []
    
    # What if we took opposite position?
    counterfactuals.append({
        'symbol': trade.symbol,
        'direction': 'SELL' if trade.direction == 'BUY' else 'BUY',
        'price': trade.price,
        'timestamp': trade.timestamp,
        'hypothetical_pnl': -trade.pnl,  # Opposite outcome
        'is_counterfactual': True
    })
    
    # What if we entered at slightly different price? (slippage simulation)
    for slippage in [-0.005, -0.002, 0.002, 0.005]:
        counterfactuals.append({
            'symbol': trade.symbol,
            'direction': trade.direction,
            'price': trade.price * (1 + slippage),
            'timestamp': trade.timestamp,
            'hypothetical_pnl': trade.pnl - (slippage * trade.notional),
            'is_counterfactual': True
        })
    
    return counterfactuals
```

**Benefits:**
- Augments limited dataset
- Simulates slippage effects
- Teaches model about opposite positions

**Implementation Effort:** Low

---

### Solution 3: Importance Weighting

**Implementation:**
```python
# Weight training samples by selection probability
def calculate_sample_weight(signal):
    """
    Weight each sample by inverse of selection probability
    """
    # Estimate probability of this signal being selected
    p_selected = estimate_selection_probability(signal)
    
    # Inverse probability weighting
    weight = 1.0 / max(p_selected, 0.1)  # Cap at 10
    
    return weight

# In training
X_train, y_train, weights = prepare_weighted_data()

model = lgb.train(
    params,
    train_data,
    sample_weight=weights  # Pass weights to LightGBM
)
```

**Benefits:**
- Corrects for selection bias
- No changes to data collection needed
- Works with existing dataset

**Implementation Effort:** Low-Medium

---

### Solution 4: Two-Stage Model (Recommended for Production)

**Architecture:**
```
Stage 1: Selection Model
├─ Input: All signals
├─ Output: P(passes filters)
└─ Trained on: All signals (executed + rejected)

Stage 2: Outcome Model
├─ Input: Executed trades only
├─ Output: P(win | executed)
└─ Trained on: Executed trades with PnL

Combined Prediction:
P(win) = P(passes filters) × P(win | executed)
```

**Implementation:**
```python
class TwoStageMLAdvisor:
    def __init__(self):
        self.selection_model = load_model('models/lgbm_selection.pkl')
        self.outcome_model = load_model('models/lgbm_outcome.pkl')
    
    def predict_trade(self, symbol, direction, price, quantity):
        # Stage 1: Selection probability
        selection_features = self._get_selection_features(symbol, direction, price)
        p_selected = self.selection_model.predict([selection_features])[0]
        
        # Stage 2: Outcome probability (conditional on execution)
        outcome_features = self._get_outcome_features(symbol, direction, price)
        p_win_given_executed = self.outcome_model.predict([outcome_features])[0]
        
        # Combined probability
        p_win = p_selected * p_win_given_executed
        
        return {
            'win_probability': p_win,
            'selection_probability': p_selected,
            'outcome_probability': p_win_given_executed,
        }
```

**Benefits:**
- Explicitly models selection process
- Unbiased outcome predictions
- Can identify filter improvements

**Implementation Effort:** High

---

### Solution 5: Rejection Sampling for Training

**Implementation:**
```python
def create_balanced_dataset(executed_trades, all_signals):
    """
    Create training dataset that matches signal distribution
    """
    # Rejection sample executed trades to match signal distribution
    sampled_trades = []
    
    # Match on key dimensions
    for symbol in all_signals['symbol'].unique():
        symbol_signals = all_signals[all_signals['symbol'] == symbol]
        symbol_trades = executed_trades[executed_trades['symbol'] == symbol]
        
        # Sample trades to match signal distribution by hour
        for hour in range(24):
            hour_signals = symbol_signals[symbol_signals['hour'] == hour]
            hour_trades = symbol_trades[symbol_trades['hour'] == hour]
            
            if len(hour_trades) > 0:
                # Sample with replacement to match signal count
                n_samples = min(len(hour_signals), len(hour_trades) * 2)
                sampled = hour_trades.sample(n=n_samples, replace=True)
                sampled_trades.append(sampled)
    
    return pd.concat(sampled_trades)
```

**Benefits:**
- Reduces distribution mismatch
- Simple to implement
- Works with existing data

**Implementation Effort:** Medium

---

## 🎯 Recommended Approach

### Phase 1: Immediate (This Week)

**Implement:** Counterfactual Augmentation + Importance Weighting

```bash
# 1. Generate counterfactuals
python scripts/generate_counterfactuals.py

# 2. Train with importance weighting
python scripts/train_weighted.py
```

**Expected Improvement:**
- Better calibration
- More realistic win rate estimates

### Phase 2: Short-term (This Month)

**Implement:** Record ALL Signals

```python
# Add to signal generation
def record_all_signals():
    # Log every signal to database
    # Include: passed_filters, executed, outcome
    pass
```

**Expected Improvement:**
- Full signal distribution
- Can train selection model

### Phase 3: Long-term (Next Quarter)

**Implement:** Two-Stage Model

```python
# Train both models
selection_model = train_selection_model(all_signals)
outcome_model = train_outcome_model(executed_trades)
```

**Expected Improvement:**
- Unbiased predictions
- Better filter optimization

---

## 📊 Bias Measurement

### Metrics to Track

```python
def measure_bias(executed_trades, all_signals):
    """
    Quantify selection bias
    """
    metrics = {}
    
    # 1. Distribution mismatch
    for feature in ['hour', 'symbol', 'direction', 'regime']:
        executed_dist = executed_trades[feature].value_counts(normalize=True)
        signals_dist = all_signals[feature].value_counts(normalize=True)
        
        # KL divergence
        kl_div = kl_divergence(executed_dist, signals_dist)
        metrics[f'kl_{feature}'] = kl_div
    
    # 2. Win rate bias
    # Compare paper vs live win rates
    paper_win_rate = executed_trades[executed_trades['is_paper']]['pnl'] > 0
    live_win_rate = executed_trades[~executed_trades['is_paper']]['pnl'] > 0
    metrics['win_rate_gap'] = paper_win_rate.mean() - live_win_rate.mean()
    
    # 3. Selection probability
    # Estimate P(executed | signal)
    overall_selection_rate = len(executed_trades) / len(all_signals)
    metrics['selection_rate'] = overall_selection_rate
    
    return metrics
```

### Acceptable Bias Levels

| Metric | Acceptable | Concerning | Critical |
|--------|------------|------------|----------|
| KL Divergence (hour) | <0.1 | 0.1-0.3 | >0.3 |
| KL Divergence (symbol) | <0.2 | 0.2-0.5 | >0.5 |
| Win Rate Gap | <5% | 5-15% | >15% |
| Selection Rate | >20% | 10-20% | <10% |

---

## 🔧 Implementation Checklist

### Counterfactual Augmentation

- [ ] Create `scripts/generate_counterfactuals.py`
- [ ] Add slippage simulation (-0.5% to +0.5%)
- [ ] Generate opposite position scenarios
- [ ] Merge with original dataset
- [ ] Retrain models

### Importance Weighting

- [ ] Estimate selection probabilities
- [ ] Add sample weights to training
- [ ] Validate weighted vs unweighted performance
- [ ] Deploy weighted model

### All Signals Recording

- [ ] Add signal logging to signal generation
- [ ] Create `all_signals` database table
- [ ] Update signal pipeline to record rejections
- [ ] Add outcome tracking for all signals

### Two-Stage Model

- [ ] Train selection model on all signals
- [ ] Train outcome model on executed trades
- [ ] Combine predictions
- [ ] Validate calibration

---

## 📈 Expected Impact

### Before Bias Correction

```
Model Accuracy (on executed trades): 77.7%
Model Accuracy (on all signals): ~55% (estimated)
Calibration Error: High
Win Rate Overstatement: +10-15%
```

### After Bias Correction

```
Model Accuracy (on all signals): ~65% (more realistic)
Calibration Error: Low
Win Rate Accuracy: ±3%
Selection Bias: Reduced by 60-80%
```

---

## 🚨 Risks

### Risk 1: Reduced Apparent Accuracy

**Issue:** Model accuracy will drop from 77% to ~65%

**Mitigation:**
- Communicate that 77% was inflated by bias
- 65% is more realistic and actionable
- Better calibration > higher accuracy

### Risk 2: More Data Needed

**Issue:** Need to collect ALL signals, not just executed

**Mitigation:**
- Start logging immediately
- Use counterfactuals to augment existing data
- Importance weighting works with current data

### Risk 3: Implementation Complexity

**Issue:** Two-stage model is more complex

**Mitigation:**
- Phase implementation
- Start with counterfactuals (easy)
- Add complexity gradually

---

## 📞 Next Steps

1. **This Week:**
   - Generate counterfactuals
   - Retrain with importance weighting
   - Measure bias metrics

2. **Next Week:**
   - Implement all-signals logging
   - Collect data for 1 week
   - Compare distributions

3. **This Month:**
   - Train two-stage model
   - Validate calibration
   - Deploy to production

---

**Summary:** Paper trading and live environment training introduces significant biases. Use counterfactual augmentation, importance weighting, and eventually two-stage modeling to correct.
