# HolonicTrader ML Integration - Clean Setup Plan

**Date:** 2026-03-21  
**Status:** ✅ Governor Patch Applied

---

## 🎯 What Was Done

### 1. Governor Integration Complete

**File Modified:** `HolonicTrader/agent_governor.py`

**Changes:**
1. **ML Advisor Import** (Line 64-70)
   - Added import for `get_ml_advisor` and `MLTradingAdvisor`
   - Graceful fallback if ML not available

2. **ML Initialization** (Line 230-243)
   - Initialize ML Advisor in `__init__`
   - Display model status and database stats
   - Create tracking dicts for predictions/performance

3. **Position Sizing Adjustment** (Line 4431-4483)
   - Query ML model before finalizing position size
   - Adjust size based on win probability:
     - >60% + HIGH → 100% size
     - >50% → 70% size
     - >40% → 30% size
     - <40% → 10% size (discourage)
   - Store predictions for validation

4. **Performance Tracking** (Line 655-683)
   - Record prediction vs actual outcome
   - Calculate rolling accuracy (last 20 trades)
   - Auto-cleanup after trade closes

---

## 📋 Clean Setup Plan

### Phase 1: Verification (Day 1)

**Goal:** Ensure ML integration doesn't break existing functionality

**Steps:**
```bash
# 1. Test Governor initialization
python -c "
from HolonicTrader.agent_governor import GovernorHolon
gov = GovernorHolon()
print('Governor initialized successfully')
print(f'ML Advisor: {gov.ml_advisor is not None}')
print(f'Models loaded: {gov.ml_advisor.get_model_status() if gov.ml_advisor else \"N/A\"}')
"

# 2. Run existing tests
python test_ml_advisor.py

# 3. Check Governor patch
python -c "
from HolonicTrader.ml_advisor import predict_trade
result = predict_trade('BTC/USDT', 'BUY', 95000.0, 0.001)
print(f'Test prediction: {result}')
"
```

**Expected Output:**
```
✓ Governor initialized
✓ ML Advisor loaded (872 trades, 50.2% win rate)
✓ Test predictions working
```

---

### Phase 2: Paper Trading (Days 2-7)

**Goal:** Validate ML predictions in live market conditions without real money

**Configuration:**
```python
# config.py
ML_ENABLED = True
ML_PAPER_MODE = True  # Log predictions but don't adjust sizes yet
ML_LOG_ALL = True     # Log all ML decisions
```

**Monitoring:**
```bash
# Daily ML accuracy check
python -c "
import json
with open('logs/governor_ml_log.json', 'r') as f:
    data = json.load(f)
    
print(f'Total predictions: {len(data)}')
if data:
    recent = data[-20:]
    correct = sum(1 for d in recent if d['predicted_win'] == d['actual_win'])
    print(f'Recent accuracy: {correct/len(recent):.1%}')
"
```

**Success Criteria:**
- No trading errors
- ML predictions logged correctly
- Accuracy >60% on 20+ trades

---

### Phase 3: Live Trading - Conservative (Week 2)

**Goal:** Enable ML sizing with conservative parameters

**Configuration:**
```python
# config.py
ML_ENABLED = True
ML_PAPER_MODE = False  # Enable actual sizing adjustments

# Conservative sizing
ML_SIZE_HIGH = 0.7     # 70% even for high confidence
ML_SIZE_MEDIUM = 0.5   # 50% for moderate
ML_SIZE_LOW = 0.2      # 20% for low
ML_VETO_THRESHOLD = 0.25  # Only veto extremely low confidence
```

**Risk Limits:**
```python
# Max 50% of normal size for first week
ML_GLOBAL_CAP = 0.5

# Max 5 ML-influenced trades per day
ML_MAX_TRADES_PER_DAY = 5
```

**Monitoring Dashboard:**
```bash
# Real-time ML status
watch -n 60 'python -c "
from HolonicTrader.ml_advisor import get_ml_advisor
advisor = get_ml_advisor()
status = advisor.get_model_status()
print(f\"Database: {status['database_trades']} trades\")
print(f\"Cache: {status['cache_size']} predictions\")
print(f\"Models: CLF={status['classifier_loaded']}, REG={status['regression_loaded']}\")
"'
```

---

### Phase 4: Full Deployment (Week 3+)

**Goal:** Full ML integration with optimized parameters

**Configuration:**
```python
# config.py
ML_ENABLED = True
ML_PAPER_MODE = False

# Optimized sizing (from Phase 2-3 data)
ML_SIZE_HIGH = 1.0     # Full size for high confidence
ML_SIZE_MEDIUM = 0.7   # 70% for moderate
ML_SIZE_LOW = 0.3      # 30% for low
ML_VETO_THRESHOLD = 0.3  # Standard veto threshold

# Performance tracking
ML_ACCURACY_TARGET = 0.65  # Target 65%+ accuracy
ML_RETRAIN_TRIGGER = 0.50  # Retrain if accuracy <50% for 50 trades
```

**Automated Monitoring:**
```python
# Add to Governor's get_status() method
def get_ml_status(self):
    if not self.ml_advisor:
        return {'enabled': False}
    
    perf = self.ml_performance[-50:] if self.ml_performance else []
    if perf:
        accuracy = sum(1 for p in perf if p['prediction_correct']) / len(perf)
        high_conf = [p for p in perf if p['predicted_confidence'] == 'HIGH']
        high_conf_acc = sum(1 for p in high_conf if p['prediction_correct']) / len(high_conf) if high_conf else 0
    else:
        accuracy = 0
        high_conf_acc = 0
    
    return {
        'enabled': True,
        'active_predictions': len(self.ml_predictions),
        'total_tracked': len(self.ml_performance),
        'accuracy_50': accuracy,
        'high_confidence_accuracy': high_conf_acc,
        'status': 'GOOD' if accuracy > 0.6 else 'NEEDS_ATTENTION',
    }
```

---

## 🔧 Maintenance Plan

### Weekly Tasks

**Every Monday:**
```bash
# 1. Review ML performance from past week
python scripts/validate_models.py

# 2. Check if retraining needed
python -c "
import json
with open('logs/train_db_trades.json', 'r') as f:
    model = json.load(f)
    
# Compare with recent performance
with open('reports/ml_performance.json', 'r') as f:
    recent = json.load(f)

print(f\"Training accuracy: {model['classification_model']['accuracy']:.1%}\")
print(f\"Recent accuracy: {recent['accuracy_50']:.1%}\")

if recent['accuracy_50'] < model['classification_model']['accuracy'] * 0.8:
    print('⚠️  Performance degradation detected - consider retraining')
"
```

**Every Friday:**
```bash
# Export new trades to database
python scripts/export_trades.py

# Check database size
python -c "
import sqlite3
conn = sqlite3.connect('holonic_trader.db')
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM trades WHERE pnl IS NOT NULL')
count = cur.fetchone()[0]
print(f'Database trades: {count}')
if count > 1000:
    print('✓ Good dataset size')
elif count > 500:
    print('⚠️  Consider retraining with new data')
else:
    print('⚠️  Dataset still growing')
conn.close()
"
```

### Monthly Tasks

**First of Month:**
```bash
# 1. Retrain models with all available data
python scripts/train_on_database.py

# 2. Validate new models
python test_ml_advisor.py

# 3. Backup old models
cp models/lgbm_*.pkl models/backup_$(date +%Y%m).pkl

# 4. Deploy new models
# (Already in place from training script)

# 5. Document performance changes
# Update reports/ML_PERFORMANCE_LOG.md
```

---

## 📊 Performance Monitoring

### Key Metrics Dashboard

Create `dashboard_ml.py`:
```python
#!/usr/bin/env python3
"""ML Performance Dashboard"""
import sqlite3
import json
from datetime import datetime, timedelta

def get_ml_dashboard():
    print("=" * 70)
    print("ML TRADING DASHBOARD")
    print("=" * 70)
    
    # 1. Database Stats
    conn = sqlite3.connect('holonic_trader.db')
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM trades WHERE pnl IS NOT NULL")
    total_trades = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM trades WHERE pnl > 0")
    wins = cur.fetchone()[0]
    
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    print(f"\n📊 DATABASE ({total_trades} trades)")
    print(f"   Win Rate: {win_rate:.1%}")
    print(f"   Wins: {wins}, Losses: {total_trades - wins}")
    
    # 2. Model Status
    try:
        from HolonicTrader.ml_advisor import get_ml_advisor
        advisor = get_ml_advisor()
        status = advisor.get_model_status()
        
        print(f"\n🤖 ML MODELS")
        print(f"   Classifier: {'✓' if status['classifier_loaded'] else '✗'}")
        print(f"   Regression: {'✓' if status['regression_loaded'] else '✗'}")
        print(f"   Features: {status['features_used']}")
    except Exception as e:
        print(f"\n🤖 ML MODELS: Error - {e}")
    
    # 3. Recent Performance (from Governor logs)
    try:
        with open('logs/governor_ml_log.json', 'r') as f:
            ml_log = json.load(f)
        
        recent = ml_log[-50:] if ml_log else []
        if recent:
            accuracy = sum(1 for r in recent if r.get('prediction_correct', False)) / len(recent)
            high_conf = [r for r in recent if r.get('confidence') == 'HIGH']
            high_conf_acc = sum(1 for r in high_conf if r.get('prediction_correct', False)) / len(high_conf) if high_conf else 0
            
            print(f"\n📈 RECENT PERFORMANCE (Last 50)")
            print(f"   Overall Accuracy: {accuracy:.1%}")
            print(f"   High Confidence: {high_conf_acc:.1%} ({len(high_conf)} trades)")
        else:
            print(f"\n📈 RECENT PERFORMANCE: No data yet")
    except FileNotFoundError:
        print(f"\n📈 RECENT PERFORMANCE: Log not found")
    
    # 4. Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    if total_trades < 500:
        print("   ⚠️  Dataset small - collect more trades before major changes")
    if total_trades >= 500 and total_trades < 1000:
        print("   ✓ Dataset growing - consider retraining soon")
    if total_trades >= 1000:
        print("   ✓ Dataset healthy - ready for retraining")
    
    if recent and accuracy < 0.55:
        print("   ⚠️  Accuracy low - review recent trades for patterns")
    elif recent and accuracy > 0.70:
        print("   ✓ Excellent accuracy - consider increasing ML sizing")
    
    print("\n" + "=" * 70)
    
    conn.close()
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'recent_accuracy': accuracy if recent else None,
    }

if __name__ == '__main__':
    get_ml_dashboard()
```

**Usage:**
```bash
# Run dashboard
python dashboard_ml.py

# Or add to crontab for hourly updates
0 * * * * cd /path/to/HolonicTrader && python dashboard_ml.py >> logs/ml_dashboard.log
```

---

## 🚨 Troubleshooting

### Issue: Governor won't start

**Symptoms:**
```
ImportError: cannot import name 'get_ml_advisor'
```

**Solution:**
```bash
# Check if ml_advisor.py exists
ls -la HolonicTrader/ml_advisor.py

# If missing, restore from backup or recreate
# Check Python path
python -c "import sys; print('\\n'.join(sys.path))"

# Ensure HolonicTrader is in path
export PYTHONPATH=/path/to/HolonicTrader:$PYTHONPATH
```

### Issue: ML predictions all same value

**Symptoms:**
```
Win Probability: 50.0% for all trades
```

**Solution:**
```bash
# Check model files
ls -la models/lgbm_*.pkl

# If files are 0 bytes or missing, retrain
python scripts/train_on_database.py

# Verify features being passed
python -c "
from HolonicTrader.ml_advisor import get_ml_advisor
advisor = get_ml_advisor()
X = advisor._prepare_features('BTC/USDT', 'BUY', 50000, 0.001)
print('Features:', X.values)
print('Unique values:', X.nunique())
"
```

### Issue: Accuracy dropping

**Symptoms:**
```
ML Accuracy (last 20): 35.0%
```

**Solution:**
```bash
# 1. Check for data drift
python scripts/export_db_trades.py

# 2. Review recent trades
python -c "
import sqlite3
conn = sqlite3.connect('holonic_trader.db')
query = '''
SELECT symbol, pnl_percent, timestamp 
FROM trades 
WHERE timestamp > datetime('now', '-7 days')
ORDER BY timestamp DESC
LIMIT 20
'''
recent = pd.read_sql_query(query, conn)
print(recent)
"

# 3. Consider retraining if >50 new trades collected
python scripts/train_on_database.py
```

---

## 📞 Support

**Files Reference:**
- `HolonicTrader/ml_advisor.py` - Main ML module
- `HolonicTrader/agent_governor.py` - Governor with ML patch
- `scripts/train_on_database.py` - Model training
- `test_ml_advisor.py` - Integration tests

**Documentation:**
- `reports/ML_INTEGRATION_COMPLETE.md` - Full integration guide
- `reports/DATABASE_INTEGRATION_GUIDE.md` - Database details
- `reports/BUY_SELL_TRAINING_GUIDE.md` - Training pipeline

**Logs:**
- `logs/governor_ml_log.json` - ML predictions and outcomes
- `logs/train_db_trades.json` - Training metrics
- `reports/ml_performance.json` - Rolling performance

---

## ✅ Checklist

### Before Going Live

- [ ] Governor initializes without errors
- [ ] ML Advisor loads successfully
- [ ] Test predictions return varied values
- [ ] Position sizing adjusts based on confidence
- [ ] Performance tracking records outcomes
- [ ] Paper trading validates logic (1 week)
- [ ] Accuracy >60% on 20+ trades
- [ ] Backup models created
- [ ] Monitoring dashboard running
- [ ] Rollback plan documented

### Weekly Maintenance

- [ ] Review ML accuracy
- [ ] Export new trades to database
- [ ] Check for performance degradation
- [ ] Update performance log
- [ ] Verify model health

### Monthly Maintenance

- [ ] Retrain models if >100 new trades
- [ ] Validate new models
- [ ] Backup old models
- [ ] Document performance changes
- [ ] Adjust parameters if needed

---

**Last Updated:** 2026-03-21  
**Next Review:** 2026-03-28
