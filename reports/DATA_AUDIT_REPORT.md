# Data Audit Report - HolonicTrader

**Date:** 2026-03-21  
**Purpose:** Identify all available data sources for ML training

---

## Executive Summary

The system has been producing trading data, but **ENTRY/EXIT events are logged differently than expected**. The primary data sources are:

1. **Live trading logs** (`live_trading_session_*.log`) - Contains actual trade executions
2. **Quant Ops reports** (`quant_ops_reports/cycle_*.json`) - Aggregated performance metrics + loss attribution
3. **Execution integrity log** (`logs/execution_integrity.json`) - System events (no TRADE events found)
4. **Feature datasets** (`.parquet` files) - Derived features, but missing price movement labels

---

## Data Source Inventory

### 1. Live Trading Logs (PRIMARY SOURCE)

**Files:** `live_trading_session_*.log` (11 files found)

| File | Lines | ENTRY | EXIT | Date Range |
|------|-------|-------|------|------------|
| live_trading_session_20260320_171733.log | 1,871 | 1 | 0 | 2026-03-20 |
| live_trading_session_20260320_182857.log | 3,623 | 3 | 2 | 2026-03-20 |
| live_trading_session_20260320_211553.log | 2,308 | 1 | 2 | 2026-03-20 |
| live_trading_session_20260320_230849.log | 30 | 0 | 0 | 2026-03-20 |
| live_trading_session_20260320_231028.log | 1,253 | 1 | 0 | 2026-03-20 |
| live_trading_session_20260320_233554.log | 4,345 | 1 | 1 | 2026-03-20 |
| live_trading_session_20260321_012601.log | 1,028 | 1 | 1 | 2026-03-21 |
| live_trading_session_20260321_015823.log | 1,114 | 1 | 1 | 2026-03-21 |
| live_trading_session_20260321_022540.log | 13,983 | 1 | 3 | 2026-03-21 |
| live_trading_session_20260321_100553.log | 17,602 | 2 | 1 | 2026-03-21 |
| live_trading_session_20260321_221805.log | 1,903 | 1 | 1 | 2026-03-21 |
| **TOTAL** | **49,060** | **13** | **12** | 2026-03-20 to 2026-03-21 |

**ENTRY Event Format:**
```
[2026-03-21 01:27:41] [TraderNexus] 🎯 EXECUTING ENTRY (Attempt 1/2): DOT/USDT (Qty: 9.9536, Lev: 2.0x)
```

**EXIT Event Format:**
```
[2026-03-21 01:59:39] [GovernorAgent] 📉 Trade Logged: DOT/USDT Profit: -0.46%. Consecutive Losses: 1/2
```

**Issues:**
- EXIT events don't contain entry/exit prices, only PnL %
- Only 3 trades could be matched (ENTRY→EXIT) due to timestamp/symbol alignment
- Most exits lack corresponding entry price data

---

### 2. Quant Ops Reports (AGGREGATED DATA)

**Files:** `quant_ops_reports/cycle_*.json` (468 files found)

**Extracted Data:**
- **240 trade records** from last 20 cycles
- **220 loss attribution records** with:
  - Symbol
  - PnL % (negative only - losses)
  - Loss category (RISK/REGIME/SIGNAL)
  - Loss reason (STOP_TOO_LOOSE, VOLATILITY_SHIFT, FALSE_POSITIVE)

**Loss Category Distribution:**
| Category | Count | % |
|----------|-------|---|
| RISK | 100 | 45.5% |
| REGIME | 100 | 45.5% |
| SIGNAL | 20 | 9.0% |

**Sample Loss Evidence:**
```
"DOT/USDT: -86.26% (STOP_TOO_LOOSE)"
"TAO/USDT: -16.15% (STOP_TOO_LOOSE)"
"XRP/USDT: -7.90% (VOLATILITY_SHIFT)"
"ADA/USDT: -1.13% (FALSE_POSITIVE)"
```

**Issues:**
- Only contains LOSSES (no winning trades)
- Aggregated data, not individual trade-level
- Missing entry/exit timestamps and prices

---

### 3. Execution Integrity Log

**File:** `logs/execution_integrity.json`

**Contents:**
- 160 log entries
- **0 TRADE events** found
- Contains SYSTEM_START, ERROR, and other system events

**Issue:** Trade logging to integrity system is not enabled or not working.

---

### 4. Feature Datasets (DERIVED DATA)

**Files:**
- `datasets/rich_features.parquet` (155 rows, 14 columns)
- `datasets/engineered_features.parquet` (155 rows, 13 columns)
- `datasets/engineered_features_v2.parquet` (131 rows, 7 columns)
- `datasets/raw_trades_snapshot.parquet` (155 rows, 10 columns)

**Problem:** All feature datasets are derived from `STOP_PLACED` events only:
- 100% of rows are `STOP_PLACED` events
- 0% are actual trade entries/exits
- **64% have zero price movement** (price_next == price)

**Why ML Models Fail:**
The models are trained on stop-loss placement events where:
- Price often doesn't move between consecutive events
- Target variable is mostly zero (no prediction signal)
- Models learn to predict the mean (-0.002) instead of patterns

---

## Root Cause Analysis

### Why No ENTRY/EXIT in Feature Datasets?

The `export_trades.py` script extracts data from logs using regex patterns:

```python
entry_re = re.compile(r"\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*ENTRY (?P<symbol>[^:]+): (?P<qty>[\d\.eE+-]+) @ (?P<price>[\d\.eE+-]+)")
exit_re = re.compile(r"\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*(EXIT|EXITED) (?P<symbol>[^:]+): (?P<qty>[\d\.eE+-]+) @ (?P<price>[\d\.eE+-]+)")
```

**Problem:** The actual log format is DIFFERENT:
- ENTRY: `[...] [TraderNexus] 🎯 EXECUTING ENTRY (Attempt 1/2): SYMBOL (Qty: X, Lev: Yx)`
- EXIT: `[...] [GovernorAgent] 📉 Trade Logged: SYMBOL Profit: Z%`

The regex expects `ENTRY SYMBOL: QTY @ PRICE` but the actual format is `EXECUTING ENTRY (Attempt X/Y): SYMBOL (Qty: Z, Lev: Wx)`.

---

## Recommendations

### Immediate Actions

1. **Fix `export_trades.py` regex patterns** to match actual log format:
   ```python
   entry_re = re.compile(r"\[(?P<ts>[^\]]+)\].*EXECUTING ENTRY.*?:\s*(?P<symbol>\S+)\s*\(Qty:\s*(?P<qty>[\d\.eE+-]+),\s*Lev:\s*(?P<lev>[\d\.]+)x\)")
   exit_re = re.compile(r"\[(?P<ts>[^\]]+)\].*Trade Logged:\s*(?P<symbol>\S+)\s*Profit:\s*(?P<pnl>[-\d\.]+)%")
   ```

2. **Add entry/exit price logging** to TraderNexus and GovernorAgent:
   - ENTRY should log: symbol, qty, leverage, **entry_price**, timestamp
   - EXIT should log: symbol, **exit_price**, **entry_price**, pnl%, pnl_usd, timestamp

3. **Enable integrity trade logging** in the execution pipeline

### Data Collection Improvements

4. **Create dedicated trade journal** (JSON/Parquet):
   - One row per complete trade (entry + exit combined)
   - Include: entry_time, exit_time, symbol, direction, qty, leverage, entry_price, exit_price, pnl_usd, pnl_percent, exit_reason

5. **Add feature computation at entry time**:
   - Capture market state (volatility, regime, etc.) at entry
   - Store as features for ML training

6. **Increase trading frequency**:
   - Only 13 entries across 11 log files is very low
   - Need more trades for statistical significance

### ML Pipeline Fixes

7. **Retrain with matched trades only**:
   - Filter out STOP_PLACED events
   - Use only complete ENTRY→EXIT pairs
   - Target: `pnl_percent` or `pnl_direction` (classification)

8. **Use quant_ops loss data for validation**:
   - Compare model predictions against actual loss patterns
   - Validate that high-risk predictions align with RISK category losses

---

## Newly Created Datasets

| File | Rows | Description |
|------|------|-------------|
| `datasets/all_entry_events.parquet` | 13 | All ENTRY events from logs |
| `datasets/all_exit_events.parquet` | 12 | All EXIT events from logs |
| `datasets/complete_trades.parquet` | 3 | Matched ENTRY→EXIT trades |
| `datasets/quant_ops_trades.parquet` | 240 | Loss attribution from quant_ops |
| `datasets/ml_training_features.parquet` | 3 | Features matched to trades |

---

## Next Steps

1. **Update `export_trades.py`** with correct regex patterns
2. **Re-run feature extraction** to create proper trade dataset
3. **Add price logging** to entry/exit handlers
4. **Collect more live trading data** (need 100+ complete trades)
5. **Retrain models** with corrected dataset

---

## Contact

For questions about this audit, review:
- `scripts/aggregate_all_trades.py` - Data extraction logic
- `scripts/create_complete_trades.py` - Trade matching logic
- `_count_all_events.py` - Event counting script
