# HolonicTrader Project Cleanup Plan
**Updated:** 2025-12-23 (Post Phase 10-11 Improvements)

---

## File Dependency Analysis

### 🎯 CORE PRODUCTION FILES (KEEP - 25 files)

#### Holonic Framework (HolonicTrader/)
- `holon_core.py` - Base classes (Holon, Disposition, Message)
- `agent_trader.py` - Supra-Holon orchestrator
- `agent_observer.py` - Data fetching (hybrid local + live)
- `agent_entropy.py` - Regime detection (Phase 10 recalibrated)
- `agent_strategy.py` - Signal generation (Phase 11 enhanced exits)
- `agent_governor.py` - Risk management
- `agent_executor.py` - Trade execution (Phase 11: PnL tracking, sigmoid fix)
- `agent_actuator.py` - Order placement
- `agent_dqn.py` - Deep Q-Learning agent
- `kalman.py` - Kalman filter for trend estimation

#### Configuration & Infrastructure
- `config.py` - System parameters
- `database_manager.py` - SQLite persistence (Phase 11: unrealized PnL columns)
- `performance_tracker.py` - Performance metrics for GUI

#### Production Entry Points
- `main_live_phase4.py` - Live trading loop (ACTIVE)
- `run_backtest.py` - Backtest simulations
- `dashboard_gui.py` - GUI control panel (ACTIVE)

#### Models & State
- `lstm_model.keras` - LSTM brain
- `scaler.pkl` - Data scaler
- `dqn_model.keras` - DQN model
- `holonic_trader.db` - State persistence (2.9 MB)
- `market_data/` - Historical CSV data (5 assets)

---

### 🧪 VALIDATION & ANALYSIS SCRIPTS (KEEP - 15 files)

#### Phase 10-11 Validation Scripts (NEW)
- `analyze_live_entropy.py` - Entropy distribution analysis
- `validate_ledger_logic.py` - HALT/REDUCE trigger validation
- `validate_thresholds.py` - Entropy threshold validation
- `test_sigmoid.py` - Sigmoid function testing
- `performance_analysis.py` - Comprehensive PnL analysis
- `perf_summary.py` - Quick performance summary
- `system_health_check.py` - System health diagnostics

#### Core Test Files
- `test_communication.py` - Message protocol tests
- `test_multi_asset.py` - Multi-asset loop tests
- `test_hybrid_data.py` - Data loading tests
- `test_live_loop.py` - Live loop tests
- `test_entropy.py` - Entropy calculation tests
- `test_executor.py` - Executor logic tests
- `test_observer.py` - Observer data fetching tests

#### Utility Scripts
- `verify_db.py` - Database verification
- `check_schema.py` - Schema inspection

---

### ⚠️ OBSOLETE/REDUNDANT FILES (SAFE TO DELETE - 15+ files)

#### One-Time Analysis Scripts
- ❌ `get_thresholds.py` - One-time threshold extraction (Phase 10)
- ❌ `quick_db_check.py` - Replaced by system_health_check.py
- ❌ `check_portfolio_state.py` - Replaced by performance_analysis.py
- ❌ `calc_liquidation_value.py` - One-time calculation
- ❌ `project_returns.py` - One-time analysis
- ❌ `extract_pdf.py` - One-time PDF extraction
- ❌ `verify_db_schema.py` - Replaced by check_schema.py
- ❌ `verify_phase2.py` - Old phase verification
- ❌ `verify_trend_decay.py` - Old verification
- ❌ `verify_math_improvements.py` - Old verification
- ❌ `validate_dqn_policy.py` - One-time DQN validation
- ❌ `test_observer_latency.py` - One-time latency test
- ❌ `test_warp_speed.py` - One-time performance test
- ❌ `test_db_persistence.py` - Covered by test_executor.py

#### Generated Output Files (CAN DELETE)
- ❌ `*.log` - Log files (5 files, ~3MB total)
- ❌ `*.txt` - Output reports (can regenerate)
- ❌ `overnight.txt` - Duplicate of log file
- ❌ `health_report.txt` - Can regenerate
- ❌ `performance_report.txt` - Can regenerate
- ❌ `dqn_validation_output.txt` - Old validation

#### Cleanup Script
- ⚠️ `cleanup_project.py` - Review before deleting (may be useful)

---

## CLEANUP ACTIONS

### Phase 1: Safe to Delete Immediately (20 files)

**One-Time Scripts:**
```bash
get_thresholds.py
quick_db_check.py
check_portfolio_state.py
calc_liquidation_value.py
project_returns.py
extract_pdf.py
verify_db_schema.py
verify_phase2.py
verify_trend_decay.py
verify_math_improvements.py
validate_dqn_policy.py
test_observer_latency.py
test_warp_speed.py
test_db_persistence.py
```

**Generated Output Files:**
```bash
*.log (5 files)
overnight.txt
health_report.txt
performance_report.txt
dqn_validation_output.txt
```

### Phase 2: Archive for Reference (Keep in archive/)

**Documentation:**
```bash
Academic_White_Paper_on_AEHML_Framework-1.pdf
white_paper_full.txt
CLEANUP_PLAN.md (this file)
```

### Phase 3: Keep (Core System - 40 files)

**Production Code:**
```
HolonicTrader/ (10 agent files)
config.py
database_manager.py
performance_tracker.py
main_live_phase4.py
run_backtest.py
dashboard_gui.py
```

**Models & Data:**
```
lstm_model.keras
scaler.pkl
dqn_model.keras
holonic_trader.db
market_data/ (5 CSV files)
```

**Validation & Testing:**
```
analyze_live_entropy.py
validate_ledger_logic.py
validate_thresholds.py
test_sigmoid.py
performance_analysis.py
perf_summary.py
system_health_check.py
test_communication.py
test_multi_asset.py
test_hybrid_data.py
test_live_loop.py
test_entropy.py
test_executor.py
test_observer.py
verify_db.py
check_schema.py
```

---

## Estimated Cleanup Impact

- **Current:** 55 files + 7 directories (~10 MB)
- **After Phase 1 Cleanup:** 35 files (~7 MB)
- **Disk Space Saved:** ~3 MB (mostly logs)
- **Clarity Improvement:** Remove 20 obsolete files

---

## Recommended Cleanup Command

```bash
# Create archive directory
mkdir archive

# Move documentation
mv Academic_White_Paper_on_AEHML_Framework-1.pdf archive/
mv white_paper_full.txt archive/
mv CLEANUP_PLAN.md archive/

# Delete one-time scripts
rm get_thresholds.py quick_db_check.py check_portfolio_state.py
rm calc_liquidation_value.py project_returns.py extract_pdf.py
rm verify_db_schema.py verify_phase2.py verify_trend_decay.py
rm verify_math_improvements.py validate_dqn_policy.py
rm test_observer_latency.py test_warp_speed.py test_db_persistence.py

# Delete generated output files
rm *.log
rm overnight.txt health_report.txt performance_report.txt
rm dqn_validation_output.txt

# Review cleanup_project.py before deleting
# rm cleanup_project.py
```

---

## Post-Cleanup File Structure

```
DEV_SPACE/
├── HolonicTrader/          # 10 agent files
├── market_data/            # 5 CSV files
├── archive/                # Documentation
├── config.py
├── database_manager.py
├── performance_tracker.py
├── main_live_phase4.py
├── run_backtest.py
├── dashboard_gui.py
├── holonic_trader.db
├── *.keras (3 models)
├── scaler.pkl
├── test_*.py (8 test files)
├── validate_*.py (3 validation files)
├── analyze_live_entropy.py
├── performance_analysis.py
├── perf_summary.py
├── system_health_check.py
├── verify_db.py
├── check_schema.py
└── requirements.txt
```

**Total:** ~35 essential files

