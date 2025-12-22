# HolonicTrader Project Cleanup Plan

## File Dependency Analysis

### 🎯 CORE PRODUCTION FILES (KEEP)

#### Holonic Framework
- `HolonicTrader/holon_core.py` - Base classes (Holon, Disposition, Message)
- `HolonicTrader/agent_trader.py` - Supra-Holon orchestrator

#### Active Agents
- `agent_observer.py` - Data fetching (hybrid local + live)
- `agent_entropy.py` - Regime detection
- `agent_strategy.py` - Signal generation (RSI, OBV, LSTM)
- `agent_governor.py` - Risk management
- `agent_executor.py` - Trade execution
- `agent_actuator.py` - Order placement

#### Configuration & Data
- `config.py` - System parameters
- `market_data/` - Historical CSV data (5 assets)

#### Production Entry Points
- `main_live_phase4.py` - Live trading loop (ACTIVELY USING)
- `run_backtest.py` - Backtest simulations (JUST CREATED)

#### Models & State
- `lstm_model.keras` - LSTM brain
- `scaler.pkl` - Data scaler
- `dqn_model.keras` - DQN model
- `holonic_trader.db` - State persistence

---

### 🧪 TEST FILES (KEEP - for validation)

- `test_communication.py` - Message protocol tests
- `test_multi_asset.py` - Multi-asset loop tests
- `test_hybrid_data.py` - Data loading tests
- `test_live_loop.py` - Live loop tests

---

### ⚠️ OBSOLETE/REDUNDANT FILES (CANDIDATES FOR REMOVAL)

#### Duplicate/Old Entry Points
- `main_simulation.py` - ❌ Replaced by run_backtest.py
- `main_backtest.py` - ❌ Replaced by run_backtest.py
- `main_live.py` - ❌ Replaced by main_live_phase4.py
- `nexus.py` - ❌ Old version, superseded by agent_trader.py
- `nexus_live.py` - ❌ Old live version
- `main_micro.py` - ❌ Micro-optimization experiment

#### Unused Agents
- `agent_sensor.py` - ❌ Duplicate of agent_observer.py
- `agent_rl.py` - ❌ Old RL agent, replaced by agent_dqn.py
- `agent_dqn.py` - ⚠️ KEEP IF USED, otherwise remove

#### Old Optimization/Analysis Scripts
- `optimize_nexus.py` - ❌ One-time optimization
- `calibrate_entropy.py` - ❌ One-time calibration
- `analyze_pareto.py` - ❌ One-time analysis
- `compare_compounding.py` - ❌ One-time comparison
- `benchmark_assets.py` - ❌ One-time benchmark
- `tune_dqn.py` - ❌ One-time tuning

#### Old Test Files
- `test_entropy.py` - ⚠️ Check if still needed
- `test_executor.py` - ⚠️ Check if still needed
- `test_observer.py` - ⚠️ Check if still needed
- `test_micro.py` - ❌ Related to obsolete main_micro.py
- `test_predator.py` - ⚠️ May be obsolete
- `verify_strategy.py` - ❌ One-time verification

#### Training Scripts
- `train_lstm.py` - ❌ LSTM already trained
- `fetch_history.py` - ❌ Data already fetched
- `fetch_multi.py` - ❌ Data already fetched

#### Utility Scripts
- `read_whitepaper.py` - ❌ One-time use
- `dashboard_gui.py` - ⚠️ Keep if you want GUI
- `database_manager.py` - ⚠️ Check if used by executor

#### Generated Files (Can Delete)
- `*.png` - Result visualizations (can regenerate)
- `*.csv` - Result CSVs (can regenerate)
- `*.json` - Result JSONs (except brain_memory, q_table if needed)
- `*.log` - Logs (can regenerate)

---

## CLEANUP ACTIONS

### Safe to Delete (30+ files):
```
main_simulation.py
main_backtest.py  
main_live.py
nexus.py
nexus_live.py
main_micro.py
agent_sensor.py
agent_rl.py
optimize_nexus.py
calibrate_entropy.py
analyze_pareto.py
compare_compounding.py
benchmark_assets.py
tune_dqn.py
test_micro.py
train_lstm.py
fetch_history.py
fetch_multi.py
read_whitepaper.py
verify_strategy.py
*.png (result images)
*.csv (except in market_data/)
paper_trading.log
```

### Keep (Core System - ~20 files):
```
HolonicTrader/
config.py
agent_*.py (observer, entropy, strategy, governor, executor, actuator)
main_live_phase4.py
run_backtest.py
test_communication.py
test_multi_asset.py
test_hybrid_data.py
test_live_loop.py
lstm_model.keras
scaler.pkl
holonic_trader.db
market_data/
```

### Review Before Deleting:
- `agent_dqn.py` + `dqn_model.keras` - Are you using DQN?
- `dashboard_gui.py` - Do you want a GUI?
- `database_manager.py` - Is it used by executor?
- Old test files - Do they have unique test cases?

---

## Estimated Cleanup Impact
- **Current:** 61 files + directories
- **After Cleanup:** ~25-30 core files
- **Disk Space Saved:** ~50MB+ (mostly images/logs)
