# HolonicTrader — Copilot Instructions

## Architecture Overview

HolonicTrader is an **AEHML (Autonomous Entropy-Holonic Machine Learning)** crypto trading system on Kraken Futures. It uses a holonic agent hierarchy where autonomous agents (Holons) communicate via `Message` dataclasses and are orchestrated by `TraderHolon` ("TraderNexus").

**Execution flow:** `main_live_phase4.py` → instantiates all Holons → `TraderHolon.start_live_loop()` runs the cycle: Observer (data) → Entropy (regime) → Oracle (signal) → Governor (risk/sizing) → Executor (ledger) → Actuator (exchange orders).

### Key Packages & Directories

| Path | Purpose |
|------|---------|
| `HolonicTrader/` | Core agent package — all Holon classes (`agent_*.py`), handlers, SMCE engine |
| `HolonicTrader/holon_core.py` | ABC `Holon` base, `Message`, `Disposition`, `PositionState` enums |
| `config.py` (root) | **All** runtime parameters — flat module, `import config` everywhere |
| `rust_engine/` | PyO3 Rust crate (`holonic_speed`) — optional accelerator |
| `core/scouts/` | Entropy scouter subsystem |
| `atlas_*.py` (root) | ATLAS Profit Architect — standalone integration, own JSON config |
| `tests/` | Pytest suite (`pytest.ini` at root) |

## Coding Conventions

### Creating a New Holon Agent

1. File: `HolonicTrader/agent_<name>.py`
2. Class: `<Name>Holon(Holon)` — inherit from `HolonicTrader.holon_core.Holon`
3. Constructor: `def __init__(self, name="<Name>Agent", ...)` → call `super().__init__(name=name, disposition=Disposition(autonomy=0.9, integration=0.9))`
4. Must implement `receive_message(self, sender, content)` (ABC requirement)
5. Wire into `TraderHolon` via `sub_holons` dict in `main_live_phase4.py`

### Config Pattern

- All parameters live in `config.py` — flat constants, no class wrapper
- Secrets: `.env` file loaded via `dotenv` (`KRAKEN_FUTURES_API_KEY`, etc.)
- Agents read config with `import config; config.SOME_PARAM`
- Config is **mutated at runtime** (capital sync, GUI overrides) — this is intentional
- Per-asset dicts keyed by base symbol: `MIN_TRADE_QTY['BTC']`, `SYMBOL_MAP['BTC/USDT']`
- Tier-based risk limits: `SMCE_POSITION_LIMITS['NANO']`, `CAPITAL_TIERS['SMALL']`

### Rust Engine (`holonic_speed`)

Always wrap in try/except with a Python fallback:
```python
try:
    import holonic_speed
    result = holonic_speed.calculate_shannon_entropy(data)
except ImportError:
    result = _python_fallback(data)
```
Build: `cd rust_engine; maturin build --release` → produces `.pyd` at project root.

### Trade Signal Pipeline

Entry/exit logic uses **free functions** (not methods) extracted from TraderHolon:
- `HolonicTrader/trader_entry_handler.py` → `handle_entry()`, `build_ppo_state()`
- `HolonicTrader/trader_exit_handler.py` → `handle_exit()`, `determine_exit_signal()`

Key dataclasses: `TradeSignal`, `TradeDecision`, `Position` (all in `agent_executor.py`).

### SMCE (Sovereign Monte Carlo Engine) — "Constitutional" Risk

SMCE modules in `HolonicTrader/smce_*.py` enforce **non-overridable** risk limits:
- Tiered by equity: `MICRO` (<$200), `SMALL` (<$500), `MEDIUM` (<$5k)
- Drawdown doctrine: 3% daily / 6% weekly → `DEFENSIVE` mode (48h cooldown)
- These rules are "Layer 0" — no strategy can bypass them

### Integration Modules (Root-Level)

Standalone systems at project root with own JSON configs (not Holon subclasses):
- `atlas_integration.py` + `atlas_profit_config.json` — profit filtering
- `atlas_capital_manager.py` — capital allocation
- `wfo_engine.py` — Walk-Forward Optimization (background thread)
- `database_manager.py` — SQLite persistence (`holonic_trader.db`)

## Developer Workflows

```powershell
# Activate venv and run bot
& .\.venv313\Scripts\Activate.ps1
python main_live_phase4.py          # Live/Paper based on config.PAPER_TRADING

# Run tests
pytest tests/ -v --tb=short

# GUI dashboard (Tkinter)
python dashboard_gui.py

# Build Rust engine
cd rust_engine; maturin build --release

# Check latest log
Get-ChildItem live_trading_session_*.log | Sort-Object LastWriteTime | Select-Object -Last 1 | Get-Content -Tail 50
```

## Critical Safety Rules

- **Never remove or weaken** SMCE drawdown limits, MonitorHolon kill-switch, or Governor veto logic
- Drawdown baselines are **reset on every startup** to prevent false FEVER lockdowns from stale state
- `config.PAPER_TRADING = False` means **real money** — double-check before any config change
- The `ExecutorHolon` maintains a SHA-256 blockchain ledger — preserve `AuditLedger` integrity
- UTF-8 logging is enforced globally (`utf8_logging.py`) — always use `get_logger()` not raw `logging`
