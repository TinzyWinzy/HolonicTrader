# HolonicTrader: AEHML Framework Proof of Concept

**Autonomous Entropy-Holonic Machine Learning (AEHML) Framework**  
A production-ready cryptocurrency trading system demonstrating holonic architecture, entropy-based regime detection, and adaptive autonomous agents.

[![Status](https://img.shields.io/badge/status-live-success)](https://github.com/TinzyWinzy/HolonicTrader)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 🎯 What is AEHML?

**AEHML (Autonomous Entropy-Holonic Machine Learning)** is a novel framework combining:
- **Holonic Architecture**: Self-organizing agents with dual autonomy/integration properties
- **Entropy Analysis**: Market regime detection using Shannon entropy
- **Sovereign Strategy (Monolith-V5)**: A PPO (Proximal Policy Optimization) brain orchestrates global risk by interpreting market entropy against portfolio health (drawdown, margin).
- **Immutable Ledger**: Blockchain-inspired audit trail for all decisions

**HolonicTrader** is the first production implementation, proving AEHML's viability in real-world financial markets.

---

## 🏗️ Architecture

### Holonic Agent System

```
┌─────────────────────────────────────────────────────────┐
│                    TraderHolon (Nexus)                  │
│                  Supra-Holon Orchestrator               │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌──────▼──────┐  ┌────────▼────────┐
│  ObserverHolon │  │EntropyHolon │  │  EntryOracle    │
│  Live/Local    [Observer] -> [Global State] -> [PPO Sovereign Brain (Monolith-V5)]
                                               |
                                        [EntryOracle (Monolith-V4)]
  │
└────────────────┘  └─────────────┘  └─────────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌──────▼──────┐  ┌────────▼────────┐
│ GovernorHolon  │  │ExecutorHolon│  │ ActuatorHolon   │
│Risk Management │  │Decision/PnL │  │Order Placement  │
└────────────────┘  └─────────────┘  └─────────────────┘
                            │
                    ┌───────▼────────┐
                    │  DQN Brain     │
                    │ Pre-trained RL │
                    └────────────────┘
```

### Core Components

| Component | Purpose | Key Features |
|:---|:---|:---|
| **ObserverHolon** | Market data acquisition | 1000-candle live sync, hybrid local CSV support |
| **EntropyHolon** | Regime classification | Shannon entropy, calibrated thresholds (0.67/0.80) |
| **EntryOracle** | Trend Prediction | **Monolith-V4**: Stacked Holon (LSTM -> XGBoost) |
| **GovernorHolon** | Risk Management | Kelly Criterion, Volatility Scaling, Conviction |
| **ExecutorHolon** | Trade Execution | Sigmoid-based autonomy, immutable audit ledger |
| **ActuatorHolon** | Order placement | Maker-only execution, post-only limit orders |
| **DQN Brain** | Deep RL Policy | **Pre-trained** on 5,352 historical experiences |

---

## 📊 Performance & Benchmarking

### System Performance (Phase 20)

| Metric | Value | Notes |
|:---|---:|:---|
| **Mean Latency** | **73.46 ms** | Cycle speed: Sense -> Think -> Govern -> Act |
| **DQN Experiences** | 5,352 | Pre-trained across 16 allowed assets |
| **Win Rate (Baseline)**| 34.7% | Based on sector-wide historical sweep |
| **Asset Universe** | 16 | BTC, Alts, Memes, Metals (PAXG) |
| **Regime Accuracy** | High | Calibrated on 10,000+ live data points |

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.10+
Virtual environment (highly recommended)
Kraken/KuCoin API keys (for live trading)
```

### Installation & Run

```bash
# Setup
git clone https://github.com/TinzyWinzy/HolonicTrader.git
cd HolonicTrader
python -m venv .venv
.\.venv\Scripts\activate

# Run Dashboard (Recommended)
python dashboard_gui.py
```

---

## 🔬 Key Innovations

### 1. Pre-trained DQN Brain 🧠
The reinforcement learning agent is no longer starting from scratch. It has been pre-trained on a diverse dataset of 16 assets, understanding how to adjust actions based on entropy and volatility before the first live trade.

### 2. Monolith-V2 Oracle 🛰️
Combines deep learning (LSTM) for direction with signal processing (Kalman Filters) for price estimation, offering a high-conviction entry logic that accounts for market noise.

### 3. Institutional Risk Management 🛡️
Features a "Multi-Brain" governance stack including **Half-Kelly** position sizing, **Volatility Scaling**, and **LSTM-based Conviction** to protect principal while maximizing trend capture.

---

## 📁 Project Structure

```
HolonicTrader/
├── HolonicTrader/           # Core agent implementations
│   ├── holon_core.py        # Base Holon architecture
│   ├── agent_trader.py      # Supra-Holon (The Nexus)
│   ├── agent_oracle.py      # Monolith-V2 Oracle
│   ├── agent_governor.py    # Institutional Risk Mgmt
│   └── ...                  # Other specialized holons
├── tests/                   # Unit & Integration tests
├── benchmarks/              # Latency and PnL audit tools
├── archive/                 # Legacy plans & documentation
├── research/                # Strategy drafts & analysis
├── config.py                # System hyperparameters
├── dashboard_gui.py         # Tkinter-based control panel
└── holonic_trader.db        # SQLite State & Audit Ledger
```

---

## 📈 Recent Improvements (Phases 15-18)

### Phase 15-16: Optimization
- ✅ Unified Scavenger/Predator stop-loss thresholds.
- ✅ Implemented direction-aware PnL for short-selling.
- ✅ Optimized cycle latency to <80ms.

### Phase 17: Pre-training
- ✅ Downloaded historical data for all 16 `ALLOWED_ASSETS`.
- ✅ Pre-trained DQN model on 5,300+ market experiences.

### Phase 18: GUI Bridge
- ✅ Connected Dashboard settings (Leverage, Allocation) to Live Bot.
- ✅ Added real-time Exposure, Margin, and Portfolio Leverage tracking.

---

## 🎓 Academic Foundation

Based on the **AEHML Framework White Paper**, this implementation demonstrates:

1. **Holonic Principles**: Arthur Koestler's concept of holons applied to ML agents
2. **Entropy Theory**: Claude Shannon's information theory for market analysis
3. **Adaptive Systems**: Self-organizing, self-regulating autonomous agents
4. **Blockchain Integration**: Immutable decision logging for transparency

**Reference**: See `archive/Academic_White_Paper_on_AEHML_Framework-1.pdf`

---

## 🛠️ Technology Stack

- **Python 3.10+**: Core language
- **TensorFlow/Keras**: LSTM neural networks
- **NumPy**: Deep Q-Learning implementation
- **CCXT**: Exchange connectivity
- **SQLite**: State persistence
- **Tkinter**: GUI dashboard
- **Pandas**: Data manipulation

---

## 📊 Monitoring & Observability

### Real-Time Dashboard
- Live portfolio value
- Regime detection status
- Agent health metrics
- Trade execution log
- Performance analytics

### Database Schema
```sql
ledger          -- Immutable decision log (9,956 entries)
trades          -- Execution history with PnL
portfolio       -- Current holdings and balance
rl_experiences  -- DQN training data
```

---

## 🔮 Future Enhancements

- [ ] Multi-exchange support (Binance, Coinbase)
- [ ] Advanced RL algorithms (PPO, A3C)
- [ ] Sentiment analysis integration
- [ ] Portfolio optimization (Markowitz)
- [ ] Real-time risk metrics (VaR, Sharpe)
- [ ] Web-based dashboard
- [ ] Backtesting framework enhancements

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- New holonic agents (e.g., SentimentHolon)
- Alternative entropy measures (Rényi, Tsallis)
- Strategy improvements
- Performance optimizations
- Documentation enhancements

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

- **AEHML Framework**: Original theoretical foundation
- **Arthur Koestler**: Holonic systems theory
- **Claude Shannon**: Information theory and entropy
- **Community**: Open-source contributors and testers

---

## 📞 Contact

**Project**: [HolonicTrader](https://github.com/TinzyWinzy/HolonicTrader)  
**Issues**: [GitHub Issues](https://github.com/TinzyWinzy/HolonicTrader/issues)

---

**⚠️ Disclaimer**: This is a proof-of-concept implementation for research and educational purposes. Cryptocurrency trading involves substantial risk. Always test thoroughly in paper trading mode before live deployment.
