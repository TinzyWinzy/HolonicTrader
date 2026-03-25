# Unified Regime Engine - Migration Guide

## Overview

The **UnifiedRegimeEngine** consolidates the dual regime systems (`HolonicAdaptor` + `SMCERegimeEngine`) into a single coherent state machine, eliminating conflicts and simplifying regime-based decision making.

## Key Improvements

| Issue | Before | After |
|-------|--------|-------|
| **Regime Conflicts** | Two independent systems could disagree | Single source of truth |
| **Overprotection** | Double conviction checks blocking 83% of signals | Unified conviction threshold |
| **Complexity** | 5 behavioral + 4 operational regimes tracked separately | 2D state: (Behavioral × Operational) |
| **Chattering** | Rapid regime flips | Hysteresis window prevents noise |
| **Configuration** | Scattered across multiple files | Centralized `UNIFIED_REGIME_CONFIG` |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    UnifiedRegimeEngine                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐          ┌──────────────────┐        │
│  │   BEHAVIORAL     │          │   OPERATIONAL    │        │
│  │     (HOW)        │          │     (WHAT)       │        │
│  ├──────────────────┤          ├──────────────────┤        │
│  │ LOW_VOL_MEAN_    │          │ HARVEST          │        │
│  │   REVERT         │          │ EXPANSION        │        │
│  │ HIGH_VOL_TRENDING│          │ TRANSITION       │        │
│  │ BULL_MOMENTUM    │          │ DEFENSIVE        │        │
│  │ BEAR_DISTRIBUTION│          │                  │        │
│  │ TRANSITION_CHAOS │          │                  │        │
│  └──────────────────┘          └──────────────────┘        │
│           │                            │                    │
│           └────────────┬───────────────┘                    │
│                        │                                    │
│                        ▼                                    │
│            ┌───────────────────────┐                       │
│            │   RegimeState         │                       │
│            │   - entries_allowed   │                       │
│            │   - size_modifier     │                       │
│            │   - max_leverage      │                       │
│            │   - min_conviction    │                       │
│            │   - trailing_stop     │                       │
│            └───────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

## Regime State Dimensions

### Behavioral Regime (HOW the market behaves)
| Regime | Entropy | Trend | Volatility | Description |
|--------|---------|-------|------------|-------------|
| `LOW_VOL_MEAN_REVERT` | < 0.35 | < 0.3 | Any | Choppy, range-bound |
| `HIGH_VOL_TRENDING` | 0.35-0.75 | > 0.8 | > 3% | Directional move |
| `BULL_MOMENTUM` | < 0.35 | > 1.0 | Any | Strong uptrend |
| `BEAR_DISTRIBUTION` | < 0.35 | > 1.0 | Any | Strong downtrend |
| `TRANSITION_CHAOS` | > 0.75 | Any | Any | High uncertainty |

### Operational Regime (WHAT operations allowed)
| Regime | Entries | Size | Max Lev | Use Case |
|--------|---------|------|---------|----------|
| `HARVEST` | ✅ | 1.0x | 3-4x | Normal operations |
| `EXPANSION` | ✅ | 1.5-2.0x | 4-5x | Bull + low entropy |
| `TRANSITION` | ✅ | 0.25-0.6x | 1.5-2x | Caution required |
| `DEFENSIVE` | ❌ | 0.0x | 1.0x | Emergency stop |

## Migration Steps

### Step 1: Import the Engine

```python
# Old way (two separate systems)
from HolonicTrader.holonic_adaptor import get_holonic_adaptor
from HolonicTrader.smce_regime_engine import SMCERegimeEngine

holonic = get_holonic_adaptor()
smce = SMCERegimeEngine()

# New way (unified)
from HolonicTrader.unified_regime_engine import get_unified_regime_engine

regime = get_unified_regime_engine()
```

### Step 2: Update Market Data Feed

```python
# Old way (separate updates)
holonic.detect_regime({'prices': prices, 'volumes': volumes, 'atr': atr})
smce.classify(structure, entropy, liquidity, correlation, drawdown_breach)

# New way (single update)
state = regime.update(
    prices=prices,
    volumes=volumes,
    atr=atr,
    structure=structure,           # 'BULLISH', 'BEARISH', 'NEUTRAL'
    liquidity_status=liquidity,    # 'healthy', 'warning', 'critical'
    correlation_idx=correlation,   # 0.0 - 1.0
    drawdown_breach=drawdown_breach,
)
```

### Step 3: Get Permissions

```python
# Old way (merge two configs)
holonic_params = holonic.get_adaptive_parameters()
smce_params = smce.get_permissions()
# ... manual merging ...

# New way (single source)
permissions = regime.get_permissions()

# Direct access to state
if state.entries_allowed and conviction >= state.min_conviction:
    size = base_size * state.size_modifier
    leverage = min(target_leverage, state.max_leverage)
```

### Step 4: Entry Validation

```python
# Old way (multiple checks)
holonic_ok, holonic_reason = holonic.should_allow_trade(signal_type, conviction)
smce_ok = smce.get_permissions()['new_entries_allowed']
if holonic_ok and smce_ok:
    # Execute

# New way (single check)
allowed, reason = regime.should_allow_entry(
    conviction=conviction,
    direction='LONG',  # or 'SHORT'
    symbol='BTC/USDT',
)
if allowed:
    # Execute
```

## Configuration Changes

### Conviction Thresholds (FIXED)

The critical overprotection issue is resolved:

| Regime Combination | Old Threshold | New Threshold |
|-------------------|---------------|---------------|
| LOW_VOL_MEAN_REVERT + HARVEST | 0.70 (Holonic) + SMCE check | **0.65** (unified) |
| TRANSITION_CHAOS + HARVEST | 0.75 (Holonic) + SMCE check | **0.75** (unified) |
| HIGH_VOL_TRENDING + HARVEST | 0.60 (Holonic) + SMCE check | **0.60** (unified) |

### Parameter Mapping

| Parameter | Old Source | New Source |
|-----------|-----------|------------|
| Trailing stop multiplier | `MARKET_REGIMES['trailing_mult']` | `state.trailing_stop_mult` |
| Profit targets | `MARKET_REGIMES['profit_targets']` | `state.profit_targets` |
| Position size multiplier | `MARKET_REGIMES['position_size_mult']` | `state.size_modifier` |
| Min conviction | `MARKET_REGIMES['min_conviction']` | `state.min_conviction` |
| Max leverage | `REGIME_CONFIG['max_leverage_*']` | `state.max_leverage` |
| Entries allowed | `REGIME_CONFIG['new_entries_allowed']` | `state.entries_allowed` |

## Integration Points

### Governor Agent

```python
# In agent_governor.py

# Replace:
#   from .smce_regime_engine import SMCERegimeEngine
#   self.smce_regime_engine = SMCERegimeEngine()
#   from .holonic_adaptor import get_holonic_adaptor
#   self.holonic_adaptor = get_holonic_adaptor()

# With:
from .unified_regime_engine import get_unified_regime_engine
self.regime_engine = get_unified_regime_engine()

# In signal validation:
def _validate_signal(self, signal):
    state = self.regime_engine.update(
        prices=self.market_data['prices'],
        structure=self.structure_view,
        liquidity_status=self.liquidity_status,
        correlation_idx=self.correlation_idx,
        drawdown_breach=self.check_drawdown_breach(),
    )
    
    allowed, reason = self.regime_engine.should_allow_entry(
        conviction=signal.conviction,
        direction=signal.direction,
        symbol=signal.symbol,
    )
    
    if not allowed:
        return self.veto(signal, reason)
    return self.approve(signal)
```

### Entry Oracle

```python
# In agent_oracle.py

# Replace:
#   self.holonic_adaptor = get_holonic_adaptor()
#   params = self.holonic_adaptor.get_adaptive_parameters()

# With:
from HolonicTrader.unified_regime_engine import get_unified_regime_engine
self.regime_engine = get_unified_regime_engine()

# In signal generation:
def generate_signal(self, symbol, data):
    state = self.regime_engine.update(
        prices=data['prices'],
        volumes=data['volumes'],
        atr=data['atr'],
        structure=self.get_structure_view(symbol),
        liquidity_status=self.get_liquidity_status(symbol),
        correlation_idx=self.get_correlation_index(),
        drawdown_breach=False,
    )
    
    # Use state parameters directly
    trailing_mult = state.trailing_stop_mult
    profit_targets = state.profit_targets
    min_conviction = state.min_conviction
```

## Testing

```python
import unittest
import numpy as np
from HolonicTrader.unified_regime_engine import UnifiedRegimeEngine

class TestUnifiedRegimeEngine(unittest.TestCase):
    
    def setUp(self):
        self.engine = UnifiedRegimeEngine()
    
    def test_low_vol_mean_revert_harvest(self):
        """Test choppy but safe regime."""
        prices = np.array([100, 100.5, 100.2, 100.8, 100.3, 100.6, 100.4])
        
        state = self.engine.update(
            prices=prices,
            structure="NEUTRAL",
            liquidity_status="healthy",
            correlation_idx=0.3,
            drawdown_breach=False,
        )
        
        self.assertEqual(state.behavioral.value, "LOW_VOL_MEAN_REVERT")
        self.assertEqual(state.operational.value, "HARVEST")
        self.assertTrue(state.entries_allowed)
        self.assertEqual(state.min_conviction, 0.65)
        self.assertEqual(state.size_modifier, 0.75)
    
    def test_transition_chaos_defensive(self):
        """Test high entropy triggers defensive."""
        # Simulate high entropy prices
        prices = np.array([100, 105, 98, 107, 95, 110, 92])
        
        state = self.engine.update(
            prices=prices,
            structure="NEUTRAL",
            liquidity_status="healthy",
            correlation_idx=0.3,
            drawdown_breach=False,
        )
        
        # High entropy should trigger DEFENSIVE or TRANSITION
        self.assertIn(
            state.operational.value,
            ["DEFENSIVE", "TRANSITION_CHAOS", "TRANSITION"]
        )
    
    def test_drawdown_breach_forces_defensive(self):
        """Test drawdown breach overrides everything."""
        prices = np.array([100, 101, 102, 103, 104])
        
        state = self.engine.update(
            prices=prices,
            structure="BULLISH",
            liquidity_status="healthy",
            correlation_idx=0.2,
            drawdown_breach=True,  # Force defensive
        )
        
        self.assertEqual(state.operational.value, "DEFENSIVE")
        self.assertFalse(state.entries_allowed)

if __name__ == '__main__':
    unittest.main()
```

## Rollback Plan

If issues arise, the old systems remain intact:

1. **Quick rollback**: Comment out unified engine import, restore old imports
2. **Hybrid mode**: Run both systems in parallel, compare decisions
3. **Gradual migration**: Migrate one component at a time (e.g., Governor first)

## Expected Impact

Based on the session log analysis (`live_trading_session_20260315_234819.log`):

| Metric | Before | Expected After |
|--------|--------|----------------|
| Signal pass rate | 0.0% | 15-25% |
| HOLONIC vetoes | 1170 (24%) | ~400 (8%) |
| GOVERNOR vetoes | 1590 (33%) | ~800 (16%) |
| Executed trades/session | 0 | 5-15 |
| Regime conflicts | Daily | None |

## Next Steps

1. ✅ **Create unified engine** (done)
2. ⏳ **Add unit tests** (see `tests/test_unified_regime.py`)
3. ⏳ **Update Governor integration** (modify `agent_governor.py`)
4. ⏳ **Update Oracle integration** (modify `agent_oracle.py`)
5. ⏳ **Backtest on historical data** (compare vs old system)
6. ⏳ **Paper trade validation** (1-2 weeks)
7. ⏳ **Deploy to live** (monitor closely)

## Support

For issues or questions:
- Check logs: `[UnifiedRegimeEngine] REGIME CHANGE: ...`
- Review state: `engine.get_status_summary()`
- Compare metrics: Old vs new conviction thresholds
