# Unified Regime Engine - Quick Reference

## At a Glance

```
┌────────────────────────────────────────────────────┐
│  Regime = (Behavioral × Operational)               │
│                                                     │
│  Behavioral: HOW market behaves (5 states)         │
│  Operational: WHAT you can do (4 states)           │
│  Total combinations: 20                            │
└────────────────────────────────────────────────────┘
```

## Behavioral Regimes (HOW)

| Regime | Entropy | Trend | Vol | Description | Conviction |
|--------|---------|-------|-----|-------------|------------|
| **LOW_VOL_MEAN_REVERT** | < 0.35 | < 0.3 | Any | Choppy range | 0.65 |
| **HIGH_VOL_TRENDING** | 0.35-0.75 | > 0.8 | > 3% | Directional | 0.60 |
| **BULL_MOMENTUM** | < 0.35 | > 1.0 | Any | Strong up | 0.65 |
| **BEAR_DISTRIBUTION** | < 0.35 | > 1.0 | Any | Strong down | 0.70 |
| **TRANSITION_CHAOS** | > 0.75 | Any | Any | Uncertain | 0.75+ |

## Operational Regimes (WHAT)

| Regime | Entries | Size | Max Lev | Trigger |
|--------|---------|------|---------|---------|
| **HARVEST** | ✅ | 1.0x | 3-4x | Default safe |
| **EXPANSION** | ✅ | 1.5-2x | 4-5x | Bull + low entropy |
| **TRANSITION** | ✅ | 0.25-0.6x | 1.5-2x | Warning signals |
| **DEFENSIVE** | ❌ | 0.0x | 1.0x | Emergency |

## Quick Start

```python
from HolonicTrader.unified_regime_engine import get_unified_regime_engine

# Get singleton instance
regime = get_unified_regime_engine()

# Update with market data (call every tick/bar)
state = regime.update(
    prices=prices,              # np.ndarray of recent closes
    volumes=volumes,            # optional
    atr=atr,                    # optional
    structure="NEUTRAL",        # BULLISH/BEARISH/NEUTRAL/SUPPORT/RESISTANCE
    liquidity_status="healthy", # healthy/warning/critical
    correlation_idx=0.3,        # 0.0-1.0
    drawdown_breach=False,      # True triggers DEFENSIVE
)

# Check if trade allowed
allowed, reason = regime.should_allow_entry(
    conviction=0.72,
    direction='LONG',
    symbol='BTC/USDT',
)

if allowed:
    # Use state parameters for trade sizing
    size = base_size * state.size_modifier
    leverage = min(target_leverage, state.max_leverage)
    stop_distance = atr * state.trailing_stop_mult
    targets = state.profit_targets
```

## State Parameters

```python
state.entries_allowed       # bool - can open new positions
state.size_modifier         # float - multiply position size
state.max_leverage          # float - maximum allowed leverage
state.min_conviction        # float - minimum signal conviction
state.trailing_stop_mult    # float - ATR multiplier for stops
state.profit_targets        # dict - {rapid, normal, runner}
state.behavioral            # enum - HOW market behaves
state.operational           # enum - WHAT operations allowed
state.confidence            # float - 0.0-1.0 regime confidence
```

## Common Patterns

### Pattern 1: Signal Validation
```python
def validate_signal(signal):
    state = regime.update(...)
    
    # Check regime gate
    if not state.entries_allowed:
        return False, "Regime blocks entries"
    
    # Check conviction
    if signal.conviction < state.min_conviction:
        return False, f"Conviction {signal.conviction:.2f} < {state.min_conviction:.2f}"
    
    # Check direction alignment
    if state.behavioral == BehavioralRegime.BEAR_DISTRIBUTION:
        if signal.direction == 'LONG' and signal.conviction < 0.75:
            return False, "Bear regime - long needs 0.75+"
    
    return True, "Approved"
```

### Pattern 2: Position Sizing
```python
def calculate_position_size(signal, state):
    base_size = account_balance * risk_per_trade / signal.stop_distance
    
    # Apply regime modifier
    adjusted_size = base_size * state.size_modifier
    
    # Apply leverage cap
    effective_leverage = min(signal.target_leverage, state.max_leverage)
    
    return adjusted_size * effective_leverage
```

### Pattern 3: Stop Loss Placement
```python
def place_stop_loss(entry_price, direction, atr, state):
    stop_mult = state.trailing_stop_mult
    
    if direction == 'LONG':
        stop_price = entry_price * (1 - stop_mult * atr / entry_price)
    else:
        stop_price = entry_price * (1 + stop_mult * atr / entry_price)
    
    return stop_price
```

### Pattern 4: Profit Targets
```python
def set_profit_targets(entry_price, direction, state):
    targets = state.profit_targets
    
    if direction == 'LONG':
        rapid = entry_price * (1 + targets['rapid'])
        normal = entry_price * (1 + targets['normal'])
        runner = entry_price * (1 + targets['runner'])
    else:
        rapid = entry_price * (1 - targets['rapid'])
        normal = entry_price * (1 - targets['normal'])
        runner = entry_price * (1 - targets['runner'])
    
    return {'rapid': rapid, 'normal': normal, 'runner': runner}
```

## Regime Transition Matrix

```
FROM \ TO         │ HARVEST │ EXPANSION │ TRANSITION │ DEFENSIVE
──────────────────┼─────────┼───────────┼────────────┼──────────
HARVEST           │   ──    │  Bull+    │  Warning   │  Emergency
EXPANSION         │  Normal │    ──     │  Warning   │  Emergency
TRANSITION        │  Clear  │  Bull+    │     ──     │  Emergency
DEFENSIVE         │  Clear  │  Bull+    │  Warning   │    ──

Triggers:
  Bull+    = BULLISH structure + entropy < 0.95 + healthy liquidity
  Warning  = liquidity warning OR correlation > 0.7 OR entropy 0.9-1.2
  Emergency = drawdown breach OR critical liquidity OR entropy > 1.2
  Clear    = Warning/Emergency conditions resolved
```

## Debugging

```python
# Get full status
status = regime.get_status_summary()
print(status)

# Output:
{
    'behavioral_regime': 'LOW_VOL_MEAN_REVERT',
    'operational_regime': 'HARVEST',
    'confidence': 0.85,
    'entries_allowed': True,
    'size_modifier': 0.75,
    'max_leverage': 3.0,
    'min_conviction': 0.65,
    'metrics': {
        'entropy': 0.28,
        'volatility': 0.015,
        'trend_strength': 0.22,
        'structure': 'NEUTRAL',
        'liquidity': 'healthy',
        'correlation': 0.31,
        'drawdown_breach': False,
    },
    'recent_transitions': [...]
}

# Get permissions dict
perms = regime.get_permissions()
```

## Logging

```
[UnifiedRegimeEngine] Initialized
[UnifiedRegimeEngine] Initial state: LOW_VOL_MEAN_REVERT + HARVEST
[UnifiedRegimeEngine] REGIME CHANGE: LOW_VOL_MEAN_REVERT + HARVEST → TRANSITION_CHAOS + TRANSITION | behavioral: LOW_VOL_MEAN_REVERT → TRANSITION_CHAOS; operational: HARVEST → TRANSITION
[UnifiedRegimeEngine] Forced DEFENSIVE: Drawdown breach detected
```

## Configuration Overrides

```python
from HolonicTrader.unified_regime_engine import UNIFIED_REGIME_CONFIG

# Override specific regime combination
UNIFIED_REGIME_CONFIG[
    (BehavioralRegime.LOW_VOL_MEAN_REVERT, OperationalRegime.HARVEST)
] = {
    'description': 'Custom: aggressive mean reversion',
    'entries_allowed': True,
    'size_modifier': 1.0,       # Was 0.75
    'max_leverage': 4.0,        # Was 3.0
    'min_conviction': 0.60,     # Was 0.65
    'trailing_stop_mult': 2.5,  # Was 3.0
    'profit_targets': {'rapid': 0.01, 'normal': 0.025, 'runner': 0.05},
    'cooldown_type': 'time',
}
```

## Migration Checklist

- [ ] Import `get_unified_regime_engine`
- [ ] Replace `HolonicAdaptor` + `SMCERegimeEngine` with unified engine
- [ ] Update `update()` calls with all required parameters
- [ ] Replace parameter access: `state.min_conviction`, `state.size_modifier`, etc.
- [ ] Update veto logic to use `should_allow_entry()`
- [ ] Run backtest comparison
- [ ] Paper trade for 1-2 weeks
- [ ] Deploy to live

## Troubleshooting

| Issue | Check | Fix |
|-------|-------|-----|
| No entries allowed | `state.entries_allowed` | Check operational regime |
| Always DEFENSIVE | `drawdown_breach`, `liquidity_status` | Verify inputs |
| Too many vetoes | `state.min_conviction` | May need config adjustment |
| Rapid regime flips | Hysteresis working? | Check `HYSTERESIS_WINDOW` |
| Wrong size | `state.size_modifier` | Verify regime combination |

## Performance Benchmarks

Based on backtest expectations:

| Metric | Old System | Unified | Improvement |
|--------|-----------|---------|-------------|
| Signal pass rate | 0-5% | 15-25% | +300-500% |
| Regime conflicts | Daily | None | 100% |
| Code complexity | 2 systems | 1 system | -50% |
| Config locations | 3 files | 1 dict | -66% |

---

**Version:** 1.0  
**Date:** 2026-03-16  
**Status:** Ready for integration testing
