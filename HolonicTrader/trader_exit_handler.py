"""
trader_exit_handler.py
Exit Logic Handler - Extracted from agent_trader.py

Handles:
- Exit transaction execution
- Guardian record keeping
- Memory replay storage
- PPO reward processing
- Telegram notifications
"""
import time
import numpy as np
from datetime import datetime
from . import config
from .trader_ppo_handler import calculate_ppo_reward


def _pa(obj, key, default=None):
    """Safe attribute/key accessor for both Position objects and legacy dicts."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _meta_copy(obj) -> dict:
    """Convert a Position object or dict to a plain dict (safe copy for downstream use)."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj.copy()
    # Position object — extract all public attributes into a plain dict
    return {k: v for k, v in vars(obj).items() if not k.startswith('_')}


def handle_exit(
    symbol: str,
    final_exit,  # TradeSignal
    reason: str,
    executor,
    guardian,
    ppo,
    memory,
    overwatch,
    current_price: float,
    regime: str,
    entropy_val: float,
    holon_name: str = "TraderNexus"
) -> dict:
    """
    Process exit signal and return result updates for row_data.

    Args:
        symbol: Asset symbol
        final_exit: TradeSignal for the exit
        reason: Exit reason (e.g., "Thesis", "Strat")
        executor: ExecutorHolon instance
        guardian: GuardianHolon instance
        ppo: PPOHolon instance (optional)
        memory: MemoryHolon instance (optional)
        overwatch: OverwatchHolon instance (optional)
        current_price: Current asset price
        regime: Current market regime
        entropy_val: Current entropy value
        holon_name: Name for logging

    Returns:
        dict: Updates to apply to row_data
    """
    result = {}

    if not final_exit or not executor:
        return result

    # Capture metadata BEFORE execution deletes it!
    # Use _meta_copy to safely handle both Position objects and legacy dicts.
    raw_meta = executor.position_metadata.get(symbol)
    meta = _meta_copy(raw_meta)

    # 2026-03-21: Store exit_reason on Position for trade recording
    if hasattr(executor, 'positions'):
        for vk, pos in executor.positions.items():
            if getattr(pos, 'symbol', None) == symbol:
                exit_reason_str = reason or 'UNKNOWN'
                if hasattr(final_exit, 'metadata') and final_exit.metadata:
                    exit_reason_str = final_exit.metadata.get('reason', exit_reason_str)
                pos.metadata['exit_reason'] = exit_reason_str
                break

    # Execute the exit
    decision = executor.decide_trade(final_exit, regime, entropy_val)
    pnl_res = executor.execute_transaction(decision, current_price)

    if pnl_res is None:
        return result

    # Record exit with Guardian
    if guardian:
        guardian.record_exit(symbol, time.time())

    # FIX 2026-02-28: Cancel ALL remaining open orders on this symbol after a successful exit.
    # This prevents stale stop-loss orders from triggering on Kraken after the position is closed.
    # Without this, an exchange stop order placed at (e.g.) $335.74 remains live and fires again
    # 18+ minutes later even though the position was already fully exited.
    try:
        actuator = getattr(executor, 'actuator', None)
        if actuator and hasattr(actuator, 'cancel_all_orders'):
            actuator.cancel_all_orders(symbol)
    except Exception as _cancel_err:
        print(f"[{holon_name}] ⚠️ Post-exit order cleanup failed for {symbol}: {_cancel_err}")

    # pnl_res from executor is actually the raw realized PnL in USD!
    pnl_usd = pnl_res

    # === ATLAS PROFIT ARCHITECT: Update Trade Result ===
    # Track trade outcome for Atlas performance monitoring
    atlas = None
    if executor and hasattr(executor, 'atlas'):
        atlas = executor.atlas
    elif guardian and hasattr(guardian, 'atlas'):
        atlas = guardian.atlas

    if atlas and meta:
        trade_result = {
            'pnl': pnl_usd,
            'symbol': symbol,
            'direction': meta.get('direction', final_exit.direction),
            'entry_price': meta.get('entry_price', 0),
            'exit_price': current_price,
            'position_size': abs(meta.get('quantity', 0.0)),
            'timestamp': time.time(),
            'strategy': meta.get('strategy', 'UNKNOWN'),
            'atlas_expected': meta.get('atlas_expected_profit', 0)
        }
        atlas.update_trade_result(trade_result)
        print(f"[{holon_name}] [ATLAS] Trade closed: PnL=${pnl_usd:+.2f} (Expected: {trade_result['atlas_expected']*100:.2f}%)")
    # =====================================

    # Calculate percentage for PPO, Memory, and Telegram
    pnl_pct = 0.0
    position_value_usd = abs(meta.get('quantity', 0.0)) * meta.get('entry_price', current_price)
    if position_value_usd > 0:
        pnl_pct = pnl_usd / position_value_usd
        
    # --- MEMORY REPLAY SAVE ---
    if memory and meta:
        _store_memory_experience(memory, meta, pnl_pct, symbol, holon_name, executor)

    # --- PPO REWARD ---
    last_reward = 0.0
    if ppo and meta:
        if meta.get('ppo_state') is not None and meta.get('ppo_conviction') is not None:
            current_eq = executor.get_portfolio_value(0.0) if executor else 100.0
            # Pass percentage to PPO, not raw USD
            reward, duration_mins = calculate_ppo_reward(pnl_pct, meta, current_eq, holon_name)

            last_reward = reward
            print(f"[{holon_name}] 🧠 PPO REWARD: {reward:.4f} (PnL {pnl_pct*100:.2f}%, {duration_mins:.0f}m)")

            # Store experience
            state = np.array(meta['ppo_state'])
            ppo.remember(state, meta['ppo_conviction'], reward, 0.0, 0.0, True)

    # --- FIX 2026-03-01: COMPOUNDING REINVESTMENT ---
    # Auto-reinvest profits from successful exits into next position
    if getattr(config, 'COMPOUNDING_ENABLED', True) and pnl_usd > 0:
        # Sanity Check: Profit shouldn't be larger than the position size itself 
        # (unless it's a 100%+ gain, which we cap for safety in micro accounts anyway)
        if position_value_usd > 0 and pnl_usd > (position_value_usd * 2.0):
             pnl_usd = 0.0
             
        if pnl_usd >= getattr(config, 'COMPOUNDING_MIN_PROFIT_USD', 0.10):
            # Calculate reinvestment amount
            reinvest_pnl = pnl_usd * getattr(config, 'COMPOUNDING_REINVEST_PCT', 0.50)
            
            # Update executor's compounding pool
            if hasattr(executor, '_compounding_pool'):
                executor._compounding_pool += reinvest_pnl
            else:
                executor._compounding_pool = reinvest_pnl
            
            print(f"[{holon_name}] 💰 COMPOUNDING: ${pnl_usd:.2f} profit -> ${reinvest_pnl:.2f} added to reinvestment pool (Total: ${executor._compounding_pool:.2f})")
    # -----------------------------------------------------------

    # --- TELEGRAM NOTIFICATION ---
    if overwatch and hasattr(overwatch, 'send_telegram_alert'):
        msg = f"📉 **EXIT** {symbol}\nPrice: {current_price}\nPnL: {pnl_pct*100:+.2f}% (${pnl_usd:+.2f}) ({reason})"
        overwatch.send_telegram_alert(msg)

    result['Action'] = f"{final_exit.direction} ({reason})"
    result['_pnl_res'] = pnl_pct
    result['_ppo_reward'] = last_reward

    return result


def _store_memory_experience(memory, meta: dict, pnl_res: float, symbol: str, holon_name: str, executor=None):
    """Store trade context in memory for replay."""
    try:
        # [RSI, BB_Width, GMB, Entropy, Volatility_Score]
        vec = [
            float(meta.get('rsi', 50.0)) / 100.0,
            0.05 * 10.0,  # Placeholder if not saved
            0.5,  # Placeholder
            0.5,  # Placeholder
            1.0 / 5.0
        ]
        outcome = 'WIN' if pnl_res > 0 else 'LOSS'
        memory.store_experience(vec, outcome, pnl_res, symbol)

        # === FIX 2026-03-15: Record Signal Outcome for Walk-Forward Validation ===
        # Get strategy from metadata and record outcome in Oracle via executor
        strategy = meta.get('strategy', 'UNKNOWN')

        # Try to get oracle reference from executor
        oracle = None
        if executor:
            oracle = getattr(executor, 'oracle', None)
            if not oracle:
                # Try via trader parent
                trader = getattr(executor, 'trader', None)
                if trader:
                    sub_holons = getattr(trader, 'sub_holons', {})
                    oracle = sub_holons.get('oracle')

        if oracle and hasattr(oracle, '_record_signal_outcome'):
            oracle._record_signal_outcome(symbol, strategy, outcome)
            print(f"[{holon_name}] 📊 WALK-FORWARD: Recorded {symbol} {strategy} = {outcome}")
    except Exception as e:
        print(f"[{holon_name}] Memory Store Error: {e}")


def determine_exit_signal(
    symbol: str,
    guardian_exit,  # TradeSignal or None
    executor,
    oracle,
    current_price: float,
    TradeSignal  # Class reference for creating new signals
) -> tuple:
    """
    Determine if an exit should occur based on thesis validation and guardian signals.
    
    Returns:
        tuple: (final_exit, reason, thesis_valid)
    """
    thesis_valid = True
    thesis_exit = None
    direction = 'BUY'
    
    qty_held = executor.held_assets.get(symbol, 0.0) if executor else 0.0
    
    # Check Dynamic TP/SL Targets (PIPs / Structural)
    if abs(qty_held) > 0.00000001 and executor:
        raw_meta = executor.position_metadata.get(symbol)
        tp_price = _pa(raw_meta, 'take_profit')
        sl_price = _pa(raw_meta, 'stop_loss')
        pos_dir = _pa(raw_meta, 'direction', 'BUY')

        exit_dir = 'SELL' if pos_dir == 'BUY' else 'BUY'

        # CRITICAL FIX: Always resolve the authoritative price for THIS symbol from the
        # executor's latest price cache. The caller's `current_price` can be stale/wrong
        # if inner loops (e.g. Monte Carlo) overwrote it with a different symbol's price.
        latest_prices = getattr(executor, 'latest_prices', {})
        symbol_price = latest_prices.get(symbol, current_price)

        # Check Take Profit
        if tp_price:
            if (pos_dir == 'BUY' and symbol_price >= tp_price) or \
               (pos_dir == 'SELL' and symbol_price <= tp_price):
                print(f"[TraderNexus] 🎯 TAKE PROFIT STRUCK: {symbol} @ {symbol_price:.8f} (Target: {tp_price})")
                return TradeSignal(symbol, exit_dir, 1.0, symbol_price, metadata={'reason': 'TAKE_PROFIT'}), "TakeProfit", True

        # Check Stop Loss
        if sl_price:
            if (pos_dir == 'BUY' and symbol_price <= sl_price) or \
               (pos_dir == 'SELL' and symbol_price >= sl_price):
                print(f"[TraderNexus] 🛑 STOP LOSS TRIGGERED: {symbol} @ {symbol_price:.8f} (Stop: {sl_price})")
                return TradeSignal(symbol, exit_dir, 1.0, symbol_price, metadata={'reason': 'STOP_LOSS'}), "StopLoss", True

    # Thesis validation
    if abs(qty_held) > 0.00000001 and hasattr(oracle, 'verify_holding_physics') and executor:
        raw_ex_meta = executor.position_metadata.get(symbol)
        # --- PATCH: ARBITRAGE EXEMPTION ---
        # Arbitrage trades are Market Neutral (or Yield-based).
        # They should NOT be killed by Directional Bias ("Thesis").
        reason_str = str(_pa(raw_ex_meta, 'reason', '')).upper()
        strat_str = str(_pa(raw_ex_meta, 'strategy', '')).upper()
        strategy_type = str(_pa(raw_ex_meta, 'strategy_type', '')).upper()

        # Explicit flag takes priority, then pattern matching as fallback
        is_arb = _pa(raw_ex_meta, 'is_arbitrage', False) or \
                 strategy_type == 'ARBITRAGE' or \
                 'ARBITRAGE' in reason_str or 'BASIS' in reason_str or 'ARB' in strat_str or \
                 'ARBITRAGE' in strat_str or 'BASIS' in strat_str or 'CARRY' in strat_str

        if not is_arb:
            # === ML EXIT OPTIMIZATION (2026-03-22) ===
            # Check ML exit optimizer for recommendation
            try:
                from .ml_exit_optimizer import predict_exit

                # Get position data - CALCULATE REAL-TIME PnL
                entry_price = _pa(raw_ex_meta, 'entry_price', symbol_price)
                
                # Calculate REAL-TIME PnL from current price vs entry
                if pos_dir == 'BUY':
                    realtime_pnl = (symbol_price - entry_price) / entry_price
                else:  # SELL
                    realtime_pnl = (entry_price - symbol_price) / entry_price
                
                pos_data = {
                    'direction': pos_dir,
                    'entry_price': entry_price,
                    'current_price': symbol_price,
                    'pnl_percent': realtime_pnl,  # USE REAL-TIME CALCULATED PnL
                    'entry_time': _pa(raw_ex_meta, 'entry_time', datetime.now()),
                }

                # Get ML exit recommendation
                exit_rec = predict_exit(symbol, pos_data)

                # Log recommendation
                print(f"[TraderNexus] 🤖 ML EXIT: {symbol} - {exit_rec['recommendation']} ({exit_rec['reason']})")

                # Act on HIGH urgency recommendations
                if exit_rec['urgency'] in ['HIGH', 'VERY_HIGH']:
                    if exit_rec['recommendation'] == 'CUT_LOSS':
                        print(f"[TraderNexus] 🤖 ML URGENT EXIT: {symbol} - Cutting loss")
                        return TradeSignal(symbol, exit_dir, 1.0, symbol_price, metadata={'reason': 'ML_CUT_LOSS'}), "ML_Exit", True
                    elif exit_rec['recommendation'] == 'TAKE_PROFIT':
                        print(f"[TraderNexus] 🤖 ML TAKE PROFIT: {symbol} - Taking profit")
                        return TradeSignal(symbol, exit_dir, 1.0, symbol_price, metadata={'reason': 'ML_TAKE_PROFIT'}), "ML_Exit", True

                # Store ML recommendation in metadata for monitoring
                raw_ex_meta['ml_exit_recommendation'] = exit_rec

            except Exception as e:
                print(f"[TraderNexus] ML Exit Optimizer error: {e}")
            # ============================================
            direction = _pa(raw_ex_meta, 'direction', 'BUY')
            thesis_valid = oracle.verify_holding_physics(symbol, direction)
        else:
            # It's an Arb trade. Thesis Validation is irrelevant.
            # Only Funding Health or Governor limits should close it.
            thesis_valid = True

    if not thesis_valid:
        print(f"[TraderNexus] 🚫 THESIS INVALIDATED for {symbol}. Exiting.")
        exit_dir = 'BUY' if direction == 'SELL' else 'SELL'
        thesis_exit = TradeSignal(symbol, exit_dir, 1.0, current_price, metadata={'reason': 'Thesis'})
    
    # Determine final exit
    final_exit = None
    reason = "IDLE"
    
    if thesis_exit:
        final_exit = thesis_exit
        reason = "Thesis"
    elif guardian_exit:
        final_exit = guardian_exit
        reason = "Strat"
    
    return final_exit, reason, thesis_valid
