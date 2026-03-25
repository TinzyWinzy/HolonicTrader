"""
trader_entry_handler.py
Entry Logic Handler - Extracted from agent_trader.py

Handles:
- Signal deduplication/debouncing
- PPO state construction
- Governor approval & sizing
- Entry execution
- Telegram notifications
"""
import time
import numpy as np
from HolonicTrader.performance_tracker import get_performance_data
import config


def build_ppo_state(
    regime: str,
    entropy_val: float,
    atr_ratio: float,
    gov_health: dict,
    tda_score: float = 0.5,
    tda_status: str = 'STABLE'
) -> np.ndarray:
    """
    Build the 8-dimensional PPO state vector.
    
    Returns:
        np.ndarray: PPO state vector
    """
    pnl_tracker = get_performance_data()
    
    tda_score_val = tda_score
    if tda_status == 'CRITICAL':
        tda_score_val = 0.0
    
    ppo_state = np.array([
        {'ORDERED': 0.0, 'TRANSITION': 0.5, 'CHAOTIC': 1.0}.get(regime, 0.5),
        entropy_val,
        pnl_tracker.get('win_rate', 0.5),
        atr_ratio,
        gov_health['drawdown_pct'],
        gov_health['margin_utilization'],
        tda_score_val,
        0.5  # Placeholder for RCMWPE specific feature
    ], dtype=np.float32)
    
    return ppo_state


def should_debounce_signal(
    symbol: str,
    entry_sig,
    governor,
    current_price: float,
    last_signal_attempts: dict,
    cooldown_seconds: int = 120
) -> tuple[bool, str]:
    """
    Check if signal should be debounced/skipped.
    
    Returns:
        tuple: (should_skip, reason)
    """
    if entry_sig is None:
        return True, "no_signal"
    
    sig_key = f"{symbol}_{entry_sig.direction}"
    last_attempt = last_signal_attempts.get(sig_key, 0)
    now_ts = time.time()
    
    # 2 Minute cooldown on retries
    if (now_ts - last_attempt) < cooldown_seconds:
        return True, "retry_cooldown"
    
    # Stack distance check
    if governor:
        pos_data = governor.positions.get(symbol)
        if pos_data:
            last_entry_price = pos_data.get('entry_price', 0.0) if isinstance(pos_data, dict) else getattr(pos_data, 'entry_price', 0.0)
            if last_entry_price > 0:
                dist_pct = abs(current_price - last_entry_price) / last_entry_price
                min_dist = getattr(config, 'GOVERNOR_MIN_STACK_DIST', 0.002)
                
                if dist_pct < min_dist:
                    return True, "stack_too_close"
    
    return False, ""


def handle_entry(
    symbol: str,
    entry_sig,  # TradeSignal
    executor,
    governor,
    oracle,
    ppo,
    overwatch,
    current_price: float,
    indicators: dict,
    regime: str,
    entropy_val: float,
    sent_score: float,
    global_bias: float,
    cycle_entries_count: int,
    limit_entries: int,
    last_signal_attempts: dict,
    tda_score: float = 0.5,
    tda_status: str = 'STABLE',
    holon_name: str = "TraderNexus",
    structure_bias: str = "NEUTRAL",  # NEW: BULLISH, BEARISH, or NEUTRAL
    smce_vol: dict = None,
    cooldown_tracker: dict = None  # NEW: For scout rotation priority
) -> tuple[bool, int, dict]:
    """
    Process entry signal.
    
    Returns:
        tuple: (executed, new_cycle_count, row_data_updates)
    """
    result = {}
    executed = False
    
    if not entry_sig or not executor or not governor or not oracle:
        return executed, cycle_entries_count, result

    # Check cycle limit
    if cycle_entries_count >= limit_entries:
        return executed, cycle_entries_count, result

    # === FIX 2026-03-15: CRISIS CHECK ===
    # Check crisis status before processing any entry
    if hasattr(governor, 'check_crisis_status'):
        crisis_status = governor.check_crisis_status()
        if not crisis_status.get('trading_allowed', True):
            print(f"[{holon_name}] ☢️ CRISIS HALT: Entry blocked for {symbol} (Crisis Score: {crisis_status.get('crisis_score', 0):.2f})")
            result['Action'] = f"{entry_sig.direction} (CRISIS HALT)"
            return executed, cycle_entries_count, result

        # Apply position reduction if crisis is elevated
        if crisis_status.get('position_reduction', 0) > 0:
            print(f"[{holon_name}] ⚠️ CRISIS REDUCTION: {symbol} position will be reduced by {crisis_status['position_reduction']*100:.0f}%")
    # === END CRISIS CHECK ===

    # === STRUCTURE BOSS VETO (AEHML Mandate) - ENHANCED 2026-03-12 ===
    # "Has absolute veto on entries violating structure."
    # Rule: "Must be in SUPPORT zone for LONGs (NEUTRAL rejected)"
    # Implied: Must be in RESISTANCE zone for SHORTs.

    # FIX 2026-03-08: structure_bias param is the MACRO TREND (BULLISH/BEARISH)
    # We need sls_zone for the actual support/resistance zone check.
    # The caller should pass sls_zone in entry_sig.metadata['structure']['sls_zone']
    structure_data = entry_sig.metadata.get('structure', {})
    sls_zone = structure_data.get('sls_zone', 'NEUTRAL')  # Default to NEUTRAL if not provided
    # FIX 2026-03-09: Defensive Price Resolution
    # Ensure we use the authoritative price for THIS symbol from the executor's
    # latest price cache. The caller's `current_price` can be stale if inner loops
    # (Hygiene, Monte Carlo) overwritten it with a different symbol's price.
    latest_prices = getattr(executor, 'latest_prices', {})
    symbol_price = latest_prices.get(symbol, current_price)

    macro_trend = structure_bias  # BULLISH/BEARISH from StructureBoss

    signal_direction = entry_sig.direction
    is_structure_aligned = False
    veto_reason = ""

    # === NEW: STRICT ZONE REQUIREMENTS ===
    # FIX 2026-03-16: Satellite assets require proper structure support (no NEUTRAL zone entries)
    is_satellite = symbol in getattr(config, 'SATELLITE_ASSETS', [])

    if signal_direction == 'BUY':
        # LONG: Require SUPPORT zone (NEUTRAL allowed with lower conviction - FIX 2026-03-23)
        if sls_zone == 'SUPPORT':
            is_structure_aligned = True
        elif sls_zone == 'NEUTRAL':
            # FIX 2026-03-23: More permissive NEUTRAL zone entries
            # Reduced conviction requirement from 0.45 to 0.40
            if entry_sig.conviction >= 0.40:
                is_structure_aligned = True
                veto_reason = "Neutral Zone + Moderate Conviction"
            elif macro_trend == 'BULLISH' and entry_sig.conviction >= 0.35:
                is_structure_aligned = True
                veto_reason = "Bullish Macro + Neutral Zone"
            else:
                veto_reason = f"Conviction {entry_sig.conviction:.2f} < 0.40 (Neutral Zone)"
        else:
            veto_reason = f"Zone {sls_zone} != SUPPORT (Trend: {macro_trend})"

    elif signal_direction == 'SELL':
        # SHORT: Require RESISTANCE zone (NEUTRAL allowed with lower conviction - FIX 2026-03-23)
        if sls_zone == 'RESISTANCE':
            is_structure_aligned = True
        elif sls_zone == 'NEUTRAL':
            # FIX 2026-03-23: More permissive NEUTRAL zone entries
            # Reduced conviction requirement from 0.45 to 0.40
            if entry_sig.conviction >= 0.40:
                is_structure_aligned = True
                veto_reason = "Neutral Zone + Moderate Conviction"
            elif macro_trend == 'BEARISH' and entry_sig.conviction >= 0.35:
                is_structure_aligned = True
                veto_reason = "Bearish Macro + Neutral Zone"
            else:
                veto_reason = f"Conviction {entry_sig.conviction:.2f} < 0.40 (Neutral Zone)"
        else:
            veto_reason = f"Zone {sls_zone} != RESISTANCE (Trend: {macro_trend})"

    # === EXCEPTIONS (Narrowed) ===
    is_arb_signal = entry_sig.metadata.get('is_arb', False)
    funding_yield = entry_sig.metadata.get('apy', 0.0)

    # Arb override: Only if yield > 200% APY (was 100%)
    if is_arb_signal or (funding_yield > 200.0):
        is_structure_aligned = True
        veto_reason = f"Arb/Yield Override (APY {funding_yield:.0f}%)"

    # === REMOVED: Whale Neutral Override ===
    # WhaleHolon can no longer bypass structure in NEUTRAL zones
    # Whales MUST align with SUPPORT (long) or RESISTANCE (short)

    # === REMOVED: High Conviction Neutral Override ===
    # Conviction > 0.8 no longer bypasses structure
    # Structure > Momentum (AEHML Principle)

    if not is_structure_aligned:
        # COMPUTE WASTE FIX: Track veto count and suppress repeated logs
        now = time.time()
        veto_entry = getattr(governor, '_veto_tracker', {}).get(symbol)
        veto_count = veto_entry.get('count', 0) if veto_entry else 0

        # Only log every 5th veto to reduce spam (after first 3)
        if veto_count < 3 or veto_count % 5 == 0:
            print(f"[{holon_name}] 🚫 STRUCTURE BOSS VETO: {symbol} {signal_direction} blocked. ({veto_reason})")

        # Record veto in StructureBoss for cross-agent caching
        structure = getattr(governor, 'sub_holons', {}).get('structure') if hasattr(governor, 'sub_holons') else None
        if structure and hasattr(structure, 'record_veto'):
            structure.record_veto(symbol, veto_reason)

        result['Action'] = f"{signal_direction} (STRUCT VETO)"
        return executed, cycle_entries_count, result
    # === END STRUCTURE BOSS VETO ===

    # === ORION CTKS PROFIT NAVIGATOR — Market Path Filter ===
    # Orion reads structure + momentum + intermarket to determine if this trade
    # aligns with the market path. Can reject or penalize conviction.
    orion_state = entry_sig.metadata.get('orion', {})
    if orion_state and getattr(config, 'ORION_ENABLED', False):
        # Lazy import: StructureBoss has orion_filter_signal
        _structure_boss = getattr(governor, 'sub_holons', {}).get('structure') if hasattr(governor, 'sub_holons') else None
        if _structure_boss and hasattr(_structure_boss, 'orion_filter_signal'):
            orion_result = _structure_boss.orion_filter_signal(
                symbol, signal_direction, entry_sig.conviction, orion_state)
        else:
            # Inline lightweight Orion path check (fallback if structure boss not reachable)
            orion_path = orion_state.get('path', 'NEUTRAL')
            orion_strictness = orion_state.get('path_strictness', 0.7)
            orion_result = {'approved': True, 'adjusted_conviction': entry_sig.conviction,
                            'reason': 'Orion passthrough', 'action': 'APPROVED'}
            if signal_direction == 'SELL' and orion_path == 'UP':
                penalty = orion_strictness
                adj = entry_sig.conviction * (1.0 - penalty)
                orion_result = {'approved': adj >= 0.20, 'adjusted_conviction': adj,
                                'reason': f'Orion: SELL against UP path (-{penalty*100:.0f}%)',
                                'action': 'REDUCED' if adj >= 0.20 else 'REJECTED'}
            elif signal_direction == 'BUY' and orion_path == 'DOWN':
                penalty = orion_strictness
                adj = entry_sig.conviction * (1.0 - penalty)
                orion_result = {'approved': adj >= 0.20, 'adjusted_conviction': adj,
                                'reason': f'Orion: BUY against DOWN path (-{penalty*100:.0f}%)',
                                'action': 'REDUCED' if adj >= 0.20 else 'REJECTED'}

        if not orion_result.get('approved', True):
            print(f"[{holon_name}] 🧭 ORION VETO: {symbol} {signal_direction} | {orion_result.get('reason', '')}")
            result['Action'] = f"{signal_direction} (ORION VETO)"
            return executed, cycle_entries_count, result

        if orion_result.get('action') == 'REDUCED':
            old_conv = entry_sig.conviction
            entry_sig.conviction = orion_result['adjusted_conviction']
            print(f"[{holon_name}] 🧭 ORION ADJUST: {symbol} conviction {old_conv:.2f} → {entry_sig.conviction:.2f} | {orion_result.get('reason', '')}")

        entry_sig.metadata['orion_filter'] = orion_result
    # === END ORION FILTER ===
    
    # Calculate ATR reference
    tr_series = indicators.get('tr')
    if tr_series is not None and hasattr(tr_series, 'rolling') and not tr_series.empty:
        atr_ref = tr_series.rolling(14).mean().rolling(14).mean().iloc[-1]
    else:
        atr_ref = indicators.get('atr', 1.0)
    
    atr_ratio = min(2.0, indicators.get('atr', 0.0) / atr_ref) if atr_ref > 0 else 1.0
    gov_health = governor.get_portfolio_health()
    
    # Build PPO state
    ppo_state = build_ppo_state(regime, entropy_val, atr_ratio, gov_health, tda_score, tda_status)
    
    # Get conviction
    conviction = ppo.get_conviction(ppo_state) if ppo else 0.5
    
    # Update signal metadata
    entry_sig.metadata.update({
        'ppo_state': ppo_state.tolist(),
        'ppo_conviction': conviction,
        'atr': indicators.get('atr', 0.0)
    })
    
    # Debounce check
    should_skip, skip_reason = should_debounce_signal(
        symbol, entry_sig, governor, symbol_price, last_signal_attempts
    )
    
    if should_skip:
        return executed, cycle_entries_count, result

    # === CHRONOS: TAIL-LOSS SUPPRESSION (PHASE 1) ===
    # Evidence: recent -3.02% tail loss was attributed to MARKET_OPEN_FVG in TRANSITION regime.
    # Rule: only allow MARKET_OPEN_FVG entries in ORDERED regime.
    sig_strategy = str(entry_sig.metadata.get('strategy', '') or '').upper()
    if sig_strategy == 'MARKET_OPEN_FVG' and regime != 'ORDERED':
        print(f"[{holon_name}] 🛑 FVG VETO: {symbol} blocked (Regime={regime}, requires ORDERED)")
        result['Action'] = f"{signal_direction} (FVG VETO)"
        return executed, cycle_entries_count, result

    # === 2026-03-20 AUDIT: HARD REGIME/ENTROPY GATE (Belt-and-Suspenders) ===
    # Oracle SHOULD block non-ORDERED regimes, but this gate catches any signal that slips through
    # (e.g. whale signals, imported signals, GUI-triggered entries)
    regime_filter = getattr(config, 'ENTRY_REGIME_FILTER', None)
    if regime_filter:
        regime_aliases = getattr(config, 'ENTRY_REGIME_ALIASES', {})
        normalized_regime = regime_aliases.get(str(regime).upper(), str(regime).upper())
        normalized_allowed = {regime_aliases.get(str(r).upper(), str(r).upper()) for r in regime_filter}
        if normalized_regime not in normalized_allowed:
            print(f"[{holon_name}] 🛑 REGIME GATE: {symbol} {signal_direction} blocked (regime={normalized_regime}, allowed={sorted(normalized_allowed)})")
            result['Action'] = f"{signal_direction} (REGIME GATE)"
            return executed, cycle_entries_count, result

    entry_max_entropy = getattr(config, 'ENTRY_MAX_ENTROPY', None)
    if entry_max_entropy is not None and entropy_val > entry_max_entropy:
        print(f"[{holon_name}] 🛑 ENTROPY GATE: {symbol} {signal_direction} blocked (entropy {entropy_val:.3f} > {entry_max_entropy})")
        result['Action'] = f"{signal_direction} (ENTROPY GATE)"
        return executed, cycle_entries_count, result
    # === END REGIME/ENTROPY GATE ===

    # === FIX 2026-03-19 (Helix): EXECUTION COST FILTER ===
    # Principle: if expected_profit < (spread + fees + slippage), DO NOT TRADE.
    # This is the core nano-account fix. Round-trip cost on a $15 position eats the edge.
    if getattr(config, 'EXECUTION_COST_FILTER_ENABLED', False):
        round_trip_cost_pct = 2 * (config.ESTIMATED_FEE_PCT + config.ESTIMATED_SLIPPAGE_PCT)
        min_edge_required = round_trip_cost_pct * getattr(config, 'MIN_EDGE_MULTIPLE', 3.0)
        # Use TP target as proxy for expected edge
        tp_target = getattr(config, 'DEFAULT_TAKE_PROFIT_PCT', 0.06)
        sl_target = getattr(config, 'DEFAULT_STOP_LOSS_PCT', 0.02)
        # Net edge after costs = TP - round_trip_cost
        net_edge = tp_target - round_trip_cost_pct
        if net_edge < min_edge_required:
            print(f"[{holon_name}] 💸 COST FILTER: {symbol} blocked (Net edge {net_edge*100:.2f}% < required {min_edge_required*100:.2f}%)")
            result['Action'] = f"{signal_direction} (COST FILTER)"
            return executed, cycle_entries_count, result
        # Also check: is position large enough that TP in USD > minimum (minimum viable profit)?
        est_position_value = getattr(config, 'MIN_ORDER_VALUE', 15.0)
        est_profit_usd = est_position_value * net_edge
        min_profit_usd = getattr(config, 'MIN_PROFIT_USD', 0.10)  # FIX 2026-03-23: Configurable
        if est_profit_usd < min_profit_usd:
            print(f"[{holon_name}] 💸 COST FILTER: {symbol} blocked (Est profit ${est_profit_usd:.2f} < ${min_profit_usd:.2f} minimum)")
            result['Action'] = f"{signal_direction} (DUST TRADE)"
            return executed, cycle_entries_count, result
    # === END EXECUTION COST FILTER ===
    
    # Governor sizing
    is_whale = entry_sig.metadata.get('is_whale', False)

    approved, safe_qty, leverage = governor.calc_position_size(
        symbol, symbol_price, indicators.get('atr', 0.0), atr_ref, conviction,
        direction=entry_sig.direction, sentiment_score=sent_score,
        whale_confirmed=is_whale, market_bias=global_bias,
        metadata=entry_sig.metadata,
        latest_prices=executor.latest_prices if executor else {}
    )

    if not approved or safe_qty <= 0:
        if not approved:
            result['Action'] = f"{signal_direction} (GOV REJECT)"
            print(f"[{holon_name}] 🛡️ Governor Vetoed {symbol}: Risk/Exposure Limits.")
        elif safe_qty <= 0:
            result['Action'] = f"{signal_direction} (NO QTY)"
        return executed, cycle_entries_count, result

    # === CHRONOS FORENSICS: EXECUTION-BOUNDARY BLACKLIST ENFORCEMENT ===
    # Evidence: session logs show symbol can be blacklisted (2 consecutive losses)
    # and still opened within the same cycle due to timing/order-of-operations.
    # This hard-check stops the entry immediately before sending any order.
    if governor and hasattr(governor, "blacklist") and symbol in getattr(governor, "blacklist", {}):
        try:
            expiry = governor.blacklist.get(symbol)
            if expiry and time.time() > float(expiry):
                # Expired -> cleanup so future cycles can trade
                del governor.blacklist[symbol]
            else:
                remaining_hr = ((float(expiry) - time.time()) / 3600.0) if expiry else None
                if remaining_hr is not None and remaining_hr > 0:
                    print(f"[{holon_name}] 🛑 BLACKLIST BLOCK: {symbol} blocked for {remaining_hr:.2f}h")
                else:
                    print(f"[{holon_name}] 🛑 BLACKLIST BLOCK: {symbol} blocked")
                result["Action"] = f"{signal_direction} (BLACKLIST)"
                return executed, cycle_entries_count, result
        except Exception:
            # If anything is off, fail safe: block the entry rather than bypassing the circuit breaker.
            print(f"[{holon_name}] 🛑 BLACKLIST BLOCK: {symbol} (fail-safe)")
            result["Action"] = f"{signal_direction} (BLACKLIST)"
            return executed, cycle_entries_count, result

    # === FIX 2026-03-15: Apply Crisis Position Reduction ===
    if hasattr(governor, 'crisis_position_reduction') and governor.crisis_position_reduction > 0:
        reduction_factor = 1.0 - governor.crisis_position_reduction
        safe_qty = safe_qty * reduction_factor
        print(f"[{holon_name}] ⚠️ CRISIS ADJUSTMENT: {symbol} qty reduced to {safe_qty:.4f} ({reduction_factor*100:.0f}% of normal)")
    # === END CRISIS ADJUSTMENT ===

    # --- PROACTIVE LIQUIDITY CHECK ---
    # Enhance Reliability: Check book depth before committing
    if getattr(config, 'TRADING_MODE', 'PAPER') != 'PAPER' and executor and hasattr(executor, 'actuator'):
        # For LIVE mode, fetch the order book and ensure we don't slip
        if not executor.actuator.check_liquidity(symbol, entry_sig.direction, safe_qty, symbol_price):
             print(f"[{holon_name}] 🛑 LIQUIDITY VETO: {symbol} rejected before execution (Insufficient depth).")
             result['Action'] = f"{signal_direction} (NO LIQUIDITY)"
             return executed, cycle_entries_count, result

    # ── SMCE v1: 3-Layer Constitutional Gate (DEFERRED EVALUATION) ──
    # Runs AFTER Governor has validated positional capacity/cooldowns to SAVE COMPUTE (Prevent Veto Storms)
    if governor and getattr(governor, 'smce_doctrine', None):
        _smce_notional = safe_qty * symbol_price
        _smce_gate = governor.run_smce_pre_trade_gate(
            symbol=symbol,
            direction=entry_sig.direction,
            proposed_notional=_smce_notional,
            proposed_leverage=leverage,
            market_context={
                'structure':         structure_bias,
                'momentum_aligned':  global_bias >= 0.55,
                'liquidity_status':  'healthy',
                'entropy':           entropy_val,
                'correlation_idx':   0.5,
            },
            portfolio_state={
                'equity':               getattr(governor, 'balance', 0.0),
                'cluster_exposure':     0.0,
                'proposed_cluster_add': 0.0,
                'cvar_95':              0.02,
            },
            volatilities=smce_vol or {},
            metadata=entry_sig.metadata,
        )

        if not _smce_gate.get('allowed', True):
            print(f"[{holon_name}] [SMCE] VETO {symbol} | {_smce_gate.get('reason','')}")
            result['Action'] = f"{signal_direction} (SMCE VETO)"
            return executed, cycle_entries_count, result
        else:
            # Honour approved size and leverage from the gate
            _approved = _smce_gate.get('approved_size', _smce_notional)
            if _approved < _smce_notional * 0.99:
                safe_qty = max(
                    safe_qty * (_approved / max(1e-9, _smce_notional)),
                    getattr(config, 'MIN_ORDER_VALUE', 5.0) / max(symbol_price, 1)
                )
            leverage = _smce_gate.get('max_leverage', leverage)
            entry_sig.metadata['smce_max_leverage'] = leverage
            entry_sig.metadata['smce_scorecard']    = _smce_gate.get('scorecard', {})

    # === ATLAS PROFIT ARCHITECT: Profit Filter Check ===
    # Runs after SMCE to ensure only profitable trades with positive expectancy
    atlas = None
    # Try to get atlas from governor (injected from main_live_phase4.py)
    if governor and hasattr(governor, 'atlas'):
        atlas = governor.atlas
    elif governor and hasattr(governor, 'sub_holons') and 'atlas' in governor.sub_holons:
        atlas = governor.sub_holons['atlas']

    if atlas:
        # Build market data for Atlas
        market_data = {
            'volatility_pct': indicators.get('atr', 0.0) / symbol_price if symbol_price > 0 else 0.01,
            'spread_pct': 0.0008,  # Typical spread estimate
            'liquidity_score': 0.8,  # Default healthy liquidity
            'regime': regime,
            'regime_score': 0.3,
            'price': symbol_price,
            'signal_direction': signal_direction
        }

        # Build portfolio state for Atlas
        portfolio_state = {
            'account_balance': getattr(governor, 'balance', executor.balance_usd if executor else 10000),
            'win_rate': get_performance_data().get('win_rate', 0.6),
            'win_loss_ratio': get_performance_data().get('win_loss_ratio', 1.2),
            'current_positions': governor.positions if governor else {}
        }

        # Build signal data for Atlas
        signal_data = {
            'direction': signal_direction,
            'strength': entry_sig.conviction if hasattr(entry_sig, 'conviction') else 0.7,
            'symbol': symbol,
            'source': str(entry_sig.metadata.get('strategy', 'UNKNOWN'))
        }

        # Process through Atlas profit filter
        approved, reason, metadata, atlas_position_size = atlas.process_trade_signal(
            signal_data, market_data, portfolio_state
        )

        if not approved:
            print(f"[{holon_name}] [ATLAS] VETO {symbol} | {reason}")
            result['Action'] = f"{signal_direction} (ATLAS VETO)"
            return executed, cycle_entries_count, result

        # Atlas approved - log and potentially adjust position size
        if atlas_position_size > 0 and atlas_position_size < safe_qty * symbol_price:
            # Atlas recommends smaller size - respect it
            new_qty = atlas_position_size / symbol_price
            if new_qty < safe_qty:
                print(f"[{holon_name}] [ATLAS] {symbol} size adjusted: {safe_qty:.4f} -> {new_qty:.4f}")
                safe_qty = new_qty

        print(f"[{holon_name}] [ATLAS] APPROVED {symbol} | {reason} | Expected: {metadata.get('expected_profit_pct', 0)*100:.2f}%")
        entry_sig.metadata['atlas_approved'] = True
        entry_sig.metadata['atlas_expected_profit'] = metadata.get('expected_profit_pct', 0)
        # 2026-03-21: Pass Atlas metadata to Position for winning pattern analysis
        entry_sig.metadata['quality_score'] = metadata.get('quality_score')
        entry_sig.metadata['is_whitelisted'] = metadata.get('is_whitelisted', False)
    # =====================================

    # === PHASE 3: TWAP EXECUTION SUPPORT ===
    # Check if signal requests TWAP execution
    execution_type = entry_sig.metadata.get('execution_type', 'MARKET')
    
    if execution_type == 'TWAP':
        # TWAP parameters from metadata
        twap_duration = entry_sig.metadata.get('twap_duration_minutes', 30)
        twap_slices = entry_sig.metadata.get('twap_num_slices', 6)
        
        # Start TWAP via phase3_execution
        try:
            from phase3_execution import get_phase3
            phase3 = get_phase3()
        except ImportError:
            phase3 = None
        
        # Start TWAP execution
        if phase3 is None:
            print(f"[{holon_name}] ⚠️ TWAP unavailable (phase3_execution missing), falling through to standard order")
        else:
            phase3.start_twap(symbol, entry_sig.direction, safe_qty, twap_duration, twap_slices)
            print(f"[{holon_name}] ⏱️ TWAP STARTED: {symbol} {entry_sig.direction} {safe_qty} over {twap_duration}min ({twap_slices} slices)")
        
            result['Action'] = f"{signal_direction} (TWAP)"
            executed = True
            cycle_entries_count += 1
            return executed, cycle_entries_count, result
    # =====================================

    # Execute entry (standard market order)
    entry_sig.size = safe_qty
    # 2026-03-21: Ensure entry context is in signal metadata for Position creation
    entry_sig.metadata['entropy'] = entropy_val
    entry_sig.metadata['regime'] = regime
    entry_sig.metadata['conviction'] = entry_sig.conviction
    decision = executor.decide_trade(entry_sig, regime, entropy_val)
    
    if decision.action == 'HALT':
        print(f"[{holon_name}] 🛑 ENTRY HALTED: {symbol} (Executor HALT)")
        result['Action'] = f"{signal_direction} (HALT)"
        return executed, cycle_entries_count, result

    # Update debounce timestamp early to prevent spam on retry loops
    sig_key = f"{symbol}_{entry_sig.direction}"
    last_signal_attempts[sig_key] = time.time()

    pnl_res = None
    retry_count = 0
    max_retries = 2  # REDUCED: 3 -> 2 (faster failure, less API spam)
    base_delay = 3.0  # INCREASED: 1.0 -> 3.0 (give API time to settle)
    last_error = "Unknown"
    last_error_msg = ""  # Store error message for diagnosis

    while retry_count < max_retries:
        # Re-check blacklist before each attempt (blacklist can be triggered asynchronously)
        if governor and hasattr(governor, "blacklist") and symbol in getattr(governor, "blacklist", {}):
            try:
                expiry = governor.blacklist.get(symbol)
                if expiry and time.time() > float(expiry):
                    del governor.blacklist[symbol]
                else:
                    print(f"[{holon_name}] 🛑 BLACKLIST BLOCK (mid-retry): {symbol} entry cancelled")
                    result["Action"] = f"{signal_direction} (BLACKLIST)"
                    return executed, cycle_entries_count, result
            except Exception:
                print(f"[{holon_name}] 🛑 BLACKLIST BLOCK (mid-retry): {symbol} (fail-safe)")
                result["Action"] = f"{signal_direction} (BLACKLIST)"
                return executed, cycle_entries_count, result

        print(f"[{holon_name}] 🎯 EXECUTING ENTRY (Attempt {retry_count+1}/{max_retries}): {symbol} (Qty: {safe_qty:.4f}, Lev: {leverage}x) @ {symbol_price}")

        pnl_res = executor.execute_transaction(decision, symbol_price)
        
        # If failed, capture error for diagnostic
        if pnl_res is None:
            last_error_msg = getattr(executor, '_last_execution_error', 'Unknown error')
        
        if pnl_res is not None:
             break

        retry_count += 1
        if retry_count < max_retries:
             # EXPONENTIAL BACKOFF: 3s, 6s (prevents API spam, gives time for rate limit reset)
             delay = base_delay * (2 ** (retry_count - 1))
             
             # DIAGNOSTIC: Log error type to help debugging
             print(f"[{holon_name}] ⚠️ EXECUTION FAILED: {last_error_msg}")
             
             # If error is non-transient (e.g., insufficient funds), abort immediately
             if "insufficientAvailableFunds" in last_error_msg or "wouldNotReducePosition" in last_error_msg:
                 print(f"[{holon_name}] 🛑 ABORTING: Non-transient error detected. No retry.")
                 break
             
             print(f"[{holon_name}] 🔄 ENTRY RETRY {retry_count}/{max_retries}: {symbol} failed. Retrying in {delay:.1f}s...")
             time.sleep(delay)

    if pnl_res is not None:
        executed = True

        # Update cooldown tracker for scout rotation priority
        if cooldown_tracker is not None:
            cooldown_tracker[symbol] = time.time()

        # Notify governor (AEHML 2.1: Pass Leverage & Strategy for Margin Accounting)
        if governor:
            governor.open_position(
                symbol,
                entry_sig.direction,
                symbol_price,
                safe_qty,
                leverage=leverage,
                strategy=entry_sig.metadata.get('strategy', 'DIRECTIONAL')
            )

        # Telegram notification
        if overwatch and hasattr(overwatch, 'send_telegram_alert'):
            msg = f"🚀 **ENTRY** {symbol}\nPrice: {symbol_price}\nSize: {entry_sig.size:.4f}"
            overwatch.send_telegram_alert(msg)

        # Determine action label
        reason_tag = entry_sig.metadata.get('reason', 'TREND')
        if entry_sig.metadata.get('is_whale'):
            result['Action'] = f"WHALE {signal_direction} 🐋"
        else:
            result['Action'] = f"{signal_direction} ({reason_tag})"

        cycle_entries_count += 1

        if cycle_entries_count >= limit_entries:
            print(f"[{holon_name}] 🛑 CYCLE LIMIT REACHED ({limit_entries} entries). Halting further entries for this cycle.")
    else:
        print(f"[{holon_name}] ⚠️ ENTRY ABORTED: {symbol} (Execution Failed after {max_retries} attempts). Governor NOT updated.")
        result['Action'] = f"{signal_direction} (FAILED)"
    
    return executed, cycle_entries_count, result
