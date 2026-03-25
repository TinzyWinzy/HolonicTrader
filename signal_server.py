import logging
import threading
import time
import os
import sys
import json
import signal
import atexit
import msvcrt
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from flask import Flask, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Import Holonic Stack
try:
    from HolonicTrader.stack_factory import create_signal_stack
    import config
    from core.scouts.listing_scouter import ListingScouter
    from state_aggregator import StateAggregator
except ImportError as e:
    print(f"Error importing Holonic components: {e}")
    sys.exit(1)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HolonHub")

app = Flask(__name__, static_folder='dashboard-ui/dist')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global State Aggregator
aggregator = StateAggregator()

# Global Event/Signal flags (Keep these for inter-thread coordination)
shutdown_event = threading.Event()
scan_trigger_event = threading.Event()  # Event to trigger immediate scan
holon_stack = None  # Global reference to holon stack for market data access

class DashboardLogHandler(logging.Handler):
    """Custom logging handler that captures logs for the dashboard."""
    def emit(self, record):
        try:
            aggregator.add_log(self.format(record), record.levelname)
        except Exception:
            pass

# Add dashboard handler to root logger
dashboard_handler = DashboardLogHandler()
dashboard_handler.setLevel(logging.INFO)
dashboard_handler.setFormatter(logging.Formatter('%(name)s - %(message)s'))
logging.getLogger().addHandler(dashboard_handler)

# Rate limiting
request_counts = defaultdict(list)
RATE_LIMIT_WINDOW = 60
MAX_REQUESTS_PER_WINDOW = 100

# Single Instance Lock
LOCK_FILE = 'signal_server.lock'
lock_file_handle = None

def validate_config():
    """Validate required configuration parameters."""
    required_configs = ['MARKET_HOLON_TYPE', 'INITIAL_CAPITAL', 'TRADING_MODE', 'ALLOWED_ASSETS']
    missing = [c for c in required_configs if not hasattr(config, c)]
    if missing:
        logger.error(f"Missing configs: {missing}")
        return False
    return True

def is_rate_limited(request_type: str) -> bool:
    """Simple rate limiting."""
    now = datetime.now()
    request_counts[request_type] = [ts for ts in request_counts[request_type] if (now - ts).total_seconds() < RATE_LIMIT_WINDOW]
    if len(request_counts[request_type]) >= MAX_REQUESTS_PER_WINDOW:
        return True
    request_counts[request_type].append(now)
    return False

def acquire_lock():
    """Prevent multiple instances from running."""
    global lock_file_handle
    try:
        lock_file_handle = open(LOCK_FILE, 'w')
        msvcrt.locking(lock_file_handle.fileno(), msvcrt.LK_NBLCK, 1)
        lock_file_handle.write(str(os.getpid()))
        lock_file_handle.flush()
        return True
    except (IOError, OSError):
        logger.error("🚫 Another instance of Signal Server is already running!")
        return False

def release_lock():
    """Release the instance lock on shutdown."""
    global lock_file_handle
    if lock_file_handle:
        try:
            lock_file_handle.close()
            os.remove(LOCK_FILE)
        except:
            pass

def run_signal_scan():
    """Background thread for market scanning."""
    global holon_stack

    logger.info("📡 Starting Signal Scanner...")
    if not validate_config(): return

    holon_stack = create_signal_stack(include_overwatch=True)
    if not holon_stack: return
    
    # Initialize Aggregator stack reference
    aggregator.stack = holon_stack

    # Initialize Listing Scouter
    listing_scouter = ListingScouter()
    last_listing_check = 0
    LISTING_CHECK_INTERVAL = 3600 # 1 Hour

    while not shutdown_event.is_set():
        try:
            aggregator.scanning_active = True
            logger.info("Scanning Market...")
            
            if not is_rate_limited('balance_check'):
                try:
                    # Get REAL balance from Kraken - don't use fallback if API returns valid 0
                    live_bal = holon_stack.market.get_balance()
                    live_equity = holon_stack.market.get_equity()
                    
                    # Only fallback to INITIAL_CAPITAL if balance is None (API error)
                    if live_bal is not None:
                        holon_stack.governor.balance = live_bal
                        holon_stack.executor.balance_usd = live_bal
                        logger.info(f"💰 Live Balance: ${live_bal:.2f}")
                    else:
                        holon_stack.governor.balance = config.INITIAL_CAPITAL
                        logger.warning("⚠️ Balance fetch returned None, using INITIAL_CAPITAL")
                    
                    if live_equity is not None:
                        holon_stack.governor.available_balance = live_equity
                        logger.info(f"📊 Live Equity: ${live_equity:.2f}")
                    
                    # === SYNC LIVE POSITIONS FROM KRAKEN ===
                    try:
                        kraken_positions = holon_stack.market.fetch_positions()
                        if kraken_positions:
                            logger.info(f"📈 Synced {len(kraken_positions)} live positions from Kraken")
                            # Sync to governor
                            for pos in kraken_positions:
                                sym = pos.get('symbol', '')
                                size = float(pos.get('contracts') or pos.get('size') or 0)
                                if size == 0:
                                    continue
                                entry = float(pos.get('entryPrice') or pos.get('averagePrice') or 0)
                                side = pos.get('side', 'long')
                                direction = 'BUY' if side == 'long' else 'SELL'
                                
                                # Update governor positions with live data
                                holon_stack.governor.positions[sym] = {
                                    'entry_price': entry,
                                    'quantity': abs(size),
                                    'direction': direction,
                                    'leverage': float(pos.get('leverage', 5.0)),
                                    'strategy': 'LIVE_SYNC',
                                    'unrealized_pnl': float(pos.get('unrealizedPnl', 0))
                                }
                    except Exception as pos_err:
                        logger.debug(f"Position sync skipped: {pos_err}")
                        
                except Exception as e:
                    logger.error(f"Balance/Position sync failed: {e}")

            # === NEW LISTING DISCOVERY ===
            # Run every hour to detect new coins
            if time.time() - last_listing_check > LISTING_CHECK_INTERVAL:
                try:
                    logger.info("🔭 Scanning for New Listings...")
                    
                    # Use Observer for market connectivity (Works in Sim and Live)
                    exchange_source = None
                    if holon_stack.observer and hasattr(holon_stack.observer, 'exchange'):
                        exchange_source = holon_stack.observer.exchange
                    elif holon_stack.market and hasattr(holon_stack.market, 'exchange'):
                        exchange_source = holon_stack.market.exchange
                        
                    if exchange_source:
                        # Force refresh of markets
                        markets = exchange_source.load_markets(reload=True)
                        new_coins = listing_scouter.check_for_new_listings(markets)
                        
                        if new_coins:
                            msg = f"✨ NEW LISTING ALERT: {', '.join(new_coins)}"
                            logger.info(msg)
                            socketio.emit('system_alert', {
                                'level': 'POSITIVE',
                                'message': msg,
                                'timestamp': time.time()
                            })
                    else:
                        logger.warning("Listing Scan skipped: No exchange source available.")
                        
                    last_listing_check = time.time()
                except Exception as e:
                    logger.error(f"Listing Scan Failed: {e}")

            # === ARBITRAGE SYNC ===
            # Explicitly sync arbitrage holon to populate funding yields/spreads
            if holon_stack.holons.get('arbitrage'):
                try:
                    logger.info("⚡ Syncing Arbitrage Data (Funding & Spreads)...")
                    holon_stack.holons['arbitrage'].perform_sync(config.ALLOWED_ASSETS)
                except Exception as arb_sync_err:
                    logger.error(f"Arbitrage Sync Failed: {arb_sync_err}")

            report = holon_stack.signal_provider.generate_signal_report(holon_stack.holons)
            if report:
                # Isolated Telegram delivery - don't let failures break the loop
                try:
                    holon_stack.signal_provider.send_to_telegram(report, overwatch=holon_stack.overwatch)
                except Exception as tg_err:
                    logger.warning(f"⚠️ Telegram delivery failed: {tg_err}")
                
                aggregator.latest_report = report
                # Broadcast immediately via SocketIO
                socketio.emit('signals_update', {'signals': report, 'time': datetime.now().isoformat()})

            aggregator.last_scan_time = datetime.now(timezone.utc).isoformat()
            aggregator.scanning_active = False

            scan_trigger_event.clear()  # Reset trigger

            interval = getattr(config, 'SIGNAL_SCAN_INTERVAL', 300)
            
            # Wait for interval OR trigger event
            # This replaces the sleep loop
            if shutdown_event.is_set(): break
            
            logger.info(f"💤 Sleeping for {interval}s (or until trigger)...")
            scan_trigger_event.wait(timeout=interval)

        except Exception as e:
            logger.error(f"Scan Loop Error: {e}")
            time.sleep(60)

    holon_stack.close()

def construct_system_state():
    """Aggregates all system data into a single, flat state dictionary."""
    return aggregator.collect()

def broadcast_status():
    """Periodically broadcast system state to web clients."""
    logger.info("📢 Starting Status Broadcaster...")
    while not shutdown_event.is_set():
        try:
            state = construct_system_state()
            socketio.emit('hub_state', state)
            time.sleep(1) # Fast broadcast (1s) for responsive UI
        except Exception as e:
            logger.error(f"Broadcaster Error: {e}")
            time.sleep(5)

# --- Routes ---

@app.route('/')
def serve_ui():
    """Serve the React production build or a fallback."""
    if os.path.exists('dashboard-ui/dist/index.html'):
        return send_from_directory('dashboard-ui/dist', 'index.html')
    return render_template_string("<h1>Holon Hub Active</h1><p>UI Build not found. Visit /api/signals for data.</p>")

@app.route('/<path:path>')
def serve_static(path):
    """Serve static assets for React UI."""
    if os.path.exists(f'dashboard-ui/dist/{path}'):
        return send_from_directory('dashboard-ui/dist', path)
    return "Not Found", 404

@app.route('/api/signals')
def get_signals():
    """Get the latest signal report."""
    # Use aggregator for state
    state = aggregator.collect()
    signals = state.get('radar', [])
    return jsonify({
        'status': 'ok', 
        'signals': signals, 
        'last_scan': aggregator.last_scan_time
    })

@app.route('/api/hub/state')
def get_hub_state():
    """Get full aggregated hub state for dashboard initial load."""
    return jsonify(aggregator.collect())

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current system configuration."""
    return jsonify({
        'status': 'ok',
        'config': aggregator.get_config_state()
    })

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update system configuration."""
    try:
        data = request.json
        updates = aggregator.update_config(data)
        return jsonify({
            'status': 'ok',
            'message': 'Configuration updated',
            'updates': updates,
            'config': aggregator.get_config_state()
        })
    except Exception as e:
        logger.error(f"Config Update Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/scan', methods=['POST'])
def trigger_scan():
    """Manually trigger a signal scan."""
    if aggregator.scanning_active:
        return jsonify({'status': 'busy', 'message': 'Scan already in progress'}), 409
    
    scan_trigger_event.set()
    return jsonify({'status': 'ok', 'message': 'Scan triggered successfully'})

@app.route('/api/emergency/panic', methods=['POST'])
def trigger_panic():
    """
    🚨 PANIC BUTTON 🚨
    Triggers emergency shutdown and liquidation.
    """
    logger.critical("🚨 PANIC BUTTON TRIGGERED! INITIATING EMERGENCY SHUTDOWN 🚨")
    
    success = False
    message = "Panic sequence initiated."
    
    if holon_stack and holon_stack.governor:
        try:
            # 1. Lock Governor
            holon_stack.governor.emergency_lock = True
            
            # 2. Halt Trading logic
            # (Assuming Governor respects lock)
            
            # 3. Attempt Liquidation (Best Effort)
            # We trigger a special scan cycle with override or call panic method if available
            # For now, just setting the lock stops *new* trades.
            # To cancel orders:
            if holon_stack.market:
                 holon_stack.market.cancel_all_orders()
                 message += " All open orders cancelled."
            
            success = True
            
            # Broadcast Alert
            socketio.emit('system_alert', {
                'level': 'CRITICAL',
                'message': "🚨 EMERGENCY STOP TRIGGERED BY USER 🚨",
                'timestamp': time.time()
            })
            
        except Exception as e:
            logger.error(f"Panic execution failed: {e}")
            message = f"Panic failed: {str(e)}"
    
    else:
        message = "System not fully initialized, cannot panic properly."
        
    return jsonify({'status': 'ok' if success else 'error', 'message': message})



@app.route('/api/trade', methods=['POST'])
def handle_manual_trade():
    """
    Executes a manual trade via Actuator.
    Expected JSON: { 'symbol': 'BTC/USDT', 'action': 'BUY'|'SELL', 'quantity': float, 'leverage': float }
    """
    if not holon_stack or not holon_stack.executor or not holon_stack.executor.actuator:
        return jsonify({'status': 'error', 'message': 'System not fully initialized (Actuator missing).'}), 503

    try:
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400

        symbol = data.get('symbol', 'BTC/USDT')
        action = data.get('action', 'BUY').upper()
        qty = float(data.get('quantity', 0.0)) # This is in USD for now, or Units? Let's assume Units to match internal API, or handle conversion.
        # User input is likely USD if from UI, but let's stick to Units for Actuator safety, 
        # OR better: Add a 'is_usd' flag.
        is_usd = data.get('is_usd', False)
        leverage = float(data.get('leverage', 1.0))
        
        actuator = holon_stack.executor.actuator
        
        # Resolve Symbol
        # If user passes 'BTC', map to 'BTC/USDT'
        if '/' not in symbol and not symbol.endswith('USD') and not symbol.endswith('USDT'):
             symbol = f"{symbol}/USDT" 
             
        # Resolve Quantity if USD
        final_qty = qty
        current_price = 0.0
        
        # Fetch price for conversion or logging
        try:
             # Try observer first (fast cache)
             if holon_stack.observer:
                 current_price = holon_stack.observer.get_latest_price(symbol)
             
             # Fallback to Exchange
             if current_price <= 0:
                 ticker = actuator.exchange.fetch_ticker(config.KRAKEN_SYMBOL_MAP.get(symbol, symbol))
                 current_price = ticker['last']
        except:
             pass
             
        if is_usd:
             if current_price <= 0:
                 return jsonify({'status': 'error', 'message': f'Cannot calculate quantity: Price for {symbol} unknown.'}), 400
             final_qty = qty / current_price
             logger.info(f"Manual Trade: Converted ${qty} -> {final_qty:.6f} {symbol} @ ${current_price}")

        # Execute
        # We use 'market' order for immediate manual entry
        order_id = actuator.place_order(
            symbol=symbol,
            direction=action,
            quantity=final_qty,
            order_type='market',
            leverage=leverage,
            urgent=True # Manual override implies urgency
        )
        
        if order_id:
             msg = f"Manual {action} {final_qty:.6f} {symbol} Placed. ID: {order_id}"
             socketio.emit('system_alert', {
                'level': 'POSITIVE',
                'message': msg,
                'timestamp': time.time()
            })
             # Force a quick cycle update or notify governor?
             # Governor Loop will pick it up on next cycle (10-60s)
             return jsonify({'status': 'ok', 'message': msg, 'order_id': order_id})
        else:
             return jsonify({'status': 'error', 'message': 'Actuator declined order (Check logs/Liquidity/Funds).'}), 400

    except Exception as e:
        logger.error(f"Manual Trade Failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/positions')
def get_positions():
    """Get current positions with PnL from Governor."""
    state = aggregator.collect()
    return jsonify({
        'positions': state.get('positions', []),
        'prices': state.get('prices', {}),
        'health': state.get('portfolio_health', {}),
        'balance': state.get('equity', 0)
    })

@app.route('/api/analyze/positions')
def analyze_positions_endpoint():
    """
    Detailed Portfolio Analysis:
    - Macro Alignment
    - Net Exposure & Leverage
    - Structural Health (Distance to TP/SL)
    - Concentration Risk
    """
    if not holon_stack or not holon_stack.governor:
        return jsonify({'status': 'error', 'message': 'Governor not initialized'}), 503

    gov = holon_stack.governor
    oracle = holon_stack.oracle
    
    # 1. Get Macro Context
    market_bias = 0.5
    macro_regime = "NEUTRAL"
    if oracle:
        try:
             # Try to get cached or fresh bias
             market_bias = oracle.get_market_bias()
        except:
             pass
             
    if market_bias > 0.6: macro_regime = "BULLISH"
    elif market_bias < 0.4: macro_regime = "BEARISH"

    # 2. Analyze Positions
    positions_data = []
    total_long_value = 0.0
    total_short_value = 0.0
    equity = gov.get_portfolio_health().get('equity', 1.0)
    if equity <= 0: equity = 1.0 # Prevent div/0
    
    prices = dict(gov.latest_prices)

    for sym, pos in gov.positions.items():
        qty = pos.get('quantity', 0)
        if abs(qty) < 0.000001: continue
        
        entry = pos.get('entry_price', 0)
        current = prices.get(sym, entry)
        value = current * abs(qty)
        direction = pos.get('direction', 'BUY')
        
        # Macro Alignment
        aligned = False
        if macro_regime == "BULLISH" and direction == 'BUY': aligned = True
        elif macro_regime == "BEARISH" and direction == 'SELL': aligned = True
        elif macro_regime == "NEUTRAL": aligned = True # Sentimental Neutrality
        
        # Exposure
        if direction == 'BUY': total_long_value += value
        else: total_short_value += value
        
        # Drift
        drift_pct = 0.0
        if entry > 0:
            if direction == 'BUY':
                drift_pct = (current - entry) / entry * 100
            else:
                drift_pct = (entry - current) / entry * 100
        
        # Structural Health (Distance to TP/SL)
        tp_dist = 0.0
        sl_dist = 0.0
        rr_ratio = 0.0
        
        tp = pos.get('take_profit')
        sl = pos.get('stop_loss')
        
        if tp and current > 0:
            tp_dist = abs(tp - current) / current * 100
        if sl and current > 0:
            sl_dist = abs(current - sl) / current * 100
            
        if sl_dist > 0:
            rr_ratio = tp_dist / sl_dist
            
        positions_data.append({
            'symbol': sym,
            'direction': direction,
            'size_usd': round(value, 2),
            'aligned': aligned,
            'drift_pct': round(drift_pct, 2),
            'tp_dist_pct': round(tp_dist, 2) if tp else None,
            'sl_dist_pct': round(sl_dist, 2) if sl else None,
            'rr_ratio': round(rr_ratio, 2) if rr_ratio > 0 else None,
            'leverage': pos.get('leverage', 1.0)
        })

    # 3. Portfolio Metrics
    total_exposure = total_long_value + total_short_value
    net_exposure = total_long_value - total_short_value
    leverage_ratio = total_exposure / equity
    net_exposure_ratio = net_exposure / equity
    
    # Concentration (Largest position % of equity)
    max_concentration = 0.0
    concentrated_asset = None
    if positions_data:
        max_pos = max(positions_data, key=lambda x: x['size_usd'])
        max_concentration = (max_pos['size_usd'] / equity) * 100
        concentrated_asset = max_pos['symbol']

    return jsonify({
        'status': 'ok',
        'macro': {
            'regime': macro_regime,
            'bias_score': round(market_bias, 2)
        },
        'metrics': {
            'equity': round(equity, 2),
            'total_exposure': round(total_exposure, 2),
            'net_exposure': round(net_exposure, 2),
            'leverage_ratio': round(leverage_ratio, 2),
            'net_exposure_ratio': round(net_exposure_ratio, 2),
            'concentration_max_pct': round(max_concentration, 2),
            'concentrated_asset': concentrated_asset
        },
        'positions': positions_data,
        'timestamp': time.time()
    })


@app.route('/api/sitrep')
def get_sitrep():
    """
    📋 SITREP (Situation Report)
    A single-glance operational brief covering:
    1. Signal Structure  - What signals are active and their anatomy
    2. Setup Rationale   - Why each signal fired (confluence factors)
    3. Expected Returns  - Projected PnL, risk/reward, and sizing
    4. Equity Chart      - Mini equity curve for trend context
    """
    state = aggregator.collect()
    gov = holon_stack.governor if holon_stack else None
    
    # --- 1. System Overview ---
    health = state.get('portfolio_health', {})
    system_overview = {
        'balance': round(state.get('equity', 0), 2),
        'risk_budget': round(health.get('risk_budget', 0), 2),
        'fortress_floor': round(health.get('fortress_balance', 0), 2),
        'metabolism': gov.get_metabolism_state() if gov else 'UNKNOWN',
        'regime': state.get('regime', 'UNKNOWN'),
        'drawdown_locked': gov.drawdown_lock if gov else False,
        'risk_multiplier': round(gov.risk_multiplier if gov else 1.0, 2),
        'scanning': state.get('scanning', False),
        'last_scan': state.get('last_scan')
    }

    # --- 2. Signal Structure ---
    signal_briefs = []
    for sig in state.get('radar', []):
        symbol = sig.get('symbol', '?')
        direction = sig.get('direction', '?')
        conviction = sig.get('conviction', 0.0)
        quality = sig.get('quality', 'UNKNOWN')
        reason = sig.get('reason', '')
        price = sig.get('price', 0.0)
        exec_details = sig.get('execution_details', {})
        qty = exec_details.get('quantity', 0.0)
        lev = exec_details.get('leverage', 1.0)
        notional = qty * price if price > 0 else 0.0

        # Rationale transformation
        setup_type = 'UNKNOWN'
        rationale = reason
        if 'BASIS_CARRY' in reason:
            setup_type = 'FUNDING CARRY'
            rationale = f"Funding rate yields profit for holding {direction}. Capture yield while hedged."
        elif 'SPATIAL_ARB' in reason:
            setup_type = 'SPATIAL ARBITRAGE'
            rationale = f"Price spread between exchanges exceeds threshold. Profit from convergence."
        elif 'WHALE_WALL' in reason:
            setup_type = 'WHALE SIGNAL'
            rationale = f"Large order wall detected. Institutional flow suggests {direction} pressure."
        elif 'MOMENTUM' in reason or 'ROCKET' in reason:
            setup_type = 'MOMENTUM'
            rationale = f"Strong directional momentum with high conviction ({conviction:.0%})."
        elif 'MEAN_REVERT' in reason:
            setup_type = 'MEAN REVERSION'
            rationale = f"Extreme deviation from mean. Statistical edge for reversion."
        elif 'CRISIS' in reason:
            setup_type = 'CRISIS HEDGE'
            rationale = f"Market stress detected. Defensive positioning into safe haven."
        elif reason:
            setup_type = reason.replace('_', ' ').title()
            rationale = f"Signal generated by {setup_type} strategy."

        # Returns
        tp = sig.get('tp', sig.get('take_profit'))
        sl = sig.get('sl', sig.get('stop_loss'))
        tp_pct = abs(tp - price) / price * 100 if tp and price > 0 else 0.0
        sl_pct = abs(price - sl) / price * 100 if sl and price > 0 else 0.0
        rr_ratio = tp_pct / sl_pct if sl_pct > 0 else 0.0

        signal_briefs.append({
            'symbol': symbol,
            'direction': direction,
            'price': round(price, 6),
            'conviction': round(conviction, 2),
            'quality': quality,
            'setup_type': setup_type,
            'rationale': rationale,
            'sizing': {
                'quantity': round(qty, 6),
                'leverage': round(lev, 1),
                'notional_usd': round(notional, 2),
                'margin_usd': round(notional / lev, 2) if lev > 0 else 0.0
            },
            'returns': {
                'tp_pct': round(tp_pct, 2),
                'sl_pct': round(sl_pct, 2),
                'rr_ratio': round(rr_ratio, 2),
                'expected_return_usd': round(notional * (tp_pct / 100), 2) if tp_pct > 0 else 0.0,
                'max_loss_usd': round(notional * (sl_pct / 100), 2) if sl_pct > 0 else 0.0
            },
            'raw_reason': reason
        })

    # --- 3. Arbitrage Opportunities ---
    arb_opps = state.get('arbitrage', [])

    # --- 4. Composed SitRep ---
    sitrep = {
        'status': 'ok',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'overview': system_overview,
        'signals': {
            'count': len(signal_briefs),
            'items': signal_briefs
        },
        'positions': {
            'count': len(state.get('positions', [])),
            'unrealized_pnl': round(sum(p.get('pnl', 0) for p in state.get('positions', [])), 2),
            'items': state.get('positions', [])
        },
        'arbitrage': {
            'count': len(arb_opps),
            'items': sorted(arb_opps, key=lambda x: abs(x['funding_apy']), reverse=True)
        },
        'chart': {
            'equity_history': state.get('equity_history', []),
            'points': len(state.get('equity_history', []))
        }
    }
    return jsonify(sitrep)


# --- Lifecycle Management ---

# Thread Registry for watchdog monitoring
thread_registry = {}
WATCHDOG_INTERVAL = 30  # Check thread health every 30 seconds
MAX_THREAD_RESTARTS = 3  # Max restart attempts before giving up

def cleanup_resources():
    """Comprehensive cleanup of all resources."""
    global lock_file_handle
    logger.info("🧹 Starting resource cleanup...")
    
    # 1. Signal all threads to stop
    shutdown_event.set()
    
    # 2. Stop SocketIO gracefully
    if socketio:
        try:
            socketio.stop()
            logger.info("✅ SocketIO stopped")
        except SystemExit:
            pass # Expected during shutdown
        except Exception as e:
            logger.warning(f"SocketIO stop warning: {e}")
    
    # 3. Wait for threads to finish
    for name, thread_info in thread_registry.items():
        thread = thread_info.get('thread')
        if thread and thread.is_alive():
            logger.info(f"⏳ Waiting for {name} thread to finish...")
            thread.join(timeout=5)
            if thread.is_alive():
                logger.warning(f"⚠️ {name} thread did not stop gracefully")
            else:
                logger.info(f"✅ {name} thread stopped")
    
    # 4. Release file lock
    release_lock()
    logger.info("✅ Lock released")
    
    # 5. Clear global state
    global latest_report, equity_history
    latest_report = []
    equity_history = []
    
    logger.info("🏁 Cleanup complete.")

def handle_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    logger.info(f"🛑 Shutdown signal ({sig_name}) received.")
    shutdown_event.set()

def start_thread(name: str, target_func, restart_count: int = 0):
    """Start a thread and register it for watchdog monitoring."""
    thread = threading.Thread(target=target_func, daemon=True, name=name)
    thread.start()
    thread_registry[name] = {
        'thread': thread,
        'target': target_func,
        'restart_count': restart_count,
        'started_at': time.time()
    }
    logger.info(f"🚀 Started thread: {name}")
    return thread

def watchdog():
    """Monitor thread health and restart dead threads."""
    logger.info("🐕 Watchdog started - monitoring thread health...")
    
    while not shutdown_event.is_set():
        try:
            time.sleep(WATCHDOG_INTERVAL)
            
            if shutdown_event.is_set():
                break
                
            for name, info in list(thread_registry.items()):
                thread = info.get('thread')
                target = info.get('target')
                restart_count = info.get('restart_count', 0)
                
                if thread and not thread.is_alive() and not shutdown_event.is_set():
                    if restart_count < MAX_THREAD_RESTARTS:
                        logger.warning(f"⚠️ Thread '{name}' died! Restarting (attempt {restart_count + 1}/{MAX_THREAD_RESTARTS})...")
                        start_thread(name, target, restart_count + 1)
                    else:
                        logger.error(f"❌ Thread '{name}' exceeded max restarts. Manual intervention required.")
                        # Emit alert to dashboard
                        try:
                            socketio.emit('system_alert', {
                                'level': 'CRITICAL',
                                'message': f"Thread '{name}' failed permanently",
                                'timestamp': time.time()
                            })
                        except:
                            pass
                            
        except Exception as e:
            logger.error(f"Watchdog error: {e}")
            time.sleep(10)
    
    logger.info("🐕 Watchdog stopped.")

# Health check route with thread status
@app.route('/health')
def health_check():
    try:
        thread_status = {}
        for name, info in thread_registry.items():
            thread = info.get('thread')
            thread_status[name] = {
                'alive': thread.is_alive() if thread else False,
                'restart_count': info.get('restart_count', 0),
                'uptime_sec': time.time() - info.get('started_at', time.time())
            }
        
        all_healthy = all(s['alive'] for s in thread_status.values())
        
        return jsonify({
            'status': 'healthy' if all_healthy else 'degraded',
            'scanning_active': aggregator.scanning_active,
            'last_scan': aggregator.last_scan_time,
            'uptime_sec': time.time() - start_time,
            'threads': thread_status
        })
    except Exception as e:
        logger.error(f"Health Check Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'status': 'error', 'message': str(e)}), 500

start_time = time.time()

# Register cleanup handlers
atexit.register(cleanup_resources)

if __name__ == "__main__":
    # Enforce single instance
    if not acquire_lock():
        logger.error("Exiting due to existing instance.")
        sys.exit(1)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Windows-specific: Handle CTRL_BREAK_EVENT
    if sys.platform == 'win32':
        try:
            signal.signal(signal.SIGBREAK, handle_shutdown)
        except (AttributeError, ValueError):
            pass

    # Start Worker Threads with registry
    start_thread('signal_scanner', run_signal_scan)
    start_thread('status_broadcaster', broadcast_status)
    start_thread('watchdog', watchdog)

    logger.info(f"🚀 Holon Hub starting on port 5000...")
    logger.info("Press Ctrl+C to shutdown gracefully.")
    
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received.")
    except Exception as e:
        logger.error(f"Server Error: {e}")
    finally:
        cleanup_resources()
        logger.info("🏁 Holon Hub Shutdown Complete.")
