"""
HolonicTrader - LIVE Execution Entry Point (Phase 4)

Supports both Python TraderHolon and Rust TraderNexus modes.
Set config.USE_RUST_NEXUS = True to enable Rust mode.
"""

import config
from HolonicTrader.agent_trader import TraderHolon
from HolonicTrader.holon_core import Disposition
from HolonicTrader.agent_observer import ObserverHolon
from HolonicTrader.agent_diagnostic import DiagnosticHolon
from HolonicTrader.agent_entropy import EntropyHolon
from HolonicTrader.agent_oracle import EntryOracleHolon
from HolonicTrader.agent_guardian import ExitGuardianHolon
from HolonicTrader.agent_monitor import MonitorHolon
from HolonicTrader.agent_governor import GovernorHolon
from HolonicTrader.agent_executor import ExecutorHolon
from HolonicTrader.agent_actuator import ActuatorHolon
from HolonicTrader.agent_ppo import PPOHolon
from HolonicTrader.agent_sentiment import SentimentHolon
from HolonicTrader.agent_overwatch import OverwatchHolon
from HolonicTrader.agent_whale import WhaleHolon
from HolonicTrader.agent_structure import CTKSStrategicHolon
from HolonicTrader.agent_arbitrage import ArbitrageHolon
from HolonicTrader.agent_signal_provider import SignalProviderHolon
from HolonicTrader.agent_dump_pump_detector import DumpPumpDetectorHolon  # Whale dump/pump detection
from HolonicTrader.agent_kraken import KrakenHolon
from wfo_engine import WalkForwardOptimizer

# ATLAS PROFIT ARCHITECT INTEGRATION
from atlas_integration import AtlasProfitIntegration
from HolonicTrader.market_real import RealMarketHolon
from HolonicTrader.market_sim import SimulationMarketHolon

# QUANT-OPS MULTI-AGENT ARCHITECTURE
from HolonicTrader.agent_quantops import QuantOpsHolon

# NEW: Rust TraderNexus (Phase 4 Rust Core)
try:
    from HolonicTrader.rust_trader_nexus import RustTraderNexus, create_rust_nexus
    RUST_NEXUS_AVAILABLE = True
except ImportError:
    RUST_NEXUS_AVAILABLE = False
    print(">> [Warning] Rust TraderNexus not available, falling back to Python TraderHolon")

# NEW: Gold Arbitrage Holons (Phase 2026-02-23)
from HolonicTrader.agent_gold_lead_lag import GoldLeadLagHolon
from HolonicTrader.agent_paxg_btc import PaxgBtcHolon

# NEW: Exponential Growth Engine (xStocks Arbitrage Auto-Compounding)
from HolonicTrader.holon_exponential_growth import ExponentialGrowthHolon
from HolonicTrader.strategy_xstocks_arb import XStocksArbitrage, scan_xstocks_arb
from HolonicTrader.growth_integration import initialize_growth_engine, get_growth_engine, get_current_allocation


from database_manager import DatabaseManager

from queue import Queue
import threading
import sys
import os
import logging
from datetime import datetime
import re

# FIX: Enable UTF-8 encoding for all logging on Windows
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Configure root logger with UTF-8 encoding
from utf8_logging import configure_root_logger, UTF8StreamHandler
configure_root_logger(logging.INFO)

# Helper function for xStocks summary (avoids unicode issues)
def get_xstocks_summary_safe():
    """Safe wrapper for xStocks summary"""
    try:
        from HolonicTrader.strategy_xstocks_arb import get_xstocks_summary
        return get_xstocks_summary()
    except Exception as e:
        return f"Could not fetch summary: {e}"

class QueueLogger:
    """Redirects stdout to a Queue for GUI display, plus file logging."""
    def __init__(self, filename, log_queue=None):
        self.terminal = sys.stdout
        self.filename = filename
        self.log_queue = log_queue
        self.lock = threading.Lock()
        # Open log file with UTF-8 encoding and 'replace' error handling
        self.log = open(filename, "a", encoding='utf-8', errors='replace')

    def write(self, message):
        with self.lock:
            # Timestamp logic
            final_msg = message
            if message.strip():
                timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
                if not message.startswith("[20"):
                    final_msg = f"{timestamp}{message}"

            # 1. Print to Real Terminal (Hidden in GUI mode usually, but good for debug)
            # Encode to UTF-8 first, then decode with 'replace' for console safety
            try:
                console_msg = final_msg.encode('utf-8').decode('utf-8', errors='replace')
                self.terminal.write(console_msg)
            except Exception:
                # Fallback: ASCII with replace for problematic consoles
                try:
                    ascii_msg = final_msg.encode('ascii', errors='replace').decode('ascii', errors='replace')
                    self.terminal.write(ascii_msg)
                except:
                    pass

            # 2. Write to File with UTF-8 encoding (already opened with utf-8)
            try:
                self.log.write(final_msg)
                self.log.flush()
            except Exception as e:
                # Prevent logging errors from crashing the bot
                try:
                    # Last resort: ASCII fallback
                    ascii_msg = final_msg.encode('ascii', errors='replace').decode('ascii', errors='replace')
                    self.log.write(ascii_msg)
                    self.log.flush()
                except:
                    pass

            # 3. Push to Queue (if exists)
            if self.log_queue:
                try:
                    # Strip ANSI codes for GUI display using regex
                    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                    clean_msg = ansi_escape.sub('', final_msg)

                    self.log_queue.put({
                        'type': 'log',
                        'message': clean_msg
                    }, block=False)
                except Exception:
                    pass # Queue full or closed

    def flush(self):
        try:
            self.terminal.flush()
            self.log.flush()
        except Exception:
            pass

def main_live(status_queue: Queue = None, stop_event: threading.Event = None, interval_seconds: int = 60, command_queue: Queue = None, disable_telegram: bool = False):
    print("==========================================")
    print("   HOLONIC TRADER - LIVE ENVIRONMENT      ")
    print("==========================================")
    
    # 0. Initialize Database
    db = DatabaseManager()

    # 0b. System Diagnostics
    diagnostic = DiagnosticHolon()
    if not diagnostic.run_system_check(db):
        print(">> 🛑 SYSTEM CHECK FAILED. HALTING STARTUP.")
        return

    # 0c. Capital Synchronization (Live & Paper)
    # Allows Paper Trading to start with REAL equity for realistic simulation
    should_sync = True # Always try to sync if possible
    if should_sync:
        try:
            import ccxt
            print(f">> 🔄 Syncing Capital from Kraken ({config.TRADING_MODE})...")
        
            if config.TRADING_MODE == 'FUTURES':
                exchange_class = ccxt.krakenfutures
                # Use specific Futures keys if available
                api_key = config.KRAKEN_FUTURES_API_KEY or config.API_KEY
                api_secret = config.KRAKEN_FUTURES_PRIVATE_KEY or config.API_SECRET
            else:
                exchange_class = ccxt.kraken
                api_key = config.API_KEY
                api_secret = config.API_SECRET
                
            exchange = exchange_class({'apiKey': api_key, 'secret': api_secret})
            bal = exchange.fetch_balance()
            info = bal.get('info', {})
            
            real_equity = 0.0
            
            if config.TRADING_MODE in ['FUTURES', 'DUAL']:
                # Futures Equity Check (Multi-Collateral 'flex')
                accounts = info.get('accounts', {})
                flex = accounts.get('flex', {})
                real_equity = float(flex.get('marginEquity', 0.0))
                
                if real_equity <= 0:
                     # Fallback to cash USD if no margin account
                     real_equity = bal.get('total', {}).get('USD', 0.0)
            else:
                # Spot Equity Check
                real_equity = float(info.get('eb', 0.0))
                if real_equity <= 0: real_equity = float(info.get('tb', 0.0))
                if real_equity <= 0: 
                     real_equity = bal['free'].get('USD', 0.0) + bal['free'].get('USDT', 0.0)
            
            if real_equity > 5.0: # Sanity check
                 print(f">> 💰 SYNC SUCCESS: Real Equity ${real_equity:.2f}")
                 config.INITIAL_CAPITAL = real_equity
                 print(f"   -> Set INITIAL_CAPITAL = ${config.INITIAL_CAPITAL:.2f}")
            else:
                 print(f">> ⚠️ Exchange Balance too low (${real_equity:.2f}), using Config Default (${config.INITIAL_CAPITAL}).")
    
        except Exception as e:
            print(f">> ⚠️ Capital Sync Failed: {e}. Using Config Defaults.")

    
    from HolonicTrader.agent_topology import TopologyHolon # <--- NEW: Structure Brain

    # NEW: Flexline Credit Facility (2026-03-09)
    flexline_enabled = getattr(config, 'FLEXLINE_ENABLED', False)
    flexline_agent = None
    if flexline_enabled:
        print(">> 💳 Initializing Kraken Flexline Credit Facility...")
        try:
            flexline_agent = FlexlineAgent(name="FlexlineAgent")
            flexline_agent.sync_credit_line()
            print(f">> ✅ Flexline Agent initialized: ${flexline_agent.available_credit:.2f} available credit")
        except Exception as e:
            print(f">> ⚠️ Flexline Agent initialization failed: {e}")
            flexline_agent = None

    # 1. Instantiate Core Agents
    observer = ObserverHolon(exchange_id='kucoin')
    kraken_observer = ObserverHolon(exchange_id='krakenfutures')
    entropy = EntropyHolon()
    topology = TopologyHolon() # <--- AEHML 2.0
    oracle = EntryOracleHolon()
    guardian = ExitGuardianHolon()
    monitor = MonitorHolon(principal=config.PRINCIPAL)
    ppo = PPOHolon()
    sentiment = SentimentHolon()
    whale = WhaleHolon()
    structure = CTKSStrategicHolon()
    arbitrage = ArbitrageHolon()
    kraken_intel = KrakenHolon()
    signal_provider = SignalProviderHolon()  # <--- NEW: Signal Report Generator
    dump_pump_detector = DumpPumpDetectorHolon()  # <--- Whale Dump/Pump Time-Window Detector
    
    # NEW: Gold Arbitrage Holons (Phase 2026-02-23)
    gold_lead_lag = GoldLeadLagHolon(name="GoldLeadLag")
    paxg_btc = PaxgBtcHolon(name="PaxgBtc")
    
    # Inject gold holons into signal provider
    signal_provider.gold_lead_lag = gold_lead_lag
    signal_provider.paxg_btc = paxg_btc

    arbitrage.kucoin_observer = observer
    arbitrage.kraken_observer = kraken_observer

    # FLEXLINE INJECTION (2026-03-09): Inject Flexline agent into Governor and Arbitrage
    if flexline_agent:
        arbitrage.flexline_agent = flexline_agent
        print(f">> 💳 Flexline injected into ArbitrageHolon")

    # FIX 2026-03-02: REMOVED kraken_spot_observer
    # xStocks are ONLY on Kraken Futures, NOT on Spot
    kraken_spot_observer = None

    # --- PHASE 46.2: ACTIVATE HIGH-FREQUENCY STREAMS ---
    # Kucoin doesn't support xStocks, filter them out
    kucoin_assets_only = [a for a in config.ALLOWED_ASSETS if a not in getattr(config, 'XSTOCKS_SYMBOLS', [])]
    observer.start_ws(kucoin_assets_only)
    kraken_observer.start_ws(config.ALLOWED_ASSETS)
    # REMOVED: kraken_spot_observer.start_ws() - xStocks are on Futures only
    
    # 2. Instantiate Execution Stack
    governor = GovernorHolon(initial_balance=config.INITIAL_CAPITAL, db_manager=db)

    # FLEXLINE INJECTION (2026-03-09): Inject Flexline agent into Governor
    if flexline_agent:
        governor.flexline_agent = flexline_agent
        print(f">> 💳 Flexline injected into GovernorHolon (Position Sizing Boost Enabled)")

    # Actuator (Execution)
    actuator = ActuatorHolon(name="ActuatorAgent", exchange_id='krakenfutures' if config.TRADING_MODE == 'FUTURES' else 'kraken', paper_mode=config.PAPER_TRADING)
    
    print(f">>> {'🧪 SIMULATION' if config.MARKET_HOLON_TYPE == 'SIMULATION' else '🚨 REAL MARKET'} MODE ACTIVE <<<")
    if config.MARKET_HOLON_TYPE == 'SIMULATION':
        market = SimulationMarketHolon(initial_capital=config.INITIAL_CAPITAL)
    else:
        market = RealMarketHolon()
        
    executor = ExecutorHolon(
        initial_capital=config.INITIAL_CAPITAL,
        governor=governor,
        market=market,
        db_manager=db,
        actuator=actuator, # Inject Actuator
        gui_queue=status_queue # NEW: Dashboard Link
    )
    
    # 2b. Sync Governor & Exchange
    executor.reconcile_exchange_positions() 
    
    # --- OPTIMIZED BALANCE SYNC (Phase 16) ---
    if market:
        # Try to get live balance from exchange
        # FIX 2026-03-10: Use get_equity instead of get_balance (availableMargin) to prevent startup drift swings
        live_bal = market.get_equity()
        if live_bal and live_bal > 0:
            executor.sync_balance(live_bal)
        else:
            # Fallback to DB state
            executor.sync_balance(executor.balance_usd)
    else:
        # Paper Trading: Trust the DB (restored in executor.__init__) over hardcoded config
        executor.sync_balance(executor.balance_usd)
    # -----------------------------------------

    governor.sync_positions(executor.held_assets, executor.position_metadata)

    # FIX 2026-03-04: Auto-clear DEFENSIVE cooldown on fresh startup.
    # The cooldown may have been triggered by a false positive (stale disk equity > live equity
    # causing a phantom drawdown calculation at day reset). Clear it here so the bot can trade
    # immediately. If a REAL drawdown triggered it, the doctrine will re-engage within the first
    # cycle when update() checks current equity against day_start_equity.
    print(">> 🛡️ Checking SMCE DEFENSIVE cooldown state on startup...")
    governor.clear_defensive_cooldown("Startup auto-clear (stale-state false positive guard)")

    # FIX 2026-03-05: Reset drawdown baselines to live equity on every startup.
    # Persisted day_start_equity / daily_start_balance can be stale, triggering a false
    # FEVER lockdown (4h hibernation) immediately on restart before any trade is made.
    # Use live_bal (confirmed exchange equity from market.get_balance() above).
    # initial_equity is not yet assigned at this point in the startup sequence.
    _live_equity = (live_bal if (live_bal and live_bal > 0) else executor.balance_usd) if 'live_bal' in dir() else executor.balance_usd

    if _live_equity > 0:
        print(f">> 🔄 Resetting drawdown baselines to live equity ${_live_equity:.2f}...")

        # 1. Reset MonitorHolon (controls FEVER / 4h lockdown)
        if monitor:
            monitor.daily_start_balance = _live_equity
            import time as _time
            monitor.last_day_reset = _time.time()
            monitor._save_state()
            monitor.is_system_healthy = True
            print(f">> ✅ Monitor daily baseline reset → ${_live_equity:.2f}")

        # 2. Reset SMCECapitalDoctrine (controls 3% daily DD → 48h DEFENSIVE cooldown)
        if governor and getattr(governor, 'smce_doctrine', None):
            governor.smce_doctrine.reset_baselines_now(
                _live_equity,
                reason=f"Startup baseline sync (live equity ${_live_equity:.2f})"
            )
            print(f">> ✅ SMCE doctrine baselines reset → ${_live_equity:.2f}")
    else:
        print(">> ⚠️ Could not reset drawdown baselines: live equity unknown at startup")

    # === ATLAS PROFIT ARCHITECT INITIALIZATION ===
    # Initialize Atlas Profit System for trade filtering and capital efficiency
    atlas = None
    try:
        atlas = AtlasProfitIntegration()
        atlas_initial_equity = executor.balance_usd if executor.balance_usd > 0 else config.INITIAL_CAPITAL
        success, message = atlas.initialize_with_account(atlas_initial_equity)
        if success:
            print(f">> ✅ ATLAS PROFIT ARCHITECT initialized: ${atlas_initial_equity:.2f}")
            print(f"   Phase: {atlas.profit_filter.config['profit_phase']}, Min Edge: {atlas.profit_filter.config['minimum_edge_pct']*100:.2f}%")
        else:
            print(f">> ⚠️ Atlas initialization failed: {message}")
            atlas = None
    except Exception as e:
        print(f">> ⚠️ Atlas initialization error: {e}")
        atlas = None
    # ================================================

    # === EXPONENTIAL GROWTH ENGINE INITIALIZATION ===
    if getattr(config, 'EXPONENTIAL_GROWTH_MODE', False):
        initial_equity = executor.balance_usd if executor.balance_usd > 0 else config.INITIAL_CAPITAL
        initialize_growth_engine(governor, executor, initial_equity=initial_equity)
    # ================================================

    # 2d. Overwatch (The Sentry: Telegram + NLP)
    try:
        overwatch = OverwatchHolon()
    except Exception as e:
        print(f">> [Warning] Overwatch failed to start: {e}")
        overwatch = None

    # 2e. Regime Controller
    from HolonicTrader.agent_regime import RegimeController
    regime_controller = RegimeController()
    governor.regime_controller = regime_controller

    # ======================================================================
    # === AEGIS QUANTSEC SECURITY FRAMEWORK INITIALIZATION ===
    # ======================================================================
    # Initialize all security components (Phases 1-4)
    # Addresses critical findings: C-01 (Ledger divergence), C-02 (Timing oracle),
    # C-03 (RL manipulation)
    aegis = None
    try:
        from HolonicTrader.aegis_integration import initialize_aegis_security

        # Get telegram bot from overwatch if available
        telegram_bot = overwatch.telegram if overwatch and hasattr(overwatch, 'telegram') else None

        aegis = initialize_aegis_security(
            executor=executor,
            governor=governor,
            kraken_holon=kraken_intel,
            trader=None,  # Will be set after trader creation
            telegram_bot=telegram_bot,
            chat_id=config.TELEGRAM_CHAT_ID if hasattr(config, 'TELEGRAM_CHAT_ID') else None,
            enable_all=True
        )
    except Exception as e:
        print(f">> [AEGIS] Initialization error: {e}")
        aegis = None
    # ======================================================================

    # ======================================================================
    # === QUANT-OPS MULTI-AGENT INTELLIGENCE LAYER ===
    # ======================================================================
    quantops = None
    if getattr(config, 'QUANTOPS_ENABLED', True):
        try:
            quantops = QuantOpsHolon(
                name="QuantOpsAgent",
                cycle_interval=getattr(config, 'QUANTOPS_CYCLE_INTERVAL', 5),
                memory_depth=getattr(config, 'QUANTOPS_MEMORY_DEPTH', 10),
                db_path=db.db_path,
                log_dir=".",
                output_dir=getattr(config, 'QUANTOPS_OUTPUT_DIR', 'quant_ops_reports'),
                governor=governor,
                executor=executor,
                atlas_integration=atlas,
                capital_manager=atlas.capital_manager if atlas else None,
                db_manager=db,
                aegis_components=aegis if aegis else None,
            )
            print(f">> \U0001f9e0 QUANT-OPS Multi-Agent Intelligence initialized")
            print(f"   Cycle interval: every {config.QUANTOPS_CYCLE_INTERVAL} trades")
            print(f"   Agents: Chronos (forensics) | Aegis (security) | Helix (repair) | Atlas (strategy)")
        except Exception as e:
            print(f">> [QUANT-OPS] Initialization error: {e}")
            quantops = None
    # ======================================================================

    # === PHASE 4: RUST TRADER NEXUS MODE ===
    use_rust_nexus = getattr(config, 'USE_RUST_NEXUS', False) and RUST_NEXUS_AVAILABLE
    
    if use_rust_nexus:
        print(">> ==========================================")
        print(">>    RUST TRADER NEXUS MODE ACTIVATED      ")
        print(">> ==========================================")
        
        # Create Rust TraderNexus
        rust_nexus = create_rust_nexus(
            initial_capital=executor.balance_usd if executor.balance_usd > 0 else config.INITIAL_CAPITAL,
            max_positions=8,
            leverage=config.PREDATOR_LEVERAGE,
            stop_loss_pct=config.DEFAULT_STOP_LOSS_PCT,
            take_profit_pct=config.DEFAULT_TAKE_PROFIT_PCT,
            cycle_interval_ms=interval_seconds * 1000,
        )
        
        if rust_nexus:
            print(">> [Rust Nexus] Initialized successfully")
            rust_nexus.start()
        else:
            print(">> [Rust Nexus] Failed to initialize, falling back to Python")
            use_rust_nexus = False
    # =======================================

    # 3. Instantiate Trader (Python mode fallback)
    primary_observer = kraken_observer if config.TRADING_MODE == 'FUTURES' else observer

    if not use_rust_nexus:
        # Python TraderHolon mode
        trader = TraderHolon("TraderNexus", sub_holons={
            'observer': primary_observer,
            'entropy': entropy,
            'topology': topology,
            'oracle': oracle,
            'guardian': guardian,
            'monitor': monitor,
            'governor': governor,
            'executor': executor,
            'ppo': ppo,
            'sentiment': sentiment,
            'overwatch': overwatch,
            'regime': regime_controller,
            'whale': whale,
            'structure': structure,
            'arbitrage': arbitrage,
            'kraken': kraken_intel,
            'signal_provider': signal_provider,
            'dump_pump_detector': dump_pump_detector,  # Whale dump/pump detector
            'quantops': quantops,  # QUANT-OPS Multi-Agent Intelligence
        })

        # ==================================================================
        # AEGIS PHASE 4: Wrap RL agents after trader creation
        # ==================================================================
        if aegis and aegis.get('enabled'):
            try:
                from HolonicTrader.rl_agent_security import wrap_dqn_agent, wrap_ppo_agent

                # Wrap DQN if present
                if hasattr(trader, 'dqn') and trader.dqn:
                    print(">> [AEGIS Phase 4] Wrapping DQN agent...")
                    trader.dqn = wrap_dqn_agent(trader.dqn, enable_all_features=True)

                # Wrap PPO if present
                if hasattr(trader, 'ppo') and trader.ppo:
                    print(">> [AEGIS Phase 4] Wrapping PPO agent...")
                    trader.ppo = wrap_ppo_agent(trader.ppo, enable_all_features=True)

                # Store trader reference in aegis for access
                aegis['trader'] = trader

                print(">> [AEGIS] RL Agent Security wrapping complete")
            except Exception as e:
                print(f">> [AEGIS] RL agent wrapping error: {e}")
        # ==================================================================

        # ==================================================================
        # ATLAS PROFIT ARCHITECT: Inject into Trader AND Governor
        # ==================================================================
        if atlas and trader:
            trader.atlas = atlas
            print(">> [Atlas] Profit Architect injected into TraderNexus")
            print(f"   Capital Allocation: BUY={atlas.capital_manager.capital_allocation.get('buy_strategy', 0):.0f}, RESERVE={atlas.capital_manager.capital_allocation.get('reserve', 0):.0f}")
        # Also inject into Governor so trader_entry_handler can access it
        if atlas and governor:
            governor.atlas = atlas
            governor.atlas_available = True
            print(">> [Atlas] Profit Architect injected into Governor")
        # ==================================================================

        # ==================================================================
        # QUANT-OPS: Inject AEGIS components after trader creation
        # ==================================================================
        if quantops and aegis and aegis.get('enabled'):
            quantops.receive_message('main', {'type': 'inject_aegis', 'components': aegis})
            print(">> [QUANT-OPS] AEGIS security components linked to QuantOps")
        if quantops:
            quantops._governor = governor  # Ensure latest governor ref
            print(">> [QUANT-OPS] Governor linked for constraint feedback")
        # ==================================================================

        # Initialize and Start WFO Engine
        if getattr(config, 'WFO_ENABLED', True):
            wfo = WalkForwardOptimizer(observer=primary_observer)
            wfo.start(cycle_hours=getattr(config, 'WFO_CYCLE_HOURS', 4.0))
            trader.wfo_engine = wfo
    else:
        trader = None  # Using Rust Nexus instead

    trader_ref_linked = False
    try:
        # 4. Link Overwatch to Trader (Python mode only)
        if overwatch and trader:
            overwatch.trader = trader
            trader_ref_linked = True
            print(">> [System] Overwatch Linked to TraderNexus.")

        # 4b. Integrate Stop Signals & Queue (Python mode)
        if trader:
            trader.gui_queue = status_queue
            trader.gui_stop_event = stop_event
            trader.command_queue = command_queue

        # 5. Start Loop
        print(">> Initializing System Components...")
        
        if use_rust_nexus and rust_nexus:
            # === RUST NEXUS MODE ===
            print(">> [Rust Nexus] Starting trading loop...")
            
            # Rust Nexus runs its own internal loop
            # For now, we keep the Python agents running for support functions
            # The Rust Nexus handles entry/exit decisions
            
            # Run Python support loop (WFO, Overwatch, etc.)
            if trader:
                trader.start_live_loop(interval_seconds=interval_seconds)
            else:
                # Minimal support loop for Rust mode
                import time
                while not stop_event.is_set() if stop_event else True:
                    # Run Rust cycle
                    if rust_nexus.is_running:
                        # Fetch market data and run cycle
                        # (This would be expanded with actual market data fetching)
                        status = rust_nexus.get_status()
                        logger.info(f"Rust Nexus Status: {status}")
                    
                    time.sleep(interval_seconds)
        else:
            # === PYTHON TRADERHOLON MODE ===
            trader.start_live_loop(interval_seconds=interval_seconds)

    except KeyboardInterrupt:
        print("\n>> STOP REQUEST RECEIVED. SHUTTING DOWN...")
    except Exception as e:
        print(f"\n>> FATAL MAIN LOOP ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Proper Resource Cleanup
        print(">> Cleaning up resources...")

        # Stop AEGIS components
        if aegis and aegis.get('enabled'):
            print(">> [AEGIS] Stopping security components...")
            try:
                if aegis.get('reconciliation_engine'):
                    aegis['reconciliation_engine'].stop()
                if aegis.get('log_manager'):
                    # Final anchor before shutdown
                    aegis['log_manager'].create_anchor()
                    print(">> [AEGIS] Final log anchor created")
            except Exception as e:
                print(f">> [AEGIS] Shutdown error: {e}")

        # Stop Rust Nexus
        if use_rust_nexus and rust_nexus:
            print(">> [Rust Nexus] Stopping...")
            rust_nexus.stop()
        
        try:
            if overwatch:
                overwatch.stop()
            if 'market' in locals() and market:
                pass
        except Exception:
             pass

        # Explicit State Save
        try:
             if 'executor' in locals() and executor:
                 executor.save_state()
        except Exception as e:
             print(f"Error saving state: {e}")

        try:
            db.close()
        except Exception:
            pass
        print(">> SYSTEM SHUTDOWN COMPLETE.")

def run_bot(stop_event, status_queue, config_dict=None, command_queue=None, disable_telegram=False):
    """Wrapper for GUI Thread"""
    # Setup Logger
    log_file = f"live_trading_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = QueueLogger(log_file, log_queue=status_queue)
    
    try:
        # Update Config from GUI if provided
        if config_dict:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Applying Dashboard Config...")
            
            # Map GUI symbols to config
            gui_symbol = config_dict.get('symbol')
            if gui_symbol and gui_symbol not in config.ALLOWED_ASSETS:
                # Add the selected symbol to the universe if it's not there
                config.ALLOWED_ASSETS.append(gui_symbol)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Added {gui_symbol} to Asset Universe.")
                
            # Ensure uniqueness
            config.ALLOWED_ASSETS = list(set(config.ALLOWED_ASSETS))
            
            # Dynamic leverage and allocation
            config.GOVERNOR_MAX_MARGIN_PCT = float(config_dict.get('max_allocation', config.GOVERNOR_MAX_MARGIN_PCT))
            config.PREDATOR_LEVERAGE = float(config_dict.get('leverage_cap', config.PREDATOR_LEVERAGE))
            
            # Dynamic Micro Mode
            if 'micro_mode' in config_dict:
                config.MICRO_CAPITAL_MODE = bool(config_dict['micro_mode'])
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Config Applied: Allocation {config.GOVERNOR_MAX_MARGIN_PCT*100:.1f}%, Leverage {config.PREDATOR_LEVERAGE}x, MicroMode: {config.MICRO_CAPITAL_MODE}")
            
        # 1. Start Loop (Check if GUI provided a specific interval, else default to 60)
        interval = config_dict.get('loop_interval', 60) if config_dict else 60
        main_live(status_queue, stop_event, interval_seconds=interval, command_queue=command_queue, disable_telegram=disable_telegram)
    except Exception as e:
        print(f"Bot Crashed: {e}")

if __name__ == "__main__":
    # Standalone Mode
    log_file = f"live_trading_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = QueueLogger(log_file)
    main_live()
