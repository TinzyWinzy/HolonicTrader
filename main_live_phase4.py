"""
HolonicTrader - LIVE Execution Entry Point (Phase 4)
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
from HolonicTrader.agent_ppo import PPOHolon

try:
    from HolonicTrader.agent_telegram import TelegramHolon
except ImportError:
    TelegramHolon = None
    print("⚠️ TelegramHolon module not found using robust import.")

from database_manager import DatabaseManager


from queue import Queue
import threading
import sys
from datetime import datetime
import re

class QueueLogger:
    """Redirects stdout to a Queue for GUI display, plus file logging."""
    def __init__(self, filename, log_queue=None):
        self.terminal = sys.stdout
        self.filename = filename
        self.log_queue = log_queue
        self.log = open(filename, "a", encoding='utf-8')
        self.lock = threading.Lock()
    
    def write(self, message):
        with self.lock:
            # Timestamp logic
            final_msg = message
            if message.strip():
                timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
                if not message.startswith("[20"): 
                    final_msg = f"{timestamp}{message}"
            
            # 1. Print to Real Terminal (Hidden in GUI mode usually, but good for debug)
            # self.terminal.write(final_msg)
            
            # 2. Write to File
            try:
                self.log.write(final_msg)
                self.log.flush()
            except Exception:
                pass # Prevent logging errors from crashing the bot
            
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

def main_live(status_queue: Queue = None, stop_event: threading.Event = None, interval_seconds: int = 60):
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
    
    # 1. Instantiate Core Agents
    observer = ObserverHolon(exchange_id='kucoin')
    entropy = EntropyHolon()
    oracle = EntryOracleHolon()
    guardian = ExitGuardianHolon()
    monitor = MonitorHolon(principal=config.INITIAL_CAPITAL)
    ppo = PPOHolon()
    
    # 2. Instantiate Execution Stack
    governor = GovernorHolon(initial_balance=config.INITIAL_CAPITAL, db_manager=db)
    
    actuator = None
    if not config.PAPER_TRADING:
        print(">>> 🚨 LIVE TRADING ENABLED - REAL MARKET EXECUTION 🚨 <<<")
        actuator = ActuatorHolon()
    else:
        print(">>> 📊 PAPER TRADING MODE ACTIVE - SIMULATED EXECUTION <<<")
        
    executor = ExecutorHolon(
        initial_capital=config.INITIAL_CAPITAL,
        governor=governor,
        actuator=actuator,
        db_manager=db
    )
    
    # 2b. Sync Governor
    governor.sync_positions(executor.held_assets, executor.position_metadata)

    # 2c. Telegram Bot (Robust Instantiation)
    if stop_event is None:
        stop_event = threading.Event()

    telegram = None
    if config.TELEGRAM_ENABLED and TelegramHolon:
        try:
            telegram = TelegramHolon(executor=executor, stop_event=stop_event)
        except Exception as e:
            print(f"⚠️ Telegram Launch Failed: {e}")

    # 3. Instantiate Trader
    trader = TraderHolon("TraderNexus", sub_holons={
        'observer': observer,
        'entropy': entropy,
        'oracle': oracle,
        'guardian': guardian,
        'monitor': monitor,
        'governor': governor,
        'executor': executor,
        'ppo': ppo,
        'telegram': telegram
    })
    
    # 4. Start Loop
    print(">> Initializing System Components...")
    
    try:
        # 4. Integrate Stop Signals & Queue
        trader.gui_queue = status_queue
        trader.gui_stop_event = stop_event
        
        # 5. Start Loop
        print(">> Initializing System Components...")
        trader.start_live_loop(interval_seconds=interval_seconds)
        
    except KeyboardInterrupt:
        print("\n>> STOP REQUEST RECEIVED. SHUTTING DOWN...")
    except Exception as e:
        print(f"\n>> FATAL MAIN LOOP ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(">> SYSTEM SHUTDOWN COMPLETE.")

def run_bot(stop_event, status_queue, config_dict=None):
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
            # We map leverage based on metabolism, but for simplicity we set PREDATOR leverage cap
            config.PREDATOR_LEVERAGE = float(config_dict.get('leverage_cap', config.PREDATOR_LEVERAGE))
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Config Applied: Allocation {config.GOVERNOR_MAX_MARGIN_PCT*100:.1f}%, Leverage {config.PREDATOR_LEVERAGE}x")
            
        # 1. Start Loop (Check if GUI provided a specific interval, else default to 60)
        interval = config_dict.get('loop_interval', 60) if config_dict else 60
        main_live(status_queue, stop_event, interval_seconds=interval)
    except Exception as e:
        print(f"Bot Crashed: {e}")

if __name__ == "__main__":
    # Standalone Mode
    log_file = f"live_trading_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = QueueLogger(log_file)
    main_live()
