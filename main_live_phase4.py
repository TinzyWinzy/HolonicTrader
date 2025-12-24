"""
HolonicTrader - LIVE Execution Entry Point (Phase 4)
"""

import config
from HolonicTrader.agent_trader import TraderHolon
from HolonicTrader.holon_core import Disposition
from HolonicTrader.agent_observer import ObserverHolon
from HolonicTrader.agent_entropy import EntropyHolon
from HolonicTrader.agent_oracle import EntryOracleHolon
from HolonicTrader.agent_guardian import ExitGuardianHolon
from HolonicTrader.agent_monitor import MonitorHolon
from HolonicTrader.agent_governor import GovernorHolon
from HolonicTrader.agent_executor import ExecutorHolon
from HolonicTrader.agent_actuator import ActuatorHolon
from HolonicTrader.agent_ppo import PPOHolon

from database_manager import DatabaseManager


from queue import Queue
import threading
import sys
from datetime import datetime

class QueueLogger:
    """Redirects stdout to a Queue for GUI display, plus file logging."""
    def __init__(self, filename, log_queue=None):
        self.terminal = sys.stdout
        self.filename = filename
        self.log_queue = log_queue
        self.log = open(filename, "a", encoding='utf-8')
    
    def write(self, message):
        # Timestamp logic
        final_msg = message
        if message.strip():
            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
            if not message.startswith("[20"): 
                final_msg = f"{timestamp}{message}"
        
        # 1. Print to Real Terminal (Hidden in GUI mode usually, but good for debug)
        # self.terminal.write(final_msg) # Optional: disable if spammy in console
        
        # 2. Write to File
        self.log.write(final_msg)
        self.log.flush()
        
        # 3. Push to Queue (if exists)
        if self.log_queue:
            self.log_queue.put({
                'type': 'log',
                'message': final_msg
            })

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main_live(status_queue: Queue = None, stop_event: threading.Event = None, interval_seconds: int = 60):
    print("==========================================")
    print("   HOLONIC TRADER - LIVE ENVIRONMENT      ")
    print("==========================================")
    
    # 0. Initialize Database
    db = DatabaseManager()
    
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
    
    # 3. Instantiate Trader
    trader = TraderHolon("TraderNexus", sub_holons={
        'observer': observer,
        'entropy': entropy,
        'oracle': oracle,
        'guardian': guardian,
        'monitor': monitor,
        'governor': governor,
        'executor': executor,
        'ppo': ppo
    })
    
    # 4. Start Loop
    print(">> Initializing System Components...")
    
    # Pass GUI controls to the Trader if supported (we need to mod Trader to support stop_event)
    # For now, we rely on Trader checking a flag or just killing the thread (messy).
    # Better: Update Trader to support external stop signal.
    
    # Inject Queue into Trader for Summary Reporting?
    # We can monkey-patch or subclass. 
    # Let's attach the queue to the Trader instance so it can push reports.
    trader.gui_queue = status_queue
    trader.gui_stop_event = stop_event
    
    trader.start_live_loop(interval_seconds=interval_seconds) 

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
                
            # Dynamic leverage and allocation
            config.GOVERNOR_MAX_MARGIN_PCT = config_dict.get('max_allocation', config.GOVERNOR_MAX_MARGIN_PCT)
            # We map leverage based on metabolism, but for simplicity we set PREDATOR leverage cap
            config.PREDATOR_LEVERAGE = config_dict.get('leverage_cap', config.PREDATOR_LEVERAGE)
            
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
