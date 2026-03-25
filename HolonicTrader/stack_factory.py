"""
stack_factory.py - Centralized Holonic Stack Initialization
Phase 49: Consolidation
"""

import logging
from typing import Dict, Any, Optional

import config
from HolonicTrader.agent_observer import ObserverHolon
from HolonicTrader.agent_entropy import EntropyHolon
from HolonicTrader.agent_topology import TopologyHolon
from HolonicTrader.agent_oracle import EntryOracleHolon
from HolonicTrader.agent_guardian import ExitGuardianHolon
from HolonicTrader.agent_governor import GovernorHolon
from HolonicTrader.agent_executor import ExecutorHolon
from HolonicTrader.market_real import RealMarketHolon
from HolonicTrader.agent_actuator import ActuatorHolon
from HolonicTrader.market_sim import SimulationMarketHolon
from HolonicTrader.agent_sentiment import SentimentHolon
from HolonicTrader.agent_whale import WhaleHolon
from HolonicTrader.agent_structure import CTKSStrategicHolon
from HolonicTrader.agent_arbitrage import ArbitrageHolon
from HolonicTrader.agent_overwatch import OverwatchHolon
from HolonicTrader.agent_signal_provider import SignalProviderHolon
from HolonicTrader.agent_regime import RegimeController
from HolonicTrader.agent_memory import MemoryHolon
from HolonicTrader.agent_ppo import PPOHolon
from HolonicTrader.agent_doomsday import DoomsdayHolon
from HolonicTrader.agent_kraken import KrakenHolon
from database_manager import DatabaseManager

logger = logging.getLogger("StackFactory")

class HolonStack:
    """
    Encapsulates a fully initialized and linked holon stack.
    """
    def __init__(self, db: DatabaseManager, market: Any, holons: Dict[str, Any]):
        self.db = db
        self.market = market
        self.holons = holons
        
        # Shortcuts for common holons
        self.observer = holons.get('observer')
        self.oracle = holons.get('oracle')
        self.governor = holons.get('governor')
        self.executor = holons.get('executor')
        self.signal_provider = holons.get('signal_provider')
        self.overwatch = holons.get('overwatch')

    def close(self):
        """Cleanup resources."""
        if self.db and hasattr(self.db, 'close'):
            try:
                self.db.close()
                logger.info("Database connection closed.")
            except Exception as e:
                logger.error(f"Error closing database: {e}")
        
        # Cleanup other holons if they have close methods
        for name, holon in self.holons.items():
            if holon and hasattr(holon, 'close') and callable(holon.close):
                try:
                    holon.close()
                except Exception as e:
                    logger.error(f"Error closing holon {name}: {e}")

def create_signal_stack(include_overwatch: bool = True) -> Optional[HolonStack]:
    """
    Initializes and links all holons required for signal generation.
    """
    try:
        # 1. Database
        db = DatabaseManager()
        
        # 2. Market
        if config.MARKET_HOLON_TYPE == 'SIMULATION':
            market = SimulationMarketHolon(initial_capital=config.INITIAL_CAPITAL)
        else:
            market = RealMarketHolon()
            
        live_bal = market.get_balance() or config.INITIAL_CAPITAL

        # 3. Core Holons
        observer = ObserverHolon(exchange_id='krakenfutures' if config.TRADING_MODE == 'FUTURES' else 'kraken')
        entropy = EntropyHolon()
        topology = TopologyHolon()
        oracle = EntryOracleHolon()
        governor = GovernorHolon(db_manager=db)
        
        # Actuator (Execution)
        actuator = ActuatorHolon(name="ActuatorAgent", exchange_id='krakenfutures' if config.TRADING_MODE == 'FUTURES' else 'kraken', paper_mode=config.PAPER_TRADING)
        executor = ExecutorHolon(actuator=actuator)
        guardian = ExitGuardianHolon()
        sentiment = SentimentHolon()
        whale = WhaleHolon()
        structure = CTKSStrategicHolon()
        arbitrage = ArbitrageHolon()
        signal_provider = SignalProviderHolon()
        
        # Kraken Intelligence (Position Management)
        kraken = None
        try:
            kraken = KrakenHolon()
            logger.info("KrakenHolon initialized for position management")
        except Exception as e:
            logger.warning(f"KrakenHolon initialization failed (position management disabled): {e}")
        regime = RegimeController()
        memory = MemoryHolon(db_manager=db)
        
        # PPO Brain (Reinforcement Learning)
        ppo = None
        try:
            ppo = PPOHolon(name="Monolith", storage_path="ppo_brain")
            logger.info("PPOHolon initialized successfully")
        except Exception as e:
            logger.warning(f"PPOHolon initialization failed: {e}")
        
        overwatch = None
        if include_overwatch:
            try:
                overwatch = OverwatchHolon()
            except Exception as e:
                logger.warning(f"Overwatch initialization failed: {e}")
        
        # Doomsday (Crisis Management)
        doomsday = None
        try:
            doomsday = DoomsdayHolon()
            logger.info("DoomsdayHolon initialized successfully")
        except Exception as e:
            logger.warning(f"DoomsdayHolon initialization failed: {e}")

        # 4. Wiring & Linking
        # Balance Sync
        governor.balance = live_bal
        governor.available_balance = live_bal
        governor.regime_controller = regime
        regime.update_state(equity=live_bal)
        
        executor.balance_usd = live_bal
        governor.executor = executor
        
        # Intelligence Links
        arbitrage.kraken_observer = observer
        oracle.memory = memory  # Link for déjà vu signals
        governor.ppo = ppo  # Link PPO brain for conviction scoring
        
        # Doomsday crisis links - BIDIRECTIONAL
        if doomsday:
            doomsday.sentiment = sentiment
            doomsday.governor = governor
            doomsday.executor = executor
            doomsday.observer = observer
            # FIX 2026-03-15: Link doomsday TO governor for crisis-aware trading
            governor.doomsday = doomsday
        
        holons = {
            'observer': observer, 'entropy': entropy, 'topology': topology,
            'oracle': oracle, 'guardian': guardian, 'governor': governor,
            'sentiment': sentiment, 'whale': whale, 'structure': structure,
            'arbitrage': arbitrage, 'overwatch': overwatch, 
            'signal_provider': signal_provider, 'executor': executor,
            'regime': regime, 'memory': memory, 'ppo': ppo, 'doomsday': doomsday,
            'kraken': kraken  # Position Management Intelligence
        }
        
        return HolonStack(db, market, holons)
        
    except Exception as e:
        logger.error(f"Failed to create holon stack: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None
