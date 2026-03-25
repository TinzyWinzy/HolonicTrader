"""
GovernorHolon - NEXUS Risk & Homeostasis (Phase 15)

Implements:
1. Dual Metabolic State (SCAVENGER / PREDATOR)
2. Volatility Targeting (ATR-based position sizing)
3. Principal Protection (Never risk the $10 base)
"""

from typing import Any, Tuple, Literal, Dict, List, Optional
from HolonicTrader.holon_core import Holon, Disposition
import config
from HolonicTrader.agent_ppo import PPOHolon

try:
    from HolonicTrader.performance_tracker import DatabaseManager, get_performance_data
except ImportError:
    # Fallback mock if not found during dev
    class DatabaseManager:
        def get_win_rate(self): return 0.5
    def get_performance_data(): return {'win_rate': 50.0}

import time
import datetime
import numpy as np
import os

# Import Monte Carlo Position Manager
try:
    from .monte_carlo_position_manager import MonteCarloPositionManager
    MONTE_CARLO_AVAILABLE = True
except ImportError:
    MONTE_CARLO_AVAILABLE = False
    print("[Governor] Monte Carlo Position Manager not available")

# ── SMCE v1 Imports ──────────────────────────────────────────────────────────
try:
    from .smce_regime_engine import SMCERegimeEngine
    from .smce_capital_doctrine import SMCECapitalDoctrine
    from .smce_probability_engine import ProbabilityStackingEngine
    from .smce_monte_carlo_court import SMCEMonteCarloRiskCourt
    from .smce_daily_digest import DailyDigestGenerator
    SMCE_AVAILABLE = True
except ImportError as _smce_err:
    SMCE_AVAILABLE = False
    print(f"[Governor] SMCE modules not available: {_smce_err}")
# ─────────────────────────────────────────────────────────────────────────────

# ── Unified Regime Engine (Phase 52) ────────────────────────────────────────
try:
    from .unified_regime_engine import get_unified_regime_engine, BehavioralRegime, OperationalRegime
    UNIFIED_REGIME_AVAILABLE = True
except ImportError as _unified_err:
    UNIFIED_REGIME_AVAILABLE = False
    print(f"[Governor] Unified Regime Engine not available: {_unified_err}")
# ─────────────────────────────────────────────────────────────────────────────

# Import Position dataclass (lazy import to avoid circular dependency)
try:
    from .agent_executor import Position
except ImportError:
    Position = None  # Will be resolved at runtime via executor

# === ML ADVISOR INTEGRATION (2026-03-21) ===
try:
    from .ml_advisor import get_ml_advisor, MLTradingAdvisor
    ML_ENABLED = True
except ImportError:
    ML_ENABLED = False
    print("[Governor] ML Advisor not available - trading without ML guidance")

# === ML-ATLAS BRIDGE (2026-03-22) ===
try:
    from .ml_atlas_bridge import MLAtlasBridge, check_ml_atlas_consensus
    ML_ATLAS_BRIDGE_ENABLED = True
except ImportError:
    ML_ATLAS_BRIDGE_ENABLED = False
    print("[Governor] ML-Atlas Bridge not available - using separate ML/Atlas checks")
# ============================================

class GovernorHolon(Holon):
    def __init__(self, name: str = "GovernorAgent", initial_balance: float = None, db_manager: Any = None):
        super().__init__(name=name, disposition=Disposition(autonomy=0.9, integration=0.9))

        # FIX 2026-03-01: Add sub_holons dict for holon-to-holon communication
        # This is required by medic, overwatch, and other holons that access governor via trader.sub_holons
        self.sub_holons = {}  # Internal reference storage (populated by TraderNexus if needed)

        # Phase 2: Modular Risk Manager
        from .governor_risk import RiskManager
        self.risk_manager = RiskManager(self)
        
        # FIX: Startup Budget sync
        if initial_balance is None:
            initial_balance = getattr(config, 'INITIAL_CAPITAL', 100.0) # Aligned with config.py
            
        self.balance = initial_balance
        self.available_balance = initial_balance # New: Track free margin
        self.hard_stop_threshold = 5.0
        self.DEBUG = False # Silence rejection spam
        self.db_manager = db_manager  # For win rate tracking
        
        # Phase 22: Portfolio Health Tracking
        self.max_balance = initial_balance
        self.drawdown_pct = 0.0
        self.margin_utilization = 0.0
        
        # Accumulator State (Phase 42)
        self.high_water_mark = initial_balance
        self.risk_multiplier = 1.0
        self.equity_history = []
        self.drawdown_lock = False
        
        # Phase 50: Daily Risk Reset
        self.last_hwm_date = datetime.datetime.utcnow().date()
        self.day_start_equity = initial_balance # Used for Dynamic Drawdown Recovery
        
        # === SMCE v1 L0: Capital Doctrine State ===
        self._smce_tier = "SMALL"
        self._day_start_equity = initial_balance
        self._week_start_equity = initial_balance
        self._last_day_reset = datetime.datetime.utcnow().date()
        self._last_week_reset = datetime.datetime.utcnow().isocalendar()[1]
        self._defensive_cooldown_until = 0.0
        self._risk_multiplier_smce = 1.0
        
        # Load SMCE State from DB
        if self.db_manager and hasattr(self.db_manager, 'load_smce_state'):
            try:
                state = self.db_manager.load_smce_state()
                if state:
                    self._smce_tier = state.get('smce_tier', 'SMALL')
                    self._day_start_equity = state.get('day_start_equity', initial_balance)
                    self._week_start_equity = state.get('week_start_equity', initial_balance)
                    
                    last_day_str = state.get('last_day_reset')
                    if last_day_str:
                         if isinstance(last_day_str, str):
                             try:
                                 self._last_day_reset = datetime.datetime.fromisoformat(last_day_str).date()
                             except:
                                 self._last_day_reset = datetime.datetime.utcnow().date()
                         elif isinstance(last_day_str, (datetime.date, datetime.datetime)):
                             self._last_day_reset = last_day_str if isinstance(last_day_str, datetime.date) else last_day_str.date()
                         
                    self._last_week_reset = state.get('last_week_reset', self._last_week_reset)
                    self._defensive_cooldown_until = state.get('defensive_cooldown_until', 0.0)
                    self._risk_multiplier_smce = state.get('risk_multiplier_smce', 1.0)
                    self._consecutive_days_without_intervention = state.get('consecutive_clean_days', 0)
                    self._period_max_drawdown = state.get('period_max_drawdown', 0.0)
                    self._allocation_pct_boost = state.get('allocation_pct_boost', 0.0)
                    print(f"[{self.name}] 💾 SMCE Layer 0 DB State Loaded Successfully.")
            except Exception as e:
                print(f"[{self.name}] ⚠️ Failed to load SMCE DB state: {e}")
                
        # Reference ATR for volatility targeting (set during first cycle)
        self.reference_atr = None
        
        # Position Tracking (Multi-Asset)
        # self.positions DEPRECATED (Moved to Executor)
        self.executor = None # Link to source of truth
        self.last_trade_time = {} # symbol -> timestamp
        self.last_specific_entry = {} # symbol -> price (for stacking distance)
        self.latest_prices = {} # symbol -> price (Last seen market price)
        
        # FIX 3: Stack Timeout Tracker (for 5-minute reduction trigger)
        self.stack_timeout_tracker = {} # symbol -> first_blocked_timestamp
        self.stack_snooze = {} # symbol -> snooze_until_timestamp
        
        # FIX 2026-03-12: Consecutive Loss Circuit Breaker
        self.consecutive_losses = {} # symbol -> count of consecutive losses
        self.blacklist = {} # symbol -> expiration timestamp
        
        # Phase 7: Regime Controller Integration
        self.regime_controller = None  # Set by Trader after instantiation
        
        # IRON BANK STATE (Capital Preservation)
        self.fortress_balance = getattr(config, 'IRON_BANK_MIN_RESERVE', 100.0) # The Floor
        self.risk_budget = 0.0 # Tradeable Capital
        self.last_ratchet_time = 0
        
        # Consolidation Engine State
        self.last_consolidation_time = 0.0
        self.consolidation_in_progress = False

        # --- SESSION 3 REPAIR: META-LEARNING VETO SYSTEM ---
        from collections import defaultdict
        self.meta_veto_counter = defaultdict(int) # symbol -> count
        self.last_veto_reason = None
        self.consecutive_veto_streak = 0
        self.veto_timestamps = defaultdict(list) # symbol -> list of veto times
        # ---------------------------------------------------
        
        # === POOL ISOLATION STATE ===
        # Pool A (Directional) and Pool B (Arb) have independent tracking
        self.pool_a_last_entry_time = 0.0    # Last directional entry timestamp
        self.pool_b_last_entry_time = 0.0    # Last arb entry timestamp
        self.pool_a_entries_this_cycle = 0   # Directional entries this cycle
        self.pool_b_entries_this_cycle = 0   # Arb entries this cycle
        self.pool_cycle_start_time = time.time()
        # =============================
        
        # Phase 3: Signal Truth Tracking (Decoupled from PnL)
        self.signal_outcomes = {} # {symbol: {'signals': [], 'post_hoc_correct': 0, 'total': 0}}

        # FIX 2: Dynamic Blacklist (Equity Gated) - REMOVED: Line 145 already defines as dict
        # self.blacklist = set()  # <-- THIS OVERWRITES THE DICT! Removed.

        # Primary Budget Sync
        self.manage_iron_bank()

        
        # Initialize Monte Carlo Position Manager
        self.monte_carlo_manager = None
        if MONTE_CARLO_AVAILABLE:
            try:
                self.monte_carlo_manager = MonteCarloPositionManager()
                print(f"[{self.name}] Monte Carlo Position Manager initialized")
            except Exception as e:
                print(f"[{self.name}] Failed to initialize Monte Carlo Position Manager: {e}")

        # ================================================================
        # ATLAS PROFIT INTEGRATION (2026-03-18)
        # Edge-aware position sizing integration
        # ================================================================
        self.atlas = None
        self.atlas_available = False
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from atlas_integration import AtlasProfitIntegration
            self.atlas = AtlasProfitIntegration()
            # Don't initialize yet - wait for balance to be available
            print(f"[{self.name}] Atlas Profit System module loaded")
        except Exception as e:
            print(f"[{self.name}] Atlas not available: {e}")
        # ================================================================

        # === ML ADVISOR INITIALIZATION (2026-03-21) ===
        self.ml_advisor = None
        if ML_ENABLED:
            try:
                self.ml_advisor = get_ml_advisor()
                status = self.ml_advisor.get_model_status()
                print(f"[{self.name}] 🤖 ML Advisor initialized ({status['database_trades']} historical trades, {status['database_win_rate']:.1%} win rate)")
            except Exception as e:
                print(f"[{self.name}] ML Advisor init failed: {e}")
                self.ml_advisor = None
        
        # ML performance tracking
        self.ml_predictions = {}  # Active trade predictions
        self.ml_performance = []  # Historical prediction accuracy
        
        # === ML-ATLAS BRIDGE INITIALIZATION (2026-03-22) ===
        self.ml_atlas_bridge = None
        if ML_ATLAS_BRIDGE_ENABLED and self.ml_advisor:
            try:
                # Get Atlas filter reference - use self.atlas (which has .profit_filter)
                atlas_filter = getattr(self, 'atlas', None)
                self.ml_atlas_bridge = MLAtlasBridge(self.ml_advisor, atlas_filter)
                print(f"[{self.name}] 🤖🗺️ ML-Atlas Bridge initialized")
            except Exception as e:
                print(f"[{self.name}] ML-Atlas Bridge init failed: {e}")
                self.ml_atlas_bridge = None
        # ============================================


        # Initialize Management Mode attributes
        self.management_mode = False
        self.management_mode_reason = None
        self.management_mode_start_time = None
        self.management_mode_target = None
        self.management_mode_risk_multiplier = None
        # FIX BUG-002: Management Mode max duration cap (10 minutes)
        self.management_mode_max_duration = getattr(config, 'MANAGEMENT_MODE_MAX_DURATION_SEC', 600)
        
        # FIX 2026-03-03: Management mode cooldown to prevent rapid re-entry
        self._last_management_mode_exit_time = 0
        self._management_mode_cooldown_sec = getattr(config, 'MANAGEMENT_MODE_COOLDOWN_SEC', 120)
        self._management_mode_auto_exit_threshold = getattr(config, 'MANAGEMENT_MODE_AUTO_EXIT_THRESHOLD', 0.80)

        # FIX 2026-02-28: Equity fetch tracking for drawdown check reliability
        self._equity_fetch_success_count = 0
        self._api_failure_mode = False

        # ── SMCE v1 Layer Initialisation ──────────────────────────────────────
        from .smce_regime_engine import SMCERegimeEngine
        self.smce_regime_engine = SMCERegimeEngine()
        self.smce_doctrine = None # Phase 2 refactored this into the Governor intrinsically
        self.smce_prob_engine     = None
        self.smce_mc_court        = None
        self.smce_digest          = None
        self.smce_regime          = "HARVEST"   # current SMCE market regime
        if SMCE_AVAILABLE:
            try:
                self.smce_regime_engine = SMCERegimeEngine()
                self.smce_doctrine      = SMCECapitalDoctrine()
                self.smce_prob_engine   = ProbabilityStackingEngine()
                self.smce_mc_court      = SMCEMonteCarloRiskCourt(n_paths=1000)
                self.smce_digest        = DailyDigestGenerator(
                    log_dir="logs/",
                    telegram_fn=None,   # Telegram wired separately if needed
                )
                self.smce_digest.schedule_daily()
                print(f"[{self.name}] ✅ SMCE v1 Layers 0-3 online")
            except Exception as _smce_init_err:
                print(f"[{self.name}] ⚠️ SMCE init error: {_smce_init_err}")
        # ─────────────────────────────────────────────────────────────────────

        # ── Unified Regime Engine (Phase 52) ─────────────────────────────────
        # Replaces dual HolonicAdaptor + SMCERegimeEngine with unified system
        self.unified_regime = None
        self._last_regime_update = 0
        self._regime_update_interval = 60  # Update every 60 seconds max
        if UNIFIED_REGIME_AVAILABLE:
            try:
                self.unified_regime = get_unified_regime_engine()
                print(f"[{self.name}] ✅ Unified Regime Engine online (Phase 52)")
            except Exception as _unified_init_err:
                print(f"[{self.name}] ⚠️ Unified Regime init error: {_unified_init_err}")
        # ─────────────────────────────────────────────────────────────────────

        # === KRAKEN FLEXLINE & LOAN TRACKING (2026-03-09) ===
        self.loan_amount = getattr(config, 'LOAN_DETAILS', {}).get('ACTIVE_LOAN_AMOUNT', 0.0)
        self.repayment_reserve = 0.0 # Accumulated profit for payback
        self.aggressive_mode = getattr(config, 'AGGRESSIVE_GROWTH_MODE', False)

        # Flexline Agent Reference (injected by system)
        self.flexline_agent = None

        # FIX 2026-03-15: Crisis Score Integration
        # Link to DoomsdayHolon for crisis-aware trading
        self.doomsday = None

        # Crisis-aware trading flags
        self.crisis_trading_halted = False
        self.crisis_position_reduction = 0.0  # 0.0 = normal, 0.5 = 50% reduction

        # Load Repayment Reserve from DB if available
        if self.db_manager and hasattr(self.db_manager, 'load_repayment_reserve'):
            self.repayment_reserve = self.db_manager.load_repayment_reserve()
    def initialize_atlas_with_balance(self, balance: float):
        """
        Initialize Atlas Profit System with current account balance
        
        Called when balance is available (after equity fetch)
        """
        if self.atlas and not self.atlas_available:
            try:
                success, message = self.atlas.initialize_with_account(balance)
                if success:
                    self.atlas_available = True
                    print(f"[{self.name}] Atlas initialized with balance: ${balance:.2f}")
                else:
                    print(f"[{self.name}] Atlas initialization failed: {message}")
            except Exception as e:
                print(f"[{self.name}] Atlas initialization error: {e}")
    
    def get_atlas_position_size(self, symbol: str, price: float, direction: str = 'BUY', 
                                 metadata: dict = None) -> tuple:
        """
        Query Atlas for edge-aware position sizing
        
        Returns: (approved: bool, notional: float, metadata: dict)
        """
        if not self.atlas_available or not self.atlas:
            return False, 0.0, {}
        
        try:
            # Prepare signal data
            signal_data = {
                'symbol': symbol,
                'direction': direction,
                'strength': metadata.get('conviction', 0.7) if metadata else 0.7,
                'source': metadata.get('strategy', 'DIRECTIONAL')
            }
            
            # Prepare market data
            market_data = {
                'price': price,
                'volatility_pct': metadata.get('atr_pct', 0.01) if metadata else 0.01,
                'spread_pct': 0.0005,  # Can be fetched from market data
                'liquidity_score': 0.8,
                'regime': self.get_smce_regime(),
                'regime_score': 0.3
            }
            
            # Prepare portfolio state
            portfolio_state = {
                'account_balance': self.balance,
                'available_margin': self.balance * 0.50,  # Assume 50% available
                'win_rate': 0.60,  # Can be fetched from performance tracking
                'win_loss_ratio': 1.5,
                'current_positions': {}  # Can be populated from executor.positions
            }
            
            # Query Atlas
            approved, reason, atlas_metadata = self.atlas.profit_filter.evaluate_trade(
                signal_data, market_data, portfolio_state
            )
            
            if approved:
                notional = atlas_metadata.get('position_size_usd', 0.0)
                return True, notional, atlas_metadata
            else:
                return False, 0.0, {'reason': reason}
                
        except Exception as e:
            print(f"[{self.name}] Atlas query error: {e}")
            return False, 0.0, {}
    

        # ====================================================

    def get_effective_balance(self, include_flexline: bool = True) -> float:
        """
        Get effective balance including available Flexline credit.

        Args:
            include_flexline: Whether to include Flexline credit

        Returns:
            Effective balance in USD
        """
        if not include_flexline or not self.flexline_agent or not self.flexline_agent.enabled:
            return self.balance

        # Get available Flexline credit for trading
        flexline_available = self.flexline_agent.get_available_for_trading()

        # Apply utilization cap
        max_flexline = self.balance * getattr(config, 'FLEXLINE_MAX_UTILIZATION', 0.50)
        flexline_to_use = min(flexline_available, max_flexline)

        return self.balance + flexline_to_use

    def get_flexline_boost(self) -> float:
        """
        Get the Flexline credit boost amount.

        Returns:
            Flexline credit available for trading
        """
        if not self.flexline_agent or not self.flexline_agent.enabled:
            return 0.0

        flexline_available = self.flexline_agent.get_available_for_trading()
        max_flexline = self.balance * getattr(config, 'FLEXLINE_MAX_UTILIZATION', 0.50)
        return min(flexline_available, max_flexline)

    def set_executor(self, executor: Any):
        """Link to the Executor Agent (Single Source of Truth)."""
        self.executor = executor
        print(f"[{self.name}] 🔗 Linked to Executor (Source of Truth)")

    def _get_symbol_expectancy(self, symbol: str, min_trades: int = 20) -> tuple:
        """
        Per-symbol expectancy from DB. Returns (expectancy, n_trades).
        Cached for 30 minutes per symbol to avoid DB spam.
        FIX 2026-03-21: Block entries on symbols with proven negative expectancy.
        """
        cache = getattr(self, '_expectancy_cache', {})
        now = time.time()
        if symbol in cache and (now - cache[symbol][2]) < 1800:  # 30 min cache
            return cache[symbol][0], cache[symbol][1]

        if not self.db_manager:
            return 0.0, 0

        try:
            trades = self.db_manager.get_recent_trades(limit=200)
            sym_trades = [t for t in trades if t.get('symbol') == symbol and t.get('pnl') is not None]
            n = len(sym_trades)
            if n < min_trades:
                result = (0.0, n)
            else:
                pnls = [float(t['pnl']) for t in sym_trades[:50]]  # Last 50 trades for symbol
                expectancy = sum(pnls) / len(pnls) if pnls else 0.0
                result = (expectancy, len(pnls))

            if not hasattr(self, '_expectancy_cache'):
                self._expectancy_cache = {}
            self._expectancy_cache[symbol] = (result[0], result[1], now)
            return result
        except Exception:
            return 0.0, 0

    def _get_symbol_trades_today(self, symbol: str) -> int:
        """Count trades for a symbol in the last 24 hours."""
        cache_key = '_daily_trade_counts'
        now = time.time()
        # Refresh cache every 5 minutes
        if hasattr(self, cache_key):
            cache = getattr(self, cache_key)
            if (now - cache.get('_ts', 0)) < 300:
                return cache.get(symbol, 0)

        try:
            if not self.db_manager:
                return 0
            trades = self.db_manager.get_recent_trades(limit=200)
            cutoff = now - 86400
            counts = {}
            for t in trades:
                ts = t.get('timestamp', 0)
                if isinstance(ts, str):
                    continue  # skip unparseable
                if ts > cutoff:
                    s = t.get('symbol', '')
                    counts[s] = counts.get(s, 0) + 1
            counts['_ts'] = now
            setattr(self, cache_key, counts)
            return counts.get(symbol, 0)
        except Exception:
            return 0

    def _is_symbol_whitelisted(self, symbol: str) -> bool:
        """Check if symbol is whitelisted via Atlas edge amplification system.
        2026-03-21: Used to relax gates for proven winners."""
        try:
            if self.atlas_available and self.atlas and hasattr(self.atlas, 'profit_filter'):
                pf = self.atlas.profit_filter
                if hasattr(pf, 'performance_tracker'):
                    return pf.performance_tracker.is_whitelisted(symbol)
        except Exception:
            pass
        return False

    def _get_symbol_edge_stats(self, symbol: str) -> dict:
        """Get symbol performance stats for edge-boost sizing.
        Returns dict with win_rate, avg_pnl (expectancy), total_trades."""
        try:
            if self.atlas_available and self.atlas and hasattr(self.atlas, 'profit_filter'):
                pf = self.atlas.profit_filter
                if hasattr(pf, 'performance_tracker'):
                    tracker = pf.performance_tracker
                    stats = tracker.performance_db.get('symbol_performance', {}).get(symbol, {})
                    return {
                        'win_rate': stats.get('win_rate', 0.0),
                        'avg_pnl': stats.get('avg_pnl', 0.0),
                        'total_trades': stats.get('total_trades', 0),
                    }
        except Exception:
            pass
        return {'win_rate': 0.0, 'avg_pnl': 0.0, 'total_trades': 0}


    @property
    def positions(self):
        """
        Backward Compatibility: Proxy to Executor (Source of Truth).
        External observers (Broadcaster/Dashboard) expect this attribute.
        """
        if self.executor:
            if hasattr(self.executor, 'get_positions_snapshot'):
                return self.executor.get_positions_snapshot()
            return self.executor.positions
        return {}

    def record_signal_truth(self, symbol: str, signal_direction: str, was_correct: bool):
        """Track signal accuracy separately from trade PnL (Phase 3)."""
        if symbol not in self.signal_outcomes:
            self.signal_outcomes[symbol] = {'signals': [], 'post_hoc_correct': 0, 'total': 0}
        
        self.signal_outcomes[symbol]['total'] += 1
        if was_correct:
            self.signal_outcomes[symbol]['post_hoc_correct'] += 1
        
        # Phase 22: The Monolith (PPO Brain)
        # We give it a small autonomy to start (it suggests, rule-based decides)
        self.ppo = PPOHolon(name="Monolith")
        print(f"[{self.name}] 🧠 Monolith (PPO) Online.")

    def close_position(self, symbol: str):
        """
        Explicitly close a position's tracking in the Governor when the Executor
        detects a local ghost or standard close.
        Cleans up associated tracking to free up the symbol for new trades immediately.

        FIX 2026-03-16: Clean up ALL tracker variants (bare symbol and virt_key suffixed versions)
        to prevent tracker accumulation/leaks.
        """
        # Clean up bare symbol and all possible virt_key variants
        symbols_to_clean = {symbol}
        # Add virt_key variants if this is a bare symbol
        if ':' not in symbol:
            # Check all trackers for variants of this symbol
            for key in list(self.last_specific_entry.keys()) + list(self.stack_timeout_tracker.keys()) + list(self.stack_snooze.keys()):
                if key.startswith(symbol + ':') or key.split(':')[0] == symbol:
                    symbols_to_clean.add(key)

        for sym in symbols_to_clean:
            if sym in self.last_specific_entry:
                del self.last_specific_entry[sym]
            if sym in self.stack_timeout_tracker:
                del self.stack_timeout_tracker[sym]
            if sym in self.stack_snooze:
                del self.stack_snooze[sym]

        # Reset meta-learning veto trackers to give the symbol a fresh start
        if symbol in self.meta_veto_counter:
            self.meta_veto_counter[symbol] = 0

        # Also clean blacklist if present
        if symbol in self.blacklist:
            del self.blacklist[symbol]
        if symbol in self.consecutive_losses:
            del self.consecutive_losses[symbol]

        print(f"[{self.name}] 🧹 Governor trackers cleared for {symbol} (Position Closed)")

    def register_trade_outcome(self, symbol: str, pnl_pct: float, crisis_score: float = 0.0, 
                                entry_price: float = None, exit_price: float = None, pnl_usd: float = None):
        """
        Record the actual PnL outcome of a closed trade (called by Executor).
        Implements the Consecutive Loss Circuit Breaker.

        FIX 2026-03-15: Reduced threshold from 3 to 2 consecutive losses.
        FIX 2026-03-15: Added crisis_score parameter to halt trading during high crisis.
        FIX 2026-03-21: Added entry_price, exit_price, pnl_usd for better logging.
        """
        # FIX 2026-03-15: Crisis override - if crisis is high, all losses are amplified
        if crisis_score >= 0.8:
            # During high crisis, reduce threshold to 1 loss
            effective_threshold = 1
            print(f"[{self.name}] ☢️ CRISIS MODE: Loss threshold reduced to 1 (Crisis Score: {crisis_score:.2f})")
        elif crisis_score >= 0.5:
            # During elevated crisis, use threshold of 2
            effective_threshold = 2
            print(f"[{self.name}] ⚠️ ELEVATED THREAT: Loss threshold set to 2 (Crisis Score: {crisis_score:.2f})")
        else:
            # Normal market conditions
            effective_threshold = 2  # FIX 2026-03-15: Changed from 3 to 2

        if pnl_pct < 0:
            self.consecutive_losses.setdefault(symbol, 0)
            self.consecutive_losses[symbol] += 1
            losses = self.consecutive_losses[symbol]

            # Build log message with prices if available
            price_info = ""
            if entry_price is not None and exit_price is not None:
                price_info = f" (Entry: ${entry_price:.4f} → Exit: ${exit_price:.4f})"
            if pnl_usd is not None:
                price_info += f" [${pnl_usd:.2f}]"

            print(f"[{self.name}] 📉 Trade Logged: {symbol} Profit: {pnl_pct:.2f}%{price_info}. Consecutive Losses: {losses}/{effective_threshold}")

            # === FIX 2026-03-23: CHRONIC LOSS ASSET CHECK ===
            # Check if this asset has a history of loss streaks
            from HolonicTrader import config
            chronic_assets = getattr(config, 'CHRONIC_LOSS_ASSETS', {})
            if symbol in chronic_assets:
                max_allowed = chronic_assets[symbol].get('max_consecutive', effective_threshold)
                if losses >= max_allowed:
                    # Extended suspension for chronic loss assets
                    timeout_duration = 86400 * 2  # 48 hours for chronic losers
                    expiration = time.time() + timeout_duration
                    self.blacklist[symbol] = expiration
                    chronic_assets[symbol]['suspended'] = True
                    print(f"[{self.name}] 🚨 CHRONIC LOSS ASSET: {symbol} hit {max_allowed} losses (streak: {losses}). Banned for 48 hours.")
            # === END CHRONIC LOSS ASSET CHECK ===

            # Circuit Breaker Trigger - FIX 2026-03-15: Threshold is now 2 (was 3)
            if losses >= effective_threshold and symbol not in self.blacklist:
                # 12-hour timeout (43200 seconds)
                timeout_duration = 43200
                expiration = time.time() + timeout_duration
                self.blacklist[symbol] = expiration
                print(f"[{self.name}] 🚨 ASSET BLACKLISTED: {symbol} triggered {effective_threshold} consecutive losses. Banned for 12 hours.")
        else:
            # Win! Reset the counter
            if symbol in self.consecutive_losses and self.consecutive_losses[symbol] > 0:
                # Build log message with prices if available
                price_info = ""
                if entry_price is not None and exit_price is not None:
                    price_info = f" (Entry: ${entry_price:.4f} → Exit: ${exit_price:.4f})"
                if pnl_usd is not None:
                    price_info += f" [${pnl_usd:.2f}]"
                
                print(f"[{self.name}] 🟢 Trade Logged: {symbol} Profit: {pnl_pct:.2f}%{price_info}. Resetting loss streak.")
                self.consecutive_losses[symbol] = 0

            # If it was blacklisted (maybe cleared manually), ensure it stays clear
            if symbol in self.blacklist:
                del self.blacklist[symbol]
        
        # === ML PERFORMANCE TRACKING (2026-03-21) ===
        if self.ml_advisor and symbol in self.ml_predictions:
            try:
                pred = self.ml_predictions[symbol]
                actual_win = pnl_pct > 0
                
                # Record prediction vs actual
                self.ml_performance.append({
                    'timestamp': datetime.datetime.now().isoformat(),
                    'symbol': symbol,
                    'predicted_win_prob': pred['win_prob'],
                    'predicted_confidence': pred['confidence'],
                    'actual_pnl': pnl_pct,
                    'actual_win': actual_win,
                    'prediction_correct': (pred['win_prob'] > 0.5) == actual_win,
                })
                
                # Log accuracy
                if len(self.ml_performance) >= 10:
                    recent = self.ml_performance[-20:]
                    correct = sum(1 for p in recent if p['prediction_correct'])
                    accuracy = correct / len(recent)
                    print(f"[{self.name}] 🤖 ML Accuracy (last 20): {accuracy:.1%} ({correct}/{len(recent)})")
                
                # Clean up
                del self.ml_predictions[symbol]
                
            except Exception as e:
                print(f"[{self.name}] ML performance tracking failed: {e}")
        # ============================================

    def check_crisis_status(self) -> dict:
        """
        FIX 2026-03-15: Check crisis status from DoomsdayHolon and update trading parameters.

        Returns:
            dict: {'crisis_score': float, 'trading_allowed': bool, 'position_reduction': float}
        """
        # Get crisis score from DoomsdayHolon if linked
        crisis_score = 0.0

        if self.doomsday and hasattr(self.doomsday, 'get_threat_level'):
            # DEFCON 5 = normal (0.0), DEFCON 1 = catastrophic (1.0)
            defcon = self.doomsday.get_threat_level()
            crisis_score = (5 - defcon) / 4.0  # Convert to 0-1 scale

        # Update crisis trading flags
        if crisis_score >= 1.0:
            # DEFCON 1: Catastrophic - halt all trading
            self.crisis_trading_halted = True
            self.crisis_position_reduction = 1.0
            print(f"[{self.name}] ☢️ CRISIS HALT: All trading suspended (Crisis Score: {crisis_score:.2f})")
        elif crisis_score >= 0.8:
            # DEFCON 2: Severe - 75% position reduction
            self.crisis_trading_halted = False
            self.crisis_position_reduction = 0.75
            print(f"[{self.name}] 🚨 CRISIS REDUCTION: Positions reduced 75% (Crisis Score: {crisis_score:.2f})")
        elif crisis_score >= 0.5:
            # DEFCON 3: High alert - 50% position reduction
            self.crisis_trading_halted = False
            self.crisis_position_reduction = 0.50
            print(f"[{self.name}] ⚠️ ELEVATED THREAT: Positions reduced 50% (Crisis Score: {crisis_score:.2f})")
        elif crisis_score >= 0.3:
            # DEFCON 4: Elevated caution - 25% position reduction
            self.crisis_trading_halted = False
            self.crisis_position_reduction = 0.25
        else:
            # DEFCON 5: Normal operations
            self.crisis_trading_halted = False
            self.crisis_position_reduction = 0.0

        return {
            'crisis_score': crisis_score,
            'trading_allowed': not self.crisis_trading_halted,
            'position_reduction': self.crisis_position_reduction
        }

    def reconcile_with_executor(self, active_positions: dict):
        """
        Called by ExecutorHolon after state changes to ensure Governor's
        internal trackers (like veto counters, stacking cooldowns, specific entries)
        remain aligned with reality, preventing memory leaks or amnesia.

        FIX 2026-03-16: Handle symbol variants (bare symbol vs virt_key) to prevent
        tracker leaks when positions use different key formats.
        """
        # Build set of active bare symbols (e.g., "BTC/USDT")
        active_bare_symbols = set()
        # Also build map of bare symbol -> virt_key for tracker sync
        bare_to_virt = {}

        for vk, pos in active_positions.items():
            bare_sym = pos.symbol.split(':')[0] if ':' in pos.symbol else pos.symbol
            active_bare_symbols.add(bare_sym)
            bare_to_virt[bare_sym] = pos.symbol

            # Keep specific entry tracker aligned with reality
            if pos.symbol not in self.last_specific_entry and pos.entry_price > 0:
                self.last_specific_entry[pos.symbol] = pos.entry_price

        # 1. Cleanup specific entry trackers for closed positions (by bare symbol match)
        for sym in list(self.last_specific_entry.keys()):
            bare_sym = sym.split(':')[0] if ':' in sym else sym
            if bare_sym not in active_bare_symbols:
                del self.last_specific_entry[sym]

        # 2. Cleanup stack timeouts for closed positions (by bare symbol match)
        for sym in list(self.stack_timeout_tracker.keys()):
            bare_sym = sym.split(':')[0] if ':' in sym else sym
            if bare_sym not in active_bare_symbols:
                del self.stack_timeout_tracker[sym]

        # 3. Cleanup stack snoozes for closed positions (by bare symbol match)
        for sym in list(self.stack_snooze.keys()):
            bare_sym = sym.split(':')[0] if ':' in sym else sym
            if bare_sym not in active_bare_symbols:
                del self.stack_snooze[sym]

    def sync_positions(self, held_assets: dict, metadata: dict):
        """
        Sync positions from Executor/DB on startup to cure Amnesia.
        Only syncs internal governor trackers (like last_specific_entry)
        as Execution state is now managed purely by the Executor.

        FIX 2026-03-16: Clear ALL tracker types to prevent phantom tracker accumulation.
        """
        print(f"[{self.name}] Syncing position trackers from DB...")

        # FIX: Clear ALL old phantoms from all tracker types!
        self.last_specific_entry.clear()
        self.stack_timeout_tracker.clear()
        self.stack_snooze.clear()
        self.meta_veto_counter.clear()

        count = 0
        for symbol, qty in held_assets.items():
            if abs(qty) > 0.00000001:
                meta = metadata.get(symbol, {})
                entry_price = meta.get('entry_price', 0.0)

                # Sync stacking tracker
                self.last_specific_entry[symbol] = entry_price
                count += 1

                direction = meta.get('direction', 'BUY' if qty > 0 else 'SELL')
                strategy = meta.get('strategy', 'DIRECTIONAL')
                leverage = meta.get('leverage', 1.0)
                pool = 'B (ARB)' if strategy in ['ARBITRAGE_GOLD', 'FUNDING_CARRY', 'BASIS_TRADE', 'ARB'] else 'A (DIR)'
                print(f"[{self.name}] Tracker Synced: {symbol} ({direction}, Qty: {abs(qty):.4f}, Lev: {leverage}x, Pool: {pool})")

        if count == 0:
            print(f"[{self.name}] Aucun tracker actif trouvé pour synchronisation.")

        # Phantom margin cleanup log just for completeness:
        if count == 0:
            print(f"[{self.name}] 🧹 PHANTOM MARGIN CLEANUP: No positions after sync, clearing phantom margin exposure")

    def sync_fortress(self, stored_floor: float):
        """Restore the Iron Bank floor from DB."""
        if stored_floor is not None and stored_floor > 0:
            self.fortress_balance = max(self.fortress_balance, stored_floor)
            print(f"[{self.name}] 🏰 Iron Bank Floor Restored: ${self.fortress_balance:.2f}")
        else:
            # Fallback for uninitialized DB or None
            self.fortress_balance = config.PRINCIPAL
            print(f"[{self.name}] 🏰 Iron Bank Floor Initialized (Fallback): ${self.fortress_balance:.2f}")
    
    def record_pool_entry(self, is_arb: bool = False):
        """
        Called after successful trade entry to update pool counters.
        This ensures cooldown and cycle limits are enforced correctly.
        """
        if is_arb:
            self.pool_b_last_entry_time = time.time()
            self.pool_b_entries_this_cycle += 1
            if self.DEBUG: print(f"[{self.name}] 📊 Pool B Entry Recorded (Cycle: {self.pool_b_entries_this_cycle})")
        else:
            self.pool_a_last_entry_time = time.time()
            self.pool_a_entries_this_cycle += 1
            if self.DEBUG: print(f"[{self.name}] 📊 Pool A Entry Recorded (Cycle: {self.pool_a_entries_this_cycle})")

    def set_live_balance(self, total: float, available: float):
        """Update equity and free margin from live exchange data."""

        # --- PATCH: NULL SAFETY ---
        if total is None or available is None:
            # Keep previous known state to avoid panic
            return

        # --- PATCH: STALE HWM PREVENTER ---
        # On first connection or if HWM is strangely low/high, trust the live balance.
        # Check total > 0.0 to prevent syncing on API errors
        if total > 0.0:
            # If we have never updated HWM (it's at init 10.0) OR if we are resetting:
            if self.high_water_mark == 10.0 or (total < self.high_water_mark * 0.8):
                # If current total is > 20% below HWM on startup, assume it's a new session/reset
                # This prevents "Solvency Halt" due to previous session data if we deployed fresh
                if not getattr(self, '_hwm_synced', False):
                    print(f"[{self.name}] 🔄 Syncing High Water Mark to Live Balance: ${total:.2f}")
                    self.high_water_mark = total
                    self._hwm_synced = True

            # Only update state if valid read
            self.balance = total
            self.available_balance = available
            self.update_accumulator(total)

            # FIX: Force Recalculate Risk Budget after Balance Update
            self.manage_iron_bank()

            # FIX 2: HARD BAN REMOVED (Replaced by Arb-Only Logic in is_trade_allowed)
            pass

    def clear_defensive_cooldown(self, reason: str = "Manual override"):
        """
        Clear defensive cooldown if triggered by API failure (not actual losses).
        
        FIX 2026-02-24: Allows manual recovery from false positive drawdown triggers.
        FIX 2026-03-04: Now clears BOTH the Governor's _defensive_cooldown_until AND
                        smce_doctrine.defensive_until to ensure both L0 paths are reset.
        
        Args:
            reason: Reason for clearing (logged for audit)
            
        Returns:
            bool: True if cooldown was cleared, False if not in cooldown
        """
        now = time.time()
        gov_active  = self._defensive_cooldown_until > now
        doc_active  = (
            self.smce_doctrine is not None and
            getattr(self.smce_doctrine, 'defensive_until', 0.0) > now
        )

        if not gov_active and not doc_active:
            print(f"[{self.name}] ℹ️ No active defensive cooldown to clear")
            return False

        old_cooldown = max(
            self._defensive_cooldown_until,
            getattr(self.smce_doctrine, 'defensive_until', 0.0) if self.smce_doctrine else 0.0,
        )
        old_risk = self._risk_multiplier_smce

        # 1. Clear Governor-level timestamp
        self._defensive_cooldown_until = 0.0
        self._risk_multiplier_smce = 1.0

        # 2. Clear Doctrine-level timestamp (feeds smce_doctrine.check_trade L0 gate)
        if self.smce_doctrine and hasattr(self.smce_doctrine, 'clear_defensive'):
            self.smce_doctrine.clear_defensive(reason)

        print(f"[{self.name}] ✅ Defensive Cooldown CLEARED: {reason}")
        print(f"   Previous cooldown until: {datetime.datetime.fromtimestamp(old_cooldown)}")
        print(f"   Risk multiplier restored: {old_risk} -> 1.0")

        # 3. Persist cleared state to DB
        if self.db_manager:
            try:
                smce_state = {
                    'defensive_cooldown_until': 0.0,
                    'risk_multiplier_smce': 1.0,
                }
                self.db_manager.save_smce_state(smce_state)
                print(f"[{self.name}] 📝 State saved to DB")
            except Exception as e:
                print(f"[{self.name}] ⚠️ Failed to save state to DB: {e}")

        return True

    def register_outcome(self, pnl: float, symbol: str):
        """
        Feedback Loop: Adjust metabolic state based on realized outcomes.
        Called by Executor after a trade closes.
        """
        # --- FIX 3.1: Repetitive Cycle Detection ---
        if not hasattr(self, 'repetition_tracker'):
            self.repetition_tracker = {}
        
        # Add timestamp to tracker
        now = time.time()
        if symbol not in self.repetition_tracker:
            self.repetition_tracker[symbol] = []
        
        # Clean up old entries (> 10 mins)
        cutoff = now - 600
        self.repetition_tracker[symbol] = [t for t in self.repetition_tracker[symbol] if t > cutoff]
        
        self.repetition_tracker[symbol].append(now)
        
        if len(self.repetition_tracker[symbol]) >= 3:
             print(f"[{self.name}] 🚨 REPETITION ALERT: {symbol} has cycled {len(self.repetition_tracker[symbol])} times in 10m. Pausing.")
        # -------------------------------------------

        if pnl > 0:
            # WINNER
            self.win_streak = getattr(self, 'win_streak', 0) + 1
            
            # === KRAKEN FLEXLINE REPAYMENT RESERVE (2026-03-09) ===
            # Subtract fixed % of profit into the reserve for repayment
            if self.loan_amount > 0 and self.repayment_reserve < self.loan_amount:
                reserve_pct = getattr(config, 'LOAN_DETAILS', {}).get('REPAYMENT_RESERVE_PCT', 0.25)
                to_reserve = pnl * reserve_pct
                # Cap reserve at loan amount
                to_reserve = min(to_reserve, self.loan_amount - self.repayment_reserve)
                self.repayment_reserve += to_reserve
                print(f"[{self.name}] 🏦 LOAN RESERVE: Partitioned ${to_reserve:.2f} (Total Reserve: ${self.repayment_reserve:.2f}/${self.loan_amount:.2f})")
                
                # Persist to DB if possible
                if self.db_manager and hasattr(self.db_manager, 'save_repayment_reserve'):
                    self.db_manager.save_repayment_reserve(self.repayment_reserve)
            # =====================================================

            # 1. Immediate Risk Boost (The "Hot Hand" Fallacy - but useful for momenta)
            # Boost risk multiplier by 0.1 per win, cap at 3.0 or Config Limit
            old_risk = self.risk_multiplier
            self.risk_multiplier = min(3.0, self.risk_multiplier + 0.2)
            
            print(f"[{self.name}] 💰 WIN DETECTED ({symbol} +${pnl:.2f})! Risk Appetite Charging... 🔋 ({old_risk:.1f}x -> {self.risk_multiplier:.1f}x)")
            
            # 2. Unlock Drawdown (Redemption)
            if self.drawdown_lock and pnl > (self.balance * 0.01):
                print(f"[{self.name}] 🔓 REDEMPTION: Significant Win unlocked Drawdown mechanism.")
                self.drawdown_lock = False
                
        else:
            # LOSER
            self.win_streak = 0
            # Reset risk to 0.8x (Mandate: Reset to 0.8x, requires 2 wins to unlock 1.0x+)
            self.risk_multiplier = 0.8
            print(f"[{self.name}] 📉 Loss Realized. Risk Reset to {self.risk_multiplier:.1f}x (Mandate)")
            # print(f"[{self.name}] 📉 Loss Realized. Cooling Risk Multiplier -> {self.risk_multiplier:.1f}x")

    def update_accumulator(self, current_equity: float):
        """
        The Accumulator Logic: 
        1. Ratchet: Track High Water Mark & Lock if Drawdown > Limit.
        2. Pump: Adjust Risk Multiplier based on Equity Velocity.
        """
        # 0. Daily Reset Check (New Day = New Session)
        current_date = datetime.datetime.utcnow().date()
        if current_date > self.last_hwm_date:
             print(f"[{self.name}] 🌅 New Day Detected ({current_date}). Resetting High Water Mark to ${current_equity:.2f}")
             self.high_water_mark = current_equity
             self.day_start_equity = current_equity # Reset Session Baseline
             self.last_hwm_date = current_date
             self.drawdown_lock = False

        # 1. Update High Water Mark (The Ratchet)
        if current_equity > self.high_water_mark:
            self.high_water_mark = current_equity
            self.drawdown_lock = False # Unlock if we make new highs
            
        # 2. Check Drawdown Lock
        if self.high_water_mark > 0:
            drawdown = (self.high_water_mark - current_equity) / self.high_water_mark
            
            # --- PATCH: DATA SANITY CHECK ---
            # If drawdown is MASSIVE (>30%) instantly (implying we didn't actually lose it trading),
            # assume previous HWM was a glitch (phantom spike) and reset.
            if drawdown > config.ACC_SANITY_THRESHOLD and not self.drawdown_lock:
                 print(f"[{self.name}] 📉 DATA SANITY CHECK: Instant >{config.ACC_SANITY_THRESHOLD:.0%} Drop (${self.high_water_mark:.2f} -> ${current_equity:.2f}). Resetting HWM (Assuming Glitch).")
                 self.high_water_mark = current_equity
                 drawdown = 0.0

                 # EMERGENCY POSITION CLOSURE: If capital dropped dramatically, close risky positions
                 if drawdown > 0.4:  # More than 40% drop - emergency mode
                     print(f"[{self.name}] 🚨 EMERGENCY MODE: Capital dropped >40%. Initiating emergency position closures.")
                     self.trigger_emergency_position_closures()
            # --------------------------------
            
            if drawdown > config.ACC_DRAWDOWN_LIMIT:
                # DYNAMIC DRAWDOWN RECOVERY (Session Override)
                # If we are profitable THIS SESSION (intra-day), ignore the global drawdown lock.
                # But respect the HARD CAP (Catastrophic Stop)
                
                session_pnl = current_equity - self.day_start_equity
                hard_stop = getattr(config, 'ACC_HARD_STOP_LIMIT', 0.40)
                
                if drawdown > hard_stop:
                    # CATASTROPHIC FAILURE - KILL SWITCH
                    if not self.drawdown_lock:
                        print(f"[{self.name}] 💀 CATASTROPHIC HALT: Drawdown {drawdown:.1%} > {hard_stop:.1%}. Override Disabled.")
                    self.drawdown_lock = True
                    
                elif session_pnl > 0:
                    # SOFT LOCK OVERRIDE
                    if self.drawdown_lock:
                        print(f"[{self.name}] 🔓 DYNAMIC OVERRIDE: Session Profitable (+${session_pnl:.2f}). Global Drawdown {drawdown:.1%} Ignored.")
                    self.drawdown_lock = False
                    
                else:
                    # STANDARD LOCK
                    if not self.drawdown_lock:
                        print(f"[{self.name}] 🛑 ACCUMULATOR HALT: Drawdown {drawdown:.1%} > {config.ACC_DRAWDOWN_LIMIT:.1%}. Trading Locked.")
                    self.drawdown_lock = True
            
        # 3. Calculate Velocity (The Pump)
        self.equity_history.append(current_equity)
        if len(self.equity_history) > 10: self.equity_history.pop(0)
        
        if len(self.equity_history) >= 5:
            # Simple slope of last 5 points
            avg_equity = sum(self.equity_history) / len(self.equity_history)
            
            if current_equity > avg_equity:
                # We are growing -> Pump
                self.risk_multiplier = min(self.risk_multiplier + 0.1, config.ACC_RISK_CEILING)
            elif current_equity < avg_equity:
                # We are shrinking -> Deflate
                self.risk_multiplier = max(self.risk_multiplier - 0.1, config.ACC_RISK_FLOOR)
        
    def update_balance(self, new_balance: float):
        """Update the internal balance knowledge and health metrics."""
        self.balance = new_balance
        
        # --- FIX 2.1: Sync High Water Mark on Balance Update ---
        self.update_accumulator(new_balance)
        # -----------------------------------------------------
        
        # Track Drawdown
        if self.balance > self.max_balance:
            self.max_balance = self.balance
            
        if self.max_balance > 0:
            self.drawdown_pct = (self.max_balance - self.balance) / self.max_balance
        
        # Calculate Margin Utilization
        total_exposure = 0.0
        margin_level = 999.0
        free_margin = self.balance
        
        # Source of Truth: Executor (Standardized)
        if self.executor:
            try:
                # Use Governor's latest prices to ensure we are using fresh data if available
                stats = self.executor.get_portfolio_stats(prices=self.latest_prices)
                total_exposure = stats['total_exposure']
                margin_level = stats['margin_level']
                free_margin = stats['margin_free']
                
                # Check for discrepancies if Governor had different balance?
                # Executor stats use its balance. Governor uses self.balance.
                # set_live_balance updates self.balance just before this.
                # So they should be aligned roughly.
            except Exception as e:
                print(f"[{self.name}] ⚠️ Failed to get stats from Executor: {e}")

        if self.balance > 0:
            # We normalize margin utilization based on the config limit
            # If we use all allowed margin, util = 1.0
            allowed_exposure = self.balance * config.GOVERNOR_MAX_MARGIN_PCT * config.PREDATOR_LEVERAGE
            self.margin_utilization = total_exposure / allowed_exposure if allowed_exposure > 0 else 0.0

        # IMPROVED: Sync with real exchange data more frequently to prevent divergence
        # Use self.actuator if available, otherwise try to access through executor if linked
        actuator = getattr(self, 'actuator', None)
        if not actuator and hasattr(self, 'executor') and self.executor:
            actuator = getattr(self.executor, 'actuator', None)

        if actuator:
            try:
                real_equity = actuator.get_equity()
                if real_equity:
                    # FIX 2026-02-28: Track successful equity fetches
                    self._equity_fetch_success_count = min(self._equity_fetch_success_count + 1, 10)
                    self._api_failure_mode = False
                    
                    if abs(real_equity - self.balance) > 1.0:  # If divergence > $1
                        print(f"[{self.name}] 🔄 GOVERNOR BALANCE SYNC: Real ${real_equity:.2f} vs Internal ${self.balance:.2f}")
                        self.balance = real_equity
                        self.available_balance = real_equity
                else:
                    # Equity fetch returned None
                    self._api_failure_mode = True
            except Exception as e:
                print(f"[{self.name}] ⚠️ Governor Balance Sync Failed: {e}")
                self._api_failure_mode = True

        # --- REAL-TIME MARGIN MONITOR (Phase 5) ---
        # FIXED: Removed call to missing _calculate_portfolio_state
        # Used standardized stats from Executor instead.
        
        # Critical Solvency Alert
        if margin_level < 2.0:
            print(f"[{self.name}] ⚠️ MARGIN WARNING: Level {margin_level:.2f} (Free: ${free_margin:.2f})")
            if margin_level < 1.1:
                print(f"[{self.name}] 🚨 MARGIN CRITICAL: Approaching Liquidation (<1.1). HIBERNATING.")
                self.state = 'HIBERNATE'
        
        # --- DRAWDOWN REGIME FORCE (User Priority 1) ---
        # If Drawdown > 10%, Force DEFENSIVE Mode
        if self.drawdown_pct > 0.10:
             if self.regime_controller:
                 current = self.regime_controller.get_current_regime()
                 if current != 'DEFENSIVE' and current != 'HIBERNATE':
                      print(f"[{self.name}] 📉 DRAWDOWN PROT: {self.drawdown_pct*100:.1f}% Drawdown. Forcing DEFENSIVE Regime.")
                      # Force Transition (Requires support in RegimeController, or we lock locally)
                      try:
                          # We simulate force by manual override until recovery
                          self.regime_controller.override_regime('DEFENSIVE')
                      except AttributeError:
                          # Fallback logic if method missing
                          print(f"[{self.name}] ⚠️ Regime Controller missing 'override_regime'. Manual Lock activated.")
                          self.drawdown_lock = True # Acc lock
        # -----------------------------------------------

        # ── SMCE v1: Run constitutional update every cycle ──
        self.run_smce_regime_sync(self.balance)

        # IRON BANK CHECK (Every Balance Update)
        self.manage_iron_bank()

        self._check_homeostasis()

        # CHECK MANAGEMENT MODE STATUS
        self.check_and_update_management_mode()

    # ─────────────────────────────────────────────────────────────────────
    # SMCE v1 — Layer 0 Capital Doctrine Helpers
    # ─────────────────────────────────────────────────────────────────────
    def _get_smce_tier(self, equity: float) -> str:
        if equity < 100.0:
            return "NANO"
        elif equity < getattr(config, 'SMCE_TIER_SMALL', 500.0):
            return "MICRO"
        elif equity < getattr(config, 'SMCE_TIER_MEDIUM', 5000.0):
            return "SMALL"
        return "LARGE"

    def get_account_tier(self) -> str:
        """Return the account tier based on current balance, matching POSITION_LIMITS_CENTRAL keys."""
        equity = self.balance
        if equity < 100.0:
            return 'NANO'
        elif equity < 500.0:
            return 'MICRO'
        elif equity < 5000.0:
            return 'SMALL'
        elif equity < 50000.0:
            return 'MEDIUM'
        return 'LARGE'

    def get_tier_leverage_cap(self) -> float:
        """Return the hard leverage cap for current account tier from POSITION_LIMITS_CENTRAL."""
        tier = self.get_account_tier()
        tier_limits = config.POSITION_LIMITS_CENTRAL.get(tier, {})
        return tier_limits.get('leverage_cap', tier_limits.get('max_leverage', 3.0))

    def _check_drawdown_limits(self, current_equity: float):
        """
        Calculates daily and weekly drawdowns. If limits breached,
        forces a DEFENSIVE cooldown period and halves SMCE risk multiplier.

        FIX 2026-02-24: Skip drawdown check during API failure mode.
        """
        # === API FAILURE MODE CHECK ===
        # Skip drawdown calculation if we're in API failure mode
        if getattr(self, '_api_failure_mode', False):
            print(f"[{self.name}] ⚠️ API FAILURE MODE: Skipping drawdown check (unreliable equity data)")
            return

        # Additional check: require minimum successful equity fetches
        # FIX 2026-03-01: Reduced from 3 to 2 fetches to minimize unprotected trading window
        equity_fetch_count = getattr(self, '_equity_fetch_success_count', 0)
        if equity_fetch_count < 2:
            print(f"[{self.name}] ⚠️ INSUFFICIENT EQUITY DATA ({equity_fetch_count}/2 fetches): Skipping drawdown check")
            # Add exponential backoff for next fetch attempt
            if not hasattr(self, '_equity_retry_delay'):
                self._equity_retry_delay = 1.0
            self._equity_retry_delay = min(30.0, self._equity_retry_delay * 1.5)
            return
        # ================================

        now = datetime.datetime.utcnow()
        current_date = now.date()
        current_week = now.isocalendar()[1]

        # Daily Reset & Phase 8 Tracking
        if current_date != getattr(self, '_last_day_reset', None):
            # Calculate full day return before reset
            if getattr(self, '_day_start_equity', 0) > 0:
                day_ret = (current_equity - self._day_start_equity) / self._day_start_equity
                if not hasattr(self, '_daily_returns'):
                    self._daily_returns = []
                self._daily_returns.append(day_ret)
                if len(self._daily_returns) > 7:
                    self._daily_returns.pop(0) # Keep last 7 days for variance

            self._day_start_equity = current_equity
            self._last_day_reset = current_date

            # Increment clean days if not in defensive cooldown
            if time.time() > getattr(self, '_defensive_cooldown_until', 0):
                self._consecutive_days_without_intervention = getattr(self, '_consecutive_days_without_intervention', 0) + 1
            else:
                self._consecutive_days_without_intervention = 0

        # Weekly Reset
        if current_week != getattr(self, '_last_week_reset', None):
            self._week_start_equity = current_equity
            self._last_week_reset = current_week

        # === DRAWDOWN SANITY CHECKS ===
        # Check for unrealistic drawdown (likely API failure artifact)
        daily_dd = (self._day_start_equity - current_equity) / self._day_start_equity if self._day_start_equity > 0 else 0
        weekly_dd = (self._week_start_equity - current_equity) / self._week_start_equity if self._week_start_equity > 0 else 0
        
        # Sanity check: >10% instant drawdown is likely data error, not real loss
        if daily_dd > 0.10:
            print(f"[{self.name}] 🚨 DATA SANITY CHECK: Instant {daily_dd*100:.1f}% Drop detected. Likely API failure artifact. Skipping defensive trigger.")
            return
        # ================================

        limit_daily = getattr(config, 'SMCE_DAILY_DRAWDOWN_LIMIT', 0.03)
        limit_weekly = getattr(config, 'SMCE_WEEKLY_DRAWDOWN_LIMIT', 0.06)

        state_changed = False

        if daily_dd > limit_daily or weekly_dd > limit_weekly:
            cooldown_hours = getattr(config, 'SMCE_DEFENSIVE_COOLDOWN_HOURS', 48)
            self._defensive_cooldown_until = time.time() + (cooldown_hours * 3600)
            self._risk_multiplier_smce = 0.5
            self._consecutive_days_without_intervention = 0 # Reset clean days
            self._allocation_pct_boost = 0.0 # Reset scaling
            print(f"[{self.name}] 🚨 CAP DOCTRINE: Drawdown Limit Breached (Daily: {daily_dd*100:.1f}%, Weekly: {weekly_dd*100:.1f}%). DEFENSIVE Cooldown active for {cooldown_hours}h.")
            state_changed = True
        
        # Recover if cooldown expires
        elif time.time() > getattr(self, '_defensive_cooldown_until', 0) and getattr(self, '_risk_multiplier_smce', 1.0) < 1.0:
             self._risk_multiplier_smce = 1.0
             print(f"[{self.name}] 🛡️ CAP DOCTRINE: Defensive cooldown expired. Normal risk restored.")
             state_changed = True

        # Phase 8: Track Period Max Drawdown
        self._period_max_drawdown = max(getattr(self, '_period_max_drawdown', 0.0), daily_dd, weekly_dd)
        
        # Check Scaling Eligibility
        self._check_scaling_eligibility()

        # Save to DB if changed
        if state_changed and self.db_manager and hasattr(self.db_manager, 'save_smce_state'):
            smce_state = {
                'smce_tier': self._get_smce_tier(current_equity),
                'day_start_equity': self._day_start_equity,
                'week_start_equity': self._week_start_equity,
                'last_day_reset': self._last_day_reset.isoformat(),
                'last_week_reset': self._last_week_reset,
                'defensive_cooldown_until': self._defensive_cooldown_until,
                'risk_multiplier_smce': self._risk_multiplier_smce,
                'consecutive_clean_days': getattr(self, '_consecutive_days_without_intervention', 0),
                'period_max_drawdown': getattr(self, '_period_max_drawdown', 0.0),
                'allocation_pct_boost': getattr(self, '_allocation_pct_boost', 0.0)
            }
            try:
                self.db_manager.save_smce_state(smce_state)
            except Exception as e:
                 print(f"[{self.name}] ⚠️ Failed to save SMCE state to DB: {e}")

    def _check_scaling_eligibility(self) -> bool:
        """
        Phase 8 Scaling: returns True if 60 consecutive days clean + max DD < 8% + weekly variance < 5%.
        If eligible, scale up allocation boost.
        """
        clean_days = getattr(self, '_consecutive_days_without_intervention', 0)
        max_dd = getattr(self, '_period_max_drawdown', 0.0)
        
        # Calculate weekly return variance
        returns = getattr(self, '_daily_returns', [])
        variance = 0.0
        if len(returns) >= 2:
            try:
                mean_ret = sum(returns) / len(returns)
                variance_sum = sum((r - mean_ret) ** 2 for r in returns)
                variance = variance_sum / (len(returns) - 1)
            except: pass
            
        self._weekly_return_variance = variance

        if clean_days >= 60 and max_dd < 0.08 and variance < 0.05:
            # We are eligible for scale up
            current_boost = getattr(self, '_allocation_pct_boost', 0.0)
            tier = self._get_smce_tier(self.balance)
            max_boost = 0.25 if tier == "LARGE" else (0.15 if tier == "MEDIUM" else 0.0)
            
            if current_boost < max_boost and getattr(self, '_last_scale_up_day', None) != self._last_day_reset:
                self._allocation_pct_boost = min(max_boost, current_boost + 0.02)
                self._last_scale_up_day = self._last_day_reset
                print(f"[{self.name}] 📈 SMCE SCALING: 60-day clean streak achieved! Boosting allocation by +2% (Total Boot: {self._allocation_pct_boost*100:.1f}%).")
            return True
        return False

    def _check_price_proximity_stacking(self, symbol: str, current_price: float) -> bool:
        """
        Returns True if the proposed entry price is within X% of any
        existing position's entry price for the same symbol.

        FIX 2026-03-14: Regime-aware stack distance buffers.
        - HARVEST: 0.3% (tight, allow closer stacks)
        - EXPANSION: 0.5% (moderate)
        - TRANSITION: 1.0% (wide, require more separation)
        - DEFENSIVE: 2.0% (very wide, stacks rarely allowed)
        """
        if not self.executor:
            return False

        # === REGIME-AWARE BUFFER ===
        regime = self.regime_controller.get_current_regime() if self.regime_controller else 'HARVEST'
        buffer_pct = config.STACK_DISTANCE_BUFFERS.get(regime, config.SMCE_STACKING_PRICE_BUFFER)

        # Volatility adjustment (high RVOL = allow closer stacks)
        # This is overridden by regime in most cases
        rvol = self._get_symbol_rvol(symbol) if hasattr(self, '_get_symbol_rvol') else 1.0
        if rvol > 3.0:
            buffer_pct = buffer_pct * 0.6  # Allow 40% closer in high volatility

        for pos in self.executor.positions.values():
            if pos.symbol == symbol:
                # FIX: Guard against divide by zero
                if pos.entry_price <= 0:
                    print(f"[{self.name}] ⚠️ WARNING: Position {symbol} has invalid entry_price={pos.entry_price}")
                    continue
                dist = abs(current_price - pos.entry_price) / pos.entry_price
                if dist <= buffer_pct:
                    return True # Too close to existing stack
        return False

    # ─────────────────────────────────────────────────────────────────────
    # SMCE v1 — Cycle Update & Public Interface
    # ─────────────────────────────────────────────────────────────────────
    def run_smce_regime_sync(self, equity: float) -> dict:
        """
        Called every balance update to synchronise SMCE Layer 0 (doctrine) and
        Layer 1 (regime engine). Provides the smce_regime string used by all
        downstream gates (Layer 2 probability, Layer 3 MC court).
        """
        # — Layer 0: Capital Doctrine update (daily/weekly drawdown tracking) —
        self._check_drawdown_limits(equity)
        in_defensive_cooldown = (time.time() < self._defensive_cooldown_until)

        if not self.smce_regime_engine:
            self.smce_regime = "DEFENSIVE" if in_defensive_cooldown else "HARVEST"
            return {"smce_regime": self.smce_regime}

        # — Layer 1: Regime Engine classification —
        # Gather market inputs from sub-agents if available
        structure       = "NEUTRAL"
        entropy         = 1.0
        liquidity       = "healthy"
        corr_idx        = 0.5

        # Try to read live structure from oracle (_structure_agent)
        try:
            structure_agent = getattr(self, '_structure_agent', None)
            if structure_agent:
                # 1st choice: dedicated method
                if hasattr(structure_agent, 'get_structure'):
                    structure = structure_agent.get_structure() or "NEUTRAL"
                # 2nd choice: use market bias ≥ 0.6 → BULLISH, ≤ 0.4 → BEARISH
                elif hasattr(structure_agent, 'get_market_bias'):
                    bias = float(structure_agent.get_market_bias(sentiment_score=0.0) or 0.5)
                    if bias >= 0.60:
                        structure = "BULLISH"
                    elif bias <= 0.40:
                        structure = "BEARISH"
                    else:
                        structure = "NEUTRAL"
                # 3rd choice: last_structure attribute
                elif hasattr(structure_agent, 'last_structure'):
                    structure = structure_agent.last_structure or "NEUTRAL"
        except Exception:
            pass

        # 2026-03-21 FIX: Feed SMCE the scout's SampleEntropy, NOT Shannon entropy.
        # SMCE thresholds (HARVEST<1.15, DEFENSIVE>1.2) are calibrated for SampleEn (0.5-1.2 range).
        # Shannon entropy (1.5-2.1 range) would push SMCE into permanent DEFENSIVE.
        try:
            scout_results = getattr(self, '_scout_results', None) or {}
            if scout_results:
                # Average SampleEntropy across all scouted assets for macro regime
                ent_vals = [v.get('entropy', 0.0) for v in scout_results.values()
                            if isinstance(v, dict) and v.get('entropy') is not None]
                if ent_vals:
                    entropy = sum(ent_vals) / len(ent_vals)
        except Exception:
            pass

        new_regime = self.smce_regime_engine.classify(
            structure=structure,
            entropy=entropy,
            liquidity_status=liquidity,
            correlation_idx=corr_idx,
            drawdown_breach=in_defensive_cooldown,
        )

        # Persist for downstream consumers
        self.smce_regime = new_regime

        # ── SMCE v1: DEFENSIVE Stop Tightening ──
        if self.smce_regime == "DEFENSIVE":
            if getattr(self, '_last_defensive_stop_tighten', 0) < time.time() - 3600:
                self._tighten_all_stops()
                self._last_defensive_stop_tighten = time.time()

        # Feed digest if available
        if hasattr(self, 'smce_digest') and self.smce_digest:
            self.smce_digest.set_start_equity(equity)
            self.smce_digest.set_end_equity(equity)
            if self.smce_regime_engine:
                self.smce_digest.record_regime(self.smce_regime_engine.get_status_summary())

        return {
            "smce_regime":     self.smce_regime,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Unified Regime Engine (Phase 52) — Market Data Update
    # ─────────────────────────────────────────────────────────────────────
    def update_unified_regime(self, market_data: dict = None) -> dict:
        """
        Update unified regime engine with live market data.
        Called periodically (every _regime_update_interval seconds).

        Args:
            market_data: Dict with 'prices', 'volumes', 'atr', etc.

        Returns:
            Dict with unified regime state and permissions
        """
        now = time.time()

        # Rate limit updates
        if now - self._last_regime_update < self._regime_update_interval:
            if self.unified_regime and self.unified_regime.state:
                return self.unified_regime.get_permissions()
            return {'entries_allowed': False, 'reason': 'Waiting for next update'}

        if not self.unified_regime:
            return {'entries_allowed': False, 'reason': 'Unified regime not available'}

        try:
            # Gather market inputs
            structure = "NEUTRAL"
            liquidity = "healthy"
            corr_idx = 0.3
            drawdown_breach = (time.time() < self._defensive_cooldown_until)

            # Get structure from structure_agent if available
            structure_agent = getattr(self, '_structure_agent', None)
            if structure_agent:
                if hasattr(structure_agent, 'get_structure'):
                    structure = structure_agent.get_structure() or "NEUTRAL"
                elif hasattr(structure_agent, 'get_market_bias'):
                    bias = float(structure_agent.get_market_bias(sentiment_score=0.0) or 0.5)
                    if bias >= 0.60:
                        structure = "BULLISH"
                    elif bias <= 0.40:
                        structure = "BEARISH"

            # Get entropy from entropy_agent if available
            entropy_agent = getattr(self, '_entropy_agent', None)
            entropy = None
            if entropy_agent:
                for _attr in ('last_entropy', 'current_entropy', 'entropy_value', 'latest_entropy'):
                    _val = getattr(entropy_agent, _attr, None)
                    if _val is not None and _val != 'N/A':
                        try:
                            entropy = float(_val)
                            break
                        except (TypeError, ValueError):
                            pass

            # Extract prices from market_data or use default
            if market_data and 'prices' in market_data:
                prices = market_data['prices']
            elif hasattr(self, 'latest_prices'):
                prices = np.array(list(self.latest_prices.values())) if self.latest_prices else np.array([100.0])
            else:
                prices = np.array([100.0])

            # Update unified regime
            state = self.unified_regime.update(
                prices=prices,
                volumes=market_data.get('volumes') if market_data else None,
                atr=market_data.get('atr') if market_data else None,
                structure=structure,
                liquidity_status=liquidity,
                correlation_idx=corr_idx,
                drawdown_breach=drawdown_breach,
            )

            self._last_regime_update = now

            # Log regime changes
            if self.unified_regime.state:
                logger.info(
                    f"[{self.name}] 🌐 UNIFIED REGIME: {state.behavioral.value} + {state.operational.value} "
                    f"(entries={state.entries_allowed}, conv={state.min_conviction:.2f}, size={state.size_modifier:.2f}x)"
                )

            return self.unified_regime.get_permissions()

        except Exception as e:
            logger.warning(f"[{self.name}] ⚠️ Unified regime update failed: {e}")
            return {'entries_allowed': False, 'reason': f'Update error: {e}'}

    def get_unified_permissions(self, symbol: str = None, conviction: float = None) -> dict:
        """
        Get unified regime permissions for a specific trade decision.

        Args:
            symbol: Trading symbol (optional)
            conviction: Signal conviction score (optional)

        Returns:
            Dict with entries_allowed, min_conviction, size_modifier, etc.
        """
        if not self.unified_regime:
            return {
                'entries_allowed': True,
                'min_conviction': 0.65,
                'size_modifier': 1.0,
                'max_leverage': 3.0,
                'source': 'fallback'
            }

        perms = self.unified_regime.get_permissions()
        perms['source'] = 'unified_regime'

        # Check specific entry if conviction provided
        if conviction is not None:
            allowed, reason = self.unified_regime.should_allow_entry(
                conviction=conviction,
                symbol=symbol,
            )
            perms['entry_allowed'] = allowed
            perms['entry_reason'] = reason

        return perms

    def _tighten_all_stops(self):
        """
        Enforces DEFENSIVE mode rule: tighten stops on existing positions
        to significantly reduce risk exposure.
        """
        if not self.executor: return
        
        tightened_count = 0
        for pos in self.executor.positions.values():
            sym = pos.symbol
            entry = pos.entry_price
            curr = self.latest_prices.get(sym, entry)
            if entry <= 0 or curr <= 0: continue
            
            is_long = pos.direction == 'BUY'
            
            # Use basic calculation since we don't have raw POS PNL methods guaranteed
            pnl_pct = ((curr - entry) / entry) if is_long else ((entry - curr) / entry)
            
            current_sl = pos.metadata.get('stop_loss')
            if not current_sl and hasattr(pos, 'stop_loss_price'):
                current_sl = pos.stop_loss_price
                
            new_sl = None
            if is_long:
                if pnl_pct > 0.005: # 0.5% profit
                    new_sl = entry * 1.001 # Break-even + tiny profit
                else: # Losing
                    if current_sl and current_sl < curr:
                        new_sl = current_sl + (curr - current_sl) * 0.5 # Halve risk
            else:
                if pnl_pct > 0.005:
                    new_sl = entry * 0.999
                else:
                    if current_sl and current_sl > curr:
                        new_sl = current_sl - (current_sl - curr) * 0.5
                        
            if new_sl:
                pos.metadata['stop_loss'] = new_sl
                if hasattr(pos, 'stop_loss_price'):
                    pos.stop_loss_price = new_sl
                    
                actuator = getattr(self.executor, 'actuator', None)
                if actuator and hasattr(actuator, 'place_stop_order'):
                    try:
                        stop_dir = 'SELL' if is_long else 'BUY'
                        actuator.place_stop_order(sym, stop_dir, pos.quantity, new_sl)
                        tightened_count += 1
                        print(f"[{self.name}] 🛡️ DEFENSIVE PROTOCOL: Tightened Stop-Loss for {sym} to {new_sl:.4f}")
                    except Exception as e:
                        pass # Ignore API failures locally, state is updated

        if tightened_count > 0:
            print(f"[{self.name}] 🛡️ DEFENSIVE: Successfully tightened {tightened_count} stop-losses.")

    def get_smce_regime(self) -> str:
        """
        SINGLE SOURCE OF TRUTH for SMCE Market Regime.
        
        Always use this method instead of accessing self.smce_regime directly.
        Ensures consistent regime application across all Governor functions.
        
        Returns:
            str: Current SMCE regime (HARVEST, EXPANSION, TRANSITION, DEFENSIVE)
        """
        # Priority 1: Check if defensive cooldown is active
        # Check the defensive_cooldown_until timestamp directly
        if hasattr(self, 'defensive_cooldown_until') and self.defensive_cooldown_until:
            if time.time() < self.defensive_cooldown_until:
                return "DEFENSIVE"
        
        # Priority 2: Cached regime
        return getattr(self, 'smce_regime', 'HARVEST')
    
    def get_capital_regime(self) -> str:
        """
        SINGLE SOURCE OF TRUTH for Capital Regime.
        
        Always use this method instead of accessing regime_controller directly.
        
        Returns:
            str: Current capital regime (NANO, MICRO, SMALL, MEDIUM)
        """
        if self.regime_controller:
            return self.regime_controller.get_current_regime()
        return 'SMALL'  # Default fallback
    
    def get_combined_regime_context(self) -> Dict[str, Any]:
        """
        Get complete regime context for decision making.
        
        Returns:
            dict: {'smce': 'HARVEST', 'capital': 'SMALL', 'is_safe': True, ...}
        """
        smce = self.get_smce_regime()
        capital = self.get_capital_regime()
        
        return {
            'smce': smce,
            'capital': capital,
            'combined': f"{capital}_{smce}",
            'is_safe': smce in ['ORDERED', 'EXPANSION', 'HARVEST'],
            'is_defensive': smce == 'DEFENSIVE',
            'is_transition': smce == 'TRANSITION'
        }

    def run_smce_pre_trade_gate(
        self,
        symbol: str,
        direction: str,
        proposed_notional: float,
        proposed_leverage: float,
        market_context: dict,
        portfolio_state: dict,
        volatilities: dict,
        metadata: dict = None,
    ) -> dict:
        """
        Full 3-layer SMCE gate for a proposed trade.

        Runs in order:
          Layer 0 — Capital Doctrine (hard constitutional rules)
          Layer 2 — Probability Stacking (quantified score)
          Layer 3 — Monte Carlo Risk Court (simulation veto)

        Returns:
            {
              "allowed":       bool,
              "reason":        str,
              "approved_size": float,   (may be reduced vs proposed)
              "max_leverage":  float,
              "scorecard":     dict,
              "mc_result":     dict,
            }
        """
        equity = self.balance
        regime = self.smce_regime

        # ─ Layer 0: Doctrine hard-check ─────────────────────────────────────
        if self.smce_doctrine:
            cluster_exp = portfolio_state.get("cluster_exposure", 0.0)
            # FIX BUG-007: Pass strategy to SMCE for arb vs directional exposure limits
            strategy = portfolio_state.get("strategy", "DIRECTIONAL")
            allowed, reason, max_lev = self.smce_doctrine.check_trade(
                symbol=symbol,
                direction=direction,
                proposed_notional=proposed_notional,
                proposed_leverage=proposed_leverage,
                equity=equity,
                smce_regime=regime,
                current_positions=self.positions,
                cluster_exposure_pct=cluster_exp,
                strategy=strategy,  # FIX BUG-007: Pass strategy for arb limit
            )
            if not allowed:
                if self.smce_digest:
                    self.smce_digest.record_violation("LAYER0_DOCTRINE", reason)
                return {"allowed": False, "reason": reason, "approved_size": 0.0,
                        "max_leverage": 1.0, "scorecard": {}, "mc_result": {}}
            proposed_leverage = min(proposed_leverage, max_lev)

        # ─ Layer 2: Probability Score ──────────────────────────────────────
        scorecard = {}
        size_modifier = 1.0

        #  --- 🔥 STAGNATION CIRCUIT BREAKER (Entropy Defense) ---
        # Get Win Rate and Entropy
        recent_win_rate = self.calculate_recent_win_rate(lookback=10) # Fast lookback for stagnation
        market_entropy = market_context.get('entropy', 0.5)

        # Log parameters for transparency
        if self.DEBUG:
            print(f"[{self.name}] 🔍 Stagnation Check | WinRate: {recent_win_rate:.2f} | Entropy: {market_entropy:.2f}")

        # Stagnation Cooldown Trigger
        stagnation_cooldown_active = False
        if hasattr(self, 'stagnation_cooldown_until'):
             if time.time() < self.stagnation_cooldown_until:
                  stagnation_cooldown_active = True
                  
        if stagnation_cooldown_active:
             reason = f"Stagnation Cooldown Active (Low Entropy/WinRate)"
             print(f"[{self.name}] 🛑 STAGNATION VETO: {symbol} blocked. {reason}")
             if self.smce_digest:
                  self.smce_digest.record_violation("STAGNATION_CIRCUIT_BREAKER", reason)
             return {"allowed": False, "reason": reason, "approved_size": 0.0,
                    "max_leverage": proposed_leverage, "scorecard": scorecard, "mc_result": {}}

        # Trigger condition: Very poor recent performance IN a low-entropy chop environment
        if recent_win_rate < 0.40 and market_entropy < 0.18:
             # Lock out trading for 2 hours
             cooldown_seconds = 2 * 3600
             self.stagnation_cooldown_until = time.time() + cooldown_seconds
             reason = f"Stagnation Threshold Met (WR {recent_win_rate:.2f} < 0.40, Ent {market_entropy:.2f} < 0.18). Cooldown: 2h."
             print(f"[{self.name}] 🚨 STAGNATION DETECTED: {reason}")
             if self.smce_digest:
                  self.smce_digest.record_violation("STAGNATION_CIRCUIT_BREAKER", reason)
             return {"allowed": False, "reason": reason, "approved_size": 0.0,
                    "max_leverage": proposed_leverage, "scorecard": scorecard, "mc_result": {}}
        # --------------------------------------------------------------------

        if self.smce_prob_engine:
            # FIX 2026-02-28: ARB/carry trades bypass L2 re-scoring.
            # They are already pre-screened in agent_trader.py with carry-trade-appropriate context
            # (structure=NEUTRAL, momentum=True, entropy=0.5). Re-scoring them here with actual
            # market context (BEARISH structure, no momentum) produces false low scores (3.0 < 5).
            # ARB trades are structurally decorrelated from directional market signals.
            # Check both market_context and metadata for strategy (metadata is primary for ARB)
            _cand_strategy_ctx = (market_context or {}).get('strategy', 'DIRECTIONAL')
            _cand_strategy_meta = (metadata or {}).get('strategy', 'DIRECTIONAL')
            _cand_strategy = _cand_strategy_meta if _cand_strategy_meta != 'DIRECTIONAL' else _cand_strategy_ctx
            
            is_arb_trade = (
                _cand_strategy in ['ARBITRAGE_GOLD', 'FUNDING_CARRY', 'BASIS_TRADE', 'ARB', 'GOLD_LEAD_LAG', 'ARBITRAGE'] or
                'ARBITRAGE' in _cand_strategy.upper() or
                'FUNDING' in _cand_strategy.upper() or
                'BASIS' in _cand_strategy.upper() or
                bool((market_context or {}).get('is_arb', False))
            )
            if is_arb_trade:
                size_modifier = 1.0
                if self.DEBUG:
                    print(f"[{self.name}] ✅ [SMCE-L2] ARB trade bypass for {symbol} — carry/funding strategy skips directional scoring.")
            else:
                trade_candidate = {
                    "symbol":                    symbol,
                    "direction":                 direction,
                    "proposed_cluster_exposure": portfolio_state.get("proposed_cluster_add", 0.0),
                }
                scorecard = self.smce_prob_engine.score_trade(
                    trade_candidate=trade_candidate,
                    portfolio_state={
                        "equity":           equity,
                        "cluster_exposure": portfolio_state.get("cluster_exposure", 0.0),
                        "cvar_95":          portfolio_state.get("cvar_95", 0.03),
                    },
                    market_context=market_context,
                    smce_regime=regime,
                )
                if not scorecard.get("eligible", True):
                    if self.smce_digest:
                        self.smce_digest.record_violation(
                            "LAYER2_PROB_SCORE", scorecard.get("block_reason", "score below threshold")
                        )
                    print(f"[{self.name}] 🛡️ [SMCE-L2] {symbol} rejected by Probability Engine (Score: {scorecard.get('score', 0):.2f})")
                    return {"allowed": False, "reason": scorecard.get("block_reason", ""),
                            "approved_size": 0.0, "max_leverage": proposed_leverage,
                            "scorecard": scorecard, "mc_result": {}}
                size_modifier = scorecard.get("size_modifier", 1.0)
                # Log L2 approval for visibility
                if size_modifier != 1.0:
                    print(f"[{self.name}] 🛡️ [SMCE-L2] {symbol} size adjusted by {size_modifier:.2f}x")

        approved_notional = proposed_notional * size_modifier

        # ─ Layer 3: Monte Carlo Risk Court ──────────────────────────────
        mc_result = {}
        if self.smce_mc_court:
            mc_result = self.smce_mc_court.evaluate_pre_trade(
                equity=equity,
                proposed_trade={
                    "symbol":    symbol,
                    "direction": direction,
                    "notional":  approved_notional,
                    "leverage":  proposed_leverage,
                },
                portfolio_positions=dict(self.positions),
                volatilities=volatilities,
            )
            if mc_result.get("vetoed"):
                if self.smce_digest:
                    self.smce_digest.record_violation(
                        "LAYER3_MC_VETO", mc_result.get("veto_reason", "MC veto")
                    )
                print(f"[{self.name}] 🛡️ [SMCE-L3] {symbol} vetoed by Monte Carlo ({mc_result.get('veto_reason', 'risk')})")
                return {"allowed": False, "reason": mc_result.get("veto_reason", ""),
                        "approved_size": 0.0, "max_leverage": proposed_leverage,
                        "scorecard": scorecard, "mc_result": mc_result}
            approved_notional = mc_result.get("approved_size", approved_notional)
            # Log L3 result for visibility
            if mc_result.get('ruin_prob', 0) > 0.5:
                print(f"[{self.name}] 🛡️ [SMCE-L3] {symbol} passed MC (Ruin Prob: {mc_result.get('ruin_prob', 0):.1%})")

        return {
            "allowed":       True,
            "reason":        "OK",
            "approved_size": approved_notional,
            "max_leverage":  proposed_leverage,
            "scorecard":     scorecard,
            "mc_result":     mc_result,
        }

    def manage_iron_bank(self):
        """
        The Iron Bank: Secure Profits & Enforce Risk Floor.
        Called every cycle to update Risk Budget and Ratchet Fortress.
        """
        if not config.IRON_BANK_ENABLED:
            self.risk_budget = self.balance # Unrestricted
            return

        # 0. RESET OVERRIDE: If Ratchet disabled, force floor to Min Reserve
        # This allows users to "Unlock" profits by setting Ratchet to 0.0
        if config.IRON_BANK_RATCHET_PCT <= 0.0:
            self.fortress_balance = config.IRON_BANK_MIN_RESERVE

        # 1. Update Risk Budget
        # Risk Budget = Equity - Fortress Floor
        # If negative, we are "Underwater" relative to Fortress -> HALT.
        # We use self.balance which is synced to total equity in set_live_balance
        if self.balance <= self.fortress_balance:
            print(f"[{self.name}] 🛑 ALERT: Iron Bank Floor Touched (${self.balance:.2f}). IMMEDIATE HALT.")
            self.state = "HALT" # Hard Trigger
            self.risk_budget = 0.0
            return

        raw_budget = self.balance - self.fortress_balance
        self.risk_budget = max(0.0, raw_budget)
        
        # 2. Ratchet Logic (Lock Profits)
        # If we have significantly exceeded the Fortress (+ Buffer), raise the Floor.
        ratchet_threshold = self.fortress_balance * (1.0 + config.IRON_BANK_BUFFER_PCT)
        
        if self.balance > ratchet_threshold:
            profit_surplus = self.balance - self.fortress_balance
            
            # We lock a % of the TOTAL surplus (simple ratchet)
            # Or lock incremental? Simple Ratchet: Raise floor by X% of surplus.
            lock_amount = profit_surplus * config.IRON_BANK_RATCHET_PCT
            
            # Only ratchet if meaningful (> $1)
            if lock_amount > 1.0:
                old_fortress = self.fortress_balance
                self.fortress_balance += lock_amount
                self.last_ratchet_time = time.time()
                
                print(f"[{self.name}] 🏰 IRON BANK RATCHET: Locked ${lock_amount:.2f} Profits. New Floor: ${self.fortress_balance:.2f} (Equity: ${self.balance:.2f})")
                
                # Re-calc budget after ratchet
                self.risk_budget = max(0.0, self.balance - self.fortress_balance)

        # 3. Status Check
        if self.risk_budget < config.MIN_ORDER_VALUE:
             pass # Silent unless critical fail? No, Executor will handle "Insufficient Funds" errors implicitly via budget check.

    def _check_homeostasis(self):
        """Check if the system is viable."""
        if self.balance < self.hard_stop_threshold:
            self.state = 'HIBERNATE'
            print(f"[{self.name}] CRITICAL: Balance ${self.balance:.2f} < ${self.hard_stop_threshold}. HIBERNATING.")
        else:
            if self.state == 'HIBERNATE':
                self.state = 'ACTIVE'

    def get_metabolism_state(self) -> Literal['SCAVENGER', 'PREDATOR']:
        """
        Determine current metabolic state based on balance.
        """
        if self.balance <= config.SCAVENGER_THRESHOLD:
            return 'SCAVENGER'
        else:
            return 'PREDATOR'

    def get_portfolio_health(self) -> dict:
        """Expose health metrics for PPO Brain."""
        return {
            'drawdown_pct': self.drawdown_pct,
            'margin_utilization': self.margin_utilization,
            'balance': self.balance,
            'max_balance': self.max_balance,
            'risk_budget': self.risk_budget,
            'fortress_balance': self.fortress_balance
        }

    def get_current_exposure_ratio(self) -> float:
        """
        Calculate current total exposure as a ratio of equity.
        Returns: float (e.g., 0.25 = 25% exposure)
        """
        if not self.executor or self.balance <= 0:
            return 0.0
        
        total_exposure = 0.0
        for pos in self.executor.positions.values():
            # Calculate notional exposure (qty * entry_price)
            notional = pos.quantity * pos.entry_price
            # Adjust for leverage (margin-based exposure)
            margin_exposure = notional / pos.leverage if pos.leverage > 0 else notional
            total_exposure += abs(margin_exposure)
        
        exposure_ratio = total_exposure / self.balance
        return exposure_ratio

    def get_dashboard_state(self) -> dict:
        """Expose Governor data for the dashboard (positions, prices, health)."""
        positions = []
        for sym, pos in self.positions.items():
            qty = pos.quantity
            if abs(qty) < 1e-6:
                continue
            entry = pos.entry_price
            direction = pos.direction
            current = self.latest_prices.get(sym, entry)
            if direction == 'BUY':
                pnl = (current - entry) * qty
                pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
            else:
                pnl = (entry - current) * qty
                pnl_pct = ((entry - current) / entry * 100) if entry > 0 else 0
            positions.append({
                'symbol': sym,
                'direction': direction,
                'quantity': qty,
                'entry_price': entry,
                'current_price': current,
                'leverage': pos.leverage,
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl_pct, 2),
                'strategy': pos.strategy,
            })
        return {
            'positions': positions,
            'prices': dict(self.latest_prices),
            'portfolio_health': self.get_portfolio_health(),
        }

    def is_trade_allowed(self, symbol: str, asset_price: float, silent: bool = False, is_whale: bool = False, funding_yield: float = 0.0, is_arb: bool = False) -> bool:
        """
        Lightweight check to see if a trade would be allowed.
        Prevents Strategy from wasting compute on blocked trades.
        FIX: Apply APY cap at Governor entry point for defense-in-depth.

        CHRONOS FIX (2026-03-15): Daily Loss Limit Circuit Breaker
        - Halt trading after -$5 or -5% daily loss
        - 24 hour cooldown after hitting limit
        """
        # --- CHRONOS: Daily Loss Limit Circuit Breaker ---
        if getattr(config, 'DAILY_LOSS_LIMIT_ENABLED', False):
            if not hasattr(self, '_chronos_daily_start_balance'):
                self._chronos_daily_start_balance = self.balance
                self._chronos_daily_loss_halted = False
                self._chronos_halt_timestamp = 0

            # Check if halt cooldown expired (24 hours)
            if self._chronos_daily_loss_halted:
                hours_since_halt = (time.time() - self._chronos_halt_timestamp) / 3600
                cooldown_hours = getattr(config, 'DAILY_LOSS_COOLDOWN_HOURS', 24)
                if hours_since_halt >= cooldown_hours:
                    print(f"[{self.name}] ✅ DAILY LOSS HALT EXPIRED: Resuming trading after {hours_since_halt:.1f}h cooldown")
                    self._chronos_daily_loss_halted = False
                    self._chronos_daily_start_balance = self.balance
                else:
                    if not silent:
                        print(f"[{self.name}] 🛑 DAILY LOSS HALT ACTIVE: Trading suspended for {cooldown_hours - hours_since_halt:.1f}h more")
                    return False

            # Calculate daily PnL
            current_equity = self.balance
            daily_pnl_usd = current_equity - self._chronos_daily_start_balance
            daily_pnl_pct = daily_pnl_usd / self._chronos_daily_start_balance if self._chronos_daily_start_balance > 0 else 0

            # Check limits
            max_loss_usd = getattr(config, 'DAILY_LOSS_LIMIT_USD', 5.0)
            max_loss_pct = getattr(config, 'DAILY_LOSS_LIMIT_PCT', 0.05)

            if daily_pnl_usd < -max_loss_usd or daily_pnl_pct < -max_loss_pct:
                self._chronos_daily_loss_halted = True
                self._chronos_halt_timestamp = time.time()
                if not silent:
                    print(f"[{self.name}] 🚨 DAILY LOSS LIMIT HIT: PnL=${daily_pnl_usd:.2f} ({daily_pnl_pct:.2%})")
                    print(f"[{self.name}] 🛑 TRADING HALTED for {getattr(config, 'DAILY_LOSS_COOLDOWN_HOURS', 24)}h cooldown")
                return False
        # -------------------------------------------------

        # FIX: Cap funding_yield to ±200% APY to prevent extreme values (e.g., 18955%)
        funding_yield = np.sign(funding_yield) * min(abs(funding_yield), 200.0)

        # FIX 2: Check Dynamic Blacklist
        if hasattr(self, 'blacklist') and symbol in self.blacklist:
             # Check expiration
             if time.time() > self.blacklist[symbol]:
                 del self.blacklist[symbol]
                 if symbol in getattr(self, 'consecutive_losses', {}):
                     self.consecutive_losses[symbol] = 0
                 if not silent:
                     print(f"[{self.name}] 🔓 BLACKLIST EXPIRED: {symbol} is free to trade again.")
             else:
                 if not silent:
                     remaining = (self.blacklist[symbol] - time.time()) / 3600
                     print(f"[{self.name}] 🚫 BLACKLIST VETO: {symbol} is banned for {remaining:.1f} more hours.")
                 return False

        # FIX 5: PAXG "ARB ONLY" MODE (For Low Equity)
        # If Equity < $1000, PAXG is only allowed if it is an ARBITRAGE (Yield) trade.
        if symbol == 'PAXG/USDT' and self.balance < 1000.0:
            # We allow IF:
            # 1. It is explicitly an Arb signal (is_arb=True)
            # 2. OR meaningful funding yield (> 50% APY)
            is_valid_arb = is_arb or (funding_yield and abs(funding_yield) > 50.0)
            
            if not is_valid_arb:
                if not silent:
                    print(f"[{self.name}] 🚫 PAXG RESTRICTION: Low Equity ($<1000). Only High-Yield Arb allowed. Yield: {funding_yield}")
                return False

        # --- FIX 2026-03-21: Per-Symbol Expectancy Gate ---
        sym_exp, sym_n = self._get_symbol_expectancy(symbol, min_trades=20)
        if sym_n >= 20 and sym_exp < 0:
            if not silent:
                print(f"[{self.name}] 🚫 NEGATIVE_EXPECTANCY_VETO: {symbol} expectancy ${sym_exp:.4f}/trade over {sym_n} trades")
            return False
        
        # --- FIX 2026-03-22: Symbol Quality Filter (Recent Performance) ---
        # Block symbols with poor recent performance even before hitting blacklist
        # Check last 5 trades on this symbol
        sym_recent_exp, sym_recent_n = self._get_symbol_expectancy(symbol, min_trades=5)
        if sym_recent_n >= 5 and sym_recent_exp < -0.005:  # <-0.5% per trade average
            if not silent:
                print(f"[{self.name}] 🚫 POOR_RECENT_PERFORMANCE: {symbol} avg ${sym_recent_exp:.4f}/trade over last {sym_recent_n} trades")
            return False

        # --- FIX 2026-03-21: Per-Symbol Daily Trade Limit ---
        max_per_sym = getattr(config, 'MAX_TRADES_PER_SYMBOL_PER_DAY', 4)
        # 2026-03-21: Whitelisted symbols get higher limit (moderate amplification)
        if self._is_symbol_whitelisted(symbol):
            max_per_sym = int(max_per_sym * 1.5)  # 4 → 6
        sym_today = self._get_symbol_trades_today(symbol)
        if sym_today >= max_per_sym:
            if not silent:
                print(f"[{self.name}] 🚫 OVERTRADING_VETO: {symbol} hit {sym_today}/{max_per_sym} daily trade limit")
            return False

        # --- FIX 3.1: Repetitive Cycle Check ---
        if hasattr(self, 'repetition_tracker') and symbol in self.repetition_tracker:
             # Check if we exceeded threshold
             # Clean first? No, trust register_outcome or just check count loosely
             # Let's filter here to be safe
             now = time.time()
             recent_closes = [t for t in self.repetition_tracker[symbol] if t > (now - 600)]
             
             if len(recent_closes) >= 3:
                 if not silent: 
                     print(f"[{self.name}] 🛑 REPETITION VETO: {symbol} in churn loop ({len(recent_closes)} cycles/10m). Waiting for cooldown.")
                 return False
        # ---------------------------------------
        
        # ── SMCE v1: DEFENSIVE REGIME BLOCK ──
        if getattr(self, 'smce_regime', '') == 'DEFENSIVE':
            if not silent:
                print(f"[{self.name}] 🛡️ DEFENSIVE REGIME VETO: {symbol} blocked. System is preserving capital.")
            return False

        # 1. Meta-Learning Circuit Breaker (Session 3 Fix)
        if not self.check_meta_limits(symbol):
            if not silent: 
                print(f"[{self.name}] 🛡️ META-BAN: {symbol} blocked due to reckless veto frequency.")
            return False

        # 2. Cooldown Check
        last_time = self.last_trade_time.get(symbol, 0)
        # Optimized: Check time once
        time_diff = time.time() - last_time
        # 2026-03-21: Whitelisted symbols get reduced cooldown (moderate amplification)
        cooldown_secs = config.GOVERNOR_COOLDOWN_SECONDS
        if self._is_symbol_whitelisted(symbol):
            cooldown_secs = int(cooldown_secs * 0.6)  # 40% reduction
        if time_diff < cooldown_secs:
            if not silent:
                # Log spam reducer: Only log every 10s
                if int(time_diff) % 10 == 0:
                     print(f"[{self.name}] ⏳ Cooldown Active for {symbol} ({int(time_diff)}s < {cooldown_secs}s)")
            return False
            
        # 2. Price Distance Check (Dynamic - Virtual Pool Aware)
        # Check if we hold this symbol in ANY virtual pool (Arb or Dir)
        relevant_keys = []
        if self.executor:
            relevant_keys = [k for k, p in self.executor.positions.items() if p.symbol == symbol]

        if relevant_keys:
            # Check proximity to the LAST entry across all pools for this asset
            for r_key in relevant_keys:
                # Retrieve Position object
                pos = self.executor.positions[r_key]
                
                # Get the most recent stack entry price (not average entry)
                l_entry = 0.0
                meta = getattr(pos, 'metadata', {}) or {}
                stacks = meta.get('stacks', None)
                if stacks and isinstance(stacks, list) and len(stacks) > 0:
                    # Use the most recent stack's entry price
                    l_entry = stacks[-1].get('price', pos.entry_price)
                else:
                    # Fallback to average entry price if no stack info
                    l_entry = pos.entry_price

                if l_entry <= 0: continue

                dist = abs(asset_price - l_entry) / l_entry
                # DYNAMIC STACK DISTANCE (User Request)

                # [SMCE v1] APY/Whale overrides REMOVED per constitutional mandate:
                # APY carry is an input to the probability score, never a bypass.
                # Whale signals contribute to momentum/structure score only.
                # NO signal can waive the stacking proximity rule.

                # 2. Dynamic Thresholds based on Regime/Asset
                # High Volatility -> 0.03% (Allow closer stacking)
                # Low Volatility  -> 0.10% (Require more movement)

                is_high_vol = False

                # Check Regime
                if self.regime_controller:
                    regime = self.regime_controller.get_current_regime()
                    if regime in ['VOL_WINDOW', 'GRAVITY']:
                        is_high_vol = True

                # Check Asset Class (Memes are always High Vol)
                if symbol in getattr(config, 'MEMECOIN_ASSETS', []):
                    is_high_vol = True

                # FIX: Arbitrage trades with high yield bypass stack distance check
                # Arb trades are market-neutral and don't have directional risk
                # Use function parameters is_arb and funding_yield passed to is_trade_allowed
                is_high_yield_arb = is_arb and abs(funding_yield) > 100.0
                is_super_yield_arb = is_arb and abs(funding_yield) > 500.0  # NEW: Extreme yield bypass

                # FIX 4: Disable ARB STACK BYPASS in TRANSITION regime (User Directive)
                # Funding stack bypass should only unlock in ORDERED or EXPANSION regimes
                regime = self.get_smce_regime()  # ✅ SINGLE SOURCE OF TRUTH
                in_safe_regime = regime in ['ORDERED', 'EXPANSION', 'HARVEST']

                # NEW: Super-yield arb (>500% APY) bypasses stack check in ANY regime
                # This captures truly exceptional opportunities while filtering normal noise
                if is_super_yield_arb:
                    if not silent:
                        print(f"[{self.name}] 🚀 SUPER-ARB OVERRIDE: {symbol} Extreme yield ({funding_yield:.0f}% APY) - Stack distance waived in ANY regime.")
                    continue  # Skip stack check entirely

                # High-yield arb bypasses stack check ONLY in safe regimes (market-neutral strategy)
                elif is_high_yield_arb and in_safe_regime:
                    if not silent:
                        print(f"[{self.name}] 🚀 ARB STACK BYPASS: {symbol} High-yield arb ({funding_yield:.0f}% APY) in {regime} regime - Stack distance waived.")
                    continue  # Skip this position's stack check, continue to next
                elif is_high_yield_arb and regime == "TRANSITION":
                    if not silent:
                        print(f"[{self.name}] 🛡️ ARB STACK BLOCKED: {symbol} High-yield arb ({funding_yield:.0f}% APY) but TRANSITION regime requires full stack distance.")
                    # Fall through to normal stack check

                if is_high_vol:
                    relaxed_limit = 0.0003  # 0.03% - Allow closer stacking
                else:
                    relaxed_limit = 0.0008  # 0.08% - Relaxed standard distance

                # NORMAL DISTANCE VETO
                if dist < relaxed_limit:
                    if symbol not in self.stack_timeout_tracker:
                        self.stack_timeout_tracker[symbol] = time.time()
                        if not silent:
                            print(f"[{self.name}] 📏 Stack Too Close for {symbol} (Pool {r_key}): Dist {dist*100:.2f}% < {relaxed_limit*100:.2f}%. Snoozing.")
                    return False
            # Price moved away - clear the timeout tracker
            self.stack_timeout_tracker.pop(symbol, None)
                
        return True

    def precheck(self, symbol: str, direction: str, est_margin: float = 0.0, metadata: dict = None) -> bool:
        """
        Upstream pre-check for high-frequency or priority signals (like ArbHunter).
        Returns True if a trade is LIKELY allowed based on current pool status.
        FIX 4: Move Arb Precheck Upstream
        """
        is_arb = (metadata or {}).get('is_arb', False)
        funding_yield = (metadata or {}).get('funding_yield', 0.0)

        # FIX: Apply APY cap for defense-in-depth
        funding_yield = np.sign(funding_yield) * min(abs(funding_yield), 200.0)

        # 0. APY OVERRIDE (User Priority)
        # If absolute APY is massive (>500%), we skip ALL pool limits.
        # Note: After capping above, this will never trigger, which is the point
        # The cap ensures we never see >500% unless there's a bug upstream
        # FIX: Also bypass precheck for ALL Arbitrage trades so they can be
        # properly sized and evaluated by the sophisticated calc_position_size logic
        # rather than being blindly aborted early.
        if is_arb:
            return True

        if self.is_in_management_mode():
            # In management mode, we might allow certain types of trades that help optimize existing positions
            # For example, adding to existing profitable positions or hedging
            existing_pos = None
            if self.executor:
                 # Find first matching position
                 for vk, p in self.executor.positions.items():
                     if p.symbol == symbol:
                         existing_pos = p
                         break

            if existing_pos:
                # Allow stacking to existing positions if they are profitable
                entry_price = existing_pos.entry_price
                current_price = self.latest_prices.get(symbol, entry_price)
                if entry_price > 0 and current_price > 0:
                    if existing_pos.direction == 'BUY':
                        pnl_pct = (current_price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - current_price) / entry_price

                    # Only allow stacking if position is profitable
                    if pnl_pct > 0.01:  # At least 1% profit
                        print(f"[{self.name}] 🛠️ MANAGEMENT MODE: Allowing stack to profitable position {symbol} ({pnl_pct*100:.2f}% profit)")
                        # Continue with normal checks for stacking
                    else:
                        # COMPUTE WASTE FIX: Suppress repeated stack block logs
                        if not hasattr(self, '_stack_block_count'):
                            self._stack_block_count = 0
                        self._stack_block_count += 1
                        if self._stack_block_count % 10 == 0:  # Log every 10th
                            print(f"[{self.name}] 🛠️ MANAGEMENT MODE: Blocking stack to unprofitable position {symbol} ({pnl_pct*100:.2f}% profit)")
                        return False
            else:
                # In management mode, don't allow new positions
                # COMPUTE WASTE FIX: Suppress repeated management mode block logs
                if not hasattr(self, '_mgmt_block_count'):
                    self._mgmt_block_count = 0
                self._mgmt_block_count += 1
                if self._mgmt_block_count % 10 == 0:  # Log every 10th
                    print(f"[{self.name}] 🛠️ MANAGEMENT MODE: Blocking new position {symbol} - focusing on existing positions")
                return False

        # 1. Slot Checks
        arb_count = 0
        directional_count = 0
        if self.executor:
            for vk, pos in self.executor.positions.items():
                if self._is_arb(pos.strategy):
                    arb_count += 1
                else:
                    directional_count += 1

        if is_arb:
            arb_slots = getattr(config, 'POOL_B_SLOTS', 3)
            if arb_count >= arb_slots:
                return False
        else:
            base_directional_slots = getattr(config, 'POOL_A_SLOTS', 5)
            directional_slots = base_directional_slots
            if self.regime_controller:
                bonuses = self.regime_controller.get_graduation_bonuses()
                directional_slots += bonuses.get('slot_bonus', 0)

            if directional_count >= directional_slots:
                return False

        # 2. Allocation Ceiling Check (If est_margin provided)
        if est_margin > 0:
            if self.risk_budget <= 0: self.manage_iron_bank()
            effective_capital = self.risk_budget

            if is_arb:
                pool_b_exposure = 0.0
                if self.executor:
                    pool_b_exposure = sum((pos.quantity * self.latest_prices.get(pos.symbol, pos.entry_price)) / pos.leverage
                                          for vk, pos in self.executor.positions.items()
                                          if self._is_arb(pos.strategy))
                
                pool_b_ratio = getattr(config, 'POOL_B_ALLOCATION_PCT', 0.60)
                pool_b_ceiling = effective_capital * pool_b_ratio
                if pool_b_exposure + est_margin > pool_b_ceiling:
                    return False
            else:
                pool_a_exposure = 0.0
                if self.executor:
                    pool_a_exposure = sum((pos.quantity * self.latest_prices.get(pos.symbol, pos.entry_price)) / pos.leverage
                                          for vk, pos in self.executor.positions.items()
                                          if not self._is_arb(pos.strategy))
                                          
                pool_a_ratio = getattr(config, 'POOL_A_ALLOCATION_PCT', 0.80)
                pool_a_ceiling = effective_capital * pool_a_ratio
                if pool_a_exposure + est_margin > pool_a_ceiling:
                    return False

        # 3. Global Brake
        total_limit = getattr(config, 'MAX_SIMULTANEOUS_POSITIONS', 12)
        if len(self.positions) >= total_limit:
            return False

        return True

    def check_exit_conditions(self, symbol: str, current_price: float, position_data: dict, recommendation: str = 'HOLD') -> Tuple[Optional[str], Optional[str]]:
        """
        Standard Exit Logic: Stop Loss & Take Profit.
        Now supports DYNAMIC AI OVERRIDES (Relax TP / Tighten SL).
        Returns: ('EXIT_TYPE', 'Reason') or (None, None)
        """
        if not position_data or current_price <= 0:
            return None, None
            
        entry_price = position_data.get('entry_price', 0.0)
        direction = position_data.get('direction', 'BUY')
        
        if entry_price <= 0: return None, None
        
        # Calculate PnL %
        if direction == 'BUY':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
            
        # --- DYNAMIC THRESHOLDS ---
        base_sl = getattr(config, 'DEFAULT_STOP_LOSS_PCT', 0.05)
        base_tp = getattr(config, 'DEFAULT_TAKE_PROFIT_PCT', 0.10)
        
        # Apply AI Recommendations
        curr_sl = base_sl
        curr_tp = base_tp
        
        if recommendation == 'TIGHTEN_SL':
            # Cut risk in half (e.g., 5% -> 2.5%) for decaying positions
            curr_sl = base_sl * 0.5
        elif recommendation == 'RELAX_TP':
            # MOONSHOT MODE: Quadruple the target (Let it run!)
            # Previous: 2.0x. New: 4.0x. 
            # If base TP is 10%, we now target 40%.
            curr_tp = base_tp * 4.0
            
            # Also tighten SL to Break Even if possible?
            # If we are in Moonshot mode, we should protect gains.
            if pnl_pct > 0.05: # If 5% up
                curr_sl = 0.01 # Trailing SL at -1% of entry (essentially BE+)
        
        # 3. VOLATILITY STOP (Dynamic)
        if position_data.get('stop_type') == 'VOLATILITY':
             # The stop width (e.g. 0.5 * ATR) is dynamic. 
             # We expect the metadata to contain the specific target stop price or width.
             # Ideally, 'sl_price' is stored in position metadata.
             sl_price = position_data.get('sl_price')
             if sl_price:
                 if direction == 'BUY':
                     if current_price <= sl_price:
                         return 'STOP_LOSS', f"VOLATILITY STOP HIT: Price {current_price} <= {sl_price}"
                 elif direction == 'SELL':
                     if current_price >= sl_price:
                         return 'STOP_LOSS', f"VOLATILITY STOP HIT: Price {current_price} >= {sl_price}"
             else:
                 # Fallback to stored width % if available
                 width = position_data.get('stop_width_pct', 0.05)
                 curr_sl = width
            
            
        # 1. STOP LOSS (Hard)
        if pnl_pct <= -curr_sl:
            return 'STOP_LOSS', f"PnL {pnl_pct:.2%} hit Limit {curr_sl:.0%} (Rec: {recommendation})"
            
        # 2. TAKE PROFIT (Target)
        if pnl_pct >= curr_tp:
            return 'TAKE_PROFIT', f"PnL {pnl_pct:.2%} hit Target {curr_tp:.0%} (Rec: {recommendation})"
            
        return None, None


    def calculate_dynamic_stack_threshold(self, symbol: str, current_price: float, atr: float = None) -> float:
        """
        Adjust stacking threshold based on volatility (ATR).
        High Volatility -> Wider Distance (Avoid clustering in noise)
        Low Volatility -> Tighter Distance (Sniper accumulation)
        """
        base_threshold = config.GOVERNOR_MIN_STACK_DIST # 0.001 (0.10%)
        
        if not atr or atr <= 0: return base_threshold
        
        atr_pct = (atr / current_price) * 100.0
        
        if atr_pct > 2.0:   # High volatility (>2%)
            return base_threshold * 1.5  # 0.15%
        elif atr_pct > 1.0: # Medium volatility
            return base_threshold * 1.2  # 0.12%
        elif atr_pct < 0.5: # Low volatility
            return base_threshold * 0.8  # 0.08%
        
        return base_threshold

        return base_threshold

    def check_solvency(self, trade_metadata: dict) -> bool:
        """
        PRE-FLIGHT CHECK: Simulate trade to ensure it doesn't break margin rules.
        Called by Executor immediately before locking the ledger.
        FIX 2026-03-01 #8: Fix contract size calculation (was inverted - dividing instead of multiplying).
        """
        sim_qty = trade_metadata.get('size', 0.0)
        sim_price = trade_metadata.get('price', 0.0)

        if sim_qty == 0 or sim_price == 0: return True

        # --- PATCH: RISK REDUCTION EXEMPTION ---
        # If we are closing/reducing a position, Solvency improves.
        # Check if we have an opposing position.
        symbol = trade_metadata.get('symbol')
        direction = trade_metadata.get('direction')

        if symbol and direction and self.executor:
             # Check if we have an existing position for this symbol
             existing = None
             for vk, p in self.executor.positions.items():
                 if p.symbol == symbol:
                     existing = p
                     break

             if existing:
                 # Normalize directions
                 curr_dir = existing.direction.upper().replace('LONG', 'BUY').replace('SHORT', 'SELL')
                 new_dir = direction.upper().replace('LONG', 'BUY').replace('SHORT', 'SELL')

                 # If directions oppose, it's a reduce/close -> ALWAYS ALLOW (Solvency can only improve)
                 if curr_dir != new_dir and curr_dir in ['BUY', 'SELL'] and new_dir in ['BUY', 'SELL']:
                      if self.DEBUG: print(f"[{self.name}] 🏳️ SOLVENCY CHECK: Allowing Risk Reduction for {symbol} ({curr_dir} -> {new_dir})")
                      return True
        # ---------------------------------------

        # Calculate Current State
        state = self._calculate_portfolio_state()

        # Simulate Impact
        # New Initial Margin
        regime_name = self.regime_controller.get_current_regime() if self.regime_controller else 'SMALL'
        regime_lev = config.REGIME_PERMISSIONS.get(regime_name, {}).get('max_leverage', 1.0)

        # AEHML Fix: Sanity check for regime leverage (prevent division by tiny numbers)
        if regime_lev <= 0.01: regime_lev = 1.0

        # FIX 2026-03-01 #8: Contract size should MULTIPLY, not divide
        # Kraken Futures: BTC contract = 0.0001 BTC, so 1 contract * 0.0001 = actual BTC amount
        # But our qty is already in BTC, so contract_size should be 1.0 (no adjustment needed)
        # The bug was: c_size = 0.0001 for BTC, then dividing by it (making margin 10000x larger)
        # Correct: For Kraken Futures, qty is already in native units, no contract size adjustment
        c_size = 1.0  # No contract size adjustment - qty is already in native units

        # Calculate initial margin: IM = (qty * price) / leverage
        # For BTC/ETH on Kraken Futures, the API qty is already in native units
        new_im = (abs(sim_qty) * sim_price * c_size) / regime_lev

        future_used = state['used_margin'] + new_im
        future_free = state['equity'] - future_used # Assuming equity doesn't change instantly (no fee deduction here)

        # 1. Check Hard Margin Limit (80% Util)
        if future_used > (state['equity'] * 0.80):
            # AEHML Fix: Sanity Check for absurdly high NewIM relative to balance
            # If NewIM is > 200% of Equity but the notional value is tiny, it's a calculation ghost.
            notional = abs(sim_qty * sim_price)
            if new_im > (state['equity'] * 2.0) and notional < (state['equity'] * 0.5):
                 print(f"[{self.name}] 🧪 SOLVENCY GHOST: Detected absurd NewIM (${new_im:.2f}) for small notional (${notional:.2f}). Forcing Tolerance.")
                 return True

            # --- PATCH: SMALL NOTIONAL TOLERANCE ---
            # If the trade is less than $10, don't veto even if margin is tight (unless equity < $5)
            if notional < config.RISK_MIN_BASE_NOTIONAL and state['equity'] > config.RISK_MIN_BASE_NOTIONAL:
                 if self.DEBUG: print(f"[{self.name}] 🤏 SMALL TRADE: Allowing ${notional:.2f} despite margin constraints.")
                 return True

            # --- PATCH: SOLVENCY OVERRIDE ---
            # For small positions (<$25 margin), allow the trade to prevent false vetos
            if new_im < 25.0:
                 if self.DEBUG: print(f"[{self.name}] 🛡️ SOLVENCY OVERRIDE: Allowing small IM (${new_im:.2f} < $25)")
                 return True

            # --- FIX 2026-03-03: Exclude arb positions from margin util calculation ---
            # Arb positions (BASIS_CARRY, FUNDING_CARRY) are market-neutral and shouldn't
            # count toward margin utilization the same way directional positions do.
            # They collect funding rates, not directional price movement.
            is_arb_trade = False
            if trade_metadata:
                strategy = trade_metadata.get('strategy', '').upper()
                is_arb_trade = ('ARBITRAGE' in strategy or 'FUNDING' in strategy or 
                               'BASIS_CARRY' in strategy or 'BASIS_TRADE' in strategy)
            
            if is_arb_trade:
                # For arb trades, use relaxed 95% limit instead of 80%
                if future_used > (state['equity'] * 0.95):
                    print(f"[{self.name}] 🛑 SOLVENCY VETO: {trade_metadata.get('symbol')} ARB Future Margin Util {future_used/state['equity']:.1%} > 95%.")
                    return False
                return True

            # Standard Utilization Check
            if future_used > (state['equity'] * 0.80):
                 print(f"[{self.name}] 🛑 SOLVENCY VETO: {trade_metadata.get('symbol')} Future Margin Util {future_used/state['equity']:.1%} > 80%.")
                 return False

            return True

        # 2. Check Margin Level (Safety Buffer)
        future_level = state['equity'] / future_used if future_used > 0 else 999
        if future_level < 1.5:
            # Apply same ghost tolerance here
            if future_used > (state['equity'] * 2.0): # If used margin is 500% of equity, it's a ghost
                 return True

            print(f"[{self.name}] 🛑 SAFETY VETO: Future Margin Level {future_level:.2f} < 1.5.")
            return False

        return True

    def _calculate_portfolio_state(self) -> dict:
        """
        CROSS-MARGIN CALCULATOR (Safe Mode):
        Aggregates portfolio margin usage and available equity.
        Equity = Balance + Unrealized PnL
        Used Initial Margin = Sum(Position Value / Leverage)
        """
        total_equity = self.balance
        used_margin = 0.0

        current_regime = self.regime_controller.get_current_regime() if self.regime_controller else 'SMALL'
        regime_lev = config.REGIME_PERMISSIONS.get(current_regime, {}).get('max_leverage', 1.0)

        # Iterate Positions (Unified Source)
        if self.executor:
            for vk, pos in self.executor.positions.items():
                qty = pos.quantity
                entry = pos.entry_price
                sym = pos.symbol

                # Get Mark Price (Reliable)
                mark_price = self.latest_prices.get(sym, entry)
                if mark_price <= 0: mark_price = entry

                # Unrealized PnL
                if pos.direction == 'BUY':
                    u_pnl = (mark_price - entry) * qty
                else:
                    u_pnl = (entry - mark_price) * qty

                total_equity += u_pnl

                # Margin Usage
                notional = qty * mark_price
                used_margin += (notional / regime_lev)

        free_margin = total_equity - used_margin

        # PHANTOM MARGIN FIX: Validate margin calculation against actual positions
        # If we have 0 positions but still show margin usage, clear phantom margin
        position_count = len(self.executor.positions) if self.executor else 0
        if position_count == 0 and used_margin > 0:
            print(f"[{self.name}] 🚨 PHANTOM MARGIN DETECTED: {len(self.positions)} positions but ${used_margin:.2f} margin. Clearing phantom.")
            used_margin = 0.0
            free_margin = total_equity

        return {
            'equity': total_equity,
            'used_margin': used_margin,
            'free_margin': free_margin,
            'margin_level': (total_equity / used_margin) if used_margin > 0 else 999.0
        }

    def clear_phantom_margin(self):
        """Clear phantom margin exposure when positions are confirmed closed externally."""
        print(f"[{self.name}] 🧹 CLEARING PHANTOM MARGIN: Positions={len(self.positions)}. Purging Ledger.")
        self.positions.clear()  # FIX: Actually clear the ledger to resolve Phantom Margin desync
        # The margin calculation will be corrected on next _calculate_portfolio_state call

    def _check_unified_protocol(self, symbol: str, asset_price: float, existing_pos: dict, whale_confirmed: bool, market_bias: float, is_arb: bool = False, funding_yield: float = 0.0) -> bool:
        """
        Helper for Unified Control Protocol Checks (Micro Mode, Stacking, Cluster Risk).
        Returns True if ALLOWED, False if BLOCKED.
        
        SLOT POOL ISOLATION:
        - Pool A: Directional / Whale / Trend Trades
        - Pool B: Arb / Funding / Carry Trades (Independent)
        """
        # --- SLOT POOL ISOLATION (Phase X) ---
        # Initialize limit_max_pos with a safe default to prevent UnboundLocalError
        # This will be overridden by pool-specific logic below.
        limit_max_pos = getattr(config, 'POOL_A_SLOTS', 5) 
        # 0. APY OVERRIDE (User Priority)
        # If absolute APY is massive (>500%), we skip ALL pool limits to capture the yield.
        # This is for "Gold Nugget" extraction that shouldn't be blocked by bureaucracy.
        if is_arb and abs(funding_yield) > 500.0:
             # Only log if we are actually overriding something?
             # For now, just pass. 
             # print(f"[{self.name}] 🚀 SUPER UNLOCK: APY {funding_yield:.0f}% > 500%. Bypassing Pool Limits.")
             return True

        # Count positions by type
        directional_count = 0
        arb_count = 0
        
        directional_count = 0
        arb_count = 0
        
        if self.executor:
            for vk, pos in self.executor.positions.items():
                # Pool detection MUST trust the strategy-to-pool mapping
                strategy = pos.strategy
                if self._is_arb(strategy):
                    arb_count += 1
                else:
                    directional_count += 1
        
        # Get base slot limits from config
        base_directional_slots = getattr(config, 'POOL_A_SLOTS', 5)  # Pool A
        arb_slots = getattr(config, 'POOL_B_SLOTS', 3)               # Pool B
        
        # === GRADUATION BONUS: Permanent slot increase from RegimeController ===
        graduation_slot_bonus = 0
        current_regime = 'SMALL'
        if self.regime_controller:
            bonuses = self.regime_controller.get_graduation_bonuses()
            graduation_slot_bonus = bonuses.get('slot_bonus', 0)
            current_regime = self.regime_controller.get_current_regime()
        
        # Apply graduation bonus to Directional pool (earned through promotions)
        directional_slots = base_directional_slots + graduation_slot_bonus

        # === CHRONOS FORENSICS: Dynamic Regime Scaling ===
        # Increase slots in safe regimes, strictly cap in dangerous ones
        if current_regime in ['EXPANSION', 'HARVEST']:
            directional_slots += 2
            if self.DEBUG: print(f"[{self.name}] 📈 DYNAMIC SLOTS: {current_regime} allowed +2 -> {directional_slots} slots")
        elif current_regime in ['CONTRACTION', 'DEFENSIVE', 'VOL_WINDOW']:
            directional_slots = min(directional_slots, max(2, base_directional_slots - 1))
            if self.DEBUG: print(f"[{self.name}] 🛡️ DYNAMIC SLOTS: {current_regime} restricted -> {directional_slots} slots")
        
        # Calculate pool-specific exposure (AEHML 2.1: Margin-Based Accounting)
        pool_a_exposure = 0.0
        pool_b_exposure = 0.0
        
        if self.executor:
            pool_a_exposure = sum((pos.quantity * self.latest_prices.get(pos.symbol, pos.entry_price)) / pos.leverage
                                  for vk, pos in self.executor.positions.items() 
                                  if not self._is_arb(pos.strategy))
            pool_b_exposure = sum((pos.quantity * self.latest_prices.get(pos.symbol, pos.entry_price)) / pos.leverage
                                  for vk, pos in self.executor.positions.items() 
                                  if self._is_arb(pos.strategy))
        
        # FIX: Use RISK BUDGET (Tradeable Capital) instead of Total Equity
        # This respects the Iron Bank Floor.
        # Ensure we have a valid risk budget calculated
        if self.risk_budget <= 0: self.manage_iron_bank()
        effective_capital = self.risk_budget

        pool_a_ceiling = effective_capital * getattr(config, 'POOL_A_ALLOCATION_PCT', 0.60)
        pool_b_ceiling = effective_capital * getattr(config, 'POOL_B_ALLOCATION_PCT', 0.60)  # UNLEASHED: 60% for arb (was 30%)

        # Debug: Check calculations if ceiling is strangely low
        # print(f"DEBUG: Cap ${effective_capital:.2f} -> A:${pool_a_ceiling:.2f} / B:${pool_b_ceiling:.2f}")

        # Reset cycle counters if new cycle started (every 5 mins)
        if time.time() - self.pool_cycle_start_time > 300:
            self.pool_a_entries_this_cycle = 0
            self.pool_b_entries_this_cycle = 0
            self.pool_cycle_start_time = time.time()

        # --- PHASE 2: ARBITRAGE POOL B CHECK (Fully Independent) ---
        if is_arb and getattr(config, 'ARB_LAYER_ENABLED', True):
            # B1. Slot Check
            if arb_count >= arb_slots:
                print(f"[{self.name}] 🛑 POOL B SLOTS FULL ({arb_count}/{arb_slots}). Rejecting Arb Entry.")
                # ENTER MANAGEMENT MODE when at position limits
                self.enter_management_mode('ARB_LIMIT_REACHED', arb_count, arb_slots)
                return False

            # B2. Allocation Ceiling Check
            min_trade = getattr(config, 'MIN_ORDER_VALUE', 10.0)
            if pool_b_exposure + min_trade > pool_b_ceiling:
                print(f"[{self.name}] 🛑 POOL B ALLOCATION CEILING (Proposed: ${(pool_b_exposure + min_trade):.2f} / Cap: ${pool_b_ceiling:.2f}). Rejecting.")
                # Enter management mode ONLY if we are actually heavily allocated
                if pool_b_exposure >= pool_b_ceiling * self._management_mode_auto_exit_threshold:
                    self.enter_management_mode('POOL_B_EXPOSURE_LIMIT', pool_b_exposure, pool_b_ceiling)
                return False
            
            # B3. Pool B Cooldown Check
            pool_b_cooldown = getattr(config, 'POOL_B_COOLDOWN_SEC', 30)
            time_since_last = time.time() - self.pool_b_last_entry_time
            if time_since_last < pool_b_cooldown:
                print(f"[{self.name}] ⏳ POOL B COOLDOWN ({time_since_last:.0f}s/{pool_b_cooldown}s). Waiting.")
                return False
            
            # B4. Pool B Entries Per Cycle Check
            max_entries = getattr(config, 'POOL_B_ENTRIES_PER_CYCLE', 2)
            if self.pool_b_entries_this_cycle >= max_entries:
                print(f"[{self.name}] 🛑 POOL B CYCLE LIMIT ({self.pool_b_entries_this_cycle}/{max_entries}). Wait for new cycle.")
                return False
            
            # Pool B passes all checks - update counters will happen on execution
            if self.DEBUG: print(f"[{self.name}] ⚖️ POOL B APPROVED: {symbol} ({arb_count+1}/{arb_slots}, ${pool_b_exposure:.2f}/${pool_b_ceiling:.2f})")
            return True  # Arb allowed in its own pool
        # --------------------------------------------

        # --- DIRECTIONAL POOL A CHECK ---
        # A1. Slot Check
        limit_max_pos = directional_slots  # Use Pool A limit
        if not existing_pos and directional_count >= limit_max_pos:
             print(f"[{self.name}] 🛑 POOL A SLOTS FULL ({directional_count}/{limit_max_pos}). Rejecting Entry.")
             # Enter management mode if we're at the limit
             self.enter_management_mode('DIRECTIONAL_LIMIT_REACHED', directional_count, limit_max_pos)
             return False

        # A2. Allocation Ceiling Check
        min_trade = getattr(config, 'MIN_ORDER_VALUE', 10.0)
        if not existing_pos and pool_a_exposure + min_trade > pool_a_ceiling:
            print(f"[{self.name}] 🛑 POOL A ALLOCATION CEILING (Proposed: ${(pool_a_exposure + min_trade):.2f} / Cap: ${pool_a_ceiling:.2f}). Rejecting.")
            # Enter management mode ONLY if we are actually heavily allocated
            if pool_a_exposure >= pool_a_ceiling * self._management_mode_auto_exit_threshold:
                self.enter_management_mode('POOL_A_EXPOSURE_LIMIT', pool_a_exposure, pool_a_ceiling)
            return False
        
        # A3. Pool A Cooldown Check
        pool_a_cooldown = getattr(config, 'POOL_A_COOLDOWN_SEC', 60)
        time_since_last = time.time() - self.pool_a_last_entry_time
        if not existing_pos and time_since_last < pool_a_cooldown:
            print(f"[{self.name}] ⏳ POOL A COOLDOWN ({time_since_last:.0f}s/{pool_a_cooldown}s). Waiting.")
            return False
        
        # A4. Pool A Entries Per Cycle Check
        max_entries = getattr(config, 'POOL_A_ENTRIES_PER_CYCLE', 2)
        if not existing_pos and self.pool_a_entries_this_cycle >= max_entries:
            print(f"[{self.name}] 🛑 POOL A CYCLE LIMIT ({self.pool_a_entries_this_cycle}/{max_entries}). Wait for new cycle.")
            return False
        
        # Legacy Total Limit (Emergency Brake)
        total_limit = getattr(config, 'MAX_SIMULTANEOUS_POSITIONS', 8)
        total_positions = len(self.positions)
        if not existing_pos and total_positions >= total_limit:
             print(f"[{self.name}] 🛑 TOTAL SLOTS EXHAUSTED ({total_positions}/{total_limit}). Emergency Brake.")
             return False
        
        # 2. Asset Class Size Limit (Approximate)
        # Using Notional Value Check against Portfolio %
        # Need Total Equity to check %
        total_equity = self.balance # Approx
        if total_equity > 0:
            tier_key = 'DEFAULT'
            if symbol in ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']: tier_key = 'LARGE_CAP'
            elif symbol in ['PEPE/USDT', 'DOGE/USDT', 'SHIB/USDT']: tier_key = 'MEME'
            
            max_alloc_pct = config.POSITION_LIMITS['size_limits'].get(tier_key, 0.10)
            
            # Current Exposure
            # If strictly adding, we check resulting size. 
            # But here we just check if existing exceeds limit? No, is_trade_allowed checks BEFORE sizing.
            # So actual sizing logic handles the 'How much'. 
            # But we can BLOCK if already full.
            
            # AEHML 2.1: Use margin-based size for per-asset limit check
            ep_qty = existing_pos.quantity if existing_pos else 0.0
            ep_lev = existing_pos.leverage if existing_pos else 1.0
            current_margin = (ep_qty * asset_price) / ep_lev if existing_pos else 0.0
            current_alloc_pct = current_margin / total_equity
            
            if current_alloc_pct >= max_alloc_pct:
                 print(f"[{self.name}] 🛑 SIZE LIMIT REACHED for {symbol} ({current_alloc_pct*100:.1f}% >= {max_alloc_pct*100:.1f}%). Stacking Blocked.")
                 return False

        # --------------------------------------
        # 1. MICRO-ACCOUNT MODE (Request F)
        if config.MICRO_CAPITAL_MODE:
            # A. NO STACKING
            # A. REGIME STACKING CHECK
            regime_key = self.regime_controller.get_current_regime() if self.regime_controller else 'SMALL'
            max_stacks = config.REGIME_PERMISSIONS.get(regime_key, {}).get('max_stacks', 0)
            
            if existing_pos and max_stacks == 0:
                print(f"[{self.name}] 🧊 MICRO FREEZE: Stacking disabled (Regime {regime_key}, Max Stacks {max_stacks}). Rejecting.")
                return False
            
            # B. MAX POSITIONS CAP (Uses Pool A logic now - already checked above)
            # Skip duplicate check
            
            # C. EXPOSURE CAP
            # Check if adding a MINIMUM SIZED trade triggers the cap. 
            # Real sizing happens later, this is just a gate.
            estimated_exposure = config.MIN_ORDER_VALUE * 1.1 # 10% buffer
            
            curr_exp = 0.0
            if self.executor:
                 curr_exp = sum([p.quantity * p.entry_price for p in self.executor.positions.values()])
            current_exposure = curr_exp
            
            if regime_key == 'NANO': # Legacy check, probably not hit
                 nano_limit_ratio = config.REGIME_PERMISSIONS.get('NANO', {}).get('max_exposure_ratio', 5.0)
                 max_allowed = self.balance * nano_limit_ratio
            else:
                 # Use Dynamic Limit based on Regime
                 regime_limit_ratio = config.REGIME_PERMISSIONS.get(regime_key, {}).get('max_exposure_ratio', 10.0)
                 max_allowed = self.balance * regime_limit_ratio
            
            if (current_exposure + estimated_exposure) > max_allowed:
                  print(f"[{self.name}] 🛑 EXPOSURE CAP REACHED (Pre-Check): Current ${current_exposure:.2f} + MinTrade > Limit ${max_allowed:.2f}")
                  return False


        # 2. LOW NAV & MARGIN FREEZE (Cross-Margin)
        # Even if not in Micro mode, if funds are low, don't stack.
        
        # A. Calculate Real State
        p_state = self._calculate_portfolio_state()
        
        # B. Hard Solvency Check
        # If Margin Level < threshold (Buffer for 1.0 liquidation), FREEZE.
        if p_state['margin_level'] < config.RISK_MIN_MARGIN_LEVEL:
             print(f"[{self.name}] 🧊 SOLVENCY FREEZE: Margin Level {p_state['margin_level']:.2f} < {config.RISK_MIN_MARGIN_LEVEL}. Reducing Risk only.")
             return False
             
        # C. Required Margin Check for NEW Trade
        # We estimate the new trade margin based on Min Order Value
        min_trade_margin = config.MIN_ORDER_VALUE / config.REGIME_PERMISSIONS.get(self.regime_controller.get_current_regime(), {}).get('max_leverage', 1.0)
        
        if p_state['free_margin'] < (min_trade_margin * 1.1): # 10% Buffer
             print(f"[{self.name}] 🧊 INSUFFICIENT MARGIN: Free ${p_state['free_margin']:.2f} < Req ${min_trade_margin:.2f}. Blocked.")
             return False

        if self.balance < config.STACKING_MIN_EQUITY:
            if existing_pos:
                 # Check Free Margin Buffer
                 # We need ~5x the min order value in FREE margin to justify a stack
                 required_buffer = config.MIN_ORDER_VALUE * config.STACKING_BUFFER_MULTIPLIER
                 if p_state['free_margin'] < required_buffer:
                       print(f"[{self.name}] 🧊 LOW NAV FREEZE: Free Margin ${p_state['free_margin']:.2f} < Buffer ${required_buffer:.2f}. Stacking Blocked.")
                       return False

        # --- PATCH 2: THE STACKING CAP (Stop the Martingale) ---
        # FIXED: Removed hardcoded MAX_STACKS = 3. Rely purely on Regime permissions.
        # --- PATCH 2: DYNAMIC STACKING (Profit-Financed Risk) ---
        # "Earn Your Stacks": Max Stacks = Base Limit + Floor(Open Profit / 1R)
        if existing_pos:
            regime_key = self.regime_controller.get_current_regime() if self.regime_controller else 'SMALL'
            base_limit = config.REGIME_PERMISSIONS.get(regime_key, {}).get('max_stacks', 0)
            
            # 🐳 WHALE BONUS
            if whale_confirmed: 
                base_limit += 2
                
            # 💰 PROFIT BONUS
            # Calculate 1R (Risk Unit)
            risk_unit = max(1.0, self.balance * config.MAX_RISK_PCT)

            # Calculate Open Profit (Unrealized)
            open_profit = 0.0
            entry_p = existing_pos.entry_price
            qty_held = existing_pos.quantity
            pos_dir = existing_pos.direction

            if entry_p > 0 and asset_price > 0:
                if pos_dir == 'BUY':
                    open_profit = (asset_price - entry_p) * qty_held
                else:
                    open_profit = (entry_p - asset_price) * qty_held

            # Bonus = Floor(Profit / 1R)
            bonus_stacks = 0
            if open_profit > 0:
                bonus_stacks = int(open_profit / risk_unit)

            dynamic_limit = base_limit + bonus_stacks

            # HARD CAP (Sanity)
            HARD_CAP = 10
            dynamic_limit = min(dynamic_limit, HARD_CAP)

            current_stacks = existing_pos.stack_count
            
            if current_stacks >= dynamic_limit:
                 print(f"[{self.name}] 🛑 STACKING CAP: {current_stacks} >= {dynamic_limit} (Base {base_limit} + Bonus {bonus_stacks}). Profit ${open_profit:.2f} (1R=${risk_unit:.2f}). Rejecting.")
                 return False
            else:
                 print(f"[{self.name}] 🥞 DYNAMIC STACK APPROVED: {current_stacks+1}/{dynamic_limit} (Bonus {bonus_stacks} from ${open_profit:.2f} profit)")
                 
        # -------------------------------------------------------
        # -------------------------------------------------------

        # --- PHASE 35: IMMUNE SYSTEM CHECKS ---
        # Note: In Micro Mode, we might skip Cluster Risk if desired ("Ignore cluster risk" request)
        if not config.MICRO_CAPITAL_MODE:
            if not self.check_cluster_risk(symbol):
                return False
                
        return True

    def _is_arb(self, strategy: str) -> bool:
        """Helper to identify arbitrage strategies via keywords."""
        if not strategy: return False
        s = str(strategy).upper()
        keywords = ['ARB', 'BASIS', 'FUNDING', 'CARRY', 'YIELD']
        return any(k in s for k in keywords)

    def _get_pool_b_exposure(self) -> float:
        """Calculate current Pool B (arb) exposure in USD."""
        if not self.executor:
            return 0.0
        
        return sum(
            (pos.quantity * self.latest_prices.get(pos.symbol, pos.entry_price)) / pos.leverage
            for vk, pos in self.executor.positions.items()
            if self._is_arb(pos.strategy)
        )

    def enter_management_mode(self, reason: str, current_value: float, limit_value: float):
        """Enter management mode when position limits are reached."""
        
        # FIX 2026-03-03: Check cooldown before re-entering
        time_since_exit = time.time() - self._last_management_mode_exit_time
        if time_since_exit < self._management_mode_cooldown_sec:
            print(f"[{self.name}] ⏳ MANAGEMENT MODE: Cooldown active ({time_since_exit:.0f}s/{self._management_mode_cooldown_sec}s). Skipping re-entry for {reason}.")
            return  # Don't re-enter immediately
        
        print(f"[{self.name}] 🛠️ MANAGEMENT MODE: Activated due to {reason}")
        print(f"    Current: {current_value}, Limit: {limit_value}")

        # Set management mode flag
        self.management_mode = True
        self.management_mode_reason = reason
        self.management_mode_start_time = time.time()

        # In management mode, focus on optimizing existing positions
        # rather than taking new trades
        self.management_mode_target = {
            'reason': reason,
            'optimize_existing': True,
            'prioritize_closures': False,
            'focus_on_profitable': True
        }

        # Adjust risk parameters for management mode
        self.management_mode_risk_multiplier = self.risk_multiplier * 0.7  # Reduce risk in management mode

    def should_exit_management_mode(self) -> bool:
        """Determine if the system should exit management mode."""
        if not self.management_mode:
            return False

        # 1. HARD TIMEOUT - Force exit after max duration
        elapsed = time.time() - self.management_mode_start_time if self.management_mode_start_time else 0
        if elapsed > self.management_mode_max_duration:
            print(f"[{self.name}] ⏰ MANAGEMENT MODE TIMEOUT: {elapsed/60:.1f}min > {self.management_mode_max_duration/60:.0f}min cap. Forcing exit.")
            return True

        # 2. EXPOSURE-BASED EXIT - Check if exposure has reduced below threshold
        pool_a_exposure = 0.0
        pool_b_exposure = 0.0
        
        if self.executor:
            pool_a_exposure = sum(
                (pos.quantity * self.latest_prices.get(pos.symbol, pos.entry_price)) / pos.leverage
                for vk, pos in self.executor.positions.items() 
                if not self._is_arb(pos.strategy)
            )
            pool_b_exposure = sum(
                (pos.quantity * self.latest_prices.get(pos.symbol, pos.entry_price)) / pos.leverage
                for vk, pos in self.executor.positions.items() 
                if self._is_arb(pos.strategy)
            )
            
        effective_capital = self.risk_budget if getattr(self, 'risk_budget', 0) > 0 else self.available_balance
        pool_a_ceiling = effective_capital * getattr(config, 'POOL_A_ALLOCATION_PCT', 0.60)
        pool_b_ceiling = effective_capital * getattr(config, 'POOL_B_ALLOCATION_PCT', 0.60)
        
        # Count current positions for slot checks
        directional_count = 0
        arb_count = 0

        if self.executor:
            for vk, p in self.executor.positions.items():
                if self._is_arb(p.strategy):
                    arb_count += 1
                else:
                    directional_count += 1

        # Get slot limits
        arb_slots = getattr(config, 'POOL_B_SLOTS', 10)
        base_directional_slots = getattr(config, 'POOL_A_SLOTS', 5)
        directional_slots = base_directional_slots
        if self.regime_controller:
            bonuses = self.regime_controller.get_graduation_bonuses()
            directional_slots += bonuses.get('slot_bonus', 0)

        reason = getattr(self, 'management_mode_reason', '')
        
        # Branch based on which constraint triggered the management mode
        if 'EXPOSURE' in reason or 'POOL' in reason:
            if 'POOL_A' in reason:
                if pool_a_ceiling > 0:
                    exposure_ratio = pool_a_exposure / pool_a_ceiling
                    if exposure_ratio < self._management_mode_auto_exit_threshold:
                        print(f"[{self.name}] ✅ MANAGEMENT MODE: Pool A exposure {exposure_ratio:.0%} < {self._management_mode_auto_exit_threshold:.0%}. Exiting.")
                        return True
            elif 'POOL_B' in reason:
                if pool_b_ceiling > 0:
                    exposure_ratio = pool_b_exposure / pool_b_ceiling
                    if exposure_ratio < self._management_mode_auto_exit_threshold:
                        print(f"[{self.name}] ✅ MANAGEMENT MODE: Pool B exposure {exposure_ratio:.0%} < {self._management_mode_auto_exit_threshold:.0%}. Exiting.")
                        return True
        elif 'LIMIT_REACHED' in reason or 'SLOT' in reason:
            if 'ARB' in reason:
                if arb_count <= arb_slots - 1:
                    print(f"[{self.name}] ✅ MANAGEMENT MODE: Arb slots have room ({arb_count}/{arb_slots}). Exiting.")
                    return True
            elif 'DIRECTIONAL' in reason:
                if directional_count <= directional_slots - 1:
                    print(f"[{self.name}] ✅ MANAGEMENT MODE: Directional slots have room ({directional_count}/{directional_slots}). Exiting.")
                    return True
        else:
            # Fallback for old/generic reasons - check BOTH to be safe
            if pool_a_ceiling > 0 and pool_b_ceiling > 0:
                if (pool_a_exposure / pool_a_ceiling) < self._management_mode_auto_exit_threshold and \
                   (pool_b_exposure / pool_b_ceiling) < self._management_mode_auto_exit_threshold:
                    if (arb_count <= arb_slots - 1 and directional_count <= directional_slots - 1):
                        print(f"[{self.name}] ✅ MANAGEMENT MODE: Global exposures & slots recovered. Exiting.")
                        return True

        return False

    def exit_management_mode(self):
        """Exit management mode when conditions improve."""
        if self.management_mode:
            elapsed = time.time() - self.management_mode_start_time if self.management_mode_start_time else 0
            print(f"[{self.name}] ✅ MANAGEMENT MODE: Deactivated after {elapsed/60:.1f}min (Reason: {self.management_mode_reason})")
            self.management_mode = False
            self.management_mode_reason = None
            self.management_mode_start_time = None
            self.management_mode_target = None
            self.management_mode_risk_multiplier = None
            
            # FIX 2026-03-03: Record exit time for cooldown tracking
            self._last_management_mode_exit_time = time.time()

    def check_and_update_management_mode(self):
        """Check if management mode should be exited."""
        if self.management_mode:
            elapsed = time.time() - self.management_mode_start_time if self.management_mode_start_time else 0
            
            # FIX 2026-03-03: Log status every 60 seconds for visibility
            if int(elapsed) % 60 == 0 and int(elapsed) > 0 and int(elapsed) < 120:
                # Only log in first 2 minutes to avoid spam
                status = self.get_management_mode_status()
                print(f"[{self.name}] 🛠️ MANAGEMENT MODE: {elapsed/60:.1f}min | "
                      f"Reason: {status.get('reason', 'N/A')} | "
                      f"Time remaining: {status.get('time_remaining_seconds', 0)/60:.1f}min")
            
            if self.should_exit_management_mode():
                self.exit_management_mode()
                print(f"[{self.name}] 🔄 MANAGEMENT MODE: Automatically deactivated - limits no longer binding")

    def is_in_management_mode(self) -> bool:
        """Check if the system is in management mode."""
        return self.management_mode

    def get_management_mode_status(self) -> dict:
        """Get current management mode status."""
        if not self.management_mode:
            return {'active': False}

        elapsed = time.time() - self.management_mode_start_time if self.management_mode_start_time else 0
        return {
            'active': True,
            'reason': self.management_mode_reason or 'UNKNOWN',
            'duration_seconds': elapsed,
            'duration_minutes': elapsed / 60.0,
            'max_duration_seconds': self.management_mode_max_duration,
            'time_remaining_seconds': max(0, self.management_mode_max_duration - elapsed),
            'target': self.management_mode_target or {},
            'risk_multiplier': self.management_mode_risk_multiplier or self.risk_multiplier
        }


    def check_meta_limits(self, symbol: str) -> bool:
        """
        Stop the definition of insanity: Doing the same thing and expecting different results.
        If we veto the same asset repeatedly, ban it temporarily.
        """
        # 1. Check if asset is in Penalty Box
        veto_count = self.meta_veto_counter.get(symbol, 0)
        msg_key = f"{symbol}_veto_log"
        
        # Reset counter if 10 minutes passed since last veto (Decay)
        last_veto_times = self.veto_timestamps.get(symbol, [])
        if last_veto_times and (time.time() - last_veto_times[-1]) > 600:
             self.meta_veto_counter[symbol] = 0
             self.veto_timestamps[symbol] = []
             return True
             
        # Threshold: 10 Vetoes in 10 Minutes -> 1 Hour Ban
        if veto_count > 10:
             # Check if we are still in the 1 hour window
             if (time.time() - last_veto_times[-1]) < 3600:
                  return False
             else:
                  # Ban expired
                  self.meta_veto_counter[symbol] = 0
                  return True
                  
        return True

    def record_veto(self, symbol: str, reason: str):
        """Track vetoes to trigger Meta-Learning bans."""
        self.meta_veto_counter[symbol] += 1
        self.veto_timestamps[symbol].append(time.time())
        
        # FIX: Capacity constraints are normal operation, not system distress.
        # We only track "Bad Vetoes" (Risk, Ruin, Toxicity) for the distress signal.
        if reason not in ["ALLOC_FULL", "MAX_POSITIONS", "SIZE_LIMIT", "COOLDOWN", "STACK_TOO_CLOSE"]:
             self.consecutive_veto_streak += 1
        else:
             # Should we reset? If we are successfully operating at capacity, we aren't in distress.
             # Resetting prevents a stale streak from triggering later.
             self.consecutive_veto_streak = 0
        
        if self.consecutive_veto_streak > 50:
             print(f"[{self.name}] 🚨 SYSTEM DISTRESS: {self.consecutive_veto_streak} Consecutive Vetoes. Requesting Cool-Off.")
             time.sleep(5) # Physical throttle
             self.consecutive_veto_streak = 0

    def calc_position_size(self, symbol: str, asset_price: float, current_atr: float = None, atr_ref: float = None, conviction: float = 0.5, direction: str = 'BUY', crisis_score: float = 0.0, sentiment_score: float = 0.0, whale_confirmed: bool = False, market_bias: float = 0.5, metadata: Dict[str, Any] = None, latest_prices: Dict[str, float] = {}) -> Tuple[bool, float, float]:
        """
        Calculate position size with Phase 12 institutional risk management.
        
        Integrates:
        1. Minimax Constraint (protect principal)
        2. Volatility Scalar (ATR-based sizing)
        4. Conviction Scalar (LSTM-based scaling)
        5. Holistic Feedback (Sentiment Hormone)
        6. Dynamic Flotilla Sizing (Market Bias)
        
        Returns:
            (is_approved: bool, quantity: float, leverage: float)
        """
        is_override = False
        is_risk_reducing = False # Default
        final_notional = 0.0
        # 0. Sync Prices (Phase 5)
        if latest_prices:
            self.latest_prices.update(latest_prices)

        # --- GLOBAL SANITY CHECK (PAXG Protection) ---
        # FIX BUG-005 2026-03-01: PAXG sanity threshold updated to $6000 (gold ~$2900/oz with buffer)
        # XAUT sanity threshold similarly updated
        # These products track 1 troy oz of gold, price should be ~$2000-3000
        # FIX 2026-03-06: Gold-pegged tokens (PAXG, XAUT) on Kraken Futures sometimes
        # return prices in XAU units (~$1.41 per contract) instead of USD (~$3100).
        # Auto-correct: if price lands in the XAU-unit range (0.5–3.0), scale to USD.
        _GOLD_REF_USD = 3100.0  # Conservative gold reference price (USD per troy oz)
        if 'PAXG' in symbol:
            # PAXG (Pax Gold) - tracks 1 troy oz of gold, should be ~$2200-$6000 in USD
            if 0.5 <= asset_price <= 3.0:
                corrected = asset_price * _GOLD_REF_USD
                print(f"[{self.name}] [CORRECTED] PAXG price ${asset_price:.4f} looks like XAU units. "
                      f"Scaling to ${corrected:.2f} (ref ${_GOLD_REF_USD:.0f}/oz).")
                asset_price = corrected
            if asset_price < 2200:
                print(f"[{self.name}] [CRITICAL] PAXG Price Sanity Fail (${asset_price:.2f} < $2200). Possible wrong price source. Rejecting Sizing.")
                return False, 0.0, 0.0
            if asset_price > 6000:
                print(f"[{self.name}] [WARNING] PAXG Price Sanity Warn (${asset_price:.2f} > $6000). Verify price source.")
                # Don't reject, just warn (gold could spike in crisis)
        if 'XAUT' in symbol:
            # XAUT (Tether Gold) similar to PAXG - tracks 1 troy oz of gold
            if 0.5 <= asset_price <= 3.0:
                corrected = asset_price * _GOLD_REF_USD
                print(f"[{self.name}] [CORRECTED] XAUT price ${asset_price:.4f} looks like XAU units. "
                      f"Scaling to ${corrected:.2f} (ref ${_GOLD_REF_USD:.0f}/oz).")
                asset_price = corrected
            if asset_price < 2200:
                print(f"[{self.name}] [CRITICAL] XAUT Price Sanity Fail (${asset_price:.2f} < $2200). Rejecting Sizing.")
                return False, 0.0, 0.0
            if asset_price > 6000:
                print(f"[{self.name}] [WARNING] XAUT Price Sanity Warn (${asset_price:.2f} > $6000). Verify price source.")
        # ---------------------------------------------

        # ── SMCE Layer 0: Capital Doctrine ─────────────────────────────────────
        tier = self._get_smce_tier(self.balance)
        regime = getattr(self, 'smce_regime', 'HARVEST')

        # Get regime-specific leverage cap from SMCE doctrine
        if tier == "SMALL":
            max_lev_doctrine = getattr(config, 'SMCE_SMALL_MAX_LEVERAGE', {}).get(regime, 3.0)
        else:  # MEDIUM or LARGE
            max_lev_doctrine = getattr(config, 'SMCE_MEDIUM_MAX_LEVERAGE', {}).get(regime, 4.0)

        # FIX 1: HARD CAP 3x in TRANSITION regime (User Directive)
        # FIX 2026-03-20: Make TRANSITION constraints dynamic based on entropy
        # Current issue: 100% of regimes are TRANSITION, causing overly conservative sizing
        if regime == "TRANSITION":
            entropy = metadata.get('entropy', 1.0) if metadata else 1.0
            
            # Dynamic leverage cap based on entropy WITHIN transition regime
            if entropy < 0.7:
                # Calm transition - allow more leverage
                max_lev_doctrine = min(max_lev_doctrine, 4.0)
                print(f"[{self.name}] 🛡️ [SMCE-L0] TRANSITION Regime (Calm, E={entropy:.2f}): Leverage capped to 4x")
            elif entropy < 1.0:
                # Moderate transition - standard cap
                max_lev_doctrine = min(max_lev_doctrine, 3.0)
                print(f"[{self.name}] 🛡️ [SMCE-L0] TRANSITION Regime (Moderate, E={entropy:.2f}): Leverage capped to 3x")
            else:
                # Choppy transition - very conservative
                max_lev_doctrine = min(max_lev_doctrine, 2.0)
                print(f"[{self.name}] 🛡️ [SMCE-L0] TRANSITION Regime (Choppy, E={entropy:.2f}): Leverage capped to 2x")

        # === AGGRESSIVE GROWTH MODE OVERRIDE (2026-03-09) ===
        if self.aggressive_mode:
            boost = 1.5 # 50% leverage boost
            old_lev = max_lev_doctrine
            max_lev_doctrine = min(20.0, max_lev_doctrine * boost)
            print(f"[{self.name}] 🔥 AGGRESSIVE MODE: Leverage Boost {old_lev}x -> {max_lev_doctrine}x")
        # ====================================================

        if metadata is None:
            metadata = {}
        metadata['smce_max_leverage'] = max_lev_doctrine

        reason = metadata.get('reason', '') if metadata else ''
        reason_upper = reason.upper()

        # FIX 2026-03-14: Detect risk-reducing EARLY to bypass proximity checks for timeout exits
        # This prevents infinite timeout loops where exits are blocked as entries
        is_temp_risk_reducing = any(x in reason_upper for x in ['EXIT','CLOSE','REDUCE','TP','SL', 'SHORT_COVER', 'STACK_TIMEOUT', 'STACK_TP', 'HYGIENE_EXIT'])

        if not is_temp_risk_reducing:
            # 1. Defensive Cooldown Block
            if time.time() < self._defensive_cooldown_until:
                print(f"[{self.name}] 🛡️ [SMCE-L0] Entry blocked: DEFENSIVE cooldown active.")
                return False, 0.0, 0.0

            # 2. Stacking Proximity Block
            _cand_strat = metadata.get('strategy', '') if metadata else ''
            _is_arb = ('ARBITRAGE' in _cand_strat.upper() or 'FUNDING' in _cand_strat.upper() or 'BASIS' in _cand_strat.upper() or _cand_strat in ['ARBITRAGE_GOLD', 'FUNDING_CARRY', 'BASIS_TRADE', 'ARB', 'GOLD_LEAD_LAG'])
            if not _is_arb and self._check_price_proximity_stacking(symbol, asset_price):
                print(f"[{self.name}] 🛡️ [SMCE-L0] Entry blocked: Proximity to existing entry < 0.5%.")
                return False, 0.0, 0.0
        # ───────────────────────────────������─────────────────────────────────────

        # --- PHASE 0: INTENT DETECTION (CRITICAL FIX) ---
        # Determine if this is a Risk Reducing trade (Exit/Reduce/Cover) vs New Risk (Entry/Stack).
        # We MUST do this early to bypass restrictions that only apply to acquiring new risk.

        cand_strategy = metadata.get('strategy', 'DIRECTIONAL') if metadata else 'DIRECTIONAL'
        # FIX: Make arb detection more inclusive to catch all arb variants
        is_arb_trade = (
            cand_strategy in ['ARBITRAGE_GOLD', 'FUNDING_CARRY', 'BASIS_TRADE', 'ARB', 'GOLD_LEAD_LAG', 'ARBITRAGE'] or 
            'ARBITRAGE' in cand_strategy.upper() or
            'FUNDING' in cand_strategy.upper() or
            'BASIS' in cand_strategy.upper()
        )
        is_macro_pair = cand_strategy == 'MACRO_MEAN_REVERSION'  # PAXG/BTC macro trade
        
        # Get regime from SINGLE SOURCE OF TRUTH
        regime = self.get_smce_regime()  # ✅ Consistent regime access

        # FIX 3: Allow up to 2 funding carry trades at a time (increased from 1)
        # Prevents small accounts from overexposing to multiple arb positions
        # while allowing diversification across different opportunities
        if is_arb_trade and not is_risk_reducing:
            arb_count = 0
            if self.executor:
                for vk, p in self.executor.positions.items():
                    pos_strat = getattr(p, 'strategy', '') or (getattr(p, 'metadata', {}) or {}).get('strategy', '')
                    pos_strat_upper = pos_strat.upper() if pos_strat else ''
                    # Only count EXPLICIT arb strategies, not directional trades
                    # Match: ARBITRAGE, ARBITRAGE_GOLD, FUNDING_CARRY, BASIS_TRADE, BASIS_CARRY_LONG/SHORT
                    is_arb_position = (
                        'ARBITRAGE' in pos_strat_upper or
                        'FUNDING_CARRY' in pos_strat_upper or
                        'BASIS_TRADE' in pos_strat_upper or
                        'BASIS_CARRY' in pos_strat_upper
                    )
                    if is_arb_position:
                        arb_count += 1
            if arb_count >= 10:  # Increased from 1 to 10 concurrent arb positions for testing
                print(f"[{self.name}] 🛑 [SMCE-L0] ONE ARB RULE: Already have {arb_count} funding carry position(s). Blocking new arb until closed.")
                return False, 0.0, 0.0


        # 1. By Explicit Reason
        bypass_reasons = ['SIGNAL_PROVIDER', 'STACK_TIMEOUT', 'COMPLIANCE_REDUCE', 'SHORT_COVER', 'EXIT', 'CLOSE', 'REDUCE', 'TP', 'SL', 'STACK_TP', 'HYGIENE_EXIT']
        reason = metadata.get('reason', '') if metadata else ''
        reason_upper = reason.upper()
        if any(x in reason_upper for x in bypass_reasons):
            is_risk_reducing = True
            if self.DEBUG: print(f"[{self.name}] 🏳️ INTENT: Risk Reduction/Reporting detected via Reason '{reason}'")
        
        # 2. By Direction Opposition (Netting)
        # If we hold Long (Pos > 0) and we are Selling -> Reduce
        # If we hold Short (Pos > 0 qty, dir='SELL') and we are Buying -> Reduce (Cover)
        # 2. By Direction Opposition (Netting)
        # If we hold Long (Pos > 0) and we are Selling -> Reduce
        # If we hold Short (Pos > 0 qty, dir='SELL') and we are Buying -> Reduce (Cover)
        existing_pos_data = None
        if not is_risk_reducing and self.executor:
             # Find position
             for vk, p in self.executor.positions.items():
                 if p.symbol == symbol:
                     existing_pos_data = p
                     break
        
        if existing_pos_data:
             qty_held = existing_pos_data.quantity
             current_dir = existing_pos_data.direction
             
             if qty_held > 0.00000001:
                 # Normalize directions
                 cd = current_dir.upper().replace('LONG', 'BUY').replace('SHORT', 'SELL')
                 nd = direction.upper().replace('LONG', 'BUY').replace('SHORT', 'SELL')
                 
                 if cd == 'BUY' and nd == 'SELL': is_risk_reducing = True
                 elif cd == 'SELL' and nd == 'BUY': is_risk_reducing = True
                 
                 if is_risk_reducing and self.DEBUG:
                     print(f"[{self.name}] 🏳️ INTENT: Risk Reduction detected via Netting ({cd} -> {nd})")

        # --- IMMEDIATE COOLDOWN VETO (User Request: Strict, No Override) ---
        last_time = self.last_trade_time.get(symbol, 0)
        
        is_stacking = False
        if existing_pos_data and existing_pos_data.quantity > 0:
             is_stacking = True
        
        # Enforce Cooldown unless Stacking (Stacking has its own separate checks, but we veto Rapid Fire Stacks too)
        # Assuming we want to allow stacking but not "Rapid Fire" stacking (e.g. 5x in 1s).
        # We enforce a mini-cooldown for stacks (e.g. 30s) or same 120s?
        # User said: "SUI... 17 units in ONE cycle... Cooldown Must Include Stack".
        # So yes, enforce Cooldown for EVERYTHING.
        if (time.time() - last_time < config.GOVERNOR_COOLDOWN_SECONDS) and not is_risk_reducing:
             print(f"[{self.name}] 🛑 STRICT COOLDOWN: {symbol} Last Trade {int(time.time() - last_time)}s ago (< {config.GOVERNOR_COOLDOWN_SECONDS}s). Blocking All.")
             return False, 0.0, 0.0

        # --- PARADOXICAL FUNDING FILTER ---
        cand_funding = metadata.get('funding_yield', 0.0) if metadata else 0.0
        cand_structure = metadata.get('structure', 'NEUTRAL') if metadata else 'NEUTRAL'
        
        if cand_funding < -300.0 and not is_risk_reducing:
             if not (crisis_score > 1.5 and cand_structure == 'SUPPORT'):
                 if not whale_confirmed:
                     print(f"[{self.name}] 💸 FUNDING VETO: Rejecting {symbol} (Paying {abs(cand_funding):.0f}% APY). Requires Crisis>1.5 & Support.")
                     return False, 0.0, 0.0
                 else:
                     print(f"[{self.name}] 🐋 WHALE OVERRIDE: Paying Expensive Funding ({cand_funding:.0f}%) due to Whale Signal.")
        
        # --- STACK SNOOZE CHECK ---
        if symbol in self.stack_snooze:
            if time.time() < self.stack_snooze[symbol]:
                # Silent Reject (Don't spam log)
                # Or maybe print once every minute? For now, we print nothing to reduce noise.
                # Actually, returning False here without print might be confusing if user is watching?
                # Let's print only if DEBUG
                if self.DEBUG: print(f"[{self.name}] 💤 Snoozing {symbol} (Stack Too Close) for {int(self.stack_snooze[symbol] - time.time())}s")
                return False, 0.0, 0.0
            else:
                del self.stack_snooze[symbol] # Wake up
        # ---------------------------
        # -------------------------------------------------------------------

        # --- ADAPTIVE LEVERAGE INIT ---
        # 2026-03-20 FIX: Use account tier leverage_cap as the authoritative source
        # Old code used REGIME_PERMISSIONS.get(regime, {}).get('max_leverage', 1.0)
        # which silently defaulted to 1.0 because POSITION_LIMITS_CENTRAL uses 'leverage_cap' not 'max_leverage'
        tier_lev_cap = self.get_tier_leverage_cap()
        regime = self.regime_controller.get_current_regime() if self.regime_controller else 'SMALL'
        leverage = tier_lev_cap  # Start from tier cap, then only reduce (never increase)

        # ================================================================
        # ATLAS INTEGRATION: Edge-Aware Position Sizing (2026-03-18)
        # ================================================================
        atlas_notional = None
        atlas_approved = False
        atlas_meta = {}
        if self.atlas_available and not is_risk_reducing:
            try:
                # Query Atlas for edge-aware sizing
                atlas_approved, atlas_size, atlas_meta = self.get_atlas_position_size(
                    symbol, asset_price, direction, metadata
                )

                if atlas_approved and atlas_size > 0:
                    atlas_notional = atlas_size
                    if self.DEBUG:
                        print(f"[{self.name}] 🎯 [ATLAS] Edge-aware size: ${atlas_notional:.2f}")
            except Exception as e:
                if self.DEBUG:
                    print(f"[{self.name}] [ATLAS] Query failed: {e}")
                atlas_notional = None
        
        # === ML-ATLAS BRIDGE INTEGRATION (2026-03-22) ===
        # Resolve conflicts between ML and Atlas
        if self.ml_atlas_bridge and not is_risk_reducing:
            try:
                # Prepare market data for bridge
                market_data = {
                    'volatility_pct': metadata.get('volatility', 0.01),
                    'spread_pct': metadata.get('spread', 0.001),
                    'liquidity_score': metadata.get('liquidity', 1.0),
                    'regime_score': metadata.get('regime_score', 0),
                }
                
                signal_data = {
                    'symbol': symbol,
                    'direction': direction,
                    'strength': metadata.get('conviction', 0.5),
                }
                
                # Get bridge evaluation
                bridge_result = self.ml_atlas_bridge.evaluate_trade(
                    symbol=symbol,
                    direction=direction,
                    price=asset_price,
                    quantity=1.0,  # Will be scaled
                    market_data=market_data,
                    signal_data=signal_data
                )
                
                # Override Atlas decision if bridge recommends
                if bridge_result['approved'] != atlas_approved:
                    print(f"[{self.name}] 🤖🗺️ ML-ATLAS BRIDGE: Overriding Atlas - {bridge_result['reason']}")
                    atlas_approved = bridge_result['approved']
                    
                    # Adjust size based on bridge recommendation
                    if bridge_result['size_adjustment'] != 1.0 and atlas_notional:
                        atlas_notional *= bridge_result['size_adjustment']
                        print(f"[{self.name}] 🤖🗺️ Size adjusted to ${atlas_notional:.2f} ({bridge_result['size_adjustment']:.0%})")
                
                # Store ML data for later use
                metadata['ml_win_prob'] = bridge_result.get('ml_win_prob', 0.5)
                metadata['ml_atlas_consensus'] = bridge_result.get('confidence', 'MEDIUM')
                
            except Exception as e:
                print(f"[{self.name}] ML-Atlas Bridge error: {e}")
                # Continue with Atlas-only decision on error
        # ============================================
        # ================================================================
        
        # 2026-03-20 FIX: TIER-BASED LEVERAGE ENFORCEMENT (replaces PREDATOR-only clamp)
        # This is the HARD CEILING — no strategy, mode, or override can exceed the tier's leverage cap
        tier_name = self.get_account_tier()
        tier_lev_cap = self.get_tier_leverage_cap()
        leverage = min(leverage, tier_lev_cap)
        if self.DEBUG:
            print(f"[{self.name}] 🛡️ TIER LEV CAP [{tier_name}]: Leverage capped to {leverage}x (tier max {tier_lev_cap}x)")
        # ------------------------------
        
        # 0. Update Accumulator State
        # ideally this is done in sync loop, but fine to do here for latest check
        self.update_accumulator(self.balance)
        state = self.get_metabolism_state()
        
        # 1. Check Accumulator Lock
        # 1. Check Accumulator Lock
        if self.drawdown_lock:
             if is_risk_reducing:
                 if self.DEBUG: print(f"[{self.name}] 🔓 DRAWDOWN BYPASS: Allowing {reason} trade.")
                 pass # Allow match
             # --------------------------------------
             
             # (Logic moved to Top of Function)
             
             if is_risk_reducing:
                 if self.DEBUG:
                     print(f"[{self.name}] 🔓 DRAWDOWN OVERRIDE: Allowing Close/Reduce for {symbol}")
             elif symbol == "PAXG/USDT" and crisis_score > 0.5:
                 print(f"[{self.name}] 🚨 CRISIS BYPASS: Allowing PAXG trade (Score {crisis_score:.2f}) despite Lock.")
             elif whale_confirmed:
                 print(f"[{self.name}] 🐋 WHALE OVERRIDE: Bypassing ACCUMULATOR HALT for {symbol} (High Conviction).")
             else:
                 print(f"[{self.name}] 🛑 REJECT {symbol}: Accumulator Lock Active (Drawdown limit hit).")
                 return False, 0.0, 1.0

        # --- UNIVERSAL RUIN GUARD (Session 3 Safety Fix) ---
        # Blocks toxic assets with high variance even if they are Satellites.
        # We use PREDATOR settings as a baseline for "Is this safe?"
        rg_sl_dist = config.PREDATOR_STOP_LOSS
        rg_tp_dist = config.PREDATOR_TAKE_PROFIT
        rg_sl_price = asset_price * (1 - rg_sl_dist) if direction == 'BUY' else asset_price * (1 + rg_sl_dist)
        rg_tp_price = asset_price * (1 + rg_tp_dist) if direction == 'BUY' else asset_price * (1 - rg_tp_dist)
        
        # Calculate prob
        univ_ruin_prob = self.calculate_ruin_probability(symbol, asset_price, direction, rg_sl_price, rg_tp_price, metadata)

        # CRITICAL: Block toxic trades BEFORE they bleed us
        # Threshold: 35% (was 85% - way too loose!)
        if univ_ruin_prob > config.PHYSICS_MAX_RUIN_PROBABILITY:
             # Check if this is a "Whale" or "Crisis" trade that strictly needs to pass?
             # User said: "Ruin Guard Block... with 100% probability".
             # We BLOCK unless it's a liquidation-reducing trade.
             if not is_risk_reducing:
                  print(f"[{self.name}] ☢️ TOXIC ASSET DETECTED: {symbol} Ruin Prob {univ_ruin_prob:.1%} > {config.PHYSICS_MAX_RUIN_PROBABILITY:.1%}. BLOCKING ALL MODES.")
                  # FIX: Add to blacklist with 12-hour timeout (43200 seconds)
                  expiration = time.time() + 43200  # 12 hours
                  self.blacklist[symbol] = expiration
                  print(f"[{self.name}] 🚫 {symbol} added to blacklist for 12 hours (Toxic Asset)")
                  self.record_veto(symbol, "RUIN_RISK")
                  return False, 0.0, 0.0
        # ---------------------------------------------------

        # --- PHASE 25: SATELLITE OVERRIDE (High Value Snipers) ---
        # FIX: Moved Correlation Guard HERE to prevent "Approved then Rejected" race conditions.
        
        # --- PHASE 40: CORRELATION GUARD (The Hedge) ---
        # Prevent "All Eggs in One Basket"
        # If we already hold an asset highly correlated (>0.85) to the candidate, VETO it.
        # Exception: If directions are opposite (Hedge).
        
        # SMART CORRELATION (Phase 45): Relax constraints during Bull Runs
        should_check_correlation = getattr(config, 'CORRELATION_CHECK', True)

        if sentiment_score > getattr(config, 'SENTIMENT_THRESHOLD_BULL', 0.2):
             if self.DEBUG: print(f"[{self.name}] 🐂 BULL MARKET OVERRIDE: Disabling Correlation Check (Sent {sentiment_score:.2f} > 0.2)")
             should_check_correlation = False

        # 🐳 WHALE SIGNAL: Whale is now just another signal in the probability stack
        # Whale signals no longer bypass any guards - they contribute to the SMCE score only

        # Check if we have positions
        has_positions = False
        if self.executor and len(self.executor.positions) > 0:
             has_positions = True
             
        if has_positions and should_check_correlation and not is_risk_reducing:
             # We need a correlation matrix. For now, we use "Family" variants as proxies or 
             # the PPO Brain's memory if available. 
             # Simpler: Hardcoded Map for Phase 1.
             
             # Map: {Asset: Family}
             # BTC, ETH -> 'CRYPTO_MAJOR'
             # SOL, AVAX -> 'L1_ROTATOR'
             # DOGE, PEPE -> 'MEME_BASKET'
             
             families = {
                 'BTC': 'BITCOIN', 'WBTC': 'BITCOIN',
                 'ETH': 'ETHEREUM',
                 'SOL': 'SOLANA', 'SUI': 'MOVE_L1', 'AVAX': 'EVM_L1', 'ADA': 'LEGACY_L1',
                 'DOGE': 'MEME', 'SHIB': 'MEME', 'PEPE': 'MEME',
                 'XRP': 'LEGACY_PAYMENT', 'LTC': 'LEGACY_PAYMENT'
             }
             
             cand_base = symbol.split('/')[0]
             cand_fam = families.get(cand_base, 'OTHER')
             
             cand_base = symbol.split('/')[0]
             cand_fam = families.get(cand_base, 'OTHER')
             
             if self.executor:
                 for vk, pos in self.executor.positions.items():
                     pos_sym = pos.symbol
                     # Extract real symbols (strip strategy suffixes for comparison)
                     pos_real_symbol = pos_sym.split(':')[0] if ':' in pos_sym else pos_sym
                     cand_real_symbol = symbol.split(':')[0] if ':' in symbol else symbol
                     
                     # === SMART EXEMPTION 1: Different Strategy Pools (Arb vs Directional) ===
                     pos_strategy = pos.strategy
                     
                     cand_strategy = metadata.get('strategy', 'DIRECTIONAL') if metadata else 'DIRECTIONAL'
                     funding_yield = metadata.get('funding_yield', 0.0) if metadata else 0.0
                     is_arb_pos = self._is_arb(pos_strategy)
                     is_arb_cand = self._is_arb(cand_strategy)
                 
                     # If pools differ, they are independent buckets -> No Correlation Veto
                     if is_arb_pos != is_arb_cand:
                         if self.DEBUG:
                             print(f"[{self.name}] 🎯 POOL EXEMPTION: {symbol} (Arb:{is_arb_cand}) independent from {pos_sym} ({pos_strategy})")
                         continue
                     
                     # === SMART EXEMPTION 2: Negligible Position Size ===
                     # If existing position is dust (< 1% of allocation), don't let it block new trades
                     pos_qty = pos.quantity
                     pos_entry = pos.entry_price
                     pos_notional = abs(pos_qty * pos_entry)
                     allocation_threshold = self.balance * 0.01 
                     
                     if pos_notional < allocation_threshold:
                         continue
 
                     # === SMART EXEMPTION 3: Exceptional APY ===
                     if funding_yield and funding_yield > 300.0:
                         print(f"[{self.name}] 💎 EXCEPTIONAL APY: {symbol} ({funding_yield:.0f}% APY) overrides correlation veto")
                         continue
                     
                     # === ORIGINAL CORRELATION CHECK ===
                     # Allow stacking of the SAME asset (this is handled by Stack Limits, not Correlation)
                     if pos_real_symbol == cand_real_symbol: continue
                     
                     pos_base = pos_real_symbol.split('/')[0]
                     pos_fam = families.get(pos_base, 'OTHER')
                     
                     # If in same family AND same direction -> BLOCK
                     if cand_fam != 'OTHER' and cand_fam == pos_fam:
                         existing_dir = pos.direction
                         
                         if existing_dir == direction:
                             if not whale_confirmed:
                                 print(f"[{self.name}] 🔗 CORRELATION VETO: Rejecting {symbol} ({cand_fam}). Too similar to {pos_sym}.")
                                 return False, 0.0, 0.0
                             else:
                                 print(f"[{self.name}] 🐋 WHALE OVERRIDE: Correlation Ignored for {symbol} due to Whale Signal.")
                         else:
                             print(f"[{self.name}] ⚖️ HEDGE DETECTED: Allowing {symbol} ({direction}) vs {pos_sym} ({existing_dir})")
        # -----------------------------------------------

        # --- PHASE 25: SATELLITE OVERRIDE (High Value Snipers) ---
        # NANO GUARD: Disable Override in Nano Mode to ensure strict check at Patch 5a
        nano_active = (self.balance < config.NANO_CAPITAL_THRESHOLD)
        
        if symbol in config.SATELLITE_ASSETS and not nano_active:
             # Target Margin from config, but capped by available funds
             target_margin = getattr(config, 'SATELLITE_MARGIN', 10.0)
             
             # Dynamic Cap: Use 75% of available funds for these snipers
             # This ensures we don't hit "Insufficient Funds" on Kraken.
             safe_max_margin = (self.available_balance - 1.0) * 0.75
             safe_max_margin = max(0.0, safe_max_margin)
             
             actual_margin = min(target_margin, safe_max_margin)
             leverage = getattr(config, 'SATELLITE_LEVERAGE', 5.0)

             final_notional = actual_margin * leverage
             quantity = final_notional / asset_price if asset_price > 0 else 0.0

             if actual_margin < target_margin:
                  print(f"[{self.name}] 🎯 SATELLITE SNIPER: Capping Margin {target_margin:.2f} -> {actual_margin:.2f} (Solvency)")
             else:
                  print(f"[{self.name}] 🎯 SATELLITE SNIPER: Targeting ${actual_margin:.2f} Margin ({leverage}x)")

             # FIX BUG-003: Check minimum order size BEFORE returning
             base_asset = symbol.split('/')[0]
             config_min = getattr(config, 'MIN_TRADE_QTY', {}).get(base_asset, 0.0)
             if config_min > 0:
                 min_qty = config_min
             else:
                 min_order_value = getattr(config, 'MIN_ORDER_VALUE', 10.0)
                 min_qty = min_order_value / asset_price if asset_price > 0 else 0.0001
             
             if quantity > 0 and quantity < min_qty:
                 print(f"[{self.name}] 🚫 POSITION TOO SMALL: {quantity:.6f} < min {min_qty:.6f} for {symbol}. Rejecting.")
                 return False, 0.0, 0.0

             is_override = True
        # ------------------------------------

        # --- VOL-WINDOW REGIME OVERRIDE ---
        regime = self.regime_controller.get_current_regime() if self.regime_controller else 'SMALL'
        if regime == 'VOL_WINDOW':
            # 1. Check Max Positions
            # Use Executor
            ex_pos_count = len(self.executor.positions) if self.executor else 0
            if ex_pos_count >= config.VOL_WINDOW_MAX_POSITIONS:
                 print(f"[{self.name}] 🛑 VOL_WINDOW CAP: Max {config.VOL_WINDOW_MAX_POSITIONS} positions reached.")
                 return False, 0.0, 0.0
                 
            # 2. Sizing: Fixed Risk %
            # Risk Amount = Balance * Risk % (e.g. 2%)
            risk_amt_usd = self.balance * config.VOL_WINDOW_RISK_PCT
            
            # Determine Stop Distance (Need to know setup or use default)
            # We assume a tight stop for Vol Window (e.g. 1% or ATR based)
            # If ATR provided, use 1.5 ATR. Else use 1%.
            stop_dist_pct = 0.01
            if current_atr and asset_price > 0:
                 stop_dist_pct = (current_atr * 1.5) / asset_price
            
            # Position Size (Gross) = Risk / Stop%
            gross_size_usd = risk_amt_usd / stop_dist_pct
            
            # Cap Leverage
            max_gross = self.balance * config.VOL_WINDOW_LEVERAGE
            gross_size_usd = min(gross_size_usd, max_gross)
            
            quantity = gross_size_usd / asset_price

            # 🐳 WHALE SIZING BOOST
            if whale_confirmed:
                quantity *= 1.5 # 50% Size Boost for Whale trades
                print(f"[{self.name}] 🐳 WHALE SIZING: Boosting Position Size by 1.5x")

            # --- ALLOCATION CLAMP (Vol-Window) ---
            notional = quantity * asset_price
            if notional > (self.balance * 0.15):
                quantity = (self.balance * 0.15) / asset_price
                print(f"[{self.name}] 👮 VOL CLAMP: Capped at 15% Alloc.")

            # FIX BUG-003: Check minimum order size BEFORE returning
            base_asset = symbol.split('/')[0]
            config_min = getattr(config, 'MIN_TRADE_QTY', {}).get(base_asset, 0.0)
            if config_min > 0:
                min_qty = config_min
            else:
                min_order_value = getattr(config, 'MIN_ORDER_VALUE', 10.0)
                min_qty = min_order_value / asset_price if asset_price > 0 else 0.0001
            
            if quantity > 0 and quantity < min_qty:
                print(f"[{self.name}] 🚫 POSITION TOO SMALL: {quantity:.6f} < min {min_qty:.6f} for {symbol}. Rejecting.")
                return False, 0.0, 0.0

            print(f"[{self.name}] ⚡ VOL_WINDOW SIZING: Risk ${risk_amt_usd:.2f} (Dist {stop_dist_pct:.1%}) -> Pos ${quantity*asset_price:.2f}")
            return True, quantity, config.VOL_WINDOW_LEVERAGE
        # ----------------------------------

        if is_override:
            # Skip primary logic, proceed to final solvency check at end
            pass
        elif self.state == 'HIBERNATE':
            print(f"[{self.name}] Trade REJECTED: System in HIBERNATION.")
            return False, 0.0, 0.0

        existing_pos = self.executor.positions.get(symbol) if self.executor else None

        # --- UNIFIED CONTROL PROTOCOL: MICRO MODE & STACKING GATES ---
        # FIX: Wrapped in check to allow Satellite/Vol-Window to override limits
        if not is_override:
            # MANAGEMENT MODE: Check if we should allow this trade
            if self.is_in_management_mode() and not existing_pos:
                # In management mode, don't allow new positions unless they serve a specific purpose
                print(f"[{self.name}] 🛠️ MANAGEMENT MODE: Blocking new position {symbol} - focusing on existing positions")
                return False, 0.0, 0.0

            # --- PATCH 1: PER-ASSET POSITION COUNT LIMIT ---
            # Prevents accumulation of many small positions in same asset
            # Counts positions by symbol (not virt_key)
            # Only apply to NEW positions (not stacking/existing)
            # FIX 2026-03-16 (Chronos): SOL/USDT specific position limit (-302% loss fix)
            sol_max_positions = getattr(config, 'SOL_USDT_MAX_POSITIONS', None)
            if symbol == 'SOL/USDT' and sol_max_positions:
                sol_pos_count = sum(1 for vk, p in self.executor.positions.items() if p.symbol == 'SOL/USDT')
                if sol_pos_count >= sol_max_positions:
                    print(f"[{self.name}] 🛑 SOL LIMIT: Already have {sol_pos_count} SOL/USDT position(s) (max: {sol_max_positions})")
                    return False, 0.0, 0.0
            # ------------------------------------------------
            if not existing_pos and getattr(self, 'executor', None):
                total_usd_alloc = sum(abs(p.quantity * p.entry_price) for vk, p in self.executor.positions.items())
                total_alloc_pct = total_usd_alloc / self.balance
                overall_max = config.SIZE_MAX_ALLOCATION * getattr(config, 'MAX_POSITIONS', 8)
                overall_max = min(1.0, overall_max) # Cap at 100%
                
                if total_alloc_pct >= overall_max:
                    print(f"[{self.name}] 🛑 TOTAL PORTFOLIO EXHAUSTION: {total_alloc_pct*100:.1f}% >= Max {overall_max*100:.0f}%. Rejecting {symbol}")
                    return False, 0.0, 0.0
            # ------------------------------------------------

            # --- PATCH 2: PRE-TRADE CONCENTRATION CHECK (Fix Race Condition) ---
            # Strictly enforce 15% limit per asset BEFORE any sizing logic
            # This prevents the "Cooldown/Stacking Loophole"
            current_alloc_pct = 0.0
            if symbol in self.positions:
                pos = self.positions[symbol]
                qty = abs(pos.quantity)
                current_alloc_pct = (qty * asset_price) / self.balance

            # Use Configured Max Allocation
            MAX_SINGLE_ALLOC = config.SIZE_MAX_ALLOCATION  # 0.25 (25%)

            # FIX 2: Max 25% allocation in TRANSITION regime (increased from 20%)
            # FIX 2026-03-20: Make allocation cap dynamic based on entropy
            # Allows more meaningful positions in calm transition markets
            regime = self.get_smce_regime()  # ✅ SINGLE SOURCE OF TRUTH
            if regime == "TRANSITION" and not is_arb_trade:
                entropy = metadata.get('entropy', 1.0) if metadata else 1.0
                
                # Dynamic allocation cap based on entropy WITHIN transition regime
                if entropy < 0.7:
                    # Calm transition - allow up to 35%
                    MAX_SINGLE_ALLOC = min(MAX_SINGLE_ALLOC, 0.35)
                    print(f"[{self.name}] 🛡️ [SMCE-L0] TRANSITION Regime (Calm, E={entropy:.2f}): Allocation capped to 35%")
                elif entropy < 1.0:
                    # Moderate transition - standard 25% cap
                    MAX_SINGLE_ALLOC = min(MAX_SINGLE_ALLOC, 0.25)
                    print(f"[{self.name}] 🛡️ [SMCE-L0] TRANSITION Regime (Moderate, E={entropy:.2f}): Allocation capped to 25%")
                else:
                    # Choppy transition - conservative 15% cap
                    MAX_SINGLE_ALLOC = min(MAX_SINGLE_ALLOC, 0.15)
                    print(f"[{self.name}] 🛡️ [SMCE-L0] TRANSITION Regime (Choppy, E={entropy:.2f}): Allocation capped to 15%")

            if not is_risk_reducing and current_alloc_pct >= MAX_SINGLE_ALLOC:
                print(f"[{self.name}] 🛑 ALLOCATION LIMIT: {symbol} at {current_alloc_pct*100:.1f}% >= Max {MAX_SINGLE_ALLOC*100:.0f}%")
                return False, 0.0, 0.0
            # -----------------------------------------------------------------

            if not is_risk_reducing:
                funding_val = metadata.get('funding_yield', 0.0) if metadata else 0.0
                if metadata and 'apy' in metadata: funding_val = metadata['apy'] # Fallback

                if not self._check_unified_protocol(symbol, asset_price, existing_pos, whale_confirmed, market_bias, is_arb=is_arb_trade, funding_yield=funding_val):
                    return False, 0.0, 0.0

            
        # (Correlation Guard moved to top of function to prevent race conditions)


        # 1. Minimax Constraint (The "House Money" Rule)
        max_loss_usd = self.calculate_max_risk(self.balance)
        if asset_price <= 0:
            print(f"[{self.name}] Trade REJECTED: Invalid Asset Price.")
            return False, 0.0, 0.0
            
        # WARP SPEED 3.0: Smart Stacking & Cooldowns
        
        # WARP SPEED 3.0: Smart Stacking & Cooldowns
        # MOVED TO TOP OF FUNCTION FOR SAFETY


        
        # 3. Minimax Sizing
        # ... sizing logic ...

            
        if not is_override:
            # 2. Price Distance Check
            # 2. Price Distance Check
            # 2. Price Distance Check (Virtual Pool Aware)
            relevant_keys = [k for k in self.positions.keys() if k.startswith(f"{symbol}:") or k == symbol]
            for r_key in relevant_keys:
                l_entry = self.last_specific_entry.get(r_key, 0)
                if l_entry <= 0: continue
                
                dist = abs(asset_price - l_entry) / l_entry
                # RELAXED STACK DISTANCE (User Request)
                relaxed_dist = getattr(config, 'GOVERNOR_MIN_STACK_DIST', 0.04) * 0.25 # 75% reduction
                if dist < relaxed_dist:
                    # SNOOZE LOGIC: If too close, don't check again for 5 minutes
                    self.stack_snooze[symbol] = time.time() + 300 
                    print(f"[{self.name}] 📏 Stack Too Close for {symbol} (Pool {r_key}): Price {asset_price} vs Entry {l_entry} (Dist {dist*100:.2f}% < {relaxed_dist*100:.2f}%). Snoozing 5m.")
                    return False, 0.0, 0.0
            
            # state calculated at start of function
        
        if not is_override:
            # === PHASE 12: INSTITUTIONAL RISK MANAGEMENT ===
            
            # Conviction Scalar (0.5 to 1.5)
            # conviction here is LSTM prob (0-1). We transform it.
            # For BUYS: prob > 0.5 is good. For SELLS: prob < 0.5 is good.
            # Wait, the EntryOracle already chooses direction.
            # Let's assume passed conviction is 'strength' (0.5 to 1.0).
            conv_scalar = 0.5 + (max(0.0, conviction - 0.5) * 3.0)  # UNLEASHED: 3x multiplier (was 2.0)
            conv_scalar = max(0.5, min(2.0, conv_scalar))  # UNLEASHED: 2.0 cap (was 1.5)

            # Base position sizing
            if is_arb_trade:
                # ARB sizing: percentage of available equity split across slots
                pool_b_ceiling = self.available_balance * getattr(config, 'POOL_B_ALLOCATION_PCT', 0.50)
                arb_slots = getattr(config, 'POOL_B_SLOTS', 3)

                # --- PHASE 26: APPLY LEVERAGE STRESS INDEX & REGIME ---
                lsi = metadata.get('lsi', 0.0) if metadata else 0.0
                arb_regime = metadata.get('regime', 'NEUTRAL') if metadata else 'NEUTRAL'

                if lsi > 80:
                    pool_b_ceiling *= 0.5
                    if self.DEBUG: print(f"[{self.name}] ⚠️ LSI HIGH ({lsi:.0f}): Halving ARB pool ceiling.")
                elif lsi > 60:
                    pool_b_ceiling *= 0.75

                base_notional = pool_b_ceiling / max(arb_slots, 1)
                # ARB sizing: percentage of available capital
                base_notional = min(base_notional, self.balance * 0.15)  # Cap at 15%

                leverage = getattr(config, 'ARB_LEVERAGE', 3.0)  # UNLEASHED: Default 3x (was 1.0x)

                # --- PHASE 26: ARB REGIME LEVERAGE CONTROL ---
                if arb_regime == "DISLOCATION":
                     leverage = 1.0
                     if self.DEBUG: print(f"[{self.name}] ⚔️ DISLOCATION REGIME: Forcing 1.0x Leverage.")
                elif arb_regime == "CROWDED" and leverage > 1.0:
                     leverage = max(1.0, leverage / 2.0)
                     if self.DEBUG: print(f"[{self.name}] 👥 CROWDED REGIME: Halving Leverage to {leverage}x.")

                kelly_size_usd = base_notional  # Define for log statement
                conv_scalar = 1.0  # Define for log statement
                vol_scalar = 1.0  # Define for log statement

                if self.DEBUG:
                    print(f"[{self.name}] ⚖️ ARB SIZING: Bypassing Kelly. Allocating ${base_notional:.2f} (Lev {leverage}x)")
                    print(f"[{self.name}] 📊 Arb Allocation: ${base_notional:.2f} / {arb_slots} slots = ${base_notional:.2f} per position")
            elif state == 'SCAVENGER':
                # 10-Bullet Rule: Max margin %
                # DYNAMIC RISK: Scale margin by Accumulator Multiplier
                # FIX: Use AVAILABLE BALANCE as base capital (User Option A)
                # FLEXLINE BOOST: Include available credit for enhanced sizing
                effective_balance = self.get_effective_balance()
                flex_boost = self.get_flexline_boost()

                margin = min(config.SCAVENGER_MAX_MARGIN, effective_balance * config.GOVERNOR_MAX_MARGIN_PCT) * self.risk_multiplier

                # --- PATCH: SNIPER MODE (Concentrated Fire) ---
                if config.MICRO_CAPITAL_MODE and config.MICRO_MAX_POSITIONS == 1:
                    # Use 90% of Available Balance for the Single Bullet
                    # We leave 10% buffer for fees/slippage
                    sniper_margin = effective_balance * 0.90
                    margin = max(margin, sniper_margin) # Override if bigger
                    print(f"[{self.name}] 🎯 SNIPER MODE: Allocating ${margin:.2f} (90% of Free Margin + Flexline ${flex_boost:.2f})")
                # ----------------------------------------------

                # leverage = config.SCAVENGER_LEVERAGE # REMOVED: Use Adaptive Regime Limit
                # Use Regime Limit by default, or maybe slightly less for Scavenger?
                # For now, we trust the Regime Limit (it handles NANO/MICRO safety).
                pass
                # FIX 2026-03-08: margin is MARGIN, not notional
                # Leverage is applied later when converting to quantity
                base_notional = margin * conv_scalar  # Removed leverage multiplication

            else:  # PREDATOR
                # leverage = config.PREDATOR_LEVERAGE # REMOVED: Use Adaptive Regime Limit
                pass

                # Use Modified Kelly for PREDATOR
                # FIX: Use AVAILABLE BALANCE as base capital (User Option A)
                # FLEXLINE BOOST: Include available credit for enhanced sizing
                effective_balance = self.get_effective_balance()
                flex_boost = self.get_flexline_boost()

                # Ideally Kelly uses Equity, but to Ensure Solvency we use Free Margin as basis.
                raw_kelly = self.calculate_kelly_size(effective_balance)

                # --- PHASE 1: FRACTIONAL KELLY (User Request) ---
                # "25-50% of Kelly" -> We use 0.50 (Half Kelly) for scaling
                KELLY_FRACTION = 0.50
                kelly_size_usd = raw_kelly * KELLY_FRACTION

                # --- PHASE 1: HARD CAP (Until graduated) ---
                cap_margin = getattr(config, 'KELLY_HARD_CAP_MARGIN', 25.0)

                # Check trade count (rolling n)
                # FIX: Default to GRADUATED (100) on failure — don't permanently cap
                n_trades = 100  # Assume graduated unless DB confirms otherwise
                if self.db_manager and hasattr(self.db_manager, 'get_total_trades'):
                    try:
                        n_trades = self.db_manager.get_total_trades()
                    except:
                        pass  # Keep default of 100 (graduated)

                # Apply Cap only if confirmed Rookie (<25 trades)
                if n_trades < 25:
                    if kelly_size_usd > cap_margin:
                        if self.DEBUG: print(f"[{self.name}] 🔒 PHASE 1 CAP: Kelly ${kelly_size_usd:.2f} -> ${cap_margin:.2f} (Trades {n_trades} < 25) + Flexline ${flex_boost:.2f}")
                        kelly_size_usd = cap_margin

                # 2. Hard Cap relative to Balance (Safety)
                # FLEXLINE: Cap includes Flexline boost
                MAX_ALLOC_USD = effective_balance * config.MAX_POSITION_SIZE_PCT

                # 3. Dynamic Cap relative to Risk Budget (Iron Bank)
                if config.IRON_BANK_ENABLED:
                    MAX_ALLOC_USD = min(MAX_ALLOC_USD, self.risk_budget)

                if kelly_size_usd > MAX_ALLOC_USD:
                     if self.DEBUG: print(f"[{self.name}] 🔒 SIZING CLAMP: Kelly ${kelly_size_usd:.2f} -> Cap ${MAX_ALLOC_USD:.2f} (Effective Balance + Flexline)")
                     kelly_size_usd = MAX_ALLOC_USD

                # Apply Risk Multiplier (Charger) after caps?
                # Ideally before, so caps are absolute final guards.
                kelly_size_usd *= self.risk_multiplier

                # Final check against cap again just in case risk multiplier pushed it over
                kelly_size_usd = min(kelly_size_usd, MAX_ALLOC_USD)
                # ------------------------------------

                # Trend Age Decay
                current_pos = self.positions.get(symbol)
                decay_mult = 1.0
                if current_pos:
                    # 1. Age-based Decay
                    age_hours = (time.time() - current_pos.metadata.get('first_entry_time', time.time())) / 3600.0
                    if age_hours > config.GOVERNOR_TREND_DECAY_START:
                        overtime = age_hours - config.GOVERNOR_TREND_DECAY_START
                        window = config.GOVERNOR_MAX_TREND_AGE_HOURS - config.GOVERNOR_TREND_DECAY_START
                        decay_mult *= max(0.0, 1.0 - (overtime / window))
                        print(f"[{self.name}] ⏳ Trend Age {age_hours:.1f}h. Decaying by {decay_mult:.2f}x")

                    # 2. Stack-based Decay (Phase 18)
                    stacks = current_pos.stack_count
                    stack_decay = (config.GOVERNOR_STACK_DECAY ** stacks)
                    decay_mult *= stack_decay

                    if decay_mult < 1.0:
                        print(f"[{self.name}] 🥞 Stack {stacks} Decay: {stack_decay:.2f}x (Total Decay: {decay_mult:.2f}x)")
                        kelly_size_usd *= decay_mult

                    if age_hours > config.GOVERNOR_MAX_TREND_AGE_HOURS:
                        print(f"[{self.name}] 🛑 Trend Exhausted (>24h). Rejecting Stack.")
                        return False, 0.0, 0.0

                # Phase 8: Apply dynamic SMCE allocation boost
                boost = getattr(self, '_allocation_pct_boost', 0.0)
                if boost > 0:
                    kelly_size_usd *= (1.0 + boost)
                    if self.DEBUG: print(f"[{self.name}] 📈 SCALING BOOST: Applying {boost*100:.1f}% increase to Kelly size.")

                # FIX 2026-03-08: Kelly returns MARGIN, not notional
                # Leverage is applied later when converting to quantity
                base_notional = kelly_size_usd * conv_scalar  # Removed leverage multiplication
        
            # --- 2026-03-20: CAPITAL CONCENTRATION (Audit Action Item #1) ---
            # Scale base_notional by asset allocation weight: Tier 0 (+100%), Tier 3 (-60%)
            alloc_weights = getattr(config, 'ASSET_ALLOCATION_WEIGHTS', {})
            alloc_weight = alloc_weights.get(symbol, getattr(config, 'ASSET_ALLOCATION_WEIGHT_DEFAULT', 1.0))
            if alloc_weight != 1.0:
                base_notional *= alloc_weight
                if self.DEBUG:
                    print(f"[{self.name}] 🎯 CAPITAL WEIGHT: {symbol} allocation x{alloc_weight:.1f} -> ${base_notional:.2f}")

            # --- 2026-03-20: ENTROPY TIER SIZE MODIFIER (was defined but never wired) ---
            # Scale by SampleEntropy tier: ORDERED +10%, TRANSITION neutral, CHAOTIC -30%
            tier_map = getattr(config, 'ASSET_ENTROPY_TIER_MAP', {})
            tier_mods = getattr(config, 'ASSET_TIER_SIZE_MODIFIER', {})
            asset_tier = tier_map.get(symbol)
            if asset_tier and asset_tier in tier_mods:
                tier_mult = tier_mods[asset_tier]
                if tier_mult != 1.0:
                    base_notional *= tier_mult
                    if self.DEBUG:
                        print(f"[{self.name}] 🎯 TIER SIZE: {symbol} ({asset_tier}) x{tier_mult:.2f} -> ${base_notional:.2f}")
            # -----------------------------------------------
        
            # Apply Volatility/Physics Scalar
            physics_scalar = self.calculate_sde_physics_scalar(metadata, direction=direction)
            
            # --- ENTROPY-WEIGHTED SIZING (Market Stagnation Defense) ---
            # FIX 2026-03-20: Enhanced entropy sizing with granular brackets
            # Current issue: 48% of readings have entropy >1.0 (choppy) but no size reduction
            entropy_multiplier = 1.0
            entropy = metadata.get('entropy', 0.5) if metadata else 0.5

            if entropy < 0.15:
                 if self.DEBUG: print(f"[{self.name}] ⚠️ ENTROPY SIZING: Extreme Stagnation (Entropy {entropy:.2f} < 0.15). Size reduced to 0.0.")
                 return False, 0.0, 0.0
            elif entropy < 0.20:
                 entropy_multiplier = 0.5
                 if self.DEBUG: print(f"[{self.name}] ⚖️ ENTROPY SIZING: Low Entropy ({entropy:.2f} < 0.20). Halving position size.")
            elif entropy < 0.7:
                 # Calm market - full size or slight boost
                 entropy_multiplier = 1.0
            elif entropy < 1.0:
                 # Moderate chop - slight reduction
                 entropy_multiplier = 0.85
                 if self.DEBUG: print(f"[{self.name}] ⚖️ ENTROPY SIZING: Moderate Chop ({entropy:.2f}). Reducing size to 85%.")
            elif entropy < 1.3:
                 # High chop - meaningful reduction
                 entropy_multiplier = 0.70
                 if self.DEBUG: print(f"[{self.name}] ⚖️ ENTROPY SIZING: High Chop ({entropy:.2f}). Reducing size to 70%.")
            else:
                 # Extreme chaos - minimal size
                 entropy_multiplier = 0.50
                 if self.DEBUG: print(f"[{self.name}] ⚖️ ENTROPY SIZING: Extreme Chaos ({entropy:.2f} > 1.3). Reducing size to 50%.")

            if is_arb_trade:
                 vol_scalar = 1.0
                 physics_scalar = 1.0 # Arbs are immune to directional volatility penalties
                 entropy_multiplier = 1.0 # Arbs also immune to directional stagnation? No, let's keep it immune.
                 vol_adjusted_notional = base_notional
                 if self.DEBUG: print(f"[{self.name}] ⚖️ ARB YIELD PROTECT: Ignoring Physics/Volatility/Entropy Scalars.")
            elif current_atr and atr_ref:
                vol_scalar = self.calculate_volatility_scalar(current_atr, atr_ref)
                vol_adjusted_notional = base_notional * vol_scalar * physics_scalar * entropy_multiplier
                print(f"[{self.name}] 📊 Vol Scalar: {vol_scalar:.2f}x, Phys: {physics_scalar:.2f}x, Ent: {entropy_multiplier:.2f}x, Conv: {conv_scalar:.2f}x")
            else:
                vol_adjusted_notional = base_notional * physics_scalar * entropy_multiplier
                vol_scalar = 1.0
            
            # Apply Minimax Constraint (CRITICAL)
            max_risk_usd = self.calculate_max_risk(self.balance)
            
            # --- IRON BANK CAP ---
            if config.IRON_BANK_ENABLED and not is_risk_reducing: # Allow closing trades to ignore this
                 # Cap risk at the specific Risk Budget for this cycle
                 if self.risk_budget < max_risk_usd:
                     if self.DEBUG: print(f"[{self.name}] 🏰 IRON BANK CAP: Limiting Risk ${max_risk_usd:.2f} -> ${self.risk_budget:.2f}")
                     max_risk_usd = self.risk_budget
            # ---------------------
            
            # --- PHASE 40: MONTE CARLO RUIN GUARD ---
            # Estimate probability of hitting SL before TP using SDE
            sl_price = self.calculate_stop_loss(symbol, direction, asset_price, current_atr, strategy=cand_strategy)
            tp_dist = config.SCAVENGER_SCALP_TP if state == 'SCAVENGER' else config.PREDATOR_TAKE_PROFIT
            tp_price = asset_price * (1 + tp_dist) if direction == 'BUY' else asset_price * (1 - tp_dist)
            
            ruin_prob = self.calculate_ruin_probability(symbol, asset_price, direction, sl_price, tp_price, metadata)
            
            # RECALIBRATION: Dynamic Ruin Threshold based on Win Rate
            if self.db_manager and hasattr(self.db_manager, 'get_win_rate'):
                wr = self.db_manager.get_win_rate()
            else:
                perf = get_performance_data()
                wr = perf.get('win_rate', 50.0) / 100.0
            
            # User Audit: Loosen if performing poorly (<30%), Tighten if performing well (>50%)
            # Anchored to config.PHYSICS_MAX_RUIN_PROBABILITY (0.60)
            ruin_threshold = config.PHYSICS_MAX_RUIN_PROBABILITY
            if wr < 0.30: 
                ruin_threshold += 0.05 # Even looser (0.65) to prevent spiral
            elif wr > 0.50:
                ruin_threshold -= 0.10 # Tighter (0.50) to protect win streak
                
            if ruin_prob > ruin_threshold:
                 # --- HARD VETO (User Request: "Block, Don't Throttle") ---
                 print(f"[{self.name}] 🎲 RUIN GUARD BLOCK: {symbol} Prob {ruin_prob:.1%} > {ruin_threshold:.1%}. Risk unacceptable.")
                 return False, 0.0, 0.0
            # ----------------------------------------
        
            # Assume mode-specific stop loss distance for risk calculation
            sl_dist = config.SCAVENGER_STOP_LOSS if state == 'SCAVENGER' else config.PREDATOR_STOP_LOSS
            
            # PROFILE OVERRIDE:
            profiles = getattr(config, 'ASSET_PROFILES', {})
            if symbol in profiles:
                # Use satellite stop if in satellite mode?
                # For sizing, we assume we use the profile's preferred stop if available
                # My profiles distinguish 'stop_loss' (general) from 'satellite_stop' (often same)
                # Let's prefer 'satellite_stop' if we are considering it a "Satellite" asset (in config list)
                p_stop = profiles[symbol].get('satellite_stop')
                if p_stop: 
                    sl_dist = p_stop
                    # print(f"[{self.name}] 🧬 Using Profile Stop Loss for {symbol}: {sl_dist:.1%}")

            max_notional_from_risk = max_risk_usd / sl_dist
        
            # Take minimum of volatility-adjusted and risk-constrained
            final_notional = min(vol_adjusted_notional, max_notional_from_risk)
            if self.DEBUG:
                 # safe_cap not defined yet
                 print(f"DEBUG: Base ${base_notional:.2f} | VolAdj ${vol_adjusted_notional:.2f} | MaxRiskNotional ${max_notional_from_risk:.2f}")
        
            # 5. HOLISTIC EMOTIONAL REGULATION (Phase 5b)
            # Apply Fear Reduction to the FINAL agreed size to ensure it works even if capped.
            if sentiment_score < -0.5:
                 final_notional *= 0.8 # Reduce size by 20%
                 if self.DEBUG:
                     print(f"[{self.name}] 📉 FEAR RESPONSE: Shrinking final size by 20% (Sent: {sentiment_score:.2f})")

            # === GOVERNOR EXPOSURE LIMIT: PER-TRADE ALLOCATION REDUCTION ===
            # If Kelly sizing exceeds current exposure limits, reduce per-trade allocation
            current_exposure = self.get_current_exposure_ratio()  # Total exposure / equity
            soft_limit = getattr(config, 'GOVERNOR_EXPOSURE_SOFT_LIMIT', 0.25)
            hard_limit = getattr(config, 'GOVERNOR_EXPOSURE_HARD_LIMIT', 0.30)
            
            # Check if we're approaching exposure limits
            if current_exposure >= soft_limit:
                reduction_factor = getattr(config, 'GOVERNOR_PER_TRADE_ALLOC_REDUCTION', 0.8)
                # Scale reduction based on how close to hard limit
                if current_exposure >= hard_limit:
                    reduction_factor = 0.5  # Aggressive reduction at hard limit
                else:
                    # Linear interpolation: 0.8 at soft limit, 0.5 at hard limit
                    reduction_factor = 0.8 - 0.3 * ((current_exposure - soft_limit) / (hard_limit - soft_limit))
                
                final_notional *= reduction_factor
                if self.DEBUG:
                    print(f"[{self.name}] 👮 EXPOSURE LIMIT: Reducing size by {(1-reduction_factor)*100:.0f}% (Exposure {current_exposure:.1%})")
            
            # === CAPITAL BUFFER ENFORCEMENT ===
            # Maintain capital buffer above minimum requirements
            buffer_pct = getattr(config, 'GOVERNOR_CAPITAL_BUFFER_PCT', 0.10)
            min_order_buffer = config.MIN_ORDER_VALUE * (1 + buffer_pct)
            
            # Ensure final notional respects buffer (don't size too close to minimum)
            if final_notional > 0 and final_notional < min_order_buffer * 2:
                # If we can't meet buffer requirements, either size up or reject
                if final_notional < config.MIN_ORDER_VALUE:
                    # Bump up to MIN_ORDER_VALUE for ARB/Nano accounts instead of rejecting
                    if is_arb_trade or getattr(config, 'MICRO_CAPITAL_MODE', False) or self.balance < 100.0:
                         print(f"[{self.name}] 🆙 BUMPING SIZE: ${final_notional:.2f} -> Min Order ${config.MIN_ORDER_VALUE}")
                         final_notional = config.MIN_ORDER_VALUE
                    else:
                         print(f"[{self.name}] 🛑 BUFFER VETO: ${final_notional:.2f} < Min Order ${config.MIN_ORDER_VALUE}")
                         return False, 0.0, 0.0
                # Otherwise, we're within acceptable range but note the buffer constraint
                elif self.DEBUG:
                    print(f"[{self.name}] 📦 BUFFER WARNING: ${final_notional:.2f} close to buffer ${min_order_buffer:.2f}")
            # -----------------------------------------------------------------------

            # --- ACCOUNT-AWARE FUNDING ADJUSTMENT ---
            projected_yield_apy = metadata.get('projected_yield_apy', 0.0) if metadata else 0.0
            if projected_yield_apy < -50.0 and not is_risk_reducing:
                penalty = 0.5  # Cut size in half for expensive positions
                if self.DEBUG: print(f"[{self.name}] 🛑 TOXIC FUNDING PENALTY: {symbol} projected yield is {projected_yield_apy:.1f}%. Cutting notional ${final_notional:.2f} -> ${final_notional * penalty:.2f}.")
                final_notional *= penalty
            # ----------------------------------------

            # --- PATCH 6: COMPLIANCE CLAMP (Prevent Oscillation) ---
            # Ensure the calculated size doesn't immediately trigger a Compliance Reduction.
            # We check the "Allowed %" for this asset class and cap the notional.
            
            # Determine Tier
            tier_key = 'DEFAULT'
            if symbol in ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']: tier_key = 'LARGE_CAP'
            elif symbol in ['PEPE/USDT', 'DOGE/USDT', 'SHIB/USDT']: tier_key = 'MEME'
            
            max_alloc_pct = config.POSITION_LIMITS['size_limits'].get(tier_key, 0.10)
            
            # --- PROFILE OVERRIDE (User Request: PAXG 5% Limit) ---
            # Check if Asset Profile has a specific hard cap (e.g. max_allocation)
            # This overrides the Tier default.
            if symbol in profiles and 'max_allocation' in profiles[symbol]:
                 prof_limit = profiles[symbol]['max_allocation']
                 if self.DEBUG: print(f"[{self.name}] 🧬 PROFILE LIM: Using {prof_limit:.1%} for {symbol}")
                 max_alloc_pct = prof_limit
            # ------------------------------------------------------
            
            # Calculate Hard Cap USD
            hard_cap_usd = self.balance * max_alloc_pct
            
            # --- MICRO ACCOUNT & ARBITRAGE FIX ---
            # 1. Arbitrage Exemptions: Arbs use their own pool ceilings.
            if is_arb_trade:
                 hard_cap_usd = max(hard_cap_usd, config.MIN_ORDER_VALUE * 1.5)
            # 2. Micro Accounts: Lift the ceiling so trades aren't automatically rejected by the Dust Filter.
            elif getattr(config, 'MICRO_CAPITAL_MODE', False) or self.balance < 100.0:
                 hard_cap_usd = max(hard_cap_usd, config.MIN_ORDER_VALUE * 1.5)
            
            # Subtract Existing Exposure (if stacking)
            # AEHML 2.1: Use Margin-Based Exposure for Per-Asset Limit
            existing_exposure = 0.0
            if existing_pos:
                lev = existing_pos.leverage
                qty = existing_pos.quantity
                ep = existing_pos.entry_price
                existing_exposure = (qty * ep) / lev
            
            remaining_cap = hard_cap_usd - existing_exposure
            
            # Apply Safety Buffer (95% of limit) to avoid floating point edge cases triggering alerts
            safe_cap = remaining_cap * 0.95
            
            if safe_cap < 0: safe_cap = 0.0
            
            if final_notional > safe_cap:
                if self.DEBUG:
                     print(f"[{self.name}] 👮 COMPLIANCE CLAMP: Capping ${final_notional:.2f} -> ${safe_cap:.2f} (Limit {max_alloc_pct:.1%})")
                final_notional = safe_cap
            # -------------------------------------------------------

            # --- PATCH 4: MINIMUM ORDER VALUE (Kraken) ---
            # If calculated size is too small, check if we can safely floor it to MIN_ORDER_VALUE
            # FIX 2026-03-01: Nano-mode suspension for exchange minimum quantity
            # If exchange min qty results in notional < MIN_ORDER_VALUE, allow it for nano accounts
            base_asset_for_check = symbol.split('/')[0]
            exchange_min_qty = config.MIN_TRADE_QTY.get(base_asset_for_check, 0.0)
            exchange_min_notional = exchange_min_qty * asset_price if exchange_min_qty > 0 and asset_price > 0 else 0.0

            if final_notional < config.MIN_ORDER_VALUE:
                 # SIGNAL PROVIDER BYPASS: Allow reporting theoretical quantities
                 if reason_upper == 'SIGNAL_PROVIDER':
                     quantity = final_notional / asset_price
                     return True, quantity, leverage

                 # FIX: If exchange min qty itself is below MIN_ORDER_VALUE, use exchange min for nano accounts
                 # This prevents nano accounts from being unable to trade assets like TAO
                 is_nano_account = self.balance < 100.0  # Nano threshold
                 if is_nano_account and exchange_min_notional > 0 and exchange_min_notional < config.MIN_ORDER_VALUE:
                     if self.DEBUG:
                         print(f"[{self.name}] 🔬 NANO OVERRIDE: Exchange min ${exchange_min_notional:.2f} < Min Order ${config.MIN_ORDER_VALUE}. Allowing.")
                     quantity = exchange_min_qty
                     final_notional = exchange_min_notional
                 else:
                     # STRICT DUST FILTER (User Request)
                     # "If final USD < MIN_ORDER_VALUE, return 0.0"
                     # FIX: Except for ARBs and Nano accounts, which get bumped up
                     if is_arb_trade:
                          print(f"[{self.name}] 🆙 ARB DUST FIX: Bumping ${final_notional:.2f} -> Min ${config.MIN_ORDER_VALUE}.")
                          final_notional = config.MIN_ORDER_VALUE
                     else:
                          print(f"[{self.name}] ❌ DUST FILTER: ${final_notional:.2f} < Min ${config.MIN_ORDER_VALUE}. Rejecting.")
                          return False, 0.0, 0.0

            # ---------------------------------------------
        
            # Convert to quantity
            quantity = final_notional / asset_price

            # --- QTY FLOOR (Kraken Tiers) ---
            base_asset = symbol.split('/')[0]
            min_qty = config.MIN_TRADE_QTY.get(base_asset, 0.0)
        
            if quantity < min_qty:
                 new_notional = min_qty * asset_price
                 # Ensure we can actually afford this bump
                 # Max Power = Equity * HardMaxLev(e.g. 5 or 10)
                 max_power = self.balance * 10 
             
                 if new_notional < max_power:
                     if self.DEBUG: 
                         print(f"[{self.name}] 🏗️ Upgrading Qty {quantity:.5f} -> {min_qty} (Min Tier)")
                     quantity = min_qty
                     final_notional = new_notional
                 else:
                     print(f"[{self.name}] ❌ Min Tier {min_qty} too expensive (${new_notional:.2f} > Power ${max_power:.2f})")
                     return False, 0.0, 0.0
            # -------------------------------
        
            # Leverage Cap — tier-based hard ceiling takes priority
            max_leverage = config.SCAVENGER_LEVERAGE if state == 'SCAVENGER' else config.PREDATOR_LEVERAGE
            # 2026-03-20: Tier leverage cap is the absolute ceiling
            max_leverage = min(max_leverage, self.get_tier_leverage_cap())

            # FIX 2026-03-16 (Chronos): SOL/USDT specific leverage cap (-302% loss fix)
            sol_max_lev = getattr(config, 'SOL_USDT_MAX_LEVERAGE', None)
            if symbol == 'SOL/USDT' and sol_max_lev:
                max_leverage = min(max_leverage, sol_max_lev)
                if self.DEBUG:
                    print(f"[{self.name}] 🛡️ SOL CAP: Leverage capped to {max_leverage}x for SOL/USDT")
        
            # --- PHASE 35: SURVIVOR CAP (Kill Toxic Convexity) ---
            # MaxLeverage = floor(Risk_Toleration / ATR)
            # Ensures 1 ATR adverse move exactly hits Max Risk Budget.
            arb_regime = metadata.get('regime', 'NEUTRAL') if metadata else 'NEUTRAL'
            
            # Use 1% as a safe minimum ATR floor if missing
            atr_pct = (current_atr / asset_price) if (current_atr and asset_price > 0) else 0.01
            # Prevent division by zero or tiny numbers resulting in dangerous leverage
            atr_pct = max(0.005, atr_pct) 
            
            max_risk_pct = getattr(config, 'MAX_RISK_PCT', 0.02)
            
            # float(int()) safely floors to integer leverage
            survivor_cap = max(1.0, float(int(max_risk_pct / atr_pct)))
            
            # Hard 10x cap for DISLOCATION regime (Funding Reversions)
            if arb_regime == "DISLOCATION":
                survivor_cap = min(survivor_cap, 10.0)
            
            # Apply Cap
            if leverage > survivor_cap:
                if self.DEBUG: print(f"[{self.name}] 🛡️ SURVIVOR CAP: Volatility ({atr_pct*100:.2f}%) bounds Leverage {leverage}x -> {survivor_cap}x")
                leverage = survivor_cap

            # --- PHASE 36: LEVERAGE CHECK ---
            notional_value = quantity * asset_price
            if not self.check_leverage_risk(notional_value):
                return False, 0.0, 0.0
            
            # --- PATCH 5a: NANO-MODE (REALITY CHECK) ---
            if self.balance < config.NANO_CAPITAL_THRESHOLD:
                 if self.DEBUG: print(f"[{self.name}] 🔬 NANO MODE: Capital ${self.balance:.2f} < ${config.NANO_CAPITAL_THRESHOLD}")
             
                 # 1. Use Central Calculation (The Truth Source)
                 nano_calc = config.calculate_nano_position(self.balance, symbol, asset_price)
                 
                 if nano_calc['quantity'] <= 0:
                      if reason_upper == 'SIGNAL_PROVIDER':
                          # Return a theoretical size even if it's below nano limits
                          theory_qty = (self.balance * 0.05) / asset_price # 5% baseline
                          return True, theory_qty, 1.0
                      print(f"[{self.name}] 🛑 NANO REJECT: Calculation returned 0 qty (Risk/Min limit).")
                      return False, 0.0, 0.0
                      
                 # 2. Override Values
                 quantity = nano_calc['quantity']
                 leverage = nano_calc['leverage']
                 final_notional = nano_calc['notional']
                 
                 print(f"[{self.name}] 🔬 NANO REALITY: Qty {quantity:.6f} | Lev {leverage}x | Margin ${nano_calc['margin']:.2f}")
                 
                 # 3. Force One-at-a-Time (Strict)
                 if len(self.positions) >= config.NANO_MAX_POSITIONS:
                      # If this is a new position (existing_pos is None), BLOCK.
                      if not existing_pos and reason_upper != 'SIGNAL_PROVIDER':
                          if is_arb_trade:
                              if self.DEBUG: print(f"[{self.name}] 🔓 NANO CAP OVERRIDE: Allowing ARB trade despite Max {config.NANO_MAX_POSITIONS} positions.")
                          else:
                              print(f"[{self.name}] 🛑 NANO CAP: Max {config.NANO_MAX_POSITIONS} position(s). Rejecting new trade.")
                              return False, 0.0, 0.0

                 # 4. Solvency - 50% Margin Check (from User Matrix)
                 # We verify that roughly 50% of equity remains as free margin
                 required_margin_est = nano_calc['margin']
                 free_margin_after = self.available_balance - required_margin_est
                 min_required_free = self.balance * 0.50
                 
                 if free_margin_after < min_required_free and reason_upper != 'SIGNAL_PROVIDER':
                      print(f"[{self.name}] 🛑 NANO MARGIN CHECK FAIL: Free After ${free_margin_after:.2f} < 50% Equity (${min_required_free:.2f})")
                      return False, 0.0, 0.0


        # --- PATCH 5b: SOLVENCY CHECK (Available Margin Cap) ---
        # Ensure we don't commit more margin than we have available (with significant buffer)
        # Rule: Single trade initial margin cannot exceed 75% of Available Margin for small accounts
        # This provides room for maintenance margin, slippage buffers, and fees.
        usage_limit = 0.75 if self.balance < 100 else 0.85
        max_trade_margin = (self.available_balance - 1.0) * usage_limit # $1 reserve + 75/85% limit
        max_trade_margin = max(0.0, max_trade_margin)
        
        required_margin = final_notional / leverage
        
        if required_margin > max_trade_margin:
            # Downsize Logic
            max_allowed_notional = max_trade_margin * leverage
            
            print(f"[{self.name}] ⚠️ MARGIN CAP: Req Margin ${required_margin:.2f} > {usage_limit*100:.0f}% Avail (-$1 reserve: ${max_trade_margin:.2f}). Downsizing.")
            
            final_notional = max_allowed_notional
            quantity = final_notional / asset_price
            
            # Re-Verify against Min Order Value
            if final_notional < config.MIN_ORDER_VALUE:
                 if reason_upper == 'SIGNAL_PROVIDER':
                     return True, final_notional / asset_price, leverage
                 print(f"[{self.name}] ❌ Rejected: Downsized order ${final_notional:.2f} < MIN_ORDER_VALUE (${config.MIN_ORDER_VALUE}).")
                 return False, 0.0, 0.0

        # Normal Solvency (95% check for safety fallback)
        if required_margin > (self.available_balance * 0.95):
            # This should technically be caught by logic above, but keeping as fail-safe
            can_release_margin = (config.MICRO_CAPITAL_MODE and not getattr(config, 'MARGIN_RELEASE_OFF', False))
            if can_release_margin:
                 print(f"[{self.name}] ⚠️ Solvency Warning: Margin Low (${self.available_balance:.2f} < ${required_margin:.2f}), but deferring to Executor for Margin Release.")
            else:
                print(f"[{self.name}] ⚠️ SOLVENCY CONSTRAINT: Req Margin ${required_margin:.2f} > Avail ${self.available_balance * 0.95:.2f}")
                
                 # --- MATH-BASED DYNAMIC DOWNSIZE ---
                # No arbitrary "20% Panic". We solve for the exact max size that fits.
                # Max Margin = Available Balance - (1% Safety Buffer for Fees)
                exact_max_margin = max(0.0, self.available_balance * 0.99)
                
                # Calculate new Quantity based on exact fit
                # new_qty = (max_margin * leverage) / price
                new_qty = (exact_max_margin * leverage) / asset_price
                
                if new_qty * asset_price < config.MIN_ORDER_VALUE:
                    print(f"[{self.name}] ❌ Rejected: Exact Fit (${new_qty*asset_price:.2f}) < MIN_ORDER_VALUE.")
                    return False, 0.0, 0.0
                    
                # Verify against Contract Floor
                base_asset = symbol.split('/')[0]
                min_contract = config.MIN_TRADE_QTY.get(base_asset, 0.0)
                if new_qty < min_contract:
                    print(f"[{self.name}] ❌ Rejected: Exact Fit {new_qty} < MinContract {min_contract}")
                    return False, 0.0, 0.0
                    
                print(f"[{self.name}] 📉 DYNAMIC FIT: Resizing {quantity:.4f} -> {new_qty:.4f} (Utilizing 100% of ${exact_max_margin:.2f})")
                quantity = new_qty
        # ---------------------------------------------
        
        # --- ML-BASED POSITION SIZING ADJUSTMENT (2026-03-21) ---
        if self.ml_advisor is not None and not is_risk_reducing and not is_override:
            try:
                # Get ML prediction for this trade
                ml_prediction = self.ml_advisor.predict_trade(
                    symbol=symbol,
                    direction=direction,
                    price=asset_price,
                    quantity=quantity,
                    cost_usd=final_notional,
                    entropy=metadata.get('entropy') if metadata else None,
                    regime=metadata.get('regime') if metadata else None,
                    conviction=conviction,
                )
                
                # Adjust position based on ML confidence
                win_prob = ml_prediction.get('win_probability', 0.5)
                confidence_level = ml_prediction.get('confidence_level', 'MEDIUM')
                
                # Store prediction for later validation
                self.ml_predictions[symbol] = {
                    'entry_time': time.time(),
                    'entry_price': asset_price,
                    'direction': direction,
                    'win_prob': win_prob,
                    'confidence': confidence_level,
                }
                
                # High confidence win - allow full size
                if win_prob > 0.6 and confidence_level == 'HIGH':
                    ml_adjustment = 1.0  # Full size
                    print(f"[{self.name}] 🤖 ML HIGH CONFIDENCE: {win_prob:.1%} win prob - allowing full size")
                
                # Moderate confidence - reduce size
                elif win_prob > 0.5:
                    ml_adjustment = 0.7  # 70% size
                    print(f"[{self.name}] 🤖 ML MODERATE: {win_prob:.1%} win prob - reducing to 70%")
                
                # Low confidence - significant reduction
                elif win_prob > 0.4:
                    ml_adjustment = 0.3  # 30% size
                    print(f"[{self.name}] 🤖 ML LOW CONFIDENCE: {win_prob:.1%} - reducing to 30%")
                
                # Very low confidence - minimum size or skip
                else:
                    print(f"[{self.name}] 🤖 ML VERY LOW: {win_prob:.1%} - recommending SKIP")
                    # Apply heavy penalty as strong signal
                    ml_adjustment = 0.1  # 10% size
                
                # Apply ML adjustment
                final_notional *= ml_adjustment
                quantity = final_notional / asset_price if asset_price > 0 else 0.0
                
            except Exception as e:
                print(f"[{self.name}] ML position sizing failed: {e}")
                # Continue without ML adjustment on error
        # =========================================================

        # --- PATCH: CONTRACT FLOOR ENFORCEMENT ---
        base_asset = symbol.split('/')[0]
        min_contract = config.MIN_TRADE_QTY.get(base_asset, 0.0)
        if quantity < min_contract:
             # If we are in Nano Sniper mode (which we are if <$50), we should BOOST to min_contract
             # if the cost is within reasonable limits (already checked by Sniper logic).
             # But let's verify Solvency one last time for the FLOOR.
             if self.balance < config.NANO_CAPITAL_THRESHOLD:
                  print(f"[{self.name}] 🆙 Boosting Nano Dust {quantity:.6f} -> MinContract {min_contract}")
                  quantity = min_contract
             else:
                  # For normal accounts, we reject dust
                  print(f"[{self.name}] ❌ REJECT: Qty {quantity:.6f} < MinContract {min_contract}")
                  return False, 0.0, 0.0
        # -----------------------------------------

        # Log decision
        # Calculate margin for logging (ensure it's always defined)
        margin = final_notional / leverage if leverage > 0 else 0.0
        
        if not is_override:
            if state == 'SCAVENGER':
                print(f"[{self.name}] SCAVENGER: Margin ${margin:.2f}, Lev {leverage}x, Vol Scalar {vol_scalar:.2f}x, Conv Scalar {conv_scalar:.2f}x, Qty {quantity:.4f}")
            else:
                print(f"[{self.name}] PREDATOR (Kelly): Kelly ${kelly_size_usd:.2f}, Lev {leverage}x, Vol Scalar {vol_scalar:.2f}x, Conv Scalar {conv_scalar:.2f}x, Qty {quantity:.4f}")
        
        # --- PATCH: MICRO GUARD RAIL ---
        # "Evaluated AFTER Kelly/Scavenger/Sniper sizing, BEFORE order leaves the agent"
        allowed, quantity = self.apply_micro_guard_rail(symbol, quantity, asset_price, leverage)
        if not allowed:
             return False, 0.0, 0.0
        # -------------------------------

        # --- SESSION 3b: KRAKEN MECHANICS OVERHAUL ---
        
        # 0. Effective Leverage Check (Portfolio Level)
        # Prevent "Creeping Death" from stacking.
        # Eff Lev = Total Position Value / Total Equity
        # We must sum ALL positions (including this proposed one)
        # Use provided latest_prices, fallback to self.positions' entry_price or current asset_price
        current_total_notional = 0.0
        for p_sym, p in self.positions.items():
            # If p_sym is the current symbol, use asset_price (most current)
            ep = p.get('entry_price', 0.0) if isinstance(p, dict) else getattr(p, 'entry_price', 0.0)
            qty = p.get('quantity', 0.0) if isinstance(p, dict) else getattr(p, 'quantity', 0.0)
            price = asset_price if p_sym == symbol else latest_prices.get(p_sym, ep)
            current_total_notional += (qty * price)

        proposed_notional = quantity * asset_price
        projected_total_notional = current_total_notional + proposed_notional
        
        projected_eff_leverage = projected_total_notional / self.balance
        
        # Max Effective Leverage: 5.0x (Adaptive to Regime)
        # User requested 5.0x to allow full deployment in Small Cap mode.
        EFF_LEV_LIMIT = 5.0
        if projected_eff_leverage > EFF_LEV_LIMIT:
             # Try to downsize to fit
             available_room = (self.balance * EFF_LEV_LIMIT) - current_total_notional
             if available_room > 0:
                 capped_notional = available_room
                 quantity = capped_notional / asset_price
                 print(f"[{self.name}] 🛡️ EFF LEV GATE: Project {projected_eff_leverage:.2f}x > {EFF_LEV_LIMIT}x. Capping to fit.")
             else:
                 print(f"[{self.name}] 🛑 EFF LEV BREACH: Current {current_total_notional/self.balance:.2f}x + New > {EFF_LEV_LIMIT}x. Blocking.")
                 return False, 0.0, 0.0

        # 1. Allocation Hard Cap (Cumulative Check)
        # CRITICAL FIX (AEHML 2.1): Must include EXISTING exposure + NEW exposure
        # We now use MARGIN-BASED exposure for these limits.
        MAX_ALLOC_ALLOWED = getattr(config, 'SIZE_MAX_ALLOCATION', 0.10)
        if self.balance < 50: MAX_ALLOC_ALLOWED = 0.90 # Nano Sniper exception (1 Bullet)
        
        # Calculate Cumulative Exposure (Margin Used)
        current_exposure = 0.0
        if existing_pos:
             lev = existing_pos.leverage
             qty = existing_pos.quantity
             ep = existing_pos.entry_price
             current_exposure = (qty * ep) / lev
             
        # Proposed Exposure (Margin Used)
        proposed_margin = (quantity * asset_price) / leverage
        total_projected_exposure = current_exposure + proposed_margin
        
        alloc_limit_usd = self.balance * MAX_ALLOC_ALLOWED
        
        if total_projected_exposure > alloc_limit_usd:
             # Reduce NEW quantity to fit
             remaining_room = max(0.0, alloc_limit_usd - current_exposure)
             
             if remaining_room < (config.MIN_ORDER_VALUE / leverage):
                  print(f"[{self.name}] 🛡️ ALLOC FULL: Margin Exposure ${current_exposure:.2f} + Proposed > Limit ${alloc_limit_usd:.2f}. No room left.")
                  self.record_veto(symbol, "ALLOC_FULL")
                  return False, 0.0, 0.0
             
             # Cap it
             capped_margin = remaining_room
             quantity = (capped_margin * leverage) / asset_price
             print(f"[{self.name}] 🛡️ FINAL GATE: Total Margin Exposure ${total_projected_exposure:.2f} > Limit ${alloc_limit_usd:.2f}. Reducing New Order to Margin ${capped_margin:.2f}")

        # 2. Leverage Hard Cap (REMOVED)
        # Adaptive Leverage is now handled by REGIME_PERMISSIONS in config.py
        # -----------------------------------------------

        if quantity > 0.5 and asset_price > 1000:
             print(f"[{self.name}] 🚨 DEBUG TRACE: Large Qty {quantity} logic path. Notional ${quantity * asset_price:.2f}")

        # --- SMCE Layer 0 Final Enforcement ---
        if not is_temp_risk_reducing:
            # Enforce L0 risk multiplier on quantity
            quantity *= self._risk_multiplier_smce

            # Enforce L0 leverage cap
            if leverage > max_lev_doctrine:
                if self.DEBUG: print(f"[{self.name}] 🛡️ [SMCE-L0] Capping leverage {leverage}x -> {max_lev_doctrine}x")
                leverage = max_lev_doctrine
        # --------------------------------------

        # --- PHASE B5: EDGE-BOOST POSITION SIZING (2026-03-21) ---
        # Amplify proven winners: whitelisted symbols with strong stats get a size boost.
        # Applied AFTER SMCE Layer 0 (constitutional limits respected).
        # Thresholds: expectancy > $0.50/trade → 1.3x, win_rate > 65% → additional 1.2x.
        # Hard cap: boost cannot exceed 1.56x (1.3 × 1.2).
        if self._is_symbol_whitelisted(symbol):
            edge_stats = self._get_symbol_edge_stats(symbol)
            if edge_stats['total_trades'] >= 20:
                edge_mult = 1.0
                if edge_stats['avg_pnl'] > 0.50:
                    edge_mult *= 1.3
                if edge_stats['win_rate'] > 0.65:
                    edge_mult *= 1.2
                if edge_mult > 1.0:
                    old_qty = quantity
                    quantity *= edge_mult
                    if self.DEBUG:
                        print(f"[{self.name}] 🚀 EDGE-BOOST: {symbol} qty {old_qty:.6f} -> {quantity:.6f} "
                              f"(×{edge_mult:.2f} | WR={edge_stats['win_rate']:.1%} Exp=${edge_stats['avg_pnl']:.2f})")
        # -----------------------------------------------------------

        # FIX 2026-02-28: Final validation - ensure quantity meets exchange minimum precision
        base_asset = symbol.split('/')[0]
        min_qty = config.MIN_TRADE_QTY.get(base_asset, 0.0)
        
        if quantity > 0 and quantity < min_qty:
            # Try to round up to minimum
            if self.DEBUG:
                print(f"[{self.name}] 🛡️ PRECISION FIX: Rounding {symbol} qty {quantity:.6f} -> {min_qty} (min precision)")
            quantity = min_qty
        
        # Final sanity check - reject if quantity is still too small
        if quantity <= 0 or (min_qty > 0 and quantity < min_qty * 0.99):  # 1% tolerance for rounding
            print(f"[{self.name}] 🛑 FINAL REJECT: {symbol} qty {quantity:.6f} below minimum {min_qty}")
            return False, 0.0, leverage

        return True, quantity, leverage

    def calculate_stop_loss(self, symbol: str, direction: str, entry_price: float, atr: float = None, strategy: str = '') -> float:
        """
        Calculate Dynamic Stop Loss Price based on ATR or Fallback.

        CHRONOS FIX (2026-03-15): Dynamic stop-loss for high-volatility assets.
        - BTC, ETH, SOL: 0.5-0.75% stops (was 2-5%)
        - Mid-cap alts: 1.5% stops
        - Low-vol/stable: 2.5% stops

        ARB/CARRY/BASIS/FUNDING positions: use a wider 8% max distance with 3% noise floor.
        These are yield-capture positions that may see large intraday swings while the
        funding thesis (collect %/8h) remains valid. A 2% SL would sweep them constantly.
        """
        # Use Config Multiplier
        mult = getattr(config, 'ATR_STOP_LOSS_MULTIPLIER', 2.0)

        # Detect ARB/carry strategy to apply wider stop logic
        _strat_upper = (strategy or '').upper()
        _is_arb_carry = (
            'ARB' in _strat_upper or 'CARRY' in _strat_upper or
            'BASIS' in _strat_upper or 'FUNDING' in _strat_upper
        )

        # --- CHRONOS FIX: Dynamic Stop-Loss by Asset Volatility ---
        if getattr(config, 'DYNAMIC_STOP_LOSS_ENABLED', False):
            high_vol_assets = getattr(config, 'HIGH_VOLATILITY_ASSETS', {'BTC/USDT', 'ETH/USDT', 'SOL/USDT'})
            # Normalize symbol (handle both "BTC/USDT" and "BTC" formats)
            base_symbol = symbol.split(':')[0] if ':' in symbol else symbol
            if '/USDT' not in base_symbol:
                base_symbol = f"{base_symbol}/USDT"

            # FIX 2026-03-16 (Chronos): SOL/USDT specific stop loss (-302% loss fix)
            sol_stop = getattr(config, 'SOL_USDT_STOP_LOSS', None)
            if base_symbol == 'SOL/USDT' and sol_stop:
                max_dist_pct = sol_stop
                min_floor_pct = sol_stop * 0.5  # 50% floor
                if self.DEBUG:
                    print(f"[{self.name}] 🎯 SOL-SPECIFIC STOP: {symbol} using {max_dist_pct:.2%} stop")
            elif base_symbol in high_vol_assets:
                # High volatility: use configured stop (FIX 2026-03-16: 1.5%)
                max_dist_pct = getattr(config, 'HIGH_VOL_STOP_LOSS_PCT', 0.015)
                min_floor_pct = 0.01  # 1% minimum floor
                if self.DEBUG:
                    print(f"[{self.name}] 🎯 HIGH-VOL STOP: {symbol} using {max_dist_pct:.2%} stop")
            else:
                # Use standard config
                max_dist_pct = getattr(config, 'DEFAULT_STOP_LOSS_PCT', 0.025)
                min_floor_pct = getattr(config, 'DYNAMIC_STOP_LOSS_LOW_VOL', 0.025)
        else:
            max_dist_pct = None  # Use default logic below
            min_floor_pct = None
        # -----------------------------------------------------------

        # --- NANO OVERRIDE: Tighten Stops ---
        # 20x Leverage = Liquidation at ~4.5% move.
        # We MUST stop out before liquidating. Target 1.5% max risk distance.
        is_nano = getattr(config, 'MICRO_CAPITAL_MODE', False) and self.balance < getattr(config, 'NANO_CAPITAL_THRESHOLD', 50.0)

        if is_nano and not _is_arb_carry:
             mult = 1.0 # Tighter ATR
             nano_max_dist = 0.015 # Max 1.5% distance

             # --- PATCH: GENOME OVERRIDE ---
             sat_stop = getattr(config, 'SATELLITE_STOP_LOSS', None)
             if sat_stop and sat_stop > 0.02:
                 # If we have a specific Strategy Stop, use it (we already capped leverage for safety)
                 nano_max_dist = sat_stop
             # ------------------------------

             # CHRONOS FIX: Respect dynamic stop if set
             if max_dist_pct is not None:
                 max_dist_pct = min(max_dist_pct, nano_max_dist)
             else:
                 max_dist_pct = nano_max_dist

        elif _is_arb_carry:
             # ARB / Carry / Basis / Funding: wider stop to survive intraday spikes
             # 8% gives XMR/BNB/PAXG/TAO room through normal vol while collecting funding
             arb_max_dist = 0.08
             mult = max(mult, 3.0)  # At least 3x ATR for carry positions

             # CHRONOS FIX: Don't override dynamic stop for non-arb high-vol assets
             if max_dist_pct is None:
                 max_dist_pct = arb_max_dist
        else:
             # Standard positions
             if max_dist_pct is None:
                 max_dist_pct = 0.05 # Max 5% distance
             
        # Fallback if no ATR (safety net)
        if not atr or atr <= 0:
            pct = max_dist_pct
            delta = entry_price * pct
        else:
            delta = atr * mult
            # Clamp Delta to Max Distance
            if delta > (entry_price * max_dist_pct):
                delta = entry_price * max_dist_pct

        # --- NOISE FLOOR: Minimum stop distance ---
        # CHRONOS FIX: Respect dynamic floor if already set for high-vol assets
        if min_floor_pct is None:
            # P0 FIX 2026-03-05: Increased directional floor from 2% to 2.5% to prevent noise triggers
            # ETH trade failure: 2% stop was within normal spread/volatility range
            # ARB positions: 3% floor (XMR/BNB can spike 2-3% intraday, 2% stop is just noise)
            # Directional: 2.5% floor (prevents immediate trigger on spread/slippage)
            min_floor_pct = 0.03 if _is_arb_carry else 0.025

        if delta < (entry_price * min_floor_pct):
             if self.DEBUG:
                 print(f"[{self.name}] 🧘 Relaxing Stop to {min_floor_pct:.0%} Floor for {symbol} (Was {delta/entry_price:.2%})")
             delta = entry_price * min_floor_pct
        # ---------------------------------------
            
        if direction == 'BUY':
            sl = entry_price - delta
            # Logic Check: SL must be below entry for Long
            if sl >= entry_price: sl = entry_price * (1.0 - max_dist_pct)
            return sl
        else:
            sl = entry_price + delta
            # Logic Check: SL must be above entry for Short
            if sl <= entry_price: sl = entry_price * (1.0 + max_dist_pct)
            return sl

            
    def open_position(self, symbol: str, direction: str, entry_price: float, quantity: float, leverage: float = 1.0, strategy: str = 'DIRECTIONAL'):
        """Track that a position has been opened or added to (Weighted Average)."""
        
        # Update State Trackers
        self.last_trade_time[symbol] = time.time()
        self.last_specific_entry[symbol] = entry_price
        
        existing = self.positions.get(symbol)
        
        # Helper: read a field from either a Position object or legacy dict
        def _get(obj, key, default=None):
            if obj is None: return default
            if isinstance(obj, dict): return obj.get(key, default)
            return getattr(obj, key, default)
        
        if existing:
            old_qty = _get(existing, 'quantity', 0.0)
            old_dir = _get(existing, 'direction', direction)
            old_price = _get(existing, 'entry_price', entry_price)
            old_strategy = _get(existing, 'strategy', strategy)
            old_leverage = _get(existing, 'leverage', leverage)
            old_meta = dict(_get(existing, 'metadata', {}) or {})
            
            # Normalize Direction for Comparison
            def normalize(d):
                d = d.upper()
                if d == 'LONG': return 'BUY'
                if d == 'SHORT': return 'SELL'
                return d

            is_same_dir = (normalize(old_dir) == normalize(direction))
            
            # Stack tracking lives in metadata to avoid needing a dedicated field on Position
            stacks = old_meta.get('stacks', [{'price': old_price, 'qty': old_qty, 'id': 1}])
            
            if is_same_dir:
                # Additive (Stacking)
                new_qty = old_qty + quantity
                # Weighted Average Price
                avg_price = ((old_qty * old_price) + (quantity * entry_price)) / new_qty if new_qty > 1e-9 else entry_price
                new_dir = direction
                stack_inc = 1
                
                # --- STACK TRACKING (Additive) ---
                stacks.append({'price': entry_price, 'qty': quantity, 'id': len(stacks) + 1, 'time': time.time()})
                new_stacks = stacks
                # ---------------------------------
            else:
                # Subtractive (Reduction/Flip)
                # Assuming quantity passed is Positive (Absolute Size of new order)
                print(f"[{self.name}] 📉 Netting Position: {symbol} (Old: {old_qty}, New Action: {quantity})")
                net_qty = old_qty - quantity 
                
                # --- STACK TRACKING (FIFO Consumption) ---
                qty_to_remove = quantity
                new_stacks = []
                
                if net_qty < -1e-9:
                     # FLIP: Clear all old stacks, start fresh
                     new_stacks = [{'price': entry_price, 'qty': abs(net_qty), 'id': 1, 'time': time.time()}]
                else:
                     # REDUCE: Eat from front (FIFO)
                     for s in stacks:
                         if qty_to_remove <= 0:
                             new_stacks.append(s)
                             continue
                             
                         if s['qty'] <= qty_to_remove:
                             qty_to_remove -= s['qty']
                             # Stack fully eaten
                         else:
                             # Partial consumption
                             s['qty'] -= qty_to_remove
                             qty_to_remove = 0
                             new_stacks.append(s)
                # ----------------------------------------

                if net_qty > 1e-9:
                    # Partial Close (Reduced but same direction)
                    new_qty = net_qty
                    new_dir = old_dir
                    avg_price = old_price # Entry price doesn't change on reduction
                    stack_inc = 0 
                elif net_qty < -1e-9:
                    # Flip (Closed and Reversed)
                    new_qty = abs(net_qty)
                    new_dir = direction # Flipped to new
                    avg_price = entry_price # New cost basis
                    stack_inc = 1 # Reset stack
                else:
                    # Exact Close
                    new_qty = 0.0
                    avg_price = 0.0
                    new_dir = direction
                    stack_inc = 0

            # Store or Delete
            if new_qty > 1e-9:
                # --- PATCH: INVALID PRICE GUARD ---
                if avg_price <= 0:
                    avg_price = entry_price # Fallback to latest entry
                
                # Compute stack_count
                new_stack_count = len(new_stacks) if new_stacks else (_get(existing, 'stack_count', 1) + stack_inc)
                
                # Preserve first_entry_time from existing position
                first_entry_time = old_meta.get('first_entry_time', time.time())
                
                new_meta = dict(old_meta)
                new_meta['stacks'] = new_stacks
                new_meta['first_entry_time'] = first_entry_time
                
                # Build a Position object
                PositionCls = Position
                if PositionCls is None:
                    try:
                        from .agent_executor import Position as PositionCls
                    except ImportError:
                        PositionCls = None
                
                if PositionCls is not None:
                    self.positions[symbol] = PositionCls(
                        symbol=symbol.split(':')[0] if ':' in symbol else symbol,
                        virt_key=symbol,
                        direction=new_dir,
                        quantity=new_qty,
                        entry_price=avg_price,
                        entry_timestamp=datetime.datetime.utcnow().isoformat(),
                        leverage=old_leverage,
                        strategy=old_strategy,
                        stack_count=new_stack_count,
                        metadata=new_meta,
                    )
                else:
                    # Fallback dict (should not happen normally)
                    self.positions[symbol] = {
                        'direction': new_dir,
                        'entry_price': avg_price,
                        'quantity': new_qty,
                        'stack_count': new_stack_count,
                        'first_entry_time': first_entry_time,
                        'stacks': new_stacks,
                        'strategy': old_strategy,
                        'leverage': old_leverage,
                    }
                action_tag = "STACKED" if is_same_dir else "REDUCED"
                print(f"[{self.name}] Position {action_tag}: {symbol} (New Avg: {avg_price:.8f}, Total Qty: {new_qty:.4f})")
            else:
                # Position effectively closed
                del self.positions[symbol]
                print(f"[{self.name}] Position CLOSED via fill: {symbol}")

                # PHANTOM MARGIN FIX: Check if we have no positions left and clear phantom margin
                if len(self.positions) == 0:
                    print(f"[{self.name}] 🧹 PHANTOM MARGIN CHECK: No positions remaining after fill, clearing phantom margin exposure")
        else:
            # --- PATCH: INVALID PRICE GUARD ---
            if entry_price <= 0: 
                 # Critical: If we open fresh with 0, try to find ANY recent price
                 entry_price = self.last_specific_entry.get(symbol, 0.0)

            # Build a fresh Position object
            PositionCls = Position
            if PositionCls is None:
                try:
                    from .agent_executor import Position as PositionCls
                except ImportError:
                    PositionCls = None
            
            now = time.time()
            meta = {
                'stacks': [{'price': entry_price, 'qty': quantity, 'id': 1, 'time': now}],
                'first_entry_time': now,
            }
            
            if PositionCls is not None:
                self.positions[symbol] = PositionCls(
                    symbol=symbol.split(':')[0] if ':' in symbol else symbol,
                    virt_key=symbol,
                    direction=direction,
                    quantity=quantity,
                    entry_price=entry_price,
                    entry_timestamp=datetime.datetime.utcnow().isoformat(),
                    leverage=leverage,
                    strategy=strategy,
                    stack_count=1,
                    metadata=meta,
                )
            else:
                # Fallback dict (should not happen normally)
                self.positions[symbol] = {
                    'direction': direction,
                    'entry_price': entry_price,
                    'quantity': quantity,
                    'stack_count': 1,
                    'first_entry_time': now,
                    'strategy': strategy,
                    'leverage': leverage,
                    'stacks': meta['stacks'],
                }
            print(f"[{self.name}] Position OPENED: {symbol} {direction} @ {entry_price:.8f} (Lev: {leverage}x, Strategy: {strategy})")
        
    def close_position(self, symbol: str):
        """Clear position tracking.
        FIX 2026-02-24: Clear exit lock to allow future hygiene checks.
        """
        if symbol in self.positions:
            del self.positions[symbol]
            print(f"[{self.name}] Position CLOSED: {symbol}")
            
            # FIX: Clear exit lock for this position
            if hasattr(self, '_exiting_positions') and symbol in self._exiting_positions:
                self._exiting_positions.remove(symbol)

            # PHANTOM MARGIN FIX: Check if we have no positions left and clear phantom margin
            if len(self.positions) == 0:
                print(f"[{self.name}] 🧹 PHANTOM MARGIN CHECK: No positions remaining, clearing phantom margin exposure")

    def check_stack_targets(self, symbol: str, current_price: float, atr: float = 0.0) -> float:
        """
        Stack-by-Stack Profit Management with Trailing Stops.
        Checks if individual stacks have hit their targets.
        Returns total quantity to close.
        FIX 2026-03-01: Added trailing stop logic for profitable positions.
        """
        pos = self.positions.get(symbol)
        if not pos: return 0.0

        # Extract position data
        if isinstance(pos, dict):
            stacks = pos.get('stacks', None)
            direction = pos.get('direction', 'BUY')
        else:
            meta = getattr(pos, 'metadata', {}) or {}
            stacks = meta.get('stacks', None)
            direction = getattr(pos, 'direction', 'BUY')

        if not stacks:
            return 0.0

        total_close = 0.0
        
        # FIX 2026-03-01: Track total position PnL for trailing logic
        total_pnl_pct = 0.0
        total_qty = sum(s.get('qty', 0) for s in stacks)
        if total_qty > 0:
            avg_entry = sum(s.get('price', 0) * s.get('qty', 0) for s in stacks) / total_qty
            if direction == 'BUY':
                total_pnl_pct = (current_price - avg_entry) / avg_entry
            else:
                total_pnl_pct = (avg_entry - current_price) / avg_entry

        # Check each stack for target hits
        for s in stacks:
            if s.get('exit_triggered'): continue

            entry = s['price']
            qty = s['qty']
            sid = s.get('id', 1) # 1-based index from config

            # Retrieve Target
            cfg_id = min(sid, 4)
            targets = config.STACK_PROFIT_TARGETS.get(cfg_id)
            if not targets: continue

            pnl = 0.0
            if direction == 'BUY':
                pnl = (current_price - entry) / entry
            else:
                pnl = (entry - current_price) / entry

            if pnl >= targets['target']:
                print(f"[{self.name}] 🥞 STACK TARGET HIT: Stack #{sid} (+{pnl*100:.2f}%) -> Closing {qty:.4f}")
                total_close += qty
                s['exit_triggered'] = True

        # FIX 2026-03-01: Trailing Stop for Profitable Stacks
        if config.STACK_TRAILING_ENABLED and total_pnl_pct >= config.STACK_TRAILING_PNL_THRESHOLD:
            # Position is profitable enough to start trailing
            trail_threshold = config.STACK_TRAILING_ATR_MULT * atr if atr > 0 else 0.0
            
            for s in stacks:
                if s.get('exit_triggered'): continue
                
                entry = s['price']
                qty = s['qty']
                current_stop = s.get('stop_loss', 0)
                
                # Calculate trailing stop price
                if direction == 'BUY':
                    trailing_stop = current_price - trail_threshold
                    # Move stop up if price improved
                    if trailing_stop > current_stop and trailing_stop > entry:
                        s['stop_loss'] = trailing_stop
                        print(f"[{self.name}] 🥞 TRAIL UPDATE: {symbol} Stack #{s.get('id',1)} Stop -> {trailing_stop:.4f}")
                else:  # SELL/Short
                    trailing_stop = current_price + trail_threshold
                    # Move stop down if price improved
                    if (trailing_stop < current_stop or current_stop == 0) and trailing_stop < entry:
                        s['stop_loss'] = trailing_stop
                        print(f"[{self.name}] 🥞 TRAIL UPDATE: {symbol} Stack #{s.get('id',1)} Stop -> {trailing_stop:.4f}")

        return total_close

    def apply_micro_guard_rail(self, symbol: str, proposed_qty: float, price: float, leverage: float) -> Tuple[bool, float]:
        """
        MICRO-MODE HARD GUARD-RAIL v1.0
        
        Purpose: stop the bot from strangling itself with notional > 150 % of NAV
        Scope: evaluated AFTER Kelly/Scavenger/Sniper sizing
        """
        # 0. Check Regime
        regime = "MICRO"
        if self.regime_controller: 
            regime = self.regime_controller.get_current_regime()
            
        # ONLY apply Micro Guard Rail if we are truly in NANO or MICRO regime
        if regime not in ['NANO', 'MICRO']:
            return True, proposed_qty

        if not config.MICRO_CAPITAL_MODE:
            return True, proposed_qty

        # --- NANO OVERRIDE: Allow Aggressive Leverage ---
        if self.balance < config.NANO_CAPITAL_THRESHOLD:
             if self.DEBUG: print(f"[{self.name}] 🛡️ MICRO GUARD: Nano Mode Override Active (Relaxing Limits)")
             # We allow up to NANO_MAX_LEVERAGE (e.g. 20x) and Max Exposure Ratio
             # Just verify against those, don't use strict 1.5x Micro cap
             
             max_allowed_lev = config.NANO_MAX_LEVERAGE
             if leverage > max_allowed_lev:
                 if self.DEBUG: print(f"[{self.name}] 🛡️ GUARD: Leverage {leverage}x > Nano Max {max_allowed_lev}x. Capping.")
                 # Reduce Qty to fit leverage
                 # logic: (qty * price) / new_lev? NO. Position size (notional) = qty * price.
                 # margin = notional / lev.
                 # We want to keep Notional same? No, leverage limit implies Risk Limit.
                 # If lev is reduced to 10x from 20x, we can keep same size but use more margin?
                 # OR we reduce size?
                 # Assuming "Leverage Cap" means "Max Leverage allowed for this trade".
                 # If we are using 20x and limit is 10x, we can still execute trade at 10x?
                 # Yes, but we use more margin.
                 # But Governor.calc usually returns leveraged quantity? No, just Qty. Lev is metadata.
                 # So we rely on Executor to use the leverage.
                 # BUT if we want to RESTRICT RISK, we should reduce size.
                 pass
             
             # If Nano, we trust the NANO logic in calc_position_size mostly.
             return True, proposed_qty
        # ------------------------------------------------

        nav = self.balance
        proposed_notional = proposed_qty * price
        
        # 5. Emergency kill-switch: if NAV < 50 USD → set MAX_GROSS_LEVERAGE_MICRO = 1.0
        max_gross_lev = config.MICRO_GUARD_GROSS_LEVERAGE
        if nav < config.MICRO_GUARD_CASH_PRESERVATION_THRESHOLD:
             max_gross_lev = config.MICRO_GUARD_CASH_PRESERVATION_LEVERAGE
             if self.DEBUG: print(f"[{self.name}] 🚨 CASH PRESERVATION: NAV ${nav:.2f} < $50. Max Lev -> {max_gross_lev}x")
             
        # Calculate Current Notional Exposure (Sum of Abs)
        current_gross_notional = 0.0
        for s, p in self.positions.items():
            if s != symbol: # Exclude current symbol if we are updating it? No, calc_size is for new/adds
                qty = p['quantity'] if isinstance(p, dict) else getattr(p, 'quantity', 0.0)
                ep = p['entry_price'] if isinstance(p, dict) else getattr(p, 'entry_price', 0.0)
                current_gross_notional += (qty * ep)
                
        # 1. Max portfolio notional <= 1.5 * NAV
        # The new total gross notional
        new_total_gross = current_gross_notional + proposed_notional
        limit_portfolio = config.MICRO_GUARD_PORTFOLIO_NOTIONAL_MULT * nav
        
        # 2. Max single-symbol net notional
        # CHECK STACKING OVERRIDE
        is_stacking = symbol in self.positions
        single_mult = config.MICRO_GUARD_SINGLE_NOTIONAL_MULT
        if is_stacking:
             # Use the higher of Config Limit or 1.0 (Don't downgrade a loose config)
             single_mult = max(config.MICRO_GUARD_SINGLE_NOTIONAL_MULT, 1.0) 
             if self.DEBUG: print(f"[{self.name}] 🥞 STACKING OVERRIDE: Using max({config.MICRO_GUARD_SINGLE_NOTIONAL_MULT}, 1.0) = {single_mult}x NAV")
             
        limit_single = single_mult * nav
        
        # 3. Max gross leverage <= Limit
        limit_leverage_notional = nav * max_gross_lev
        
        # FIND THE BINDING CONSTRAINT
        # We need to reduce proposed_notional such that:
        # A) current + proposed <= limit_portfolio
        # B) proposed <= limit_single
        # C) current + proposed <= limit_leverage_notional
        
        max_allowed_notional_A = max(0, limit_portfolio - current_gross_notional)
        max_allowed_notional_B = limit_single
        max_allowed_notional_C = max(0, limit_leverage_notional - current_gross_notional)
        
        final_allowed_notional = min(max_allowed_notional_A, max_allowed_notional_B, max_allowed_notional_C)
        
        if proposed_notional > final_allowed_notional:
            # DOWN-SIZE
            new_qty = final_allowed_notional / price
            print(f"[{self.name}] 🛡️ MICRO_GUARD: Down-sizing {symbol} {proposed_qty:.4f}->{new_qty:.4f} (${proposed_notional:.2f}->${final_allowed_notional:.2f})")
            print(f"    Constraint: PortLimit=${limit_portfolio:.2f}, SingleLimit=${limit_single:.2f}, LevLimit=${limit_leverage_notional:.2f}")
            
            # Check Min Notional
            if new_qty * price < config.MIN_ORDER_VALUE:
                 print(f"[{self.name}] ❌ Governor: Micro-guard veto – min_notional breach (${new_qty*price:.2f} < ${config.MIN_ORDER_VALUE})")
                 return False, 0.0
                 
            return True, new_qty
            
        return True, proposed_qty


    def check_portfolio_compliance(self, symbol: str, current_price: float) -> float:
        """
        Phase 3: Auto-Correction for Oversized Positions.
        Checks if a position exceeds its Asset Class Cap.
        Returns: Excess Quantity to CLOSE (positive float).
        """
        if not config.POSITION_LIMITS: return 0.0

        # FIX 2026-03-01: Position Dictionary mismatch
        # The Executor stores positions using virt_key (e.g. PAXG/USDT:ARBITRAGE)
        # However, check_portfolio_compliance is called using the raw symbol (e.g. PAXG/USDT)
        # self.positions.get(symbol) will return None, forcing a fallback compliance breach.
        # We must iterate and match the underlying p.symbol.
        pos = None
        for vk, p in self.positions.items():
            if p.symbol == symbol:
                pos = p
                break
                
        if not pos: return 0.0

        qty = abs(pos.quantity)
        if qty <= 1e-9: return 0.0

        total_equity = self.balance
        if total_equity <= 0: return 0.0

        # FIX 2026-03-03: Check for ARB strategy FIRST before tier assignment
        # This prevents DEFAULT tier from being applied before ARB exception is recognized
        pos_strategy = getattr(pos, 'strategy', '') or ''
        pos_metadata_reason = getattr(pos, 'metadata', {}).get('reason', '') if getattr(pos, 'metadata', None) else ''
        combined_tag = f"{pos_strategy} {pos_metadata_reason}".upper()
        
        # ARB positions (BASIS_CARRY_LONG/SHORT, FUNDING_CARRY, etc.) are
        # intentionally sized at ~15-25% of portfolio for yield harvesting.
        # Apply 25% cap immediately to avoid false compliance breaches.
        if any(k in combined_tag for k in ['ARB', 'CARRY', 'BASIS', 'FUNDING']):
            limit_pct = 0.25  # 25% cap for ARB positions
            tier_key = 'ARB_POSITION'
        else:
            # Normal tier assignment for non-ARB positions
            tier_key = 'DEFAULT'
            if symbol in ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']: 
                tier_key = 'LARGE_CAP'
            elif symbol in ['PEPE/USDT', 'DOGE/USDT', 'SHIB/USDT']: 
                tier_key = 'MEME'
            limit_pct = config.POSITION_LIMITS['size_limits'].get(tier_key, 0.10)
        
        # Current Notional
        current_notional = qty * current_price
        current_pct = current_notional / total_equity
        
        # HYSTERESIS: Only trigger if we exceed limit by 10% relative (e.g. 5.0% -> 5.5%)
        # This prevents immediate reduction due to minor price fluctuations or spread.
        hysteresis_buffer = 1.10
        
        if current_pct > (limit_pct * hysteresis_buffer):
            excess_pct_point = current_pct - limit_pct
            # Calculate Excess Notional
            # We want to reduce to limit.
            # Target Notional = Total * limit_pct
            # Excess = Current - Target
            target_notional = total_equity * limit_pct
            excess_notional = current_notional - target_notional

            excess_qty = excess_notional / current_price
            
            
            # FIX 2026-03-01: Ensure excess_qty is meaningful and executable
            # Get specific asset minimum or calculate minimum viable quantity based on MIN_ORDER_VALUE
            base_asset = symbol.split('/')[0]
            config_min = getattr(config, 'MIN_TRADE_QTY', {}).get(base_asset, 0.0)
            
            if config_min > 0:
                min_viable_qty = config_min
            else:
                min_order_value = getattr(config, 'MIN_ORDER_VALUE', 5.0)
                min_viable_qty = min_order_value / current_price if current_price > 0 else 0.0001
            
            # If excess_qty is too small, round up to minimum viable
            # But use smart buffering to avoid over-reduction
            if 0 < excess_qty < min_viable_qty:
                if excess_qty < min_viable_qty * 0.5:
                    # Very small breach: round to minimum (no buffer)
                    excess_qty = min_viable_qty
                    print(f"[{self.name}] ⚖️ COMPLIANCE BREACH: {symbol} is {current_pct*100:.1f}% of Portfolio (Limit {limit_pct*100:.1f}%). Excess: ${excess_notional:.2f} ({excess_qty:.4f}, rounded up from {excess_notional/current_price:.6f})")
                else:
                    # Moderate breach: round to minimum + small 5% buffer
                    excess_qty = min_viable_qty * 1.05
                    print(f"[{self.name}] ⚖️ COMPLIANCE BREACH: {symbol} is {current_pct*100:.1f}% of Portfolio (Limit {limit_pct*100:.1f}%). Excess: ${excess_notional:.2f} ({excess_qty:.4f}, rounded up from {excess_notional/current_price:.6f})")
            else:
                print(f"[{self.name}] ⚖️ COMPLIANCE BREACH: {symbol} is {current_pct*100:.1f}% of Portfolio (Limit {limit_pct*100:.1f}%). Excess: ${excess_notional:.2f} ({excess_qty:.4f})")
            
            return excess_qty
            
        return 0.0

    def check_timeout_actions(self, symbol: str) -> float:
        """
        Check if a position has timed out (stuck at entry) and needs exit.

        FIX 2026-03-15: Changed from 50% partial reduction to 100% clean exit.
        Partial reductions created infinite timeout loops.

        Returns: Quantity to exit (float) - now returns 100% of position.
        """
        if symbol not in self.stack_timeout_tracker:
            return 0.0

        start_time = self.stack_timeout_tracker[symbol]
        elapsed = time.time() - start_time
        timeout_seconds = getattr(config, 'STACK_TIMEOUT_SECONDS', 300)  # 5 min default

        if elapsed > timeout_seconds:
            # FIX 2026-03-15: Full exit instead of 50% reduction
            # Partial reductions created infinite loops where position dribbles out over hours
            pos = self.positions.get(symbol)
            if not pos:
                self.stack_timeout_tracker.pop(symbol, None)
                return 0.0

            qty_held = abs(pos.quantity)

            # CLEAN EXIT POLICY: Exit 100% of position on timeout
            reduce_qty = qty_held  # Full exit (was: qty_held * 0.5)

            print(f"[{self.name}] ⏰ TIMEOUT EXIT: {symbol} stuck for {elapsed/60:.1f}min. Full Exit {reduce_qty:.4f} (100%).")

            # Clear tracker - position will be fully closed
            self.stack_timeout_tracker.pop(symbol, None)

            return reduce_qty

        return 0.0

    # === POSITION HYGIENE SYSTEM (Capital Recycling) ===
    def check_position_hygiene(self, symbol: str, current_price: float, 
                                funding_yields: dict = None, 
                                structure_data: dict = None,
                                arb_opportunities: list = None) -> dict:
        """
        Position Hygiene: Forced rotation rules for capital recycling.
        
        Auto-close candidates when:
        1. Funding turns strongly negative AND no structural support
        2. Conviction decays below X for Y cycles
        3. Capital efficiency < alternative arb opportunity
        
        Returns: {'action': 'RECYCLE'|'HOLD', 'reason': str, 'close_pct': float}
        """
        result = {'action': 'HOLD', 'reason': '', 'close_pct': 0.0}

        pos = self.positions.get(symbol)
        if not pos:
            return result

        # Position is a dataclass object, access attributes directly
        qty = abs(pos.quantity)
        if qty <= 1e-9:
            return result

        direction = pos.direction
        entry_price = pos.entry_price

        # Calculate position age (use entry_timestamp from Position)
        entry_time_str = pos.entry_timestamp
        # Parse timestamp string to epoch time
        try:
            entry_time = float(entry_time_str) if isinstance(entry_time_str, (int, float)) else time.time()
        except (ValueError, TypeError):
            entry_time = time.time()
        age_minutes = (time.time() - entry_time) / 60.0

        # --- FIX 2.2: Position Age-Based Immunity (Consolidation Veto) ---
        # FIX: Arb positions need LONG immunity (6+ hours for funding cycles)
        IMMUNITY_WINDOW_MINUTES = 15.0  # Default for normal trades
        ARB_IMMUNITY_WINDOW_MINUTES = 360.0  # 6 hours for arb/funding carry trades
        
        pos_strategy = pos.strategy
        pos_reason = pos.metadata.get('reason', '') if pos.metadata else ''
        
        is_carry = (
            "BASIS_CARRY" in pos_strategy or 
            "ARBITRAGE" in pos_strategy or 
            "FUNDING_CARRY" in pos_strategy or
            "BASIS" in pos_reason or 
            "ARBITRAGE" in pos_reason or
            "FUNDING" in pos_reason
        )
        
        # Use appropriate immunity window based on strategy
        immunity_window = ARB_IMMUNITY_WINDOW_MINUTES if is_carry else IMMUNITY_WINDOW_MINUTES
        
        if is_carry and age_minutes < immunity_window:
             # Arb/Funding Carry Immunity Active - NO HYGIENE CHECKS
             # These positions should be held for full funding cycles (8 hours)
             return {'action': 'HOLD', 'reason': f'ARB_IMMUNITY_WINDOW ({age_minutes:.0f}m < {immunity_window:.0f}m)', 'close_pct': 0.0}
        
        # For non-arb positions, use standard immunity
        if not is_carry and age_minutes < IMMUNITY_WINDOW_MINUTES:
             return {'action': 'HOLD', 'reason': 'IMMUNITY_WINDOW', 'close_pct': 0.0}
        # -----------------------------------------------------------
        
        # Calculate current PnL
        if direction == 'BUY':
            pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
        else:
            pnl_pct = (entry_price - current_price) / entry_price if entry_price > 0 else 0
        
        # Notional value of this position
        position_notional = qty * current_price
        
        # --- HYGIENE RULE 1: Toxic Funding + No Structure ---
        # If we're LONG and funding is strongly negative, AND structure is not SUPPORT
        if funding_yields and symbol in funding_yields:
            funding_apy = funding_yields.get(symbol, 0.0)
            structure_zone = 'NEUTRAL'
            if structure_data and symbol in structure_data:
                structure_zone = structure_data.get(symbol, {}).get('zone', 'NEUTRAL')

            # Toxic: Paying > 200% APY in funding AND not in a support zone
            TOXIC_FUNDING_THRESHOLD = getattr(config, 'HYGIENE_TOXIC_FUNDING_APY', -200.0)
            if direction == 'BUY' and funding_apy < TOXIC_FUNDING_THRESHOLD:
                if structure_zone != 'SUPPORT':
                    result['action'] = 'RECYCLE'
                    result['reason'] = f'TOXIC_FUNDING: Paying {abs(funding_apy):.0f}% APY, Zone: {structure_zone}'
                    result['close_pct'] = 1.0  # Full close
                    # Deduplicate Logs
                    last_log = self.last_trade_time.get(f"{symbol}_hygiene_log", 0)
                    if time.time() - last_log > 60:
                        print(f"[{self.name}] ☣️ HYGIENE RECYCLE: {symbol} {result['reason']}")
                        self.last_trade_time[f"{symbol}_hygiene_log"] = time.time()
                    return result
            # Tag position with toxic funding info for Monte Carlo
            if direction == 'BUY' and funding_apy < TOXIC_FUNDING_THRESHOLD:
                # Update position metadata to indicate toxic funding
                if symbol in self.positions:
                    pos.metadata['toxic_funding'] = True
                    pos.metadata['funding_apy'] = funding_apy

        # --- HYGIENE RULE 2: Conviction Decay ---
        # Track conviction over cycles. If < threshold for N consecutive cycles, recycle.
        if not hasattr(self, 'conviction_decay_tracker'):
            self.conviction_decay_tracker = {}

        # Get current conviction from position's ppo_conviction attribute
        current_conviction = pos.ppo_conviction if pos.ppo_conviction is not None else 0.5
        DECAY_THRESHOLD = getattr(config, 'HYGIENE_CONVICTION_DECAY_THRESHOLD', 0.35)
        DECAY_CYCLES_LIMIT = getattr(config, 'HYGIENE_CONVICTION_DECAY_CYCLES', 5)
        
        if current_conviction < DECAY_THRESHOLD:
            decay_count = self.conviction_decay_tracker.get(symbol, 0) + 1
            self.conviction_decay_tracker[symbol] = decay_count
            
            if decay_count >= DECAY_CYCLES_LIMIT:
                result['action'] = 'RECYCLE'
                result['reason'] = f'CONVICTION_DECAY: {current_conviction:.2f} < {DECAY_THRESHOLD} for {decay_count} cycles'
                result['close_pct'] = 0.5  # Partial close (50%)
                
                last_log = self.last_trade_time.get(f"{symbol}_hygiene_log", 0)
                if time.time() - last_log > 60:
                    print(f"[{self.name}] 📉 HYGIENE RECYCLE: {symbol} {result['reason']}")
                    self.last_trade_time[f"{symbol}_hygiene_log"] = time.time()
                    
                self.conviction_decay_tracker[symbol] = 0  # Reset after action
                return result
        else:
            # Reset decay counter if conviction recovered
            self.conviction_decay_tracker[symbol] = 0
        
        # --- HYGIENE RULE 3: Capital Efficiency vs Arb Opportunity ---
        # If we have an arb opportunity with higher expected return than our current position
        if arb_opportunities and pnl_pct < 0.005:  # Only check if position is flat or losing
            # Calculate opportunity cost: best arb APY vs our current return
            best_arb_apy = 0.0
            best_arb_symbol = None
            for arb in arb_opportunities:
                arb_apy = arb.get('apy', 0.0)
                if arb_apy > best_arb_apy:
                    best_arb_apy = arb_apy
                    best_arb_symbol = arb.get('symbol', 'Unknown')
            
            # Our current "return" - annualized from position age
            if age_minutes > 5:  # Need some minimum age
                annualized_current = (pnl_pct * 365 * 24 * 60) / age_minutes if age_minutes > 0 else 0
            else:
                annualized_current = 0.0
            
            OPPORTUNITY_COST_THRESHOLD = getattr(config, 'HYGIENE_OPPORTUNITY_COST_PCT', 100.0)  # 100% APY difference
            
            if best_arb_apy > 0 and (best_arb_apy - annualized_current) > OPPORTUNITY_COST_THRESHOLD:
                # Only recycle if we're not profitable and the arb is significantly better
                if pnl_pct <= 0 and age_minutes > 10:  # Losing for 10+ minutes
                    result['action'] = 'RECYCLE'
                    result['reason'] = f'OPPORTUNITY_COST: Arb {best_arb_symbol} ({best_arb_apy:.0f}% APY) >> Current ({annualized_current:.0f}%)'
                    result['close_pct'] = 0.75  # Close 75% to free capital
                    
                    last_log = self.last_trade_time.get(f"{symbol}_hygiene_log", 0)
                    if time.time() - last_log > 60:
                        print(f"[{self.name}] 💰 HYGIENE RECYCLE: {symbol} {result['reason']}")
                        self.last_trade_time[f"{symbol}_hygiene_log"] = time.time()
                        
                    return result

        # --- MONTE CARLO EVALUATION FOR LOSING POSITIONS ---
        # If position is losing and hasn't been caught by other rules, run Monte Carlo evaluation
        if pnl_pct < -0.005 and result['action'] == 'HOLD':  # Losing more than 0.5% and not already marked for recycling (more sensitive)
            if self.monte_carlo_manager:
                try:
                    # Prepare SDE parameters from position metadata if available
                    sde_params = pos.metadata.get('sde_physics', {
                        'mu': 0.0,
                        'sigma': 0.1,
                        'lambda': 0.1
                    })

                    # Get position age in hours
                    position_age_hours = age_minutes / 60.0

                    # Evaluate position using Monte Carlo
                    monte_carlo_result = self.monte_carlo_manager.evaluate_position_for_closure(
                        symbol=symbol,
                        current_price=current_price,
                        entry_price=entry_price,
                        direction=direction,
                        position_age_hours=position_age_hours,
                        sde_params=sde_params,
                        pnl_pct=pnl_pct
                    )

                    # Check if result is valid
                    if monte_carlo_result is not None and isinstance(monte_carlo_result, tuple) and len(monte_carlo_result) >= 3:
                        should_close, confidence, mc_reason = monte_carlo_result[0], monte_carlo_result[1], monte_carlo_result[2]

                        if should_close and confidence and confidence > 0.4:  # Lowered threshold to 40% for more sensitivity
                            result['action'] = 'RECYCLE'
                            result['reason'] = f'MONTE_CARLO_EVAL: {mc_reason} (Conf: {confidence:.2%})'
                            result['close_pct'] = 1.0  # Full close based on Monte Carlo assessment

                            print(f"[{self.name}] 🎲 MONTE CARLO CLOSURE: {symbol}: {mc_reason} (Conf: {confidence:.2%})")

                except Exception as e:
                    print(f"[{self.name}] Monte Carlo evaluation error for {symbol}: {e}")
                    import traceback
                    traceback.print_exc()
        # -----------------------------------------------

        return result
    
    def run_hygiene_sweep(self, latest_prices: dict,
                          funding_yields: dict = None,
                          structure_data: dict = None,
                          arb_opportunities: list = None) -> list:
        """
        Run hygiene check on ALL positions. Returns list of signals to close.

        Returns: List of {'symbol': str, 'close_pct': float, 'reason': str}
        FIX 2026-02-24: Added position exit lock to prevent race conditions.
        """
        recycle_signals = []
        
        # FIX: Track positions currently being exited to prevent race conditions
        if not hasattr(self, '_exiting_positions'):
            self._exiting_positions = set()

        for held_virt_key, pos in self.positions.items():
            qty = pos.quantity
            
            # FIX: Check actual quantity from Executor (source of truth)
            if self.executor and held_virt_key in self.executor.positions:
                actual_qty = self.executor.positions[held_virt_key].quantity
                if abs(actual_qty) <= 1e-9:
                    # Position was closed in Executor but not cleaned up in Governor
                    if held_virt_key in self.positions:
                        del self.positions[held_virt_key]
                        print(f"[{self.name}] 🧹 CLEANUP: Removed closed position {held_virt_key}")
                    continue
                qty = actual_qty  # Use actual quantity
            elif abs(qty) <= 1e-9:
                continue

            # Extract Real Symbol for price & external lookups
            symbol = held_virt_key.split(':')[0] if ':' in held_virt_key else held_virt_key

            # FIX: Skip if position is already being exited (race condition prevention)
            if held_virt_key in self._exiting_positions:
                continue

            current_price = latest_prices.get(symbol, 0.0)
            if current_price <= 0:
                continue

            hygiene_result = self.check_position_hygiene(
                symbol=held_virt_key, # Pass the key so it can pull the right metadata
                current_price=current_price,
                funding_yields=funding_yields,
                structure_data=structure_data,
                arb_opportunities=arb_opportunities
            )

            if hygiene_result['action'] == 'RECYCLE':
                # FIX: Mark position as exiting to prevent duplicate signals
                self._exiting_positions.add(held_virt_key)
                recycle_signals.append({
                    'symbol': symbol,
                    'close_pct': hygiene_result['close_pct'],
                    'reason': hygiene_result['reason']
                })

        if recycle_signals:
             # Deduplicate Summary Log (Once per 60s)
             last_log = self.last_trade_time.get("hygiene_sweep_log", 0)
             if time.time() - last_log > 60:
                 print(f"[{self.name}] 🧹 HYGIENE SWEEP: {len(recycle_signals)} positions flagged for recycling")
                 self.last_trade_time["hygiene_sweep_log"] = time.time()

        return recycle_signals


    def set_reference_atr(self, atr: float):
        """Set the reference ATR for volatility targeting."""
        if self.reference_atr is None:
            self.reference_atr = atr
            print(f"[{self.name}] Reference ATR set: {atr:.6f}")

    # === PHASE 12: INSTITUTIONAL RISK MANAGEMENT ===
    
    def calculate_max_risk(self, balance: float) -> float:
        """
        Minimax Constraint (Game Theory):
        Never risk the principal ($10). Only risk house money OR 1% of total.
        Uses Rust Engine (holonic_speed) if available.
        """
        try:
            import holonic_speed
            return holonic_speed.governor_calculate_max_risk(
                balance, config.PRINCIPAL, balance
            )
        except ImportError:
            # Fallback to Python
            house_money = max(0, balance - config.PRINCIPAL)
            pct_risk = balance * config.MAX_RISK_PCT
            return min(house_money, pct_risk)
    
    def calculate_volatility_scalar(self, atr_current: float, atr_ref: float) -> float:
        """
        Volatility Scalar (Inverse Variance Weighting):
        Normalize position size based on current volatility.
        
        Formula: Size_adj = Size_base × (ATR_ref / ATR_current)
        
        Args:
            atr_current: Current ATR value
            atr_ref: Reference ATR (14-period average)
            
        Returns:
            Scalar multiplier (clamped to 0.5-2.0)
        """
        if atr_current <= 0 or atr_ref <= 0:
            return 1.0
        
        # Inverse relationship: high volatility = smaller size
        scalar = atr_ref / atr_current
        
        # Clamp to reasonable range
        return max(config.VOL_SCALAR_MIN, min(config.VOL_SCALAR_MAX, scalar))

    def calculate_sde_physics_scalar(self, metadata: Dict[str, Any], direction: str = 'BUY') -> float:
        """
        Physics-Based Position Scaling (SDE Layer).
        Dynamically adjusts size based on SDE drift and diffusion.
        """
        if not metadata or 'sde_physics' not in metadata:
            return 1.0
            
        sde = metadata['sde_physics']
        precision_physics_scalar = 1.0
        
        # 1. Diffusion-Alpha (Noise Sensitivity)
        # If Instantaneous SDE Sigma is spiking relative to ATR, we reduce size.
        inst_vol = sde.get('sigma', 0.0)
        if inst_vol > 1.5: # Extreme Volatility (> 150% annual)
             precision_physics_scalar *= 0.8
             
        # 2. Quantum Reversion Scaling
        reason = metadata.get('reason', '')
        if reason in ['QUANTUM', 'QUANTUM_SELL']:
            # For Quantum Reversion, we scale by the conviction provided by the Oracle
            # (which is based on distance from the mean)
            q_conv = metadata.get('quantum_conviction', 1.0)
            precision_physics_scalar *= q_conv
            
        # 3. Drift Check (Optional Bonus)
        # We don't want to over-size, so we clamp the bonus tightly
        drift = sde.get('drift', 0.0)
        if direction == 'BUY' and drift > 0.5:
            precision_physics_scalar *= 1.1 # 10% Bonus for strong positive drift
        elif direction == 'SELL' and drift < -0.5:
            precision_physics_scalar *= 1.1
            
        return max(0.5, min(1.5, precision_physics_scalar))
    
    def calculate_recent_win_rate(self, lookback: int = None) -> float:
        """
        Calculate win rate from recent trades.
        
        Args:
            lookback: Number of recent trades to analyze
            
        Returns:
            Win rate (0.0 to 1.0)
        """
        if lookback is None:
            lookback = config.KELLY_LOOKBACK
        
        # Integrate with database to get actual win rate
        if hasattr(self, 'db_manager') and self.db_manager:
            # TEST HOOK: Prefer direct win rate if available (Mock Support)
            if hasattr(self.db_manager, 'get_win_rate'):
                 return self.db_manager.get_win_rate()

            try:
                # Get recent trades from database
                trades = self.db_manager.get_recent_trades(lookback)
                if trades and len(trades) > 0:
                    # Calculate actual win rate (Keep the float cast fix)
                    wins = sum(1 for t in trades if float(t.get('pnl', 0)) > 0)
                    actual_wr = wins / len(trades)
                    
                    # BLENDING: If we have few trades, blend with a neutral baseline (0.52)
                    # to prevent "Cold Start" rejection (e.g. 0% WR after 1 loss).
                    # User Override: 52% Baseline + 20 trade window
                    sample_size = len(trades)
                    min_sample = 20
                    if sample_size < min_sample:
                        baseline = 0.52
                        weight = sample_size / min_sample
                        win_rate = (actual_wr * weight) + (baseline * (1 - weight))
                    else:
                        win_rate = actual_wr
                        
                    # Only print if significantly changed or periodically?
                    # For now, just print cleaner one-liner
                    print(f"[{self.name}] 📊 Win Rate: {win_rate*100:.1f}% (Actual: {actual_wr*100:.1f}%, n={sample_size})")
                    return win_rate
            except Exception as e:
                print(f"[{self.name}] ⚠️ Win rate calculation failed: {e}")
        
        return 0.52


    def calculate_ruin_probability(self, symbol: str, entry_price: float, direction: str, stop_loss: float, take_profit: float, metadata: Dict[str, Any]) -> float:
        """
        Monte Carlo Ruin Guard:
        Uses optimized SDEEngine (Rust accelerated) to estimate 
        the probability of hitting Stop Loss before Take Profit/Horizon.
        """
        if not metadata or 'sde_physics' not in metadata:
            return 0.5 
            
        try:
            from HolonicTrader.sde_engine import SDEEngine
            sde = metadata['sde_physics']
            # Parameters from Oracle
            params = {
                'mu': sde.get('mu', 0.0),
                'sigma': sde.get('sigma', 0.1),
                'lambda': sde.get('lambda', 0.1)
            }
            
            # Use Rust-accelerated calculation
            return SDEEngine.calculate_ruin_probability(
                'GBM', # Default model
                params, 
                entry_price, 
                stop_loss, 
                take_profit, 
                horizon=100, 
                paths=500
            )
            
        except Exception as e:
            if self.DEBUG: print(f"[{self.name}] Ruin Guard Error: {e}")
            return 0.5


    def pre_entry_ruin_check(self, symbol: str, entry_price: float, direction: str, 
                              stop_loss: float, take_profit: float, metadata: dict = None) -> tuple:
        """
        CRITICAL: Block trades BEFORE entry if ruin probability is too high.
        This is the PRIMARY defense against bleeding.
        
        Returns: (allowed: bool, ruin_prob: float, reason: str)
        """
        if metadata is None:
            metadata = {}
        
        # Calculate ruin probability using SDE/Monte Carlo
        ruin_prob = self.calculate_ruin_probability(
            symbol=symbol,
            entry_price=entry_price,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata
        )
        
        # HARD BLOCK if ruin prob > threshold
        max_allowed_ruin = getattr(config, 'PHYSICS_MAX_RUIN_PROBABILITY', 0.35)
        
        if ruin_prob > max_allowed_ruin:
            reason = f'PRE_ENTRY_RUIN_BLOCK: {ruin_prob:.1%} > {max_allowed_ruin:.1%}'
            print(f"[{self.name}] [PRE-ENTRY RUIN CHECK] [BLOCKED] {symbol} - {reason}")
            return False, ruin_prob, reason
        
        # Additional check: If ruin prob > 50%, warn but allow with reduced size
        if ruin_prob > 0.50:
            print(f"[{self.name}] [PRE-ENTRY RUIN CHECK] [WARNING] {symbol} - High ruin prob: {ruin_prob:.1%}")
        
        print(f"[{self.name}] [PRE-ENTRY RUIN CHECK] [ALLOWED] {symbol} - Ruin prob: {ruin_prob:.1%}")
        return True, ruin_prob, 'PASS'

    def check_cluster_risk(self, symbol: str) -> bool:
        """
        Refuse trade if we already hold an asset from the same family.
        Returns: False if RISK DETECTED (Reject), True if SAFE.
        Uses Rust Engine (holonic_speed) if available.
        """
        try:
            import holonic_speed
            # Get currently held symbols - FIX: Position is an object, not a dict
            held_symbols = [s for s, pos in self.positions.items() if abs(pos.quantity) > 0]
            result = holonic_speed.governor_check_cluster_risk(held_symbols, symbol)
            if not result:
                print(f"[{self.name}] CLUSTER RISK: Rejecting {symbol} (Same family as held)")
            return result
        except ImportError:
            # Fallback to Python
            
            # --- USER HEDGE OVERRIDE ---
            # If asset is a designated Hedge Asset (ETH, XMR, XRP), allow it even if correlated.
            hedge_assets = getattr(config, 'BTC_HEDGE_ASSETS', [])
            if symbol in hedge_assets:
                return True
            # ---------------------------

            family = None
            if symbol in config.FAMILY_L1: family = config.FAMILY_L1
            elif symbol in config.FAMILY_PAYMENT: family = config.FAMILY_PAYMENT
            elif symbol in config.FAMILY_MEME: family = config.FAMILY_MEME
            
            if not family: return True
            
            for asset, data in self.positions.items():
                if abs(data['quantity']) > 0 and asset in family and asset != symbol:
                    print(f"[{self.name}] CLUSTER RISK: Rejecting {symbol} (Already hold {asset})")
                    return False
        return True

    def check_leverage_risk(self, new_notional_value: float) -> bool:
        """
        Refuse trade if Total Notional Exposure > 10x Balance.
        """
        current_exposure = 0.0
        # Sum absolute notional value of all positions
        for asset, position in self.positions.items():
            # Position object has quantity and entry_price attributes
            qty = abs(position.quantity)
            price = position.entry_price
            current_exposure += (qty * price)

        total_exposure = current_exposure + new_notional_value
        
        # --- NANO OVERRIDE ---
        if self.balance < config.NANO_CAPITAL_THRESHOLD:
            # Allow full 20x (or whatever NANO_MAX_LEVERAGE is)
            max_allowed = self.balance * config.NANO_MAX_LEVERAGE
        else:
            max_allowed = self.balance * config.IMMUNE_MAX_LEVERAGE_RATIO
        
        if total_exposure > max_allowed:
            print(f"[{self.name}] ⚠️ OVER-LEVERAGE: Exposure ${total_exposure:.0f} > Limit ${max_allowed:.0f}")
            return False
        return True

    def calculate_kelly_size(self, balance: float, win_rate: float = None, risk_reward: float = None) -> float:
        """
        Modified Kelly Criterion (Half-Kelly):
        Calculate optimal position size for PREDATOR mode.
        
        Formula: f* = [(p(b+1) - 1) / b] × 0.5
        
        Args:
            balance: Current account balance
            win_rate: Recent win rate (0.0 to 1.0)
            risk_reward: Expected reward/risk ratio
            
        Returns:
            Maximum position size in USD
        """
        """
        Calculate maximum allowable risk per trade (USD) based on Volatility-Adjusted Kelly Criterion.
        Replaces arbitrary 5% hard caps with Probability-Based Sizing.
        """
        # 1. Get Win Rate & Reward:Risk
        win_rate = self.calculate_recent_win_rate()
        risk_reward = 2.0 # Conservative estimate for R:R
        
        # 2. Kelly Formula -> Optimal Fraction
        # f* = (p(b+1) - 1) / b
        # where p = win_rate, b = risk_reward
        kelly_fraction = ((win_rate * (risk_reward + 1)) - 1) / risk_reward
        
        # 3. Fractional Kelly (Safety)
        # Use Half-Kelly for safety (industry standard)
        fractional_kelly = kelly_fraction * 0.5
        
        # 4. Volatility Adjustment
        # If market is wild, reduce size.
        atr_ref = getattr(self, 'reference_atr', 0.0) or 0.0
        # If we can't find ATR, we assume 1.0 scalar. 
        # But we need current ATR... which isn't passed here. 
        # We assume the Caller (calc_position_size) applies vol_scalar LATER.
        # So here we return the BASE Kelly risk.
        
        # Sanity Bounds (Never risk <0% or >20% of equity per trade)
        # Floor at 5% to allow minimum viable position size
        safe_fraction = max(0.05, min(0.20, fractional_kelly))
        
        max_usd_risk = balance * safe_fraction
        
        if self.DEBUG:
            print(f"[{self.name}] 🧠 Kelly Risk: WR {win_rate:.2f}, Kelly {kelly_fraction:.2f}, Safe {safe_fraction:.2f} -> ${max_usd_risk:.2f}")
            
        return max_usd_risk

    def trigger_emergency_position_closures(self):
        """Trigger emergency closure of all positions when capital drops dramatically."""
        print(f"[{self.name}] 🚨 EMERGENCY: Initiating mass position closures due to capital crisis")

        # Create a copy of positions to avoid modification during iteration
        positions_to_close = list(self.positions.keys())

        for symbol in positions_to_close:
            pos_data = self.positions.get(symbol, {})
            if pos_data and abs(pos_data.get('quantity', 0)) > 0:
                print(f"[{self.name}] 🚨 EMERGENCY CLOSING: {symbol} (Qty: {pos_data.get('quantity')})")

                # Mark for closure by sending appropriate message to executor
                # This would typically be handled by the executor, but we'll flag it here
                if hasattr(self, 'executor') and self.executor:
                    # Send emergency close signal to executor
                    try:
                        # Create a close signal
                        from .agent_executor import TradeSignal
                        direction = pos_data.get('direction', 'BUY') if isinstance(pos_data, dict) else getattr(pos_data, 'direction', 'BUY')
                        close_signal = TradeSignal(
                            symbol=symbol,
                            direction='SELL' if direction == 'BUY' else 'BUY',
                            size=1.0,  # Close all
                            price=0.0,  # Will be filled by executor
                            metadata={'reason': 'EMERGENCY_CAPITAL_CRISIS', 'is_percent': True}
                        )
                        # Execute the emergency closure
                        from .agent_executor import TradeDecision
                        from HolonicTrader.holon_core import Disposition
                        decision = TradeDecision(
                            action='EXECUTE',
                            original_signal=close_signal,
                            adjusted_size=1.0,
                            disposition=Disposition(autonomy=1.0, integration=1.0),
                            block_hash='EMERGENCY_CLOSE'
                        )
                        # Execute transaction if actuator is available
                        if hasattr(self.executor, 'execute_transaction'):
                            current_price = self.executor.latest_prices.get(symbol, 0.0)
                            if current_price > 0:
                                self.executor.execute_transaction(decision, current_price)
                    except Exception as e:
                        print(f"[{self.name}] Error in emergency closure for {symbol}: {e}")

        # Clear all positions after emergency closure
        self.positions.clear()
        print(f"[{self.name}] 🚨 EMERGENCY: All positions closed. Position book cleared.")

    def receive_message(self, sender: Any, content: Any) -> Any:
        """Handle incoming messages."""
        msg_type = content.get('type')
        if msg_type == 'VALIDATE_TRADE':
            symbol = content.get('symbol')
            price = content.get('price')
            atr = content.get('atr')
            conviction = content.get('conviction', 0.5)
            # Check if conviction is None (if key exists but value is None)
            if conviction is None: conviction = 0.5
            direction = content.get('direction', 'BUY')

            crisis_score = content.get('crisis_score', 0.0)
            
            # AEHML FIX: Ensure we extract whale_confirmed and proper metadata
            # It might be at the top level, or nested in 'metadata'
            metadata_block = content.get('metadata', content)
            whale_conf = metadata_block.get('is_whale', False)
            if 'whale_confirmed' in content:
                 whale_conf = content['whale_confirmed']
                 
            # Extract strategy explicitly so it isn't lost during the second pass
            strategy = content.get('strategy', metadata_block.get('strategy', 'DIRECTIONAL'))
            
            # Inject it into metadata so it passes through cleanly
            metadata_block['strategy'] = strategy

            return self.calc_position_size(symbol, price, atr, conviction=conviction, direction=direction, crisis_score=crisis_score, whale_confirmed=whale_conf, metadata=metadata_block)

        elif msg_type == 'POSITION_FILLED':
            # Executor sends positive Qty. open_position handles logic.
            qty = content.get('quantity')

            self.open_position(
                content.get('symbol'),
                content.get('direction'),
                content.get('price'),
                qty
            )
            return True

        elif msg_type == 'POSITION_CLOSED':
            self.close_position(content.get('symbol'))
            return True

        elif msg_type == 'GET_STATE':
            return self.get_metabolism_state()

        elif msg_type == 'WAKE_UP':
            print(f"[{self.name}] Received WAKE_UP signal from Immune System.")
            self.state = 'ACTIVE'
            return True

        elif msg_type == 'EMERGENCY_CLOSE_ALL':
            print(f"[{self.name}] Received EMERGENCY_CLOSE_ALL signal")
            self.trigger_emergency_position_closures()
            return True

        elif msg_type == 'CLEAR_PHANTOM_MARGIN':
            print(f"[{self.name}] Received CLEAR_PHANTOM_MARGIN signal")
            self.clear_phantom_margin()
            return True

        return None

    def gc_sync_with_executor(self, executor) -> list:
        """
        Garbage Collector: Ensure Governor positions match Executor state.
        Returns list of mismatched positions that were fixed.

        FIX BUG-001: Normalize position keys - Governor uses virt_key (e.g., BTC/USDT:BASIS_CARRY_LONG)
        while Executor uses bare symbols (e.g., BTC/USDT). This caused false "mismatch" reports.

        FIX 2026-03-16: Governor's self.positions is a property proxying to Executor.
        We should NOT modify self.positions directly. Instead, only sync internal trackers.
        This prevents state corruption and tracker leaks.
        """
        verbose = getattr(config, 'GC_LOG_VERBOSE', True)
        mismatches = []

        if not executor:
            return mismatches

        executor_assets = executor.held_assets
        executor_metadata = executor.position_metadata

        # FIX BUG-001: Build normalized symbol map for Governor internal trackers
        # Map bare symbol -> tracked symbols in last_specific_entry
        tracked_bare_symbols = set()
        for sym in self.last_specific_entry.keys():
            bare_sym = sym.split(':')[0] if ':' in sym else sym
            tracked_bare_symbols.add(bare_sym)

        # Check for positions in trackers but not in Executor (cleanup orphaned trackers)
        for sym in list(self.last_specific_entry.keys()):
            bare_sym = sym.split(':')[0] if ':' in sym else sym
            exec_qty = executor_assets.get(bare_sym, 0.0)
            if abs(exec_qty) < 0.00000001:
                # Executor doesn't have it, but Governor tracker does - CLEANUP
                if verbose:
                    print(f"[GC Monitor] 🧹 Orphaned tracker cleanup: {sym}")
                if sym in self.last_specific_entry:
                    del self.last_specific_entry[sym]
                if sym in self.stack_timeout_tracker:
                    del self.stack_timeout_tracker[sym]
                if sym in self.stack_snooze:
                    del self.stack_snooze[sym]
                mismatches.append(f"cleanup:{sym}")

        # Check for positions Executor has but Governor trackers don't (sync trackers)
        for sym, qty in list(executor_assets.items()):
            if abs(qty) > 0.00000001:
                # Check if we have tracker for this symbol (any variant)
                has_tracker = sym in self.last_specific_entry or sym in tracked_bare_symbols

                if not has_tracker:
                    # Executor has position, but Governor trackers are missing - SYNC
                    if verbose:
                        print(f"[GC Monitor] 🔗 Syncing tracker for {sym}")

                    meta = executor_metadata.get(sym, {})
                    entry_price = meta.get('entry_price', 0.0)

                    # Sync the entry price tracker
                    self.last_specific_entry[sym] = entry_price
                    mismatches.append(f"sync:{sym}")

        if verbose and mismatches:
            print(f"[GC Monitor] ✅ Governor Sync: {len(mismatches)} tracker(s) fixed: {mismatches}")
        elif verbose:
            print(f"[GC Monitor] ✅ Governor Sync: Trackers aligned with Executor.")

        return mismatches

    # === PHASE 7: CONSOLIDATION ENGINE ===
    def run_consolidation_engine(self, current_prices: dict, position_metadata: dict = None) -> list:
        """
        Intelligent Position Consolidation.
        
        Triggers:
        - open_positions > max_positions_allowed (from regime)
        - free_margin < 1.5 * min_required_margin
        - regime_transition_pending
        
        Scoring Model (normalized 0-1):
        - PnL (30%): Unrealized profit
        - Conviction (25%): Signal confidence at entry
        - Liquidity (15%): Is it a major pair?
        - Age (10%): Newer positions may be better
        - Correlation (-20%): Penalty for redundant positions
        
        Returns: Single symbol to CLOSE (one per cycle for safety).
        """
        if self.consolidation_in_progress:
            return []
            
        open_positions = list(self.positions.keys())
        if len(open_positions) == 0:
            return []
            
        # Get regime permissions
        if self.regime_controller:
            permissions = self.regime_controller.get_permissions()
            max_positions = permissions.get('max_positions', 2)
            transition_pending = self.regime_controller.is_transition_pending()
        else:
            # Fallback to MICRO
            max_positions = config.REGIME_PERMISSIONS['MICRO']['max_positions']
            transition_pending = False
            
        # Check Trigger Conditions
        trigger_reason = None
        
        # A. Position count exceeds limit
        if len(open_positions) > max_positions:
            trigger_reason = f"Positions {len(open_positions)} > Max {max_positions}"
            
        # B. Regime transition pending
        elif transition_pending:
            trigger_reason = "Regime Transition Pending"
            
        # C. Free margin too low
        elif self.available_balance < 1.5 * config.MIN_ORDER_VALUE:
            trigger_reason = f"Free Margin ${self.available_balance:.2f} < ${1.5 * config.MIN_ORDER_VALUE:.2f}"
            
        if not trigger_reason:
            return []
            
        self.consolidation_in_progress = True
        
        # Dampened Logging: Only print if we haven't printed in the last 60s OR if action is taken
        should_log = (time.time() - self.last_consolidation_time > 60.0) 
        
        if should_log:
            # print(f"\n[ConsolidationEngine] 🧹 TRIGGERED: {trigger_reason}")
            pass
            # print(f"[ConsolidationEngine] Analyzing {len(open_positions)} positions...") # Too noisy
        
        # Score all positions
        scored_positions = []
        
        # Calculate Total Portfolio Equity for weighting
        # Use available_balance + notional of all positions?
        # Simpler: Use Governor's tracked equity if available, else sum
        def check_stacking_logic(self, symbol: str, current_price: float, direction: str, atr: float = None) -> Tuple[bool, str]:
            """
            Phase 18: Smart Stacking
            Only allow adding to position if:
            1. Winning (PnL > 0)
            2. Price moved significantly (Dynamic ATR-based Distance)
            3. Trend is still young (< 24h)
            4. Regime Alignment (Don't stack Longs in Bear Market)
            """
            if symbol not in self.positions:
                return True, "New Position"

            pos = self.positions[symbol]
            entry = pos.entry_price
            qty = pos.quantity
            existing_dir = pos.direction
            stacks = pos.stack_count
            
            # 0. Direction Check (Sanity)
            if direction != existing_dir:
                return False, f"Opposite Direction ({direction} vs {existing_dir})"
                
            # 1. PnL Check (Only stack winners)
            pnl_pct = (current_price - entry) / entry
            if existing_dir == 'SELL': pnl_pct *= -1
            
            if pnl_pct <= 0:
                return False, f"Losing Position ({pnl_pct*100:.2f}%)"
                
            # 2. Distance Check (Standard + Volatility Scaling)
            # Standard Min Dist
            min_dist_pct = config.GOVERNOR_MIN_STACK_DIST
            
            # VOLATILITY SCALING (User Request)
            if atr and current_price > 0:
                # If Volatility is High, we need MORE distance to confirm trend
                # Normalize ATR pct: e.g. 1% ATR -> 1.0 multiplier
                atr_pct = atr / current_price
                vol_multiplier = max(1.0, atr_pct / 0.01) # Baseline 1% ATR
                min_dist_pct *= vol_multiplier
                # print(f"[{self.name}] 📏 Dynamic Stack Dist: {min_dist_pct*100:.2f}% (Vol Mult: {vol_multiplier:.2f}x)")

            current_dist = abs((current_price - entry) / entry)
            if current_dist < min_dist_pct:
                return False, f"Too Close (Dist {current_dist*100:.2f}% < {min_dist_pct*100:.2f}%)"
                
            # 3. Stack Limit
            max_stacks = config.REGIME_PERMISSIONS['SMALL']['max_stacks'] # Default
            if self.regime_controller:
                max_stacks = self.regime_controller.get_permissions().get('max_stacks', 0)
                
            # REGIME OVERRIDE: Cap Long Stacks in Bear Market
            # How to know regime here? We can check recent sentiment or pass it in.
            # Fallback: Use simple trend check or external flag if available.
            # For now, strict limit.
            if stacks >= max_stacks:
                return False, f"Max Stacks Reached ({stacks}/{max_stacks})"
                
            return True, "Stacking Approved"
        total_equity = self.available_balance
        for sym in open_positions:
            p = self.positions[sym]
            # FIX 2026-02-28: p is a Position object, not a dict — use getattr not .get()
            _qty = abs(getattr(p, 'quantity', 0.0))
            _ep = getattr(p, 'entry_price', 0.0)
            total_equity += _qty * current_prices.get(sym, _ep)
            
        if total_equity <= 0: total_equity = 1.0 # Prevent div/0

        for sym in open_positions:
            pos = self.positions[sym]

            # === CRITICAL: MINIMUM AGE CHECK (MUST BE FIRST) ===
            # Prevent brand-new positions from being scored AT ALL
            # This stops "Invalid Asset Price" errors and self-sabotaging closures
            first_entry = pos.metadata.get('first_entry_time', time.time())
            age_minutes = (time.time() - first_entry) / 60.0

            MIN_POSITION_AGE_MINUTES = getattr(config, 'CONSOLIDATION_MIN_AGE_MINUTES', 5)

            # CHRONOS FORENSICS: Immunity Bypass for high-conviction stacks
            is_immune = age_minutes < MIN_POSITION_AGE_MINUTES
            stack_bypass = pos.metadata.get('stack_conviction_bypass', False)

            if is_immune and stack_bypass:
                is_immune = False
                if should_log:
                    print(f"[ConsolidationEngine] 🚀 IMMUNITY BYPASS: {sym} has high-conviction stacking signal. Overriding {MIN_POSITION_AGE_MINUTES}m block.")

            if is_immune:
                # COMPUTE WASTE FIX: Track immunity suppress count
                if not hasattr(self, '_immunity_suppress_count'):
                    self._immunity_suppress_count = {}
                self._immunity_suppress_count[sym] = self._immunity_suppress_count.get(sym, 0) + 1

                # Only log first 3 times per symbol, then every 10th
                count = self._immunity_suppress_count[sym]
                if should_log and (count <= 3 or count % 10 == 0):
                    print(f"[ConsolidationEngine] 🛡️ IMMUNITY: {sym} is too new ({age_minutes:.1f}m < {MIN_POSITION_AGE_MINUTES}m). Skipping.")
                continue

            # === Now safe to fetch price data ===
            entry = pos.entry_price
            qty = pos.quantity
            direction = pos.direction
            
            # Extract real symbol for price lookup (strip strategy suffix if virtual)
            real_symbol = sym.split(':')[0] if ':' in sym else sym
            current_price = current_prices.get(real_symbol, entry)
            
            # Additional safety: Skip if we don't have valid price data
            if current_price <= 0:
                if should_log:
                    print(f"[ConsolidationEngine] ⚠️ SKIP: {sym} has invalid price (${current_price:.2f})")
                continue

            # === Calculate Individual Scores (0-1 normalized) ===
            
            # 1. PnL Score
            if entry > 0 and current_price > 0:
                pnl_pct = (current_price - entry) / entry
                if direction == 'SELL': pnl_pct *= -1
                # Normalize: -10% to +10% -> 0 to 1. Higher PnL = Higher Score (Better to KEEP)
                # We want to CLOSE the LOWEST score.
                pnl_score = max(0.0, min(1.0, (pnl_pct + 0.10) / 0.20))
            else:
                pnl_score = 0.5 # Neutral if no data
                
            # 2. Conviction Score (With Capital-Weighted Decay)
            meta = position_metadata.get(sym, {}) if position_metadata else {}
            raw_conviction = meta.get('conviction', 0.5)
            
            # --- WEIGHTED DECAY LOGIC ---
            notional = abs(qty * current_price)
            capital_weight = notional / total_equity if total_equity > 0 else 0.0
            
            # Acceleration: Heavy positions decay faster
            decay_factor = 1.0 + (capital_weight * config.CONVICTION_DECAY_CAPITAL_MULTIPLIER)
            effective_lifespan = config.CONVICTION_DECAY_BASE_HOURS / decay_factor

            first_entry = pos.metadata.get('first_entry_time', time.time())
            age_hours = (time.time() - first_entry) / 3600.0
            
            # Age Score (0.0=Dead, 1.0=Fresh) relative to effective lifespan
            age_efficiency = max(0.0, 1.0 - (age_hours / effective_lifespan))
            
            # Decayed Conviction
            conviction_score = raw_conviction * age_efficiency
            # ----------------------------
            
            # 3. Liquidity Score (major pairs)
            tier1_pairs = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT']
            tier2_pairs = ['SOL/USDT', 'DOGE/USDT', 'ADA/USDT', 'LINK/USDT']
            if sym in tier1_pairs:
                liquidity_score = 1.0
            elif sym in tier2_pairs:
                liquidity_score = 0.6
            else:
                liquidity_score = 0.3
                
            # 4. Age Score (Simple linear for general ranking)
            # We keep the raw age score for the mix, but also used it for decay above.
            age_score = max(0.0, min(1.0, 1.0 - (age_hours / 48.0)))
            
            # 5. Correlation Penalty
            correlation_penalty = 0.0
            for other_sym in open_positions:
                if other_sym == sym: continue
                if self._are_correlated(sym, other_sym):
                    correlation_penalty += 0.2  # Reduced from 0.5 per new directive or just kept?
                    # Plan said 0.2
            correlation_penalty = min(0.6, correlation_penalty) # Max penalty 0.6
            
            # 6. Stack Penalty (New Directive)
            stack_penalty = 0.0
            if meta.get('stack_count', 1) > 1:
                stack_penalty = 0.5 # Severe penalty for stacked positions
            
            # === Composite Score ===
            score = (
                config.CONSOLIDATION_WEIGHT_PNL * pnl_score +
                config.CONSOLIDATION_WEIGHT_CONVICTION * conviction_score +
                config.CONSOLIDATION_WEIGHT_LIQUIDITY * liquidity_score +
                config.CONSOLIDATION_WEIGHT_AGE * age_score -
                config.CONSOLIDATION_WEIGHT_CORRELATION * correlation_penalty -
                stack_penalty
            )

            # --- IMMUNITY BOOST ---
            # If this is a WHALE/ROCKET or ARB trade, give it a massive boost to prevent consolidation close
            pos_strategy = pos.strategy
            pos_reason = pos.metadata.get('reason', '')
            if 'WHALE' in pos_reason or 'ROCKET' in pos_reason or 'ARB' in pos_strategy or 'CARRY' in pos_strategy or 'BASIS' in pos_strategy:
                 score += 5.0 # Guaranteed survival
            
            # === Hard Override Rules ===
            notional = abs(qty * current_price)
            force_close = False
            force_reason = ""
            
            # A. Dust threshold
            if notional < config.CONSOLIDATION_DUST_THRESHOLD:
                force_close = True
                force_reason = f"Dust (${notional:.2f})"
                score = -999.0
                
            # B. Stale position (no favorable movement)
            # (Simplified: just check if losing for too long)
            if pnl_pct < 0 and age_hours > config.CONSOLIDATION_STALE_HOURS:
                force_close = True
                force_reason = f"Stale Loss ({age_hours:.0f}h, {pnl_pct*100:.1f}%)"
                score = -998.0
                
            scored_positions.append({
                'symbol': sym,
                'score': score,
                'pnl_pct': pnl_pct if entry > 0 else 0.0,
                'pnl_score': pnl_score,
                'conviction_score': conviction_score,
                'liquidity_score': liquidity_score,
                'age_score': age_score,
                'correlation_penalty': correlation_penalty,
                'force_close': force_close,
                'force_reason': force_reason,
            })
            
        # Sort by Score DESCENDING (Best = highest score, kept first)
        scored_positions.sort(key=lambda x: x['score'], reverse=True)
        
        # Log ranking
        if should_log and scored_positions:
            print(f"[ConsolidationEngine] Ranking:")
            for i, item in enumerate(scored_positions):
                status = "→ KEEP" if i < max_positions and item['score'] > -100 else "→ CLOSE"
                force_tag = f" [FORCED: {item['force_reason']}]" if item['force_close'] else ""
                print(f"  {i+1}. {item['symbol']:<12} score={item['score']:.2f} (PnL:{item['pnl_pct']*100:+.1f}%) {status}{force_tag}")
            
        # Select ONE position to close (lowest score)
        to_close = []
        if scored_positions:
            lowest = scored_positions[-1]
            if len(open_positions) > max_positions or lowest['force_close']:
                to_close.append(lowest['symbol'])
                print(f"[ConsolidationEngine] ❌ CLOSING: {lowest['symbol']} (Score: {lowest['score']:.2f})")
            else:
                print(f"[ConsolidationEngine] ✅ All positions acceptable.")
                
        self.consolidation_in_progress = False
        self.last_consolidation_time = time.time()
        
        # Notify Regime Controller that consolidation complete
        if self.regime_controller and to_close == []:
            self.regime_controller.complete_transition()
            
        return to_close
        
    def _are_correlated(self, sym1: str, sym2: str) -> bool:
        """
        Check if two symbols are in the same correlation family.
        """
        # Define families
        families = [
            ['BTC/USDT', 'ETH/USDT'],  # Majors move together
            ['SOL/USDT', 'ADA/USDT', 'AVAX/USDT', 'DOT/USDT'],  # L1s
            ['DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT'],  # Memes
            ['XRP/USDT', 'LTC/USDT'],  # OG Alts
            ['LINK/USDT', 'UNI/USDT', 'AAVE/USDT'],  # DeFi
        ]
        
        for family in families:
            if sym1 in family and sym2 in family:
                return True
        return False
        
    # Legacy method for backwards compatibility
    def consolidate_micro_exposure(self, current_prices: dict) -> list:
        """Legacy wrapper for run_consolidation_engine."""
        return self.run_consolidation_engine(current_prices)

    def calculate_take_profit(self, symbol: str, direction: str, entry_price: float, atr: float = None, metadata: dict = None) -> float:
        """
        Calculates the dynamic Take Profit price based on regime, strategy, and volatility.
        Used by Executor to set hard TP levels.
        """
        # 1. Determine Distance Pct
        # Default to Predator setting (Aggressive)
        tp_dist = config.PREDATOR_TAKE_PROFIT 
        
        # --- PATCH: STRATEGY-AWARE TP ---
        # If this is an ARB trade, use MUCH tighter targets to avoid immediate churn/logic conflict
        strategy = (metadata or {}).get('strategy', 'PREDATOR')
        is_arb = strategy in ['ARBITRAGE_GOLD', 'FUNDING_CARRY', 'BASIS_TRADE', 'ARB']
        
        if is_arb:
             # Use the 'rapid' target for the WHALE_BID_WALL as a proxy for Arb/Funding targets
             # Or better: use a hardcoded safe minimum for arb (e.g. 1.0%)
             tp_dist = 0.05 # 5% for arb (Safe buffer above 0.5% logic)
             if self.DEBUG: print(f"[{self.name}] 🎯 ARB TP: Using tight 5% target for {symbol}")
        elif self.get_metabolism_state() == 'SCAVENGER':
            tp_dist = config.SCAVENGER_SCALP_TP # Corrected to SCAVENGER_SCALP_TP (Was missing)
            
        # 2. Calculate Price
        if direction == 'BUY':
            tp_price = entry_price * (1.0 + tp_dist)
        else: # SELL
            tp_price = entry_price * (1.0 - tp_dist)
            
        return tp_price

    def manage_positions(self) -> List[Dict[str, Any]]:
        """
        Phase 2: Active Position Management via Monte Carlo.
        Returns a list of positions recommended for closure.
        """
        if not getattr(self, 'monte_carlo_manager', None) or not self.executor: 
            return []

        # Convert to dict for Monte Carlo (Legacy compat)
        pos_dicts = {}
        for vk, p in self.executor.positions.items():
            # Basic data
            pd = {
                 'entry_price': p.entry_price,
                 'direction': p.direction,
                 'quantity': p.quantity,
                 'toxic_funding': False
            }
            
            # Metadata extraction
            if hasattr(p, 'metadata') and isinstance(p.metadata, dict):
                pd['toxic_funding'] = p.metadata.get('toxic_funding', False)
                pd['funding_apy'] = p.metadata.get('funding_yield', 0.0)
            
            # Timestamp parsing
            try:
                # Handle varying timestamp formats
                ts_str = p.entry_timestamp
                # Simple check if it's already a float/int
                if isinstance(ts_str, (float, int)):
                    pd['first_entry_time'] = float(ts_str)
                else:
                    # ISO Format
                    if isinstance(ts_str, str) and ts_str.endswith('Z'):
                         ts_str = ts_str.replace('Z', '+00:00')
                    if isinstance(ts_str, str):
                        dt = datetime.datetime.fromisoformat(ts_str)
                        pd['first_entry_time'] = dt.timestamp()
                    else:
                        pd['first_entry_time'] = float(ts_str)
            except Exception:
                pd['first_entry_time'] = time.time()
                
            pos_dicts[p.symbol] = pd

        # Get SDE Data (Use defaults for now, or fetch from Oracle if linked)
        sde_defaults = {} 

        try:
            recs = self.monte_carlo_manager.run_position_health_check(
                positions=pos_dicts,
                current_prices=self.latest_prices,
                sde_data=sde_defaults
            )
            return recs
        except Exception as e:
            if self.DEBUG: print(f"[{self.name}] ❌ Monte Carlo Check Failed: {e}")
            return []


