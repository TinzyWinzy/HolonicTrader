import time
import traceback
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from scipy.stats import linregress
from rich.live import Live
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.console import Group
import sys
import os

# Initialize Console for Real Terminal (sys.__stdout__)
# We bypass sys.stdout (QueueLogger) so the Table doesn't spam the GUI/LogFile
console = Console(file=sys.__stdout__, force_terminal=True, width=120)

# UTF-8 Logging Support
from utf8_logging import get_logger

from HolonicTrader.holon_core import Holon, Disposition, Message
from HolonicTrader.agent_executor import TradeSignal, TradeDecision
from HolonicTrader.performance_tracker import get_performance_data
import config
from HolonicTrader.agent_trinity import TrinityStrategy # NEW: Phase 46
from core.scouts.entropy_scouter import EntropyScouter # AEHML Upgrade
from HolonicTrader.agent_medic import MedicHolon # The Field Physician

from HolonicTrader.exceptions import DeadMansSwitchTriggered

# Refactored Handler Modules
from HolonicTrader.trader_exit_handler import handle_exit, determine_exit_signal
from HolonicTrader.trader_entry_handler import handle_entry, build_ppo_state


class TraderHolon(Holon):
    """
    TraderHolon (Supra-Holon)
    The central coordinator that orchestrates the trading lifecycle using a 
    concurrency-first architecture (Phase 28: Warp Velocity).
    """
    
    def __init__(self, name: str = "TraderNexus", sub_holons: Dict[str, Holon] = None):
        super().__init__(name=name, disposition=Disposition(autonomy=0.9, integration=0.9))
        self.sub_holons = sub_holons if sub_holons else {}
        self.market_state = {'price': 0.0, 'regime': 'UNKNOWN', 'entropy': 0.0, 'signal': None}
        self.gui_queue = None
        self.gui_stop_event = None
        self.last_ppo_conviction = 0.5
        
        # Cycle Counters
        self.cycle_counter = 0
        self.scout_last_run = 0
        self.last_ppo_reward = 0.0  # PPO reward tracking
        self.gc_cycle_counter = 0   # Garbage collection counter
        
        # State Flags
        self.is_paused = False
        
        # Scout State
        self.scout_results = {}  # Cache for scout findings
        self.scout_active_list = []  # Persistent scout findings
        self.scout_batch_index = 0  # Batch rotation index (0, 1, 2 for 3 batches)
        self.scout_last_rotation = time.time()  # Track last batch rotation
        self.scout_cooldown_tracker = {}  # symbol -> last_entry_time (for cooldown-aware priority)

        # COMPUTE WASTE FIX: Profiling cache to avoid redundant analysis
        self._profile_cache = {}  # {symbol: {'ts': float, 'data': dict}}
        self._profile_cache_ttl = 60  # 1 minute - only re-profile if new data arrives

        # COMPUTE WASTE FIX: Veto tracking to avoid repeated rejections
        self._veto_tracker = {}  # {symbol: {'count': int, 'last_ts': float, 'reason': str}}
        self._veto_suppress_threshold = 3  # After 3 vetoes, suppress further logs for 60s

        # FIX 2026-03-21: Whale forced entry cooldown to prevent spam (DOT fired 8x in 8 min)
        self._whale_forced_cooldown = {}  # {symbol: expiry_timestamp}

        # 3D Holospace Memory (NEW - Phase 46)
        self.market_phase_data = {}  # symbol -> [{'entropy': ..., 'tda': ..., 'price': ...}]
        
        # === DYNAMIC ASSET SELECTION ===
        # Check if dynamic selection is enabled
        if hasattr(config, 'DYNAMIC_ASSET_SELECTION_ENABLED') and config.DYNAMIC_ASSET_SELECTION_ENABLED:
            # Start with full watchlist, let dynamic selector optimize
            self.active_session_whitelist = config.ALLOWED_ASSETS.copy()
            print(f"[{self.name}] 🔓 DYNAMIC MODE: Selecting best {config.MAX_CONCURRENT_ASSETS} assets from {len(self.active_session_whitelist)} available")
        elif hasattr(config, 'PHASE1_CONCENTRATED_ASSETS') and config.PHASE1_CONCENTRATED_ASSETS:
            # Only restrict if explicitly configured
            print(f"[{self.name}] 🎯 PHASE 1: Concentrated Asset List: {config.PHASE1_CONCENTRATED_ASSETS}")
            self.active_session_whitelist = config.PHASE1_CONCENTRATED_ASSETS.copy()
        else:
            # Default to hot list
            self.active_session_whitelist = config.ACTIVE_WATCHLIST.copy()
            print(f"[{self.name}] 📊 Using Active Watchlist: {len(self.active_session_whitelist)} assets")
        
        self._load_whitelist_from_disk()
        
        # Evolution Engine Watcher
        self.last_genome_mtime = 0
        
        self.verbose_logging = True # Request C: Enable transparency logs
        
        # Phase 46: Trinity Strategy
        self.trinity = TrinityStrategy()
        
        # AEHML: Entropy Scouter
        self.entropy_scouter = EntropyScouter()
        
        # Order Flow Tracking (for Dashboard Visualization)
        self.order_flow_data = {}  # symbol -> {'cvd_history': [], 'buy_ratio': 0.5, ...}
        self.volume_profile_data = {}  # symbol -> [[price, volume], ...]

        # Medic Holon (The Field Physician)
        self.medic = MedicHolon()
        self.register_agent('medic', self.medic)

        self.cycle_counter = 0 # General cycle counter (Phase 46 Fix)
        
        # --- SHARED EXECUTOR (Prevent Sticky Threading) ---
        self.executor = ThreadPoolExecutor(
            max_workers=config.TRADER_MAX_WORKERS, 
            thread_name_prefix=f"{self.name}_Worker"
        )

    def _load_whitelist_from_disk(self):
        import json
        try:
            path = os.path.join(os.getcwd(), 'scout_whitelist.json')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    saved_list = json.load(f)

                    strict_universe = getattr(config, 'STRICT_ASSET_UNIVERSE', False)
                    allowed_assets = set(getattr(config, 'ALLOWED_ASSETS', []))
                    if strict_universe and allowed_assets:
                        saved_list = [s for s in saved_list if s in allowed_assets]

                    # Dynamic Mode: Merge scout discoveries
                    if getattr(config, 'DYNAMIC_ASSET_SELECTION_ENABLED', False):
                        # Add new discoveries only if scout expansion is enabled
                        allow_expansion = getattr(config, 'SCOUT_ALLOW_EXPANSION', True)
                        if not allow_expansion:
                            saved_list = [s for s in saved_list if s in self.active_session_whitelist]

                        new_assets = set(saved_list) - set(self.active_session_whitelist)
                        if new_assets:
                            self.active_session_whitelist.extend(list(new_assets))
                            print(f"[{self.name}] 📂 Added {len(new_assets)} scout discoveries: {list(new_assets)[:5]}")
                    # Phase 1: Block Scout Expansion (only if explicitly configured)
                    elif hasattr(config, 'PHASE1_CONCENTRATED_ASSETS') and config.PHASE1_CONCENTRATED_ASSETS:
                        print(f"[{self.name}] 🔒 PHASE 1 LOCKED: Ignoring {len(saved_list)} assets from disk cache.")
                    else:
                        # Standard mode: merge lists
                        self.active_session_whitelist = list(set(self.active_session_whitelist + saved_list))
                        if strict_universe and allowed_assets:
                            self.active_session_whitelist = [s for s in self.active_session_whitelist if s in allowed_assets]
                        print(f"[{self.name}] 📂 Loaded {len(self.active_session_whitelist)} assets from local scout sync.")
        except Exception as e:
            print(f"[{self.name}] ⚠️ Failed to load scout whitelist: {e}")

    def _sync_whitelist_to_disk(self):
        import json
        try:
            path = os.path.join(os.getcwd(), 'scout_whitelist.json')
            with open(path, 'w') as f:
                json.dump(self.active_session_whitelist, f)
        except Exception as e:
            print(f"[{self.name}] ⚠️ Failed to sync scout whitelist: {e}")

    def _sync_scout_status_to_disk(self, results: dict):
        import json
        try:
            path = os.path.join(os.getcwd(), 'scout_status.json')
            # Add timestamp
            data = {
                'timestamp': time.time(),
                'results': results
            }
            with open(path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[{self.name}] ⚠️ Failed to sync scout status: {e}")

    def _sync_order_flow_to_disk(self):
        """Write order flow data for dashboard visualization."""
        import json
        try:
            path = os.path.join(os.getcwd(), 'order_flow_status.json')
            data = {
                'last_update': time.time(),
                'symbols': self.order_flow_data,
                'volume_profile': self.volume_profile_data
            }
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[{self.name}] ⚠️ Failed to sync order flow: {e}")

    def _update_order_flow(self, symbol: str, observer, df: pd.DataFrame = None):
        """Collect order flow data for a symbol."""
        oracle = self.sub_holons.get('oracle')
        if not oracle or not observer:
            return
            
        try:
            # 1. Get CVD from Oracle's analyze_order_flow
            flow = oracle.analyze_order_flow(symbol, observer)
            
            # Initialize if new symbol
            if symbol not in self.order_flow_data:
                self.order_flow_data[symbol] = {
                    'cvd_history': [],
                    'buy_ratio': 0.5,
                    'current_delta': 0.0,
                    'signal': 'NEUTRAL'
                }
            
            # Update CVD history (keep last 50)
            self.order_flow_data[symbol]['cvd_history'].append(flow.get('delta', 0.0))
            if len(self.order_flow_data[symbol]['cvd_history']) > 50:
                self.order_flow_data[symbol]['cvd_history'].pop(0)
            
            self.order_flow_data[symbol]['buy_ratio'] = flow.get('buy_ratio', 0.5)
            self.order_flow_data[symbol]['current_delta'] = flow.get('delta', 0.0)
            self.order_flow_data[symbol]['signal'] = flow.get('signal', 'NEUTRAL')
            
            # 2. Calculate Volume Profile from OHLCV
            if df is not None and len(df) >= 20:
                self._calculate_volume_profile(symbol, df)
                
        except Exception as e:
            # Silently ignore to not spam logs
            pass

    def _calculate_volume_profile(self, symbol: str, df: pd.DataFrame, bins: int = 20):
        """Calculate volume at price levels for heatmap."""
        try:
            # Use last 100 candles
            recent = df.tail(100)
            
            # Get price range
            price_min = recent['low'].min()
            price_max = recent['high'].max()
            
            if price_max <= price_min:
                return
            
            # Create price bins
            bin_size = (price_max - price_min) / bins
            volume_at_price = []
            
            for i in range(bins):
                bin_low = price_min + (i * bin_size)
                bin_high = bin_low + bin_size
                bin_mid = (bin_low + bin_high) / 2
                
                # Sum volume where price touched this bin
                vol_in_bin = recent[
                    (recent['low'] <= bin_high) & (recent['high'] >= bin_low)
                ]['volume'].sum()
                
                volume_at_price.append([bin_mid, float(vol_in_bin)])
            
            self.volume_profile_data[symbol] = volume_at_price
            
        except Exception as e:
            pass

    def _calculate_genome_fitness(self, metrics: dict) -> float:
        """
        Calculate Fitness Score per System Mandate:
        Fitness = (Win Rate * Avg Win) - (Loss Rate * Avg Loss) - (Slippage Cost)
        """
        win_rate = metrics.get('win_rate', 0.0) # 0.0 to 1.0
        avg_win = metrics.get('avg_win', 0.0)   # %
        avg_loss = abs(metrics.get('avg_loss', 0.0)) # %
        
        # Loss Rate = 1 - Win Rate
        loss_rate = 1.0 - win_rate
        
        # Slippage Cost (Assume per trade cycle, or avg slippage per trade)
        # Mandate says "Slippage Cost". We'll assume a fixed tax or the metric from sim.
        slippage = metrics.get('avg_slippage', 0.001) # Default 0.1%
        
        fitness = (win_rate * avg_win) - (loss_rate * avg_loss) - slippage
        return fitness

    def _run_scout_cycle(self):
        """
        The Slow Loop: Scans the Cold List -> Entropy Filter -> Whitelist.
        AEHML Upgrade: Physics-based asset selection with batch rotation.

        IMPROVEMENTS (2026-03-14):
        - Batch rotation: Cycles through 3 batches of 16 for full 48-asset coverage
        - Cooldown-aware priority: Assets off cooldown get priority boost
        - Held assets always bypass the batch cap (critical for exit management)
        """
        if time.time() - self.scout_last_run < config.SCOUT_CYCLE_INTERVAL:
            return

        observer = self.sub_holons.get('observer')
        if not observer: return

        # FIX-3: Reset per-cycle dedup set for 0.0 entropy warnings
        self._zero_entropy_warned = set()
        print_verbose = getattr(self, 'verbose_logging', False)

        # Initialize batch rotation state
        if not hasattr(self, 'scout_batch_index'):
            self.scout_batch_index = 0
        if not hasattr(self, 'scout_cooldown_tracker'):
            self.scout_cooldown_tracker = {}  # sym -> last_entry_time

        if print_verbose:
            print(f"[{self.name}] 🔭 SCOUT CYCLE STARTING... (Batch {self.scout_batch_index + 1}/3, Physics Mode)")

        try:
            # 1. Gather Candidates (Trinity + Config + Current)
            scan_list = set(config.SCOUT_CANDIDATES)
            if hasattr(self, 'trinity'):
                scan_list.update(self.trinity.get_watch_list())
            scan_list.update(self.active_session_whitelist)

            # Get held assets EARLY (they bypass all caps)
            held_assets = []
            executor = self.sub_holons.get('executor')
            if executor:
                held_assets = list(set([k.split(':')[0] for k in executor.held_assets.keys()]))

            # Hard universe guard: restrict to ALLOWED_ASSETS (always keep held assets for exit management)
            strict_universe = getattr(config, 'STRICT_ASSET_UNIVERSE', False)
            allowed_assets = set(getattr(config, 'ALLOWED_ASSETS', []))
            if strict_universe and allowed_assets:
                scan_list = {s for s in scan_list if s in allowed_assets}
                scan_list.update(held_assets)

            # Separate held assets (always scan) from candidates (batch rotation)
            candidate_list = scan_list - set(held_assets)

            # Apply batch rotation for full 48-asset coverage over 3 cycles
            batch_rotation_enabled = getattr(config, 'SCOUT_BATCH_ROTATION', False)
            batch_size = getattr(config, 'SCOUT_BATCH_SIZE', 16)

            if batch_rotation_enabled and len(candidate_list) > batch_size:
                candidate_list = list(candidate_list)

                # Sort by priority score (volume + cooldown status + regime stability)
                try:
                    tickers = observer.fetch_tickers_batch(candidate_list)
                    if tickers:
                        def get_priority_score(sym):
                            """
                            Priority = (Volume × 0.4) + (Cooldown Status × 0.3) + (Regime Stability × 0.3)
                            """
                            # Volume component (0-1 normalized)
                            t = tickers.get(sym)
                            if not t:
                                mapped = config.KRAKEN_SYMBOL_MAP.get(sym)
                                if mapped: t = tickers.get(mapped)
                            vol_score = float(t.get('quoteVolume') or 0.0) if t else 0.0

                            # Cooldown component (assets off cooldown get boost)
                            last_entry = self.scout_cooldown_tracker.get(sym, 0)
                            cooldown_sec = getattr(config, 'POOL_A_COOLDOWN_SEC', 60)
                            time_since_entry = time.time() - last_entry
                            cooldown_score = 1.0 if time_since_entry > cooldown_sec else 0.3

                            # Regime stability (prefer ORDERED over TRANSITION)
                            prev_result = getattr(self, 'scout_results', {}).get(sym, {})
                            prev_regime = prev_result.get('regime', 'TRANSITION')
                            regime_score = 1.0 if prev_regime == 'ORDERED' else (0.7 if prev_regime == 'TRANSITION' else 0.0)

                            # Normalize volume (divide by 1e9 for scaling)
                            vol_score = vol_score / 1e9 if vol_score > 0 else 0.0

                            return (vol_score * 0.4) + (cooldown_score * 0.3) + (regime_score * 0.3)

                        # Sort by priority (descending)
                        candidate_list = sorted(candidate_list, key=get_priority_score, reverse=True)

                        # Apply batch rotation: select different slice each cycle
                        start_idx = (self.scout_batch_index * batch_size) % len(candidate_list)
                        end_idx = start_idx + batch_size

                        if start_idx + batch_size > len(candidate_list):
                            # Wrap around
                            rotated = candidate_list[start_idx:] + candidate_list[:end_idx - len(candidate_list)]
                        else:
                            rotated = candidate_list[start_idx:end_idx]

                        candidate_list = rotated

                        if print_verbose:
                            print(f"[{self.name}] 🔭 Batch Rotation: Scanning assets {start_idx + 1}-{end_idx} of {len(candidate_list)}")
                    else:
                        candidate_list = candidate_list[:batch_size]
                except Exception as e:
                    print(f"[{self.name}] ⚠️ Priority Sort Error: {e}")
                    candidate_list = candidate_list[:batch_size]

            # Combine held assets (always included) + batch-selected candidates
            scan_list = held_assets + list(candidate_list)[:batch_size]

            if print_verbose:
                print(f"[{self.name}] 🔭 Scanning {len(scan_list)} assets ({len(held_assets)} held + {len(candidate_list)} candidates)")

            # 2. Fetch Data (1h data for stable regime detection)
            start_fetch = time.time()
            tickers_data = observer.fetch_market_data_batch(list(scan_list), timeframe='1h', limit=100)
            fetch_duration = time.time() - start_fetch

            if fetch_duration > 10.0:
                print(f"[{self.name}] ⚠️ Scout fetch took {fetch_duration:.1f}s (slow network)")

            # Early exit if no data
            if not tickers_data or len(tickers_data) == 0:
                print(f"[{self.name}] ⚠️ Scout fetch returned no data. Skipping cycle.")
                self.scout_last_run = time.time()
                return

            # 3. Entropy Scout Analysis
            scout_results = self.entropy_scouter.scout_regimes(tickers_data)
            self.scout_results = scout_results  # Update generic cache
            # Forward SampleEntropy results to Governor for SMCE regime classification
            governor = self.sub_holons.get('governor')
            if governor:
                governor._scout_results = scout_results

            # 4. Filter Whitelist
            # We only trade ORDERED or TRANSITION markets. CHAOS is vetoed (unless specialized).
            # Held assets are always kept for safe exit management.

            approved_list = []
            promoted_count = 0

            for sym in scan_list:
                # Always keep what we hold (exit management priority)
                if sym in held_assets:
                    approved_list.append(sym)
                    continue

                res = scout_results.get(sym, {})
                regime = res.get('regime', 'UNKNOWN')
                entropy = res.get('entropy', 0.0)

                if regime in ['ORDERED', 'TRANSITION']:
                    approved_list.append(sym)

                    # Promotion Log
                    if sym not in self.active_session_whitelist:
                        print(f"[{self.name}] 🚀 SCOUT PROMOTION: {sym} (Regime: {regime}, Ent: {entropy:.2f})")
                        promoted_count += 1
                        # Track cooldown entry time for priority scoring
                        self.scout_cooldown_tracker[sym] = time.time()
                else:
                    # CHAOTIC -> Reject
                    # FIX: Don't demote if entropy is exactly 0.0 (Likely Data Failure)
                    if entropy == 0.0:
                        if sym in self.active_session_whitelist:
                             # Keep it, assume data glitch
                             approved_list.append(sym)
                             # FIX-3 2026-03-05: Deduplicate — only warn once per symbol per scout cycle
                             if not hasattr(self, '_zero_entropy_warned'):
                                 self._zero_entropy_warned = set()
                             if sym not in self._zero_entropy_warned:
                                 self._zero_entropy_warned.add(sym)
                                 print(f"[{self.name}] ⚠️ SCOUT WARNING: {sym} has 0.0 Entropy (Data Gap). Preserving in Whitelist.")
                    elif sym in self.active_session_whitelist and sym not in held_assets:
                        print(f"[{self.name}] 📉 SCOUT DEMOTION: {sym} entered CHAOS (Ent: {entropy:.2f}). Dropping.")

            # 5. Commit
            if strict_universe and allowed_assets:
                approved_list = [s for s in approved_list if s in allowed_assets or s in held_assets]
            self.scout_active_list = list(set(approved_list))
            self.active_session_whitelist = self.scout_active_list.copy()
            self._sync_whitelist_to_disk()
            self._sync_scout_status_to_disk(scout_results)

            # 6. Advance batch rotation for next cycle
            if batch_rotation_enabled:
                self.scout_batch_index = (self.scout_batch_index + 1) % 3
                if print_verbose:
                    print(f"[{self.name}] 🔭 Batch rotation advanced to {self.scout_batch_index + 1}/3")

            # --- PHASE 46.2: WS SYNC ---
            if observer:
                observer.update_ws_symbols(self.active_session_whitelist)
            # ---------------------------

            self.scout_last_run = time.time()
            self._zero_entropy_warned = set()  # FIX-3: Clear for next cycle

        except Exception as e:
            print(f"[{self.name}] ❌ SCOUT CYCLE ERROR: {e}")
            # Update timestamp to prevent immediate retry
            self.scout_last_run = time.time()


    def register_agent(self, role: str, agent: Holon):
        self.sub_holons[role] = agent
        print(f"[{self.name}] Registered {role}: {agent.name}")

    def perform_health_check(self):
        observer = self.sub_holons.get('observer')
        if observer:
            try:
                status = observer.receive_message(self, {'type': 'GET_STATUS'})
                if not (isinstance(status, dict) and status.get('status') == 'OK'):
                    observer.receive_message(self, {'type': 'FORCE_FETCH'})
            except Exception as e:
                print(f"[{self.name}] ❌ OBSERVER HEALTH FAIL: {e}")

    def run_cycle(self):
        self.cycle_counter += 1
        self.perform_health_check()

        # 2026-03-21: Dynamic rebalancing — adjust allocation weights every 100 cycles
        if self.cycle_counter % 100 == 0:
            try:
                governor = self.sub_holons.get('governor')
                if governor and getattr(governor, 'atlas_available', False) and governor.atlas:
                    pf = getattr(governor.atlas, 'profit_filter', None)
                    if pf and hasattr(pf, 'performance_tracker'):
                        pf.performance_tracker.auto_rebalance_weights(min_trades=20)
            except Exception as e:
                print(f"[{self.name}] ⚠️ Auto-rebalance failed: {e}")
        
        executor = self.sub_holons.get('executor')
        monitor = self.sub_holons.get('monitor')
        governor = self.sub_holons.get('governor') # Moved up for Balance Sync Patch
        observer = self.sub_holons.get('observer')
        
        # --- WARM-UP: Pre-fetch prices on first cycle to prevent cold-start "Price Unknown" ---
        if observer and not getattr(self, '_warmup_done', False):
            print(f"[{self.name}] 🌡️ WARM-UP: Pre-fetching prices for {len(config.ALLOWED_ASSETS)} assets...")
            for sym in config.ALLOWED_ASSETS:
                try:
                    p = observer.get_latest_price(sym)
                    if p > 0:
                        print(f"[{self.name}] 🌡️ WARMUP: {sym} = ${p:.2f}")
                except Exception as e:
                    print(f"[{self.name}] ⚠️ WARMUP MISS: {sym} ({e})")
            self._warmup_done = True
        
        # --- PATCH 2: SAFETY-FIRST LOOP ORDERING (Resilience Update) ---
        # 1. Check Connectivity & Equity Status
        current_equity = None
        blind_mode = False
        
        if executor and executor.actuator:
             # Circuit Breaker Check is implicit in Actuator calls
             current_equity = executor.actuator.get_equity()

             # --- PATCH: CRITICAL BALANCE SYNC ---
             # We must sync BOTH Equity and Free Margin to Governor/Executor
             # to prevent "Insufficient Funds" errors due to drift.
             free_margin = executor.actuator.get_account_balance()

             if current_equity is not None and free_margin is not None:
                 if governor:
                     governor.set_live_balance(current_equity, free_margin)
                 if executor:
                     # FIX 2026-02-25: Sync with EQUITY (marginEquity), not availableMargin
                     # Using availableMargin caused artificial $67 drift because:
                     # - Ledger was set to marginEquity (~$112)
                     # - sync_balance was called with availableMargin (~$45)
                     # - This created phantom $67 slippage
                     executor.sync_balance(current_equity) # Updates DB and Internal

                 # Also update Monitor if present
                 if monitor:
                     monitor.metrics['current_equity'] = current_equity
             # ------------------------------------
             
             if current_equity is None:
                 print(f"[{self.name}] ⚠️ BLIND MODE: Cannot fetch Equity (API/Network Issue). Skipping Entry Logic.")
                 blind_mode = True
                 # If we are blind, we generally shouldn't trade, but maybe we can manage exits?
                 # Safest is to skip entries but allow exits if Order Book allows.
                 # But if Actuator is down, we can't do anything.
                 if getattr(executor.actuator, 'circuit_open', False):
                     print(f"[{self.name}] 💤 API CIRCUIT OPEN. Skipping Cycle.")
                     self.publish_agent_status()
                     time.sleep(10) 
                     return []
        
        # 2. IMMEDIATE HEALTH CHECK (The Fever Check)
        # FIX 2026-03-19 (Helix/Chronos): Reduced from 4h→30min hibernation.
        # Old 4h sleep caused F-03 paralysis: system missed ALL recovery trades.
        # SMCE doctrine already handles multi-day drawdown (48h cooldown).
        # Fever check only needs a short pause to prevent immediate panic re-entry.
        if monitor and current_equity:
            is_healthy, risk_msg = monitor.perform_live_check(current_equity)
            if not is_healthy:
                print(f"[{self.name}] 🛑 HEALTH LOCKDOWN: {risk_msg}")
                print(f"[{self.name}] 💤 Cooling down for 30 minutes (SMCE handles extended drawdown)...")
                self.publish_agent_status()
                time.sleep(1800) # 30 min cooldown (was 4h — caused paralysis)
                return [] # Skip cycle
        
        # 3. Old Check (Backup)
        # FIX 2026-03-19 (Helix): Allow exit management even during lockdown.
        # Old behavior: silently returned [] which meant NO exits were processed either.
        # New: skip new entries but still run exit logic for open positions.
        if monitor:
             is_healthy_old, _ = monitor.check_vital_signs()
             if not is_healthy_old:
                 print(f"[{self.name}] ⚠️ HEALTH LOCKDOWN (Persistent State). Entries blocked, exits allowed.")
                 # Don't return [] — fall through to exit management below
        
        # 4. MEDIC TRIAGE (Live Resuscitation)
        self.medic.perform_triage(self)
        # -------------------------------------------------------------
        
        interval = getattr(self, '_active_interval', 60)
        # print(f"\n[{self.name}] --- Starting Warp Cycle (Interval: {interval}s) ---")

        cycle_report = []
        entropies = []
        cycle_data_cache = {}
        print_verbose = getattr(self, 'verbose_logging', False)

        # AEHML FIX: Initialize safe defaults to avoid Cycle Errors (local variable access)
        # Python 3.11+ strict scoping requires explicit module references in try/except blocks
        _time = time
        _traceback = traceback
        global_bias = 0.5 # Neutral

        oracle = self.sub_holons.get('oracle')
        observer = self.sub_holons.get('observer')
        executor = self.sub_holons.get('executor')
        governor = self.sub_holons.get('governor')
        
        # --- CRITICAL WIRING: LINK GOVERNOR TO EXECUTOR ---
        if governor and executor and not getattr(governor, 'executor', None):
            try:
                governor.set_executor(executor)
            except AttributeError:
                print(f"[{self.name}] ⚠️ Governor missing set_executor method!")
        # --------------------------------------------------

        # ── SMCE v1: Link market agents to Governor once per session ──────────
        if governor and not getattr(governor, '_smce_agents_linked', False):
            entropy_agent = self.sub_holons.get('entropy')
            oracle_agent  = self.sub_holons.get('oracle')
            if entropy_agent:
                governor._entropy_agent = entropy_agent
            if oracle_agent:
                # Oracle doubles as structure provider in existing codebase
                governor._structure_agent = oracle_agent
            governor._smce_agents_linked = True
        # ─────────────────────────────────────────────────────────────────────   

        ppo = self.sub_holons.get('ppo')
        guardian = self.sub_holons.get('guardian')
        # monitor fetched above
        sentiment = self.sub_holons.get('sentiment')
        overwatch = self.sub_holons.get('overwatch') 
        regime_controller = self.sub_holons.get('regime')
        arbitrage = self.sub_holons.get('arbitrage')

        # --- PHASE -2: REGIME STATE UPDATE ---
        if regime_controller and executor:
            equity = executor.get_portfolio_value(0.0)
            health_metrics = {
                'trade_completed': False,
                'solvency_rejection': False,
                'gc_correction': False,
                'slippage': 0.0,
            }
            regime_controller.update_state(equity, health_metrics)

            regime_status = regime_controller.get_status_summary()

            if self.gui_queue:
                self.gui_queue.put({
                    'type': 'regime_status',
                    'data': {
                        'regime': regime_status['regime'],
                        'health': regime_status['health_score'],
                        'peak': regime_status['peak_equity']
                    }
                })

        # ── SMCE v1: Wire Telegram to digest once (first cycle) ───────────────
        if governor and getattr(governor, 'smce_digest', None):
            if not getattr(governor, '_smce_telegram_wired', False):
                # Try to find any Telegram-capable notifier in sub_holons
                for role, agent in self.sub_holons.items():
                    send_fn = getattr(agent, 'send_message', None) or \
                              getattr(agent, 'send_telegram', None) or \
                              getattr(agent, 'notify', None)
                    if send_fn and callable(send_fn):
                        governor.smce_digest.telegram_fn = send_fn
                        print(f"[{self.name}] 📬 SMCE Digest wired to Telegram via '{role}'")
                        break
                governor._smce_telegram_wired = True
        # ─────────────────────────────────────────────────────────────────────

        # --- PHASE -1: OVERWATCH AUDIT (The Sentry) ---
        if overwatch:
            overwatch.perform_audit()
            
        # --- PHASE -0.8: GLOBAL BIAS SYNC (Moved Upstream) ---
        # We must calculate bias early because it informs Kraken Ghost Resolution
        sent_score = 0.0
        if sentiment and oracle:
            sent_score = sentiment.fetch_and_analyze()
            oracle.set_crisis_score(getattr(sentiment, 'crisis_score', 0.0))
            # Pass equity for tier-aware crisis decisions
            if governor:
                oracle._last_known_equity = getattr(governor, 'balance', 100.0)
        
        if oracle and observer:
            fear = 0.0
            greed = 1.0
            if monitor:
                fear = monitor.metrics.get('current_drawdown', 0.0)
            if governor:
                greed = getattr(governor, 'risk_multiplier', 1.0)
            oracle.set_emotional_bias(fear, greed)
            global_bias = oracle.get_market_bias(sentiment_score=sent_score)

        # --- PHASE -0.6: WALK-FORWARD OPTIMIZATION (WFO) SYNC ---
        # FIX 2026-03-13: Block WFO from overriding critical safety parameters
        WFO_BLOCKED_PARAMS = {'DEFAULT_STOP_LOSS_PCT', 'DEFAULT_TAKE_PROFIT_PCT', 'STOP_LOSS_PCT'}

        if getattr(self, 'wfo_engine', None) and self.wfo_engine.is_running:
            new_params = self.wfo_engine.get_current_parameters()
            if new_params:
                for p_key, p_val in new_params.items():
                    # Skip blocked parameters (safety first!)
                    if p_key in WFO_BLOCKED_PARAMS:
                        # Silently block - don't clutter logs
                        continue

                    if hasattr(config, p_key):
                        current_val = getattr(config, p_key)
                        if current_val != p_val:
                            setattr(config, p_key, p_val)
                            print(f"[{self.name}] 🧬 WFO Parameter Update: {p_key} = {p_val}")

        # --- PHASE -0.5: ARBITRAGE SYNC (The Silent Miner) ---
        if arbitrage:
            arbitrage.perform_sync(self.active_session_whitelist)
            
        # --- PHASE -0.4: KRAKEN PLATFORM INTEL ---
        kraken = self.sub_holons.get('kraken')
        if kraken:
            kraken.update_market_intel(self.active_session_whitelist)
            k_status = kraken.get_system_status()
            # FIX 2026-03-05: Only warn on actual ERROR, not UNKNOWN (timeout/transient).
            if "ERROR" in k_status:
                print(f"[{self.name}] ⚠️ KRAKEN SYSTEM STATUS: {k_status}")
            
            # 👻 GHOST DETECTION: Sync Reality vs Ledger
            if executor:
                ghost_report = kraken.detect_ghost_positions(executor.held_assets)
                if any(ghost_report.values()):
                    # Call Resolution Engine
                    actions = kraken.resolve_ghosts(ghost_report, global_bias)
                    for action in actions:
                        act_type = action['type']
                        sym = action['symbol']
                        
                        if act_type == 'EXORCISE':
                            print(f"[{self.name}] 🗡️ EXORCISING GHOST: Closing {sym} ({action['qty']}) - {action['reason']}")
                            if executor.actuator:
                                executor.actuator.close_position(sym)
                                
                        elif act_type == 'ADOPT':
                            price = executor.latest_prices.get(sym, 0.0)
                            executor.adopt_position(sym, action['qty'], price, action['reason'])
                            
                        elif act_type == 'PURGE':
                            executor.purge_position(sym, action['reason'])
                            
                        elif act_type == 'RECONCILE':
                            executor.reconcile_position_size(sym, action['new_qty'], action['reason'])
            
            # 🛡️ EQUITY TRUTH SYNC: Heal Divergence Loops
            k_truth = kraken.get_equity_truth()
            if 'error' not in k_truth:
                if executor:
                    # NEW: Apply Collateral Haircuts
                    haircuts = kraken.get_collateral_haircuts()
                    # (In a real system, we'd iterate through held assets and apply ratios)
                    # For now, we use the raw equity truth but log if we detect non-USD assets

                    internal_equity = executor.get_portfolio_value(0.0)
                    
                    diff = abs(k_truth['equity'] - internal_equity)
                    if diff > 1.0:
                        print(f"[{self.name}] 🧘 SELF-HEALING: Syncing Equity to Kraken Truth: ${k_truth['equity']:.2f} (Fixing ${diff:.2f} Divergence)")
                        # FIX: Pass total equity, NOT available margin (was causing ledger divergence)
                        executor.sync_balance(k_truth['equity'])
                        if governor:
                             governor.set_live_balance(k_truth['equity'], k_truth['available'])
            
            # 🚑 SAFETY RAILS: Server-Side Stop Check
            if executor and executor.held_assets:
                rails = kraken.sync_server_side_stops(executor.held_assets, config.STOP_LOSS_PCT, executor)
                if rails.get('errors'):
                    print(f"[{self.name}] 🛡️ SAFETY ALERT: {len(rails['errors'])} assets ({rails['errors']}) failed to place Server-Side Stops! Manual check required.")
                elif rails.get('missing'):
                    print(f"[{self.name}] 🛡️ SAFETY WARNING: {len(rails['missing'])} assets ({rails['missing']}) lack Server-Side Stops (placement attempted but failed)")
            
            # 📉 FLASH CRASH CIRCUIT BREAKER
            env = kraken.monitor_execution_environment(self.active_session_whitelist)
            if env.get('status') == 'CRASH_RISK':
                dangerous = [d['symbol'] for d in env['dangerous_assets']]
                print(f"[{self.name}] 🛑 CIRCUIT BREAKER: Dangerous Spreads detected in {dangerous}. HALTING Entry Logic.")
                blind_mode = True # Use blind_mode to skip entries
            
            # Sync health metrics to Governor if critically low
            k_health = kraken.get_account_health()
            if k_health.get('status') == 'WARNING':
                print(f"[{self.name}] 🚨 KRAKEN MARGIN WARNING: Liquidation Distance {k_health.get('liquidation_distance', 0):.2f}")
                if governor:
                    governor.risk_multiplier = min(governor.risk_multiplier, 0.5) # Force defensive
            
            # ⛏️ MINER HOOK: Check for Gold Nuggets
            nuggets = arbitrage.mine_liquidity()
            if nuggets:
                 # Filter to allowed assets only (Phase 1: avoid wasting Governor calls on vetoed assets)
                 # FIX 2026-02-24: Handle None PHASE1_CONCENTRATED_ASSETS (dynamic mode)
                 phase1_assets = getattr(config, 'PHASE1_CONCENTRATED_ASSETS', None)
                 if phase1_assets:
                     # Phase 1 mode: use concentrated list
                     allowed = set(phase1_assets)
                 else:
                     # Dynamic mode: use full allowed assets
                     allowed = set(getattr(config, 'ALLOWED_ASSETS', []))
                 
                 nuggets = [n for n in nuggets if n['symbol'] in allowed]
                 if not nuggets:
                     pass  # All nuggets filtered out, skip silently
                 else:
                   print(f"[{self.name}] ⚒️ PROCESSING {len(nuggets)} ARB NUGGETS...")

                  
                   # === FIX 2026-03-04: ARB COOLDOWN & FREQUENCY CHECK ===
                   # Check if any nugget is in cooldown after previous stop-loss
                   from HolonicTrader.arb_safety_monitor import get_safety_monitor
                   safety_monitor = get_safety_monitor()
                  
                 for nug in nuggets:
                     # === FIX 2026-03-04: CHECK COOLDOWN & FREQUENCY ===
                     nug_sym = nug['symbol']
                     
                     # Check 1: Arb cooldown (1 hour after stop-loss)
                     if safety_monitor and safety_monitor.is_in_cooldown(nug_sym):
                         print(f"[{self.name}] 🛑 ARB COOLDOWN: {nug_sym} in cooldown after recent stop-loss. Skipping.")
                         continue
                     
                     # Check 2: Trade frequency limit
                     if hasattr(executor, 'check_trade_frequency_limit'):
                         allowed, reason = executor.check_trade_frequency_limit()
                         if not allowed:
                             print(f"[{self.name}] 🛑 FREQUENCY LIMIT: {nug_sym} blocked - {reason}")
                             continue
                     
                     # Create TradeSignal from Nugget
                     # Fetch price proactively
                     curr_price = 0.0
                     # FIX: Always fetch fresh price for Arb Entries to ensure accuracy (avoid Stale/Leaked prices)
                     if observer:
                         curr_price = observer.get_latest_price(nug_sym)
                     
                     if curr_price <= 0:
                         # Fallback to cache if live fetch failed
                         curr_price = executor.latest_prices.get(nug_sym, 0.0)

                     if curr_price <= 0:
                          # Fallback to cache if live fetch failed
                          curr_price = executor.latest_prices.get(nug_sym, 0.0)

                     # --- SANITY CHECK (Price Contamination Guard) ---
                     # Detect if PAXG (Gold) is priced like SOL ($90-$200)
                     if 'PAXG' in nug_sym and curr_price < 1500:
                         print(f"[{self.name}] 🛑 CRITICAL PRICE ERROR: {nug_sym} price is ${curr_price:.2f} (Likely SOL Contamination). REJECTING.")
                         continue
                         
                     if curr_price <= 0:
                          print(f"[{self.name}] ⚠️ SKIPPING NUGGET {nug_sym}: Price unknown (Live & Cache failed).")
                          continue

                     # Nugget keys: direction, confidence, reason, symbol
                     # Extract strategy from nugget reason for ARB bypass and metadata
                     nugget_reason = nug.get('reason', '')
                     nugget_strategy = 'ARBITRAGE'  # Default
                     if 'BASIS_CARRY_LONG' in nugget_reason:
                         nugget_strategy = 'BASIS_CARRY_LONG'
                     elif 'BASIS_CARRY_SHORT' in nugget_reason:
                         nugget_strategy = 'BASIS_CARRY_SHORT'
                     elif 'FUNDING_CARRY' in nugget_reason:
                         nugget_strategy = 'FUNDING_CARRY'
                     elif 'SPATIAL_ARB' in nugget_reason:
                         nugget_strategy = 'SPATIAL_ARB'
                     elif 'GOLD_ORACLE' in nugget_reason:
                         nugget_strategy = 'ARBITRAGE_GOLD'

                     # --- PATCH: PRE-CALCULATE SIZE (Arb Sizing) ---
                     # We must ask Governor for valid size to avoid "Solvency Veto" loop (1.0 Unit vs 100% Equity)
                     safe_qty = 0.0
                     if governor:
                         # ── SMCE v1 Layer 2: Probability Stacking Engine ──
                         is_prob_eligible = True
                         prob_size_mod = 1.0

                         # FIX: ARB trades bypass L2 probability scoring (they are structurally decorrelated)
                         # FIX 2026-03-01: Enhanced ARB detection to catch all variants
                         is_arb_trade = (
                             nugget_strategy in ['ARBITRAGE_GOLD', 'FUNDING_CARRY', 'BASIS_TRADE', 'ARB', 'GOLD_LEAD_LAG', 'ARBITRAGE', 'BASIS_CARRY_LONG', 'BASIS_CARRY_SHORT'] or
                             'BASIS' in nugget_strategy.upper() or
                             'FUNDING' in nugget_strategy.upper() or
                             'CARRY' in nugget_strategy.upper() or
                             'ARBITRAGE' in nugget_strategy.upper()
                         )

                         if hasattr(governor, 'smce_prob_engine') and not is_arb_trade:
                             smce_regime = "TRANSITION"
                             if hasattr(governor, 'get_smce_regime'):
                                 smce_regime = governor.get_smce_regime()

                             # Approximate portfolio & market state for Arbitrage checking
                             cand = {"direction": nug['direction'], "symbol": nug_sym, "proposed_cluster_exposure": 0.05}
                             p_state = {"equity": getattr(governor, 'balance', 1000.0), "cluster_exposure": 0.0, "cvar_95": 0.01}
                             m_ctx = {
                                 "structure": "NEUTRAL",  # Arb is usually structure-neutral
                                 "momentum_aligned": True, # Arb implies yield momentum
                                 "liquidity_status": "healthy",
                                 "entropy": 0.5, # Assume ordered
                                 "correlation_idx": 0.1 # Arb is decorrelated
                             }

                             prob_result = governor.smce_prob_engine.score_trade(cand, p_state, m_ctx, smce_regime)
                             is_prob_eligible = prob_result.get("eligible", False)
                             prob_size_mod = prob_result.get("size_modifier", 1.0)

                             if not is_prob_eligible:
                                 print(f"[{self.name}] 🛑 ARB PROBABILITY VETO: {nug_sym} rejected by Layer 2 Engine (Score: {prob_result.get('score')}).")

                         if is_prob_eligible:
                             # Use nugget_strategy and nugget_reason extracted above
                             is_app, qty, lev = governor.calc_position_size(
                                 symbol=nug_sym,
                                 asset_price=curr_price,
                                 conviction=nug['confidence'],
                                 direction=nug['direction'],
                                 whale_confirmed=True, # Treat Arb as Whale/High Priority
                                 metadata={'strategy': nugget_strategy, 'reason': nugget_reason, 'is_xstock': nug.get('is_xstock', False), 'leverage': 10.0},
                                 latest_prices=executor.latest_prices if executor else {}
                             )
                             if is_app:
                                 safe_qty = qty * prob_size_mod # Apply Layer 2 modifier
                                 print(f"[{self.name}] ⚖️ ARB SIZING APPLIED: {nug_sym} -> {safe_qty:.6f} (Lev {lev}x)")
                             else:
                                 print(f"[{self.name}] 🛑 ARB GOVERNOR VETO: Sizing rejected completely for {nug_sym}.")
                                 continue
                         else:
                             continue # Probability Veto
                     else:
                         # Fallback for no governor (shouldn't happen in live)
                         safe_qty = (config.MIN_ORDER_VALUE * 1.05) / curr_price

                     if safe_qty <= 0: continue
                     # ----------------------------------------------

                     sig = TradeSignal(
                         symbol=nug_sym,
                         direction=nug['direction'],
                         size=safe_qty, # Use Calculated Quantity
                         price=curr_price,
                         conviction=nug['confidence']
                     )

                     # Use nugget_strategy extracted above for metadata
                     sig.metadata = {
                         'reason': nugget_reason,
                         'strategy': nugget_strategy,
                         'is_whale': True,
                         'is_percent': False, # Explicitly Units
                         'is_xstock': nug.get('is_xstock', False)  # Pass xStock flag
                     }
                     
                     # Check if we already hold it? (Executor check)
                     # Or just fire it into decision engine.
                     # Let's fire it.
                     if executor:
                          decision = executor.decide_trade(sig, 'ORDERED', 0.0)
                          if decision.action != 'HALT':
                              # Get Price for execution log (Already fetched)
                              res = executor.execute_transaction(decision, curr_price)
                              # if res:
                              #     print(f"[{self.name}] 💰 NUGGET SECURED: {nug['symbol']} {nug['direction']} ({nug['reason']})")
            
        # Evolution Watcher (Conditional)
        if getattr(config, 'ENABLE_EVOLUTION', False):
            self._scan_for_genome_updates()
        
        self._run_scout_cycle()
        
        # --- PHASE 0.1: EXIT MANAGEMENT SCAN (MOVED DOWNSTREAM) ---
        # Logic moved after Matrix Sync to ensure fresh data (Growth Logic)
        # ----------------------------------------------------------

        # --- PHASE 46: TRINITY ASSET ROTATION ---
        # 1. Determine Market Regime (from History/Entropy)
        m_regime = self.market_state.get('regime', 'TRANSITION')
        # global_bias is now synced upstream (Phase -0.8)
        btc_trend = 'BULL' if global_bias >= 0.50 else 'BEAR'
            
        # 2. Get Targets from Trinity Strategy
        trinity_targets = self.trinity.get_allocation_target(m_regime, btc_trend)
        
        # 3. Update Whitelist (Dynamic Focus)
        # Only trade what the strategy wants + Any open positions (to manage exits)
        open_positions = []
        if executor: 
             # Extract unique base symbols from virtual keys (Phase 47 Virtual Isolation)
             open_positions = list(set([k.split(':')[0] for k in executor.held_assets.keys()]))
        
        # Merge and Dedup (Trinity + OpenPositions + ScoutRockets)
        target_list = list(trinity_targets.keys()) + open_positions + self.scout_active_list
        strict_universe = getattr(config, 'STRICT_ASSET_UNIVERSE', False)
        allowed_assets = set(getattr(config, 'ALLOWED_ASSETS', []))
        if strict_universe and allowed_assets:
            target_list = [s for s in target_list if s in allowed_assets or s in open_positions]
        self.active_session_whitelist = list(set(target_list))
        
        # --- PHASE 46.1: UNIFIED MATRIX FETCH (Warp Speed) ---
        target_assets = self.active_session_whitelist
        if print_verbose:
            print(f"[{self.name}] ⚡ Matrix Syncing {len(target_assets)} assets...")
        
        # Unified parallel fetch (15m, 1h, Books, Funding)
        cycle_data_cache = observer.fetch_matrix_data(target_assets)
        
        # Warm up Oracle (Kalman) with synced data
        for sym, data in cycle_data_cache.items():
            try:
                oracle.get_kalman_estimate(sym, data['df_15m'])
            except: pass
        
        # -----------------------------------------------------
        
        # --- PATCH: BATCH PRICE UPDATE & METRICS ---
        # Ensure Executor has fresh prices for ALL targeted assets (especially held ones)
        # This is critical for MAE/MFE and PnL Calculation.
        if executor:
            new_prices = {}
            for sym, data in cycle_data_cache.items():
                df = data.get('df_15m')
                if df is not None and not df.empty:
                     new_prices[sym] = df['close'].iloc[-1]
            
            executor.latest_prices.update(new_prices)
            
            # Update Performance Metrics (MAE/MFE)
            executor.update_position_metrics(executor.latest_prices)
        # ---------------------------------------------
            

            
        # --- PACK HUNT DATA PREP ---
        # Calculate 24h % Change for all assets to find Alpha/Beta dispersion
        pack_changes = []
        self.session_ticker_data = {} # Store for _analyze_asset lookup
        
        for sym, unit in cycle_data_cache.items():
            d = unit.get('df_15m')
            if d is not None and len(d) >= 90: # Need approx 24h data (96 bars of 15m)
                try:
                    # Use first available bar if < 96, else -96
                    start_idx = -96 if len(d) >= 96 else 0
                    start_p = d['close'].iloc[start_idx]
                    end_p = d['close'].iloc[-1]
                    if start_p > 0:
                        pct_change = ((end_p - start_p) / start_p) * 100.0
                        pack_changes.append(pct_change)
                        self.session_ticker_data[sym] = {'percentage': pct_change}
                except: pass
        
        if pack_changes:
            pack_arr = np.array(pack_changes)
            self.session_pack_stats = {
                'mean': float(np.mean(pack_arr)),
                'std': float(np.std(pack_arr))
            }
            # print(f"[{self.name}] 🐺 PACK STATS: Mean {self.session_pack_stats['mean']:.2f}% | Std {self.session_pack_stats['std']:.2f}%")
        else:
            self.session_pack_stats = {'mean': 0.0, 'std': 1.0}
        # ---------------------------

        # --- PHASE 46.3: RUST BATCH ANALYSIS (Tier 3) ---
        if print_verbose:
            print(f"[{self.name}] 🦀 Rust-Accelerating Signals...")
            
        batch_prices = {s: d['df_15m']['close'].values.tolist() for s, d in cycle_data_cache.items() if d.get('df_15m') is not None and not d['df_15m'].empty and 'close' in d['df_15m'].columns}
        batch_highs = {s: d['df_15m']['high'].values.tolist() for s, d in cycle_data_cache.items() if d.get('df_15m') is not None and not d['df_15m'].empty and 'high' in d['df_15m'].columns}
        batch_lows = {s: d['df_15m']['low'].values.tolist() for s, d in cycle_data_cache.items() if d.get('df_15m') is not None and not d['df_15m'].empty and 'low' in d['df_15m'].columns}
        
        try:
            import holonic_speed
            rust_signals = holonic_speed.calculate_signals_matrix(
                list(batch_prices.keys()),
                batch_prices,
                batch_highs,
                batch_lows
            )
            
            # Pack rust signals into cache for propagation
            for s, signals in rust_signals.items():
                if s in cycle_data_cache:
                    cycle_data_cache[s]['rust_signals'] = signals
            
            if print_verbose: print(f"[{self.name}] ✅ Rust-Acceleration Complete.")
        except Exception as e:
            print(f"[{self.name}] ⚠️ Rust Acceleration Failed: {e}. Falling back to Pandas.")
        
        # --- PHASE 46.4: ORDER FLOW DATA COLLECTION ---
        # Collect CVD and Volume Profile for dashboard visualization
        for s, cache in cycle_data_cache.items():
            try:
                df_15 = cache.get('df_15m')
                self._update_order_flow(s, observer, df_15)
            except:
                pass  # Silently skip if data missing
        # -----------------------------------------------

        # --- PHASE 0.1: EXIT MANAGEMENT SCAN (REFACTORED) ---
        # Now runs AFTER Data Fetch to ensure fresh data (Growth Logic)
        if executor and governor:
             for held_virt_key, held_qty in list(executor.held_assets.items()):
                 if abs(held_qty) < 0.0000001: continue
                 
                 # Extract Real Symbol for price and cache lookups
                 held_sym = held_virt_key.split(':')[0] if ':' in held_virt_key else held_virt_key
                 
                 curr_p = executor.latest_prices.get(held_sym, 0.0)
                 # Pull position data from internal ledger truth
                 # Try position_metadata first (dict), then positions (Position object)
                 pos_data = executor.position_metadata.get(held_virt_key)
                 if not pos_data:
                     # Fallback: get Position object and convert to dict-like access
                     pos_obj = executor.positions.get(held_virt_key)
                     if pos_obj:
                         pos_data = {
                             'entry_price': pos_obj.entry_price,
                             'direction': pos_obj.direction,
                             'strategy': pos_obj.strategy,
                             'quantity': pos_obj.quantity,
                             'symbol': pos_obj.symbol,
                             'virt_key': pos_obj.virt_key,
                         }
                     else:
                         continue

                 # --- EXTRACT MOMENTUM DATA ---
                 momentum_meta = {}
                 if held_sym in cycle_data_cache:
                     cache = cycle_data_cache[held_sym]
                     df_15 = cache.get('df_15m')

                     if df_15 is not None and not df_15.empty:
                         # 1. RSI
                         if 'rsi' in df_15.columns:
                             momentum_meta['rsi'] = df_15['rsi'].iloc[-1]

                         # 2. RVOL (Approximate vs 20-period SMA)
                         if 'volume' in df_15.columns and len(df_15) > 20:
                             vol_now = df_15['volume'].iloc[-1]
                             vol_avg = df_15['volume'].rolling(20).mean().iloc[-1]
                             if vol_avg > 0:
                                 momentum_meta['rvol'] = vol_now / vol_avg

                         # 3. BB Width (Optional)
                         if 'bb_upper' in df_15.columns:
                             u = df_15['bb_upper'].iloc[-1]
                             l = df_15['bb_lower'].iloc[-1]
                             momentum_meta['bb_width'] = (u - l) / curr_p if curr_p > 0 else 0.0

                         # 4. Order Flow (Injected - Phase 46.5)
                         if held_sym in self.order_flow_data:
                             momentum_meta['order_flow'] = {
                                 'signal': self.order_flow_data[held_sym].get('signal', 'NEUTRAL'),
                                 'buy_ratio': self.order_flow_data[held_sym].get('buy_ratio', 0.5)
                             }

                         # 5. Yield Data (Injected - Phase 47: Farming)
                         if arbitrage:
                             momentum_meta['yield_apy'] = arbitrage.funding_yields.get(held_sym, 0.0)

                 # --- DYNAMIC AI EXIT CHECK ---
                 rec = 'HOLD'
                 if oracle:
                     entry_p = pos_data.get('entry_price', 0.0)
                     direction = pos_data.get('direction', 'BUY')
                     
                     # PASS METADATA TO ORACLE
                     phys_res = oracle.verify_holding_physics(
                        held_sym, direction, current_price=curr_p, entry_price=entry_p, 
                        metadata=momentum_meta # <--- NEW INJECTION
                     )
                     
                     exit_type = None
                     
                     if not phys_res.get('valid', True):
                         action = getattr(config, 'THESIS_FAILURE_ACTION', 'WARN')
                         if action == 'FLAT':
                             exit_type = 'THESIS_INVALID'
                             exit_reason = phys_res.get('reason', 'Physics Veto')
                         else:
                             # Just warn but allow standard Governor checks (TP/SL)
                             # if getattr(self, 'verbose_logging', False):
                             #    print(f"[{self.name}] ⚠️ THESIS FAILED ({held_sym}): {phys_res.get('reason')} (Action: {action} - HOLDING)")
                             exit_type = None
                     
                     if not exit_type:
                         rec = phys_res.get('recommendation', 'HOLD')
                         exit_type, exit_reason = governor.check_exit_conditions(held_sym, curr_p, pos_data, recommendation=rec)
                 else:
                     exit_type, exit_reason = governor.check_exit_conditions(held_sym, curr_p, pos_data)
                 
                 # --- FORCE EXIT (Road to Profitability) ---
                 if held_sym in getattr(config, 'FORCE_EXIT_ASSETS', []) and not exit_type:
                     print(f"[{self.name}] 🚨 FORCE EXIT: {held_sym} found in Kill List.")
                     exit_type = "FORCE_KILL"
                     exit_reason = "Manual Config Request"

                 # --- FIX 2026-03-21: SOFTWARE HARD-STOP (Belt & Suspenders) ---
                 # Catches positions with no exchange stop-loss or failed stop placement
                 if not exit_type:
                     entry_p_val = pos_data.get('entry_price', 0)
                     if entry_p_val and curr_p and entry_p_val > 0:
                         direction = pos_data.get('direction', 'BUY')
                         if direction == 'BUY':
                             unrealized_pct = (curr_p - entry_p_val) / entry_p_val
                         else:
                             unrealized_pct = (entry_p_val - curr_p) / entry_p_val
                         hard_stop_pct = getattr(config, 'SOFTWARE_HARD_STOP_PCT', -0.05)
                         if unrealized_pct <= hard_stop_pct:
                             print(f"[{self.name}] 🚨 SOFTWARE HARD-STOP: {held_sym} at {unrealized_pct:.2%} (limit {hard_stop_pct:.0%})")
                             exit_type = "EMERGENCY_STOP"
                             exit_reason = f"Software hard-stop {unrealized_pct:.2%}"

                 if exit_type:
                     print(f"[{self.name}] 🚨 EXIT SIGNAL: {held_sym} -> {exit_type} ({exit_reason})")
                     # 2026-03-21: Store exit_reason on position metadata for trade recording
                     pos_vk = held_sym
                     if hasattr(executor, 'positions') and pos_vk in executor.positions:
                         executor.positions[pos_vk].metadata['exit_reason'] = exit_type
                     is_long = pos_data.get('direction', 'BUY') == 'BUY'
                     close_signal = TradeSignal(
                            symbol=held_sym,
                            direction='SELL' if is_long else 'BUY',
                            size=1.0, 
                            price=curr_p,
                            conviction=1.0,
                            metadata={
                                'reason': f"AUTO_{exit_type}", 
                                'reduce_only': True,
                                'strategy': pos_data.get('strategy', 'DIRECTIONAL') # Fix pool mapping
                            }
                        )
                     executor.execute_transaction(TradeDecision(
                         action='EXECUTE', original_signal=close_signal, adjusted_size=1.0, disposition=Disposition(autonomy=1.0, integration=1.0),
                         block_hash='AUTO_EXIT_OVERRIDE'
                     ), curr_p)

        # --- PHASE 1: PARALLEL ANALYSIS PASS ---
        analysis_results = []
        # Use Shared Executor
        futures = []
            
        # --- ZOMBIE FIX: UNION of Whitelist + Held Assets ---
        # Ensure we analyze assets we own, even if they aren't in the current Phase 1 focus list.
        # Sanitizing to base symbols (Phase 47: Virtual Isolation aware)
        held_base_syms = list(set([k.split(':')[0] for k in executor.held_assets.keys()])) if executor else []
        analysis_universe = list(set(self.active_session_whitelist + held_base_syms))

        # ── SMCE v1 Layer 1: Macro Regime Gate ──
        smce_regime = "TRANSITION"
        if governor and hasattr(governor, 'get_smce_regime'):
            smce_regime = governor.get_smce_regime()
            
        is_defensive_regime = (smce_regime == "DEFENSIVE")
        if is_defensive_regime:
            print(f"[{self.name}] 🛡️ SMCE-REGIME GATE: DEFENSIVE Macro State. Skipping new entry analysis.")

        for s in analysis_universe:
            cache = cycle_data_cache.get(s, {})
            df_15m = cache.get('df_15m')
            # Pre-fetch price for check
            p_check = 0.0
            if df_15m is not None and not df_15m.empty:
                p_check = df_15m['close'].iloc[-1]
            
            # --- UPSTREAM PRE-FILTER (Cooldown Audit & Regime) ---
            # Stop "Boy Who Cried Wolf". If Governor won't allow trade, don't ask Strategy.
            # Exception: Always analyze if we HOLD the asset (need to check for Exits/Thesis)
            is_held = (s in held_base_syms)
            
            if not is_held:
                if is_defensive_regime:
                     continue # SMCE L1 Gate Block
                # FIX 2026-03-05: PAXG is an arb-only asset handled via mine_liquidity() nuggets.
                # Passing it through the directional is_trade_allowed() check (without is_arb=True)
                # always fails the low-equity ($1000) guard. Skip the pre-filter for arb-only assets.
                _arb_only = getattr(config, 'ARB_ONLY_ASSETS', {'PAXG/USDT'})
                _is_arb_only = s in _arb_only or s.replace('/USDT', '') in _arb_only
                if _is_arb_only:
                    continue  # Arb-only: handled by nuggets pipeline, not directional scan
                if governor and not governor.is_trade_allowed(s, p_check, silent=True):
                    # Silent Skip (Verbose only)
                    if getattr(self, 'verbose_logging', False):
                        print(f"[{self.name}] ⏳ PRE-FILTER: Skipping {s} (Cooldown/Stack).")
                    continue

            # --------------------------------------------

            # FIX 2026-03-03: Validate data before submitting to analysis
            if df_15m is None or df_15m.empty:
                if getattr(self, 'verbose_logging', False):
                    print(f"[{self.name}] ⏳ PRE-FILTER: Skipping {s} - No 15m data available.")
                continue
            
            df_1h = cache.get('df_1h')
            book = cache.get('book')
            funding = cache.get('funding')
            rust_sigs = cache.get('rust_signals')
            rust_sigs = cache.get('rust_signals')
            futures.append(self.executor.submit(self._analyze_asset, s, df_15m, df_1h, global_bias, book, funding, rust_sigs))
        
        try:
            for f in as_completed(futures, timeout=240):
                try:
                    res = f.result()
                    if res and isinstance(res, dict): 
                        analysis_results.append(res)
                    elif res is None:
                        # Asset analysis returned None - likely data issue
                        pass  # Silently skip, already logged in _analyze_asset
                except Exception as e:
                    print(f"[{self.name}] ⚠️ Analysis Logic Error: {e}")
                    print(f"[{self.name}] ⚠️ Stack trace: {_traceback.format_exc()[:500]}")
        except TimeoutError:
             print(f"[{self.name}] ⚠️ Analysis Cycle Timed Out (proceeding with partial results)")

        # FIX: Sort by CONVICTION (Highest First) instead of Symbol
        # This ensures we take the BEST trades first if capital is limited.
        def get_conviction(res):
            if not res or not isinstance(res, dict):
                return 0.0
            sig = res.get('entry_signal')
            return sig.conviction if sig else 0.0

        analysis_results.sort(key=get_conviction, reverse=True)
        # analysis_results.sort(key=lambda x: x['symbol']) # OLD NAIVE SORT

        # --- PHASE 2: SEQUENTIAL EXECUTION PASS ---
        cycle_entries_count = 0
        limit_entries = getattr(config, 'TRADER_MAX_CYCLE_ENTRIES', 3)

        for res in analysis_results:
            # Safety check: Skip malformed results
            if not res or not isinstance(res, dict):
                print(f"[{self.name}] ⚠️ Skipping malformed analysis result: {type(res)}")
                continue
            if 'symbol' not in res:
                print(f"[{self.name}] ⚠️ Skipping analysis result missing 'symbol' key: {res.keys() if hasattr(res, 'keys') else 'N/A'}")
                continue
                
            symbol, data, current_price = res['symbol'], res['data'], res['price']
            row_data, indicators = res['row_data'], res['indicators']
            entropy_val, regime = res.get('entropy_val', 0.0), res.get('regime', 'TRANSITION')
            tda_score, tda_status = res.get('tda_score', 0.5), res.get('tda_status', 'STABLE')
            
            if entropy_val > 0: entropies.append(entropy_val)

            try:
                if executor: executor.latest_prices[symbol] = current_price
                if executor and governor: 
                    # --- PATCH: SOLVENCY UPDATE ---
                    e_tot, e_free = executor.get_balance_details()
                    governor.set_live_balance(e_tot, e_free)
                    
                    # --- PHASE 7: CONSOLIDATION ENGINE ---
                    # Run intelligent consolidation (replaces simple MICRO logic)
                    to_close = governor.run_consolidation_engine(
                        executor.latest_prices,
                        position_metadata=executor.position_metadata
                    )
                    
                    # --- PATCH: DRAWDOWN EMERGENCY LIQUIDATION ---
                    # If Drawdown Lock is active, aggressively cut losing positions
                    if governor.drawdown_lock:
                         for pos_sym, pos_data in governor.positions.items():
                             if pos_sym in to_close: continue
                             
                             entry_p = pos_data.get('entry_price', 0)
                             curr_p = executor.latest_prices.get(pos_sym, entry_p)
                             direction = pos_data.get('direction', 'BUY')
                             
                             if entry_p > 0:
                                 if direction == 'BUY':
                                     pnl = (curr_p - entry_p) / entry_p
                                 else:
                                     pnl = (entry_p - curr_p) / entry_p
                                     
                                 if pnl < -0.01: # -1% Loss Threshold (Quick Cut)
                                     print(f"[{self.name}] 🚨 DRAWDOWN EMERGENCY: Liquidating Loser {pos_sym} (PnL {pnl:.2%}) due to Lock.")
                                     to_close.append(pos_sym)
                    # ---------------------------------------------
                    
                    for c_sym in to_close:
                        print(f"[{self.name}] 🧹 EXECUTING CONSOLIDATION CLOSE: {c_sym}")
                        # Safe Metadata Access
                        meta = executor.position_metadata if executor and executor.position_metadata else {}
                        direction = meta.get(c_sym, {}).get('direction', 'BUY')
                        is_long = direction == 'BUY'
                        close_qty = abs(executor.held_assets.get(c_sym, 0.0))
                        close_price = executor.latest_prices.get(c_sym, 0.0)
                        
                        # DEBUG: Inspect quantity
                        real_holding = executor.held_assets.get(c_sym, "MISSING")
                        print(f"[{self.name}] 🔍 CONSOLIDATION DEBUG: Sym={c_sym}, Held={real_holding}, CloseQty={close_qty:.8f}")
                        
                        # Construct proper TradeSignal and TradeDecision for Executor
                        # Imports moved to global scope
                        
                        close_signal = TradeSignal(
                            symbol=c_sym,
                            direction='SELL' if is_long else 'BUY',
                            size=1.0, # FIXED: Use 1.0 (100%) percent multiplier
                            price=close_price,
                            conviction=1.0,
                            metadata={
                                'reason': 'CONSOLIDATION', 
                                'reduce_only': True,
                                'strategy': 'DIRECTIONAL' # TODO: Can we infer? Consolidate usually Directional
                            }
                        )
                        close_decision = TradeDecision(
                            action='EXECUTE',
                            original_signal=close_signal,
                            adjusted_size=1.0, # FIXED: Use 1.0 (100%) percent multiplier
                            disposition=Disposition(autonomy=1.0, integration=1.0),
                            block_hash='CONSOLIDATION',
                            entropy_score=0.0
                        )
                        
                        # Execute the close
                        executor.execute_transaction(close_decision, close_price)
                        # NOTE: Do NOT call sync_positions here — it calls .clear() which corrupts
                        # executor.positions while the consolidation loop is still iterating.
                    # -----------------------------------
                    
                    # --- PATCH: IMMUNE SYSTEM ACTIVATION ---
                    if monitor:
                        # Feed the Immune System (Check Drawdown/Solvency)
                        perf = get_performance_data()
                        is_healthy = monitor.update_health(
                            executor_summary=executor.get_execution_summary(),
                            performance_data=perf
                        )
                        if not is_healthy:
                            print(f"[{self.name}] ☠️ IMMUNE SYSTEM TRIGGERED: HALTING CYCLE.")
                            # Optional: Panic Close?
                            # executor.panic_close_all(executor.latest_prices)
                            return [] # Abort Cycle
                    # ------------------------------
                    
                    # --- POSITION HYGIENE SWEEP (Capital Recycling) ---
                    if getattr(config, 'HYGIENE_ENABLED', True):
                        # Gather data for hygiene analysis
                        funding_yields = {}
                        arb_opportunities = []
                        structure_data = {}
                        
                        # Get funding yields from Arbitrage Holon
                        if arbitrage:
                            funding_yields = getattr(arbitrage, 'funding_yields', {})
                            # Build arb opportunities list for comparison
                            for arb_sym, arb_apy in funding_yields.items():
                                if arb_apy > 50.0:  # Only positive yield opportunities
                                    arb_opportunities.append({'symbol': arb_sym, 'apy': arb_apy})
                        
                        # Get structure from StructureBoss holon (not Oracle)
                        structure_holon = self.sub_holons.get('structure')
                        if structure_holon:
                            # Use executor keys for truth (virt_keys)
                            for held_virt_key in executor.held_assets.keys():
                                pos_sym = held_virt_key.split(':')[0] if ':' in held_virt_key else held_virt_key
                                # Use htf_bias and sls_levels from StructureBoss
                                sls = structure_holon.sls_levels.get(pos_sym, {})
                                htf = structure_holon.htf_bias.get(pos_sym, 'NEUTRAL')
                                # Determine zone from sls_levels if available
                                zone = 'NEUTRAL'
                                if sls:
                                    _recycle_p = executor.latest_prices.get(pos_sym, 0)
                                    supports = sls.get('support', [])
                                    resistances = sls.get('resistance', [])
                                    if supports and _recycle_p > 0:
                                        nearest_sup = max([s for s in supports if s < _recycle_p], default=0)
                                        if nearest_sup > 0 and (_recycle_p - nearest_sup) / _recycle_p < 0.015:
                                            zone = 'SUPPORT'
                                structure_data[held_virt_key] = {'zone': zone, 'bias': htf}
                        
                        # Run the hygiene sweep
                        recycle_signals = governor.run_hygiene_sweep(
                            latest_prices=executor.latest_prices,
                            funding_yields=funding_yields,
                            structure_data=structure_data,
                            arb_opportunities=arb_opportunities
                        )
                        
                        # Execute recycle signals
                        for recycle in recycle_signals:
                            held_virt_key = recycle['symbol'] # Governor returns the key it iterated over
                            r_pct = recycle['close_pct']
                            r_reason = recycle['reason']
                            
                            r_sym = held_virt_key.split(':')[0] if ':' in held_virt_key else held_virt_key
                            
                            # Pull from truth
                            raw_meta_h = executor.position_metadata.get(held_virt_key)
                            if not raw_meta_h: continue

                            def _pa_h(obj, key, default=None):
                                if obj is None: return default
                                if isinstance(obj, dict): return obj.get(key, default)
                                return getattr(obj, key, default)

                            direction = _pa_h(raw_meta_h, 'direction', 'BUY')
                            recycle_price = executor.latest_prices.get(r_sym, 0.0)
                            qty_held = abs(executor.held_assets.get(held_virt_key, 0.0))
                            
                            if qty_held > 0 and recycle_price > 0:
                                print(f"[{self.name}] 🧹 HYGIENE EXIT: {held_virt_key} ({r_pct*100:.0f}%) - {r_reason}")
                                
                                hygiene_signal = TradeSignal(
                                    symbol=r_sym,
                                    direction='SELL' if direction == 'BUY' else 'BUY',
                                    size=r_pct,
                                    price=recycle_price,
                                    metadata={
                                        'reason': f'HYGIENE_{r_reason.split(":")[0]}', 
                                        'is_percent': True,
                                        'strategy': _pa_h(raw_meta_h, 'strategy', 'DIRECTIONAL')
                                    }
                                )
                                hygiene_decision = TradeDecision(
                                    action='EXECUTE',
                                    original_signal=hygiene_signal,
                                    adjusted_size=r_pct,
                                    disposition=Disposition(autonomy=1.0, integration=1.0),
                                    block_hash='HYGIENE',
                                    entropy_score=0.0
                                )
                                executor.execute_transaction(hygiene_decision, recycle_price)
                                # NOTE: Do NOT call sync_positions here — it calls .clear() which corrupts
                                # executor.positions while the hygiene loop is still iterating.
                    # --------------------------------------------------

                    # --- MONTE CARLO POSITION EVALUATION ---
                    # Run Monte Carlo evaluation for losing positions
                    if governor.monte_carlo_manager:
                        try:
                            # Get current prices for all held positions
                            current_prices = executor.latest_prices

                            # Prepare SDE data for all positions (could come from Oracle or other sources)
                            sde_data = {}
                            for held_virt_key in list(executor.held_assets.keys()):  # Snapshot to avoid dict-size-changed errors
                                pos_sym = held_virt_key.split(':')[0] if ':' in held_virt_key else held_virt_key
                                # Use default SDE parameters if not available
                                sde_data[pos_sym] = {
                                    'mu': 0.0,
                                    'sigma': 0.1,
                                    'lambda': 0.1
                                }

                                # If Oracle has SDE data, use it
                                if oracle and hasattr(oracle, 'sde_cache') and pos_sym in oracle.sde_cache:
                                    sde_data[pos_sym] = oracle.sde_cache[pos_sym]

                            # Extract base symbols from executor's virtual keys to pass to Monte Carlo
                            live_positions = {}
                            for vk, pos_data in list(executor.position_metadata.items()):
                                base_sym = vk.split(':')[0] if ':' in vk else vk
                                live_positions[base_sym] = pos_data

                            # Run Monte Carlo health check
                            monte_carlo_recommendations = governor.monte_carlo_manager.run_position_health_check(
                                positions=live_positions,
                                current_prices=current_prices,
                                sde_data=sde_data
                            )

                            # Execute Monte Carlo closure signals
                            # Check if recommendations is not None before iterating
                            if monte_carlo_recommendations is None:
                                monte_carlo_recommendations = []

                            for rec in monte_carlo_recommendations:
                                symbol = rec['symbol']
                                reason = rec['reason']
                                confidence = rec['confidence']

                                # Find the virtual key for this symbol
                                virt_key = None
                                for vk in executor.held_assets.keys():
                                    # Extract base symbol from virtual key (remove strategy suffix if present)
                                    base_vk = vk.split(':')[0] if ':' in vk else vk
                                    if base_vk == symbol:
                                        virt_key = vk
                                        break

                                if virt_key:
                                    raw_meta = executor.position_metadata.get(virt_key)
                                    # Support both Position objects and legacy dicts
                                    def _pa_mc(obj, key, default=None):
                                        if obj is None: return default
                                        if isinstance(obj, dict): return obj.get(key, default)
                                        return getattr(obj, key, default)
                                    if raw_meta:
                                        direction = _pa_mc(raw_meta, 'direction', 'BUY')
                                        _mc_p = current_prices.get(symbol, 0.0)
                                        qty_held = abs(executor.held_assets.get(virt_key, 0.0))

                                        if qty_held > 0 and _mc_p > 0:
                                            print(f"[{self.name}] 🎲 MONTE CARLO EXIT: {symbol} - {reason} (Conf: {confidence:.2%})")

                                            mc_signal = TradeSignal(
                                                symbol=symbol,
                                                direction='SELL' if direction == 'BUY' else 'BUY',
                                                size=1.0,  # Full close based on Monte Carlo assessment
                                                price=_mc_p,
                                                conviction=confidence  # Add confidence as conviction
                                            )
                                            mc_signal.metadata = {
                                                'reason': f'MONTE_CARLO_{reason.split(":")[0]}',
                                                'is_percent': True,
                                                'strategy': _pa_mc(raw_meta, 'strategy', 'DIRECTIONAL'),
                                                'monte_carlo_confidence': confidence
                                            }
                                            mc_decision = TradeDecision(
                                                action='EXECUTE',
                                                original_signal=mc_signal,
                                                adjusted_size=1.0,
                                                disposition=Disposition(autonomy=1.0, integration=1.0),
                                                block_hash='MONTE_CARLO',
                                                entropy_score=0.0
                                            )
                                            executor.execute_transaction(mc_decision, _mc_p)
                                            governor.sync_positions(executor.held_assets, executor.position_metadata)

                        except Exception as e:
                            print(f"[{self.name}] Monte Carlo evaluation error: {e}")
                            traceback.print_exc()
                    # -------------------------------------

                    # --- ENHANCED HYGIENE EXECUTION ---
                    # Ensure hygiene signals are properly executed
                    if governor and executor:
                        # Get current positions to compare with hygiene signals
                        for held_virt_key, held_qty in list(executor.held_assets.items()):
                            if abs(held_qty) < 0.0000001: continue

                            # Extract Real Symbol for price and cache lookups
                            held_sym = held_virt_key.split(':')[0] if ':' in held_virt_key else held_virt_key

                            # Check if this position should be closed based on hygiene
                            pos_data = executor.position_metadata.get(held_virt_key)
                            if pos_data:
                                _status_p = executor.latest_prices.get(held_sym, 0.0)
                                if _status_p > 0:
                                    # Get funding yields for this specific symbol
                                    current_funding_yields = getattr(arbitrage, 'funding_yields', {}) if arbitrage else {}
                                    symbol_funding = {held_sym: current_funding_yields.get(held_sym, 0.0)} if held_sym in current_funding_yields else {}

                                    # Run individual position hygiene check
                                    hygiene_result = governor.check_position_hygiene(
                                        symbol=held_virt_key,
                                        current_price=_status_p,
                                        funding_yields=symbol_funding,  # Only pass funding for this symbol
                                        structure_data={},
                                        arb_opportunities=[]
                                    )

                                    if hygiene_result['action'] == 'RECYCLE':
                                        r_pct = hygiene_result['close_pct']
                                        r_reason = hygiene_result['reason']

                                        print(f"[{self.name}] 🧹 ENHANCED HYGIENE EXIT: {held_virt_key} ({r_pct*100:.0f}%) - {r_reason}")

                                        # Support both Position objects and legacy dicts
                                        def _pa(obj, key, default=None):
                                            if obj is None: return default
                                            if isinstance(obj, dict): return obj.get(key, default)
                                            return getattr(obj, key, default)

                                        direction = _pa(pos_data, 'direction', 'BUY')
                                        hygiene_signal = TradeSignal(
                                            symbol=held_sym,
                                            direction='SELL' if direction == 'BUY' else 'BUY',
                                            size=r_pct,
                                            price=current_price,
                                            metadata={
                                                'reason': f'ENHANCED_HYGIENE_{r_reason.split(":")[0]}',
                                                'is_percent': True,
                                                'strategy': _pa(pos_data, 'strategy', 'DIRECTIONAL')
                                            }
                                        )
                                        hygiene_decision = TradeDecision(
                                            action='EXECUTE',
                                            original_signal=hygiene_signal,
                                            adjusted_size=r_pct,
                                            disposition=Disposition(autonomy=1.0, integration=1.0),
                                            block_hash='ENHANCED_HYGIENE',
                                            entropy_score=0.0
                                        )
                                        executor.execute_transaction(hygiene_decision, _status_p)
                                        # NOTE: Do NOT call sync_positions here — it calls .clear() which corrupts
                                        # executor.positions while the outer symbol loop is still iterating.
                    # -------------------------------------

                    # --- ADDITIONAL TOXIC FUNDING CHECK ---
                    # Perform additional check for toxic funding positions that may have been missed
                    if governor and executor and arbitrage:
                        toxic_funding_yields = getattr(arbitrage, 'funding_yields', {})
                        toxic_threshold = getattr(config, 'HYGIENE_TOXIC_FUNDING_APY', -200.0)
                        processed_toxic = set()  # Prevent multiple exits per symbol per cycle

                        # Cross-cycle cooldown: avoids re-firing when exit consistently fails (e.g. NO_LIQUIDITY)
                        if not hasattr(self, '_toxic_exit_cooldown'):
                            self._toxic_exit_cooldown = {}
                        toxic_cooldown_secs = getattr(config, 'TOXIC_EXIT_COOLDOWN_SECS', 90)

                        for toxic_sym, apy in list(toxic_funding_yields.items()):
                            # Skip if already processed this cycle, or cooled down from a recent attempt
                            last_toxic_attempt = self._toxic_exit_cooldown.get(toxic_sym, 0)
                            if toxic_sym in processed_toxic or (time.time() - last_toxic_attempt) < toxic_cooldown_secs:
                                continue

                            if apy < toxic_threshold:  # Paying high negative funding
                                # Check if we hold this symbol
                                for held_virt_key, held_qty in list(executor.held_assets.items()):  # Snapshot to prevent dict mutation during iteration
                                    held_sym = held_virt_key.split(':')[0] if ':' in held_virt_key else held_virt_key

                                    if held_sym == toxic_sym and abs(held_qty) > 0.0000001:
                                        pos_data = executor.position_metadata.get(held_virt_key)
                                        # Support both Position objects and legacy dicts
                                        def _td(obj, key, default=None):
                                            if obj is None: return default
                                            if isinstance(obj, dict): return obj.get(key, default)
                                            return getattr(obj, key, default)
                                        
                                        pos_direction = _td(pos_data, 'direction', 'BUY')
                                        if pos_direction == 'BUY':  # Only for long positions paying negative funding
                                            _toxic_p = executor.latest_prices.get(toxic_sym, 0.0)
                                            if _toxic_p > 0:
                                                print(f"[{self.name}] 🚨 URGENT TOXIC FUNDING EXIT: {toxic_sym} paying {abs(apy):.0f}% APY")
                                                processed_toxic.add(toxic_sym)  # Mark as processed for this cycle
                                                self._toxic_exit_cooldown[toxic_sym] = time.time()  # Arm cooldown

                                                direction = pos_direction
                                                toxic_signal = TradeSignal(
                                                    symbol=toxic_sym,
                                                    direction='SELL' if direction == 'BUY' else 'BUY',
                                                    size=1.0,  # Full close for toxic funding
                                                    price=_toxic_p,
                                                    metadata={
                                                        'reason': f'TOXIC_FUNDING_IMMEDIATE_CLOSE_{abs(apy):.0f}',
                                                        'is_percent': True,
                                                        'strategy': _td(pos_data, 'strategy', 'DIRECTIONAL')
                                                    }
                                                )
                                                toxic_decision = TradeDecision(
                                                    action='EXECUTE',
                                                    original_signal=toxic_signal,
                                                    adjusted_size=1.0,
                                                    disposition=Disposition(autonomy=1.0, integration=1.0),
                                                    block_hash='TOXIC_FUNDING_IMMEDIATE',
                                                    entropy_score=0.0
                                                )
                                                executor.execute_transaction(toxic_decision, _toxic_p)
                                                # NOTE: Do NOT call sync_positions here — it calls .clear() which corrupts
                                                # executor.positions while the outer symbol loop is still iterating.
                                                break  # Stop checking more held_keys for this toxic symbol
                    # -------------------------------------

                # --- PHASE 7: TRANSITION FREEZE ---
                # Block new entries during regime transition
                if regime_controller and regime_controller.is_transition_pending():
                    print(f"[{self.name}] ⏸️ Regime Transition Pending. Entries PAUSED.")
                    entry_sig = None  # Override any signal

                # A. Handle Entry
                entry_sig = res.get('entry_signal') if not (regime_controller and regime_controller.is_transition_pending()) else None
                
                # --- SESSION 3 FIX: STRICT CYCLE LIMIT (Recalibration Hotfix) ---
                # Check BEFORE processing the signal
                limit_entries = getattr(config, 'MAX_ENTRIES_PER_CYCLE', 2)
                # RELAX LIMIT FOR ARB:
                if entry_sig and entry_sig.metadata.get('is_arb', False):
                    limit_entries = 5 # Allow more nuggets in one burst
                
                if entry_sig and cycle_entries_count >= limit_entries:
                     if getattr(self, 'verbose_logging', False):
                         print(f"[{self.name}] 🛑 SKIPPING ENTRY {symbol}: Cycle Limit ({limit_entries}) Reached.")
                     entry_sig = None
                
                # --- NEW: ANTI-STACKING THROTTLE ---
                # "One Bullet Per Cycle Per Asset"
                if entry_sig and executor:
                    # Satellite Logic: "Post-entry averaging, not simultaneous execution."
                    satellites = getattr(config, 'SATELLITE_ASSETS', [])
                    if symbol in satellites:
                        # Check if ANY virtual pool holds this symbol
                        is_held = any(k.startswith(f"{symbol}:") or k == symbol for k, v in executor.held_assets.items() if abs(v) > 1e-9)
                        if is_held:
                             # We hold it. This is a stack attempt.
                             # Ensure we bypass "Sniper" entry logic if it ignores stacking rules.
                             # We delegate to Governor (handled downstream by calc_position_size check for existing exposure).
                             # BUT we must set a flag or metadata to ensure it's treated as "Averaging" not "New Position".
                             # Actually, just ensuring we don't 'Force' a new trade is key.
                             entry_sig.metadata['is_stack'] = True
                             if getattr(self, 'verbose_logging', False):
                                  print(f"[{self.name}] 🛰️ SATELLITE: Existing position in {symbol}. Treating as Stacking Candidate.")
                    
                    last_trade = governor.last_trade_time.get(symbol, 0) if governor else 0
                    time_since_trade = time.time() - last_trade
                    throttle = getattr(config, 'MIN_SECONDS_BETWEEN_STACKS', 60)
                    
                    if time_since_trade < throttle:
                        # RELAX THROTTLE FOR HIGH-YIELD ARB
                        is_urgent_arb = entry_sig.metadata.get('is_arb') and abs(entry_sig.metadata.get('apy', 0)) > 100.0
                        if is_urgent_arb:
                            if getattr(self, 'verbose_logging', False):
                                print(f"[{self.name}] 🚀 ARB THROTTLE BYPASS: {symbol} yield is high ({entry_sig.metadata.get('apy'):.0f}%).")
                        else:
                            print(f"[{self.name}] 🛑 STACK THROTTLE: {symbol} traded {int(time_since_trade)}s ago. Waiting {throttle}s.")
                            entry_sig = None
                # -----------------------------------------

                # --- SESSION 3: SIGNAL DEBOUNCER (Anti-Spam) ---
                if entry_sig:
                    if not hasattr(self, 'last_signal_times'): self.last_signal_times = {}
                    last_trigger = self.last_signal_times.get(symbol, 0)
                    time_since = time.time() - last_trigger
                    
                    # 30 Minute Debounce (unless it's a specific urgent type? No, strict.)
                    if time_since < 1800:
                         print(f"[{self.name}] ⏳ DEBOUNCE: Skipping {symbol} signal (Last: {int(time_since)}s ago < 30m).")
                         entry_sig = None
                    # Update of self.last_signal_times is moved downstream to ensure
                    # we only debounce if the trade actually executes, preventing shadow bans.
                # -----------------------------------------------
                
                # --- PATCH: HARD TOPOLOGY VETO ---
                if tda_status == 'CRITICAL' and entry_sig:
                     print(f"[{self.name}] 🚨 TOPOLOGY HARD VETO: Structure Collapse detected for {symbol}. Blocking Entry.")
                     entry_sig = None
                # ---------------------------------
                
                # BLIND MODE GUARD: No Entries if we could not verify Equity
                if blind_mode: entry_sig = None

                # ── SMCE Volatility Calculation (Deferred Gate) ──
                _smce_vol = {}
                if entry_sig and governor and getattr(governor, 'smce_doctrine', None):
                    try:
                        _df = res.get('data')
                        if _df is not None and not _df.empty:
                            _closes = _df['close'].tail(96)
                            if len(_closes) > 5:
                                _ret = _closes.pct_change().dropna()
                                _smce_vol[symbol] = float(_ret.std() * (96 ** 0.5))
                    except Exception:
                        pass

                if entry_sig and executor and governor and oracle:
                    # A. Handle Entry (REFACTORED - Using trader_entry_handler module)
                    overwatch = self.sub_holons.get('overwatch')
                    
                    # Initialize last_signal_attempts if needed
                    if not hasattr(self, 'last_signal_attempts'):
                        self.last_signal_attempts = {}
                    
                    # ── ORION: Inject market-path state into signal metadata ──
                    structure_boss = self.sub_holons.get('structure')
                    if structure_boss and hasattr(structure_boss, 'get_orion_state'):
                        try:
                            _macro_oracle = getattr(oracle, 'macro_oracle', None)
                            _rsi = indicators.get('rsi', 50.0)
                            orion_state = structure_boss.get_orion_state(
                                symbol, observer, macro_oracle=_macro_oracle,
                                rsi=_rsi, entropy=entropy_val)
                            entry_sig.metadata['orion'] = orion_state
                        except Exception as _oe:
                            print(f"[{self.name}] ⚠️ Orion state error: {_oe}")

                    tda_score_val = tda_score if 'tda_score' in locals() else 0.5
                    
                    executed, cycle_entries_count, entry_result = handle_entry(
                        symbol=symbol,
                        entry_sig=entry_sig,
                        executor=executor,
                        governor=governor,
                        oracle=oracle,
                        ppo=ppo,
                        overwatch=overwatch,
                        current_price=current_price,
                        indicators=indicators,
                        regime=regime,
                        entropy_val=entropy_val,
                        sent_score=sent_score,
                        global_bias=global_bias,
                        cycle_entries_count=cycle_entries_count,
                        limit_entries=limit_entries,
                        last_signal_attempts=self.last_signal_attempts,
                        tda_score=tda_score_val,
                        tda_status=tda_status,
                        holon_name=self.name,
                        structure_bias=res.get('macro_trend', 'NEUTRAL'),  # Trend alignment filter
                        smce_vol=_smce_vol if '_smce_vol' in locals() else {},
                        cooldown_tracker=self.scout_cooldown_tracker  # Pass cooldown tracker for scout rotation
                    )
                    
                    row_data.update(entry_result)
                    if executed:
                        self.last_signal_times[symbol] = time.time()
                        if ppo:
                            self.last_ppo_conviction = entry_sig.metadata.get('ppo_conviction', 0.5)
                    
                else:
                    # No entry signal - set WAIT action for GUI clarity
                    if not executor.held_assets.get(symbol) if executor else True:
                        if not row_data.get('Action'):
                            row_data['Action'] = "WAIT"
                            if self.verbose_logging:
                                print(f"[{self.name}] 💤 No Entry {symbol}: Regime={regime} (Expected)")

                # B. Handle Exit (REFACTORED - Using trader_exit_handler module)
                guardian_exit = res.get('guardian_exit')
                
                # Determine exit signal using handler
                final_exit, reason, thesis_valid = determine_exit_signal(
                    symbol, guardian_exit, executor, oracle, current_price, TradeSignal
                )
                
                if final_exit and executor:
                    memory = self.sub_holons.get('memory')
                    overwatch = self.sub_holons.get('overwatch')
                    
                    exit_result = handle_exit(
                        symbol=symbol,
                        final_exit=final_exit,
                        reason=reason,
                        executor=executor,
                        guardian=guardian,
                        ppo=ppo,
                        memory=memory,
                        overwatch=overwatch,
                        current_price=current_price,
                        regime=regime,
                        entropy_val=entropy_val,
                        holon_name=self.name
                    )
                    
                    row_data.update(exit_result)
                    if '_ppo_reward' in exit_result:
                        self.last_ppo_reward = exit_result['_ppo_reward']

            except Exception as e:
                print(f"[{self.name}] ❌ Error processing {symbol}: {e}")
                traceback.print_exc()

            # --- LIQUIDITY & HEALTH MONITOR ---
            # If we hold a position, check its liquidity health on the EXECUTION VENUE
            qty_held = executor.held_assets.get(symbol, 0.0) if executor else 0.0
            actuator = self.sub_holons.get('executor', {}).actuator if executor else None
            
            # Note: Executor holds the Actuator reference
            
            if abs(qty_held) > 0.00000001 and guardian and actuator:
                try:
                    # Fetch live book from KRAKEN FUTURES (via Actuator)
                    book = actuator.fetch_order_book(symbol)
                    
                    # Determine Exit Direction for check
                    exit_dir = 'SELL' if qty_held > 0 else 'BUY'
                    liq_status = guardian.check_liquidity_health(symbol, exit_dir, abs(qty_held), book)
                    
                    if liq_status != "HEALTHY":
                        warn_msg = f"⚠️ LIQUIDITY WARNING for {symbol}: {liq_status}"
                        print(f"[{self.name}] {warn_msg}")
                        row_data['Note'] = liq_status
                        
                except Exception as e:
                    # print(f"[{self.name}] LiqCheck error: {e}")
                    pass

            cycle_report.append(row_data)

        # --- PHASE 3: AGGREGATE & UI ---
        if entropies and self.sub_holons.get('entropy'):
            avg_e = sum(entropies) / len(entropies)
            self.market_state['entropy'] = avg_e
            self.market_state['regime'] = self.sub_holons['entropy'].determine_regime(avg_e)

        # Removed redundant _print_summary call
        if monitor and executor: 
            exec_summary = executor.get_execution_summary()
            is_solvent = monitor.update_health(exec_summary, get_performance_data())
            
            if not is_solvent:
                # TRIGGER LIQUIDATION
                print(f"[{self.name}] 📞 MARGIN CALL RECEIVED. LIQUIDATING...")
                executor.panic_close_all(executor.latest_prices)
                self.last_ppo_reward = -100.0 # Severe Penalty for Liquidation
                # Maybe pause for a bit?
                time.sleep(5)

        # Run Overwatch Audit
        if self.sub_holons.get('overwatch'):
            try: self.sub_holons['overwatch'].perform_audit()
            except Exception as e: print(f"[{self.name}] ⚠️ Overwatch Error: {e}")

        # Sync Order Flow Data for Dashboard
        self._sync_order_flow_to_disk()
        
        # --- PHASE 4: CONDITIONAL PROMOTION CHECK ---
        # self.check_promotion_eligibility() # TEMPORARILY DISABLED (Stabilization Phase)

        self.publish_agent_status()
        return cycle_report

    def check_promotion_eligibility(self):
        """
        PHASE 4: Conditional Promotion Logic.
        Checks if we can graduate from Phase 1 (Concentrated) to Phase 2 (Full List).
        """
        # Only relevant if we are in Restricted Mode
        if not hasattr(config, 'PHASE1_CONCENTRATED_ASSETS'): return
        
        # Check 1: Capital Growth (Simple Check)
        # If we have grown capital by > 5% ($262.50), we consider promotion.
        current_equity = self.sub_holons['executor'].total_value if 'executor' in self.sub_holons else 0.0
        
        target = config.INITIAL_CAPITAL * 1.05
        
        if current_equity > target:
             print(f"[{self.name}] 🏆 PROMOTION ELIGIBILITY CHECK: Capital ${current_equity:.2f} > Target ${target:.2f}")
             
             # Check 2: Win Rate (Quality)
             # We need to ask Governor or DB Manager
             win_rate = 0.0
             if 'governor' in self.sub_holons:
                 stats = self.sub_holons['governor'].get_portfolio_health()
                 # stats doesn't have win rate. Governor doesn't track it natively effectively without DB.
                 # Let's assume we can peek at DB via Executor or Governor
                 pass
                 
             # For now, simplest promotion: Capital Growth + No Drawdown
             dd_pct = self.sub_holons['governor'].drawdown_pct
             if dd_pct < 0.10:
                 print(f"[{self.name}] 🚀 CONDITIONAL PROMOTION GRANTED! Unlocking Full Whitelist.")
                 
                 # UNLOCK
                 self.active_session_whitelist = list(set(self.active_session_whitelist + config.ACTIVE_WATCHLIST))
                 
                 # Persist to disk so we don't regress on restart?
                 # Or maybe we want to re-prove it each session? 
                 # Let's re-prove for now (safer).
                 
                 # Note: We don't delete PHASE1_CONCENTRATED_ASSETS from config, we just ignore it for the rest of session by expanding list.
                 self._sync_whitelist_to_disk()
             else:
                  if self.cycle_counter % 50 == 0:
                      print(f"[{self.name}] 🔒 Promotion Deferred: Drawdown {dd_pct:.1%} >= 10%.")
        else:
             if self.cycle_counter % 50 == 0:
                 print(f"[{self.name}] 🔒 Promotion Eligibility Status: ${current_equity:.2f} < ${target:.2f} (Target) | DD: {self.sub_holons['governor'].drawdown_pct:.1%}")

    def _analyze_asset(self, symbol, data, df_1h, global_bias, book_data, funding_rate, rust_sigs=None):
        # Sanitize symbol for external calls (e.g., KRAKEN FUTURES doesn't know about :ARBITRAGE_GOLD)
        base_sym = symbol.split(':')[0] if ':' in symbol else symbol
        
        observer = self.sub_holons.get('observer')
        if data is None and observer:
             # Fallback for manual/single calls (not in batch cycle)
            try: data = observer.fetch_market_data(limit=100, symbol=base_sym)
            except: return None
        if data is None: return None
        
        # SAFETY CHECK: Ensure columns exist
        required_cols = ['close', 'high', 'low', 'open', 'volume']
        if not all(col in data.columns for col in required_cols):
             print(f"[{self.name}] ⚠️ Data Validation Error for {symbol}. Missing Columns. Keys: {data.columns.tolist()}")
             return None

        # Dashboard Layout Alignment
        current_close = data['close'].iloc[-1]
        if current_close < 0.01:
             price_str = f"{current_close:.8f}"
        elif current_close < 1.0:
             price_str = f"{current_close:.6f}"
        else:
             price_str = f"{current_close:.2f}"
            
        row_data = {
            'Symbol': symbol, 
            'Price': price_str, 
            'Regime': '?', 
            'Struct': '-',
            'Entropy': '0.000',
            'RSI': '-',
            'LSTM': '0.50',
            'XGB': '0.50',
            'PnL': '-', 
            'Action': 'HOLD', 
            'Note': ''
        }
        indicators = {}
        
        # Calculate TR (Always available as Series for downstream RL logic)
        tr = pd.concat([(data['high']-data['low']), (data['high']-data['close'].shift()).abs(), (data['low']-data['close'].shift()).abs()], axis=1).max(axis=1)
        
        current_price = data['close'].iloc[-1]
        # Safe Access for Rust Signals
        safe_rust = rust_sigs if rust_sigs else {}
        entropy_val = safe_rust.get('entropy', 0.0)
        
        # ... existing variables ...
        regime = 'TRANSITION'
        tda_score = 0.5
        tda_status = 'STABLE'
        obv_slope = 0.0
        bb_vals = {'upper': current_price, 'middle': current_price, 'lower': current_price}
        
        if current_price <= 0:
             print(f"[{self.name}] ⚠️ Price Data Invalid for {symbol}: {current_price}. Skipping.")
             return None

        # COMPUTE WASTE FIX: Profiling cache to reduce redundant logs
        import time
        now = time.time()
        profile_key = f"{symbol}_{len(data)}_{current_price:.6f}"
        cache_entry = self._profile_cache.get(symbol)

        # Only log profiling if cache miss or significant price change (>0.5%)
        should_log = False
        if not cache_entry or (now - cache_entry.get('ts', 0)) > self._profile_cache_ttl:
            should_log = True
            self._profile_cache[symbol] = {'ts': now, 'price': current_price, 'len': len(data)}
        elif cache_entry:
            price_change = abs(current_price - cache_entry['price']) / cache_entry['price']
            if price_change > 0.005:  # 0.5% change
                should_log = True
                cache_entry['ts'] = now
                cache_entry['price'] = current_price

        # PROFILING LOG (User Request) - Now cached
        if should_log and symbol in ['SOL/USDT', 'XRP/USDT', 'BTC/USDT', 'XTZ/USDT', 'TBTC/USDT', 'PAXG/USDT']:
             scout_res = getattr(self, 'scout_results', {}) or {}
             pers = scout_res.get(symbol, "Unknown")
             print(f"[{self.name}] 🕵️ PROFILING {symbol}: Price ${current_price:.8f} | Personality: {pers} | Rows: {len(data)}")
        
        # --- PATCH: MULTI-TIMEFRAME & POLYMARKET CONTEXT ---
        # 1. Calculate Minutes into Candle (15m)
        last_ts = data['timestamp'].iloc[-1]
        current_time = pd.Timestamp.now(tz=timezone.utc if getattr(last_ts, 'tzinfo', None) else None)
        delta_mins = (current_time - last_ts).total_seconds() / 60.0
        
        # 2. Get 1h Trend (Macro)
        macro_trend = 'NEUTRAL'
        if df_1h is not None and not df_1h.empty:
            # User Request: Stable Trend Filter using 20 EMA
            ema_trend = df_1h['close'].ewm(span=20, adjust=False).mean().iloc[-1]
            macro_trend = 'BULLISH' if df_1h['close'].iloc[-1] > ema_trend else 'BEARISH'
        elif observer:
            # Fallback (Slow)
            try:
                df_1h = observer.fetch_market_data(timeframe='1h', limit=100, symbol=symbol)
                if not df_1h.empty:
                    ema_trend = df_1h['close'].ewm(span=20, adjust=False).mean().iloc[-1]
                    macro_trend = 'BULLISH' if df_1h['close'].iloc[-1] > ema_trend else 'BEARISH'
            except: pass
        
        structure_ctx = {
            'minutes_into_candle': delta_mins,
            'macro_trend': macro_trend
        }
        

        
        
        entropy_agent, oracle = self.sub_holons.get('entropy'), self.sub_holons.get('oracle')
        guardian, governor = self.sub_holons.get('guardian'), self.sub_holons.get('governor')
        executor = self.sub_holons.get('executor')
        topology = self.sub_holons.get('topology') # <--- AEHML 2.0
        
        if rust_sigs:
            entropy_val = rust_sigs.get('shannon_entropy', 0.0)
            # GUARD: If rust entropy is unreasonably high (>2.1), it was likely
            # computed on raw prices instead of returns. Recompute from returns.
            # NOTE: 10-bin Shannon H(nats) max = ln(10) ≈ 2.303; normal returns ≈ 1.5-2.0
            if entropy_val > 2.1 and entropy_agent and not data.empty:
                try:
                    returns = data['close'].pct_change().dropna()
                    entropy_val = entropy_agent.calculate_shannon_entropy(returns)
                except Exception:
                    pass  # Keep the rust value as fallback
            regime = entropy_agent.determine_regime(entropy_val) if entropy_agent else 'UNKNOWN'
            # SCOUTER FALLBACK (2026-03-19): SampleEntropy (structural complexity) and
            # Shannon entropy (return distribution) measure different market properties.
            # If Shannon says CHAOTIC but the scouter's structural analysis says
            # ORDERED or TRANSITION, trust the less restrictive reading so we don't
            # blanket-block assets that have genuine price structure.
            scout_res = getattr(self, 'scout_results', {}) or {}
            scout_data = scout_res.get(symbol) or {}
            scouter_regime = scout_data.get('regime')
            if regime == 'CHAOTIC' and scouter_regime in ('ORDERED', 'TRANSITION'):
                regime = scouter_regime  # Defer to structural analysis
                # Propagate scout's SampleEntropy so downstream gates see consistent value
                scout_ent = scout_data.get('entropy')
                if scout_ent is not None:
                    entropy_val = float(scout_ent)
                row_data['Entropy'] = f"{entropy_val:.3f}"
                row_data['Regime'] = f"{regime}(scout)"
            else:
                row_data['Entropy'] = f"{entropy_val:.3f}"
                row_data['Regime'] = regime
            structure_ctx['entropy_val'] = entropy_val
            structure_ctx['entropy_regime'] = regime
            structure_ctx['entropy_type'] = 'sample' if scouter_regime and regime == scouter_regime else 'shannon'

            # === PHASE 2: ONNX TREND PREDICTION INJECTION ===
            if config.USE_ONNX and entropy_agent:
                try:
                    returns = data['close'].pct_change().dropna()
                    onnx_trend_prob = entropy_agent.predict_trend(returns)
                    row_data['LSTM'] = f"{onnx_trend_prob:.2f}"
                    structure_ctx['onnx_trend_prob'] = onnx_trend_prob
                except Exception as e:
                    pass  # Silent fallback
        elif entropy_agent:
            # Fallback
            returns = data['close'].pct_change().dropna()
            entropy_val = entropy_agent.calculate_shannon_entropy(returns)
            regime = entropy_agent.determine_regime(entropy_val)
            # SCOUTER FALLBACK: same logic as Rust path above
            scout_res = getattr(self, 'scout_results', {}) or {}
            scout_data = scout_res.get(symbol) or {}
            scouter_regime = scout_data.get('regime')
            if regime == 'CHAOTIC' and scouter_regime in ('ORDERED', 'TRANSITION'):
                regime = scouter_regime
                scout_ent = scout_data.get('entropy')
                if scout_ent is not None:
                    entropy_val = float(scout_ent)
            row_data['Entropy'], row_data['Regime'] = f"{entropy_val:.3f}", regime
            structure_ctx['entropy_val'] = entropy_val
            structure_ctx['entropy_regime'] = regime
            structure_ctx['entropy_type'] = 'sample' if scouter_regime and regime == scouter_regime else 'shannon'
            
            # === PHASE 2: ONNX TREND PREDICTION INJECTION ===
            if config.USE_ONNX:
                try:
                    onnx_trend_prob = entropy_agent.predict_trend(returns)
                    row_data['LSTM'] = f"{onnx_trend_prob:.2f}"
                    structure_ctx['onnx_trend_prob'] = onnx_trend_prob
                except Exception as e:
                    pass  # Silent fallback

        # AEHML 2.0: Topological Check
        if topology:
            tda_res = topology.analyze_structure(data)
            tda_score = tda_res.get('score', 0.5)
            tda_status = tda_res.get('status', 'STABLE')
            
            # Wire to Dashboard
            row_data['Struct'] = f"{tda_status} ({tda_score:.2f})"
            
            # If topology is collapsing, override regime display to warn user
            if tda_status == 'CRITICAL':
                row_data['Regime'] = f"CRASH WARNING (TDA {tda_score:.2f})"
                structure_ctx['tda_critical'] = True

        # Structure Scan (CTKS Integration)
        structure = self.sub_holons.get('structure')
        if structure:
             ctx = structure.get_structural_context(symbol, observer)
             structure_ctx.update(ctx)
             # Append, don't overwrite
             current_struct = row_data.get('Struct', '-')
             row_data['Struct'] = f"{current_struct} | {ctx.get('sls_zone', 'N')}"

             # Orion Path display (if available from cache)
             if hasattr(structure, '_orion_cache'):
                 _orion = structure._orion_cache.get(symbol, {}).get('state', {})
                 if _orion:
                     _path = _orion.get('path', '?')
                     _conf = _orion.get('confidence', '?')
                     row_data['Struct'] += f" | 🧭{_path}({_conf[0]})"
        elif oracle:
             # Fallback (Legacy)
             base_ctx = oracle.get_structural_context(symbol, data, current_price) if hasattr(oracle, 'get_structural_context') else {}
             structure_ctx.update(base_ctx)

        # Indicators (RUST ACCELERATED)
        if rust_sigs:
            rsi_val = rust_sigs.get('rsi', 50.0)
            atr = rust_sigs.get('atr', 0.0)
            bb_vals = {
                'upper': rust_sigs.get('bb_upper', current_price),
                'middle': current_price, # Middle not explicitly in my rust batch but can be added or ignored
                'lower': rust_sigs.get('bb_lower', current_price)
            }
            row_data['RSI'] = f"{rsi_val:.1f}"
            indicators['rsi'] = rsi_val
            indicators['atr'] = atr
            indicators['bb'] = bb_vals
            # Add SDE Params for Physics Oracle
            for k, v in rust_sigs.items():
                if k.startswith('ou_'):
                    indicators[k] = v
        else:
            # LEGACY FALLBACK (Pandas)
            # tr already calculated above

            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            
            # Divide by zero safety
            with np.errstate(divide='ignore', invalid='ignore'):
                rs = gain / loss
                rsi_val = 100 - (100 / (1 + rs)).iloc[-1]
            
            row_data['RSI'] = f"{rsi_val:.1f}"
            indicators['rsi'] = rsi_val

            rolling_mean, rolling_std = data['close'].rolling(20).mean(), data['close'].rolling(20).std()
            bb_vals = {
                'upper': (rolling_mean + 2*rolling_std).iloc[-1], 
                'middle': rolling_mean.iloc[-1], 
                'lower': (rolling_mean - 2*rolling_std).iloc[-1]
            }
            indicators['bb'] = bb_vals
            
            # Correctly handle ATR in fallback
            if tr is not None:
                atr = tr.rolling(14).mean().iloc[-1]
            else:
                atr = 0.0
            indicators['atr'] = atr
        
        obv = (np.sign(data['close'].diff()).fillna(0) * data['volume']).cumsum()
        obv_slope, _, _, _, _ = linregress(np.arange(14), obv.iloc[-14:].values)

        metabolism = 'PREDATOR' if executor and executor.get_portfolio_value(current_price) > config.SCAVENGER_THRESHOLD else 'SCAVENGER'
        
        # --- HOISTED: DATA PREP & WHALE DETECTION (REC-2026-01-31) ---
        # We need this knowledge BEFORE asking Governor for permission (to allow overrides)
        
        # 1. Fetch Data Wrappers
        if book_data is None and observer:
            try: book_data = observer.fetch_order_book(symbol)
            except: book_data = {}
        
        if funding_rate == 0.0 and observer:
            try: funding_rate = observer.fetch_funding_rate(symbol)
            except: funding_rate = 0.0
            
        # 2. Whale Detection - ROUTE THROUGH ORACLE
        whale = self.sub_holons.get('whale')
        oracle = self.sub_holons.get('oracle')
        is_whale_signal = False
        if whale:
            # Calculate Daily Volume (Approx) for Dynamic Thresholds
            daily_vol_usd = 0.0
            try:
                recent = data.iloc[-96:] if len(data) >= 96 else data
                daily_vol_usd = (recent['close'] * recent['volume']).sum()
            except:
                daily_vol_usd = 0.0

            # Check for "Whale-Scalper" Setup (Bid Wall)
            is_whale_signal = whale.check_bid_wall(symbol, book_data, daily_vol=daily_vol_usd)

            # FIX 2026-03-21: Throttle whale signals — same symbol can only fire once per 300s
            if is_whale_signal:
                cooldown_until = self._whale_forced_cooldown.get(symbol, 0)
                if time.time() < cooldown_until:
                    is_whale_signal = None  # Suppress duplicate detection

            # FIX 2026-03-14: Route whale signals through oracle for holonic processing
            if is_whale_signal and oracle and hasattr(oracle, 'process_holonic_signal'):
                is_whale_signal = oracle.process_holonic_signal(
                    symbol=symbol,
                    signal_type='WHALE_BID_WALL',
                    signal_data=is_whale_signal,
                    market_data={'prices': data['close'].values, 'volumes': data['volume'].values}
                )

            if is_whale_signal:
                print(f"[{self.name}] 🐋 WHALE SIGNAL DETECTED: {symbol} (Bid Wall)")

        # 3. Calculate Funding Yield (APY Approx) for Poison/Override
        # FIX: Use capped value from ArbitrageHolon to prevent extreme APY values (e.g., 18955%)
        arb_holon = self.sub_holons.get('arbitrage')
        if arb_holon and hasattr(arb_holon, 'funding_yields'):
            funding_apy = arb_holon.funding_yields.get(symbol, 0.0)
        else:
            # Fallback: Calculate with sanity cap (±200% APY)
            raw_apy = abs(funding_rate) * 3 * 365 * 100
            funding_apy = np.sign(funding_rate) * min(abs(raw_apy), 200.0)

        # 4. PRE-CHECK: Arbitrage Opportunity (Layer 0) - ROUTE THROUGH ORACLE
        # We check this EARLY to allow Governor Bypass
        arb_signal_raw = getattr(arb_holon, 'get_active_signal', lambda x, y: None)(symbol, current_price)

        # FIX 2026-03-14: Route arb signals through oracle for holonic processing
        if arb_signal_raw and oracle and hasattr(oracle, 'process_holonic_signal'):
            arb_signal = oracle.process_holonic_signal(
                symbol=symbol,
                signal_type='ARB_FUNDING',
                signal_data=arb_signal_raw,
                market_data={'prices': data['close'].values, 'volumes': data['volume'].values}
            )
        else:
            arb_signal = arb_signal_raw

        is_arb_opportunity = bool(arb_signal)
        if is_arb_opportunity:
             print(f"[{self.name}] 💰 ARB OPPORTUNITY: {symbol}")

        entry_sig = None
        
        # CHECK: Governor Permission (With Overrides)
        allowed = True
        if governor:
             allowed = governor.is_trade_allowed(symbol, current_price, is_whale=is_whale_signal, funding_yield=funding_apy, is_arb=is_arb_opportunity)
        
        if allowed and oracle:
            last_exit = guardian.last_exit_times.get(symbol) if guardian else None
            
            # ...
            
            # --- INTEGRATION: ARBITRAGE HUNTER (PHASE 46.5) ---
            # We already checked arb_signal above, just use it
            if arb_signal:
                # FIX 4: Upstream Precheck
                if governor and hasattr(governor, 'precheck'):
                    # Estimate margin (e.g. $10 min or based on balance)
                    if not governor.precheck(symbol, arb_signal['direction'], metadata={'is_arb': True, 'funding_yield': funding_apy}):
                        print(f"[{self.name}] 🚫 ARB STRIKE ABORTED: Governor precheck failed for {symbol}.")
                        arb_signal = None
                
                if arb_signal:
                    # FIX 2026-03-03: Check minimum funding rate before strike
                    # Signal has 'gross_yield_8h' in % (e.g., 0.025 = 0.025% per 8H)
                    funding_8h = arb_signal.get('gross_yield_8h', 0.0)
                    # Convert to APY: 0.025%/8H * 3 * 365 = ~27% APY
                    funding_apy = abs(funding_8h) * 3 * 365  # 8H to APY conversion
                    min_funding = getattr(config, 'MIN_FUNDING_PER_8H', 0.10)  # 10% per 8H = 3650% APY
                    min_apy_threshold = min_funding * 3 * 365  # Convert to APY
                    
                    if funding_apy < min_apy_threshold:
                        print(f"[{self.name}] 🚫 ARB SKIPPED: {symbol} funding {funding_apy:.1f}% APY ({funding_8h:.3f}%/8H) < {min_apy_threshold:.1f}% minimum")
                        arb_signal = None
                    else:
                        print(f"[{self.name}] 💰 ARB HUNTER STRIKE: {symbol} -> {arb_signal['direction']} ({arb_signal['reason']}, {funding_apy:.1f}% APY)")
                        # Force Entry Signal
                        entry_sig = TradeSignal(
                            symbol=symbol,
                            direction=arb_signal['direction'],
                            size=1.0,
                            price=current_price,
                            conviction=arb_signal['confidence'],
                            metadata={
                                'reason': arb_signal['reason'],
                                'is_whale': True,
                                'is_arb': True,  # PHASE 2: Flag for Governor Bypass
                                'strategy': 'ARBITRAGE' # Ensure alignment with compliance bounds
                            }
                        )
            else:
                # --- STRATEGY: VOLATILITY COMPRESSION (User Request) ---
                if not entry_sig:
                     try:
                         entry_sig = oracle.analyze_volatility_compression(symbol, data)
                         if entry_sig:
                             # Reinforce reason
                             if 'reason' not in entry_sig.metadata:
                                 entry_sig.metadata['reason'] = 'VOLATILITY_SQUEEZE'

                             # FIX 2026-03-16: Add structure data for Structure Boss veto
                             # VOLATILITY_COMPRESSION signals must go through structure check
                             if 'structure' not in entry_sig.metadata:
                                 structure_metadata = {'sls_zone': 'NEUTRAL'}
                                 structure_boss = self.sub_holons.get('structure')
                                 if structure_boss and observer:
                                     s_bias, s_zone, s_sup, s_res, s_piv = structure_boss.get_structure(symbol, observer)
                                     structure_metadata['sls_zone'] = s_zone
                                     structure_metadata['macro_trend'] = s_bias
                                 entry_sig.metadata['structure'] = structure_metadata
                     except Exception as e:
                         print(f"[{self.name}] ⚠️ Volatility Strategy Error: {e}")
                
                # --- STANDARD ENTRY ANALYSIS ---
                if not entry_sig:
                    # === FIX 2026-03-14: Sync SMCE Regime to Oracle for Macro Stack Weight Adaptation ===
                    governor = self.sub_holons.get('governor')
                    if governor and hasattr(oracle, '_current_smce_regime'):
                        oracle._current_smce_regime = governor.get_smce_regime()

                    entry_sig = oracle.analyze_for_entry(
                    symbol, data, bb_vals, obv_slope, metabolism,
                    structure_ctx=structure_ctx,
                    book_data=book_data,
                    ticker_data=getattr(self, 'session_ticker_data', {}).get(symbol, {}),
                    pack_stats=getattr(self, 'session_pack_stats', {}),
                    funding_rate=funding_rate,
                    observer=observer,
                    is_whale=is_whale_signal
                )

                # === PHASE 2: ONNX CONFIDENCE INJECTION ===
                # Weight oracle confidence by ONNX trend prediction
                if entry_sig and config.USE_ONNX:
                    onnx_prob = structure_ctx.get('onnx_trend_prob', 0.5)
                    # Blend: 70% oracle, 30% ONNX (preserves oracle logic while adding ML signal)
                    blended_confidence = (0.7 * entry_sig.conviction) + (0.3 * onnx_prob)
                    entry_sig.conviction = blended_confidence
                    entry_sig.metadata['onnx_trend_prob'] = onnx_prob
                    entry_sig.metadata['conviction_blended'] = True

            # === ACCOUNT-AWARE FUNDING YIELD ===
            if entry_sig:
                # funding_apy is positive if longs pay shorts.
                # If BUYING and funding_apy > 0, we pay (- yield).
                projected_yield_apy = funding_apy if entry_sig.direction == 'SELL' else -funding_apy
                entry_sig.metadata['projected_yield_apy'] = projected_yield_apy
                    
                if projected_yield_apy < -50.0:
                    entry_sig.conviction *= 0.8 # 20% penalty for toxic funding
                    entry_sig.metadata['reason'] = entry_sig.metadata.get('reason', 'SYSTEM') + ' (Toxic Funding)'
                    if getattr(self, 'verbose_logging', False):
                        print(f"[{self.name}] ⚠️ TOXIC FUNDING for {symbol}: Projected {projected_yield_apy:.1f}% APY. Conviction penalized.")
                        
                elif projected_yield_apy > 50.0:
                    entry_sig.conviction = min(1.0, entry_sig.conviction * 1.1) # 10% boost
                    entry_sig.metadata['reason'] = entry_sig.metadata.get('reason', 'SYSTEM') + ' (Funding Sponsor)'
            
            # Inject Whale Signal into Oracle Metadata if Oracle missed it or to reinforce
            if entry_sig:
                 if is_whale_signal:
                     entry_sig.metadata['is_whale'] = True
                     entry_sig.metadata['reason'] = 'WHALE_SCALPER'
                     
                     # 📢 Moltbook: Announce the Leviathan
                     # We do this here (Pre-Trade) or later (Post-Trade)?
                     # Pre-Trade adds hype.
                     moltbook = self.sub_holons.get('moltbook')
                     if moltbook:
                         moltbook.post_event('WHALE_SIGHTING', {'symbol': symbol})

            elif is_whale_signal:
                 # Force Entry if Whale Detected and Oracle hasn't approved
                 print(f"[{self.name}] 🐋 WHALE FORCED ENTRY: Oracle missed the signal, but WhaleHolon detected a massive bid wall.")

                 # Fetch Structural Integrity
                 structure_metadata = {'sls_zone': 'NEUTRAL'}
                 structure_boss = self.sub_holons.get('structure')
                 s_bias, s_zone = 'NEUTRAL', 'NEUTRAL'
                 if structure_boss and observer:
                     s_bias, s_zone, s_sup, s_res, s_piv = structure_boss.get_structure(symbol, observer)
                     structure_metadata['sls_zone'] = s_zone
                     structure_metadata['macro_trend'] = s_bias
                 
                 # Require StructureBoss alignment even for forced entries (LIVE safety).
                 # Longs must be at SUPPORT (or NEUTRAL only if macro is BULLISH).
                 allow_forced = (s_zone == 'SUPPORT') or (s_zone == 'NEUTRAL' and s_bias == 'BULLISH')
                 if not allow_forced:
                     print(f"[{self.name}] 🐋🚫 WHALE FORCED ENTRY VETO: {symbol} blocked (Zone {s_zone} != SUPPORT, Trend: {s_bias})")
                     entry_sig = None
                     # FIX 2026-03-21: Set cooldown even on veto to prevent 130x/session spam
                     self._whale_forced_cooldown[symbol] = time.time() + 300
                 else:
                     
                     entry_sig = TradeSignal(
                         symbol=symbol,
                         direction='BUY',  # Bid wall equates to strong floor / buy signal
                         size=1.0,
                         conviction=0.9,   # High conviction for whales
                         metadata={
                             'is_whale': True,
                             'reason': 'WHALE_SCALPER_FORCED',
                             'strategy': 'WHALE_SCALPER',
                             'structure': structure_metadata
                         }
                     )
                     # FIX 2026-03-21: Set cooldown after forced entry to prevent spam
                     self._whale_forced_cooldown[symbol] = time.time() + 300  # 5 min cooldown
                 
                 moltbook = self.sub_holons.get('moltbook')
                 if moltbook:
                     moltbook.post_event('WHALE_SIGHTING', {'symbol': symbol})


        guardian_exit = None
        entry_p = executor.entry_prices.get(symbol, 0.0) if executor else 0.0
        pnl_pct = 0.0
        row_data['PnL'] = "0.00%"  # Default to avoid KeyError/UnboundLocalError
        if entry_p > 0 and guardian:
            def _pa_ga(obj, key, default=None):
                if obj is None: return default
                if isinstance(obj, dict): return obj.get(key, default)
                return getattr(obj, key, default)
            _raw_pmeta = executor.position_metadata.get(symbol)
            direction = _pa_ga(_raw_pmeta, 'direction', 'BUY')
            age_h = 0.0
            if executor.entry_timestamps.get(symbol):
                from datetime import datetime, timezone
                try: age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(executor.entry_timestamps[symbol])).total_seconds() / 3600
                except: pass
            # Extract RSI and Meta for Guardian
            indicators_ctx = {
                'rsi': float(row_data['RSI']) if row_data['RSI'] != '-' else 50.0,
                'atr': atr
            }
            # Safely convert Position object to a plain dict for guardian
            _pm_raw = executor.position_metadata.get(symbol)
            if isinstance(_pm_raw, dict):
                meta_ctx = _pm_raw.copy()
            elif _pm_raw is not None:
                meta_ctx = {k: v for k, v in vars(_pm_raw).items() if not k.startswith('_')}
            else:
                meta_ctx = {}
            
            guardian_exit = guardian.analyze_for_exit(
                symbol, current_price, entry_p, bb_vals, atr, metabolism, age_h, direction,
                indicators=indicators_ctx,
                meta=meta_ctx
            )
            # FIX 2026-02-28: Debug logging for BNB exit signal tracing
            if symbol == 'BNB/USDT' and self.verbose_logging:
                print(f"[{self.name}] 🔍 BNB DEBUG: guardian_exit={guardian_exit}, atr={atr}, pnl={pnl_pct*100:.2f}%")
            pnl_pct = (current_price - entry_p) / entry_p if direction == 'BUY' else (entry_p - current_price) / entry_p
            row_data['PnL'] = f"{pnl_pct*100:+.2f}%"

            # --- STACK PROFIT MANAGEMENT (Phase 2) ---
            if not guardian_exit and governor:
                # 1. Timeout Checks (Critical Fix)
                timeout_qty = governor.check_timeout_actions(symbol)
                if timeout_qty > 0:
                     qty_held = abs(executor.held_assets.get(symbol, 1.0))
                     if qty_held > 0:
                         red_pct = min(1.0, timeout_qty / qty_held)
                         print(f"[{self.name}] ⏰ TIMEOUT EXIT: {symbol} Force Reducing {timeout_qty:.4f} ({red_pct*100:.1f}%)")
                         guardian_exit = TradeSignal(
                             symbol=symbol,
                             direction='SELL' if direction == 'BUY' else 'BUY',
                             size=red_pct,
                             price=current_price,
                             metadata={'reason': 'STACK_TIMEOUT', 'is_percent': True}
                         )

                # 2. Profit Targets
                if not guardian_exit:
                    # Pass ATR for trailing stop calculation
                    atr_val = indicators.get('atr', 0.0) if 'indicators' in dir() else 0.0
                    stack_exit_qty = governor.check_stack_targets(symbol, current_price, atr_val)
                    if stack_exit_qty > 0:
                        qty_held = abs(executor.held_assets.get(symbol, 0.0))
                        if qty_held > 0:
                            exit_pct = min(1.0, stack_exit_qty / qty_held)
                            if exit_pct > 0.01: # Min 1% to avoid dust loops
                                print(f"[{self.name}] 🥞 STACK EXIT TRIGGERED: {symbol} Qty {stack_exit_qty:.4f} ({exit_pct*100:.1f}%)")
                                guardian_exit = TradeSignal(
                                    symbol=symbol, 
                                    direction='SELL' if direction == 'BUY' else 'BUY', 
                                    size=exit_pct, 
                                    price=current_price,
                                    metadata={'reason': 'STACK_TP', 'is_percent': True}
                                )
            
            # --- COMPLIANCE CHECK (Phase 3: Auto-Reduction) ---
            # FIX 2026-03-01: Compliance reductions are RISK-REDUCING and should NOT be
            # blocked by MIN_ORDER_VALUE. Even small reductions help meet limits.
            if not guardian_exit and governor:
                excess_qty = governor.check_portfolio_compliance(symbol, current_price)
                if excess_qty > 0:
                     qty_held = abs(executor.held_assets.get(symbol, 0.0))
                     if qty_held > 0:
                         # Calculate exact reduction (Absolute Quantity)
                         # min(excess vs held) to ensure we don't oversell (rare race condition)
                         reduce_qty = min(qty_held, excess_qty)

                         # FIX 2026-03-01: Skip MIN_ORDER_VALUE check for compliance reductions
                         # Risk-reducing trades should never be blocked by minimum size
                         print(f"[{self.name}] ⚖️ COMPLIANCE REDUCTION: {symbol} Closing {reduce_qty:.4f} (${reduce_qty*current_price:.2f})")
                         guardian_exit = TradeSignal(
                             symbol=symbol,
                             direction='SELL' if direction == 'BUY' else 'BUY',
                             size=reduce_qty,
                             price=current_price,
                             # FIX 2026-03-01: bypass_validation ensures compliance reductions
                             # skip ALL Governor validation layers (SMCE, min order, etc.)
                             # This is risk-reducing - always allow execution.
                             metadata={
                                 'reason': 'COMPLIANCE_REDUCE',
                                 'is_percent': False,
                                 'is_exit': True,
                                 'reduce_only': True,
                                 'bypass_validation': True  # CRITICAL: Skip all validation
                             }
                         )
            # --------------------------------------------------
            # -----------------------------------------

        # Enrichment for Dashboard
        probes = oracle.last_probes.get(symbol, {'lstm': 0.5, 'xgb': 0.5}) if oracle else {'lstm': 0.5, 'xgb': 0.5}
        row_data['LSTM'] = f"{probes['lstm']:.2f}"
        row_data['XGB'] = f"{probes['xgb']:.2f}"

        # 3D Holospace Data Injection
        row_data['_entropy'] = entropy_val
        row_data['_tda'] = tda_score
        row_data['_price'] = current_price
        row_data['_vol'] = atr if 'atr' in locals() else 0.0

        return {
            'symbol': symbol, 'data': data, 'price': current_price, 'row_data': row_data,
            'entropy_val': entropy_val, 'regime': regime, 'metabolism': metabolism,
            'entry_signal': entry_sig, 'guardian_exit': guardian_exit,
            'tda_score': tda_score, 'tda_status': tda_status,
            'indicators': {'bb_vals': bb_vals, 'obv_slope': obv_slope, 'atr': atr, 'tr': tr},
            'macro_trend': macro_trend  # For trend alignment filter
        }

    def _create_summary_layout(self, cycle_report: List[Dict]):
        oracle = self.sub_holons.get('oracle')
        sentiment = self.sub_holons.get('sentiment')
        bias = oracle.get_market_bias(sentiment_score=sentiment.current_sentiment_score if sentiment else 0.0) if oracle else 0.5
        
        # 1. Market Status Panel
        bias_color = "green" if bias >= config.GMB_THRESHOLD else ("yellow" if bias >= 0.4 else "red")
        status_text = f"[bold {bias_color}]GLOBAL MARKET BIAS: {bias:.2f}[/bold {bias_color}] | " \
                      f"Status: [{'bold green' if bias >= config.GMB_THRESHOLD else 'bold red'}]" \
                      f"{'BULLISH' if bias >= config.GMB_THRESHOLD else 'CAUTIOUS'}[/]"
        
        header = Panel(status_text, title=f"[{self.name}] Live Dashboard", border_style="blue", expand=False)

        # 2. Detail Table
        table = Table(title="Asset Register", box=box.SIMPLE_HEAD, show_lines=False)
        table.add_column("Symbol", style="cyan", no_wrap=True)
        table.add_column("Price", style="white")
        table.add_column("Regime", style="magenta")
        table.add_column("Entropy", justify="right")
        table.add_column("Brains (LSTM/XGB)", justify="center")
        table.add_column("Action", style="bold")
        table.add_column("PnL", justify="right", style="green")

        for row in cycle_report:
            probes = oracle.last_probes.get(row['Symbol'], {'lstm': 0.5, 'xgb': 0.5}) if oracle else {'lstm': 0.5, 'xgb': 0.5}
            
            # Colorize Action
            action = row['Action']
            act_style = "dim"
            if "BUY" in action: act_style = "bold green"
            elif "SELL" in action: act_style = "bold red"
            elif "HOLD" in action: act_style = "dim white"
            
            # Colorize Regime
            reg_style = "white"
            if row['Regime'] == 'CHAOTIC': reg_style = "red"
            elif row['Regime'] == 'ORDERED': reg_style = "green"
            
            table.add_row(
                row['Symbol'],
                row['Price'],
                f"[{reg_style}]{row['Regime']}[/{reg_style}]",
                row.get('Entropy', 'N/A'),
                f"{probes['lstm']:.2f} / {probes['xgb']:.2f}",
                f"[{act_style}]{action}[/{act_style}]",
                row['PnL']
            )
            
        # Combine into group (or just return a group/layout)
        from rich.console import Group
        return Group(header, table)

    def _get_signal_report(self) -> list:
        """Get cached signal report from SignalProvider (refreshes every 60s)."""
        sp = self.sub_holons.get('signal_provider')
        if not sp:
            return []
        try:
            now = time.time()
            if now - getattr(self, '_last_signal_report_time', 0) > 60:
                self._cached_signal_report = sp.generate_signal_report(self.sub_holons) or []
                self._last_signal_report_time = now
            return getattr(self, '_cached_signal_report', [])
        except Exception as e:
            print(f"[{self.name}] ⚠️ Signal Report Error: {e}")
            return []

    def publish_agent_status(self):
        # Build status data regardless of queue existence
        gov, executor = self.sub_holons.get('governor'), self.sub_holons.get('executor')
        oracle = self.sub_holons.get('oracle')
        perf = get_performance_data()
        # Real-time Valuation for Asset Allocation
        latest_prices = executor.latest_prices if executor else {}
        holdings = {'CASH': gov.balance if gov else 0.0}
        total_exp = 0.0
        
        if gov:
            for s, p in gov.positions.items():
                curr_p = latest_prices.get(s, p.entry_price)
                val = p.quantity * curr_p
                holdings[s] = val
                total_exp += val

        portfolio_val = executor.get_portfolio_value(0.0) if executor else 1.0
        
        # --- NEW: ARCHIPELAGO EVOLUTION STATUS ---
        evo_stats = {'best_fitness': '0.00', 'kings': '0'}
        try:
            hof_path = os.path.join(os.getcwd(), 'hall_of_fame.json')
            if os.path.exists(hof_path):
                import json
                with open(hof_path, 'r') as f:
                    hof_data = json.load(f)
                    if isinstance(hof_data, list) and len(hof_data) > 0:
                        best = hof_data[0]
                        evo_stats['best_fitness'] = f"{best.get('fitness', 0.0):.2f}"
                        evo_stats['kings'] = str(len(hof_data))
        except: pass
        # ------------------------------------------

        status_msg = {
            'type': 'agent_status',
            'data': {
                'gov_state': f"{gov.get_metabolism_state() if gov else 'OFFLINE'}",
                'gov_alloc': f"{config.GOVERNOR_MAX_MARGIN_PCT*100:.1f}%",
                'gov_lev': f"{config.PREDATOR_LEVERAGE}x",
                'gov_trends': str(len(gov.positions)) if gov else "0",
                'evo_fitness': evo_stats['best_fitness'],
                'evo_kings': evo_stats['kings'],
                'gov_micro': f"{'ACTIVE' if config.MICRO_CAPITAL_MODE else 'OFF'}",
                'risk_budget': f"${gov.risk_budget:.2f}" if gov else "$0.00",
                'fortress_balance': f"${gov.fortress_balance:.2f}" if gov else "$300.00",
                'solvency_status': 'SOLVENT' if (gov and gov.balance >= config.MIN_ORDER_VALUE) else 'INSOLVENT', # Explicit Warning
                'regime': self.market_state['regime'],
                'entropy': f"{self.market_state['entropy']:.4f}",
                'strat_model': 'Warp-V4 (Hybrid)',
                'kalman_active': 'True' if oracle and oracle.kalman_filters else 'False',
                'ppo_conv': f"{self.last_ppo_conviction:.2f}",
                'ppo_reward': f"{self.last_ppo_reward:.2f}",
                'sentiment_score': f"{self.sub_holons['sentiment'].current_sentiment_score:.2f}" if 'sentiment' in self.sub_holons else "0.00",
                'lstm_prob': f"{oracle.get_health().get('last_lstm', 0.5):.2f}",
                'xgb_prob': f"{oracle.get_health().get('last_xgb', 0.5):.2f}",
                'last_order': executor.last_order_details if executor else 'NONE',
                'win_rate': f"{perf.get('win_rate', 0.0):.1f}%",
                'pnl': f"${perf.get('total_pnl', 0.0):.2f}",
                'omega': f"{perf.get('omega_ratio', 0.0):.2f}",
                'exposure': f"${total_exp:.2f}",
                'margin': f"${executor.get_execution_summary()['margin_used']:.2f}" if executor else "$0.00",
                'actual_lev': f"{total_exp/portfolio_val:.2f}x",
                'holdings': holdings,
                # === Dashboard Wiring Fix ===
                'balance': gov.balance if gov else 0.0,
                'equity': portfolio_val,
                'entry_prices': {s: p.entry_price for s, p in (gov.positions.items() if gov else {})},
                'current_prices': latest_prices,
                'pending_count': len(executor.actuator.pending_orders) if executor and getattr(executor, 'actuator', None) else 0,
                'news_feed': self.sub_holons['sentiment'].latest_news if 'sentiment' in self.sub_holons else [],
                # === Regime/Health Data ===
                'health_score': self.sub_holons['regime'].get_status_summary().get('health_score', 0.0) if 'regime' in self.sub_holons else 0.0,
                'promo_progress': self.sub_holons['regime'].get_status_summary().get('promotion_progress', 0.0) if 'regime' in self.sub_holons else 0.0,
                # === Consolidation Radar ===
                'scout_data': [
                    {
                        'symbol': s, 
                        'score': 0.95, 
                        'reason': 'ROCKET' if p.get('regime') == 'ORDERED' else ('ANCHOR' if p.get('regime') == 'TRANSITION' else 'DEAD')
                    } 
                    for s, p in self.scout_results.items()
                ],
                'consolidation_data': [
                    {'symbol': r[0], 'score': float(r[1]), 'reason': r[2]} 
                    for r in (oracle.get_consolidation_rankings()[:10] if oracle and hasattr(oracle, 'get_consolidation_rankings') else [])
                    if isinstance(r, (list, tuple)) and len(r) >= 3
                ],
                # === Signal Provider Report ===
                'signal_report': self._get_signal_report(),
            }
        }
        
        # Send to queue if available
        if self.gui_queue:
            self.gui_queue.put(status_msg)
        
        # ALWAYS write to file for external dashboard sync
        try:
            import json
            from datetime import datetime, timezone
            status_file_path = os.path.join(os.path.dirname(__file__), '..', 'dashboard_status.json')
            
            # Add timestamp and flatten for file format
            file_data = status_msg.get('data', {}).copy()
            file_data['last_updated'] = datetime.now(timezone.utc).isoformat()
            file_data['type'] = 'agent_status'
            
            # Atomic write (write to temp, then rename)
            temp_path = status_file_path + '.tmp'
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(file_data, f, indent=2, default=str)
            os.replace(temp_path, status_file_path)
        except Exception as e:
            print(f"[{self.name}] Error synchronizing dashboard status: {e}")

    def start_live_loop(self, interval_seconds: int = 60):
        self._active_interval = interval_seconds
        
        # User requested to remove terminal table and reduce noise
        # with Live(console=console, screen=False, refresh_per_second=4) as live:
            
        while True:
            if self.gui_stop_event and self.gui_stop_event.is_set(): break
            
            # --- COMMAND QUEUE PROCESSING ---
            if hasattr(self, 'command_queue') and self.command_queue and not self.command_queue.empty():
                try:
                    cmd = self.command_queue.get_nowait()
                    if cmd['type'] == 'update_config':
                        print(f"[{self.name}] ⚙️ Applying Runtime Config: {cmd['data']}")
                        # Apply updates
                        data = cmd['data']
                        if 'max_allocation' in data:
                            config.GOVERNOR_MAX_MARGIN_PCT = float(data['max_allocation'])
                        if 'leverage_cap' in data:
                            config.PREDATOR_LEVERAGE = float(data['leverage_cap'])
                        if 'micro_mode' in data:
                            config.MICRO_CAPITAL_MODE = bool(data['micro_mode'])
                        
                        # Apply to Sub-Holdons if needed
                        gov = self.sub_holons.get('governor')
                        if gov:
                             gov.max_allocation = config.GOVERNOR_MAX_MARGIN_PCT
                    
                    elif cmd['type'] == 'panic_close':
                        print(f"[{self.name}] 🚨 PANIC SIGNAL RECEIVED via Queue")
                        ex = self.sub_holons.get('executor')
                        if ex: ex.panic_close_all(ex.latest_prices)
                        
                    elif cmd['type'] == 'c2_order':
                        print(f"[{self.name}] ⚔️ C2 ORDER: {cmd['direction']} {cmd['symbol']} (${cmd['qty_usd']})")
                        ex = self.sub_holons.get('executor')
                        if ex and hasattr(ex, 'actuator'):
                            # Calculate required qty from USD
                            price = ex.latest_prices.get(cmd['symbol'])
                            if price and price > 0:
                                qty = cmd['qty_usd'] / price
                                print(f"[{self.name}] Executing Manual {cmd['direction']} for {qty:.6f} {cmd['symbol']} @ ~${price}")
                                # Force market/taker logic for urgent C2 execution
                                ex.actuator.place_order(cmd['symbol'], cmd['direction'], qty, price, order_type='limit', urgent=True)
                            else:
                                print(f"[{self.name}] ❌ C2 FAILED: Cannot get price for {cmd['symbol']}")
                                
                    elif cmd['type'] == 'c2_close':
                        print(f"[{self.name}] ⚔️ C2 CLOSE: {cmd['symbol']}")
                        ex = self.sub_holons.get('executor')
                        if ex and hasattr(ex, 'actuator'):
                            ex.actuator.close_position(cmd['symbol'], reason="MANUAL_C2_CLOSE")
                            
                    elif cmd['type'] == 'c2_pause':
                        print(f"[{self.name}] ⏸️ SYSTEM PAUSED by C2 command.")
                        self.is_paused = True
                        
                    elif cmd['type'] == 'c2_resume':
                        print(f"[{self.name}] ▶️ SYSTEM RESUMED by C2 command.")
                        self.is_paused = False
                        
                except Exception as e:
                    print(f"[{self.name}] Command Error: {e}")
            # --------------------------------
            
            start = time.time()
            try: 
                # --- CHECK FOR GENOME UPDATES (WINNING BRAIN) ---
                self._scan_for_genome_updates()
                self._scan_for_graft_requests() # <--- NEW: Ecological Grafting
                # -----------------------------------------------
                # -----------------------------------------------

                # Reduced Log Noise: Commented out cycle start print
                # print(f"\n[{self.name}] --- Starting Warp Cycle (Interval: {interval_seconds}s) ---") 
                
                # --- PHASE X: TRAILING STOPS ---
                # Manage stops before running new signals
                ex = self.sub_holons.get('executor')
                if ex: self._manage_trailing_stops(ex)
                # -------------------------------

                # --- PHASE 2: MONTE CARLO MANAGER ---
                gov = self.sub_holons.get('governor')
                if gov and hasattr(gov, 'manage_positions'):
                    mc_closures = gov.manage_positions()
                    for c in mc_closures:
                         print(f"[{self.name}] 📉 EXECUTING MONTE CARLO CLOSURE: {c['symbol']}")
                         if ex and hasattr(ex, 'actuator'):
                             ex.actuator.close_position(c['symbol'], reason=c['reason'])
                # ------------------------------------

                report = {}
                if self.is_paused:
                    print(f"[{self.name}] ⏸️ System Paused. Skipping oracle evaluation cycle. Maintaining stops.")
                    report = {'assets': [], 'summary': 'SYSTEM PAUSED'}
                else:
                    report = self.run_cycle()
                
                # --- QUANT-OPS MULTI-AGENT INTELLIGENCE TICK ---
                quantops = self.sub_holons.get('quantops')
                if quantops and hasattr(quantops, 'tick') and not self.is_paused:
                    try:
                        cycle_result = quantops.tick()
                        if cycle_result:
                            print(f"[{self.name}] 🧠 QUANT-OPS Cycle #{cycle_result.get('cycle_id', '?')} "
                                  f"completed ({cycle_result.get('duration_sec', 0):.1f}s)")
                    except Exception as e:
                        print(f"[{self.name}] [QUANT-OPS] Tick error: {e}")
                # -----------------------------------------------

                # --- SENTRY & SOCIAL AUDIT ---
                overwatch = self.sub_holons.get('overwatch')
                if overwatch and hasattr(overwatch, 'perform_audit'):
                    overwatch.perform_audit()
                    
                moltbook = self.sub_holons.get('moltbook')
                if moltbook and hasattr(moltbook, 'perform_audit'):
                    moltbook.perform_audit()
                # -----------------------------
                
                # GC Monitor: Run every N cycles
                self.gc_cycle_counter += 1
                gc_interval = getattr(config, 'GC_INTERVAL_CYCLES', 5)
                if self.gc_cycle_counter >= gc_interval:
                    self.run_gc_cycle()
                    self.gc_cycle_counter = 0
                
                if self.gui_queue: 
                    self.gui_queue.put({'type': 'summary', 'data': report})
                    self.publish_agent_status() # NEW: Push full agent metrics
                
                # ALWAYS write summary to file for external dashboard sync
                try:
                    import json
                    from datetime import datetime, timezone
                    status_file_path = os.path.join(os.path.dirname(__file__), '..', 'dashboard_status.json')
                    
                    # Read existing file to merge with summary
                    existing_data = {}
                    if os.path.exists(status_file_path):
                        with open(status_file_path, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                    
                    # Update with summary data
                    existing_data['summary'] = report
                    
                    # Inject Loan Info into summary for UI (2026-03-09)
                    gov = self.sub_holons.get('governor')
                    if gov:
                        existing_data['summary']['loan_amount'] = getattr(gov, 'loan_amount', 0.0)
                        existing_data['summary']['repayment_reserve'] = getattr(gov, 'repayment_reserve', 0.0)
                        existing_data['summary']['net_worth'] = getattr(gov, 'balance', 0.0) - getattr(gov, 'loan_amount', 0.0)

                    existing_data['summary_updated'] = datetime.now(timezone.utc).isoformat()
                    existing_data['last_updated'] = existing_data['summary_updated']
                    
                    # Atomic write
                    temp_path = status_file_path + '.tmp'
                    with open(temp_path, 'w', encoding='utf-8') as f:
                        json.dump(existing_data, f, indent=2, default=str)
                    os.replace(temp_path, status_file_path)
                except Exception:
                    pass  # Silently fail file write
                
                # --- CAPITAL CRISIS CHECK ---
                # Check if we have a severe capital discrepancy that requires emergency action
                executor = self.sub_holons.get('executor')
                governor = self.sub_holons.get('governor')
                if executor and governor:
                    current_equity = executor.get_portfolio_value(0.0)
                    db_balance = getattr(governor, 'balance', current_equity)

                    # If there's a significant discrepancy (>40%), trigger emergency protocols
                    if db_balance > 0:
                        discrepancy_pct = abs(db_balance - current_equity) / db_balance
                        if discrepancy_pct > 0.4:  # 40% discrepancy
                            print(f"[{self.name}] 🚨 CAPITAL CRISIS: {discrepancy_pct:.1%} discrepancy between DB (${db_balance:.2f}) and Exchange (${current_equity:.2f})")
                            print(f"[{self.name}] 🚨 INITIATING EMERGENCY PROTOCOLS...")

                            # Send emergency close signal to governor
                            if hasattr(governor, 'receive_message'):
                                governor.receive_message(self, {'type': 'EMERGENCY_CLOSE_ALL'})

                            # Also trigger executor's panic close
                            if hasattr(executor, 'panic_close_all'):
                                executor.panic_close_all(executor.latest_prices)

                            # Clear phantom margin after emergency close
                            if hasattr(governor, 'receive_message'):
                                governor.receive_message(self, {'type': 'CLEAR_PHANTOM_MARGIN'})
                # ---------------------------------

                # --- MANAGEMENT MODE STATUS CHECK ---
                # FIX 2026-03-05: Actually evaluate whether management mode should exit.
                # check_and_update_management_mode() was never called in the cycle,
                # making management mode permanent once activated. Now called every cycle.
                governor = self.sub_holons.get('governor')
                if governor and hasattr(governor, 'check_and_update_management_mode'):
                    governor.check_and_update_management_mode()

                # Report status
                governor = self.sub_holons.get('governor')
                if governor and hasattr(governor, 'get_management_mode_status'):
                    mm_status = governor.get_management_mode_status()
                    if mm_status.get('active', False):
                        print(f"[{self.name}] 🛠️ MANAGEMENT MODE ACTIVE: {mm_status.get('reason', 'Unknown')}")
                        print(f"    Duration: {mm_status.get('duration_seconds', 0):.0f}s, Risk Multiplier: {mm_status.get('risk_multiplier', 1.0):.2f}")

                        # Update GUI queue if available
                        if self.gui_queue:
                            self.gui_queue.put({
                                'type': 'management_mode_status',
                                'data': mm_status
                            })
                # ---------------------------------

                # --- PHANTOM MARGIN CHECK ---
                # Verify that position counts match between executor and governor
                executor = self.sub_holons.get('executor')
                governor = self.sub_holons.get('governor')
                if governor and executor:
                    # Get actual positions from executor vs positions in governor
                    actual_positions = len(executor.held_assets) if hasattr(executor, 'held_assets') else 0
                    governor_positions = len(governor.positions) if hasattr(governor, 'positions') else 0

                    # If executor shows 0 positions but governor still has positions, clear phantom
                    if actual_positions == 0 and governor_positions > 0:
                        print(f"[{self.name}] 🚨 PHANTOM MARGIN DETECTED: Executor shows 0 positions but Governor has {governor_positions}. Clearing phantom.")
                        if hasattr(governor, 'receive_message'):
                            governor.receive_message(self, {'type': 'CLEAR_PHANTOM_MARGIN'})

                    # Also check if positions are out of sync and force resync
                    elif actual_positions != governor_positions:
                        print(f"[{self.name}] ⚠️ POSITION COUNT MISMATCH: Executor has {actual_positions} but Governor has {governor_positions}. Forcing resync.")
                        if hasattr(governor, 'sync_positions') and hasattr(executor, 'held_assets') and hasattr(executor, 'position_metadata'):
                            governor.sync_positions(executor.held_assets, executor.position_metadata)
                # ---------------------------------

                # --- PHASE 22: PPO REWARD HOSE ---
                # Feed the Monolith with the results of this cycle (Action=Previous Conviction, Reward=Equity Delta)
                self._feed_the_monolith(report)
                # ---------------------------------
                
                # Disable Terminal Table Update
                # layout = self._create_summary_layout(report)
                # live.update(layout)
                
            except DeadMansSwitchTriggered as dms_msg:
                print(f"\n=======================================================")
                print(f"[{self.name}] ☠️ FATAL: DEAD MAN'S SWITCH TRIGGERED ☠️")
                print(f"[{self.name}] Reason: {dms_msg}")
                print(f"[{self.name}] HALTING ALL TRADING OPERATIONS IMMEDIATELY.")
                print(f"=======================================================\n")
                # Try to notify via Overwatch/Telegram if possible before dying
                overwatch = self.sub_holons.get('overwatch')
                if overwatch and hasattr(overwatch, 'send_alert'):
                    try: overwatch.send_alert(f"☠️ DEAD MAN'S SWITCH SPUN: {dms_msg}. BOT HALTED.")
                    except: pass
                
                # Attempt to set GUI stop event so dashboard knows
                if self.gui_stop_event:
                    self.gui_stop_event.set()
                
                # Break the loop entirely
                break
                
            except Exception as e:
                # import traceback (Moved to top)
                print(f"[{self.name}] ☠️ Cycle Error: {e}")
                traceback.print_exc()
                _time.sleep(30)

            wait = max(0, interval_seconds - (time.time() - start))
            for _ in range(int(wait * 2)):
                if self.gui_stop_event and self.gui_stop_event.is_set(): break
                time.sleep(0.5)

    def run_gc_cycle(self):
        """
        Garbage Collector Monitor: Run cleanup across all components.
        Called periodically every GC_INTERVAL_CYCLES.
        """
        gc_interval = getattr(config, 'GC_INTERVAL_CYCLES', 5)
        verbose = getattr(config, 'GC_LOG_VERBOSE', True)
        
        if verbose:
            print(f"\n[GC Monitor] 🧹 Starting Garbage Collection Cycle...")
        
        actuator = self.sub_holons.get('actuator')
        executor = self.sub_holons.get('executor')
        governor = self.sub_holons.get('governor')
        
        # 1. Actuator: Clean stale pending orders
        stale_orders = 0
        if actuator and hasattr(actuator, 'gc_clean_stale_orders'):
            stale_orders = actuator.gc_clean_stale_orders()
        
        # 2. Executor: Reconcile positions with exchange
        ghosts = []
        if executor and hasattr(executor, 'gc_reconcile_positions'):
            ghosts = executor.gc_reconcile_positions()
        
        # 3. Governor: Sync with Executor
        mismatches = []
        if governor and hasattr(governor, 'gc_sync_with_executor'):
            mismatches = governor.gc_sync_with_executor(executor)
        
        if verbose:
            print(f"[GC Monitor] ✅ GC Complete: {stale_orders} stale orders, {len(ghosts)} ghosts, {len(mismatches)} gov mismatches.")

    def receive_message(self, sender, content): pass
    def _adapt_to_regime(self, regime): pass

    def _feed_the_monolith(self, report):
        """
        Calculate Step Reward and feed the PPO Brain.
        Reward = (Realized PnL + Unrealized Delta) / Volatility
        """
        governor = self.sub_holons.get('governor')
        if not governor or not hasattr(governor, 'ppo'): return
        
        # 1. Calculate Reward
        # We need equity change since last cycle.
        current_equity = governor.balance
        if not hasattr(self, 'last_ppo_equity'): self.last_ppo_equity = current_equity
        
        equity_delta = current_equity - self.last_ppo_equity
        self.last_ppo_equity = current_equity
        
        # Normalize reward (e.g. $0.10 gain on $10 account = 1%)
        # Scale up because PPO likes ~1.0 range
        raw_reward = (equity_delta / config.INITIAL_CAPITAL) * 100.0
        
        # simple clipping to avoid exploding gradients from wild swings
        reward = max(-5.0, min(5.0, raw_reward))
        
        # 2. Get State (Simplified for now)
        # [WinRate, Drawdown, MarginUsed, ... ]
        state = [
            governor.db_manager.get_win_rate() if governor.db_manager else 0.5,
            governor.drawdown_pct,
            governor.margin_utilization,
            0.0, 0.0, 0.0 # placeholders
        ]
        
        # 3. Remember (We assume Action was 0.5 for now, finding exact action requires tracking)
        # ideally we track what action PPO suggested at start of cycle.
        # For now, we just train it to associate State -> Reward.
        # This is a loose approximation to prime the memory.
        import numpy as np
        governor.ppo.remember(
            state=np.array(state, dtype=np.float32), 
            action=0.5, # Placeholder action
            reward=reward, 
            prob=0.5, 
            val=0.5, 
            done=False
        )
        
        # 4. Learn periodically
        if (self.cycle_counter % 10) == 0:
            a_loss, c_loss = governor.ppo.learn()
            if abs(a_loss) > 0:
                print(f"[{self.name}] 🧠 PPO LEARN: Rewards={reward:.4f} | Loss A={a_loss:.4f} C={c_loss:.4f}")

    def _scan_for_genome_updates(self):
        """
        Phase 46 + Ensemble: Checks for new Evolution Result & Hall of Fame.
        Hot-swaps parameters and Ensemble Strategies.
        """
        import os
        import json

        # In LIVE trading, genome hot-swaps apply evolved parameters to config.
        # Safety is enforced via SANITY CLAMP below (clamped to min/max bounds).
        evolution_enabled = bool(getattr(config, 'ENABLE_EVOLUTION', True))
        
        # 1. LIVE GENOME (Single Best) - Parameter Tuning
        path_genome = os.path.join(os.getcwd(), 'live_genome.json')
        if os.path.exists(path_genome):
            try:
                mtime = os.path.getmtime(path_genome)
                if not hasattr(self, 'last_genome_mtime'): self.last_genome_mtime = 0

                if mtime > self.last_genome_mtime:
                    self.last_genome_mtime = mtime
                    with open(path_genome, 'r') as f:
                        data = json.load(f)
                    
                    genome = data.get('genome', {})
                    if genome:
                        if not evolution_enabled:
                            if not hasattr(self, '_evolution_disabled_notice_printed'):
                                self._evolution_disabled_notice_printed = True
                                print(f"[{self.name}] 🧬 Genome update detected but ENABLE_EVOLUTION=False — skipping brain transplant.")
                            genome = {}
                        if genome:
                            print(f"[{self.name}] 🧬 DETECTED NEW EVOLVED BRAIN (Fitness: {data.get('fitness', 0):.2f})")
                            
                            # =========================================================
                            # 🛡️ SANITY CLAMP (Protects against "Lottery Genomes")
                            # =========================================================
                            # STOP LOSS: respect config floors/ceilings (avoid silently forcing 5%)
                            sl = float(genome.get('stop_loss', getattr(config, 'DEFAULT_STOP_LOSS_PCT', 0.025)))
                            sl_min = float(getattr(config, 'MIN_STOP_LOSS_PCT', getattr(config, 'DEFAULT_STOP_LOSS_PCT', 0.015)))
                            sl_max = float(getattr(config, 'MAX_STOP_LOSS_PCT', 0.08))
                            if sl < sl_min:
                                sl = max(sl_min, float(getattr(config, 'DEFAULT_STOP_LOSS_PCT', sl_min)))
                                print(f"[{self.name}] 🛡️ SANITY CLAMP: SL too tight, set to {sl:.1%}")
                            elif sl > sl_max:
                                sl = sl_max
                                print(f"[{self.name}] 🛡️ SANITY CLAMP: SL too loose, set to {sl:.1%}")
                            
                            # TAKE PROFIT: respect config floors/ceilings
                            tp = float(genome.get('take_profit', getattr(config, 'DEFAULT_TAKE_PROFIT_PCT', 0.045)))
                            tp_min = float(getattr(config, 'MIN_TAKE_PROFIT_PCT', getattr(config, 'DEFAULT_TAKE_PROFIT_PCT', 0.01)))
                            tp_max = float(getattr(config, 'MAX_TAKE_PROFIT_PCT', 0.25))
                            if tp < tp_min:
                                tp = max(tp_min, float(getattr(config, 'DEFAULT_TAKE_PROFIT_PCT', tp_min)))
                                print(f"[{self.name}] 🛡️ SANITY CLAMP: TP too tight, set to {tp:.1%}")
                            elif tp > tp_max:
                                tp = tp_max
                                print(f"[{self.name}] 🛡️ SANITY CLAMP: TP too loose, set to {tp:.1%}")
                            # =========================================================
                            
                            # Update Config (Global Defaults) with clamped values
                            config.STRATEGY_RSI_OVERSOLD = float(genome.get('rsi_buy', config.STRATEGY_RSI_OVERSOLD))
                            config.STRATEGY_RSI_OVERBOUGHT = float(genome.get('rsi_sell', config.STRATEGY_RSI_OVERBOUGHT))
                            # Sync RSI_ENTRY_MAX with evolved overbought threshold
                            config.STRATEGY_RSI_ENTRY_MAX = config.STRATEGY_RSI_OVERBOUGHT
                            config.SATELLITE_STOP_LOSS = sl
                            config.SATELLITE_TAKE_PROFIT_1 = tp
                            # Also apply SL/TP to global defaults (not just satellite)
                            config.DEFAULT_STOP_LOSS_PCT = sl
                            config.DEFAULT_TAKE_PROFIT_PCT = tp
                            
                            print(f"[{self.name}] ✅ Brain Transplant: RSI_OS={config.STRATEGY_RSI_OVERSOLD:.0f}, RSI_OB={config.STRATEGY_RSI_OVERBOUGHT:.0f}, SL={sl:.1%}, TP={tp:.1%}")
            except Exception as e:
                print(f"[{self.name}] ⚠️ Genome Read Error: {e}")

        # 2. HALL OF FAME (Ensemble) - Logic Update
        path_hof = os.path.join(os.getcwd(), 'hall_of_fame.json')
        if os.path.exists(path_hof):
            try:
                mtime = os.path.getmtime(path_hof)
                if not hasattr(self, 'last_hof_mtime'): self.last_hof_mtime = 0
                
                if mtime > self.last_hof_mtime:
                    self.last_hof_mtime = mtime
                    
                    oracle = self.sub_holons.get('oracle')
                    if oracle and hasattr(oracle, 'load_ensemble'):
                        oracle.load_ensemble(path_hof)
                        print(f"[{self.name}] 🎭 ENSEMBLE DEPLOYED: Hall of Fame Loaded into Oracle.")
            except Exception as e:
                print(f"[{self.name}] ⚠️ HOF Read Error: {e}")

    def publish_agent_status(self):
        """
        Gathers health and status metrics from all sub-holons
        and pushes them to the GUI Queue for the Dashboard.
        """
        if not self.gui_queue: return
        
        statuses = {}
        
        # 1. Governor (Risk)
        gov = self.sub_holons.get('governor')
        if gov:
            statuses['governor'] = {
                'allocation': getattr(gov, 'allocation_percentage', 0.0),
                'leverage': getattr(gov, 'leverage_cap', 1.0),
                'risk_mult': getattr(gov, 'risk_multiplier', 1.0),
                'state': 'MANAGEMENT' if getattr(gov, 'management_mode', False) else 'ACTIVE',
                'drawdown': getattr(gov, 'drawdown_pct', 0.0),
                'loan_amount': getattr(gov, 'loan_amount', 0.0),
                'repayment_reserve': getattr(gov, 'repayment_reserve', 0.0),
                'aggressive_mode': getattr(gov, 'aggressive_mode', False)
            }
            
        # 2. Actuator (Execution)
        act = self.sub_holons.get('actuator')
        if act:
            statuses['actuator'] = {
                'circuit': 'OPEN' if getattr(act, 'circuit_open', False) else 'CLOSED',
                'errors': getattr(act, 'error_count', 0),
                'pending': len(getattr(act, 'pending_orders', [])),
                'held_locks': len(getattr(act, 'failed_orders', {}))
            }
            
        # 3. Brains (Oracle/Entropy/PPO)
        oracle = self.sub_holons.get('oracle')
        if oracle:
             statuses['oracle'] = {
                 'bias': getattr(oracle, 'market_bias', 0.5),
                 'regime': self.market_state.get('regime', 'UNKNOWN')
             }
             
        # 4. Performance (Executor)
        executor = self.sub_holons.get('executor')
        if executor:
             # Try to get real equity
             eq = 0.0
             if hasattr(executor, 'get_portfolio_value'):
                 eq = executor.get_portfolio_value(0.0)
             elif hasattr(executor, 'balance_usd'):
                 eq = executor.balance_usd
                 
             statuses['performance'] = {
                 'equity': eq,
                 'positions': len(getattr(executor, 'held_assets', {})),
                 'pnl_24h': 0.0 # Placeholder
             }
             
        self.gui_queue.put({'type': 'agent_status', 'data': statuses})

    def _scan_for_graft_requests(self):
        """
        Phase 47: Ecological Grafting.
        Allows hot-swapping strategies by dragging a 'graft_genome.json' into the folder.
        """
        import os
        import json
        
        path = os.path.join(os.getcwd(), 'graft_genome.json')
        if os.path.exists(path):
            try:
                # 1. Read the DNA
                with open(path, 'r') as f:
                    genome = json.load(f)
                
                print(f"[{self.name}] 🌿 GRAFT REQUEST DETECTED: Assimilating Alien DNA...")
                
                # 2. Inject into Runtime Config (Hot Swap)
                # Supports core params: stop_loss, take_profit, rsi_buy, rsi_sell
                changes = []
                
                if 'stop_loss' in genome:
                    val = float(genome['stop_loss'])
                    config.SATELLITE_STOP_LOSS = val # Update default
                    # Also try to update Gov? Gov reads from config usually.
                    changes.append(f"SL={val:.1%}")
                    
                if 'take_profit' in genome:
                    val = float(genome['take_profit'])
                    config.SATELLITE_TAKE_PROFIT_1 = val
                    changes.append(f"TP={val:.1%}")
                    
                if 'rsi_buy' in genome:
                    val = float(genome['rsi_buy'])
                    config.STRATEGY_RSI_OVERSOLD = val
                    changes.append(f"RSI_BUY={val}")
                    
                if 'rsi_sell' in genome:
                    val = float(genome['rsi_sell'])
                    config.STRATEGY_RSI_OVERBOUGHT = val
                    changes.append(f"RSI_SELL={val}")
                    
                # 3. Rename to prevent loops
                new_path = path + ".grafted"
                if os.path.exists(new_path): os.remove(new_path)
                os.rename(path, new_path)
                
                print(f"[{self.name}] ✅ GRAFT SUCCESSFUL: {', '.join(changes)} active.")
                
            except Exception as e:
                print(f"[{self.name}] ⚠️ GRAFT REJECTION: {e}")

    def _manage_trailing_stops(self, executor):
        """
        PHASE X: Active Trailing Stop Manager.
        Scans open positions. If PnL > Activation Threshold (e.g. 1.5R),
        Moves Stop Loss towards price to lock in gains.
        """
        if not executor or not executor.actuator: return
        
        # 1. Get Activation Parameters
        # TODO: Pull from Live Genome?
        # Default: Activate at 1.5% profit, Trail by 1.0% distance
        activation_pct = 0.015 
        trail_dist_pct = 0.010
        
        for held_virt_key, qty in executor.held_assets.items():
            if abs(qty) < 0.00000001: continue
            
            # Extract Real Symbol for price & actuator
            symbol = held_virt_key.split(':')[0] if ':' in held_virt_key else held_virt_key
            
            # Get Current Price
            curr_price = executor.latest_prices.get(symbol, 0.0)
            if curr_price <= 0: continue
            
            # Get Entry Data from metadata truth
            _raw_meta_ts = executor.position_metadata.get(held_virt_key)
            def _pa_ts(obj, key, default=None):
                if obj is None: return default
                if isinstance(obj, dict): return obj.get(key, default)
                return getattr(obj, key, default)
            entry_price = _pa_ts(_raw_meta_ts, 'entry_price', curr_price)
            if entry_price <= 0: continue
            
            direction = _pa_ts(_raw_meta_ts, 'direction', 'BUY')
            
            # Calculate PnL %
            if direction == 'BUY':
                pnl_pct = (curr_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - curr_price) / entry_price
                
            # Check Activation
            if pnl_pct > activation_pct:
                # We are in profit zone!
                
                # Check Expected Stop Price
                # Long: Price * (1 - trail)
                # Short: Price * (1 + trail)
                
                if direction == 'BUY':
                    new_stop_price = curr_price * (1.0 - trail_dist_pct)
                else:
                    new_stop_price = curr_price * (1.0 + trail_dist_pct)
                    
                # Find Existing Stop Order
                # We look for pending stop orders for this symbol
                existing_stop_id = None
                existing_stop_price = 0.0
                
                for order in executor.actuator.pending_orders:
                    if order.get('symbol') == symbol and order.get('type') == 'stop-market':
                         existing_stop_id = order.get('id')
                         existing_stop_price = float(order.get('stop_price', 0.0))
                         break
                         
                # Decide Update
                should_update = False
                if existing_stop_id:
                    if direction == 'BUY':
                        # Move UP only
                        if new_stop_price > (existing_stop_price * 1.001): # 0.1% buffer
                            should_update = True
                    else:
                        # Move DOWN only
                        if new_stop_price < (existing_stop_price * 0.999):
                            should_update = True
                else:
                    # No stop? Create one!
                    should_update = True
                    
                if should_update:
                    print(f"[{self.name}] 🥅 TRAILING STOP UPDATE: {symbol} PnL {pnl_pct*100:.1f}%. Moving Stop -> {new_stop_price:.2f}")
                    
                    # Cancel Old
                    if existing_stop_id:
                        executor.actuator.cancel_order(existing_stop_id, symbol)
                        
                    # Place New
                    # Stop Direction is Opposite to Position
                    stop_dir = 'SELL' if direction == 'BUY' else 'BUY'
                    executor.actuator.place_stop_order(symbol, stop_dir, abs(qty), new_stop_price)

    def shutdown(self):
        """
        Gracefully shut down the Trader, closing threads.
        """
        print(f"[{self.name}] 🛑 Shutting down Trader...")
        if self.executor:
            self.executor.shutdown(wait=False)
            print(f"[{self.name}] 🛑 Shared Executor Shutdown.")
