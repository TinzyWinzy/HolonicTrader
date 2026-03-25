"""
SignalProviderHolon - Manual Entry Intelligence (Phase 48)

Aggregates signals from all core holons to provide high-quality
manual entry signals with TP/SL.
"""

import pandas as pd
import numpy as np
import re
import time
import logging
from collections import defaultdict
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from .holon_core import Holon, Disposition
from .agent_executor import TradeSignal
import config

# Setup Logging
logger = logging.getLogger("SignalProvider")
logger.setLevel(logging.INFO)

class SignalProviderHolon(Holon):
    """
    SignalProviderHolon - Manual Entry Intelligence (Phase 48)

    A specialized holon responsible for aggregating signals from all core holons
    to provide high-quality manual entry signals with take-profit (TP) and
    stop-loss (SL) levels. This holon acts as the central intelligence hub
    for generating trade recommendations based on multiple data sources and
    analytical approaches.

    Attributes:
        last_signals (dict): Cache of previously generated signals to detect changes
        request_counts (defaultdict): Tracks API request counts by type for rate limiting
        rate_limit_window (int): Time window in seconds for rate limiting
        max_requests_per_window (int): Maximum requests allowed per window
    """

    def __init__(self, name: str = "SignalProvider"):
        """
        Initialize the SignalProviderHolon with default parameters.

        Args:
            name (str): Name identifier for this holon instance
        """
        super().__init__(name=name, disposition=Disposition(autonomy=0.9, integration=0.6))
        self.last_signals = {}

        # Rate limiting attributes
        self.request_counts = defaultdict(list)
        self.rate_limit_window = 60  # Time window in seconds
        self.max_requests_per_window = 100  # Maximum requests allowed per window

        # NEW: Signal Debouncing (Phase 2026-02-23)
        self.signal_cooldowns = {}  # {fingerprint: timestamp}
        self.signal_cooldown_seconds = 300  # 5 minutes cooldown for identical signals
        self.last_signal_per_asset = {}  # {symbol: {'direction': str, 'reason': str, 'timestamp': float}}
        
        # Order Flow Data Storage
        self.order_flow_data = {}
        self.volume_profile_data = {}

    def is_rate_limited(self, request_type: str) -> bool:
        """
        Check if a request type is currently rate limited.

        Args:
            request_type (str): Type of request to check (e.g., 'market_data', 'order_book')

        Returns:
            bool: True if rate limited, False otherwise
        """
        now = datetime.now()
        # Clean old requests outside the window
        self.request_counts[request_type] = [
            timestamp for timestamp in self.request_counts[request_type]
            if (now - timestamp).seconds < self.rate_limit_window
        ]

        # Check if we're over the limit
        if len(self.request_counts[request_type]) >= self.max_requests_per_window:
            return True

        # Add current request
        self.request_counts[request_type].append(now)
        return False

    def generate_signal_fingerprint(self, symbol: str, direction: str, reason: str) -> str:
        """
        Generate a unique fingerprint for a signal to detect duplicates.
        
        Args:
            symbol: Trading pair symbol
            direction: BUY or SELL
            reason: Signal reason (e.g., 'WHALE_BID_WALL', 'FUNDING_ARB')
        
        Returns:
            str: Unique fingerprint string
        """
        # Normalize reason (remove dynamic parts like prices)
        clean_reason = re.sub(r'\d+\.\d+', 'X', reason)  # Replace numbers with X
        clean_reason = re.sub(r'\s+', '_', clean_reason.upper())  # Normalize whitespace
        
        return f"{symbol}_{direction}_{clean_reason}"
    
    def is_signal_on_cooldown(self, symbol: str, direction: str, reason: str) -> bool:
        """
        Check if an identical signal is still on cooldown (debouncing).
        
        Args:
            symbol: Trading pair symbol
            direction: BUY or SELL
            reason: Signal reason
        
        Returns:
            bool: True if signal is duplicate (on cooldown), False if fresh
        """
        fingerprint = self.generate_signal_fingerprint(symbol, direction, reason)
        current_time = time.time()
        
        # Check if this exact signal was generated recently
        if fingerprint in self.signal_cooldowns:
            time_since_last = current_time - self.signal_cooldowns[fingerprint]
            if time_since_last < self.signal_cooldown_seconds:
                remaining = int(self.signal_cooldown_seconds - time_since_last)
                logger.debug(f"[{self.name}] 🕒 Signal on cooldown: {symbol} {direction} ({remaining}s remaining)")
                return True
        
        # Clean old cooldowns
        expired = [fp for fp, ts in self.signal_cooldowns.items() 
                   if current_time - ts > self.signal_cooldown_seconds]
        for fp in expired:
            del self.signal_cooldowns[fp]
        
        return False
    
    def record_signal(self, symbol: str, direction: str, reason: str):
        """
        Record a signal for debouncing (call after signal is generated).
        
        Args:
            symbol: Trading pair symbol
            direction: BUY or SELL
            reason: Signal reason
        """
        fingerprint = self.generate_signal_fingerprint(symbol, direction, reason)
        self.signal_cooldowns[fingerprint] = time.time()
        
        self.last_signal_per_asset[symbol] = {
            'direction': direction,
            'reason': reason,
            'timestamp': time.time()
        }
        
        logger.debug(f"[{self.name}] 📝 Signal recorded: {fingerprint}")

    def check_hps_confluence(self, symbol: str, signal: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates if a signal meets the criteria for a High Probability Setup (HPS).
        
        The HPS Protocol requires confluence of at least 3 of 5 pillars:
        1. Structure: Key Pivot or Zone support.
        2. Momentum: RSI/OBV confirmation.
        3. Probability: Monte Carlo > 60%.
        4. Physics: Low Entropy or Whale Volume.
        5. Bias: Global Market Alignment.
        
        Returns:
            dict: {'is_hps': bool, 'score': int, 'pillars': List[str]}
        """
        score = 0
        pillars = []
        
        direction = signal.direction # 'BUY' or 'SELL'
        data = context.get('data')
        
        # 1. STRUCTURE PILLAR
        # Is price at S2/S3 (Long) or R2/R3 (Short)?
        structure = context.get('structure', {})
        pivots = structure.get('pivots', {})
        current_price = signal.price
        
        is_structure_good = False
        if direction == 'BUY':
            s2 = pivots.get('S2')
            if s2 and current_price <= s2 * 1.01: # Within 1% of S2 or lower
                is_structure_good = True
        elif direction == 'SELL':
            r2 = pivots.get('R2')
            if r2 and current_price >= r2 * 0.99: # Within 1% of R2 or higher
                is_structure_good = True
                
        if is_structure_good:
            score += 1
            pillars.append('STRUCTURE')

        # 2. MOMENTUM PILLAR
        # RSI not overextended (unless reversal) AND OBV confirming
        rsi = context.get('rsi', 50)
        is_momentum_good = False
        if direction == 'BUY':
            if rsi < 70: is_momentum_good = True
        elif direction == 'SELL':
            if rsi > 30: is_momentum_good = True
            
        if is_momentum_good:
            score += 1
            pillars.append('MOMENTUM')

        # 3. PROBABILITY PILLAR (Monte Carlo)
        # > 60% Hit Rate
        hit_prob = context.get('hit_prob', 0.0)
        if hit_prob > 0.60:
            score += 1
            pillars.append('PROBABILITY')

        # 4. PHYSICS PILLAR
        # Entropy Low (Trend) OR Whale Signal
        entropy_val = context.get('entropy', 1.0)
        is_whale = signal.metadata.get('is_whale', False)
        
        if entropy_val < 0.6 or is_whale:
            score += 1
            pillars.append('PHYSICS')
            
        # 5. BIAS PILLAR
        # Alignment with Global Trend (mocked here, ideally from oracle.market_bias)
        # Assuming if filtered by Oracle logic, it's somewhat aligned, but let's be strict.
        # User Oracle.get_market_bias() if available
        # For now, we trust the Oracle's 'fair weather' check passed.
        # Let's verify 'regime'
        regime = context.get('regime', 'UNKNOWN')
        is_bias_good = False
        if direction == 'BUY' and regime in ['BULLISH_TREND', 'STABLE']:
            is_bias_good = True
        elif direction == 'SELL' and regime in ['BEARISH_TREND']:
            is_bias_good = True
            
        if is_bias_good:
            score += 1
            pillars.append('BIAS')

        return {
            'is_hps': score >= 3,
            'score': score,
            'pillars': pillars
        }

    def generate_signal_report(self, sub_holons: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Scan all allowed assets and generate a comprehensive report of high-quality trading signals.

        This method performs a multi-layered analysis combining technical indicators,
        market entropy, structural context, whale detection, and arbitrage opportunities
        to produce ranked trading signals with appropriate risk management parameters.

        The process includes:
        1. Fetching market data and order book information
        2. Calculating entropy and market regime
        3. Performing TDA (Topological Data Analysis)
        4. Checking for whale wall formations
        5. Generating entry signals using the oracle
        6. Applying asset personality characteristics
        7. Calculating appropriate TP/SL levels
        8. Validating signals against structural alignment
        9. Determining position sizing via the governor

        Args:
            sub_holons (Dict[str, Any]): Dictionary containing all required holon instances
                including observer, oracle, whale, arbitrage, governor, structure, entropy,
                and topology holons

        Returns:
            List[Dict[str, Any]]: A list of signal dictionaries, each containing:
                - symbol (str): Trading pair symbol
                - direction (str): BUY or SELL
                - price (float): Current market price
                - tp (float): Take profit price
                - sl (float): Stop loss price
                - conviction (float): Signal confidence level (0.0-1.0)
                - quality (str): Signal quality rating (HIGH, MEDIUM, VETOED)
                - reason (str): Explanation for the signal
                - regime (str): Market regime classification
                - execution_details (dict): Position sizing information
                - account_context (dict): Account health information
                - timestamp (str): ISO format timestamp of signal generation
        """
        observer = sub_holons.get('observer')
        oracle = sub_holons.get('oracle')
        whale = sub_holons.get('whale')
        arbitrage = sub_holons.get('arbitrage')
        governor = sub_holons.get('governor')
        structure = sub_holons.get('structure')
        entropy_agent = sub_holons.get('entropy')
        topology = sub_holons.get('topology')
        kraken = sub_holons.get('kraken')  # Added for Position Management
        dump_pump_detector = sub_holons.get('dump_pump_detector')  # NEW: Dump/Pump Whale Detector

        if not all([observer, oracle, governor]):
            logger.error(f"[{self.name}] ❌ Missing core holons for signal generation.")
            return []

        # 获取账户信息
        try:
            health = governor.get_portfolio_health()
            equity = health.get('equity', 0.0)
            available = health.get('available_margin', 0.0)
            drawdown = health.get('drawdown_pct', 0.0)
        except Exception as e:
            logger.error(f"[{self.name}] Failed to get portfolio health: {e}")
            # Use default values if health check fails
            equity = 0.0
            available = 0.0
            drawdown = 0.0

        report = []

        # --- FETCH VIX LEVEL (MacroOracle) ---
        vix_level = 0.0
        try:
            if oracle and hasattr(oracle, 'macro_oracle') and hasattr(oracle.macro_oracle, 'get_vix_level'):
                vix_level = oracle.macro_oracle.get_vix_level()
                if vix_level > 0:
                    vix_regime = (
                        'PANIC' if vix_level >= getattr(config, 'VIX_PANIC_THRESHOLD', 30.0) else
                        'FEAR'  if vix_level >= getattr(config, 'VIX_FEAR_THRESHOLD', 20.0) else
                        'CALM'  if vix_level < getattr(config, 'VIX_CALM_THRESHOLD', 15.0) else 'NORMAL'
                    )
                    logger.info(f"[{self.name}] 🌡️ VIX: {vix_level:.1f} [{vix_regime}]")
        except Exception as e:
            logger.debug(f"[{self.name}] VIX fetch skipped: {e}")

        # --- AI POSITION MANAGEMENT LOOP ---
        if kraken and oracle and hasattr(kraken, 'detect_ghost_positions'):
            try:
                # 1. Fetch Reality (All active positions on exchange)
                # Pass empty ledger to get ALL positions
                platform_scan = kraken.detect_ghost_positions({})
                active_positions = platform_scan.get('ghosts', {})
                
                # Fetch Health & Account Data
                platform_info = kraken.get_platform_info()
                account_health = platform_info.get('account_health', {})
                
                # 2. Iterate & Analyze
                for symbol, qty in active_positions.items():
                    # We need market data context for the Oracle
                    # Try to fetch from observer if possible
                    df = None
                    if observer:
                        try:
                            # Try standard fetch 
                            df = observer.fetch_market_data(symbol, limit=100)
                        except:
                            pass
                            
                    if df is not None and not df.empty:
                         # Get Structure Context
                        structure_ctx = {}
                        if structure:
                            try:
                                structure_ctx = structure.get_structural_context(symbol, observer)
                            except:
                                pass
                                
                        # 3. ASK THE ORACLE
                        position_data = {'qty': qty, 'symbol': symbol}
                        # Call the new method we added to Oracle
                        if hasattr(oracle, 'analyze_active_position'):
                            decision = oracle.analyze_active_position(
                                symbol, 
                                position_data, 
                                account_health, 
                                df, 
                                structure_ctx
                            )
                            
                            if decision['type'] != 'HOLD':
                                # Create a Management Signal
                                report.append({
                                    'symbol': symbol,
                                    'direction': decision['type'], # CLOSE / REDUCE / STACK
                                    'conviction': decision.get('urgency', 0.5), # Urgency = Conviction
                                    'quality': 'HIGH' if decision.get('urgency', 0) > 0.8 else 'MEDIUM',
                                    'reason': f"[AI MANAGER] {decision['reason']}",
                                    'price': df['close'].iloc[-1],
                                    'timestamp': datetime.now(timezone.utc).isoformat(),
                                    'metadata': {
                                        'strategy': 'POSITION_MANAGEMENT', 
                                        'original_action': decision
                                    },
                                    # Fill required fields with SAFE defaults to pass validation
                                    'tp': 0.0,
                                    'sl': 0.0, 
                                    'regime': 'MANAGEMENT',
                                    'tda_status': 'N/A',
                                    'expected_yield': 0.0,
                                    'hit_probability': 0.0,
                                    'decay_score': 1.0,
                                    'optimal_horizon': 0,
                                    'pips_potential': 0.0,
                                    'execution_details': {}, 
                                    'account_context': {'equity': equity}
                                })
            except Exception as e:
                logger.error(f"[{self.name}] Position Management Error: {e}")


        # Validate ALLOWED_ASSETS configuration
        try:
            assets = config.ALLOWED_ASSETS
            if not isinstance(assets, (list, tuple)):
                logger.error(f"[{self.name}] ALLOWED_ASSETS is not properly configured as a list/tuple")
                return []
        except AttributeError:
            logger.error(f"[{self.name}] ALLOWED_ASSETS configuration is missing")
            return []

        for symbol in assets:
            try:
                # Validate symbol format
                if not isinstance(symbol, str) or not symbol.strip():
                    logger.warning(f"[{self.name}] Invalid symbol: {symbol}")
                    continue

                # Additional symbol format validation
                if not re.match(r'^[A-Z0-9_]+/[A-Z0-9_]+$', symbol) and not re.match(r'^[A-Z0-9_]+$', symbol):
                    logger.warning(f"[{self.name}] Symbol format does not match expected pattern (e.g., BTC/USD or BTCUSD): {symbol}")
                    continue

                # Check rate limit for market data fetch
                if self.is_rate_limited('market_data'):
                    logger.warning(f"[{self.name}] Rate limit exceeded for market data fetch, skipping {symbol}")
                    continue

                # 1. Fetch Data
                try:
                    data = observer.fetch_market_data(symbol=symbol, limit=100)
                except ConnectionError as ce:
                    logger.error(f"[{self.name}] Network connection error while fetching market data for {symbol}: {ce}")
                    continue
                except TimeoutError as te:
                    logger.error(f"[{self.name}] Request timeout while fetching market data for {symbol}: {te}")
                    continue
                except Exception as e:
                    logger.error(f"[{self.name}] Failed to fetch market data for {symbol}: {e}")
                    continue

                if data is None or data.empty:
                    logger.warning(f"[{self.name}] No market data for {symbol}")
                    continue

                current_price = data['close'].iloc[-1]

                # Check rate limit for order book fetch
                if self.is_rate_limited('order_book'):
                    logger.warning(f"[{self.name}] Rate limit exceeded for order book fetch, skipping {symbol}")
                    book_data = None
                else:
                    try:
                        book_data = observer.fetch_order_book(symbol)
                    except ConnectionError as ce:
                        logger.error(f"[{self.name}] Network connection error while fetching order book for {symbol}: {ce}")
                        book_data = None
                    except TimeoutError as te:
                        logger.error(f"[{self.name}] Request timeout while fetching order book for {symbol}: {te}")
                        book_data = None
                    except Exception as e:
                        logger.error(f"[{self.name}] Failed to fetch order book for {symbol}: {e}")
                        book_data = None

                # Check rate limit for funding rate fetch
                if self.is_rate_limited('funding_rate'):
                    logger.warning(f"[{self.name}] Rate limit exceeded for funding rate fetch, skipping {symbol}")
                    funding_rate = 0.0
                else:
                    try:
                        funding_rate = observer.fetch_funding_rate(symbol)
                    except ConnectionError as ce:
                        logger.error(f"[{self.name}] Network connection error while fetching funding rate for {symbol}: {ce}")
                        funding_rate = 0.0
                    except TimeoutError as te:
                        logger.error(f"[{self.name}] Request timeout while fetching funding rate for {symbol}: {te}")
                        funding_rate = 0.0
                    except Exception as e:
                        logger.error(f"[{self.name}] Failed to fetch funding rate for {symbol}: {e}")
                        funding_rate = 0.0

                funding_apy = abs(funding_rate) * 3 * 365 * 100

                # 2. Indicators & Physics
                try:
                    returns = data['close'].pct_change().dropna()
                    
                    # --- DASHBOARD ANALYTICS ---
                    # 1. Order Flow
                    self.measure_order_flow(symbol, oracle, observer, data)
                    
                    # 2. Monte Carlo Projections
                    if len(data) > 100:
                        volatility = returns.std()
                        # Simple params for visualization
                        mc_params = {
                            'mu': 0.0,
                            'sigma': float(volatility * (365 * 24)**0.5), # Annualized
                            'lambda': 0.1
                        }
                        self._write_monte_carlo_results(symbol, current_price, mc_params)
                    # ---------------------------

                    entropy_val = entropy_agent.calculate_shannon_entropy(returns) if entropy_agent else 0.0
                    regime = entropy_agent.determine_regime(entropy_val) if entropy_agent else 'UNKNOWN'
                except Exception as e:
                    logger.error(f"[{self.name}] Failed to calculate entropy for {symbol}: {e}")
                    entropy_val = 0.0
                    regime = 'UNKNOWN'

                # TDA Analysis
                tda_score = 0.5
                tda_status = 'STABLE'
                if topology:
                    try:
                        tda_res = topology.analyze_structure(data)
                        tda_score = tda_res.get('score', 0.5)
                        tda_status = tda_res.get('status', 'STABLE')
                    except Exception as e:
                        logger.error(f"[{self.name}] Failed to analyze structure for {symbol}: {e}")

                # 3. Structural Context
                structure_ctx = {}
                if structure:
                    try:
                        structure_ctx = structure.get_structural_context(symbol, observer)
                    except Exception as e:
                        logger.error(f"[{self.name}] Failed to get structural context from structure holon for {symbol}: {e}")
                elif oracle:
                    try:
                        structure_ctx = oracle.get_structural_context(symbol, data, current_price) if hasattr(oracle, 'get_structural_context') else {}
                    except Exception as e:
                        logger.error(f"[{self.name}] Failed to get structural context from oracle for {symbol}: {e}")

                # 4. Signal Generation
                # Check for Arbitrage
                arb_signal = None
                if arbitrage:
                    try:
                        arb_signal = getattr(arbitrage, 'get_active_signal', lambda x, y: None)(symbol, current_price)
                    except Exception as e:
                        logger.error(f"[{self.name}] Failed to get arbitrage signal for {symbol}: {e}")

                # Check for Gold Lead-Lag Arbitrage (NEW - Gold.com vs Kraken XAUT/PAXG)
                gold_lag_signal = None
                if symbol in ['XAUT/USDT', 'PAXG/USDT'] and hasattr(self, 'gold_lead_lag'):
                    try:
                        gold_lag_signal = self.gold_lead_lag.detect_lead_lag_opportunity(observer)
                        if gold_lag_signal:
                            logger.info(f"[{self.name}] 🏆 GOLD LEAD-LAG DETECTED: {gold_lag_signal['reason']}")
                    except Exception as e:
                        logger.error(f"[{self.name}] Failed to check gold lead-lag for {symbol}: {e}")

                # Check for PAXG/BTC Macro Arbitrage (NEW - Gold/Bitcoin Ratio)
                paxg_btc_signal = None
                if symbol == 'PAXG/BTC' and hasattr(self, 'paxg_btc'):
                    try:
                        paxg_btc_signal = self.paxg_btc.detect_opportunity(observer)
                        if paxg_btc_signal:
                            logger.info(f"[{self.name}] 📊 PAXG/BTC MACRO SIGNAL: {paxg_btc_signal['reason']}")
                    except Exception as e:
                        logger.error(f"[{self.name}] Failed to check PAXG/BTC for {symbol}: {e}")

                # Check for Whale
                is_whale_signal = False
                if whale:
                    try:
                        # Daily Volume for dynamic thresholds
                        recent = data.iloc[-96:] if len(data) >= 96 else data
                        daily_vol_usd = (recent['close'] * recent['volume']).sum()
                        is_whale_signal = whale.check_bid_wall(symbol, book_data, daily_vol=daily_vol_usd)
                    except Exception as e:
                        logger.error(f"[{self.name}] Failed to check whale signal for {symbol}: {e}")

                # --- DUMP/PUMP WHALE DETECTOR (Time-Window Pattern) ---
                dump_event = None
                if dump_pump_detector and getattr(config, 'DUMP_PUMP_ENABLED', True):
                    try:
                        recent_vol = (data.iloc[-96:] if len(data) >= 96 else data)
                        dv_usd = float((recent_vol['close'] * recent_vol['volume']).sum())
                        cvd_data = self.order_flow_data.get(symbol, {})
                        dump_event = dump_pump_detector.analyze(
                            symbol=symbol,
                            book_data=book_data,
                            df=data,
                            cvd_data=cvd_data,
                            daily_vol_usd=dv_usd,
                            vix_level=vix_level,
                        )
                    except Exception as e:
                        logger.debug(f"[{self.name}] DumpPump detector error on {symbol}: {e}")

                # If dump detected, handle immediately before entry signal generation
                if dump_event:
                    d_phase = dump_event.get('phase', '')
                    d_dir   = dump_event.get('direction', 'SELL')
                    d_conf  = dump_event.get('confidence', 0.6)
                    d_reason = dump_event.get('reason', 'WHALE_DUMP')
                    d_vix   = dump_event.get('vix_level', 0.0)

                    if d_phase in ('DUMP_EXHAUSTED', 'RE_ACCUMULATION'):
                        # HIGH confidence contrarian BUY — inject directly
                        try:
                            from HolonicTrader.agent_executor import TradeSignal
                            dump_buy_sig = TradeSignal(
                                symbol=symbol,
                                direction='BUY',
                                size=1.0,
                                price=current_price,
                                conviction=d_conf,
                                metadata={
                                    'reason': d_reason,
                                    'strategy': 'WHALE_DUMP_EXHAUSTION',
                                    'is_whale': True,  # Bypasses RVOL/Fair-Weather gates
                                    'vix_level': d_vix,
                                    'dump_phase': d_phase,
                                }
                            )
                            sl_pct = getattr(config, 'DEFAULT_STOP_LOSS_PCT', 0.015)
                            tp_pct = getattr(config, 'DEFAULT_TAKE_PROFIT_PCT', 0.025)
                            dump_buy_sig.stop_loss_price = current_price * (1.0 - sl_pct)
                            dump_buy_sig.take_profit_price = current_price * (1.0 + tp_pct)
                            report.append({
                                'symbol': symbol,
                                'direction': 'BUY',
                                'price': current_price,
                                'tp': dump_buy_sig.take_profit_price,
                                'sl': dump_buy_sig.stop_loss_price,
                                'conviction': d_conf,
                                'quality': 'HIGH',
                                'reason': d_reason,
                                'regime': regime,
                                'tda_status': tda_status,
                                'hps_score': 4,
                                'hps_pillars': ['WHALE_EXHAUSTION', 'CVD_ABSORPTION', 'TIME_WINDOW', 'VOLUME_SPIKE'],
                                'metadata': dump_buy_sig.metadata,
                                'expected_yield': round(tp_pct * d_conf * 100, 2),
                                'hit_probability': round(d_conf * 100, 1),
                                'decay_score': 0.9,
                                'optimal_horizon': 4,
                                'pips_potential': round(current_price * tp_pct, 4),
                                'vix_level': d_vix,
                                'execution_details': {
                                    'quantity': 0.0,  # Governor will size
                                    'leverage': 1.0,
                                    'order_type': 'LIMIT',
                                    'position_type': 'LONG'
                                },
                                'account_context': {'equity': equity, 'available': available, 'drawdown': drawdown},
                                'timestamp': datetime.now(timezone.utc).isoformat()
                            })
                            logger.warning(f"[{self.name}] 🔄 DUMP EXHAUSTION BUY INJECTED: {symbol} | conf={d_conf:.2f} | VIX={d_vix:.1f}")
                        except Exception as e:
                            logger.error(f"[{self.name}] Failed to inject dump exhaustion signal: {e}")

                    elif d_phase == 'DUMP_IN_PROGRESS':
                        # Warn: any BUY signals for this symbol this cycle should be downgraded
                        logger.warning(f"[{self.name}] ⚠️ DUMP_IN_PROGRESS on {symbol} — BUY signals will be downgraded | VIX={d_vix:.1f}")
                        # We flag it; the entry_sig generation below will still run but gets downgraded
                        is_whale_signal = False  # Prevent whale boost from conflicting

                # Standard Oracle Signals
                entry_sig = None
                if paxg_btc_signal:
                    # Priority 1: PAXG/BTC Macro (highest conviction, mean reversion)
                    try:
                        entry_sig = TradeSignal(
                            symbol=symbol,
                            direction=paxg_btc_signal['direction'],
                            size=1.0,
                            price=paxg_btc_signal.get('entry_ratio', 0.033),
                            conviction=paxg_btc_signal['confidence'],
                            metadata={
                                'reason': paxg_btc_signal['reason'],
                                'is_arb': False,  # Not traditional arb, it's macro
                                'strategy': 'MACRO_MEAN_REVERSION',
                                'zscore': paxg_btc_signal['metadata']['zscore'],
                                'paxg_btc_ratio': paxg_btc_signal['metadata']['paxg_btc_ratio'],
                            }
                        )
                        # Set TP/SL from signal
                        if paxg_btc_signal.get('target_ratio'):
                            entry_sig.take_profit_price = paxg_btc_signal['target_ratio']
                        if paxg_btc_signal.get('stop_loss_ratio'):
                            entry_sig.stop_loss_price = paxg_btc_signal['stop_loss_ratio']
                    except Exception as e:
                        logger.error(f"[{self.name}] Failed to create PAXG/BTC signal for {symbol}: {e}")
                elif gold_lag_signal:
                    try:
                        entry_sig = TradeSignal(
                            symbol=symbol,
                            direction=arb_signal['direction'],
                            size=1.0,
                            price=current_price,
                            conviction=arb_signal['confidence'],
                            metadata={'reason': arb_signal['reason'], 'is_arb': True}
                        )
                    except Exception as e:
                        logger.error(f"[{self.name}] Failed to create arbitrage trade signal for {symbol}: {e}")
                else:
                    try:
                        # Compute indicators
                        tr = pd.concat([(data['high']-data['low']), (data['high']-data['close'].shift()).abs(), (data['low']-data['close'].shift()).abs()], axis=1).max(axis=1)
                        atr = tr.rolling(14).mean().iloc[-1]

                        rolling_mean = data['close'].rolling(20).mean()
                        rolling_std = data['close'].rolling(20).std()
                        bb_vals = {
                            'upper': (rolling_mean + 2*rolling_std).iloc[-1],
                            'middle': rolling_mean.iloc[-1],
                            'lower': (rolling_mean - 2*rolling_std).iloc[-1]
                        }

                        obv = (np.sign(data['close'].diff()).fillna(0) * data['volume']).cumsum()
                        obv_slope = 0.0
                        if len(obv) >= 14:
                            from scipy.stats import linregress
                            obv_slope, _, _, _, _ = linregress(np.arange(14), obv.iloc[-14:].values)

                        metabolism = 'PREDATOR'

                        entry_sig = oracle.analyze_for_entry(
                            symbol, data, bb_vals, obv_slope, metabolism,
                            structure_ctx=structure_ctx,
                            book_data=book_data,
                            funding_rate=funding_rate,
                            observer=observer,
                            is_whale=is_whale_signal
                        )
                    except Exception as e:
                        logger.error(f"[{self.name}] Failed to analyze for entry for {symbol}: {e}")
                        continue

                # 5. Enrichment & Filtering
                if entry_sig:
                    try:
                        # Apply Asset Personality (Injects PIP-based TP)
                        # Apply Asset Personality (Injects PIP-based TP)
                        entry_sig = oracle.apply_asset_personality(symbol, entry_sig, prices=data['close'].tolist())
                        
                        if not entry_sig:
                            logger.warning(f"[{self.name}] Asset personality rejected signal for {symbol}")
                            continue

                    except Exception as e:
                        logger.error(f"[{self.name}] Failed to apply asset personality for {symbol}: {e}")

                    # Calculate Stop Loss if not present
                    if entry_sig and not entry_sig.stop_loss_price:
                        try:
                            profile = getattr(config, 'ASSET_PROFILES', {}).get(symbol, {})
                            sl_pct = profile.get('stop_loss', config.DEFAULT_STOP_LOSS_PCT)
                            if entry_sig.direction == 'BUY':
                                entry_sig.stop_loss_price = current_price * (1.0 - sl_pct)
                            else:
                                entry_sig.stop_loss_price = current_price * (1.0 + sl_pct)
                        except Exception as e:
                            logger.error(f"[{self.name}] Failed to calculate stop loss for {symbol}: {e}")

                    # Calculate Take Profit if not present
                    if entry_sig and not entry_sig.take_profit_price:
                        try:
                            tp_pct = profile.get('take_profit', config.DEFAULT_TAKE_PROFIT_PCT)
                            if entry_sig.direction == 'BUY':
                                entry_sig.take_profit_price = current_price * (1.0 + tp_pct)
                            else:
                                entry_sig.take_profit_price = current_price * (1.0 - tp_pct)
                        except Exception as e:
                            logger.error(f"[{self.name}] Failed to calculate take profit for {symbol}: {e}")

                    # Filter based on "Boss Veto" (Trend/Structure Alignment)
                    macro_trend = structure_ctx.get('macro_trend', 'NEUTRAL')
                    is_aligned = False
                    if entry_sig.direction == 'BUY' and macro_trend == 'BULLISH': is_aligned = True
                    elif entry_sig.direction == 'SELL' and macro_trend == 'BEARISH': is_aligned = True

                    if entry_sig.metadata.get('is_arb') or entry_sig.metadata.get('is_whale') or entry_sig.conviction > 0.8:
                        is_aligned = True

                    quality = "HIGH" if (is_aligned and entry_sig.conviction > 0.7) else "MEDIUM"

                    # --- NEW: Get Kraken Execution Details from Governor ---
                    # We use the oracle's calculated ATR for precision
                    try:
                        tr = pd.concat([(data['high']-data['low']), (data['high']-data['close'].shift()).abs(), (data['low']-data['close'].shift()).abs()], axis=1).max(axis=1)
                        current_atr = tr.rolling(14).mean().iloc[-1]

                        # Create signal metadata with bypass for reporting
                        exec_metadata = entry_sig.metadata.copy()
                        exec_metadata['reason'] = 'SIGNAL_PROVIDER'  # Bypass cooldowns for reporting

                        # ── SMCE v1 Layer 2: Probability Stacking Engine ──
                        is_prob_eligible = True
                        prob_size_mod = 1.0
                        
                        if governor and hasattr(governor, 'smce_prob_engine'):
                            smce_regime = "TRANSITION"
                            if hasattr(governor, 'get_smce_regime'):
                                smce_regime = governor.get_smce_regime()
                            
                            # Build context
                            cand = {"direction": entry_sig.direction, "symbol": symbol, "proposed_cluster_exposure": 0.05}
                            p_state = {"equity": equity, "cluster_exposure": 0.0, "cvar_95": 0.03} # Approximate base limits
                            m_ctx = {
                                "structure": macro_trend,
                                "momentum_aligned": is_aligned,
                                "liquidity_status": "healthy", # Assume observer filtered bad liquidity
                                "entropy": entropy_val,
                                "correlation_idx": 0.5
                            }
                            
                            prob_result = governor.smce_prob_engine.score_trade(cand, p_state, m_ctx, smce_regime)
                            is_prob_eligible = prob_result.get("eligible", False)
                            prob_size_mod = prob_result.get("size_modifier", 1.0)
                            
                            # Attach scorecard to metadata
                            exec_metadata['smce_scorecard'] = prob_result
                            
                            if not is_prob_eligible:
                                logger.info(f"[{self.name}] 🛑 PROBABILITY VETO: {symbol} rejected by Layer 2 Engine (Score: {prob_result.get('score')}).")
                                is_approved = False
                                gov_qty = 0.0
                                gov_lev = 1.0
                                
                        if getattr(locals(), 'is_prob_eligible', True):
                            is_approved, gov_qty, gov_lev = governor.calc_position_size(
                                symbol=symbol,
                                asset_price=current_price,
                                current_atr=current_atr,
                                conviction=entry_sig.conviction,
                                direction=entry_sig.direction,
                                whale_confirmed=is_whale_signal,
                                metadata=exec_metadata
                            )
                            # Apply Layer 2 Sizing Modifier
                            gov_qty *= prob_size_mod

                    except Exception as e:
                        logger.error(f"[{self.name}] Failed to calculate position size for {symbol}: {e}")
                        is_approved = False
                        gov_qty = 0.0
                        gov_lev = 5.0  # Default min leverage

                    reason = entry_sig.metadata.get('reason')
                    if not reason:
                        reason = entry_sig.metadata.get('strategy', 'TECHNICAL_SIGNAL')

                    # Add structural context to reason
                    if macro_trend != 'NEUTRAL':
                        reason += f" | {macro_trend}_FLOW"
                    if is_whale_signal:
                        reason += " | WHALE_WALL"

                    # NEW: Signal Debouncing Check (5-minute cooldown for identical signals)
                    signal_reason_base = entry_sig.metadata.get('reason', 'TECHNICAL')
                    if self.is_signal_on_cooldown(symbol, entry_sig.direction, signal_reason_base):
                        logger.info(f"[{self.name}] 🕒 SIGNAL DEBOUNCED: {symbol} {entry_sig.direction} (identical signal within 5 min)")
                        continue  # Skip this signal, it's a duplicate
                    
                    # Manual Trading Enhancement: Show VETO reason but keep signal visible
                    quality_final = quality
                    if not is_approved:
                        quality_final = "VETOED"
                        reason += " | 🛑 GOV_VETO"

                    # If high conviction but vetoed, suggest manual review
                    if not is_approved and entry_sig.conviction > 0.75:
                        reason += " (MANUAL REVIEW)"

                    # === 🚪 SIGNAL QUALITY GATE (2026-03-22) ===
                    # Early filtering to prevent churn - reject BEFORE expensive analysis
                    try:
                        from .signal_quality_gate import get_signal_quality_gate
                        
                        gate = get_signal_quality_gate()
                        
                        # Prepare signal data for quality check
                        signal_data = {
                            'symbol': symbol,
                            'direction': entry_sig.direction,
                            'quantity': gov_qty if gov_qty > 0 else 0.01,  # Estimate if not sized yet
                            'price': current_price,
                        }
                        
                        # Prepare market data
                        market_data = {
                            'spread_pct': 0.001,  # Estimate if not fetched
                            'liquidity_score': 1.0,
                        }
                        
                        # Check quality
                        passed, reject_reason = gate.passes_quality_check(signal_data, market_data)
                        
                        if not passed:
                            logger.debug(f"[{self.name}] 🚪 SIGNAL REJECTED EARLY: {symbol} {entry_sig.direction} - {reject_reason}")
                            continue  # Skip expensive analysis (Structure, Orion, ML, etc.)
                        
                    except Exception as e:
                        logger.debug(f"[{self.name}] Signal Quality Gate error: {e}")
                        # Continue if gate fails (graceful degradation)
                    # ============================================

                    # Record signal for debouncing (before adding to report)
                    self.record_signal(symbol, entry_sig.direction, signal_reason_base)

                    # === MONTE CARLO YIELD PROJECTION ===
                    expected_yield = 0.0
                    hit_probability = 0.0
                    decay_score = 1.0  # Fresh signal = 1.0, decays over time
                    optimal_horizon = 24  # Default 24 hours
                    pips_potential = 0.0
                    
                    try:
                        from HolonicTrader.sde_engine import SDEEngine
                        
                        # Estimate GBM parameters from price history
                        prices_arr = data['close'].values[-50:]
                        if len(prices_arr) >= 10:
                            gbm_params = SDEEngine.estimate_gbm_parameters(prices_arr)
                            
                            # Calculate hit probability for TP
                            if entry_sig.take_profit_price and entry_sig.take_profit_price > 0:
                                hit_probability = SDEEngine.calculate_hit_probability(
                                    'GBM', gbm_params, current_price, 
                                    entry_sig.take_profit_price, 
                                    horizon=100, paths=200
                                )
                                
                                # Calculate expected yield (TP% * probability)
                                if entry_sig.direction == 'BUY':
                                    tp_pct = (entry_sig.take_profit_price - current_price) / current_price
                                else:
                                    tp_pct = (current_price - entry_sig.take_profit_price) / current_price
                                
                                expected_yield = tp_pct * hit_probability * 100  # As percentage
                                
                                # Calculate pips (price movement potential)
                                pips_potential = abs(entry_sig.take_profit_price - current_price)
                                
                            # Estimate optimal horizon (when to take profit)
                            drift = gbm_params.get('drift', 0)
                            diffusion = gbm_params.get('diffusion', 0.1)
                            if diffusion > 0:
                                # Higher drift = faster signal, higher vol = slower decay
                                optimal_horizon = min(72, max(4, int(24 * (diffusion / (abs(drift) + 0.01)))))
                            
                            # Signal decay: starts at 1.0, decays based on market conditions
                            # Faster decay if volatility is high or drift is unfavorable
                            direction_sign = 1 if entry_sig.direction == 'BUY' else -1
                            drift_alignment = direction_sign * drift
                            decay_score = min(1.0, max(0.3, 0.7 + drift_alignment - (diffusion * 0.5)))
                            
                    except Exception as e:
                        logger.debug(f"[{self.name}] Monte Carlo projection failed for {symbol}: {e}")


                    # Evaluate HPS Confluence
                    rsi_val = 50.0
                    if 'rsi' in data.columns:
                        rsi_val = data['rsi'].iloc[-1] if 'rsi' in data else 50.0 # Placeholder if not computed
                        # Actually compute RSI if missing
                        if 'rsi' not in data and len(data) > 14:
                            delta = data['close'].diff()
                            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                            rs = gain / loss
                            rsi_val = 100 - (100 / (1 + rs)).iloc[-1]

                    hps_ctx = {
                        'data': data,
                        'structure': structure_ctx,
                        'rsi': rsi_val,
                        'entropy': entropy_val,
                        'hit_prob': hit_probability,
                        'regime': regime
                    }
                    
                    hps_result = self.check_hps_confluence(symbol, entry_sig, hps_ctx)
                    
                    # FILTER: If not HPS (Score < 3) -> REJECT or Downgrade
                    # User request: "Signal provider should be looking for high probability setups"
                    # We will Mark non-HPS as 'LOW_QUALITY' or simply 'VETOED' unless manual override.
                    
                    if not hps_result['is_hps']:
                        # Downgrade logic
                        if quality_final == 'HIGH': 
                            quality_final = 'MEDIUM' # Downgrade HIGH to MEDIUM
                            reason += f" | Low HPS ({hps_result['score']}/5)"
                        else:
                            quality_final = 'LOW' # Reject
                            # Or veto completely?
                            # Let's veto if score < 2 for safety.
                            if hps_result['score'] < 2:
                                quality_final = 'VETOED'
                                reason += f" | 🚫 WEAK SETUP ({hps_result['score']}/5)"
                    else:
                        # Upgrade or Confirm
                        reason += f" | ⭐ HPS {hps_result['score']}/5"
                        quality_final = 'HIGH' # Force HIGH if HPS confirmed

                    report.append({
                        'symbol': symbol,
                        'direction': entry_sig.direction,
                        'price': current_price,
                        'tp': entry_sig.take_profit_price,
                        'sl': entry_sig.stop_loss_price,
                        'conviction': entry_sig.conviction,
                        'quality': quality_final,
                        'reason': reason,
                        'metadata': entry_sig.metadata,
                        'regime': regime,
                        'tda_status': tda_status,
                        'hps_score': hps_result['score'], # Visualize on dashboard
                        'hps_pillars': hps_result['pillars'],
                        # === NEW: MONTE CARLO ANTICIPATION ===
                        'expected_yield': round(expected_yield, 2),  # e.g., "+3.2%"
                        'hit_probability': round(hit_probability * 100, 1),  # e.g., "72%"
                        'decay_score': round(decay_score, 2),  # 1.0 = fresh, 0.3 = stale
                        'optimal_horizon': optimal_horizon,  # Hours until signal decays
                        'pips_potential': round(pips_potential, 4),  # Raw price movement
                        'execution_details': {
                            'quantity': gov_qty,
                            'leverage': gov_lev,
                            'order_type': 'LIMIT', # Suggested for manual entry
                            'position_type': 'LONG' if entry_sig.direction == 'BUY' else 'SHORT'
                        },
                        'account_context': {
                            'equity': equity,
                            'available': available,
                            'drawdown': drawdown
                        },
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })

            except Exception as e:
                logger.error(f"[{self.name}] Unexpected error processing {symbol}: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        # --- SYNC DASHBOARD ANALYTICS ---
        self._sync_order_flow_to_disk()

        # === ML ENTRY FILTER (2026-03-22) ===
        # Filter out low-confidence signals before returning
        try:
            from .ml_advisor import predict_trade
            
            filtered_report = []
            ml_filtered_count = 0
            
            for signal in report:
                # Skip ML filter for management signals
                if signal.get('metadata', {}).get('strategy') == 'POSITION_MANAGEMENT':
                    filtered_report.append(signal)
                    continue
                
                # Get ML prediction for entry signals
                try:
                    ml_pred = predict_trade(
                        symbol=signal['symbol'],
                        direction=signal['direction'],
                        price=signal.get('price', 0),
                        quantity=1.0  # Dummy for prediction
                    )
                    
                    # Add ML data to signal metadata
                    if 'metadata' not in signal:
                        signal['metadata'] = {}
                    signal['metadata']['ml_win_prob'] = ml_pred['win_probability']
                    signal['metadata']['ml_confidence'] = ml_pred['confidence_level']
                    
                    # FIX 2026-03-22: Raised threshold from 35% to 45% for higher quality
                    if ml_pred['win_probability'] < 0.45:
                        logger.info(f"[{self.name}] 🤖 ML FILTER: Skipping {signal['symbol']} - {ml_pred['win_probability']:.1%} win prob")
                        ml_filtered_count += 1
                        continue
                    
                    # Downgrade quality for moderate confidence
                    if ml_pred['win_probability'] < 0.55 and signal['quality'] == 'HIGH':
                        signal['quality'] = 'MEDIUM'
                    
                    filtered_report.append(signal)
                    
                except Exception as e:
                    logger.debug(f"[{self.name}] ML prediction failed for {signal['symbol']}: {e}")
                    # Include signal if ML fails (graceful degradation)
                    filtered_report.append(signal)
            
            if ml_filtered_count > 0:
                logger.info(f"[{self.name}] 🤖 ML Entry Filter: Filtered {ml_filtered_count} low-confidence signals")
            
            report = filtered_report
            
        except ImportError:
            logger.info(f"[{self.name}] ML Advisor not available - skipping entry filter")
        except Exception as e:
            logger.error(f"[{self.name}] ML Entry Filter error: {e}")
        # ============================================

        # === SOLON PRIME FINAL GATEKEEPER (2026-03-22) ===
        # Final capital intelligence check before returning signals
        try:
            from .solon_prime import evaluate_trade, get_solon
            
            solon_filtered = []
            solon_rejected = 0
            
            # Get portfolio state for Solon
            portfolio_state = {
                'equity': getattr(self, 'last_known_equity', 100.0),
                'drawdown': 0.0,  # Would be fetched from Governor
                'consecutive_losses': 0,
                'positions': {},  # Current positions
            }
            
            for signal in filtered_report:
                # Run through Solon's 7-layer filter
                solon_decision = evaluate_trade(signal, portfolio_state)
                
                if solon_decision['action'] == 'APPROVE':
                    # Add Solon risk parameters to signal
                    if 'metadata' not in signal:
                        signal['metadata'] = {}
                    signal['metadata']['solon_approved'] = True
                    signal['metadata']['solon_risk'] = solon_decision.get('risk_params', {})
                    signal['metadata']['entropy_level'] = solon_decision.get('entropy_level', 0)
                    solon_filtered.append(signal)
                else:
                    solon_rejected += 1
                    logger.info(f"[{self.name}] 🏛️ SOLON REJECT: {signal['symbol']} - {solon_decision['reason']}")
            
            if solon_rejected > 0:
                logger.info(f"[{self.name}] 🏛️ Solon Prime: Rejected {solon_rejected} signals")
            
            report = solon_filtered
            
        except ImportError:
            logger.info(f"[{self.name}] Solon Prime not available - skipping final gate")
        except Exception as e:
            logger.error(f"[{self.name}] Solon Prime error: {e}")
        # ============================================

        # Sort: HIGH quality first, then conviction. VETOED signals at the bottom.
        report.sort(key=lambda x: (x['quality'] == 'HIGH', x['quality'] == 'MEDIUM', x['conviction']), reverse=True)
        return report

    def format_telegram_report(self, report: List[Dict[str, Any]]) -> str:
        """
        Format the signal report for Telegram with detailed descriptions and rich formatting.
        Includes both entry signals and position management alerts.
        """
        if not report:
            return "📭 **Signal Provider**: No high-quality signals detected in this cycle."

        # Split into management vs entry signals
        mgmt_signals = [s for s in report if s.get('metadata', {}).get('strategy') == 'POSITION_MANAGEMENT']
        entry_signals = [s for s in report if s.get('metadata', {}).get('strategy') != 'POSITION_MANAGEMENT']

        msg = ""
        
        # --- MANAGEMENT ALERTS (Priority) ---
        if mgmt_signals:
            msg += "🛡️ **POSITION MANAGEMENT ALERTS**\n"
            msg += "================================\n\n"
            for sig in mgmt_signals:
                action = sig['direction']
                urgency = sig.get('conviction', 0)
                
                # Action-specific emoji
                if action == 'URGENT_CLOSE': icon = "🚨"
                elif action == 'CLOSE': icon = "🔴"
                elif action == 'REDUCE': icon = "⚠️"
                elif action == 'STACK': icon = "🟢"
                else: icon = "📋"
                
                # Urgency bar
                urgency_bar = "█" * int(urgency * 5) + "░" * (5 - int(urgency * 5))
                
                msg += f"{icon} **{sig['symbol']}** → `{action}`\n"
                msg += f"├─ Urgency: [{urgency_bar}] `{urgency:.0%}`\n"
                msg += f"├─ Price: `{sig['price']:.4f}`\n"
                msg += f"└─ _{sig['reason']}_\n\n"
            
            msg += "--------------------------------\n\n"

        # --- ENTRY SIGNALS ---
        if entry_signals:
            acc = entry_signals[0].get('account_context', {})
            msg += f"🏛️ **Entry Signal Intelligence**\n"
            msg += f"Equity: `${acc.get('equity', 0):.2f}` | Available: `${acc.get('available', 0):.2f}`\n"
            msg += f"Drawdown: `{acc.get('drawdown', 0):.1%}`\n"
            msg += f"--------------------------------\n\n"

            for i, sig in enumerate(entry_signals[:5]):
                status_emoji = "🔥" if sig['quality'] == "HIGH" else "⚡"
                if sig['quality'] == "VETOED": status_emoji = "🛡️"
                
                exec_ctx = sig.get('execution_details', {})
                dir_emoji = "🟢 LONG" if sig['direction'] == 'BUY' else "🔴 SHORT"
                
                msg += f"{status_emoji} **{sig['symbol']}** | {dir_emoji}\n"
                msg += f"├─ **Entry**: `{sig['price']:.4f}` ({exec_ctx.get('order_type', 'LIMIT')})\n"
                msg += f"├─ **Size**: `{exec_ctx.get('quantity', 0):.4f}` | **Lev**: `{exec_ctx.get('leverage', 1):.1f}x`\n"
                msg += f"├─ 🎯 **TP**: `{sig.get('tp', 0):.4f}`\n"
                msg += f"├─ 🛑 **SL**: `{sig.get('sl', 0):.4f}`\n"
                msg += f"├─ 🧠 **Conviction**: `{sig['conviction']:.2%}`\n"
                msg += f"├─ 📊 **Status**: `{sig.get('regime', '---')}` | `{sig.get('tda_status', '---')}`\n"
                msg += f"└─ 📜 **Reason**: _{sig['reason']}_\n\n"
        elif not mgmt_signals:
            msg += "📭 No entry signals this cycle.\n\n"

        msg += f"🕒 _Generated at {datetime.now().strftime('%H:%M:%S')} UTC_"
        return msg

    def send_to_telegram(self, report: List[Dict[str, Any]], overwatch: Any = None):
        """
        Send the formatted signal report to Telegram using the Overwatch holon.

        This method handles the transmission of trading signals to Telegram,
        with fallback behavior if the Overwatch holon is unavailable.

        Args:
            report (List[Dict[str, Any]]): The list of signal dictionaries to send
            overwatch (Any, optional): The Overwatch holon instance for Telegram transmission.
                                     If None, the method will log that Telegram is disabled.

        Returns:
            None: Outputs to Telegram or logs a warning if transmission is not possible
        """
        msg = self.format_telegram_report(report)
        if overwatch and hasattr(overwatch, 'send_telegram_alert'):
            overwatch.send_telegram_alert(msg)
        else:
            print(f"[{self.name}] ⚠️ Telegram not sent (Overwatch/Token missing).")
            print(f"--- TELEGRAM MESSAGE ---\n{msg}\n-----------------------")

    # === DASHBOARD ANALYTICS GENERATION ===
    
    def _sync_order_flow_to_disk(self):
        """Write order flow data for dashboard visualization."""
        import json
        import os
        import numpy as np # Added for np.integer, np.floating, np.ndarray
        import time # Added for time.time()
        try:
            path = os.path.join(os.getcwd(), 'order_flow_status.json')
            
            # Convert numpy types to float for JSON serialization
            def convert(o):
                if isinstance(o, np.integer): return int(o)
                if isinstance(o, np.floating): return float(o)
                if isinstance(o, np.ndarray): return o.tolist()
                return o

            data = {
                'last_update': time.time(),
                'symbols': self.order_flow_data,
                'volume_profile': self.volume_profile_data
            }
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=convert)
        except Exception as e:
            logger.error(f"[{self.name}] ⚠️ Failed to sync order flow: {e}")

    def _update_order_flow(self, symbol: str, oracle, df: pd.DataFrame = None):
        """Collect order flow data for a symbol."""
        # Need access to Oracle but SignalProvider doesn't hold reference to sub_holons in self.sub_holons usually.
        # It receives them in generate_signal_report.
        # But here we need to pass them or store them.
        # I'll rely on passing `oracle` explicitly or assuming it's available.
        # Wait, the method signature needs to change if I don't have self.sub_holons.
        # I'll update the signature to accept `oracle`.
        pass

    def measure_order_flow(self, symbol: str, oracle, observer, df: pd.DataFrame = None):
        """
        Measure and store order flow data.
        """
        if not oracle or not observer: return

        try:
            # 1. Get CVD from Oracle's analyze_order_flow
            if hasattr(oracle, 'analyze_order_flow'):
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
                delta = flow.get('delta', 0.0)
                # Cap extremely large values to prevent chart distortion
                if abs(delta) > 1000000: delta = 0.0
                
                self.order_flow_data[symbol]['cvd_history'].append(float(delta))
                if len(self.order_flow_data[symbol]['cvd_history']) > 50:
                    self.order_flow_data[symbol]['cvd_history'].pop(0)
                
                self.order_flow_data[symbol]['buy_ratio'] = float(flow.get('buy_ratio', 0.5))
                self.order_flow_data[symbol]['current_delta'] = float(delta)
                self.order_flow_data[symbol]['signal'] = flow.get('signal', 'NEUTRAL')
            
            # 2. Calculate Volume Profile from OHLCV
            if df is not None and not df.empty and len(df) >= 20:
                self._calculate_volume_profile(symbol, df)
                
        except Exception as e:
            # logger.debug(f"Order Flow Error {symbol}: {e}")
            pass

    def _calculate_volume_profile(self, symbol: str, df: pd.DataFrame, bins: int = 20):
        """Calculate volume at price levels for heatmap."""
        try:
            # Use last 100 candles
            recent = df.tail(100)
            
            # Get price range
            price_min = recent['low'].min()
            price_max = recent['high'].max()
            
            if price_max <= price_min: return
            
            # Create price bins
            bin_size = (price_max - price_min) / bins
            volume_at_price = []
            
            # Iterate bins
            for i in range(bins):
                level_low = price_min + (i * bin_size)
                level_high = level_low + bin_size
                
                # Sum volume for candles touching this level
                # Approximation: if candle mid-point is in bin
                mask = (recent['close'] >= level_low) & (recent['close'] < level_high)
                vol = recent.loc[mask, 'volume'].sum()
                
                volume_at_price.append([float(level_low), float(vol)])
            
            self.volume_profile_data[symbol] = volume_at_price
            
        except Exception as e:
            pass

    def _write_monte_carlo_results(self, symbol: str, current_price: float, sde_params: dict):
        """
        Generate and write Monte Carlo projection paths for the dashboard.
        """
        import json
        import os
        import numpy as np # Added for np.median, np.percentile
        import time # Added for time.time()
        from HolonicTrader.sde_engine import SDEEngine
        
        try:
            # Generate paths
            # 50 paths, 48 hours horizon?
            paths = SDEEngine.simulate_paths(
                model='GBM',
                params=sde_params,
                start_price=current_price,
                horizon=48, # 48 steps (hours)
                paths=50,
                dt=1/24 # Hourly steps
            )
            
            # Calculate percentiles
            p50 = np.median(paths, axis=0).tolist()
            p95_upper = np.percentile(paths, 95, axis=0).tolist()
            p95_lower = np.percentile(paths, 5, axis=0).tolist()
            
            path_count = len(paths)
            
            data = {
                'timestamp': time.time(),
                'symbol': symbol,
                'current_price': current_price,
                'horizon': 48,
                'paths': path_count,
                'p50': p50,
                'p95_upper': p95_upper,
                'p95_lower': p95_lower,
                'params': sde_params
            }
            
            # Write to single file (overwriting previous asset? Or accumulating?)
            # Dashboard expects a single object for the "Active" view or map.
            # Let's write just the LATEST one analyzed (usually BTC/ETH).
            # Or if we want to separate, we'd need a map. dashboard_gui.py seemed to handle one or map.
            # React component expects `data.paths`, `data.p50`. So single object.
            # We will prioritize BTC, then ETH, then Majors.
            
            # Prioritize writing Major assets or high conviction signals
            should_write = False
            if symbol in ['BTC/USDT', 'XBT/USD', 'ETH/USDT', 'SOL/USDT']: should_write = True
            elif sde_params and entry_sig and entry_sig.conviction > 0.6: should_write = True
            elif not os.path.exists('monte_carlo_results.json'): should_write = True
            
            if should_write:
                file_path = os.path.join(os.getcwd(), 'monte_carlo_results.json')
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                    
        except Exception as e:
            logger.error(f"[{self.name}] MC Write Error: {e}")

    def get_dashboard_state(self) -> dict:
        """Expose SignalProvider data for the dashboard."""
        return {
            'order_flow': getattr(self, 'order_flow_data', {}),
            'monte_carlo': getattr(self, 'monte_carlo_data', {}),
        }

    def receive_message(self, sender: Any, content: Any) -> Any:
        """
        Handle incoming messages.
        """
        pass
