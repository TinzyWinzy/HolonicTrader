"""
ArbitrageHolon - The "Yield Hunter" Brain (Phase 40)

Specialized in:
1. Spatial Spread Monitoring (KuCoin vs. Kraken).
2. Funding Rate Yield Detection (Basis Trading).
3. Cross-Exchange Conviction Injection.
4. xStocks Funding Arbitrage (SPYX, AAPLX, etc.)
"""

import time
from typing import Any, Dict, List, Optional
from HolonicTrader.holon_core import Holon, Disposition
import config
import numpy as np

# xStocks symbols for funding arbitrage
XSTOCKS_SYMBOLS = [
    'SPYX/USDT', 'QQQX/USDT',  # ETFs
    'NVDAX/USDT', 'AAPLX/USDT', 'GOOGLX/USDT', 'TSLAX/USDT', 'MSTRX/USDT',  # Tech
    'CRCLX/USDT', 'HOODX/USDT',  # Others
]

class ArbitrageHolon(Holon):
    def __init__(self, name: str = "ArbHunter"):
        # High Autonomy (decides what is a 'good' arb), High Integration (feeds into Oracle)
        super().__init__(name=name, disposition=Disposition(autonomy=0.8, integration=0.8))
        self.kucoin_connected = False
        self.kraken_connected = False
        self.price_spreads = {} # symbol -> {long: %, short: %}
        self.funding_yields = {} # symbol -> 8H Yield
        self.funding_context = {} # symbol -> {regime, lsi}
        self.last_sync = 0

        # Load Risk Parameters
        self.min_profit = getattr(config, 'ARBITRAGE_MIN_PROFIT_PCT', 0.005)
        self.max_drawdown = getattr(config, 'ARBITRAGE_MAX_DRAWDOWN', 0.10)
        self.max_positions = getattr(config, 'ARBITRAGE_MAX_POSITIONS', 5)

        # Flexline Agent Reference (injected by system)
        self.flexline_agent = None

        # Phase 3 Integration
        try:
            from phase3_execution import RUST_AVAILABLE, get_phase3
            self.rust_available = RUST_AVAILABLE
            self.phase3 = get_phase3()
        except ImportError:
            self.rust_available = False
            self.phase3 = None
        self.kucoin_observer = None
        self.kraken_observer = None
        self.xstocks_arb = None  # Lazy-loaded xStocks arbitrage scanner
        self.xstocks_funding_cache = {}  # Cache xStocks funding data
        self.last_xstocks_fetch = 0

        # === FIX 2026-03-04: ATR CACHE ===
        # Cache ATR values to avoid recalculating every cycle
        self._atr_cache = {}  # {symbol: {'value': float, 'timestamp': float}}
        self._atr_cache_duration = 300  # Cache ATR for 5 minutes

    def get_effective_capital(self, base_capital: float) -> float:
        """
        Get effective capital including Flexline credit boost.

        Args:
            base_capital: Base equity/capital available

        Returns:
            Effective capital with Flexline boost
        """
        if not self.flexline_agent or not self.flexline_agent.enabled:
            return base_capital

        # Get available Flexline credit for arbitrage
        flexline_available = self.flexline_agent.get_available_for_trading()

        # Apply arb allocation percentage (50% of available credit for arb)
        arb_allocation = flexline_available * getattr(config, 'FLEXLINE_ARB_ALLOCATION_PCT', 0.50)

        # Cap at max utilization
        max_flexline = base_capital * getattr(config, 'FLEXLINE_MAX_UTILIZATION', 0.50)
        flexline_to_use = min(arb_allocation, max_flexline)

        effective = base_capital + flexline_to_use

        if flexline_to_use > 0:
            print(f"[{self.name}] 💳 FLEXLINE BOOST: ${base_capital:.2f} + ${flexline_to_use:.2f} = ${effective:.2f}")

        return effective

    def get_atr(self, symbol: str, period: int = None) -> float:
        """
        FIX 2026-03-04: Calculate Average True Range for volatility-adjusted arb filter.
        
        Args:
            symbol: Asset symbol (e.g., 'BTC/USDT')
            period: ATR calculation period (default from config)
        
        Returns:
            ATR value in price terms (e.g., $50 for BTC)
        """
        if period is None:
            period = getattr(config, 'ARB_ATR_PERIOD', 14)
        
        # Check cache first
        now = time.time()
        if symbol in self._atr_cache:
            cache_age = now - self._atr_cache[symbol]['timestamp']
            if cache_age < self._atr_cache_duration:
                return self._atr_cache[symbol]['value']
        
        # Fetch OHLCV data from Kraken observer
        if not self.kraken_observer:
            return 0.0
        
        try:
            # Map symbol to exchange format
            exec_symbol = config.KRAKEN_SYMBOL_MAP.get(symbol, symbol)
            
            # Fetch OHLCV (1-hour timeframe for arb volatility)
            ohlcv = self.kraken_observer._fetch_ohlcv_resilient(exec_symbol, '1h', since=None, limit=period + 1)
            
            if not ohlcv or len(ohlcv) < period + 1:
                return 0.0
            
            # Calculate True Range for each period
            tr_list = []
            for i in range(1, len(ohlcv)):
                # OHLCV format: [timestamp, open, high, low, close, volume]
                high = float(ohlcv[i][2])
                low = float(ohlcv[i][3])
                prev_close = float(ohlcv[i-1][4])
                
                # True Range = max of:
                # 1. High - Low (current range)
                # 2. |High - Previous Close| (gap up)
                # 3. |Low - Previous Close| (gap down)
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                tr_list.append(tr)
            
            # Calculate Average True Range
            if len(tr_list) == 0:
                return 0.0
            
            atr = sum(tr_list) / len(tr_list)
            
            # Cache the result
            self._atr_cache[symbol] = {
                'value': atr,
                'timestamp': now
            }
            
            return atr
            
        except Exception as e:
            print(f"[{self.name}] ⚠️ ATR calculation failed for {symbol}: {e}")
            return 0.0

    def check_funding_convergence(self, symbol: str, current_funding_rate: float) -> bool:
        """
        FIX 2026-03-03: Close ARB position if funding rate converges below threshold.
        Returns True if position should be closed.
        """
        threshold = getattr(config, 'FUNDING_CONVERGENCE_THRESHOLD', 0.10)  # 10% per 8H
        
        # If funding rate drops below threshold, close position
        if abs(current_funding_rate) < threshold:
            print(f"[{self.name}] 📉 FUNDING CONVERGENCE: {symbol} funding {current_funding_rate*100:.2f}% < {threshold*100:.1f}% threshold -> CLOSE POSITION")
            return True
        
        return False

    def perform_sync(self, symbols: List[str]):
        """
        Fetch data from both observers and update internal arb state.
        UPGRADE: Uses Bid/Ask and Timestamp for precise 'Lead-Lag' detection.
        UPGRADE 2: Now includes xStocks funding arbitrage data.
        """
        if not self.kucoin_observer or not self.kraken_observer:
            return

        # FIX: Filter to KuCoin-compatible symbols (spot only, no xStocks/PF_ symbols)
        kucoin_symbols = []
        for s in symbols:
            # Skip xStocks (SPYX, QQQX, NVDAX, etc.) - KuCoin doesn't have them
            if any(xs in s for xs in ['SPYX', 'QQQX', 'NVDAX', 'AAPLX', 'GOOGLX',
                                       'TSLAX', 'MSTRX', 'CRCLX', 'HOODX']):
                continue
            # Skip any PF_ prefix symbols (Kraken Futures only)
            if s.startswith('PF_'):
                continue
            kucoin_symbols.append(s)

        # 1. Fetch Tickers in parallel
        kucoin_tickers = self.kucoin_observer.fetch_tickers_batch(kucoin_symbols)

        # Kraken Futures - ALL symbols (crypto + xStocks)
        # FIX 2026-03-02: xStocks are on Futures, NOT Spot
        kraken_symbols = [config.KRAKEN_SYMBOL_MAP.get(s, s) for s in symbols]
        kraken_tickers = self.kraken_observer.fetch_tickers_batch(kraken_symbols)

        # REMOVED: kraken_spot_observer - xStocks don't exist on Kraken Spot

        current_time = time.time()
        max_staleness = getattr(config, 'ARB_MAX_STALENESS', 10.0) # Default 10s staleness limit

        for sym in symbols:
            # KuCoin Data (Source/Lead)
            k_ticker = kucoin_tickers.get(sym)
            if not k_ticker:
                continue

            ku_bid = float(k_ticker.get('bid', 0.0) or 0.0)
            ku_ask = float(k_ticker.get('ask', 0.0) or 0.0)
            ku_bid_size = float(k_ticker.get('bidVolume', 1.0) or 1.0)
            ku_ask_size = float(k_ticker.get('askVolume', 1.0) or 1.0)
            ku_last = float(k_ticker.get('last', 0.0) or 0.0)
            ku_ts = k_ticker.get('timestamp') # CCXT returns ms usually

            # Normalize timestamp to seconds
            if ku_ts and ku_ts > 1e10: ku_ts /= 1000.0

            # Check Staleness
            if ku_ts and (current_time - ku_ts) > max_staleness:
                # print(f"[{self.name}] ⚠️ KuCoin data stale for {sym} ({current_time - ku_ts:.1f}s)")
                continue

            # Kraken Data (Target/Lag)
            kr_sym = config.KRAKEN_SYMBOL_MAP.get(sym, sym)
            kr_ticker = kraken_tickers.get(kr_sym)
            if not kr_ticker:
                 continue

            kr_bid = float(kr_ticker.get('bid', 0.0) or 0.0)
            kr_ask = float(kr_ticker.get('ask', 0.0) or 0.0)
            kr_bid_size = float(kr_ticker.get('bidVolume', 1.0) or 1.0)
            kr_ask_size = float(kr_ticker.get('askVolume', 1.0) or 1.0)
            kr_last = float(kr_ticker.get('last', 0.0) or 0.0)
            kr_ts = kr_ticker.get('timestamp')

             # Normalize timestamp to seconds
            if kr_ts and kr_ts > 1e10: kr_ts /= 1000.0

            if kr_ts and (current_time - kr_ts) > max_staleness:
                 continue

            # Valid Data?
            if ku_bid > 0 and ku_ask > 0 and kr_bid > 0 and kr_ask > 0:
                 self.calculate_spatial_spread_precise(sym, ku_bid, ku_ask, kr_bid, kr_ask)
                 if self.rust_available:
                     self.phase3.update_arb_price(sym, "kucoin", ku_bid, ku_ask, ku_bid_size, ku_ask_size)
                     self.phase3.update_arb_price(sym, "kraken", kr_bid, kr_ask, kr_bid_size, kr_ask_size)
            elif ku_last > 0 and kr_last > 0:
                 # Fallback to last price if bid/ask missing (rare)
                 self.calculate_spatial_spread(sym, ku_last, kr_last)

            # Funding & Open Interest (Kraken only)
            funding, oi = self.kraken_observer.fetch_funding_and_oi(sym)
            if funding != 0 or oi != 0:
                # === FIX 2026-03-04: Fetch ATR for volatility-adjusted filter ===
                atr = self.get_atr(sym)
                self.analyze_funding_and_oi(sym, funding, oi, atr)

        # 2. Fetch xStocks Funding Data (CRITICAL ADDITION)
        # xStocks are NOT in the main symbols list for spatial arb, but we need their funding data
        # Funding data must be pulled from the Futures client (kraken_observer), since spot lacks funding
        self._sync_xstocks_funding()
        
    def _sync_xstocks_funding(self):
        """
        Fetch and cache funding rates for xStocks (SPYX, AAPLX, etc.)
        
        xStocks have extreme funding rates (-7000% to +1000% APY) due to:
        - High demand for leverage on traditional equity exposure
        - Limited supply of xStock contracts
        - Crypto market structure inefficiencies
        
        This data is used for funding arbitrage signals.
        """
        current_time = time.time()
        
        # FIX 2026-03-05: Gate entire xStocks sync with the ENABLE_XSTOCKS_ARB config flag.
        # Previously mine_liquidity() was gated but this sync function still ran every 2 min,
        # fetching data and logging xSTOCKS SHORT ARB lines even when xStocks were disabled.
        if not getattr(config, 'ENABLE_XSTOCKS_ARB', False):
            return  # Silently skip — xStocks disabled

        # Cache for 2 minutes (funding rates change slowly)
        if current_time - self.last_xstocks_fetch < 120 and self.xstocks_funding_cache:

            # Restore cached data to funding_yields
            for sym, yield_8h in self.xstocks_funding_cache.items():
                self.funding_yields[sym] = yield_8h
                if hasattr(self, 'funding_context') and sym in self.funding_context:
                     pass # keep context if we have it
            return
        
        # Lazy-load xStocks arbitrage scanner
        if self.xstocks_arb is None:
            try:
                from .strategy_xstocks_arb import XStocksArbitrage
                self.xstocks_arb = XStocksArbitrage(exchange=self.kraken_observer.exchange if hasattr(self.kraken_observer, 'exchange') else None)
            except Exception as e:
                print(f"[{self.name}] ⚠️ Failed to load xStocks arb scanner: {e}")
                return
        
        try:
            # Fetch xStocks funding data
            funding_data = self.xstocks_arb.fetch_xstocks_funding()
            
            xstocks_count = 0
            for symbol, data in funding_data.items():
                if 'error' in data or 'apy' not in data:
                    continue
                
                apy = data['apy']
                oi = data.get('open_interest', 0)

                # === FIX 2026-03-04: OI FILTER FOR XSTOCKS ===
                # Skip xStocks with very low OI to avoid slippage
                min_oi_xstocks = getattr(config, 'ARB_MIN_OI_XSTOCKS', 500.0)
                if oi < min_oi_xstocks:
                    continue  # Skip illiquid xStocks
                
                # Skip negligible funding rates
                if abs(apy) < 10.0:  # Ignore < 10% APY for xStocks
                    continue
                
                # Convert marketing APY into mathematical 8H Gross Yield
                gross_yield_8h = apy / 10.95
                
                # Store in both cache and active funding_yields
                # Note: We use the USDT symbol format for consistency
                self.funding_yields[symbol] = gross_yield_8h
                self.xstocks_funding_cache[symbol] = gross_yield_8h
                xstocks_count += 1
                
                # Log extreme funding rates (these are the arb opportunities!)
                if abs(apy) > 100:
                    direction = "SHORT" if apy < 0 else "LONG"
                    print(f"[{self.name}] 📊 xSTOCKS {direction} ARB | {symbol}: {gross_yield_8h:.2f}%/8H (OI: ${data.get('open_interest', 0):,.0f})")
            
            self.last_xstocks_fetch = current_time
            
            if xstocks_count > 0:
                print(f"[{self.name}] ✅ Synced {xstocks_count} xStocks funding rates")
                
        except Exception as e:
            print(f"[{self.name}] ⚠️ xStocks funding sync failed: {e}")
        
    def calculate_spatial_spread_precise(self, symbol: str, ku_bid: float, ku_ask: float, kr_bid: float, kr_ask: float):
        """
        Calculate Realizable Spread considering Bid/Ask (Slippage Awareness).
        
        Scenario A: LONG Signal (Kraken Cheap)
        We BUY at Kraken ASK. We Sell at KuCoin BID (Theoretical Exit).
        Spread = (KuCoin_Bid - Kraken_Ask) / Kraken_Ask

        Scenario B: SHORT Signal (Kraken Expensive)
        We SELL at Kraken BID. We Buy at KuCoin ASK (Theoretical Exit).
        Spread = (Kraken_Bid - KuCoin_Ask) / KuCoin_Ask
        """
        # 1. Long Opportunity (Kraken Discount)
        # Yield = (Exit - Entry) / Entry
        spread_long = (ku_bid - kr_ask) / kr_ask

        # 2. Short Opportunity (Kraken Premium)
        spread_short = (kr_bid - ku_ask) / ku_ask

        # Fee Adjustment (Effective Spread)
        fee_buffer = getattr(config, 'ARB_MIN_PROFIT_SPREAD', 0.002)
        spread_long -= fee_buffer
        spread_short -= fee_buffer

        # Store specialized spreads
        self.price_spreads[symbol] = {
            'long': spread_long,
            'short': spread_short,
            'simple': (kr_ask - ku_bid) / ku_bid # For legacy reference
        }

        threshold = getattr(config, 'ARB_SPATIAL_THRESHOLD', 0.005)

        if spread_long > threshold:
             print(f"[{self.name}] 📉 DISCOUNT ARB (LONG) | {symbol}: {spread_long*100:.2f}% (Buy Kra: {kr_ask} -> Sell KuC: {ku_bid})")
        
        if spread_short > threshold:
             print(f"[{self.name}] 📈 PREMIUM ARB (SHORT) | {symbol}: {spread_short*100:.2f}% (Sell Kra: {kr_bid} -> Buy KuC: {ku_ask})")


    def calculate_spatial_spread(self, symbol: str, kucoin_price: float, kraken_price: float) -> float:
        """Legacy: Calculate the percentage spread between two prices (Last Price)."""
        if not kucoin_price or not kraken_price:
            return 0.0
        
        # Spread = (Target - Source) / Source
        spread = (kraken_price - kucoin_price) / kucoin_price
        
        # Legacy compatibility store
        if isinstance(self.price_spreads.get(symbol), dict):
             self.price_spreads[symbol]['simple'] = spread
        else:
             self.price_spreads[symbol] = spread
        
        return spread

    def analyze_funding_and_oi(self, symbol: str, funding_rate: float, oi: float, atr: float = None) -> float:
        """
        Calculates 8H Gross Yield, Liquidity Weighting, and Leverage Stress Index (LSI).
        Replaces legacy APY methodology.
        
        FIX 2026-03-04: Filter out low OI assets to prevent slippage bleed.
        FIX 2026-03-04: Add volatility-adjusted filter - funding must exceed ATR*2
        """
        # === FIX 2026-03-04: MINIMUM OI FILTER ===
        # Skip assets with low open interest to avoid slippage on illiquid markets
        min_oi = getattr(config, 'ARB_MIN_OPEN_INTEREST', 1000.0)  # Default $1000
        if oi > 0 and oi < min_oi:
            # Silently skip - don't log to avoid spam
            return 0.0
        
        # 1. Normalize Rate
        corrected_rate = funding_rate
        if abs(funding_rate) > 0.005:
            if abs(funding_rate) > 0.20:
                 corrected_rate = funding_rate / 100.0
            elif abs(funding_rate) > 0.01:
                 corrected_rate = funding_rate / 10.0

        gross_yield_8h = corrected_rate * 100.0 # Percentage per epoch

        # === FIX 2026-03-04: VOLATILITY-ADJUSTED FILTER ===
        # Funding yield must exceed volatility risk (ATR * multiplier)
        # ATR represents expected price move per period
        # If funding < ATR*2, price can move more than funding collected
        if atr is not None and atr > 0:
            min_funding_multiplier = getattr(config, 'ARB_MIN_FUNDING_VS_VOLATILITY', 2.0)
            min_funding_required = atr * min_funding_multiplier
            
            # Convert funding to $/hour for comparison
            # gross_yield_8h is % per 8H, assume $100 notional for comparison
            funding_per_hour = abs(gross_yield_8h) / 8  # % per hour
            
            if funding_per_hour < min_funding_required:
                # Funding doesn't justify volatility risk
                return 0.0

        # 2. Track Funding Sub-State for Stress Mathematics
        now = time.time()
        if not hasattr(self, '_funding_history'): self._funding_history = {}
        if symbol not in self._funding_history: self._funding_history[symbol] = []

        self._funding_history[symbol].append({'ts': now, 'rate': corrected_rate, 'oi': oi})
        self._funding_history[symbol] = self._funding_history[symbol][-50:] # Keep last 50 updates

        # 3. Calculate Derivatives (Context Models)
        velocity = 0.0
        oi_delta = 0.0
        if len(self._funding_history[symbol]) >= 2:
            prev = self._funding_history[symbol][-2]
            dt_hours = max(0.1, (now - prev['ts']) / 3600.0)
            velocity = (corrected_rate - prev['rate']) / dt_hours
            if prev['oi'] > 0:
                oi_delta = (oi - prev['oi']) / prev['oi']

        # 4. Liquidity Normalization (Fade extreme rates on ghost-town books)
        liquidity_multiplier = 1.0
        if oi > 0:
            if oi < 10_000_000:       # Noise Protocol
                liquidity_multiplier = 0.1
            elif oi < 50_000_000:     # Scaling up...
                liquidity_multiplier = 0.1 + 0.9 * ((oi - 10_000_000) / 40_000_000)

        weighted_yield_8h = gross_yield_8h * liquidity_multiplier

        # 5. Leverage Stress Index (LSI) Calculation
        mag_stress = min(100, abs(gross_yield_8h) * 100) # 1% per 8h = max 100 mag stress
        # Penalize if velocity is accelerating in direction of the trend
        vel_stress = 0
        if np.sign(velocity) == np.sign(corrected_rate) and velocity != 0:
            vel_stress = min(50, abs(velocity) * 1000)
        # Penalize if crowd is piling in
        oi_stress = min(50, max(0, oi_delta * 1000))

        # Composite LSI
        lsi = min(100, (mag_stress * 0.5) + (vel_stress * 0.25) + (oi_stress * 0.25))

        # 6. Regime Classification
        regime = "NEUTRAL"
        if lsi > 70 and oi_delta < 0 and np.sign(velocity) != np.sign(corrected_rate):
            regime = "DISLOCATION" # Reversion starting
        elif lsi > 50:
            regime = "CROWDED"     # Trend exhaustive

        # Store for downstream modules
        self.funding_yields[symbol] = weighted_yield_8h

        if not hasattr(self, 'funding_context'): self.funding_context = {}
        self.funding_context[symbol] = {
            'rate': corrected_rate,
            'oi': oi,
            'velocity': velocity,
            'oi_delta': oi_delta,
            'lsi': lsi,
            'regime': regime,
            'gross_yield_8h': gross_yield_8h,
            'liquidity_multiplier': liquidity_multiplier
        }

        threshold = getattr(config, 'ARB_FUNDING_THRESHOLD', 0.0001)
        if abs(corrected_rate) >= threshold and gross_yield_8h != 0.0:
             status = "YIELD (LONG)" if corrected_rate < 0 else "COST (LONG)"
             display_yield = abs(gross_yield_8h)
             # Log only if moderate or high stress to avoid purely noisy output
             if lsi > 10.0:
                 print(f"[{self.name}] 💰 {status} | {symbol}: {display_yield:.3f}%/8h (LSI: {lsi:.1f} | Regime: {regime})")

        return weighted_yield_8h


    def get_arb_conviction(self, symbol: str) -> float:
        """
        Returns a conviction boost [-1.0, 1.0] based on arbitrage opportunities.
        """
        boost = 0.0
        
        # Funding Yield Logic (Basis Trade)
        # If Funding is Negative (-0.25%), Shorts PAY Longs.
        # Yield for LONG = abs(NegativeRate).
        # Yield for SHORT = NegativeRate (Cost).
        
        funding_yield = self.funding_yields.get(symbol, 0.0)
        
        # Handle new dict structure or legacy float
        spread_data = self.price_spreads.get(symbol, 0.0)
        if isinstance(spread_data, dict):
            spread = spread_data.get('simple', 0.0)
        else:
            spread = spread_data
        
        # Scenario A: LONG (Farm Negative Funding)
        if funding_yield > 0.045: # Huge Positive Yield (Longs earn > 0.045% per 8H)
             # If Spread is small (< 1%), it's a Gold Nugget.
             if spread < 0.01:
                 boost += 0.3 # Strong Long Boost (Yield Farm)
                 
        # Scenario B: SHORT (Spatial Arb)
        if spread > 0.01:
             # Price is 1% premium. We want to SHORT.
             if funding_yield > -0.045: # Cost is less than 0.045% per 8H (or we earn yield)
                 boost -= 0.3 # Strong Short Boost
                 
        return boost

    def get_active_signal(self, symbol: str, current_price: float, spread: float = None, funding_yield: float = None) -> Optional[Dict]:
        """
        Generates a direct trade signal if a 'Gold Nugget' arb exists.
        Integrates Context Regimes (Funding Reversion vs Carry Trade).
        Returns: Dict with 'direction', 'confidence', 'reason'
        """
        if funding_yield is None:
            funding_yield = self.funding_yields.get(symbol, 0.0)

        # Retrieve Regime context
        context = getattr(self, 'funding_context', {}).get(symbol, {})
        regime = context.get('regime', 'NEUTRAL')
        lsi = context.get('lsi', 0.0)

        # Check if this is an xStock (they have extreme funding rates)
        is_xstock = any(xs in symbol for xs in ['SPYX', 'QQQX', 'NVDAX', 'AAPLX', 'GOOGLX', 
                                                  'TSLAX', 'MSTRX', 'CRCLX', 'HOODX'])
        
        # DEFENSIVE CAP: 8H yields cap
        if is_xstock:
            if abs(funding_yield) > 10.0:  # Cap at ±10% per 8H for xStocks (~11000% APY)
                funding_yield = np.sign(funding_yield) * 10.0
        else:
            if abs(funding_yield) > 0.5:  # Cap at ±0.5% per 8H for crypto (~550% APY)
                funding_yield = np.sign(funding_yield) * 0.5

        # Extract precise spreads
        spread_data = self.price_spreads.get(symbol, 0.0)
        spread_long = spread_data.get('long', 0.0) if isinstance(spread_data, dict) else -spread_data
        spread_short = spread_data.get('short', 0.0) if isinstance(spread_data, dict) else spread_data
        
        # 1. NEW REGIME: FUNDING REVERSION
        # If funding is extreme but the regime is Dislocating (velocity flipped, crowd unwinding)
        # We actively trade the unwind direction.
        if regime == "DISLOCATION" and lsi > 70:
            # If the yield is positive (Shorts heavily paid), but unwinding -> Go LONG targeting short squeeze
            if funding_yield > 0.1:
                 return {
                    'direction': 'BUY',
                    'confidence': 0.95,
                    'reason': f"FUNDING_REVERSION (LSI {lsi:.0f}, Reverting Yield)",
                    'is_xstock': is_xstock,
                    'lsi': lsi,
                    'regime': regime
                 }
            elif funding_yield < -0.1:
                 return {
                    'direction': 'SELL',
                    'confidence': 0.95,
                    'reason': f"FUNDING_REVERSION (LSI {lsi:.0f}, Reverting Yield)",
                    'is_xstock': is_xstock,
                    'lsi': lsi,
                    'regime': regime
                 }

        # 2. TOXIC FUNDING FILTER (Cost-side)
        if not is_xstock and abs(funding_yield) > 1.0: # Block > 1% per 8H pure cost positions
            print(f"[{self.name}] ☣️ TOXIC INSTRUMENT DETECTED: {symbol} - Funding {funding_yield:.2f}%/8h is extreme. Avoiding.")
            return None

        # 3. BASIS YIELD TRADE (Capture Funding)
        
        # CASE A: POSITIVE RATE (>0) -> Longs Pay Shorts. Get SHORT to earn yield.
        max_short_yield = 10.0 if is_xstock else 0.5
        min_short_yield = 0.05 if is_xstock else 0.02 # Lower threshold: 0.02% per 8H (approx 22% APY)
        
        if min_short_yield <= funding_yield <= max_short_yield:
            # Reduce confidence if crowded
            base_confidence = 0.90 if not is_xstock else 0.95
            confidence = base_confidence - (lsi / 500.0) # slightly dock confidence as stress goes up
            if spread_short > -0.005:
                return {
                    'direction': 'SELL', # SHORT to capture Positive Funding
                    'confidence': confidence,
                    'reason': f"BASIS_CARRY_SHORT ({funding_yield:.3f}%/8H)",
                    'gross_yield_8h': abs(funding_yield),
                    'is_xstock': is_xstock,
                    'lsi': lsi,
                    'regime': regime
                }

        # CASE B: NEGATIVE RATE (<0) -> Shorts pay Longs. Get LONG to earn yield.
        min_long_yield = -10.0 if is_xstock else -0.5
        max_long_yield = -0.05 if is_xstock else -0.02
        
        if min_long_yield <= funding_yield <= max_long_yield:
            base_confidence = 0.95 if is_xstock else 0.90
            confidence = base_confidence - (lsi / 500.0)
            if spread_long > -0.005:
                yield_val = abs(funding_yield)
                return {
                    'direction': 'BUY', # LONG to capture Negative Funding
                    'confidence': confidence,
                    'reason': f"BASIS_CARRY_LONG ({yield_val:.3f}%/8H)",
                    'gross_yield_8h': yield_val,
                    'is_xstock': is_xstock,
                    'lsi': lsi,
                    'regime': regime
                }
            
        # 2. SPATIAL REVERSION (Spread Arb)
        # Uses Precise Bid/Ask Spreads now.
        if not getattr(self, 'rust_available', False):
            # SHORT Signal: Kraken Bid > KuCoin Ask (Premium)
            if spread_short > 0.010: # > 1% Realizable Premium
                # Cost check: If funding < -100, Shorting costs >100% APY. Veto.
                if funding_yield > -100.0:
                    confidence = min(0.95, 0.6 + (abs(spread_short) * 10))
                    return {
                        'direction': 'SELL',
                        'confidence': confidence,
                        'reason': f"SPATIAL_ARB_SHORT (Premium {spread_short*100:.2f}%)",
                        'lsi': lsi,
                        'regime': regime
                    }

            # LONG Signal: KuCoin Bid > Kraken Ask (Discount)
            if spread_long > 0.010: # > 1% Realizable Discount
                # Cost check: If funding > 100, Longing costs >100% APY. Veto.
                if funding_yield < 100.0:
                    confidence = min(0.95, 0.6 + (abs(spread_long) * 10))
                    return {
                        'direction': 'BUY',
                        'confidence': confidence,
                        'reason': f"DISCOUNT_ARB_LONG (Discount {spread_long*100:.2f}%)",
                        'lsi': lsi,
                        'regime': regime
                    }
            
        return None

    def is_toxic_funding_instrument(self, symbol: str) -> bool:
        """
        Check if an instrument has toxic funding rates that should trigger position closure.
        Returns True if the instrument has funding rates that exceed safe thresholds.
        """
        funding_yield = self.funding_yields.get(symbol, 0.0)
        # 0.45% per 8H is roughly 500% APY. Anything higher is completely toxic to hold.
        toxic_threshold = 0.45  
        return abs(funding_yield) > toxic_threshold

    def get_funding_risk_level(self, symbol: str) -> str:
        """
        Return the risk level of funding for an instrument (Calculated on 8H basis).
        """
        funding_yield = self.funding_yields.get(symbol, 0.0)
        abs_yield = abs(funding_yield)

        if abs_yield > 0.9:       # ~1000% APY
            return 'TOXIC'
        elif abs_yield > 0.45:    # ~500% APY
            return 'HIGH'
        elif abs_yield > 0.09:    # ~100% APY
            return 'MODERATE'
        else:
            return 'LOW'

    def get_dashboard_state(self) -> dict:
        """Expose arbitrage data for the dashboard."""
        opportunities = []
        all_symbols = set(self.funding_yields.keys()) | set(self.price_spreads.keys())
        for sym in all_symbols:
            funding_apy = self.funding_yields.get(sym, 0.0)
            spread_data = self.price_spreads.get(sym, 0.0)
            spread = spread_data.get('simple', 0.0) if isinstance(spread_data, dict) else spread_data
            sig = self.get_active_signal(sym, 0, spread=spread, funding_apy=funding_apy)
            if funding_apy != 0.0 or sig or abs(spread) > 0.001:
                opportunities.append({
                    'symbol': sym,
                    'funding_apy': round(funding_apy, 1),
                    'spread_pct': round(spread * 100, 2),
                    'signal': sig.get('direction') if sig else None,
                    'reason': sig.get('reason', '') if sig else '',
                    'confidence': sig.get('confidence', 0) if sig else 0,
                    'has_opportunity': sig is not None,
                })
        return {'arbitrage': opportunities}

    def receive_message(self, sender: Any, content: Any) -> None:
        """Process price updates from Observer."""
        pass

    def check_gold_arb(self, kraken_ticker: dict) -> Optional[Dict]:
        """
        Specialized Logic for Gold Oracle Arbitrage (Spot vs PAXG).
        """
        if not kraken_ticker: return None
        
        # We need the Oracle Holon to access the GoldOracle, OR we instantiate it locally?
        # Architecture: Oracle Holon owns Data. Arb Holon owns Logic.
        # But for speed, Arb Holon often fetches directly.
        # Let's instantiate GoldOracle here if not passed, or better:
        # If we can't access EntryOracle, we make a local lightweight instance.
        if not hasattr(self, 'gold_oracle'):
             from .agent_oracle import GoldOracle
             self.gold_oracle = GoldOracle()
             
        spot_price = self.gold_oracle.fetch_spot_price()
        if spot_price <= 0: return None
        
        # Kraken PAXG/USD
        paxg_bid = float(kraken_ticker.get('bid', 0.0) or 0.0)
        paxg_ask = float(kraken_ticker.get('ask', 0.0) or 0.0)
        
        if paxg_bid <= 0 or paxg_ask <= 0: return None
        
        # Logic:
        # If Spot > PAXG Ask (Kraken Cheap) -> Buy PAXG
        # spread = (Spot - Ask) / Ask
        spread_long = (spot_price - paxg_ask) / paxg_ask
        
        # If Spot < PAXG Bid (Kraken Expensive) -> Sell PAXG
        # spread = (Bid - Spot) / Spot
        spread_short = (paxg_bid - spot_price) / spot_price
        
        threshold = 0.002 # 0.2% Net (Fees are ~0.26% taker, so need ~0.5% gross?)
        # Let's start with 0.5% Gross
        gross_threshold = 0.005
        
        if spread_long > gross_threshold:
             return {
                 'symbol': 'PAXG/USD',
                 'direction': 'BUY',
                 'confidence': 0.9,
                 'reason': f"GOLD_ORACLE_LONG (Spot ${spot_price:.1f} > PAXG ${paxg_ask:.1f} by {spread_long*100:.2f}%)",
                 'metadata': {'spot_gold': spot_price, 'spread': spread_long}
             }
             
        if spread_short > gross_threshold:
             return {
                 'symbol': 'PAXG/USD',
                 'direction': 'SELL',
                 'confidence': 0.9,
                 'reason': f"GOLD_ORACLE_SHORT (Spot ${spot_price:.1f} < PAXG ${paxg_bid:.1f} by {spread_short*100:.2f}%)",
                 'metadata': {'spot_gold': spot_price, 'spread': spread_short}
             }
             
        return None

    def mine_liquidity(self) -> List[Dict]:
        """
        The Silent Miner ⛏️
        Scans all monitored assets for 'Gold Nuggets' (High Value Arbitrage).
        Only returns signals that meet strict criteria to convert silence into action.
        
        UPGRADE: Now includes xStocks funding arbitrage opportunities.
        """
        nuggets = []

        if not hasattr(self, '_nugget_cooldowns'):
            self._nugget_cooldowns = {}

        current_time = time.time()

        # 1. Standard Arb & Funding (Crypto futures only)
        # xStocks (CRCLX, HOODX, SPYX, etc.) are disabled via ENABLE_XSTOCKS_ARB config flag
        # due to extreme/unpredictable funding rates that create unacceptable drawdown risk.
        _xstocks_enabled = getattr(config, 'ENABLE_XSTOCKS_ARB', False)
        _xstock_tickers  = {'SPYX', 'QQQX', 'NVDAX', 'AAPLX', 'GOOGLX', 'TSLAX', 'MSTRX', 'CRCLX', 'HOODX'}

        allowed_assets = set(config.ALLOWED_ASSETS)
        for symbol in list(self.funding_yields.keys()):
            # Skip xStocks unless explicitly enabled
            if not _xstocks_enabled and any(xs in symbol for xs in _xstock_tickers):
                continue

            # Allow crypto and (if enabled) xStocks symbols
            # xStocks may be in format 'SPYX/USDT' or 'SPYX/USD:USD'
            is_allowed = symbol in allowed_assets
            if not is_allowed:
                # Check if it's an xStock in Kraken format (e.g., 'SPYX/USD:USD')
                base_symbol = symbol.replace('/USD:USD', '/USDT')
                if base_symbol in allowed_assets:
                    is_allowed = True
            
            if not is_allowed:
                continue
                
            # Re-evaluate signal
            sig = self.get_active_signal(symbol, 0.0)
            if sig and sig['confidence'] >= 0.8:
                # Cooldown check to prevent spamming the same nugget every cycle
                last_time_info = self._nugget_cooldowns.get(symbol, {'time': 0, 'yield': 0.0})

                # Support old format where it was just a float timestamp
                if isinstance(last_time_info, (int, float)):
                    last_time = last_time_info
                    old_yield = 0.0
                else:
                    last_time = last_time_info.get('time', 0)
                    old_yield = last_time_info.get('yield', 0.0)

                current_yield = sig.get('gross_yield_8h', 0.0)
                time_passed = current_time - last_time

                # FIX 2026-03-04: xStocks now use the same 5-min cooldown as crypto.
                # Previously 2-min cooldown was causing xStocks to dominate the miner log
                # 2.5x more frequently than crypto signals, giving false impression of over-focus.
                is_xstock = sig.get('is_xstock', False) or any(xs in symbol for xs in ['SPYX', 'QQQX', 'NVDAX', 'AAPLX', 'GOOGLX', 'TSLAX', 'MSTRX', 'CRCLX', 'HOODX'])
                cooldown_threshold = 300   # 5 min for ALL assets (was 120 for xStocks)
                yield_surge_threshold = 4.5 if is_xstock else 0.45  # Keep higher surge bar for xStocks
                
                if time_passed < cooldown_threshold:
                    if current_yield <= old_yield + yield_surge_threshold:
                        continue
                    else:
                        print(f"[{self.name}] 🚀 COOLDOWN SHORT-CIRCUIT: {symbol} Yield surged from {old_yield:.3f}% to {current_yield:.3f}%")

                # FIX: Set cooldown IMMEDIATELY when valid signal is detected
                # This prevents repeated signaling even if downstream components veto
                self._nugget_cooldowns[symbol] = {'time': current_time, 'yield': current_yield}
                print(f"[{self.name}] ⏱️ COOLDOWN SET: {symbol} for {cooldown_threshold}s")

                # Normalize symbol to base format ('SPYX/USD:USD' -> 'SPYX/USDT')
                # This ensures the nuggets bypass TraderNexus' strict ALLOWED_ASSETS filter.
                sig['symbol'] = symbol.replace('/USD:USD', '/USDT')
                sig['is_xstock'] = is_xstock  # Tag for downstream slot cap
                nuggets.append(sig)

        # 1.5 Rust Phase 3 Spatial Arb Scanning
        if getattr(self, 'rust_available', False):
            arb_opps = self.phase3.scan_arbitrage()
            for opp in arb_opps:
                symbol = f"{opp.base_asset}/{opp.quote_asset}"
                if symbol not in allowed_assets:
                    continue
                
                # We want to place the order on Kraken (since we only execute there right now)
                # If Kraken is the buy exchange, we go LONG. If Kraken is the sell exchange, we go SHORT.
                if opp.buy_exchange == 'kraken':
                    direction = 'BUY'
                elif opp.sell_exchange == 'kraken':
                    direction = 'SELL'
                else:
                    continue
                
                nugget = {
                    'symbol': symbol,
                    'direction': direction,
                    'confidence': min(0.95, 0.6 + (opp.spread_pct / 10)),
                    'reason': f"SPATIAL_ARB_RUST ({opp.buy_exchange}->{opp.sell_exchange} {opp.spread_pct:.2f}%)",
                    'metadata': {
                        'strategy': 'ARBITRAGE',
                        'buy_exchange': opp.buy_exchange,
                        'sell_exchange': opp.sell_exchange,
                        'expected_profit_pct': opp.expected_profit_pct,
                        'max_quantity': opp.max_quantity
                    }
                }
                
                # Cooldown check
                last_time_info = self._nugget_cooldowns.get(symbol + "_rust", {'time': 0, 'yield': 0.0})
                last_time = last_time_info.get('time', 0) if isinstance(last_time_info, dict) else last_time_info
                
                if current_time - last_time >= 60: # 60s cooldown for Rust arb
                    self._nugget_cooldowns[symbol + "_rust"] = {'time': current_time, 'yield': opp.expected_profit_pct}
                    nuggets.append(nugget)
                    print(f"[{self.name}] 🦀 RUST ARB DETECTED: {symbol} {direction} (Profit {opp.expected_profit_pct:.3f}%)")

        # 2. Gold Oracle Injection
        # We need PAXG ticker. It might be in price_spreads or we fetch it?
        # Currently perform_sync fetches tickers. But we don't store raw tickers in self.
        # We need to access the observer or store last ticker.
        # Workaround: fetch fresh PAXG ticker here (Mining is less freq than sync)
        if self.kraken_observer:
             # Check PAXG
             if 'PAXG/USD' in allowed_assets or 'PAXG/USDT' in allowed_assets:
                 # Use fetch_tickers_batch for robustness (uses existing cache logic)
                 tickers = self.kraken_observer.fetch_tickers_batch(['PAXG/USD'])
                 paxg_ticker = tickers.get('PAXG/USD')
                 if paxg_ticker: # Safety check
                     gold_sig = self.check_gold_arb(paxg_ticker)
                     if gold_sig:
                          nuggets.append(gold_sig)
                          print(f"[{self.name}] 🌟 GOLD ORACLE SIGNAL (PAXG): {gold_sig['reason']}")

             # Check XAUT (Tether Gold) - User Request
             if 'XAUT/USD' in allowed_assets or 'XAUT/USDT' in allowed_assets:
                 # xaut_ticker = self.kraken_observer.fetch_ticker('XAUT/USD') # Or XAUT/USDT depending on map
                 # Use batch fetch
                 xaut_tickers = self.kraken_observer.fetch_tickers_batch(['XAUT/USD', 'XAUT/USDT'])
                 xaut_ticker = xaut_tickers.get('XAUT/USD') or xaut_tickers.get('XAUT/USDT')

                 if xaut_ticker:
                     xaut_sig = self.check_gold_arb(xaut_ticker)
                     if xaut_sig:
                          # Fix symbol in signal to match Kraken format
                          xaut_sig['symbol'] = 'XAUT/USDT'
                          xaut_sig['reason'] = xaut_sig['reason'].replace('GOLD_ORACLE', 'GOLD_ORACLE_XAUT')
                          nuggets.append(xaut_sig)
                          print(f"[{self.name}] 🌟 GOLD ORACLE SIGNAL (XAUT): {xaut_sig['reason']}")

        if nuggets:
            # Separate xStocks from crypto for reporting
            xstock_nuggets = [n for n in nuggets if n.get('is_xstock', False) or any(xs in n.get('symbol', '') for xs in ['SPYX', 'QQQX', 'NVDAX', 'AAPLX', 'GOOGLX', 'TSLAX', 'MSTRX', 'CRCLX', 'HOODX'])]
            crypto_nuggets = [n for n in nuggets if n not in xstock_nuggets]
            
            if xstock_nuggets:
                print(f"[{self.name}] ⛏️ MINING LIQUIDITY: Extracted {len(nuggets)} Nuggets! ({len(xstock_nuggets)} xStocks, {len(crypto_nuggets)} crypto)")
            else:
                print(f"[{self.name}] ⛏️ MINING LIQUIDITY: Extracted {len(nuggets)} Nuggets!")

        return nuggets
