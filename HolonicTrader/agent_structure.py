"""
CTKSStrategicHolon - The "Institution of One" + Orion CTKS Profit Navigator
(Structure Overrides Momentum → Market Path Alignment)

Objective:
- Map "Stanfield Levels" (SLS) using Higher Timeframe (HTF) Data.
- Enforce BORSOG Protocol (Buy On Red, Sell On Green).
- Override Impulse/Momentum signals if they conflict with Structure.
- [ORION] Determine Market Path (UP/DOWN/NEUTRAL) via structure + momentum alignment.
- [ORION] Apply intermarket intelligence (DXY, Yields) as macro confirmation.
- [ORION] Produce Orion market state assessment for trade filtering.

Methodology:
- Tier 1: Weekly/Daily Trend & Levels.
- Tier 2: 4H Market Structure.
- Tier 3: Execution (handled by Oracle/Executor).
- Tier 4: [ORION] Market Path + Intermarket Confirmation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from HolonicTrader.holon_core import Holon, Disposition
import config

class CTKSStrategicHolon(Holon):
    def __init__(self, name: str = "StructureBoss"):
        # High Autonomy (It sets the rules), High Integration (Must be obeyed)
        super().__init__(name=name, disposition=Disposition(autonomy=0.9, integration=0.8))
        self.sls_levels = {} # {symbol: {'support': [], 'resistance': []}}
        self.htf_bias = {}   # {symbol: 'BULLISH' | 'BEARISH' | 'RANGING'}
        self._ctx_cache = {} # {symbol: {'ts': float, 'ctx': dict, 'str': str}}
        self._cache_ttl = 300 # 5 minutes
        self._veto_cache = {} # {symbol: {'ts': float, 'reason': str, 'count': int}} - Compute waste fix
        self._veto_cache_ttl = 60 # 1 minute veto cache
        self._alert_cooldown = {} # {alert_type: last_ts} - Topology alert spam fix
        self._alert_cooldown_ttl = 30 # 30s between identical alerts

        # ── ORION CTKS Profit Navigator State ──
        self._orion_cache = {}     # {symbol: {'ts': float, 'state': dict}}
        self._orion_cache_ttl = 120  # 2 minutes (faster than structure, Orion is lightweight)
        self._intermarket_state = {}  # Cached DXY/Yield data from MacroOracle
        self._intermarket_ts = 0

    def receive_message(self, sender: Any, content: Any) -> None:
        """Process incoming messages (CTKS Protocol)."""
        pass # Currently pull-based architecture

    def get_structural_context(self, symbol: str, observer: Any) -> Dict[str, Any]:
        """
        Primary Output: Analyze HTF structure and return constraints.
        """
        # 1. Tier 1 Analysis (Daily/Weekly)
        # We need Observer to fetch HTF data.
        if not observer:
            return {'structure_mode': 'UNKNOWN', 'sls_zone': 'NONE', 'bias': 'NEUTRAL'}

        import time
        now = time.time()
        # Cache Check
        cache_entry = self._ctx_cache.get(symbol)
        if cache_entry and (now - cache_entry['ts']) < self._cache_ttl:
            # Re-print log for dashboard context if needed, or suppress
            if 'str' in cache_entry and cache_entry['str']:
                print(cache_entry['str'])
            return cache_entry['ctx']

        # COMPUTE WASTE FIX: Check veto cache to avoid re-analyzing recently vetoed symbols
        veto_entry = self._veto_cache.get(symbol)
        if veto_entry and (now - veto_entry['ts']) < self._veto_cache_ttl:
            veto_entry['count'] += 1
            # Only log every 5th veto to reduce spam
            if veto_entry['count'] % 5 == 0:
                print(f"[{self.name}] ⚠️ {symbol} still vetoed ({veto_entry['count']}x): {veto_entry['reason']}")
            return cache_entry['ctx'] if cache_entry else {'structure_mode': 'VETOED', 'sls_zone': 'NONE'}

        try:
            # Fetch Daily Data (Tier 2/1 Hybrid for Crypto)
            df_daily = observer.fetch_market_data(timeframe='1d', limit=60, symbol=symbol)
            if df_daily.empty:
                return {'structure_mode': 'NO_DATA'}
            
            # 2. Map Stanfield Levels (SLS)
            # Simplified: Use Swing Highs/Lows and Volume Nodes
            supports, resistances = self._map_sls_levels(df_daily)
            self.sls_levels[symbol] = {'support': supports, 'resistance': resistances}
            
            current_price = df_daily['close'].iloc[-1]
            
            # 3. Determine Bias (Higher Highs / Lower Lows)
            bias = self._determine_bias(df_daily)
            self.htf_bias[symbol] = bias
            
            # 4. Calculate Pivot Points (New)
            pivots = self._calculate_pivots(df_daily)
            
            # 5. Check BORSOG Alignment
            # Are we at Support (Buy Zone) or Resistance (Sell Zone)?
            
            # Find nearest levels
            nearest_sup = max([s for s in supports if s < current_price], default=0)
            nearest_res = min([r for r in resistances if r > current_price], default=float('inf'))
            
            dist_to_sup = (current_price - nearest_sup) / current_price if nearest_sup > 0 else 1.0
            dist_to_res = (nearest_res - current_price) / current_price if nearest_res < float('inf') else 1.0
            
            # SLS Zone Definition (e.g. within 1.5% of level)
            zone_threshold = 0.015 
            
            sls_zone = 'NEUTRAL'
            if dist_to_sup < zone_threshold:
                sls_zone = 'SUPPORT'
            elif dist_to_res < zone_threshold:
                sls_zone = 'RESISTANCE'
                
            dist_to_sup_pct = (current_price - nearest_sup) / nearest_sup if nearest_sup > 0 else 0
            
            # Fallback Logic (Prevent 0/Inf)
            if nearest_sup == 0: nearest_sup = current_price * 0.8 # Broad floor
            if nearest_res == float('inf'): nearest_res = current_price * 1.2 # Broad ceiling
            
            # Context Object
            ctx = {
                'structure_mode': 'Valid',
                'macro_trend': bias,
                'sls_zone': sls_zone,
                'nearest_support': nearest_sup,
                'nearest_resistance': nearest_res,
                'dist_to_sup_pct': -abs(dist_to_sup_pct), # Always negative/zero
                'dist_to_res_pct': dist_to_res,
                'pivots': pivots,
                'bias': bias,  # Orion alias for macro_trend
            }

            # FIX 2026-03-23: Report macro trend to Oracle for GMB calculation
            # This ensures GMB reflects actual market structure, not Kalman noise
            # Access oracle through governor (parent holon)
            governor = getattr(self, 'governor', None)
            if governor and hasattr(governor, 'sub_holons'):
                oracle = governor.sub_holons.get('oracle')
                if oracle and hasattr(oracle, 'update_structure_bias'):
                    oracle.update_structure_bias(symbol, bias)

            # Formatted String (Prevent overflow in logs)
            res_str = f"{nearest_res:.8f}" if nearest_res < 1000 else f"{nearest_res:.2f}"
            sup_str = f"{nearest_sup:.8f}" if nearest_sup < 1000 else f"{nearest_sup:.2f}"
            
            pivot_str = f"| Piv: {pivots.get('P',0):.2f}" if pivots else ""
            log_str = f"[{self.name}] 🏛️ {symbol} Structure: {bias} | Zone: {sls_zone} | Sup: {sup_str}, Res: {res_str} {pivot_str}"
            print(log_str)
            
            self._ctx_cache[symbol] = {'ts': now, 'ctx': ctx, 'str': log_str}
            return ctx

        except Exception as e:
            print(f"[{self.name}] Error analyzing structure: {e}")
            return {}

    def record_veto(self, symbol: str, reason: str):
        """
        COMPUTE WASTE FIX: Record a veto to prevent repeated analysis of the same rejection.
        Call this when Governor/Trader vetoes a signal.
        """
        import time
        now = time.time()
        veto_entry = self._veto_cache.get(symbol)
        if veto_entry:
            veto_entry['ts'] = now
            veto_entry['reason'] = reason
            veto_entry['count'] += 1
        else:
            self._veto_cache[symbol] = {'ts': now, 'reason': reason, 'count': 1}
        # Clear cache entry after TTL expires
        self._veto_cache = {k: v for k, v in self._veto_cache.items() if (now - v['ts']) < self._veto_cache_ttl}

    def should_suppress_alert(self, alert_type: str) -> bool:
        """
        COMPUTE WASTE FIX: Check if an alert should be suppressed due to cooldown.
        Returns True if alert should be suppressed.
        """
        import time
        now = time.time()
        last_ts = self._alert_cooldown.get(alert_type, 0)
        if (now - last_ts) < self._alert_cooldown_ttl:
            return True
        self._alert_cooldown[alert_type] = now
        return False

    def get_structure(self, symbol: str, observer: Any = None) -> tuple:
        """
        Legacy compatibility method: Returns structure as a tuple.
        Returns: (bias, zone, support, resistance, pivot)
        """
        # Use get_structural_context internally and convert to tuple format
        ctx = self.get_structural_context(symbol, observer)

        bias = ctx.get('macro_trend', 'NEUTRAL')
        zone = ctx.get('sls_zone', 'NEUTRAL')
        support = ctx.get('nearest_support', 0.0)
        resistance = ctx.get('nearest_resistance', 0.0)
        pivot = ctx.get('pivots', {}).get('P', 0.0)

        return (bias, zone, support, resistance, pivot)

    def _map_sls_levels(self, df: pd.DataFrame) -> (List[float], List[float]):
        """
        Identify Significant Swing Points (Fractals).
        """
        highs = df['high']
        lows = df['low']
        
        # Simple Fractal Logic (5-candle)
        # High is higher than 2 previous and 2 next
        swing_highs = []
        swing_lows = []
        
        # We scan up to -3 to allow for '2 next' confirmation
        # Ideally we use a rolling window
        for i in range(2, len(df)-2):
            if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i-2] and \
               highs.iloc[i] > highs.iloc[i+1] and highs.iloc[i] > highs.iloc[i+2]:
                swing_highs.append(highs.iloc[i])
                
            if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i-2] and \
               lows.iloc[i] < lows.iloc[i+1] and lows.iloc[i] < lows.iloc[i+2]:
                swing_lows.append(lows.iloc[i])
                
        # Return all discovered swing points in the lookback window
        # Filtering by 'Top 5 Highest' via sorted() breaks when price drops below them,
        # leading to persistent "NEUTRAL" zones where the bot cannot find local support/resistance.
        return swing_lows, swing_highs

    def _determine_bias(self, df: pd.DataFrame) -> str:
        # Simple SMA Check + Structure
        sma50 = df['close'].rolling(50).mean().iloc[-1]
        price = df['close'].iloc[-1]
        
        # Check Lower Highs / Higher Lows?
        # For robustness, use SMA alignment
        if price > sma50: return 'BULLISH'
        else: return 'BEARISH'

    # ══════════════════════════════════════════════════════════════════
    # ORION CTKS PROFIT NAVIGATOR — Market Path Alignment Engine
    # ══════════════════════════════════════════════════════════════════

    def get_orion_state(self, symbol: str, observer: Any, macro_oracle: Any = None,
                        rsi: float = 50.0, entropy: float = 1.5) -> Dict[str, Any]:
        """
        Orion Market State Assessment.
        Combines: Structure (CTKS) + Momentum (RSI/Bias) + Intermarket (DXY/Yields/VIX).

        Returns:
            {
                'path': 'UP' | 'DOWN' | 'NEUTRAL',
                'structure': 'SUPPORT' | 'RESISTANCE' | 'NEUTRAL',
                'momentum': 'BULLISH' | 'BEARISH' | 'NEUTRAL',
                'intermarket': 'RISK_ON' | 'RISK_OFF' | 'NEUTRAL',
                'confidence': 'HIGH' | 'MEDIUM' | 'LOW',
                'alignment_score': int (0-3),
                'conviction_modifier': float (0.0-1.0),
                'dxy_signal': str,
                'yield_signal': str,
                'path_strictness': float,
            }
        """
        import time
        if not getattr(config, 'ORION_ENABLED', False):
            return self._orion_default()

        now = time.time()
        cache = self._orion_cache.get(symbol)
        if cache and (now - cache['ts']) < self._orion_cache_ttl:
            return cache['state']

        # 1. Get CTKS structural context (uses its own cache)
        ctx = self.get_structural_context(symbol, observer)
        sls_zone = ctx.get('sls_zone', 'NEUTRAL')
        macro_trend = ctx.get('macro_trend', 'NEUTRAL')

        # 2. Determine momentum alignment
        momentum = self._orion_momentum(rsi, macro_trend)

        # 3. Intermarket intelligence (DXY, Yields, VIX)
        intermarket = self._orion_intermarket(macro_oracle)

        # 4. Determine market path
        path, alignment_score = self._orion_path(macro_trend, momentum, intermarket, entropy)

        # 5. Confidence assessment
        confidence = 'HIGH' if alignment_score >= 3 else ('MEDIUM' if alignment_score >= 2 else 'LOW')

        # 6. Conviction modifier (how much to trust/penalize trades)
        conviction_mod = self._orion_conviction_modifier(path, sls_zone, alignment_score)

        state = {
            'path': path,
            'structure': sls_zone,
            'momentum': momentum,
            'intermarket': intermarket.get('signal', 'NEUTRAL'),
            'confidence': confidence,
            'alignment_score': alignment_score,
            'conviction_modifier': conviction_mod,
            'dxy_signal': intermarket.get('dxy_signal', 'N/A'),
            'yield_signal': intermarket.get('yield_signal', 'N/A'),
            'path_strictness': getattr(config, 'ORION_PATH_STRICTNESS', 0.7),
        }

        # Log
        print(f"[Orion] 🧭 {symbol} Path: {path} | Struct: {sls_zone} | Mom: {momentum} "
              f"| Macro: {intermarket.get('signal', '?')} | Align: {alignment_score}/3 | Conf: {confidence}")

        self._orion_cache[symbol] = {'ts': now, 'state': state}
        return state

    def orion_filter_signal(self, symbol: str, direction: str, conviction: float,
                            orion_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orion Trade Filter — approve/reject/modify a trade signal based on market path.

        Returns:
            {
                'approved': bool,
                'adjusted_conviction': float,
                'reason': str,
                'action': 'APPROVED' | 'REJECTED' | 'REDUCED',
            }
        """
        if not getattr(config, 'ORION_ENABLED', False):
            return {'approved': True, 'adjusted_conviction': conviction,
                    'reason': 'Orion disabled', 'action': 'APPROVED'}

        path = orion_state.get('path', 'NEUTRAL')
        strictness = orion_state.get('path_strictness', 0.7)
        alignment = orion_state.get('alignment_score', 0)
        min_alignment = getattr(config, 'ORION_MIN_ALIGNMENT_SCORE', 2)
        conv_mod = orion_state.get('conviction_modifier', 1.0)

        # ── PATH ALIGNMENT CHECK ──
        # Orion's core rule: trade WITH the path, not against it
        if direction == 'BUY' and path == 'DOWN':
            if strictness >= 1.0:
                return {'approved': False, 'adjusted_conviction': 0.0,
                        'reason': f'Orion VETO: BUY against DOWN path (strictness={strictness})',
                        'action': 'REJECTED'}
            else:
                penalty = strictness  # 0.7 → 70% conviction reduction
                adj = conviction * (1.0 - penalty)
                return {'approved': adj >= 0.20, 'adjusted_conviction': adj,
                        'reason': f'Orion penalty: BUY in DOWN path (-{penalty*100:.0f}%)',
                        'action': 'REDUCED' if adj >= 0.20 else 'REJECTED'}

        if direction == 'SELL' and path == 'UP':
            if strictness >= 1.0:
                return {'approved': False, 'adjusted_conviction': 0.0,
                        'reason': f'Orion VETO: SELL against UP path (strictness={strictness})',
                        'action': 'REJECTED'}
            else:
                penalty = strictness
                adj = conviction * (1.0 - penalty)
                return {'approved': adj >= 0.20, 'adjusted_conviction': adj,
                        'reason': f'Orion penalty: SELL in UP path (-{penalty*100:.0f}%)',
                        'action': 'REDUCED' if adj >= 0.20 else 'REJECTED'}

        # ── ALIGNMENT SCORE CHECK ──
        if alignment < min_alignment:
            penalty = 0.3  # 30% conviction reduction for low alignment
            adj = conviction * (1.0 - penalty)
            return {'approved': adj >= 0.20, 'adjusted_conviction': adj,
                    'reason': f'Orion low alignment ({alignment}/{min_alignment})',
                    'action': 'REDUCED' if adj >= 0.20 else 'REJECTED'}

        # ── ALIGNED: Apply conviction modifier (can boost or maintain) ──
        adj = min(1.0, conviction * conv_mod)
        return {'approved': True, 'adjusted_conviction': adj,
                'reason': f'Orion aligned: {direction} with {path} path (score {alignment}/3)',
                'action': 'APPROVED'}

    def _orion_default(self) -> Dict[str, Any]:
        """Default Orion state when disabled."""
        return {
            'path': 'NEUTRAL', 'structure': 'NEUTRAL', 'momentum': 'NEUTRAL',
            'intermarket': 'NEUTRAL', 'confidence': 'LOW', 'alignment_score': 0,
            'conviction_modifier': 1.0, 'dxy_signal': 'N/A', 'yield_signal': 'N/A',
            'path_strictness': 0.0,
        }

    def _orion_momentum(self, rsi: float, macro_trend: str) -> str:
        """
        Determine momentum direction from RSI + macro trend.
        Momentum must CONFIRM structure for Orion alignment.
        """
        # RSI zones
        rsi_bullish = rsi > 55
        rsi_bearish = rsi < 45
        # Macro trend from CTKS _determine_bias (SMA50)
        trend_bullish = macro_trend == 'BULLISH'
        trend_bearish = macro_trend == 'BEARISH'

        if rsi_bullish and trend_bullish:
            return 'BULLISH'
        elif rsi_bearish and trend_bearish:
            return 'BEARISH'
        elif rsi_bullish or trend_bullish:
            return 'BULLISH'  # Lean bullish if either confirms
        elif rsi_bearish or trend_bearish:
            return 'BEARISH'
        return 'NEUTRAL'

    def _orion_intermarket(self, macro_oracle: Any = None) -> Dict[str, Any]:
        """
        Read intermarket intelligence from MacroOracle (DXY, Yields, VIX).
        Returns signal classification and component signals.
        """
        import time
        now = time.time()

        # Use cached intermarket state if fresh
        if self._intermarket_state and (now - self._intermarket_ts) < 120:
            return self._intermarket_state

        result = {'signal': 'NEUTRAL', 'dxy_signal': 'N/A', 'yield_signal': 'N/A',
                  'vix_signal': 'N/A', 'score': 0.0}

        if not macro_oracle:
            self._intermarket_state = result
            self._intermarket_ts = now
            return result

        if not getattr(config, 'ORION_INTERMARKET_FILTER', False):
            self._intermarket_state = result
            self._intermarket_ts = now
            return result

        macro_state = macro_oracle.fetch_macro_context() if hasattr(macro_oracle, 'fetch_macro_context') else {}
        details = macro_state.get('details', {})
        score = 0.0

        # ── DXY (Dollar Index) ──
        # DXY ↑ → risk assets ↓ (crypto, stocks)
        dxy_change = details.get('DX-Y.NYB', 0.0)
        dxy_threshold = getattr(config, 'ORION_DXY_RISK_OFF_PCT', 0.005)
        if isinstance(dxy_change, (int, float)):
            if dxy_change > dxy_threshold:
                result['dxy_signal'] = 'RISK_OFF'
                score -= 1.5  # DXY rising strongly = pressure on crypto
            elif dxy_change < -dxy_threshold:
                result['dxy_signal'] = 'RISK_ON'
                score += 1.0  # DXY falling = tailwind for crypto
            else:
                result['dxy_signal'] = 'NEUTRAL'

        # ── Yields (US10Y - US02Y spread) ──
        us10y = details.get('^TNX', None)
        us2y = details.get('^IRX', None)
        inversion_threshold = getattr(config, 'ORION_YIELD_INVERSION_THRESHOLD', -0.10)
        if isinstance(us10y, (int, float)) and isinstance(us2y, (int, float)):
            # ^TNX is 10Y yield level, ^IRX is 13-week T-bill rate (proxy for short end)
            # When yield change is negative → risk assets may benefit
            if us10y > 0.003:  # 10Y yield rising significantly
                result['yield_signal'] = 'PRESSURE'
                score -= 0.5
            elif us10y < -0.003:
                result['yield_signal'] = 'EASING'
                score += 0.5
            else:
                result['yield_signal'] = 'STABLE'
        elif isinstance(us10y, (int, float)):
            if us10y > 0.003:
                result['yield_signal'] = 'PRESSURE'
                score -= 0.5
            else:
                result['yield_signal'] = 'STABLE'

        # ── VIX (already in MacroOracle, leverage it) ──
        vix = macro_state.get('vix', 0.0)
        if vix >= getattr(config, 'VIX_PANIC_THRESHOLD', 30.0):
            result['vix_signal'] = 'PANIC'
            score -= 2.0
        elif vix >= getattr(config, 'VIX_FEAR_THRESHOLD', 20.0):
            result['vix_signal'] = 'FEAR'
            score -= 0.5
        elif vix < getattr(config, 'VIX_CALM_THRESHOLD', 15.0):
            result['vix_signal'] = 'CALM'
            score += 0.5
        else:
            result['vix_signal'] = 'NEUTRAL'

        # Classify overall intermarket signal
        if score >= 1.0:
            result['signal'] = 'RISK_ON'
        elif score <= -1.0:
            result['signal'] = 'RISK_OFF'
        else:
            result['signal'] = 'NEUTRAL'

        result['score'] = score
        self._intermarket_state = result
        self._intermarket_ts = now
        return result

    def _orion_path(self, macro_trend: str, momentum: str,
                    intermarket: Dict[str, Any], entropy: float) -> tuple:
        """
        Determine Market Path: UP, DOWN, or NEUTRAL.
        Score = structure_vote + momentum_vote + intermarket_vote.
        Also considers entropy (chaotic = penalize confidence).

        Returns: (path: str, alignment_score: int)
        """
        votes_up = 0
        votes_down = 0

        # Vote 1: Structure (from CTKS bias)
        if macro_trend == 'BULLISH':
            votes_up += 1
        elif macro_trend == 'BEARISH':
            votes_down += 1

        # Vote 2: Momentum
        if momentum == 'BULLISH':
            votes_up += 1
        elif momentum == 'BEARISH':
            votes_down += 1

        # Vote 3: Intermarket
        inter_signal = intermarket.get('signal', 'NEUTRAL')
        if inter_signal == 'RISK_ON':
            votes_up += 1
        elif inter_signal == 'RISK_OFF':
            votes_down += 1

        # High entropy (chaotic market) → force NEUTRAL if votes aren't unanimous
        entropy_ceiling = getattr(config, 'ENTRY_MAX_ENTROPY', 2.1)
        if entropy > entropy_ceiling and abs(votes_up - votes_down) < 2:
            return ('NEUTRAL', 0)

        # Determine path
        if votes_up >= 2 and votes_down == 0:
            return ('UP', votes_up)
        elif votes_down >= 2 and votes_up == 0:
            return ('DOWN', votes_down)
        elif votes_up > votes_down:
            return ('UP', votes_up - votes_down)
        elif votes_down > votes_up:
            return ('DOWN', votes_down - votes_up)
        else:
            return ('NEUTRAL', 0)

    def _orion_conviction_modifier(self, path: str, sls_zone: str,
                                   alignment_score: int) -> float:
        """
        Calculate conviction multiplier based on Orion alignment.
        Perfect alignment (3/3) at structural level → boost conviction.
        Misalignment → penalize.
        """
        mod = 1.0

        # Alignment boost/penalty
        if alignment_score >= 3:
            mod = 1.15  # 15% boost for full alignment
        elif alignment_score == 2:
            mod = 1.0   # Neutral — acceptable
        elif alignment_score == 1:
            mod = 0.8   # 20% penalty
        else:
            mod = 0.6   # 40% penalty for zero alignment

        # Structure zone bonus
        if path == 'UP' and sls_zone == 'SUPPORT':
            mod *= 1.1  # At support in uptrend = ideal BUY
        elif path == 'DOWN' and sls_zone == 'RESISTANCE':
            mod *= 1.1  # At resistance in downtrend = ideal SELL

        return min(1.25, mod)  # Cap at 25% boost

    def _calculate_pivots(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Standard, Fibonacci, and Camarilla Pivot Points.
        Uses the PREVIOUS completed day (iloc[-2]).
        """
        if len(df) < 2: return {}
        
        # Previous Day Data
        prev = df.iloc[-2]
        H = prev['high']
        L = prev['low']
        C = prev['close']
        
        # 1. Standard (Floor) Pivots
        P = (H + L + C) / 3
        R1 = (2 * P) - L
        S1 = (2 * P) - H
        R2 = P + (H - L)
        S2 = P - (H - L)
        R3 = H + 2 * (P - L)
        S3 = L - 2 * (H - P)
        
        # 2. Fibonacci Pivots
        range_hl = H - L
        Fib_R1 = P + (0.382 * range_hl)
        Fib_S1 = P - (0.382 * range_hl)
        Fib_R2 = P + (0.618 * range_hl)
        Fib_S2 = P - (0.618 * range_hl)
        Fib_R3 = P + (1.000 * range_hl)
        Fib_S3 = P - (1.000 * range_hl)
        
        # 3. Camarilla Pivots (Mean Reversion)
        Cam_R3 = C + (range_hl * 1.1 / 4)
        Cam_S3 = C - (range_hl * 1.1 / 4)
        Cam_R4 = C + (range_hl * 1.1 / 2) # Breakout
        Cam_S4 = C - (range_hl * 1.1 / 2) # Breakdown

        return {
            'P': P,
            'R1': R1, 'S1': S1,
            'R2': R2, 'S2': S2,
            'R3': R3, 'S3': S3,
            'Fib_R1': Fib_R1, 'Fib_S1': Fib_S1,
            'Fib_R2': Fib_R2, 'Fib_S2': Fib_S2,
            'Fib_R3': Fib_R3, 'Fib_S3': Fib_S3,
            'Cam_R3': Cam_R3, 'Cam_S3': Cam_S3,
            'Cam_R4': Cam_R4, 'Cam_S4': Cam_S4
        }
