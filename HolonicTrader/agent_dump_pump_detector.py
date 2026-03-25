"""
DumpPumpDetectorHolon - Time-Window Whale Manipulation Detector (Phase 2026-03-05)

Detects institutional dump-and-pump algo patterns at known manipulation windows:
  - Gold (XAUT/PAXG): 8:00 AM ET  — COMEX open + paper-gold flooding
  - Bitcoin (BTC/XBT): 10:00 AM ET — US macro session cross-contamination

Detection Methods:
  1. Time-Window Guard    → Only active during configurable pre/post windows
  2. Ask-Wall Detection   → Large sell walls ≥ 0.5% of 24h volume near mid price
  3. Dump Velocity        → Price drop % + volume spike on latest 1m candle
  4. CVD Divergence       → Buying absorption vs falling price = exhaustion signal
  5. VIX Amplifier        → VIX > 25 boosts confidence; VIX > 30 = panic tag

Output Phases:
  DUMP_IN_PROGRESS  → SELL bias (whale dump active — avoid/fade longs)
  DUMP_EXHAUSTED    → BUY  bias (absorption complete — contrarian long)
  RE_ACCUMULATION   → BUY  bias (whale reloading after stop-run)
"""

import time
import logging
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo  # stdlib Python 3.9+
from datetime import datetime, timezone

import pandas as pd

from HolonicTrader.holon_core import Holon, Disposition
import config

logger = logging.getLogger("DumpPumpDetector")

# Time zone constant
ET_TZ = ZoneInfo("America/New_York")

# Gold symbols monitored for 8am ET window
GOLD_SYMBOLS = {'XAUT/USDT', 'XAUT/USD:USD', 'PAXG/USDT', 'PAXG/USD:USD', 'PAXG/BTC'}

# Bitcoin symbols monitored for 10am ET window
BTC_SYMBOLS = {'BTC/USDT', 'BTC/USD:USD', 'XBT/USD', 'XBT/USD:USD'}


class DumpPumpDetectorHolon(Holon):
    """
    Time-window whale manipulation detector for gold (8am ET) and BTC (10am ET).
    Produces structured event dicts consumed by SignalProvider.
    """

    def __init__(self, name: str = "DumpPumpDetector"):
        super().__init__(name=name, disposition=Disposition(autonomy=0.8, integration=0.9))

        # Config-driven thresholds
        self.gold_et_hour = getattr(config, 'DUMP_GOLD_WINDOW_ET_HOUR', 8)
        self.btc_et_hour = getattr(config, 'DUMP_BTC_WINDOW_ET_HOUR', 10)
        self.window_before_min = getattr(config, 'DUMP_WINDOW_MINUTES_BEFORE', 10)
        self.window_after_min = getattr(config, 'DUMP_WINDOW_MINUTES_AFTER', 25)
        self.dump_velocity_pct = getattr(config, 'DUMP_VELOCITY_PCT', 0.0035)
        self.dump_rvol_threshold = getattr(config, 'DUMP_RVOL_THRESHOLD', 2.5)
        self.ask_wall_dist = getattr(config, 'DUMP_ASK_WALL_DIST', 0.005)
        self.exhaustion_cvd_ratio = getattr(config, 'DUMP_EXHAUSTION_CVD_RATIO', 0.60)

        # VIX thresholds
        self.vix_fear = getattr(config, 'VIX_FEAR_THRESHOLD', 20.0)
        self.vix_panic = getattr(config, 'VIX_PANIC_THRESHOLD', 30.0)
        self.vix_calm = getattr(config, 'VIX_CALM_THRESHOLD', 15.0)

        # Internal state
        self._last_events: Dict[str, Dict] = {}  # {symbol: last_event}
        self._event_cooldown: Dict[str, float] = {}  # {symbol: expiry_ts}
        self.COOLDOWN_SEC = 180  # 3-min cooldown per symbol

    # ──────────────────────────────────────────────────────────────
    # TIME WINDOW
    # ──────────────────────────────────────────────────────────────

    def _get_et_now(self) -> datetime:
        """Return current time in US/Eastern timezone."""
        return datetime.now(tz=ET_TZ)

    def is_in_window(self, symbol: str) -> bool:
        """
        Returns True if the current time falls within the monitoring window
        for the given symbol's asset class.
        """
        if not getattr(config, 'DUMP_PUMP_ENABLED', True):
            return False

        et_now = self._get_et_now()
        et_minutes = et_now.hour * 60 + et_now.minute

        # Determine target window
        if symbol in GOLD_SYMBOLS:
            target_minutes = self.gold_et_hour * 60
            asset_label = "GOLD"
        elif symbol in BTC_SYMBOLS:
            target_minutes = self.btc_et_hour * 60
            asset_label = "BTC"
        else:
            return False

        window_start = target_minutes - self.window_before_min
        window_end = target_minutes + self.window_after_min

        in_window = window_start <= et_minutes <= window_end

        if in_window:
            et_str = et_now.strftime('%H:%M')
            logger.info(
                f"[{self.name}] 🕐 {asset_label} DUMP WINDOW ACTIVE "
                f"({window_start//60:02d}:{window_start%60:02d}→"
                f"{window_end//60:02d}:{window_end%60:02d} ET) | Now: {et_str} ET"
            )

        return in_window

    # ──────────────────────────────────────────────────────────────
    # DETECTION METHODS
    # ──────────────────────────────────────────────────────────────

    def detect_ask_wall(self, book_data: Dict, daily_vol_usd: float = 0.0) -> Optional[Dict]:
        """
        Detect a large sell (ask) wall near the current mid price.
        Mirrors WhaleHolon.check_bid_wall() logic but on the ask side.

        Returns wall dict or None.
        """
        if not book_data or 'asks' not in book_data:
            return None

        asks = book_data.get('asks', [])
        if not asks:
            return None

        best_ask = asks[0][0]

        # Dynamic threshold: 0.5% of 24h volume, floor $50k
        if daily_vol_usd > 0:
            wall_threshold_usd = max(50_000.0, daily_vol_usd * 0.005)
        else:
            wall_threshold_usd = 500_000.0

        for price, vol in asks[:15]:
            notional = price * vol
            if notional >= wall_threshold_usd:
                # Distance from best ask (must be close = within ask_wall_dist %)
                dist = (price - best_ask) / best_ask
                if dist <= self.ask_wall_dist:
                    return {
                        'type': 'ASK_WALL',
                        'price': price,
                        'vol': vol,
                        'notional': notional,
                        'distance': dist,
                        'threshold_used': wall_threshold_usd,
                    }

        return None

    def detect_dump_velocity(self, df: pd.DataFrame) -> bool:
        """
        Returns True if the latest candle shows dump velocity:
          - Close-to-close drop > dump_velocity_pct
          - Volume > dump_rvol_threshold × rolling 20-bar avg

        Expects a DataFrame with 'close' and 'volume' columns (1m candles preferred).
        """
        if df is None or len(df) < 5:
            return False

        try:
            last_close = float(df['close'].iloc[-1])
            prev_close = float(df['close'].iloc[-2])

            if prev_close <= 0:
                return False

            price_change = (last_close - prev_close) / prev_close  # negative = dump

            # Volume spike check
            recent_vol = float(df['volume'].iloc[-1])
            avg_vol = float(df['volume'].iloc[-21:-1].mean()) if len(df) >= 21 else float(df['volume'].mean())

            if avg_vol <= 0:
                return False

            rvol = recent_vol / avg_vol

            is_dump = price_change <= -self.dump_velocity_pct
            is_spike = rvol >= self.dump_rvol_threshold

            if is_dump and is_spike:
                logger.info(
                    f"[{self.name}] ⚡ DUMP VELOCITY: "
                    f"Δprice={price_change*100:.3f}% | RVOL={rvol:.2f}×"
                )
                return True

        except Exception as e:
            logger.debug(f"[{self.name}] dump_velocity error: {e}")

        return False

    def detect_exhaustion(self, cvd_buy_ratio: float, price_change_pct: float) -> bool:
        """
        Returns True if dump exhaustion pattern detected:
          - Price is falling (price_change_pct < -0.2%)
          - CVD buy ratio is high (> exhaustion_cvd_ratio)
          → Whales absorbed the dump; reversal likely

        Args:
            cvd_buy_ratio: Fraction of buy volume in CVD (0.0–1.0)
            price_change_pct: Recent price change as fraction (negative = down)
        """
        price_falling = price_change_pct < -0.002  # -0.2%
        absorption = cvd_buy_ratio > self.exhaustion_cvd_ratio

        if price_falling and absorption:
            logger.info(
                f"[{self.name}] 🔄 DUMP EXHAUSTION: "
                f"CVD buy={cvd_buy_ratio:.1%} while price={price_change_pct*100:.3f}%"
            )
            return True
        return False

    # ──────────────────────────────────────────────────────────────
    # MAIN ANALYSIS ENTRY POINT
    # ──────────────────────────────────────────────────────────────

    def analyze(
        self,
        symbol: str,
        book_data: Optional[Dict] = None,
        df: Optional[pd.DataFrame] = None,
        cvd_data: Optional[Dict] = None,
        daily_vol_usd: float = 0.0,
        vix_level: float = 0.0,
    ) -> Optional[Dict]:
        """
        Main analysis. Only fires within the configured time window for each symbol.

        Returns event dict if a pattern is detected, else None.

        Event dict structure:
          {
            'event': 'WHALE_DUMP_GOLD' | 'WHALE_DUMP_BTC' | 'PUMP_UNWIND_GOLD' | 'PUMP_UNWIND_BTC',
            'phase': 'DUMP_IN_PROGRESS' | 'DUMP_EXHAUSTED' | 'RE_ACCUMULATION',
            'direction': 'SELL' | 'BUY',
            'confidence': float (0.0–1.0),
            'reason': str,
            'vix_level': float,
            'vix_regime': 'CALM' | 'NORMAL' | 'FEAR' | 'PANIC',
            'symbol': str,
            'timestamp': float,
          }
        """
        # 1. Gate: enabled?
        if not getattr(config, 'DUMP_PUMP_ENABLED', True):
            return None

        # 2. Gate: time window
        if not self.is_in_window(symbol):
            return None

        # 3. Gate: cooldown
        now = time.time()
        if self._event_cooldown.get(symbol, 0) > now:
            return None

        # 4. Determine asset class label
        if symbol in GOLD_SYMBOLS:
            asset_class = 'GOLD'
        elif symbol in BTC_SYMBOLS:
            asset_class = 'BTC'
        else:
            return None

        # 5. VIX regime label + confidence modifier
        vix_regime, vix_boost = self._get_vix_regime(vix_level)
        vix_tag = f"VIX={vix_level:.1f} [{vix_regime}]" if vix_level > 0 else ""

        # 6. Run detectors
        event = None

        # --- Ask wall detection (dump in progress indicator)
        ask_wall = self.detect_ask_wall(book_data, daily_vol_usd) if book_data else None

        # --- Dump velocity (1m candle spike)
        velocity_dump = self.detect_dump_velocity(df) if df is not None else False

        # --- CVD divergence (exhaustion)
        cvd_buy_ratio = 0.5
        price_change = 0.0
        if cvd_data:
            cvd_buy_ratio = float(cvd_data.get('buy_ratio', 0.5))
            if df is not None and len(df) >= 2:
                p_now = float(df['close'].iloc[-1])
                p_prev = float(df['close'].iloc[-2])
                price_change = (p_now - p_prev) / p_prev if p_prev > 0 else 0.0

        exhausted = self.detect_exhaustion(cvd_buy_ratio, price_change)

        # 7. Classify phase
        if exhausted:
            # Absorption detected → contrarian BUY
            confidence = min(0.92, 0.60 + (cvd_buy_ratio - self.exhaustion_cvd_ratio) * 2.0 + vix_boost)
            phase = 'DUMP_EXHAUSTED'
            direction = 'BUY'
            reason = (
                f"🔄 WHALE_DUMP_EXHAUSTION on {symbol}: "
                f"CVD buy {cvd_buy_ratio:.0%} while price {price_change*100:.2f}% "
                f"| {vix_tag}"
            )
            event_type = f"PUMP_UNWIND_{asset_class}"

        elif velocity_dump or ask_wall:
            # Dump in progress → SELL / warning
            base_conf = 0.55
            if velocity_dump:
                base_conf += 0.15
            if ask_wall:
                base_conf += 0.15
                wall_info = f" | Ask Wall ${ask_wall['notional']/1000:.0f}k @ {ask_wall['price']:.2f}"
            else:
                wall_info = ""
            confidence = min(0.90, base_conf + vix_boost)
            phase = 'DUMP_IN_PROGRESS'
            direction = 'SELL'
            reason = (
                f"⚠️ WHALE_DUMP_ACTIVE on {symbol}: "
                f"{'Velocity dump ' if velocity_dump else ''}"
                f"{wall_info} "
                f"| {vix_tag}"
            )
            event_type = f"WHALE_DUMP_{asset_class}"

        else:
            # Nothing concrete detected
            return None

        # 8. Build event
        event = {
            'event': event_type,
            'phase': phase,
            'direction': direction,
            'confidence': round(confidence, 3),
            'reason': reason.strip(),
            'vix_level': vix_level,
            'vix_regime': vix_regime,
            'symbol': symbol,
            'timestamp': now,
            'details': {
                'ask_wall': ask_wall,
                'velocity_dump': velocity_dump,
                'cvd_buy_ratio': cvd_buy_ratio,
                'price_change_pct': price_change,
            }
        }

        # 9. Set cooldown & cache
        self._event_cooldown[symbol] = now + self.COOLDOWN_SEC
        self._last_events[symbol] = event

        logger.warning(
            f"[{self.name}] 🐻 {phase} | {symbol} | Confidence: {confidence:.2f} | {reason}"
        )

        return event

    # ──────────────────────────────────────────────────────────────
    # VIX HELPERS
    # ──────────────────────────────────────────────────────────────

    def _get_vix_regime(self, vix_level: float):
        """
        Returns (regime_label, confidence_boost) based on VIX level.
        Higher fear = higher confidence in dump pattern.
        """
        if vix_level <= 0:
            return 'UNKNOWN', 0.0
        elif vix_level >= self.vix_panic:
            return 'PANIC', 0.15
        elif vix_level >= self.vix_fear:
            return 'FEAR', 0.08
        elif vix_level < self.vix_calm:
            return 'CALM', -0.05  # Low fear = less likely a coordinated dump
        else:
            return 'NORMAL', 0.0

    # ──────────────────────────────────────────────────────────────
    # STATUS / DASHBOARD
    # ──────────────────────────────────────────────────────────────

    def get_last_events(self) -> Dict[str, Dict]:
        """Return the most recent dumped event per symbol."""
        return self._last_events.copy()

    def get_status(self) -> Dict:
        """Dashboard-compatible status dict."""
        et_now = self._get_et_now()
        return {
            'enabled': getattr(config, 'DUMP_PUMP_ENABLED', True),
            'et_time': et_now.strftime('%H:%M ET'),
            'gold_window_active': self.is_in_window('XAUT/USDT'),
            'btc_window_active': self.is_in_window('BTC/USDT'),
            'last_events': self._last_events,
        }

    def receive_message(self, sender: Any, content: Any) -> None:
        pass
