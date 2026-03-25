"""
Tests for DumpPumpDetectorHolon
Tests: time window, ask wall, dump velocity, CVD exhaustion, VIX amplification
"""

import unittest
import time
import sys
import os
from unittest.mock import patch
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

# Path setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from HolonicTrader.agent_dump_pump_detector import DumpPumpDetectorHolon, GOLD_SYMBOLS, BTC_SYMBOLS, ET_TZ


def _make_df(length: int = 25, last_close: float = 100.0, drop_pct: float = 0.0, volume_mult: float = 1.0) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame for testing."""
    base = 100.0
    closes = [base] * (length - 1) + [last_close]
    volumes = [1000.0] * (length - 1) + [1000.0 * volume_mult]
    return pd.DataFrame({
        'open':   closes,
        'high':   [c + 0.1 for c in closes],
        'low':    [c - 0.1 for c in closes],
        'close':  closes,
        'volume': volumes,
    })


def _make_book(ask_notional: float = 0.0, ask_price: float = 100.5) -> dict:
    """Build a synthetic order book."""
    if ask_notional > 0:
        ask_vol = ask_notional / ask_price
        return {
            'bids': [[99.9, 10.0], [99.8, 5.0]],
            'asks': [[ask_price, ask_vol], [ask_price + 0.5, 10.0]],
        }
    return {
        'bids': [[99.9, 10.0]],
        'asks': [[100.1, 5.0]],
    }


class TestDumpPumpDetector(unittest.TestCase):

    def setUp(self):
        import config
        config.DUMP_PUMP_ENABLED = True
        config.DUMP_GOLD_WINDOW_ET_HOUR = 8
        config.DUMP_BTC_WINDOW_ET_HOUR = 10
        config.DUMP_WINDOW_MINUTES_BEFORE = 10
        config.DUMP_WINDOW_MINUTES_AFTER = 25
        config.DUMP_VELOCITY_PCT = 0.0035
        config.DUMP_RVOL_THRESHOLD = 2.5
        config.DUMP_ASK_WALL_DIST = 0.005
        config.DUMP_EXHAUSTION_CVD_RATIO = 0.60
        config.VIX_FEAR_THRESHOLD = 20.0
        config.VIX_PANIC_THRESHOLD = 30.0
        config.VIX_CALM_THRESHOLD = 15.0
        self.detector = DumpPumpDetectorHolon()

    # ── 1. Time window: inside gold window ──────────────────────────────────
    def test_inside_gold_window(self):
        """is_in_window() returns True for XAUT/USDT at 08:05 ET."""
        fake_et = datetime(2024, 3, 5, 8, 5, 0, tzinfo=ET_TZ)
        with patch.object(self.detector, '_get_et_now', return_value=fake_et):
            result = self.detector.is_in_window('XAUT/USDT')
        self.assertTrue(result, "Should be inside gold window at 08:05 ET")

    # ── 2. Time window: outside any window ──────────────────────────────────
    def test_outside_window(self):
        """is_in_window() returns False at 12:00 ET for both symbols."""
        fake_et = datetime(2024, 3, 5, 12, 0, 0, tzinfo=ET_TZ)
        with patch.object(self.detector, '_get_et_now', return_value=fake_et):
            gold_result = self.detector.is_in_window('XAUT/USDT')
            btc_result  = self.detector.is_in_window('BTC/USDT')
        self.assertFalse(gold_result, "Should be outside gold window at 12:00 ET")
        self.assertFalse(btc_result,  "Should be outside BTC window at 12:00 ET")

    # ── 3. Ask wall detection ────────────────────────────────────────────────
    def test_ask_wall_detected(self):
        """detect_ask_wall() returns wall dict when notional >= threshold."""
        # 100k ask wall with 24h vol of 10M (threshold = 0.5% * 10M = 50k)
        book = _make_book(ask_notional=150_000.0, ask_price=100.1)
        wall = self.detector.detect_ask_wall(book, daily_vol_usd=10_000_000.0)
        self.assertIsNotNone(wall, "Large ask wall should be detected")
        self.assertEqual(wall['type'], 'ASK_WALL')
        self.assertGreaterEqual(wall['notional'], 50_000.0)

    def test_ask_wall_not_detected_when_small(self):
        """detect_ask_wall() returns None when notional is below threshold."""
        book = _make_book(ask_notional=1_000.0, ask_price=100.1)
        wall = self.detector.detect_ask_wall(book, daily_vol_usd=10_000_000.0)
        self.assertIsNone(wall, "Small ask order should NOT be a wall")

    # ── 4. Dump velocity detection ──────────────────────────────────────────
    def test_dump_velocity_fires(self):
        """detect_dump_velocity() returns True on -0.5% candle with 3× volume."""
        # 24 bars at 100, last bar -0.5% at 3× volume
        df = _make_df(length=25, last_close=99.5, volume_mult=3.0)
        result = self.detector.detect_dump_velocity(df)
        self.assertTrue(result, "Should detect dump velocity: -0.5% + 3× RVOL")

    def test_dump_velocity_no_fire_small_drop(self):
        """detect_dump_velocity() returns False on -0.1% with normal volume."""
        df = _make_df(length=25, last_close=99.9, volume_mult=1.0)
        result = self.detector.detect_dump_velocity(df)
        self.assertFalse(result, "Small drop with normal volume should NOT fire")

    # ── 5. CVD exhaustion pattern ───────────────────────────────────────────
    def test_dump_exhaustion_pattern(self):
        """detect_exhaustion() returns True when price falling + high CVD buy ratio."""
        result = self.detector.detect_exhaustion(cvd_buy_ratio=0.70, price_change_pct=-0.005)
        self.assertTrue(result, "68% buy CVD while price -0.5% = exhaustion")

    def test_dump_exhaustion_not_triggered_on_diverge(self):
        """detect_exhaustion() returns False when both price and CVD are selling."""
        result = self.detector.detect_exhaustion(cvd_buy_ratio=0.30, price_change_pct=-0.005)
        self.assertFalse(result, "Low CVD buy ratio = no absorption, no exhaustion")

    # ── 6. VIX amplifies confidence ─────────────────────────────────────────
    def test_vix_panic_boosts_confidence(self):
        """analyze() produces higher confidence when VIX > 30 (panic)."""
        fake_et = datetime(2024, 3, 5, 8, 5, 0, tzinfo=ET_TZ)

        # Setup: ask wall + velocity dump to guarantee an event fires
        book = _make_book(ask_notional=200_000.0, ask_price=100.1)
        df   = _make_df(length=25, last_close=99.5, volume_mult=3.5)

        with patch.object(self.detector, '_get_et_now', return_value=fake_et):
            # Normal VIX
            self.detector._event_cooldown.clear()
            ev_normal = self.detector.analyze('XAUT/USDT', book_data=book, df=df,
                                               daily_vol_usd=10_000_000.0, vix_level=18.0)
            # Panic VIX
            self.detector._event_cooldown.clear()
            ev_panic = self.detector.analyze('XAUT/USDT', book_data=book, df=df,
                                              daily_vol_usd=10_000_000.0, vix_level=35.0)

        self.assertIsNotNone(ev_normal, "Event should fire in gold window")
        self.assertIsNotNone(ev_panic,  "Event should fire in gold window with panic VIX")
        self.assertGreater(
            ev_panic['confidence'], ev_normal['confidence'],
            "Panic VIX should yield higher confidence than normal VIX"
        )
        self.assertEqual(ev_panic['vix_regime'], 'PANIC')


if __name__ == '__main__':
    unittest.main()
