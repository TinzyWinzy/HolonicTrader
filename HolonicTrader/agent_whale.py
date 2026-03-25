"""
WhaleHolon - The "Hunter" Brain (Phase X)

Specialized in:
1. Detecting massive Bid/Ask Walls (Whale intent).
2. "Whale-Scalper" Strategy execution.
3. High-frequency 1-minute time checks.
4. Volume confirmation over time (anti-spoofing).
"""

from typing import Any, Dict, List, Optional, Literal
from HolonicTrader.holon_core import Holon, Disposition
import config
import time
from collections import defaultdict

class WhaleHolon(Holon):
    def __init__(self, name: str = "WhaleHunter"):
        # Medium Autonomy, High Integration (Works closely with Oracle/Actuator)
        super().__init__(name=name, disposition=Disposition(autonomy=0.6, integration=0.8))
        self.last_scan_time = 0.0
        self.active_whales = {} # {symbol: {price: float, side: 'bid'|'ask', size: float}}
        # FIX 2026-03-23: Track whale signals over time to confirm persistence (anti-spoofing)
        self.whale_signal_history = defaultdict(list)  # {symbol: [(timestamp, notional, price), ...]}
        self.whale_confirmation_window = 180  # 3 minutes confirmation window
        self.whale_min_confirmations = 2  # Require 2+ sightings in window

    def check_bid_wall(self, symbol: str, depth: Dict, daily_vol: float = 0.0) -> Optional[Dict]:
        """
        Check for a "Bid Wall" condition:
        - Wall Size >= 0.5% of 24h Volume (Dynamic)
        - Floor: $50,000 (To avoid noise in illiquid pairs)
        - Wall Price <= Mid Price + 0.2% (Close to market)
        - FIX 2026-03-23: Requires persistence (2+ sightings in 3min) to avoid spoofing

        Returns wall details if found, else None.
        """
        if not depth or 'bids' not in depth:
            return None

        bids = depth['bids']
        if not bids: return None

        best_bid = bids[0][0]

        # 1. Dynamic Threshold Logic
        if daily_vol > 0:
            wall_threshold_usd = max(50000.0, daily_vol * 0.005) # 0.5% or $50k min
        else:
            wall_threshold_usd = 500000.0 # Legacy fallback

        for price, vol in bids[:15]:
            notional = price * vol

            if notional >= wall_threshold_usd:
                # 2. Distance Check
                dist = (best_bid - price) / best_bid
                if dist <= 0.005:
                    # === FIX 2026-03-23: Volume Confirmation Over Time ===
                    # Record this sighting
                    now = time.time()
                    self.whale_signal_history[symbol].append((now, notional, price))

                    # Clean old entries outside confirmation window
                    cutoff = now - self.whale_confirmation_window
                    self.whale_signal_history[symbol] = [
                        (t, n, p) for t, n, p in self.whale_signal_history[symbol] if t > cutoff
                    ]

                    # Count confirmations (similar price within 0.2%)
                    confirmations = sum(
                        1 for t, n, p in self.whale_signal_history[symbol]
                        if abs(p - price) / price <= 0.002  # Same price level
                    )

                    # Require multiple confirmations to avoid spoofed walls
                    if confirmations < self.whale_min_confirmations:
                        # Wall detected but not confirmed yet - log for debugging
                        # print(f"[{self.name}] 🐋 {symbol} BID_WALL detected ({confirmations}/{self.whale_min_confirmations} confirmations)")
                        return None  # Not confirmed yet

                    print(f"[{self.name}] 🐋 {symbol} BID_WALL CONFIRMED: ${notional:,.0f} @ {price} ({confirmations} sightings in {self.whale_confirmation_window}s)")
                    # === END VOLUME CONFIRMATION ===

                    return {
                        'type': 'BID_WALL',
                        'price': price,
                        'vol': vol,
                        'notional': notional,
                        'distance': dist,
                        'threshold_used': wall_threshold_usd,
                        'confirmations': confirmations,
                        'confirmed': True
                    }

        return None

    def check_ask_wall(self, symbol: str, depth: Dict, daily_vol: float = 0.0) -> Optional[Dict]:
        """
        Check for an "Ask Wall" condition (sell-side, dump pressure):
        - Wall Size >= 0.5% of 24h Volume (Dynamic)
        - Floor: $50,000
        - Wall Price <= Best Ask + 0.5% (Close to market)

        Returns wall details if found, else None.
        Used by DumpPumpDetectorHolon for whale dump detection.
        """
        if not depth or 'asks' not in depth:
            return None

        asks = depth['asks']
        if not asks:
            return None

        best_ask = asks[0][0]

        # Dynamic Threshold Logic (same as bid_wall)
        if daily_vol > 0:
            wall_threshold_usd = max(50000.0, daily_vol * 0.005)
        else:
            wall_threshold_usd = 500000.0

        for price, vol in asks[:15]:
            notional = price * vol

            if notional >= wall_threshold_usd:
                # Distance Check: close to market (within 0.5%)
                dist = (price - best_ask) / best_ask
                if dist <= 0.005:
                    return {
                        'type': 'ASK_WALL',
                        'price': price,
                        'vol': vol,
                        'notional': notional,
                        'distance': dist,
                        'threshold_used': wall_threshold_usd
                    }

        return None

    def check_scout_entry(self, symbol: str, observer_data: Dict) -> Optional[Dict]:
        """
        Main Logic Check.
        Called by Trader during 1m loop.
        """
        # 1. Depth Check
        depth = observer_data.get(f"depth_{symbol}")
        curr_price = observer_data.get('price', 0.0)
        
        if not depth or curr_price == 0:
            return None
            
        wall = self.check_bid_wall(symbol, depth)
        
        if wall:
            print(f"[{self.name}] 🐋 WHALE DETECTED on {symbol}: ${wall['notional']:,.0f} Bid Wall @ {wall['price']}")
            
            # CONFIRMATION: Price Action must be bouncing off it or holding
            # For now, we signal entry if we are just above it (front-run the wall)
            
            # Setup: BUY just above wall
            entry_price = wall['price'] * 1.001 # +0.1% front-run
            
            if curr_price <= entry_price * 1.002: # Ensure we haven't missed it
                return {
                    'signal': 'BUY',
                    'symbol': symbol,
                    'price': curr_price,
                    'reason': f"Whale Wall Support (${wall['notional']/1000:.0f}k)",
                    'stop_loss': wall['price'] * 0.995, # tight stop below wall
                    'target': curr_price * 1.015 # 1.5% scalp target
                }
                
        return None

    def receive_message(self, sender: Any, content: Any) -> None:
        pass
