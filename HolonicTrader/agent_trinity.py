"""
Trinity Strategy Module (Phase 46)

Implements the "Trinity" Asset Rotation Logic:
- ORDERED (Bull): 60% TAO / 40% BTC (The Amplifier)
- TRANSITION: 100% BTC (The Anchor)
- CHAOTIC (Bear): 100% XAUT (The Shield)

This module acts as a sub-advisor to the Trader Holon.
"""

from typing import Dict, List, Optional
import config

class TrinityStrategy:
    def __init__(self):
        self.name = "TrinityStrategy"
        
        # Load Preferences from Config (Scientific Taxonomy)
        self.scavengers = getattr(config, 'ASSET_PREF_SCAVENGER', ['XRP/USDT', 'DOGE/USDT'])
        self.predators = getattr(config, 'ASSET_PREF_PREDATOR', ['ETH/USDT', 'SOL/USDT'])
        self.anchor = 'BTC/USDT'
        
    def get_allocation_target(self, market_regime: str, btc_trend: str = 'BULL', current_capital_regime: str = 'SMALL') -> Dict[str, float]:
        """
        Returns target portfolio weights based on regime.
        Args:
            market_regime: 'ORDERED', 'CHAOTIC', or 'TRANSITION'
            btc_trend: 'BULL' or 'BEAR'
            current_capital_regime: 'MICRO', 'SMALL', 'MEDIUM' (for permissions)
        """
        targets = {}
        
        # Get allowed assets for current capital regime
        allowed_assets = config.REGIME_PERMISSIONS[current_capital_regime]['allowed_pairs']
        
        # Helper to find ALL eligible assets (Basket Approach)
        def get_valid_basket(preference_list, limit=3):
            basket = []
            for asset in preference_list:
                if asset in allowed_assets:
                    basket.append(asset)
                    if len(basket) >= limit: break
            
            # Ensure Anchor is included if we have nothing else
            if not basket and self.anchor in allowed_assets:
                basket.append(self.anchor)
            return basket
            
        # 1. CHAOTIC / BEAR TREND (Crisis or Correction)
        if market_regime == 'CHAOTIC' or btc_trend == 'BEAR':
            # USER ANALYSIS: ETH, XMR, XRP perform when BTC is down
            # Prioritize specific Hedge Assets if allowed
            hedge_candidates = getattr(config, 'BTC_HEDGE_ASSETS', ['ETH/USDT', 'XMR/USDT', 'XRP/USDT'])
            basket = get_valid_basket(hedge_candidates, limit=3)
            
            # If no hedge assets allowed, fallback to Scavengers or Anchor
            if not basket:
                if market_regime == 'CHAOTIC':
                    basket = get_valid_basket(self.scavengers, limit=3)
                else: 
                     # Bear Trend but Ordered Regime? Anchor default.
                     if self.anchor in allowed_assets: basket = [self.anchor]

            weight_per_asset = 1.0 / len(basket) if basket else 0
            for a in basket: targets[a] = weight_per_asset
            return targets
            
        # 2. TRANSITION -> ANCHOR (Safety)
        if market_regime == 'TRANSITION':
            targets[self.anchor] = 1.0
            return targets
            
        # 3. ORDERED (Bullish) -> PREDATOR + SCAVENGER MIX (Diversified Attack)
        if market_regime == 'ORDERED':
            if btc_trend == 'BULL':
                # Aggressive Beta Capture: Mixed Squad
                # 2 Predators (Majors) + 2 Scavengers (Alts)
                predators = get_valid_basket(self.predators, limit=3)
                scavengers = get_valid_basket(self.scavengers, limit=2)
                
                combined_basket = list(set(predators + scavengers))
                
                # Split weight evenly or favor Anchor?
                # Let's favor Anchor slightly if present, but allow width.
                if combined_basket:
                    weight_per_asset = 1.0 / len(combined_basket)
                    for a in combined_basket: targets[a] = weight_per_asset
                    
            else:
                # Bullish Regime but Bearish Trend? (Correction) -> Anchor + Top Scavengers (Defensive Rotation)
                # Just Anchor for safety
                targets[self.anchor] = 1.0
                
        # Default Logic
        if not targets:
            # NANO FIX: Only default to anchor if allowed
            if self.anchor in allowed_assets:
                targets[self.anchor] = 1.0
            
        return targets

    def get_watch_list(self) -> List[str]:
        """Return list of assets to monitor (All Predators + Scavengers + Anchor)."""
        # Unique set of all assets
        candidates = set(self.predators + self.scavengers + [self.anchor])
        return list(candidates)
