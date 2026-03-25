"""
RegimeController Holon (Phase 7)

Manages stateful capital regime graduation:
- MICRO ($0-$49): Maximum safety, minimal risk
- SMALL ($50-$249): Unlocks limited autonomy
- MEDIUM ($250-$999): Full trading capabilities

Promotion requires: Capital + Stability + Behavior Integrity
Demotion is fast and ruthless.
"""

import time
from typing import Dict, Optional, List
from collections import deque
import config
# Force Reload: AEHML Compliance Check


class RegimeController:
    """
    The Regime Controller is the authority on risk permissions.
    It tracks capital regime, handles promotion/demotion, and freezes trading during transitions.
    
    GRADUATION BONUSES (Capital Allocator Mindset):
    - Permanent whitelist unlock tiers
    - Iron Bank profit siphoning on promotion
    - Incremental slot increases
    """
    
    def __init__(self):
        self.name = "RegimeController"
        
        # Current State - Auto-detect based on Mandate
        if config.INITIAL_CAPITAL >= config.REGIME_ATOM_FLOOR:
            self.current_regime = 'ATOM'
        elif config.INITIAL_CAPITAL >= config.REGIME_LARGE_CEILING:
            self.current_regime = 'GRAVITY'
        elif config.INITIAL_CAPITAL >= config.REGIME_SMALL_CEILING:
            self.current_regime = 'LARGE'
        else:
            self.current_regime = 'SMALL'
            
        print(f"[{self.name}] Initialized. Auto-Detected Regime: {self.current_regime} (Cap: ${config.INITIAL_CAPITAL})")
        self.previous_regime = self.current_regime
        
        # Promotion Tracking
        self.equity_history: deque = deque(maxlen=1000)  # (timestamp, equity) tuples
        self.promotion_consecutive_cycles = 0 # New: Count cycles instead of hours
        self.promotion_candidate_regime = None
        
        # Health Tracking
        self.health_events: deque = deque(maxlen=100)  # Track recent issues
        self.trade_count = 0
        self.solvency_rejections = 0
        self.gc_corrections = 0
        self.hwm_resets = 0
        self.avg_slippage = 0.0
        
        # Transition State
        self.transition_pending = False
        self.transition_target: Optional[str] = None
        
        # High Water Mark for Demotion
        self.peak_equity = 0.0
        
        # === GRADUATION BONUSES (Capital Allocator) ===
        self.unlock_tier = 0        # 0=Base, 1=Tier1, 2=Tier2, 3=Full
        self.iron_bank_balance = 0.0  # Profit siphoned for compounding
        self.graduation_slot_bonus = 0  # Extra slots earned
        self.total_promotions = 0     # Track lifetime promotions
        
        print(f"[{self.name}] Initialized. Starting Regime: {self.current_regime}")
        
    def update_state(self, equity: float, health_metrics: Dict = None):
        """
        Called every cycle to update regime state.
        """
        now = time.time()
        
        # 1. Record Equity History
        self.equity_history.append((now, equity))
        
        # 2. Update Peak (for drawdown calculation)
        if equity > self.peak_equity:
            self.peak_equity = equity
            
        # 3. Update Health Metrics
        if health_metrics:
            if health_metrics.get('solvency_rejection'):
                self.solvency_rejections += 1
                self.health_events.append(('solvency_rejection', now))
            if health_metrics.get('gc_correction'):
                self.gc_corrections += 1
                self.health_events.append(('gc_correction', now))
            if health_metrics.get('hwm_reset'):
                self.hwm_resets += 1
                self.health_events.append(('hwm_reset', now))
            if 'slippage' in health_metrics:
                self.avg_slippage = (self.avg_slippage * 0.9) + (health_metrics['slippage'] * 0.1)
            if health_metrics.get('trade_completed'):
                self.trade_count += 1
                
        # 4. Check for Demotion (FAST - Instant)
        demoted = self._check_demotion(equity)
        if demoted:
            return
            
        # 5. Check for Promotion (3 Cycles > Threshold)
        self._check_promotion(equity)
        
    def _check_demotion(self, equity: float) -> bool:
        """
        Demotion is fast and ruthless.
        """
        demoted = False
        target_regime = None
        reason = ""
        
        # Health Demotion (Mandate: Health < 0.3)
        current_health = self._calculate_health_score()
        if current_health < config.REGIME_DEMOTION_HEALTH_THRESHOLD:
             # Demote one step down if possible
             if self.current_regime == 'ATOM': target_regime = 'GRAVITY'
             elif self.current_regime == 'GRAVITY': target_regime = 'LARGE'
             elif self.current_regime == 'LARGE': target_regime = 'SMALL'
             
             if target_regime:
                 reason = f"Health Score {current_health:.2f} < {config.REGIME_DEMOTION_HEALTH_THRESHOLD}"
                 self._execute_transition(target_regime, reason, is_demotion=True)
                 return True

        # Capital Demotion (Instant if below floor)
        # SMALL has no floor (0)
        # LARGE floor is 1000
        # GRAVITY floor is 5M
        # ATOM floor is 50M
        
        if self.current_regime == 'ATOM' and equity < config.REGIME_ATOM_FLOOR:
            target_regime = 'GRAVITY'
            reason = f"Equity ${equity:.2f} < ${config.REGIME_ATOM_FLOOR}"
        elif self.current_regime == 'GRAVITY' and equity < config.REGIME_LARGE_CEILING: # 5M
             target_regime = 'LARGE'
             reason = f"Equity ${equity:.2f} < ${config.REGIME_LARGE_CEILING}"
        elif self.current_regime == 'LARGE' and equity < config.REGIME_SMALL_CEILING: # 1K
             target_regime = 'SMALL'
             reason = f"Equity ${equity:.2f} < ${config.REGIME_SMALL_CEILING}"
                    
        if target_regime:
            self._execute_transition(target_regime, reason, is_demotion=True)
            demoted = True
            
        return demoted
        
    def _check_promotion(self, equity: float):
        """
        Promotion requires:
        1. Capital > Threshold
        2. 3 Consecutive Cycles
        3. Health > 0.6
        """
        target_regime = None
        threshold = 9999999999999.0 # Infinity
        
        # Determine Potential Next Level
        if self.current_regime == 'SMALL':
            if equity >= config.REGIME_SMALL_CEILING:
                target_regime = 'LARGE'
                threshold = config.REGIME_SMALL_CEILING
        elif self.current_regime == 'LARGE':
            if equity >= config.REGIME_LARGE_CEILING:
                target_regime = 'GRAVITY'
                threshold = config.REGIME_LARGE_CEILING
        elif self.current_regime == 'GRAVITY':
            if equity >= config.REGIME_GRAVITY_CEILING:
                target_regime = 'ATOM'
                threshold = config.REGIME_GRAVITY_CEILING
                
        if not target_regime:
            self.promotion_consecutive_cycles = 0
            self.promotion_candidate_regime = None
            return
            
        # Check Candidates
        if self.promotion_candidate_regime != target_regime:
            # Reset if target changed (e.g. was aiming for LARGE, suddenly crashed? No wait, equity check passed)
            self.promotion_consecutive_cycles = 0
            self.promotion_candidate_regime = target_regime
            
        # Increment Counter
        self.promotion_consecutive_cycles += 1
        print(f"[{self.name}] ⏳ Promotion Progress: {self.current_regime} -> {target_regime} ({self.promotion_consecutive_cycles}/{config.REGIME_PROMOTION_CYCLES})")
        
        # Check if Requirements Met
        if self.promotion_consecutive_cycles >= config.REGIME_PROMOTION_CYCLES:
            # Check Health
            health_score = self._calculate_health_score()
            if health_score > config.REGIME_PROMOTION_HEALTH_THRESHOLD:
                # PROMOTE!
                self._execute_transition(target_regime, f"Capital > ${threshold} for {config.REGIME_PROMOTION_CYCLES} cycles. Health {health_score:.2f}", is_demotion=False)
            else:
                print(f"[{self.name}] ⚠️ Promotion Held: Health {health_score:.2f} <= {config.REGIME_PROMOTION_HEALTH_THRESHOLD}")


    def check_vol_window_conditions(self, observer_data: Dict[str, float]) -> bool:
        """
        Check if we should enter the VOL_WINDOW High-Entropy Regime.
        Requires:
        1. BTC 24h Realized Vol > 45%
        2. Avg Funding > 0.03% (Positive)
        3. Spread < 0.4%
        4. "Meme" Listing < 14 days (Optional check, assumed checked by caller or config)
        """
        btc_vol = observer_data.get('btc_vol', 0.0)
        avg_funding = observer_data.get('avg_funding', 0.0) # 8h rate
        avg_spread = observer_data.get('avg_spread', 0.0)
        
        # 1. Vol Check
        if btc_vol < config.VOL_WINDOW_BTC_VOL_THRESHOLD:
            return False
            
        # 2. Funding Check (Positive Bullish Sentiment)
        if avg_funding < config.VOL_WINDOW_FUNDING_THRESHOLD:
            return False
            
        # 3. Spread Check (Liquidity)
        if avg_spread > config.VOL_WINDOW_SPREAD_THRESHOLD:
            return False
            
        # All Clear
        return True

    def attempt_vol_window_entry(self, observer_data: Dict[str, float]):
        """
        Public method to trigger VOL_WINDOW entry if conditions met.
        Overrules standard regimes.
        """
        if self.current_regime == 'VOL_WINDOW':
            # Check exit conditions? (Reverse logic)
            # If Vol drops OR Funding turns negative -> Exit
            if not self.check_vol_window_conditions(observer_data):
                print(f"[{self.name}] 📉 VOL_WINDOW Conditions Lost. Reverting to NORMAL.")
                # Revert to appropriate capital regime
                # For safety, go to SMALL first
                self._execute_transition('SMALL', "VOL_WINDOW Conditions Lost", is_demotion=True)
            return

        # Check Entry
        if self.check_vol_window_conditions(observer_data):
            self._execute_transition('VOL_WINDOW', "High-Entropy Conditions Detected (Vol+Funding+Spread)", is_demotion=False)

    def override_regime(self, target_regime: str):
        """
        Manual override method to force a specific regime (e.g. DEFENSIVE).
        Called by Governor during emergencies or via User Command.
        """
        if target_regime not in config.REGIME_PERMISSIONS and target_regime != 'DEFENSIVE' and target_regime != 'HIBERNATE':
             print(f"[{self.name}] ⚠️ Invalid Override Target: {target_regime}")
             return

        print(f"[{self.name}] 🔧 MANUAL OVERRIDE: Forcing Regime to {target_regime}")
        self._execute_transition(target_regime, "Manual/Governor Override", is_demotion=(target_regime=='DEFENSIVE'))
        
    def _calculate_health_score(self) -> float:
        """
        Calculate behavior integrity score (0.0 to 1.0).
        Penalizes: Solvency rejections, GC corrections, HWM resets, high slippage.
        """
        # Mandate doesn't specify min trades, but we avoid premature judgment.
        if self.trade_count < 5:
            return 0.5  # Neutral Start (Fixes 'Coma' bug)
            
        # Start at 1.0, deduct for each issue
        score = 1.0
        
        # Recent events (last 20 trades worth of time)
        recent_window = 20 * 3600  # Rough estimate: 1h per trade
        now = time.time()
        
        recent_solvency = sum(1 for e, t in self.health_events if e == 'solvency_rejection' and now - t < recent_window)
        recent_gc = sum(1 for e, t in self.health_events if e == 'gc_correction' and now - t < recent_window)
        recent_hwm = sum(1 for e, t in self.health_events if e == 'hwm_reset' and now - t < recent_window)
        
        # Deductions
        score -= recent_solvency * 0.05
        score -= recent_gc * 0.03
        score -= recent_hwm * 0.10
        score -= min(0.10, self.avg_slippage * 10)  # 1% slippage = 0.10 deduction
        
        return max(0.0, min(1.0, score))
        
    def _execute_transition(self, target_regime: str, reason: str, is_demotion: bool):
        """
        Execute a regime transition with proper handshake.
        On PROMOTION: Apply graduation bonuses (permanent unlocks, convex growth).
        """
        direction = "DEMOTION" if is_demotion else "PROMOTION"
        
        print(f"\n{'='*50}")
        print(f"[{self.name}] 🔄 REGIME {direction}: {self.current_regime} → {target_regime}")
        print(f"[{self.name}] Reason: {reason}")
        print(f"{'='*50}\n")
        
        # 1. Freeze new entries
        self.transition_pending = True
        self.transition_target = target_regime
        
        # 2. Update regime
        self.previous_regime = self.current_regime
        self.current_regime = target_regime
        
        # 3. Reset promotion eligibility
        self.promotion_consecutive_cycles = 0
        self.promotion_candidate_regime = None
        
        # === GRADUATION BONUSES (On Promotion Only) ===
        if not is_demotion:
            self.total_promotions += 1
            
            # a. Permanent Whitelist Tier Unlock (Never Demoted)
            old_tier = self.unlock_tier
            if target_regime == 'LARGE' and self.unlock_tier < 1:
                self.unlock_tier = 1  # Tier 1: Mid-cap altcoins unlocked
            elif target_regime == 'GRAVITY' and self.unlock_tier < 2:
                self.unlock_tier = 2  # Tier 2: All assets except memes
            elif target_regime == 'ATOM' and self.unlock_tier < 3:
                self.unlock_tier = 3  # Tier 3: Full asset universe
                
            if self.unlock_tier > old_tier:
                print(f"[{self.name}] 🔓 WHITELIST UNLOCK: Tier {old_tier} → Tier {self.unlock_tier} (PERMANENT)")
            
            # b. Slot Bonus (Incremental, +1 per promotion, max +4)
            if self.graduation_slot_bonus < 4:
                self.graduation_slot_bonus += 1
                print(f"[{self.name}] 📊 SLOT BONUS: +{self.graduation_slot_bonus} extra positions unlocked")
            
            # c. Iron Bank Profit Siphon (10% of gains go to fortress)
            siphon_pct = getattr(config, 'GRADUATION_SIPHON_PCT', 0.10)
            recent_equity = [eq for ts, eq in self.equity_history if time.time() - ts < 3600]
            if len(recent_equity) >= 2:
                recent_gain = max(0, recent_equity[-1] - recent_equity[0])
                siphon_amount = recent_gain * siphon_pct
                if siphon_amount > 0:
                    self.iron_bank_balance += siphon_amount
                    print(f"[{self.name}] 🏦 IRON BANK SIPHON: +${siphon_amount:.2f} → Total: ${self.iron_bank_balance:.2f}")
            
            print(f"[{self.name}] 🎓 GRADUATION SUMMARY: Tier {self.unlock_tier}, Slots +{self.graduation_slot_bonus}, Bank ${self.iron_bank_balance:.2f}")
        
        # 4. On demotion, reset peak to current (to avoid immediate re-demotion)
        # NOTE: Graduation bonuses are PERMANENT (convex growth)
        if is_demotion:
            print(f"[{self.name}] ⚠️ Demotion does NOT reset graduation bonuses (Tier, Slots, Bank remain)")
            pass
            
        # 5. Clear transition flag (will be handled by Governor during next consolidation)
        # The transition_pending flag tells Governor to re-evaluate all positions
        
    def complete_transition(self):
        """
        Called by Governor after consolidation is complete.
        Unlocks trading.
        """
        if self.transition_pending:
            print(f"[{self.name}] ✅ Transition Complete. Regime: {self.current_regime}")
            self.transition_pending = False
            self.transition_target = None
            
    def get_current_regime(self) -> str:
        return self.current_regime
        
    def get_permissions(self) -> Dict:
        """
        Return the permissions dict for the current regime.
        """
        # Handle VOL_WINDOW specially if not in config dict yet (it should be)
         # If not in config yet, return a hardcoded high-entropy set
        if self.current_regime == 'VOL_WINDOW':
            return {
                'max_positions': config.VOL_WINDOW_MAX_POSITIONS,
                'max_stacks': 0,
                'max_exposure_ratio': config.VOL_WINDOW_LEVERAGE,
                'max_leverage': config.VOL_WINDOW_LEVERAGE,
                'allowed_pairs': config.ALLOWED_ASSETS, # All allowed
                'correlation_check': False # Speed over safety
            }
            
        return config.REGIME_PERMISSIONS.get(self.current_regime, config.REGIME_PERMISSIONS['SMALL'])
        
    def is_transition_pending(self) -> bool:
        return self.transition_pending
    
    def get_graduation_bonuses(self) -> Dict:
        """
        Returns currently unlocked graduation bonuses for Governor/Trader use.
        These are PERMANENT and survive demotions.
        """
        return {
            'unlock_tier': self.unlock_tier,          # 0-3
            'slot_bonus': self.graduation_slot_bonus,  # 0-4
            'iron_bank': self.iron_bank_balance,       # USD
            'total_promotions': self.total_promotions
        }
        
    def get_status_summary(self) -> Dict:
        """
        Return a summary for Dashboard display.
        """
        return {
            'regime': self.current_regime,
            'peak_equity': self.peak_equity,
            'health_score': self._calculate_health_score(),
            'promotion_consecutive_cycles': self.promotion_consecutive_cycles, # Updated name
            'promotion_target': self.promotion_candidate_regime,
            'transition_pending': self.transition_pending,
            'trade_count': self.trade_count,
            # Graduation Bonuses (Permanent)
            'unlock_tier': self.unlock_tier,
            'slot_bonus': self.graduation_slot_bonus,
            'iron_bank': self.iron_bank_balance,
        }
