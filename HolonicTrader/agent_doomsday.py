"""
DoomsdayHolon - The Crisis Management Brain (Phase 3.1)

Implements:
1. DEFCON threat level system (1-5)
2. SemanticRadar for crisis keyword detection
3. PAXG safe haven rotation on market crashes
4. Emergency shorting logic

Dependencies: SentimentHolon, GovernorHolon, ObserverHolon
"""

import time
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime, timedelta
from HolonicTrader.holon_core import Holon, Disposition
import config


# Crisis keyword weights (negative = bearish crisis)
CRISIS_KEYWORDS = {
    'crash': -1.0,
    'hack': -1.0,
    'hacked': -1.0,
    'exploit': -0.8,
    'regulation': -0.5,
    'ban': -0.8,
    'SEC': -0.6,
    'lawsuit': -0.5,
    'fraud': -0.8,
    'ponzi': -0.9,
    'rug': -1.0,
    'bankrupt': -1.0,
    'insolvency': -0.9,
    'liquidity crisis': -0.9,
    'flash crash': -1.0,
    'black swan': -1.0,
    'delisting': -0.7,
    'subpoena': -0.6,
    'arrest': -0.7,
    'indictment': -0.8,
    'war': -0.7,
    'sanctions': -0.6,
}


class DoomsdayHolon(Holon):
    """
    Crisis detection and defensive action coordinator.
    
    DEFCON Levels:
    - 5: Normal operations
    - 4: Elevated caution (tighten stops)
    - 3: High alert (reduce exposure 50%)
    - 2: Severe crisis (close directional positions)
    - 1: Catastrophic (PAXG rotation, short BTC)
    """
    
    def __init__(self, name: str = "DoomsdayHolon"):
        super().__init__(name=name, disposition=Disposition(autonomy=0.9, integration=0.8))
        
        # Core state
        self.defcon_level = 5  # Start at normal
        self.last_defcon_change = time.time()
        self.defcon_cooldown = 300  # 5 min cooldown between level changes
        
        # Safe haven config
        self.safe_haven = getattr(config, 'PAXG_SYMBOL', 'PAXG/USDT')
        self.safe_haven_allocation = getattr(config, 'PAXG_CRISIS_ALLOCATION', 0.50)  # 50% to PAXG
        
        # Price tracking for crash detection
        self.price_history = {}  # symbol -> [(timestamp, price), ...]
        self.price_history_window = 4 * 60 * 60  # 4 hours
        
        # Crisis state
        self.crisis_active = False
        self.crisis_start_time = None
        self.crisis_actions_taken = []
        
        # Linked holons (set by stack_factory)
        self.sentiment = None
        self.governor = None
        self.observer = None
        self.executor = None
        
        print(f"[{self.name}] ☢️ Doomsday Protocol Initialized. DEFCON: {self.defcon_level}")
    
    def update_price_history(self, symbol: str, price: float):
        """Track price for crash detection."""
        now = time.time()
        
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append((now, price))
        
        # Prune old entries
        cutoff = now - self.price_history_window
        self.price_history[symbol] = [
            (t, p) for t, p in self.price_history[symbol] if t > cutoff
        ]
    
    def get_price_change(self, symbol: str, hours: float = 1.0) -> float:
        """Calculate price change % over specified hours."""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
            return 0.0
        
        now = time.time()
        target_time = now - (hours * 3600)
        
        # Find price closest to target time
        history = self.price_history[symbol]
        current_price = history[-1][1]
        
        old_price = None
        for t, p in history:
            if t <= target_time:
                old_price = p
        
        if old_price is None or old_price <= 0:
            return 0.0
        
        return (current_price - old_price) / old_price
    
    def assess_threat_level(self) -> int:
        """
        Analyze all inputs and determine current DEFCON level.
        Returns: DEFCON level 1-5
        """
        threat_score = 0.0
        reasons = []
        
        # 1. Sentiment Analysis
        if self.sentiment:
            sentiment_score = getattr(self.sentiment, 'score', 0.0)
            if sentiment_score < -0.5:
                threat_score += 2
                reasons.append(f"Sentiment Critical: {sentiment_score:.2f}")
            elif sentiment_score < -0.3:
                threat_score += 1
                reasons.append(f"Sentiment Bearish: {sentiment_score:.2f}")
        
        # 2. BTC Price Crash Detection
        btc_symbol = 'BTC/USDT'
        btc_1h = self.get_price_change(btc_symbol, 1.0)
        btc_4h = self.get_price_change(btc_symbol, 4.0)
        
        if btc_1h < -0.20:  # >20% crash in 1h = catastrophic
            threat_score += 4
            reasons.append(f"BTC Flash Crash: {btc_1h*100:.1f}% in 1h")
        elif btc_4h < -0.20:  # >20% in 4h
            threat_score += 3
            reasons.append(f"BTC Severe Crash: {btc_4h*100:.1f}% in 4h")
        elif btc_4h < -0.10:  # >10% in 4h
            threat_score += 2
            reasons.append(f"BTC Major Drop: {btc_4h*100:.1f}% in 4h")
        elif btc_1h < -0.05:  # >5% in 1h
            threat_score += 1
            reasons.append(f"BTC Elevated Drop: {btc_1h*100:.1f}% in 1h")
        
        # 3. Crisis Keywords in recent news
        if self.sentiment and hasattr(self.sentiment, 'recent_headlines'):
            crisis_hits = self._scan_headlines_for_crisis(self.sentiment.recent_headlines)
            if crisis_hits >= 3:
                threat_score += 2
                reasons.append(f"Crisis Keywords: {crisis_hits} headlines")
            elif crisis_hits >= 1:
                threat_score += 1
                reasons.append(f"Crisis Keywords: {crisis_hits} headlines")
        
        # 4. Governor drawdown check
        if self.governor:
            drawdown = getattr(self.governor, 'drawdown_pct', 0.0)
            if drawdown > 0.15:  # >15% portfolio drawdown
                threat_score += 1
                reasons.append(f"Portfolio Drawdown: {drawdown*100:.1f}%")
        
        # Map score to DEFCON
        if threat_score >= 5:
            new_defcon = 1  # Catastrophic
        elif threat_score >= 3:
            new_defcon = 2  # Severe
        elif threat_score >= 2:
            new_defcon = 3  # High
        elif threat_score >= 1:
            new_defcon = 4  # Elevated
        else:
            new_defcon = 5  # Normal
        
        # Apply cooldown before changing level
        now = time.time()
        if new_defcon != self.defcon_level:
            if now - self.last_defcon_change > self.defcon_cooldown:
                old_level = self.defcon_level
                self.defcon_level = new_defcon
                self.last_defcon_change = now
                
                # Log transition
                direction = "⬆️ ELEVATED" if new_defcon < old_level else "⬇️ RELAXED"
                print(f"[{self.name}] ☢️ {direction} DEFCON {old_level} → {new_defcon}")
                for r in reasons:
                    print(f"    └─ {r}")
                
                # Trigger crisis actions if entering high threat
                if new_defcon <= 2 and old_level > 2:
                    self._enter_crisis_mode(reasons)
        
        return self.defcon_level
    
    def _scan_headlines_for_crisis(self, headlines: List[str]) -> int:
        """Count crisis keyword matches in headlines."""
        if not headlines:
            return 0
        
        count = 0
        for headline in headlines:
            headline_lower = headline.lower()
            for keyword in CRISIS_KEYWORDS:
                if keyword.lower() in headline_lower:
                    count += 1
                    break  # One match per headline
        return count
    
    def _enter_crisis_mode(self, reasons: List[str]):
        """Activate crisis protocols."""
        self.crisis_active = True
        self.crisis_start_time = time.time()
        self.crisis_actions_taken = []
        
        print(f"[{self.name}] 🚨 CRISIS MODE ACTIVATED 🚨")
        print(f"[{self.name}] Reasons: {', '.join(reasons)}")
    
    def get_crisis_action(self) -> Dict[str, Any]:
        """
        Get recommended action based on current DEFCON level.
        
        Returns dict with action details for Governor/Executor to execute.
        """
        action = {
            'defcon': self.defcon_level,
            'action_type': None,
            'details': {}
        }
        
        if self.defcon_level == 5:
            action['action_type'] = 'NORMAL'
            action['details'] = {'message': 'Normal operations'}
            
        elif self.defcon_level == 4:
            action['action_type'] = 'TIGHTEN_STOPS'
            action['details'] = {
                'stop_loss_multiplier': 0.8,  # Tighten stops by 20%
                'message': 'Elevated caution - tightening stops'
            }
            
        elif self.defcon_level == 3:
            action['action_type'] = 'REDUCE_EXPOSURE'
            action['details'] = {
                'target_reduction': 0.50,  # Reduce exposure by 50%
                'block_new_longs': True,
                'message': 'High alert - reducing exposure'
            }
            
        elif self.defcon_level == 2:
            action['action_type'] = 'CLOSE_DIRECTIONAL'
            action['details'] = {
                'close_all_directional': True,
                'keep_arbitrage': True,  # Keep arb positions (delta neutral)
                'message': 'Severe crisis - closing directional positions'
            }
            
        elif self.defcon_level == 1:
            action['action_type'] = 'PAXG_ROTATION'
            action['details'] = {
                'safe_haven': self.safe_haven,
                'allocation': self.safe_haven_allocation,
                'allow_btc_short': True,
                'message': 'CATASTROPHIC - rotating to PAXG safe haven'
            }
        
        return action
    
    def trigger_safe_haven(self) -> bool:
        """
        Execute PAXG rotation via Governor/Executor.
        Called when DEFCON 1 reached.
        
        Returns: True if rotation initiated
        """
        if not self.governor or not self.executor:
            print(f"[{self.name}] ⚠️ Cannot execute safe haven - missing holons")
            return False
        
        if 'PAXG_ROTATION' in self.crisis_actions_taken:
            print(f"[{self.name}] ⏸️ PAXG rotation already executed this crisis")
            return False
        
        print(f"[{self.name}] 🏦 INITIATING PAXG SAFE HAVEN ROTATION")
        
        try:
            # Calculate PAXG position size
            balance = self.governor.balance
            allocation = balance * self.safe_haven_allocation
            
            # Get current PAXG price
            paxg_price = self.governor.latest_prices.get(self.safe_haven, 2000.0)
            
            if paxg_price > 0:
                quantity = allocation / paxg_price
                
                print(f"[{self.name}] 🛡️ Rotating ${allocation:.2f} to {self.safe_haven}")
                
                # Mark action taken
                self.crisis_actions_taken.append('PAXG_ROTATION')
                
                # Return signal for executor (actual execution handled by trader loop)
                return True
                
        except Exception as e:
            print(f"[{self.name}] ❌ Safe haven rotation failed: {e}")
        
        return False
    
    def run_doomsday_scan(self, latest_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Main scan loop. Called by TraderHolon or SignalProvider.
        
        Args:
            latest_prices: Dict of symbol -> price for crash detection
            
        Returns:
            Crisis action recommendation
        """
        # Update price history
        for symbol, price in latest_prices.items():
            self.update_price_history(symbol, price)
        
        # Assess current threat level
        self.assess_threat_level()
        
        # Get recommended action
        action = self.get_crisis_action()
        
        # Execute PAXG rotation at DEFCON 1
        if self.defcon_level == 1:
            if self.trigger_safe_haven():
                action['safe_haven_triggered'] = True
        
        return action
    
    def get_status(self) -> Dict[str, Any]:
        """Get current doomsday status for dashboard."""
        return {
            'defcon_level': self.defcon_level,
            'crisis_active': self.crisis_active,
            'crisis_start': self.crisis_start_time,
            'actions_taken': self.crisis_actions_taken,
            'safe_haven': self.safe_haven,
            'last_change': self.last_defcon_change
        }

    def get_dashboard_state(self) -> dict:
        """Expose doomsday data for the dashboard."""
        return {'doomsday': self.get_status()}
    
    def receive_message(self, sender: Any, content: Any) -> None:
        """Handle holon messages."""
        if content == 'HEALTH_CHECK':
            return {'status': 'ACTIVE', 'defcon': self.defcon_level}
        elif content == 'GET_STATUS':
            return self.get_status()
