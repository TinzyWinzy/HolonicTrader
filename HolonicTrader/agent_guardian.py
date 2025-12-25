"""
ExitGuardianHolon - The "Defense" Brain (Phase 16)

Specialized in:
1. Unrealized PnL Protection
2. Time-at-Risk Management
3. Volatility-based Trailing Stops
"""

from typing import Any, Optional, Literal
from HolonicTrader.holon_core import Holon, Disposition
import config

class ExitGuardianHolon(Holon):
    def __init__(self, name: str = "ExitGuardian"):
        super().__init__(name=name, disposition=Disposition(autonomy=0.8, integration=0.4))
        self.last_exit_times = {} # {symbol: timestamp}

    def manage_satellite_positions(self, symbol: str, current_price: float, entry_price: float, direction: Literal['BUY', 'SELL']):
        """
        Hit & Run Management for Satellite Assets.
        Breakeven at +1.5%, Take Profit 50% at +3%.
        """
        from .agent_executor import TradeSignal
        
        if entry_price <= 0: return None
        
        # PnL Calculation
        if direction == 'BUY':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
            
        # 1. Breakeven Trigger (Move SL to Entry + 0.1%)
        # Logic: If we are > 1.5% profit, we assume the actuator moves the SL.
        # But here, we simulate the "Close" if price drops back.
        # Ideally, actuator handles hard stops. Guardian handles 'decisions'.
        # For simplicity in this Phase 4 architecture:
        # If PnL was high (>1.5%) and now drops to <= 0.1%, we exit.
        # But we don't store "high watermark" here yet. 
        # So we will implement the TP logic first.
        
        # 2. Take Profit (Scale Out 50%)
        # Note: Phase 4 Executor doesn't support partial closes well yet (binary ON/OFF per signal).
        # We will trigger a 'REDUCE' signal (which Executor treats as Exit for now, or we implement partials later).
        # For now, we take FULL PROFIT at 3% to be safe and simple.
        if pnl_pct >= config.SATELLITE_TAKE_PROFIT_1:
             print(f"[{self.name}] 🚀 SATELLITE HIT & RUN: {symbol} (+{pnl_pct*100:.2f}%) -> TAKING PROFIT")
             return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price)
             
        # 3. Hard Stop (ATR-based, approx 1.5x) or fixed %
        # We use a fixed 2% stop for simplicity if ATR is 0
        if pnl_pct <= -0.02:
             print(f"[{self.name}] 💥 SATELLITE STOP LOSS: {symbol} ({pnl_pct*100:.2f}%)")
             return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price)
             
        return None

    def analyze_for_exit(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        bb: dict,
        atr: float,
        metabolism_state: Literal['SCAVENGER', 'PREDATOR'],
        position_age_hours: float = 0.0,
        direction: Literal['BUY', 'SELL'] = 'BUY'
    ):
        
        # --- PHASE 25: SATELLITE ROUTING ---
        if symbol in config.SATELLITE_ASSETS:
            return self.manage_satellite_positions(symbol, current_price, entry_price, direction)
        # -----------------------------------

        from .agent_executor import TradeSignal
        
        if entry_price <= 0:
            return None
            
        # PnL Calculation
        if direction == 'BUY':
            pnl_pct = (current_price - entry_price) / entry_price
        else: # SELL (Short)
            pnl_pct = (entry_price - current_price) / entry_price
        
        # 1. HARD STOP LOSS (Circuit Breaker)
        sl_target = config.SCAVENGER_STOP_LOSS if metabolism_state == 'SCAVENGER' else config.PREDATOR_STOP_LOSS
        if pnl_pct <= -sl_target:
            print(f"[{self.name}] 🚨 EMERGENCY EXIT: {symbol} ({direction}) PnL {pnl_pct*100:.2f}%")
            return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price)

        if metabolism_state == 'SCAVENGER':
            # Scavenger Exits: Quick Mean Reversion
            if position_age_hours >= 4.0:
                print(f"[{self.name}] ⏳ TIME EXIT: {symbol} (4h reached)")
                return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price)
            
            if pnl_pct >= config.SCAVENGER_SCALP_TP:
                print(f"[{self.name}] ✅ SCALP TP: {symbol} (+{pnl_pct*100:.2f}%)")
                return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price)
                
            # Mean Reversion: Hit BB Middle
            if direction == 'BUY' and current_price >= bb['middle']:
                print(f"[{self.name}] 🔄 MEAN REVERSION (Long): {symbol} hit BB Middle")
                return TradeSignal(symbol=symbol, direction='SELL', size=1.0, price=current_price)
            elif direction == 'SELL' and current_price <= bb['middle']:
                print(f"[{self.name}] 🔄 MEAN REVERSION (Short): {symbol} hit BB Middle")
                return TradeSignal(symbol=symbol, direction='BUY', size=1.0, price=current_price)

        else: # PREDATOR
            # Predator Exits: Momentum Following
            if position_age_hours >= 8.0:
                print(f"[{self.name}] ⏳ TREND EXPIRY: {symbol} (8h reached)")
                return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price)
                
            if pnl_pct >= config.PREDATOR_TAKE_PROFIT:
                print(f"[{self.name}] 💰 PREDATOR TP: {symbol} (+{pnl_pct*100:.2f}%)")
                return TradeSignal(symbol=symbol, direction='SELL' if direction == 'BUY' else 'BUY', size=1.0, price=current_price)
            
            # Trailing ATR Stop
            if atr > 0:
                if direction == 'BUY':
                    trailing_stop = entry_price - (atr * config.PREDATOR_TRAILING_STOP_ATR_MULT)
                    if current_price <= trailing_stop:
                        print(f"[{self.name}] 🛡️ TRAILING STOP (Long): {symbol} @ {current_price:.4f}")
                        return TradeSignal(symbol=symbol, direction='SELL', size=1.0, price=current_price)
                else: # SELL
                    trailing_stop = entry_price + (atr * config.PREDATOR_TRAILING_STOP_ATR_MULT)
                    if current_price >= trailing_stop:
                        print(f"[{self.name}] 🛡️ TRAILING STOP (Short): {symbol} @ {current_price:.4f}")
                        return TradeSignal(symbol=symbol, direction='BUY', size=1.0, price=current_price)

        return None

    def record_exit(self, symbol: str, timestamp: Any):
        self.last_exit_times[symbol] = timestamp

    def get_health(self) -> dict:
        return {
            'status': 'OK',
            'exits_tracked': len(self.last_exit_times)
        }

    def receive_message(self, sender: Any, content: Any) -> None:
        pass
