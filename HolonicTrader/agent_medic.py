"""
MedicHolon - The "Field Physician"
"I heal the wounded, I wake the sleeping."

Responsibilities:
1.  **Vitals Monitoring**: Pulse (Trade Activity), BP (Margin Level), O2 (Solvency).
2.  **Triage**: Close bleeding positions that are draining "blood" (Equity) too fast.
3.  **Resuscitation (CPR)**: Force-resetting Governor state when HIBERNATION is prolonged but safe.
4.  **Adrenaline**: Injecting capital/risk appetite override to restart the heart.
"""

from HolonicTrader.holon_core import Holon, Disposition
import time
import config

class MedicHolon(Holon):
    def __init__(self, name: str = "MedicHolon"):
        super().__init__(name=name, disposition=Disposition(autonomy=0.5, integration=0.8))
        self.last_pulse_check = time.time()
        self.cpr_attempts = 0
        self.max_cpr_attempts = 3
        self.patient_status = "STABLE"

    def check_vitals(self, trader_ref) -> dict:
        """
        Diagnose the patient (Trading System).
        """
        gov = trader_ref.sub_holons.get('governor')
        executor = trader_ref.sub_holons.get('executor')
        
        if not (gov and executor):
            return {'status': 'DEAD', 'reason': 'Missing Organs (Agents)'}

        current_time = time.time()
        
        # 1. Check Pulse (Last Trade Activity)
        last_trade_time = max(gov.pool_a_last_entry_time, gov.pool_b_last_entry_time, getattr(gov, 'last_ratchet_time', 0), self.last_pulse_check)
        seconds_since_pulse = current_time - last_trade_time
        
        # 2. Check Blood Pressure (Margin Level)
        # Re-calc margin check
        p_state = gov._calculate_portfolio_state()
        margin_level = p_state.get('margin_level', 999.0)
        
        # 3. Check Shock State (Drawdown Lock)
        is_hibernating = (gov.state == 'HIBERNATE') or gov.drawdown_lock
        
        vitals = {
            'heart_rate': 'FLATLINE' if seconds_since_pulse > 3600 else 'BEATING',
            'seconds_since_pulse': seconds_since_pulse,
            'margin_level': margin_level,
            'is_hibernating': is_hibernating,
            'drawdown': gov.drawdown_pct,
            'equity': gov.balance
        }
        
        return vitals

    def perform_triage(self, trader_ref):
        """
        Main Loop: Check vitals and intervene if necessary.
        """
        vitals = self.check_vitals(trader_ref)
        self.patient_status = "STABLE"
        
        # LOG VITALS (Dashboard Pulse)
        # print(f"[{self.name}] 🏥 Vitals: Rate={vitals['heart_rate']} | BP={vitals['margin_level']:.1f} | Shock={vitals['is_hibernating']}")

        gov = trader_ref.sub_holons.get('governor')

        # CONDITION 1: COMA DETECTED (Hibernating + Healthy Margin + Long Silence)
        # If we are stuck in HIBERNATE due to a "Phantom Spike" or "Fixed Drawdown" from yesterday,
        # but our Margin is totally fine (> 2.0), we should WAKE UP.
        if vitals['is_hibernating'] and vitals['margin_level'] > 2.0:
            if vitals['seconds_since_pulse'] > 1800: # 30 mins of silence
                print(f"[{self.name}] 🚑 COMA DETECTED: Patient healthy (margin > 2.0) but stuck in HIBERNATION.")
                self.administer_cpr(gov)
                
        # CONDITION 2: CARDIAC ARREST (No trades for 2 hours in ACTIVE mode)
        # Maybe config is too tight?
        elif not vitals['is_hibernating'] and vitals['seconds_since_pulse'] > 7200:
             print(f"[{self.name}] 🩺 LOW PULSE: System idle for 2+ hours. Injecting Adrenaline (Reset Scouters).")
             # Force Scout Refresh
             trader_ref.scout_last_run = 0

             # Force Risk Reset (Small bump)
             if gov.risk_multiplier < 1.0:
                 gov.risk_multiplier = 1.0
                 print(f"[{self.name}] 💉 ADRENALINE: Reset Risk Multiplier to 1.0x")

        # CONDITION 3: TOXIC POSITION DETECTION (Check for positions with extreme funding costs)
        self.check_for_toxic_positions(trader_ref)
        
        # Update pulse timestamp (system is alive if triage is running)
        self.last_pulse_check = time.time()

    def check_for_toxic_positions(self, trader_ref):
        """
        Check for positions with toxic funding rates that should be closed immediately.
        """
        gov = trader_ref.sub_holons.get('governor')
        executor = trader_ref.sub_holons.get('executor')
        arbitrage = trader_ref.sub_holons.get('arbitrage')

        if not (gov and executor and arbitrage):
            return

        # Check if arbitrage module has funding yield data
        if hasattr(arbitrage, 'funding_yields'):
            toxic_positions = []

            for symbol, funding_apy in arbitrage.funding_yields.items():
                # Check if this symbol is in our positions
                if symbol in executor.held_assets:
                    # Define toxic funding threshold (same as in arbitrage module)
                    if abs(funding_apy) > 500.0:  # More than 500% annualized funding cost
                        toxic_positions.append({
                            'symbol': symbol,
                            'funding_apy': funding_apy,
                            'position_size': executor.held_assets[symbol],
                            'direction': 'LONG' if executor.held_assets[symbol] > 0 else 'SHORT'
                        })

                        print(f"[{self.name}] ☣️ TOXIC POSITION DETECTED: {symbol} ({'LONG' if executor.held_assets[symbol] > 0 else 'SHORT'}) - Funding: {funding_apy:.1f}% APY")

            # If toxic positions are detected, initiate emergency closure
            if toxic_positions:
                print(f"[{self.name}] 🚨 EMERGENCY: {len(toxic_positions)} toxic positions detected. Initiating closures...")

                for toxic_pos in toxic_positions:
                    symbol = toxic_pos['symbol']
                    print(f"[{self.name}] 🧨 INITIATING EMERGENCY CLOSURE: {symbol} (Funding: {toxic_pos['funding_apy']:.1f}% APY)")

                    # Close the position via executor's actuator (Real Exchange Order)
                    try:
                        if hasattr(executor, 'actuator') and executor.actuator:
                            executor.actuator.close_position(symbol)
                            print(f"[{self.name}] ✅ EMERGENCY EXCHANGE CLOSURE INITIATED: {symbol}")
                            
                            # Clean up paper ledger directly if in paper mode
                            if getattr(executor.actuator, 'paper_mode', False):
                                current_price = 0.0
                                if hasattr(executor, 'latest_prices'):
                                    current_price = executor.latest_prices.get(symbol, 0.0)
                                if current_price > 0:
                                    executor.actuator._log_paper_trade(
                                        symbol,
                                        toxic_pos['direction'],
                                        toxic_pos['position_size'],
                                        current_price,
                                        True, # is_exit
                                        'TOXIC_EXIT',
                                        str(time.time())
                                    )
                                
                                # Remove from executor tracking safely
                                if hasattr(executor, 'position_lock'):
                                    with executor.position_lock:
                                        keys_to_delete = [k for k in getattr(executor, 'positions', {}) if getattr(executor.positions[k], 'symbol', '') == symbol]
                                        for k in keys_to_delete:
                                            del executor.positions[k]
                                        if hasattr(executor, 'held_assets'): executor.held_assets.pop(symbol, None)
                                        if hasattr(executor, 'position_metadata'): executor.position_metadata.pop(symbol, None)
                                        if hasattr(executor, 'entry_prices'): executor.entry_prices.pop(symbol, None)
                        
                        # Also sync the Governor immediately so it doesn't think we still have it
                        if gov and hasattr(gov, 'close_position'):
                            gov.close_position(symbol)
                    except Exception as e:
                        print(f"[{self.name}] ❌ EMERGENCY CLOSURE FAILED for {symbol}: {e}")

    def administer_cpr(self, governor):
        """
        Force-Reset Governor State to recover from Drawdown Lock.
        """
        print(f"[{self.name}] ⚡ CLEAR! Administering CPR (State Reset)...")
        
        # 1. Reset High Water Mark to Current (Accept Loss)
        old_hwm = governor.high_water_mark
        governor.high_water_mark = governor.balance
        
        # 2. Unlock Drawdown
        governor.drawdown_lock = False
        
        # 3. Reset State
        governor.state = 'ACTIVE'
        
        # 4. Reset Risk (Safety Mode)
        governor.risk_multiplier = 0.5 # Start slow
        
        # 5. Clear Veto Counters
        if hasattr(governor, 'meta_veto_counter'):
            governor.meta_veto_counter.clear()
            
        print(f"[{self.name}] ✅ PATIENT REVIVED: HWM Reset (${old_hwm:.2f} -> ${governor.balance:.2f}). State: ACTIVE. Risk: 0.5x")

    def receive_message(self, sender, content):
        """
        Handle incoming messages (required by Holon base class).
        """
        # For now, Medic only reacts to perform_triage polls, but we can listen for emergency signals.
        pass
