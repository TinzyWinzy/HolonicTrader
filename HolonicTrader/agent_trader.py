import time
import pandas as pd
from typing import Dict, Any, Optional
from HolonicTrader.holon_core import Holon, Disposition, Message
from performance_tracker import get_performance_data
import config

class TraderHolon(Holon):
    """
    TraderHolon (Supra-Holon)
    
    The central coordinator that orchestrates the trading lifecycle.
    It manages a set of sub-holons (Observer, Entropy, Strategy, Governor, Executor)
    and routes messages between them to execute the trading loop.
    """
    
    def __init__(self, name: str = "TraderNexus", sub_holons: Dict[str, Holon] = None):
        # High integration (orchestrator), High autonomy (decision maker)
        super().__init__(name=name, disposition=Disposition(autonomy=0.9, integration=0.9))
        
        self.sub_holons = sub_holons if sub_holons else {}
        self.market_state = {
            'price': 0.0,
            'regime': 'UNKNOWN',
            'entropy': 0.0,
            'signal': None
        }
        
        # GUI integration
        self.gui_queue = None
        self.gui_stop_event = None

    def register_agent(self, role: str, agent: Holon):
        """Register a sub-holon with a specific role."""
        self.sub_holons[role] = agent
        print(f"[{self.name}] Registered {role}: {agent.name}")

    def perform_health_check(self):
        """
        IMMUNE SYSTEM: Ping sub-holons to ensure system health.
        """
        # 1. Check Observer
        observer = self.sub_holons.get('observer')
        if observer:
            try:
                status = observer.receive_message(self, {'type': 'GET_STATUS'})
                if isinstance(status, dict) and status.get('status') == 'OK':
                     pass # All good
                else: 
                     if status:
                         print(f"[{self.name}] ⚠️ OBSERVER HEALTH WARN: {status}")
                     # Self-Healing: Force Fetch
                     observer.receive_message(self, {'type': 'FORCE_FETCH'})
            except Exception as e:
                print(f"[{self.name}] ❌ OBSERVER HEALTH FAIL: {e}")

    def run_cycle(self):
        """
        Execute one full trading cycle (Heartbeat).
        Iterates through all ALLOWED_ASSETS defined in config.
        """
        import config # Lazy import to avoid circular dep if any
        # print(f"\n[{self.name}] --- Starting Multi-Asset Cycle ---") # Silence excessive log
        
        # IMMUNE SYSTEM CHECK
        self.perform_health_check()
        
        cycle_report = [] # List to hold row data for summary table
        entropies = [] # List to hold entropy values from all assets
        
        # 1. ORCHESTRATION LOOP
        for symbol in config.ALLOWED_ASSETS:
            # Initialize Report Row
            row_data = {
                'Symbol': symbol,
                'Price': 'ERR',
                'Regime': '?',
                'Action': 'HOLD',
                'PnL': '-',
                'Note': ''
            }

            # A. FETCH DATA
            observer = self.sub_holons.get('observer')
            if not observer:
                print(f"[{self.name}] CRITICAL: No Observer attached.")
                continue

            try:
                # Fetch Data Specific to Asset
                data = observer.fetch_market_data(limit=100, symbol=symbol)
                current_price = data['close'].iloc[-1]
                row_data['Price'] = f"{current_price:.4f}"
                
                # Update Executor's price view immediately
                executor = self.sub_holons.get('executor')
                if executor:
                    executor.latest_prices[symbol] = current_price
            except Exception as e:
                # print(f"[{self.name}] Error fetching {symbol}: {e}") # Silence error slightly or log shorter
                row_data['Note'] = "Data Fetch Err"
                cycle_report.append(row_data)
                continue

            # B. ENTROPY & REGIME
            entropy_agent = self.sub_holons.get('entropy')
            regime = 'UNKNOWN'
            entropy_val = 0.0
            
            if entropy_agent:
                try:
                    returns = data['returns']
                    entropy_val = entropy_agent.calculate_shannon_entropy(returns)
                    entropies.append(entropy_val) # Collect for aggregation
                    row_data['Entropy'] = f"{entropy_val:.3f}"
                    regime = entropy_agent.determine_regime(entropy_val)
                    row_data['Regime'] = regime
                except Exception:
                    row_data['Entropy'] = "ERR"
                    pass

            # C. STRATEGY (Signal Generation)
            strategy = self.sub_holons.get('strategy')
            signal = None
            
            if strategy:
                try:
                    # Calculate OBV & Technicals
                    obv_series = strategy.calculate_obv(data)
                    obv_slope = strategy.calculate_obv_slope(obv_series)
                    
                    # Store OBV Trend in Note if significant? 
                    # row_data['Note'] = f"OBV:{obv_slope:.1f}"

                    # Calculate BB (20, 2)
                    rolling_mean = data['close'].rolling(window=20).mean()
                    rolling_std = data['close'].rolling(window=20).std()
                    bb = {
                        'upper': rolling_mean + (2 * rolling_std),
                        'middle': rolling_mean.iloc[-1],
                        'lower': rolling_mean - (2 * rolling_std)
                    }
                    bb_vals = {k: (v.iloc[-1] if hasattr(v, 'iloc') else v) for k,v in bb.items()}
                    
                    # Calculate ATR (14)
                    high_low = data['high'] - data['low']
                    high_close = (data['high'] - data['close'].shift()).abs()
                    low_close = (data['low'] - data['close'].shift()).abs()
                    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    atr = tr.rolling(window=14).mean().iloc[-1]
                    
                    # Calculate RSI
                    entry_rsi = strategy.calculate_rsi(data['close'])
                    row_data['RSI'] = f"{entry_rsi:.1f}"

                    # Fetch Stack Count from Governor
                    stack_count = 0
                    governor = self.sub_holons.get('governor')
                    if governor:
                         stack_count = governor.positions.get(symbol, {}).get('stack_count', 0)
                    row_data['Stack'] = str(stack_count)

                    # E. METABOLISM & SIGNAL
                    metabolism = 'SCAVENGER' # Default
                    current_capital = config.INITIAL_CAPITAL
                    
                    if executor:
                        current_capital = executor.get_portfolio_value(current_price)
                        if current_capital > config.SCAVENGER_THRESHOLD:
                            metabolism = 'PREDATOR'
                    
                    # Generate Entry Signal
                    # PRE-CHECK: Skip if Governor will reject
                    is_allowed = True
                    if governor:
                         is_allowed = governor.is_trade_allowed(symbol, current_price)
                    
                    entry_signal = None
                    if is_allowed:
                        entry_signal = strategy.analyze_for_entry(
                            symbol=symbol,
                            window_data=data,
                            bb=bb_vals,
                            obv_slope=obv_slope,
                            metabolism_state=metabolism
                        )
                    # else:
                    #     if row_data.get('Stack') not in ['0', 'ERR']: 
                    #        row_data['Note'] = "Gov Block" # Optional UI feedback
                    
                    if entry_signal:
                        # Log immediately
                        print(f"[{self.name}] ⚡ SIGNAL: {entry_signal.direction} {symbol} @ {entry_signal.price:.2f}")
                        
                        if executor:
                            # 1. DQN CONTEXT
                            dqn = self.sub_holons.get('dqn')
                            rl_state = []
                            dqn_action = "HOLD"
                            dqn_idx = 1 # Halt default
                            
                            if dqn:
                                # State: [Entropy, RSI, Volatility(ATR), Bias]
                                # We need ATR. Strategy has calculate_atr usually, or we approx
                                # For now: [Entropy, RSI, 0.0, 1.0]
                                rl_state = [entropy_val, entry_rsi, 0.0, 1.0] 
                                dqn_action = dqn.get_action(rl_state)
                                dqn_idx = dqn.get_action_index(dqn_action)
                                print(f"[{self.name}] 🧠 DQN Suggests: {dqn_action} (Epsilon: {dqn.epsilon:.2f})")
                            
                            # 2. DECISION & EXECUTION
                            decision = executor.decide_trade(entry_signal, regime, entropy_val)
                            
                            if decision.action != 'HALT':
                                strategy.update_reputation(0.01)
                                executor.execute_transaction(decision, current_price)
                                
                                # 3. METADATA INJECTION (For RL Training later)
                                if dqn and symbol in executor.position_metadata:
                                    executor.position_metadata[symbol]['rl_state'] = rl_state
                                    executor.position_metadata[symbol]['rl_action_idx'] = 0 # Assume we EXECUTED (idx 0) since we are here
                                    # Note: If we followed DQN 'HALT', we wouldn't be here. 
                                    # Ideally we train on 'refusal' too, but for now we train on 'did we make money?'
                                
                                row_data['Action'] = f"BUY ({metabolism})"
                            else:
                                row_data['Action'] = "BUY (HALT)"
                    
                    # F. EXIT MANAGEMENT & REPORTING
                    if executor:
                        # 1. Strategy Analysis (Exit v2)
                        entry_p = executor.entry_prices.get(symbol, 0.0)
                        
                        if entry_p > 0:
                            # Add PnL to table
                            pnl_pct = (current_price - entry_p) / entry_p
                            row_data['PnL'] = f"{pnl_pct*100:+.2f}%"
                            
                            # Add Proximity Note
                            if metabolism == 'SCAVENGER':
                                scalptp_dist = ((entry_p * (1+config.SCAVENGER_SCALP_TP)) - current_price) / current_price
                                row_data['Note'] = f"TP: {scalptp_dist*100:+.1f}%"
                            else:
                                predtp_dist = ((entry_p * (1+config.PREDATOR_TAKE_PROFIT)) - current_price) / current_price
                                row_data['Note'] = f"BigTP: {predtp_dist*100:+.1f}%"

                            # Calculate position age
                            from datetime import datetime, timezone
                            entry_timestamp = executor.entry_timestamps.get(symbol)
                            position_age_hours = 0.0
                            if entry_timestamp:
                                try:
                                    entry_dt = datetime.fromisoformat(entry_timestamp)
                                    age_seconds = (datetime.now(timezone.utc) - entry_dt).total_seconds()
                                    position_age_hours = age_seconds / 3600
                                except Exception:
                                    position_age_hours = 0.0

                            strategy_exit = strategy.analyze_for_exit(
                                symbol=symbol,
                                current_price=current_price,
                                entry_price=entry_p,
                                bb=bb_vals,
                                atr=atr,
                                metabolism_state=metabolism,
                                entry_timestamp=entry_timestamp,
                                position_age_hours=position_age_hours
                            )
                            if strategy_exit:
                                print(f"[{self.name}] 🧠 STRATEGY EXIT for {symbol}: {strategy_exit.direction}")
                                decision = executor.decide_trade(strategy_exit, regime, entropy_val)
                                pnl_res = executor.execute_transaction(decision, current_price)
                                
                                if pnl_res is not None:
                                    strategy.update_reputation(pnl_res * 10.0)
                                    print(f"[{self.name}] Reputation Updated: {strategy.reputation:.3f}")
                                    strategy.record_exit(symbol, data.index[-1])
                                    
                                    # DQN TRAIN
                                    dqn = self.sub_holons.get('dqn')
                                    if dqn:
                                        meta = executor.position_metadata.get(symbol, {})
                                        entry_state = meta.get('rl_state')
                                        entry_action_idx = meta.get('rl_action_idx')
                                        if entry_state and entry_action_idx is not None:
                                            # Risk-Adjusted Reward: Sortino/Sharpe Proxy
                                            # Reward = PnL / (Volatility + epsilon)
                                            # We use normalized ATR as volatility measure
                                            atr_pct = (atr / entry_p) if entry_p > 0 else 0.01
                                            risk_adj_reward = (pnl_res * 10.0) / max(0.001, atr_pct)
                                            
                                            # Clip reward to avoid exploding gradients [-10, 10]
                                            risk_adj_reward = max(-10.0, min(10.0, risk_adj_reward))
                                            
                                            current_state = [entropy_val, entry_rsi, 0.0, 1.0]
                                            dqn.remember(entry_state, entry_action_idx, risk_adj_reward, current_state, done=True)
                                            loss = dqn.replay()
                                            current_state = [entropy_val, entry_rsi, 0.0, 1.0]
                                            dqn.remember(entry_state, entry_action_idx, risk_adj_reward, current_state, done=True)
                                            loss = dqn.replay()
                                            
                                            # IMP_003: Persist Experience
                                            if executor and executor.db_manager:
                                                executor.db_manager.save_experience({
                                                    'symbol': symbol,
                                                    'state': entry_state,
                                                    'action_idx': entry_action_idx,
                                                    'reward': risk_adj_reward,
                                                    'next_state': current_state,
                                                    'done': True
                                                })
                                                
                                            # IMP_003: Persist Experience (Strategy Exit)
                                            if executor and executor.db_manager:
                                                executor.db_manager.save_experience({
                                                    'symbol': symbol,
                                                    'state': entry_state,
                                                    'action_idx': entry_action_idx,
                                                    'reward': risk_adj_reward,
                                                    'next_state': current_state,
                                                    'done': True
                                                })
                                                
                                            print(f"[{self.name}] 🧠 DQN TRAINED on {symbol}: PnL={pnl_res*100:.2f}%, Vol={atr_pct*100:.2f}%, Reward={risk_adj_reward:.4f}, Loss={loss:.6f}")
                                
                                row_data['Action'] = "SELL (Strat)"
                                cycle_report.append(row_data)
                                continue # Position closed
                        
                        # 2. Hard Guardrails (SL/TP in Executor)
                        exit_type = executor.check_stop_loss_take_profit(symbol, current_price)
                        if exit_type:
                            print(f"[{self.name}] 🚨 HARD EXIT: {symbol} - {exit_type}")
                            from .agent_executor import TradeSignal as TS
                            exit_signal = TS(symbol=symbol, direction='SELL', size=1.0, price=current_price)
                            
                            decision = executor.decide_trade(exit_signal, regime, entropy_val)
                            pnl_res = executor.execute_transaction(decision, current_price)
                            
                            # DQN TRAIN (HARD EXIT)
                            if pnl_res is not None:
                                strategy.update_reputation(pnl_res * 10.0)
                                dqn = self.sub_holons.get('dqn')
                                if dqn:
                                    # Retrieve Entry State
                                    meta = executor.position_metadata.get(symbol, {})
                                    entry_state = meta.get('rl_state')
                                    entry_action_idx = meta.get('rl_action_idx')
                                    
                                    if entry_state and entry_action_idx is not None:
                                        # Construct Reward (PnL %)
                                        # Risk-Adjusted
                                        atr_pct = (atr / entry_p) if entry_p > 0 else 0.01
                                        risk_adj_reward = (pnl_res * 10.0) / max(0.001, atr_pct)
                                        # Clip
                                        risk_adj_reward = max(-10.0, min(10.0, risk_adj_reward))
                                        
                                        # Current State
                                        current_state = [entropy_val, entry_rsi, 0.0, 1.0] # approx
                                        
                                        dqn.remember(entry_state, entry_action_idx, risk_adj_reward, current_state, done=True)
                                        loss = dqn.replay()
                                        # IMP_003: Persist Experience (Hard Exit)
                                        if executor and executor.db_manager:
                                            executor.db_manager.save_experience({
                                                'symbol': symbol,
                                                'state': entry_state,
                                                'action_idx': entry_action_idx,
                                                'reward': risk_adj_reward,
                                                'next_state': current_state,
                                                'done': True
                                            })
                                        
                                        print(f"[{self.name}] 🧠 DQN TRAINED on {symbol}: PnL={pnl_res*100:.2f}%, Vol={atr_pct*100:.2f}%, Reward={risk_adj_reward:.4f}, Loss={loss:.6f}")

                            row_data['Action'] = f"SELL ({exit_type})"

                except Exception as e:
                     print(f"[{self.name}] Error in Strategy analysis for {symbol}: {e}")
                     # import traceback
                     # traceback.print_exc() 
            
            cycle_report.append(row_data)

        # AGGREGATE MARKET STATE
        avg_entropy = 0.0
        if entropies and self.sub_holons.get('entropy'):
            avg_entropy = sum(entropies) / len(entropies)
            self.market_state['entropy'] = avg_entropy
            # Determine global regime based on average (or max?) - using Average for now
            self.market_state['regime'] = self.sub_holons['entropy'].determine_regime(avg_entropy)

        # PRINT SUMMARY TABLE
        # -------------------
        print("-" * 95)
        print(f"{'SYMBOL':<10} {'PRICE':<12} {'REGIME':<10} {'ENTROPY':<8} {'ACTION':<15} {'PNL':<10} {'NOTE':<20}")
        print("-" * 95)
        for row in cycle_report:
            entropy_str = row.get('Entropy', 'N/A')
            print(f"{row['Symbol']:<10} {row['Price']:<12} {row['Regime']:<10} {entropy_str:<8} {row['Action']:<15} {row['PnL']:<10} {row['Note']}")
        print("-" * 95)
        
        # PUBLISH AGENT STATUS
        self.publish_agent_status()
        
        return cycle_report 

    def publish_agent_status(self):
        """Send specific agent metrics to the GUI."""
        if not self.gui_queue: return
        
        # 1. Governor Data
        gov = self.sub_holons.get('governor')
        gov_data = {}
        if gov:
            # Calculate total allocation usage? For now just static config or dynamic limits
            # Showing Config values for now as 'State'
            gov_data = {
                'state': 'PRE-CHECK ACTIVE',
                'alloc': f"{config.GOVERNOR_MAX_MARGIN_PCT*100:.1f}%",
                'lev': f"{config.PREDATOR_LEVERAGE}x"
            }
            
        # 2. Brain Data
        regime = self.market_state.get('regime', 'UNKNOWN')
        entropy = self.market_state.get('entropy', 0.0)
        
        # 3. Actuator Data (Last Order from Ledger)
        last_action = "IDLE"
        executor = self.sub_holons.get('executor')
        if executor:
            ledger_sum = executor.get_ledger_summary()
            last_blk = ledger_sum.get('last_block')
            if last_blk:
                # Format: "EXECUTE (15:30)" or "HALT"
                ts = last_blk.get('timestamp', '')[11:16] # Extract time HH:MM
                act = last_blk.get('action', 'NONE')
                last_action = f"{act} @ {ts}"
        
        # 4. Global Performance Data
        perf_data = get_performance_data()
        
        # 5. Brain Health Checks
        strat_health = self.sub_holons['strategy'].get_health() if self.sub_holons.get('strategy') else {}
        dqn_health = self.sub_holons['dqn'].get_health() if self.sub_holons.get('dqn') else {}
        
        msg = {
            'type': 'agent_status',
            'data': {
                'gov_state': gov_data.get('state', 'OFFLINE'),
                'gov_alloc': gov_data.get('alloc', '-'),
                'gov_lev': gov_data.get('lev', '-'),
                'gov_trends': f"{len(gov.positions) if gov else 0}",
                'regime': regime,
                'entropy': f"{entropy:.4f}",
                'last_order': last_action,
                'win_rate': f"{perf_data.get('win_rate', 0.0):.1f}%",
                'omega': f"{perf_data.get('omega_ratio', 0.0):.2f}",
                # Brain Health
                'strat_model': strat_health.get('model', 'N/A'),
                'kalman_active': f"{strat_health.get('kalman_count', 0)}",
                'dqn_epsilon': dqn_health.get('epsilon', '-'),
                'dqn_mem': f"{dqn_health.get('memory', 0)}",
                # Holdings for Pie Chart
                'holdings': self._get_holdings_breakdown()
            }
        }
        self.gui_queue.put(msg)

    def _get_holdings_breakdown(self):
        """Helper to get {Symbol: USD_Value} for GUI."""
        gov = self.sub_holons.get('governor')
        if not gov: return {}
        
        breakdown = {'CASH': gov.balance}
        for sym, pos in gov.positions.items():
            # Estimate value using entry price if current price not avail inside Governor directly easily
            # But Governor tracks `last_specific_entry` or `positions` entry price.
            # To be accurate we need current price, but entry price * qty is decent approx for allocation view.
            val = pos['quantity'] * pos['entry_price']
            breakdown[sym] = val
        return breakdown

    def _adapt_to_regime(self, regime: str):
        """Alter disposition based on market regime."""
        if regime == 'CHAOTIC':
            # High Integration (Safety in numbers), Low Autonomy (Don't go rogue)
            self.disposition.integration = 0.9
            self.disposition.autonomy = 0.1
        elif regime == 'ORDERED':
            # Low Integration, High Autonomy (Aggressive)
            self.disposition.integration = 0.2
            self.disposition.autonomy = 0.9
        else: # TRANSITION
            self.disposition.integration = 0.5
            self.disposition.autonomy = 0.5

    def start_live_loop(self, interval_seconds: int = 60):
        """
        Start the infinite live execution loop.
        """
        # LOAD EXPERIENCE MEMORY
        if self.sub_holons.get('dqn') and self.sub_holons.get('executor'):
            try:
                db = self.sub_holons['executor'].db_manager
                exps = db.get_experiences(limit=2000)
                if exps:
                    print(f"[{self.name}] 🧠 Loading {len(exps)} partial experiences from DB to Brain...")
                    for e in exps:
                        self.sub_holons['dqn'].remember(e['state'], e['action_idx'], e['reward'], e['next_state'], e['done'])
            except Exception as e:
                print(f"[{self.name}] Failed to load RL experiences: {e}")

        print(f"[{self.name}] 🚀 Starting LIVE Loop (Interval: {interval_seconds}s)")
        print(f"[{self.name}] Press Ctrl+C to stop.")
        
        try:
            while True:
                start_time = time.time()
                
                cycle_report = self.run_cycle()
                
                elapsed = time.time() - start_time
                sleep_time = max(0, interval_seconds - elapsed)
                
                if self.gui_queue:
                    try:
                        self.gui_queue.put({
                            'type': 'summary',
                            'data': cycle_report
                        })
                    except Exception:
                        pass
                        
                elapsed = time.time() - start_time
                sleep_time = max(0, interval_seconds - elapsed)
                
                # HEADLESS OR THREADED SLEEP
                # We need to sleep in chunks to responsive to stop_event
                chunks = int(sleep_time / 0.5)
                for _ in range(chunks):
                    if self.gui_stop_event and self.gui_stop_event.is_set():
                        print(f"[{self.name}] ⏹ Stop Signal Received.")
                        return # Exit Loop
                    time.sleep(0.5)
                
                # Sleep remainder
                rem = sleep_time % 0.5
                if rem > 0:
                    time.sleep(rem)
                
                if sleep_time <= 0:
                    print(f"[{self.name}] ⚠️ Cycle took {elapsed:.1f}s (> {interval_seconds}s interval). Skipping sleep.")
                
                # Check stop event at end of loop too
                if self.gui_stop_event and self.gui_stop_event.is_set():
                    print(f"[{self.name}] ⏹ Stop Signal Received.")
                    return
                    
        except KeyboardInterrupt:
            print(f"\n[{self.name}] 🛑 Stopped by User.")
        except Exception as e:
            print(f"[{self.name}] ☠️ CRITICAL LOOP ERROR: {e}")
            print(f"[{self.name}] Self-Healing: Sleeping for 30s before restarting cycle...")
            time.sleep(30)
            # The loop continues because we are inside the 'while True' but wait...
            # The try-except is OUTSIDE the while loop in the original code? 
            # Let's check lines 266-267.
            # line 266: try:
            # line 267:     while True:
            # So if exception happens inside, it breaks the loop and goes to line 283.
            # To make it persistent, I must restart the loop.
            # Recursive call? Or just wrap the try block properly?
            # Better checking the structure.
            self.start_live_loop(interval_seconds)

    def receive_message(self, sender: Any, content: Any) -> None:
        """Handle incoming messages (reports from agents)."""
        if isinstance(content, Message):
            print(f"[{self.name}] Received {content.type} from {content.sender}")
            if content.type == 'ALERT':
                print(f"[{self.name}] ⚠️ ALERT: {content.payload}")
        else:
            # Legacy support
            pass
