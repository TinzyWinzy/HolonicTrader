import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from scipy.stats import linregress
from rich.live import Live
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.console import Group
import sys

# Initialize Console for Real Terminal (sys.__stdout__)
# We bypass sys.stdout (QueueLogger) so the Table doesn't spam the GUI/LogFile
console = Console(file=sys.__stdout__, force_terminal=True, width=120)

from HolonicTrader.holon_core import Holon, Disposition, Message
from HolonicTrader.agent_executor import TradeSignal
from performance_tracker import get_performance_data
import config

class TraderHolon(Holon):
    """
    TraderHolon (Supra-Holon)
    The central coordinator that orchestrates the trading lifecycle using a 
    concurrency-first architecture (Phase 28: Warp Velocity).
    """
    
    def __init__(self, name: str = "TraderNexus", sub_holons: Dict[str, Holon] = None):
        super().__init__(name=name, disposition=Disposition(autonomy=0.9, integration=0.9))
        self.sub_holons = sub_holons if sub_holons else {}
        self.market_state = {'price': 0.0, 'regime': 'UNKNOWN', 'entropy': 0.0, 'signal': None}
        self.gui_queue = None
        self.gui_stop_event = None
        self.last_ppo_conviction = 0.5
        self.last_ppo_reward = 0.0

    def register_agent(self, role: str, agent: Holon):
        self.sub_holons[role] = agent
        print(f"[{self.name}] Registered {role}: {agent.name}")

    def perform_health_check(self):
        observer = self.sub_holons.get('observer')
        if observer:
            try:
                status = observer.receive_message(self, {'type': 'GET_STATUS'})
                if not (isinstance(status, dict) and status.get('status') == 'OK'):
                    observer.receive_message(self, {'type': 'FORCE_FETCH'})
            except Exception as e:
                print(f"[{self.name}] ❌ OBSERVER HEALTH FAIL: {e}")

    def run_cycle(self):
        self.perform_health_check()
        interval = getattr(self, '_active_interval', 60)
        print(f"\n[{self.name}] --- Starting Warp Cycle (Interval: {interval}s) ---") 
        
        cycle_report = []
        entropies = []
        cycle_data_cache = {}

        oracle = self.sub_holons.get('oracle')
        observer = self.sub_holons.get('observer')
        executor = self.sub_holons.get('executor')
        governor = self.sub_holons.get('governor')
        ppo = self.sub_holons.get('ppo')
        guardian = self.sub_holons.get('guardian')
        monitor = self.sub_holons.get('monitor')

        # --- PHASE 0: PARALLEL PRE-FLIGHT (GMB Sync) ---
        if oracle and observer:
            def fetch_and_warmup(sym):
                try:
                    data = observer.fetch_market_data(limit=100, symbol=sym)
                    oracle.get_kalman_estimate(sym, data)
                    return sym, data
                except: return sym, None

            with ThreadPoolExecutor(max_workers=config.TRADER_MAX_WORKERS) as t_pool:
                futures = [t_pool.submit(fetch_and_warmup, s) for s in config.ALLOWED_ASSETS]
                try:
                    for f in as_completed(futures, timeout=30):
                        try:
                            sym, d = f.result()
                            if d is not None: cycle_data_cache[sym] = d
                        except Exception as e:
                            print(f"[{self.name}] ⚠️ Data Fetch Error: {e}")
                except TimeoutError:
                    print(f"[{self.name}] ⚠️ Data Fetch Cycle Timed Out (skipping remaining)")
            
            print(f"[{self.name}] 📊 GLOBAL BIAS: {oracle.get_market_bias():.2f}")

        # --- PHASE 1: PARALLEL ANALYSIS PASS ---
        analysis_results = []
        with ThreadPoolExecutor(max_workers=config.TRADER_MAX_WORKERS) as t_pool:
            futures = [t_pool.submit(self._analyze_asset, s, cycle_data_cache.get(s)) for s in config.ALLOWED_ASSETS]
            try:
                for f in as_completed(futures, timeout=30):
                    try:
                        res = f.result()
                        if res: analysis_results.append(res)
                    except Exception as e:
                        print(f"[{self.name}] ⚠️ Analysis Logic Error: {e}")
            except TimeoutError:
                 print(f"[{self.name}] ⚠️ Analysis Cycle Timed Out (proceeding with partial results)")

        analysis_results.sort(key=lambda x: x['symbol'])

        # --- PHASE 2: SEQUENTIAL EXECUTION PASS ---
        for res in analysis_results:
            symbol, data, current_price = res['symbol'], res['data'], res['price']
            row_data, indicators = res['row_data'], res['indicators']
            entropy_val, regime = res['entropy_val'], res['regime']
            
            if entropy_val > 0: entropies.append(entropy_val)

            try:
                if executor: executor.latest_prices[symbol] = current_price
                if executor and governor: 
                    governor.update_balance(executor.get_portfolio_value(current_price))

                # A. Handle Entry
                entry_sig = res.get('entry_signal')
                if entry_sig and executor and governor and oracle:
                    pnl_tracker = get_performance_data()
                    atr_ref = indicators['tr'].rolling(14).mean().rolling(14).mean().iloc[-1]
                    atr_ratio = min(2.0, indicators['atr'] / atr_ref) if atr_ref > 0 else 1.0
                    gov_health = governor.get_portfolio_health()
                    
                    ppo_state = np.array([
                        {'ORDERED': 0.0, 'TRANSITION': 0.5, 'CHAOTIC': 1.0}.get(regime, 0.5),
                        entropy_val, pnl_tracker.get('win_rate', 0.5), atr_ratio, 
                        gov_health['drawdown_pct'], gov_health['margin_utilization']
                    ], dtype=np.float32)

                    conviction = ppo.get_conviction(ppo_state) if ppo else 0.5
                    self.last_ppo_conviction = conviction
                    entry_sig.metadata = {'ppo_state': ppo_state.tolist(), 'ppo_conviction': conviction}

                    approved, safe_qty, leverage = governor.calc_position_size(
                        symbol, current_price, indicators['atr'], atr_ref, conviction
                    )

                    if approved and safe_qty > 0:
                        entry_sig.size = safe_qty
                        decision = executor.decide_trade(entry_sig, regime, entropy_val)
                        if decision.action != 'HALT':
                            executor.execute_transaction(decision, current_price)
                            
                            # Safe Telegram Notification
                            telegram = self.sub_holons.get('telegram')
                            if telegram and hasattr(telegram, 'send_message'):
                                msg = f"🚀 **ENTRY** {symbol}\nPrice: {current_price}\nSize: {entry_sig.size:.4f}"
                                telegram.send_message(msg)
                                
                            row_data['Action'] = f"BUY ({res['metabolism']})"
                    else:
                        row_data['Action'] = "BUY (GOV REJECT)"

                # B. Handle Exit
                guardian_exit = res.get('guardian_exit')
                hard_exit_type = executor.check_stop_loss_take_profit(symbol, current_price) if executor else None
                
                final_exit = None
                reason = "IDLE"
                if hard_exit_type:
                    final_exit = TradeSignal(symbol, 'SELL', 1.0, current_price)
                    reason = hard_exit_type
                elif guardian_exit:
                    final_exit = guardian_exit
                    reason = "Strat"

                if final_exit and executor:
                    # Capture metadata BEFORE execution deletes it!
                    meta = executor.position_metadata.get(symbol, {}).copy()
                    
                    decision = executor.decide_trade(final_exit, regime, entropy_val)
                    pnl_res = executor.execute_transaction(decision, current_price)
                    
                    if pnl_res is not None:
                        if guardian: guardian.record_exit(symbol, time.time())
                        if ppo and meta:
                            if meta.get('ppo_state') is not None and meta.get('ppo_conviction') is not None:
                                # --- PPO REWARD REFACTOR (Hybrid Velocity/Pain) ---
                                # Previous: reward = pnl_res - (drawdown * 2.0) [Death Spiral]
                                # New: Asymmetric Loss Aversion + Time Decay Efficiency
                                
                                pnl_pct = pnl_res # pnl_res is percentage
                                is_win = pnl_pct > 0
                                
                                # 1. Base Utility
                                # Scale small % (0.01) to recognizable scalar (0.1)
                                reward = pnl_pct * 10.0
                                
                                # 2. Asymmetric Punishment (Loss Aversion)
                                if not is_win:
                                    reward *= 2.5 # Pain Factor
                                    
                                # 3. Time Duration
                                from datetime import datetime
                                entry_ts_iso = meta.get('entry_timestamp')
                                duration_mins = 1.0 # Default minimum
                                if entry_ts_iso:
                                    try:
                                        # Handle ISO parsing manually if needed, but pd.to_datetime is robust
                                        t_entry = pd.to_datetime(entry_ts_iso)
                                        t_exit = pd.Timestamp.now(tz='UTC')
                                        duration_mins = (t_exit - t_entry).total_seconds() / 60.0
                                    except: pass
                                    
                                duration_mins = max(1.0, duration_mins)
                                
                                # 4. Time Decay / Velocity
                                # log1p(duration) -> log(2) ~ 0.69, log(61) ~ 4.1
                                time_factor = np.log1p(duration_mins)
                                if time_factor < 1.0: time_factor = 1.0 # Floor at 1
                                
                                if is_win:
                                    reward = reward / time_factor # Fast wins > Slow wins
                                else:
                                    reward = reward * time_factor # Long losses > Fast losses (Double Pain)
                                    
                                self.last_ppo_reward = reward
                                print(f"[{self.name}] 🧠 PPO REWARD: {reward:.4f} (PnL {pnl_pct*100:.2f}%, {duration_mins:.0f}m)")
                                
                                # Convert list back to numpy if needed
                                state = np.array(meta['ppo_state']) 
                                ppo.remember(state, meta['ppo_conviction'], reward, 0.0, 0.0, True)

                        # Safe Telegram Notification
                        telegram = self.sub_holons.get('telegram')
                        if telegram and hasattr(telegram, 'send_message'):
                            msg = f"📉 **EXIT** {symbol}\nPrice: {current_price}\nPnL: {pnl_res*100:+.2f}% ({reason})"
                            telegram.send_message(msg)

                        row_data['Action'] = f"SELL ({reason})"

            except Exception as e:
                print(f"[{self.name}] ❌ Error processing {symbol}: {e}")

            cycle_report.append(row_data)

        # --- PHASE 3: AGGREGATE & UI ---
        if entropies and self.sub_holons.get('entropy'):
            avg_e = sum(entropies) / len(entropies)
            self.market_state['entropy'] = avg_e
            self.market_state['regime'] = self.sub_holons['entropy'].determine_regime(avg_e)

        # Removed redundant _print_summary call
        if monitor and executor: monitor.update_health(executor.get_portfolio_value(0.0), get_performance_data())
        self.publish_agent_status()
        return cycle_report

    def _analyze_asset(self, symbol: str, data: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        observer = self.sub_holons.get('observer')
        if data is None and observer:
            try: data = observer.fetch_market_data(limit=100, symbol=symbol)
            except: return None
        if data is None: return None

        row_data = {'Symbol': symbol, 'Price': f"{data['close'].iloc[-1]:.4f}", 'Regime': '?', 'Action': 'HOLD', 'PnL': '-', 'Note': ''}
        current_price = data['close'].iloc[-1]
        
        entropy_agent, oracle = self.sub_holons.get('entropy'), self.sub_holons.get('oracle')
        guardian, governor = self.sub_holons.get('guardian'), self.sub_holons.get('governor')
        executor = self.sub_holons.get('executor')
        
        if entropy_agent:
            entropy_val = entropy_agent.calculate_shannon_entropy(data['returns'])
            regime = entropy_agent.determine_regime(entropy_val)
            row_data['Entropy'], row_data['Regime'] = f"{entropy_val:.3f}", regime

        # Indicators
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        row_data['RSI'] = f"{(100 - (100 / (1 + (gain / loss))).iloc[-1]):.1f}"

        rolling_mean, rolling_std = data['close'].rolling(20).mean(), data['close'].rolling(20).std()
        bb_vals = {'upper': (rolling_mean + 2*rolling_std).iloc[-1], 'middle': rolling_mean.iloc[-1], 'lower': (rolling_mean - 2*rolling_std).iloc[-1]}
        
        tr = pd.concat([(data['high']-data['low']), (data['high']-data['close'].shift()).abs(), (data['low']-data['close'].shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        obv = (np.sign(data['close'].diff()).fillna(0) * data['volume']).cumsum()
        obv_slope, _, _, _, _ = linregress(np.arange(14), obv.iloc[-14:].values)

        metabolism = 'PREDATOR' if executor and executor.get_portfolio_value(current_price) > config.SCAVENGER_THRESHOLD else 'SCAVENGER'
        
        entry_sig = None
        if (not governor or governor.is_trade_allowed(symbol, current_price)) and oracle:
            last_exit = guardian.last_exit_times.get(symbol) if guardian else None
            if not (last_exit and (time.time() - last_exit) < (config.STRATEGY_POST_EXIT_COOLDOWN_CANDLES * 3600)):
                entry_sig = oracle.analyze_for_entry(symbol, data, bb_vals, obv_slope, metabolism)

        guardian_exit = None
        entry_p = executor.entry_prices.get(symbol, 0.0) if executor else 0.0
        if entry_p > 0 and guardian:
            direction = executor.position_metadata.get(symbol, {}).get('direction', 'BUY')
            age_h = 0.0
            if executor.entry_timestamps.get(symbol):
                from datetime import datetime, timezone
                try: age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(executor.entry_timestamps[symbol])).total_seconds() / 3600
                except: pass
            guardian_exit = guardian.analyze_for_exit(symbol, current_price, entry_p, bb_vals, atr, metabolism, age_h, direction)
            pnl_pct = (current_price - entry_p) / entry_p if direction == 'BUY' else (entry_p - current_price) / entry_p
            row_data['PnL'] = f"{pnl_pct*100:+.2f}%"

        # Enrichment for Dashboard
        probes = oracle.last_probes.get(symbol, {'lstm': 0.5, 'xgb': 0.5}) if oracle else {'lstm': 0.5, 'xgb': 0.5}
        row_data['LSTM'] = f"{probes['lstm']:.2f}"
        row_data['XGB'] = f"{probes['xgb']:.2f}"

        return {
            'symbol': symbol, 'data': data, 'price': current_price, 'row_data': row_data,
            'entropy_val': entropy_val, 'regime': regime, 'metabolism': metabolism,
            'entry_signal': entry_sig, 'guardian_exit': guardian_exit,
            'indicators': {'bb_vals': bb_vals, 'obv_slope': obv_slope, 'atr': atr, 'tr': tr}
        }

    def _create_summary_layout(self, cycle_report: List[Dict]):
        oracle = self.sub_holons.get('oracle')
        bias = oracle.get_market_bias() if oracle else 0.5
        
        # 1. Market Status Panel
        bias_color = "green" if bias >= config.GMB_THRESHOLD else ("yellow" if bias >= 0.4 else "red")
        status_text = f"[bold {bias_color}]GLOBAL MARKET BIAS: {bias:.2f}[/bold {bias_color}] | " \
                      f"Status: [{'bold green' if bias >= config.GMB_THRESHOLD else 'bold red'}]" \
                      f"{'BULLISH' if bias >= config.GMB_THRESHOLD else 'CAUTIOUS'}[/]"
        
        header = Panel(status_text, title=f"[{self.name}] Live Dashboard", border_style="blue", expand=False)

        # 2. Detail Table
        table = Table(title="Asset Register", box=box.SIMPLE_HEAD, show_lines=False)
        table.add_column("Symbol", style="cyan", no_wrap=True)
        table.add_column("Price", style="white")
        table.add_column("Regime", style="magenta")
        table.add_column("Entropy", justify="right")
        table.add_column("Brains (LSTM/XGB)", justify="center")
        table.add_column("Action", style="bold")
        table.add_column("PnL", justify="right", style="green")

        for row in cycle_report:
            probes = oracle.last_probes.get(row['Symbol'], {'lstm': 0.5, 'xgb': 0.5}) if oracle else {'lstm': 0.5, 'xgb': 0.5}
            
            # Colorize Action
            action = row['Action']
            act_style = "dim"
            if "BUY" in action: act_style = "bold green"
            elif "SELL" in action: act_style = "bold red"
            elif "HOLD" in action: act_style = "dim white"
            
            # Colorize Regime
            reg_style = "white"
            if row['Regime'] == 'CHAOTIC': reg_style = "red"
            elif row['Regime'] == 'ORDERED': reg_style = "green"
            
            table.add_row(
                row['Symbol'],
                row['Price'],
                f"[{reg_style}]{row['Regime']}[/{reg_style}]",
                row.get('Entropy', 'N/A'),
                f"{probes['lstm']:.2f} / {probes['xgb']:.2f}",
                f"[{act_style}]{action}[/{act_style}]",
                row['PnL']
            )
            
        # Combine into group (or just return a group/layout)
        from rich.console import Group
        return Group(header, table)

    def publish_agent_status(self):
        if not self.gui_queue: return
        gov, executor = self.sub_holons.get('governor'), self.sub_holons.get('executor')
        oracle = self.sub_holons.get('oracle')
        perf = get_performance_data()
        # Real-time Valuation for Asset Allocation
        latest_prices = executor.latest_prices if executor else {}
        holdings = {'CASH': gov.balance if gov else 0.0}
        total_exp = 0.0
        
        if gov:
            for s, p in gov.positions.items():
                curr_p = latest_prices.get(s, p['entry_price'])
                val = p['quantity'] * curr_p
                holdings[s] = val
                total_exp += val

        portfolio_val = executor.get_portfolio_value(0.0) if executor else 1.0
        
        self.gui_queue.put({
            'type': 'agent_status',
            'data': {
                'gov_state': f"{gov.get_metabolism_state() if gov else 'OFFLINE'}",
                'gov_alloc': f"{config.GOVERNOR_MAX_MARGIN_PCT*100:.1f}%",
                'gov_lev': f"{config.PREDATOR_LEVERAGE}x",
                'gov_trends': str(len(gov.positions)) if gov else "0",
                'regime': self.market_state['regime'],
                'entropy': f"{self.market_state['entropy']:.4f}",
                'strat_model': 'Warp-V4 (Hybrid)',
                'kalman_active': 'True' if oracle and oracle.kalman_filters else 'False',
                'ppo_conv': f"{self.last_ppo_conviction:.2f}",
                'ppo_reward': f"{self.last_ppo_reward:.2f}",
                'lstm_prob': f"{oracle.get_health().get('last_lstm', 0.5):.2f}",
                'xgb_prob': f"{oracle.get_health().get('last_xgb', 0.5):.2f}",
                'last_order': executor.last_order_details if executor else 'NONE',
                'win_rate': f"{perf.get('win_rate', 0.0):.1f}%",
                'pnl': f"${perf.get('total_pnl', 0.0):.2f}",
                'omega': f"{perf.get('omega_ratio', 0.0):.2f}",
                'exposure': f"${total_exp:.2f}",
                'margin': f"${executor.get_execution_summary()['margin_used']:.2f}" if executor else "$0.00",
                'actual_lev': f"{total_exp/portfolio_val:.2f}x",
                'holdings': holdings
            }
        })

    def start_live_loop(self, interval_seconds: int = 60):
        self._active_interval = interval_seconds
        
        # User requested to remove terminal table and reduce noise
        # with Live(console=console, screen=False, refresh_per_second=4) as live:
            
        while True:
            if self.gui_stop_event and self.gui_stop_event.is_set(): break
            start = time.time()
            try: 
                # Reduced Log Noise: Commented out cycle start print
                # print(f"\n[{self.name}] --- Starting Warp Cycle (Interval: {interval_seconds}s) ---") 
                
                report = self.run_cycle()
                
                if self.gui_queue: self.gui_queue.put({'type': 'summary', 'data': report})
                
                # Disable Terminal Table Update
                # layout = self._create_summary_layout(report)
                # live.update(layout)
                
            except Exception as e:
                print(f"[{self.name}] ☠️ Cycle Error: {e}")
                time.sleep(30)
            
            wait = max(0, interval_seconds - (time.time() - start))
            for _ in range(int(wait * 2)):
                if self.gui_stop_event and self.gui_stop_event.is_set(): break
                time.sleep(0.5)

    def receive_message(self, sender, content): pass
    def _adapt_to_regime(self, regime): pass
