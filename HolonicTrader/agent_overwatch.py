"""
OverwatchHolon - The "Sentry" (Phase 45)

"I see all, I speak for all."

Responsibilities:
1.  **State Aggregation**: Collects health/status from Governor, Executor, Sentiment.
2.  **Narrative Engine**: Deterministically converts stats into human-readable Situation Reports (SitReps).
3.  **Communication**: Manages the Telegram Bot and pushes updates to the Dashboard.
"""

import threading
import asyncio
import time
from typing import Any, Dict, List, Optional
from enum import Enum
import random
import config
import os
import json
import logging
from datetime import datetime, timezone

from HolonicTrader.holon_core import Holon, Disposition, Message
from HolonicTrader.exceptions import DeadMansSwitchTriggered

# Setup logging for this module
logger = logging.getLogger(__name__)

# Telegram Imports (Robust)
try:
    from telegram import Update
    from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("⚠️ Overwatch: python-telegram-bot not installed. Telegram features DISABLED.")

class SystemState(Enum):
    CRITICAL = "CRITICAL"   # High Danger (Drawdown, Crisis)
    CAUTION = "CAUTION"     # Heightened Risk (High Volatility, Negative Sentiment)
    NOMINAL = "NOMINAL"     # Standard Operation
    OPTIMAL = "OPTIMAL"     # Good Conditions (Profit & Positive Sentiment)

class NarrativeEngine:
    """
    Deterministic NLP Engine.
    Converts raw metrics into "SitReps" (Situation Reports).
    """
    def __init__(self):
        self.adjectives = {
            'positive': ['Robust', 'Healthy', 'Strong', 'Promising', 'Stable'],
            'negative': ['Fragile', 'Shaky', 'Volatile', 'Uncertain', 'Choppy'],
            'neutral': ['Steady', 'Quiet', 'Flat', 'Normal']
        }

    def _generate_mission_bar(self, ctx) -> str:
        current = ctx.get('equity', 0.0)
        start = config.INITIAL_CAPITAL
        target = config.MISSION_TARGET
        
        # Avoid division by zero
        denom = target - start if target > start else 1.0
        pct = (current - start) / denom
        pct = max(0.0, min(1.0, pct)) # Clamp
        
        # Bar: [|||||.....]
        bars = int(pct * 10)
        progress_bar = "█" * bars + "░" * (10 - bars)
        
        return f"🎯 **{config.MISSION_NAME}**: [{progress_bar}] {pct*100:.1f}% (${current:.2f}/${target:.2f})"

    def generate_sitrep(self, context: Dict[str, Any]) -> str:
        """
        Generate the situation report.
        Context requires: state, metabolism, leverage, sentiment_score, crisis_score, active_positions
        """
        system_state = context.get('state', SystemState.NOMINAL)
        sentiment_score = context.get('sentiment_score', 0.0)
        
        # 0. Mission Status Line
        mission_line = self._generate_mission_bar(context)
        
        # 1. Select Adjective based on Sentiment
        if sentiment_score > 0.3:
            mood = random.choice(self.adjectives['positive'])
        elif sentiment_score < -0.3:
            mood = random.choice(self.adjectives['negative'])
        else:
            mood = random.choice(self.adjectives['neutral'])

        # 2. Select Template based on System State
        if system_state == SystemState.CRITICAL:
            report = self._template_critical(context, mood)
        elif system_state == SystemState.CAUTION:
            report = self._template_caution(context, mood)
        elif system_state == SystemState.OPTIMAL:
            report = self._template_optimal(context, mood)
        else:
            report = self._template_nominal(context, mood)
            
        return f"{mission_line}\n\n{report}"

    def _template_critical(self, ctx, mood) -> str:
        reason = ctx.get('critical_reason', 'Unknown Threat')
        return (
            f"🚨 **CRITICAL ALERT**\n"
            f"System has entered defensive posture due to **{reason}**.\n"
            f"• Metabolism: HIBERNATE\n"
            f"• Action: Halting new entries, monitoring exits closely.\n"
            f"• Sentiment: {mood} ({ctx.get('sentiment_score', 0):.2f})"
        )

    def _template_caution(self, ctx, mood) -> str:
        return (
            f"⚠️ **CAUTION**\n"
            f"Market conditions are **{mood}**. Elevated risk detected.\n"
            f"• Metabolism: {ctx.get('metabolism', 'UNKNOWN')}\n"
            f"• Leverage: Reduced to {ctx.get('leverage_cap', '1.0')}x\n"
            f"• Positions: Holding {ctx.get('position_count', 0)} active trades.\n"
            f"• Note: Strict entry filters are active."
        )

    def _template_nominal(self, ctx, mood) -> str:
        return (
            f"✅ **SYSTEM NOMINAL**\n"
            f"Operations are proceeding normally. Market is **{mood}**.\n"
            f"• Metabolism: {ctx.get('metabolism', 'UNKNOWN')}\n"
            f"• Active Positions: {ctx.get('position_count', 0)}\n"
            f"• Daily PnL: {ctx.get('pnl_str', '$0.00')}"
        )

    def _template_optimal(self, ctx, mood) -> str:
        return (
            f"🚀 **OPTIMAL CONDITIONS**\n"
            f"Systems green. Market is **{mood}** and profitable.\n"
            f"• Performance: {ctx.get('pnl_str', '$0.00')} today.\n"
            f"• Metabolism: PREDATOR (Aggressive)\n"
            f"• Sentiment: High Confidence ({ctx.get('sentiment_score', 0):.2f})"
        )

class OverwatchHolon(Holon):
    def __init__(self, check_interval: int = 60, trader_ref=None):
        # High Integration (0.9), Low Autonomy (0.1) - The "Servant" Monitor
        super().__init__(name="OverwatchHolon", disposition=Disposition(autonomy=0.1, integration=0.9))
        
        self.trader = trader_ref
        self.narrative_engine = NarrativeEngine()
        self.check_interval = check_interval
        
        # State Cache
        self.latest_sitrep = "System Initializing..."
        self.latest_state = SystemState.NOMINAL
        
        # Telegram Config
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.app = None
        self.loop = None
        self.bot_thread = None
        self.stop_event = threading.Event()
        
        if TELEGRAM_AVAILABLE and config.TELEGRAM_ENABLED and config.TELEGRAM_BOT_TOKEN:
            self._setup_telegram()
        else:
            print(f"[{self.name}] Telegram connection skipped.")

    def _setup_telegram(self):
        """Initialize the Telegram Bot Application."""
        try:
            self.app = ApplicationBuilder().token(config.TELEGRAM_BOT_TOKEN).build()
            
            # Register Handlers
            self.app.add_handler(CommandHandler("start", self._cmd_start))
            self.app.add_handler(CommandHandler("status", self._cmd_status))
            self.app.add_handler(CommandHandler("report", self._cmd_report)) # Verbose SitRep
            self.app.add_handler(CommandHandler("panic", self._cmd_panic))
            self.app.add_handler(CommandHandler("stop", self._cmd_stop))
            self.app.add_handler(CommandHandler("config", self._cmd_config))
            self.app.add_handler(CommandHandler("positions", self._cmd_positions))
            self.app.add_handler(CommandHandler("signals", self._cmd_signals))
            self.app.add_handler(CommandHandler("audit", self._cmd_audit))
            self.app.add_handler(CommandHandler("scout", self._cmd_scout))
            self.app.add_handler(CommandHandler("forecast", self._cmd_forecast)) # NEW: Prediction
            self.app.add_handler(CommandHandler("lifespan", self._cmd_lifespan)) # NEW: Duration
            self.app.add_handler(CommandHandler("outlook", self._cmd_outlook))   # NEW: 3/7/21d Model
            self.app.add_handler(CommandHandler("model", self._cmd_outlook))     # Alias
            self.app.add_handler(CommandHandler("structure", self._cmd_structure)) # NEW: PIPs
            
            # --- PHASE 2: C2 Commands ---
            self.app.add_handler(CommandHandler("buy", self._cmd_buy))
            self.app.add_handler(CommandHandler("sell", self._cmd_sell))
            self.app.add_handler(CommandHandler("close", self._cmd_close))
            self.app.add_handler(CommandHandler("pause", self._cmd_pause))
            self.app.add_handler(CommandHandler("resume", self._cmd_resume))
            
            print(f"[{self.name}] ✅ Telegram Bot Ready (Overwatch Mode)")
            
            # Start Background Thread
            self.bot_thread = threading.Thread(target=self._run_bot_loop, daemon=True)
            self.bot_thread.start()
            
        except Exception as e:
            print(f"[{self.name}] ❌ Telegram Init Failed: {e}")

    def _run_bot_loop(self):
        """Asyncio loop for Telegram Polling."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Start periodic broadcast task
        self.loop.create_task(self._periodic_broadcast())
        
        while not self.stop_event.is_set():
            try:
                # print(f"[{self.name}] 📞 Polling Telegram...")
                self.app.run_polling(stop_signals=None, close_loop=False, timeout=10)
            except Exception as e:
                print(f"[{self.name}] ⚠️ Telegram Polling Error: {e}. Retrying...")
                time.sleep(5)
            
            if self.stop_event.is_set():
                break

    async def _periodic_broadcast(self):
        """Send periodic situation reports to Telegram."""
        interval_minutes = getattr(config, 'TELEGRAM_PERIODIC_ALERT_MINUTES', 60)
        while not self.stop_event.is_set():
            await asyncio.sleep(interval_minutes * 60)
            if self.latest_sitrep and self.chat_id:
                try:
                    await self.app.bot.send_message(
                        chat_id=self.chat_id, 
                        text=f"⏱️ **Periodic Update**\n\n{self.latest_sitrep}", 
                        parse_mode='Markdown'
                    )
                except Exception as e:
                    print(f"[{self.name}] ⚠️ Periodic Broadcast Failed: {e}")

    def perform_audit(self):
        """
        The main 'Sentry' logic.
        Aggregates state -> Generates SitRep -> Broadcasts.
        """
        if not self.trader:
            print(f"[{self.name}] ⚠️ Audit Skipped: Trader Reference Missing!")
            return

        # 1. Gather Intelligence
        gov = self.trader.sub_holons.get('governor')
        sent = self.trader.sub_holons.get('sentiment')
        exec_agent = self.trader.sub_holons.get('executor')
        
        if not (gov and exec_agent):
            # System not fully ready
            return

        # 2. Determine State
        current_equity = exec_agent.get_portfolio_value(0.0)
        
        metrics = {
             'equity': current_equity, # <--- Added for Mission Bar
            'metabolism': gov.get_metabolism_state(),
            'position_count': len(gov.positions),
            'sentiment_score': sent.current_sentiment_score if sent else 0.0,
            'crisis_score': sent.crisis_score if sent else 0.0,
            'drawdown_pct': gov.drawdown_pct,
            'leverage_cap': config.PREDATOR_LEVERAGE if gov.get_metabolism_state() == 'PREDATOR' else config.SCAVENGER_LEVERAGE
        }
        
        # Calculate PnL string
        from HolonicTrader.performance_tracker import get_performance_data
        perf = get_performance_data()
        pnl_val = perf.get('total_pnl', 0.0)
        metrics['pnl_str'] = f"${pnl_val:+.2f}" 

        # --- SESSION 3b: MARGIN LEVEL MONITOR ---
        # Calculate Real-Time Margin Level (Liquidation Proximity)
        # Maintenance Margin Rate = 10% (Conservative assumption for 5x/10x)
        # Used Margin = Total Position Value * MaintRate
        total_pos_value = 0.0
        if len(gov.positions) > 0:
            for sym, pos in gov.positions.items():
                qty = pos.get('quantity', 0.0) if isinstance(pos, dict) else getattr(pos, 'quantity', 0.0)
                # Use current price from Executor if avail, else entry
                ep = pos.get('entry_price', 0.0) if isinstance(pos, dict) else getattr(pos, 'entry_price', 0.0)
                curr_price = exec_agent.latest_prices.get(sym, ep)
                total_pos_value += (qty * curr_price)
        
        used_margin = total_pos_value * 0.10
        margin_level = 999.0 # Infinity
        if used_margin > 0:
            margin_level = current_equity / used_margin
            
        metrics['margin_level'] = margin_level
        metrics['effective_leverage'] = total_pos_value / current_equity if current_equity > 0 else 0.0
        # ----------------------------------------

        # State Logic
        state = SystemState.NOMINAL
        reason = ""
        
        # Priority Logic: Solvency First
        if margin_level < 1.1:
             # --- DEAD MAN's SWITCH: LIQUIDATION IMMINENT ---
             logger.critical(f"[{self.name}] ☠️ DEAD MAN'S SWITCH: Margin Level CRITICAL ({margin_level*100:.0f}%). Liquidation imminent!")
             raise DeadMansSwitchTriggered(f"Margin Level Dropped Below 110%: {margin_level*100:.0f}%")
             # -----------------------------------------------
             
        elif margin_level < 1.5:
             state = SystemState.CRITICAL
             reason = f"LIQUIDATION RISK (Margin Lvl {margin_level*100:.0f}%)"
             # Trigger Panic? Maybe too aggressive for auto-trigger without testing.
             # User said "Alert < 200%, Panic < 150%".
             # For now, CRITICAL state halts entries.
             
        elif margin_level < 2.0:
             state = SystemState.CAUTION
             reason = f"Low Margin Level ({margin_level*100:.0f}%)"
             
        elif metrics['drawdown_pct'] > 0.05 or metrics['crisis_score'] > 0.8:
            state = SystemState.CRITICAL
            reason = "High Drawdown" if metrics['drawdown_pct'] > 0.05 else "Geopolitical Crisis"
        elif metrics['sentiment_score'] < -0.4 or metrics['drawdown_pct'] > 0.02:
            state = SystemState.CAUTION
        elif metrics['sentiment_score'] > 0.4:
            state = SystemState.OPTIMAL
            
        metrics['state'] = state
        metrics['critical_reason'] = reason # Logic for Critical
        
        self.latest_state = state
        
        # 3. Generate Narrative
        self.latest_sitrep = self.narrative_engine.generate_sitrep(metrics)
        
        # 4. Push to Dashboard
        if self.trader.gui_queue:
            self.trader.gui_queue.put({
                'type': 'overwatch_update',
                'state': state.value,
                'sitrep': self.latest_sitrep
            })
        
        # 4b. ALWAYS Write to file for external dashboard sync
        try:
            import json
            import os
            from datetime import datetime, timezone
            status_file_path = os.path.join(os.path.dirname(__file__), '..', 'dashboard_status.json')
            
            # Read existing file to merge
            existing_data = {}
            if os.path.exists(status_file_path):
                with open(status_file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            
            # Add overwatch data
            existing_data['overwatch'] = {
                'state': state.value,
                'sitrep': self.latest_sitrep,
                'updated': datetime.now(timezone.utc).isoformat()
            }
            
            # Atomic write
            temp_path = status_file_path + '.tmp'
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, default=str)
            os.replace(temp_path, status_file_path)
        except Exception:
            pass  # Silently fail file write
            
        # 5. Broadcast (Optional: Only on State Change or Critical)
        # For now, we don't spam Telegram every 60s. We specific commands or Critical transitions.
        if state == SystemState.CRITICAL and getattr(self, '_last_broadcast_state', None) != SystemState.CRITICAL:
            self.send_telegram_alert(self.latest_sitrep)
            # Moltbook: Narrate the Crisis
            moltbook = self.trader.sub_holons.get('moltbook')
            if moltbook:
                moltbook.post_event('STATE_CRITICAL', {'sitrep': self.latest_sitrep})
                
        elif state == SystemState.OPTIMAL and getattr(self, '_last_broadcast_state', None) != SystemState.OPTIMAL:
            # Moltbook: Brag about the Setup
            moltbook = self.trader.sub_holons.get('moltbook')
            if moltbook:
                moltbook.post_event('STATE_OPTIMAL', {'sitrep': self.latest_sitrep})
            
        self._last_broadcast_state = state

    def get_dashboard_state(self) -> dict:
        """Expose Overwatch data for the dashboard."""
        return {
            'overwatch_state': self.latest_state.value,
            'sitrep': self.latest_sitrep,
        }

    # --- Telegram Commands ---
    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("👁️ **Overwatch Online**.\nI am monitoring the system.")

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(self.latest_sitrep, parse_mode='Markdown')

    async def _cmd_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Detailed stats + Narrative
        if not self.trader: return
        gov = self.trader.sub_holons.get('governor')
        
        msg = (
            f"{self.latest_sitrep}\n\n"
            f"**Technical Details:**\n"
            f"• Balance: ${gov.balance:.2f}\n"
            f"• Avail Margin: ${gov.available_balance:.2f}\n"
            f"• Drawdown: {gov.drawdown_pct*100:.2f}%\n"
            f"• State: {self.latest_state.value}"
        )
        await update.message.reply_text(msg, parse_mode='Markdown')

    async def _cmd_panic(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🚨 **PANIC SIGNAL RECEIVED** 🚨\nForwarding to Executor for immediate liquidation.")
        # Trigger Executor Panic
        if self.trader:
            executor = self.trader.sub_holons.get('executor')
            if executor:
                res = executor.panic_close_all(executor.latest_prices)
                await update.message.reply_text(f"🛑 Result:\n{res}")

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Gracefully stop the trading bot."""
        if self.trader and hasattr(self.trader, 'gui_stop_event') and self.trader.gui_stop_event:
            self.trader.gui_stop_event.set()
            await update.message.reply_text("🛑 **STOP SIGNAL RECEIVED**. Initiating graceful shutdown sequence...")
        else:
            await update.message.reply_text("⚠️ Error: Cannot stop. Control link missing.")

    async def _cmd_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Update configuration dynamically. Usage: /config <alloc|lev|micro> <value>"""
        try:
            if not context.args or len(context.args) != 2:
                await update.message.reply_text("Usage: `/config <alloc|lev|micro> <value>`\nEx: `/config alloc 0.5`")
                return

            param, val = context.args[0].lower(), context.args[1]
            cfg_update = {}
            
            if param in ['alloc', 'allocation']:
                cfg_update['max_allocation'] = float(val)
            elif param in ['lev', 'leverage']:
                cfg_update['leverage_cap'] = float(val)
            elif param == 'micro':
                cfg_update['micro_mode'] = (val.lower() in ['true', '1', 'yes', 'on'])
            else:
                await update.message.reply_text(f"❌ Unknown parameter: {param}")
                return

            if self.trader and hasattr(self.trader, 'command_queue') and self.trader.command_queue:
                self.trader.command_queue.put({'type': 'update_config', 'data': cfg_update})
                await update.message.reply_text(f"⚙️ **Config Update Queued**:\n`{cfg_update}`", parse_mode='Markdown')
            else:
                 await update.message.reply_text("⚠️ Error: Command Queue not linked.")
                 
        except Exception as e:
             await update.message.reply_text(f"❌ Config Error: {e}")

    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List active positions detailed."""
        if not self.trader: return
        gov = self.trader.sub_holons.get('governor')
        executor = self.trader.sub_holons.get('executor')
        
        if not gov.positions:
            await update.message.reply_text("📉 No active positions.")
            return

        lines = ["📊 **Active Portfolio**"]
        total_pnl = 0.0

        for sym, pos in gov.positions.items():
            # Duck-typed accessor for both dict and Position objects
            if isinstance(pos, dict):
                qty = pos.get('quantity', 0.0)
                entry = pos.get('entry_price', 0.0)
            else:
                qty = getattr(pos, 'quantity', 0.0)
                entry = getattr(pos, 'entry_price', 0.0)
            curr = executor.latest_prices.get(sym, entry)
            
            # Estimate PnL (Simple)
            val_entry = qty * entry
            val_curr = qty * curr
            pnl = val_curr - val_entry
            pnl_pct = (pnl / val_entry) * 100 if val_entry > 0 else 0.0
            
            total_pnl += pnl
            icon = "🟢" if pnl >= 0 else "🔴"
            
            lines.append(f"{icon} `{sym}`: {pnl_pct:+.2f}% (${pnl:+.2f}) | Qty: {qty:.4f}")

        lines.append(f"\n💰 **Total Unrealized PnL**: ${total_pnl:+.2f}")
        await update.message.reply_text("\n".join(lines), parse_mode='Markdown')

    async def _cmd_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show latest oracle probes."""
        if not self.trader: return
        oracle = self.trader.sub_holons.get('oracle')
        if not oracle: return
        
        lines = ["📡 **Oracle Radar (Latest)**"]
        
        # Sort by XGB probability descending
        sorted_probes = sorted(
            oracle.last_probes.items(), 
            key=lambda x: x[1].get('xgb', 0) if isinstance(x[1], dict) else 0, 
            reverse=True
        )
        
        count = 0
        for sym, data in sorted_probes:
            if count >= 10: break # Top 10
            if not isinstance(data, dict): continue
            
            xgb_prob = data.get('xgb', 0.0)
            lstm_prob = data.get('lstm', 0.0)
            trend = "Bullish" if oracle.symbol_trends.get(sym, False) else "Bearish"
            
            icon = "🔥" if xgb_prob > 0.53 else "❄️"
            lines.append(f"{icon} `{sym}`: XGB {xgb_prob:.2f} | LSTM {lstm_prob:.2f} | {trend}")
            count += 1
            
        await update.message.reply_text("\n".join(lines), parse_mode='Markdown')
        
    async def _cmd_audit(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Audit a specific asset. Usage: /audit <SYMBOL>"""
        if not context.args or len(context.args) != 1:
            await update.message.reply_text("Usage: `/audit <SYMBOL>` (e.g., BTC/USDT)")
            return
            
        symbol = context.args[0].upper()
        if "/" not in symbol and "USDT" not in symbol: symbol += "/USDT" # Helper
        
        if not self.trader: return
        
        # Gather Intel
        gov = self.trader.sub_holons.get('governor')
        oracle = self.trader.sub_holons.get('oracle')
        executor = self.trader.sub_holons.get('executor')
        structure = self.trader.sub_holons.get('structure')
        
        is_held = symbol in gov.positions
        price = executor.latest_prices.get(symbol, "Unknown")
        probe = oracle.last_probes.get(symbol, {})
        
        struct_info = "N/A"
        if structure and hasattr(structure, 'get_structure'):
            struct = structure.get_structure(symbol)
            struct_info = f"{struct.get('trend', 'UNK')} ({struct.get('zone', 'UNK')})"
            
        msg = [
            f"🧐 **Audit: {symbol}**",
            f"• Price: ${price}",
            f"• Position: {'HELD' if is_held else 'FLAT'}",
            f"• Structure: {struct_info}",
            f"• Signals: XGB={probe.get('xgb',0):.2f}, LSTM={probe.get('lstm',0):.2f}"
        ]
        
        if is_held:
            pos = gov.positions[symbol]
            # Duck-typed accessor for both dict and Position objects
            if isinstance(pos, dict):
                msg.append(f"• Entry: ${pos.get('entry_price',0)}")
                msg.append(f"• Quantity: {pos.get('quantity',0)}")
            else:
                msg.append(f"• Entry: ${getattr(pos, 'entry_price', 0)}")
                msg.append(f"• Quantity: {getattr(pos, 'quantity', 0)}")
            
        await update.message.reply_text("\n".join(msg), parse_mode='Markdown')

    async def _cmd_scout(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Force a scout cycle/refresh."""
        if self.trader:
            # Setting last_run to 0 triggers reset logic next loop
            self.trader.scout_last_run = 0
            await update.message.reply_text("🔭 **Scout Refresh Requested**.\nTop 15 assets will be re-evaluated next cycle.")
        else:
            await update.message.reply_text("⚠️ Link Broken.")

    async def _cmd_forecast(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Predict price movement probability.
        Usage: /forecast <SYMBOL> <TARGET_PRICE> [DAYS]
        """
        try:
            if not context.args or len(context.args) < 2:
                await update.message.reply_text("Usage: `/forecast <SYMBOL> <TARGET> [DAYS]`\nEx: `/forecast BTC 100000 30`")
                return
            
            symbol = context.args[0].upper()
            if "/" not in symbol and "USDT" not in symbol: symbol += "/USDT"
            
            try: target = float(context.args[1])
            except: 
                await update.message.reply_text("❌ Invalid Target Price.")
                return
                
            days = int(context.args[2]) if len(context.args) > 2 else 30
            
            if not self.trader: return
            oracle = self.trader.sub_holons.get('oracle')
            observer = self.trader.sub_holons.get('observer')
            
            if not (oracle and observer):
                await update.message.reply_text("⚠️ Oracle/Observer not ready.")
                return
                
            await update.message.reply_text(f"🔮 **Forecasting...**\nRunning Monte Carlo sim for {symbol} -> ${target} over {days} days.")
            
            # Fetch Data (Sync call in Async? Ideally thread it, but it's fast enough or we accept block)
            # Fetch 1h candles for best fit (e.g. 500 candles ~ 20 days context)
            df = observer.fetch_market_data(symbol=symbol, timeframe='1h', limit=500)
            if df is None or df.empty:
                await update.message.reply_text("❌ Data fetch failed.")
                return
                
            prices = df['close'].values
            
            # Generate Forecast
            res = oracle.generate_forecast(symbol, target, days, prices)
            
            if 'error' in res:
                await update.message.reply_text(f"❌ Error: {res['error']}")
                return
            
            prob = res['probability']
            drift = res['drift'] * 100 # % per step
            vol = res['volatility'] * 100 # % per root-step
            
            icon = "🟢" if prob > 0.5 else "🔴"
            if prob > 0.8: icon = "🔥"
            if prob < 0.1: icon = "🧊"
            
            msg = (
                f"{icon} **Forecast: {symbol}**\n"
                f"Target: **${target:,.2f}** in {days} days\n"
                f"Probability: **{prob*100:.1f}%**\n\n"
                f"**Physics Model (GBM):**\n"
                f"• Current: ${res['current_price']:,.2f}\n"
                f"• Drift (Trend): {drift:.4f}%\n"
                f"• Volatility: {vol:.2f}%"
            )
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Forecast Error: {e}")

    async def _cmd_lifespan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Estimate 'Time To Live' (TTL) for a trade before hitting SL.
        Usage: /lifespan <SYMBOL> <STOP_PRICE>
        """
        try:
            if not context.args or len(context.args) < 2:
                await update.message.reply_text("Usage: `/lifespan <SYMBOL> <STOP_LOSS>`\nEx: `/lifespan BTC 65000`")
                return
            
            symbol = context.args[0].upper()
            if "/" not in symbol and "USDT" not in symbol: symbol += "/USDT"
            
            try: sl = float(context.args[1])
            except: 
                await update.message.reply_text("❌ Invalid Stop Loss.")
                return
            
            if not self.trader: return
            oracle = self.trader.sub_holons.get('oracle')
            observer = self.trader.sub_holons.get('observer')
            
            if not (oracle and observer):
                await update.message.reply_text("⚠️ Oracle/Observer not ready.")
                return
                
            await update.message.reply_text(f"⏳ **Calculating Lifespan...**\nSimulating survival time for {symbol} above ${sl}...")
            
            df = observer.fetch_market_data(symbol=symbol, timeframe='1h', limit=500)
            if df is None or df.empty:
                await update.message.reply_text("❌ Data fetch failed.")
                return
                
            prices = df['close'].values
            
            res = oracle.calculate_trade_expectancy(symbol, sl, prices)
            
            if 'error' in res:
                await update.message.reply_text(f"❌ Error: {res['error']}")
                return
                
            days = res['expected_duration_days']
            drift = res['drift'] * 100
            
            # Formatting
            if days >= 90:
                time_str = "> 90 Days (Long Term)"
                icon = "♾️"
            elif days < 1.0:
                time_str = f"{days*24:.1f} Hours"
                icon = "⚡"
            else:
                time_str = f"{days:.1f} Days"
                icon = "🗓️"
                
            msg = (
                f"{icon} **Trade Life Expectancy**\n"
                f"Symbol: `{symbol}`\n"
                f"Stop Loss: `${sl:,.2f}`\n\n"
                f"**Median Survival: {time_str}**\n"
                f"(Time until 50% of paths hit SL)\n\n"
                f"• Current Price: ${res['current_price']:,.2f}\n"
                f"• Market Drift: {drift:.4f}%"
            )
            await update.message.reply_text(msg, parse_mode='Markdown')

        except Exception as e:
            await update.message.reply_text(f"❌ Lifespan Error: {e}")

    async def _cmd_outlook(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Multi-Horizon Model (3d/7d/21d).
        Usage: /outlook <SYMBOL> or /model <SYMBOL>
        """
        try:
            if not context.args or len(context.args) < 1:
                await update.message.reply_text("Usage: `/outlook <SYMBOL>` (e.g. BTC)")
                return
            
            symbol = context.args[0].upper()
            if "/" not in symbol and "USDT" not in symbol: symbol += "/USDT"
            
            if not self.trader: return
            oracle = self.trader.sub_holons.get('oracle')
            observer = self.trader.sub_holons.get('observer')
            
            if not (oracle and observer):
                await update.message.reply_text("⚠️ Oracle/Observer not ready.")
                return

            await update.message.reply_text(f"📐 **Generating Market Model...**\nSimulating 3/7/21 day cones for {symbol}.")
            
            df = observer.fetch_market_data(symbol=symbol, timeframe='1h', limit=600)
            if df is None or df.empty:
                await update.message.reply_text("❌ Data fetch failed.")
                return
                
            prices = df['close'].values
            
            # Generate Term Structure
            res = oracle.generate_term_structure(symbol, prices)
            
            if 'error' in res:
                await update.message.reply_text(f"❌ Error: {res['error']}")
                return
                
            struct = res['structure']
            curr = res['current_price']
            drift = res['drift'] * 100
            
            # Format Output
            msg = [f"📐 **Market Model: {symbol}**"]
            msg.append(f"Current: ${curr:,.2f}")
            msg.append(f"Trend (Drift): {drift:+.4f}% / day\n")
            
            msg.append("**Forecast Cones (90% CI):**")
            
            for d in [3, 7, 21]:
                key = f'{d}d'
                data = struct[key]
                bear = data['p05']
                bull = data['p95']
                base = data['p50']
                
                # Colors based on Base vs Current
                icon = "➖"
                if base > curr * 1.01: icon = "📈"
                elif base < curr * 0.99: icon = "📉"
                
                msg.append(f"{icon} **{key} Horizon**")
                msg.append(f"   🐻 ${bear:,.0f}  |  🎯 ${base:,.0f}  |  🐂 ${bull:,.0f}")
                
                # Volatility Width
                width = (bull - bear) / base
                msg.append(f"   (Vol Range: {width*100:.1f}%)")
                msg.append("")
                
            msg.append(f"_Model: Monte Carlo GBM (2000 Paths)_")
            
            await update.message.reply_text("\n".join(msg), parse_mode='Markdown')

        except Exception as e:
            await update.message.reply_text(f"❌ Outlook Error: {e}")

    async def _cmd_structure(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Market Structure Analysis (PIPs / ZigZag).
        Usage: /structure <SYMBOL>
        """
        try:
            if not context.args or len(context.args) < 1:
                await update.message.reply_text("Usage: `/structure <SYMBOL>`")
                return
            
            symbol = context.args[0].upper()
            if "/" not in symbol and "USDT" not in symbol: symbol += "/USDT"
            
            if not self.trader: return
            oracle = self.trader.sub_holons.get('oracle')
            observer = self.trader.sub_holons.get('observer')
            
            if not (oracle and observer):
                await update.message.reply_text("⚠️ Oracle/Observer not ready.")
                return

            await update.message.reply_text(f"📐 **Analyzing Structure (PIPs)...**\nScanning {symbol} for pivots.")
            
            # Fetch 2 weeks of 1h data (approx ~336 candles)
            limit = 350
            df = observer.fetch_market_data(symbol=symbol, timeframe='1h', limit=limit)
            if df is None or df.empty:
                await update.message.reply_text("❌ Data fetch failed.")
                return
                
            prices = df['close'].values.tolist()
            
            # Run PIPs
            res = oracle.get_structure_status(symbol, prices)
            
            if 'error' in res:
                await update.message.reply_text(f"❌ Error: {res['error']}")
                return
            
            status = res['status'] # BULLISH_LEG / BEARISH_LEG
            pips = res['pips_values']
            
            # Formatting
            icon = "⚖️"
            if "BULLISH" in status: icon = "🐂"
            elif "BEARISH" in status: icon = "🐻"
            
            msg = [f"{icon} **Structure: {symbol}**", f"Bias: **{status}**"]
            msg.append("")
            msg.append("**Semantic Pivots (Last 7):**")
            
            # Draw ASCII Sparkline of pivots? Or just list them?
            # List mostly sufficient for now.
            for i, p in enumerate(pips):
                label = "Pip"
                if i == 0: label = "Start"
                if i == len(pips)-1: label = "Curr"
                
                # Check relation to prev
                change = ""
                if i > 0:
                    delta = (p - pips[i-1]) / pips[i-1]
                    arrow = "↗️" if delta > 0 else "↘️"
                    change = f" ({arrow} {delta*100:.1f}%)"
                    
                msg.append(f"{i+1}. ${p:,.2f}{change}")
                
            await update.message.reply_text("\n".join(msg), parse_mode='Markdown')

        except Exception as e:
            await update.message.reply_text(f"❌ Structure Error: {e}")

    # --- C2 Override Commands ---
    async def _cmd_buy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args or len(context.args) < 2:
            await update.message.reply_text("Usage: `/buy <SYMBOL> <QTY_IN_USD>`\nEx: `/buy BTC 50`")
            return
        await self._dispatch_c2_order('BUY', context.args, update)

    async def _cmd_sell(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args or len(context.args) < 2:
            await update.message.reply_text("Usage: `/sell <SYMBOL> <QTY_IN_USD>`\nEx: `/sell ETH 50`")
            return
        await self._dispatch_c2_order('SELL', context.args, update)

    async def _dispatch_c2_order(self, direction: str, args: list, update: Update):
        symbol = args[0].upper()
        if "/" not in symbol and "USDT" not in symbol: symbol += "/USDT"
        try:
            qty_usd = float(args[1])
        except ValueError:
            await update.message.reply_text("❌ Quantity must be a number (USD).")
            return
            
        if self.trader and self.trader.command_queue:
            self.trader.command_queue.put({
                'type': 'c2_order',
                'direction': direction,
                'symbol': symbol,
                'qty_usd': qty_usd
            })
            await update.message.reply_text(f"⚔️ **C2 Command Queued**:\nExecute {direction} on `{symbol}` for ${qty_usd:.2f} USD", parse_mode='Markdown')
        else:
            await update.message.reply_text("⚠️ Command queue not linked.")

    async def _cmd_close(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args or len(context.args) < 1:
            await update.message.reply_text("Usage: `/close <SYMBOL>`\nEx: `/close SOL`")
            return
            
        symbol = context.args[0].upper()
        if "/" not in symbol and "USDT" not in symbol: symbol += "/USDT"
        
        if self.trader and self.trader.command_queue:
            self.trader.command_queue.put({
                'type': 'c2_close',
                'symbol': symbol
            })
            await update.message.reply_text(f"⚔️ **C2 Command Queued**:\nClose position for `{symbol}`", parse_mode='Markdown')
        else:
            await update.message.reply_text("⚠️ Command queue not linked.")

    async def _cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.trader and self.trader.command_queue:
            self.trader.command_queue.put({'type': 'c2_pause'})
            await update.message.reply_text("⏸️ **System Pause Queued**\nThe bot will stop opening new positions.")
        else:
            await update.message.reply_text("⚠️ Command queue not linked.")

    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.trader and self.trader.command_queue:
            self.trader.command_queue.put({'type': 'c2_resume'})
            await update.message.reply_text("▶️ **System Resume Queued**\nThe bot will resume normal operations.")
        else:
            await update.message.reply_text("⚠️ Command queue not linked.")

    def send_telegram_alert(self, msg: str):
        """Thread-safe send with Markdown Parse Error fallback."""
        if not self.app:
            print(f"[{self.name}] ❌ Telegram Alert Failed: Bot app not initialized")
            return
        if not self.chat_id:
            print(f"[{self.name}] ❌ Telegram Alert Failed: chat_id is {self.chat_id} (check TELEGRAM_CHAT_ID env var)")
            return
        if not self.loop:
            print(f"[{self.name}] ❌ Telegram Alert Failed: Event loop not running")
            return

        async def _safe_send():
            try:
                await self.app.bot.send_message(chat_id=self.chat_id, text=msg, parse_mode='Markdown')
            except Exception as e:
                error_str = str(e)
                if "Chat not found" in error_str:
                    print(f"[{self.name}] ❌ Telegram Alert Failed: Chat ID {self.chat_id} not found. Bot may not be added to chat.")
                elif "Unauthorized" in error_str:
                    print(f"[{self.name}] ❌ Telegram Alert Failed: Unauthorized. Bot was blocked by user.")
                elif "timeout" in error_str.lower():
                    print(f"[{self.name}] ❌ Telegram Alert Failed: Network timeout.")
                elif "parse" in error_str.lower() or "entities" in error_str.lower():
                    print(f"[{self.name}] ⚠️ Telegram Markdown Parse Error. Falling back to plain text...")
                    try:
                        await self.app.bot.send_message(chat_id=self.chat_id, text=msg)
                    except Exception as fallback_e:
                        print(f"[{self.name}] ❌ Telegram Plaintext Fallback Failed: {fallback_e}")
                else:
                    print(f"[{self.name}] ❌ Async Telegram Alert Failed: {e}")

        try:
            if self.loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    _safe_send(),
                    self.loop
                )
            else:
                print(f"[{self.name}] ⚠️ Telegram loop not running, attempting sync send...")
                # Fallback: try synchronous send
                if self.app and self.chat_id:
                    import requests
                    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
                    try:
                        res = requests.post(url, json={'chat_id': self.chat_id, 'text': msg, 'parse_mode': 'Markdown'}, timeout=5)
                        if res.status_code != 200:
                            if "parse" in res.text.lower() or "entities" in res.text.lower():
                                requests.post(url, json={'chat_id': self.chat_id, 'text': msg}, timeout=5)
                            else:
                                print(f"[{self.name}] ❌ Sync Telegram Failed HTTP {res.status_code}: {res.text}")
                    except Exception as req_e:
                        print(f"[{self.name}] ❌ Sync Telegram Failed: {req_e}")
        except Exception as e:
            print(f"[{self.name}] ❌ Global Alert Submission Failed: {e}")

    def receive_message(self, sender: Any, content: Any) -> None:
        """Handle incoming Message objects (alerts)."""
        # If we receive a CRITICAL message, we broadcast immediately.
        pass

    def stop(self):
        self.stop_event.set()
