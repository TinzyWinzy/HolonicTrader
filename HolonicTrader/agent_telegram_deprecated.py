"""
TelegramHolon - Remote Command & Control (Phase 4 Extension)

Allows remote monitoring and "Kill Switch" functionality via Telegram.
"""

import threading
import asyncio
import config
from typing import Any, Dict

# Telegram Imports
try:
    from telegram import Update
    from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
    from telegram.error import NetworkError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("⚠️ python-telegram-bot not installed. TelegramHolon DISABLED.")

from HolonicTrader.holon_core import Holon, Disposition, Message

class TelegramHolon(Holon):
    def __init__(self, executor=None, stop_event=None, trader_ref=None):
        super().__init__(name="TelegramHolon", disposition=Disposition(autonomy=0.5, integration=0.5))
        self.executor = executor
        self.stop_event = stop_event
        self.trader = trader_ref
        
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.app = None
        self.loop = None
        self.bot_thread = None
        
        if TELEGRAM_AVAILABLE and config.TELEGRAM_ENABLED and config.TELEGRAM_BOT_TOKEN:
            self.setup_bot()
        else:
            print(f"[{self.name}] ❌ Telegram Disabled (Missing Token or Dependency)")

    def setup_bot(self):
        """Initialize the Telegram Bot Application."""
        try:
            self.app = ApplicationBuilder().token(config.TELEGRAM_BOT_TOKEN).build()
            
            # Register Handlers
            self.app.add_handler(CommandHandler("start", self._start_command))
            self.app.add_handler(CommandHandler("status", self._status_command))
            self.app.add_handler(CommandHandler("panic", self._panic_command))
            self.app.add_handler(CommandHandler("pause", self._pause_command))
            self.app.add_handler(CommandHandler("resume", self._resume_command))
            self.app.add_handler(CommandHandler("help", self._help_command))
            
            print(f"[{self.name}] ✅ Bot Initialized")
            
            # Start Background Thread
            self.bot_thread = threading.Thread(target=self._run_bot_loop, daemon=True)
            self.bot_thread.start()
            
        except Exception as e:
            print(f"[{self.name}] ❌ Init Failed: {e}")

    def _run_bot_loop(self):
        """Entry point for the bot thread with auto-retry."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        while not (self.stop_event and self.stop_event.is_set()):
            try:
                # Polling is blocking, so this will run until stop or error
                # We specify close_loop=False so we can restart it if needed
                
                print(f"[{self.name}] 🔄 Starting Polling...")
                # Increased timeout and read_timeout for stability
                self.app.run_polling(stop_signals=None, close_loop=False, timeout=20, read_timeout=30, write_timeout=30)
                
            except NetworkError as e:
                print(f"[{self.name}] ⚠️ Telegram Network Error: {e}. Retrying in 5s...")
                import time
                time.sleep(5)
            except Exception as e:
                # If we are here, polling crashed.
                print(f"[{self.name}] ⚠️ Telegram Connection Lost: {e}. Retrying in 10s...")
                import time
                time.sleep(10)
                
            if self.stop_event and self.stop_event.is_set():
                break

    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🤖 HolonicTrader Online.\n\n"
            "Commands:\n"
            "/status - View Portfolio Health\n"
            "/pause - ⏸️ Pause New Entries\n"
            "/resume - ▶️ Resume Trading\n"
            "/panic - 🚨 EMERGENCY STOP (Close All)\n"
        )

    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Commands:\n"
            "/status - View Portfolio Health\n"
            "/pause - ⏸️ Pause New Entries\n"
            "/resume - ▶️ Resume Trading\n"
            "/panic - 🚨 EMERGENCY STOP (Close All)"
        )

    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Report current system status."""
        if not self.executor:
            await update.message.reply_text("❌ Executor not linked.")
            return

        summary = self.executor.get_execution_summary()
        held = self.executor.held_assets
        
        # Format positions
        pos_str = ""
        for sym, qty in held.items():
            if abs(qty) > 0.00000001:
                pos_str += f"- {sym}: {qty:.4f}\n"

        # Check Trader Pause State
        paused_status = "🟢 ACTIVE"
        if self.trader and getattr(self.trader, 'is_paused', False):
            paused_status = "⏸️ PAUSED"

        msg = (
            f"📊 **System Status**\n"
            f"State: {paused_status}\n"
            f"Balance: ${summary['balance']:.2f}\n"
            f"Equity: ${summary['equity']:.2f}\n"
            f"Margin Used: ${summary['margin_used']:.2f}\n"
            f"Positions: {summary['active_positions']}\n"
            f"------------------\n"
            f"{pos_str if pos_str else 'No Active Positions'}"
        )
        await update.message.reply_text(msg, parse_mode='Markdown')

    async def _pause_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Pause trading (No new entries)."""
        if self.trader:
            self.trader.is_paused = True
            await update.message.reply_text("⏸️ **Trading PAUSED**. No new positions will be opened.")
        else:
             await update.message.reply_text("❌ Trader Agent not available.")

    async def _resume_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Resume trading."""
        if self.trader:
            self.trader.is_paused = False
            await update.message.reply_text("▶️ **Trading RESUMED**.")
        else:
             await update.message.reply_text("❌ Trader Agent not available.")

    async def _panic_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """TRIGGER THE KILL SWITCH."""
        await update.message.reply_text("🚨 **PANIC RECEIVED** 🚨\nInitiating Emergency Shutdown Protocol...")
        
        if self.executor:
            # We need current prices for the panic close.
            # Using executor's latest known prices.
            results = self.executor.panic_close_all(self.executor.latest_prices)
            
            res_str = "\n".join(results)
            await update.message.reply_text(f"🛑 **Positions Closed:**\n{res_str}")
            
        else:
            await update.message.reply_text("❌ Executor not linked! Manual close required on Exchange.")

        # Shut down the loop
        if self.stop_event:
            self.stop_event.set()
            await update.message.reply_text("💀 System Stopping...")

    def send_message(self, message: str):
        """
        Send a notification to the owner.
        Thread-safe wrapper for asyncio logic.
        """
        if not (self.app and self.chat_id): return
        
        try:
           # Fire and forget (ish) - safely insert into the bot's loop
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.app.bot.send_message(chat_id=self.chat_id, text=message),
                    self.loop
                )
        except Exception as e:
            print(f"[{self.name}] ❌ Send Failed: {e}")

    def receive_message(self, sender: Any, content: Any) -> None:
        pass
