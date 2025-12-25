import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import queue
from datetime import datetime

# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates

# Import the bot runners
# main_live_phase4 now contains run_bot
from main_live_phase4 import run_bot
from run_backtest import run_backtest

class HolonicDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("HolonicTrader Command Center")
        self.root.geometry("1400x900") # Expanded for Tabs
        
        # Threading vars
        self.gui_queue = queue.Queue()
        self.gui_stop_event = threading.Event()
        self.bot_thread = None
        self.is_running_live = False
        self.is_running_backtest = False
        
        # PPO Tracking
        self.last_ppo_conviction = 0.5
        self.last_ppo_reward = 0.0
        
        # Configuration Vars
        self.conf_symbol = tk.StringVar(value="XRP/USDT")
        self.conf_timeframe = tk.StringVar(value="1h")
        self.conf_alloc = tk.DoubleVar(value=0.10)
        self.conf_leverage = tk.DoubleVar(value=5.0)

        self._setup_ui()
        
        # Start the polling loop
        self.root.after(100, self.process_queue)

    def _setup_ui(self):
        # Styles
        style = ttk.Style()
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("SubHeader.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Data.TLabel", font=("Consolas", 11))
        style.configure("Status.TLabel", font=("Segoe UI", 12, "bold"))
        
        # --- Header ---
        header_frame = ttk.Frame(self.root, padding="10 10 10 0")
        header_frame.pack(fill=tk.X)
        
        ttk.Label(header_frame, text="AEHML HOLONIC TRADER", style="Header.TLabel").pack(side=tk.LEFT)
        
        self.status_var = tk.StringVar(value="SYSTEM READY")
        self.status_label = ttk.Label(header_frame, textvariable=self.status_var, style="Status.TLabel", foreground="blue")
        self.status_label.pack(side=tk.RIGHT)

        ttk.Separator(self.root, orient='horizontal').pack(fill=tk.X, pady=10)

        # --- Main Notebook ---
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Tabs
        self.tab_live = ttk.Frame(self.notebook, padding=10)
        self.tab_agents = ttk.Frame(self.notebook, padding=10)
        self.tab_config = ttk.Frame(self.notebook, padding=10)
        self.tab_backtest = ttk.Frame(self.notebook, padding=10)
        
        self.notebook.add(self.tab_live, text="  Live Operations  ")
        self.notebook.add(self.tab_agents, text="  Holon Status  ")
        self.notebook.add(self.tab_config, text="  Configuration  ")
        self.notebook.add(self.tab_backtest, text="  Backtesting  ")
        
        self._setup_live_tab()
        self._setup_agents_tab()
        self._setup_config_tab()
        self._setup_backtest_tab()

    # ========================== TAB 1: LIVE ==========================
    def _setup_live_tab(self):
        # Top Controls
        ctl_frame = ttk.Frame(self.tab_live)
        ctl_frame.pack(fill=tk.X, pady=5)
        
        self.start_btn = ttk.Button(ctl_frame, text="▶ START LIVE BOT", command=self.start_live_bot)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(ctl_frame, text="⏹ STOP BOT", command=self.stop_bot, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Main Grid
        grid_frame = ttk.Frame(self.tab_live)
        grid_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left Col: Metrics & Table
        left_col = ttk.Frame(grid_frame)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # > Market Table (Treeview)
        tbl_frame = ttk.LabelFrame(left_col, text="Market Overview", padding=10)
        tbl_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        cols = ("Symbol", "Price", "Regime", "Entropy", "RSI", "LSTM", "XGB", "PnL", "Action")
        self.tree = ttk.Treeview(tbl_frame, columns=cols, show='headings', height=10)
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Configure columns with appropriate widths
        self.tree.column("Symbol", width=70)
        self.tree.column("Price", width=80)
        self.tree.column("Regime", width=80)
        self.tree.column("Entropy", width=60)
        self.tree.column("RSI", width=50)
        self.tree.column("LSTM", width=50)
        self.tree.column("XGB", width=50)
        self.tree.column("PnL", width=70)
        self.tree.column("Action", width=110)
        
        for col in cols:
            self.tree.heading(col, text=col)
            
        # > Financials (Simple)
        # fin_frame = ttk.LabelFrame(left_col, text="Financials", padding=10)
        # fin_frame.pack(fill=tk.X, pady=5)
        # self.val_label = self._metric(fin_frame, "Portfolio:", "$0.00", 0)
        
        # Right Col: Logs
        right_col = ttk.Frame(grid_frame)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        lbl = ttk.Label(right_col, text="Live Activity Log", style="SubHeader.TLabel")
        lbl.pack(anchor="w")
        self.log_text = scrolledtext.ScrolledText(right_col, height=12, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.see(tk.END)
        
        # Configure Tags for Syntax Highlighting
        self.log_text.tag_config("POSITIVE", foreground="#00e676") # Bright Green
        self.log_text.tag_config("NEGATIVE", foreground="#ff5252") # Bright Red
        self.log_text.tag_config("WARNING", foreground="#ffd740")  # Amber
        self.log_text.tag_config("INFO", foreground="#40c4ff")     # Light Blue
        self.log_text.tag_config("DIM", foreground="#78909c")      # Grey

        # Asset Allocation Pie Chart (Small)
        pie_frame = ttk.LabelFrame(right_col, text="Asset Allocation", padding=5)
        pie_frame.pack(fill=tk.X, pady=5)
        
        self.fig_pie = Figure(figsize=(3, 2), dpi=80)
        self.ax_pie = self.fig_pie.add_subplot(111)
        self.canvas_pie = FigureCanvasTkAgg(self.fig_pie, master=pie_frame)
        self.canvas_pie.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # Init empty
        self.ax_pie.text(0.5, 0.5, "Waiting for Data...", ha='center')
        self.canvas_pie.draw()

    # ========================== TAB 2: AGENTS ==========================
    def _setup_agents_tab(self):
        # Container
        container = ttk.Frame(self.tab_agents)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Governor (Risk)
        gov_frame = ttk.LabelFrame(container, text="🛡️ Governor Holon (Risk Management)", padding=15)
        gov_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.gov_status = self._metric(gov_frame, "State:", "ACTIVE", 0)
        self.gov_alloc = self._metric(gov_frame, "Max Allocation:", "10.0%", 1)
        self.gov_lev = self._metric(gov_frame, "Leverage Cap:", "5.0x", 2)
        self.gov_trends = self._metric(gov_frame, "Active Trends:", "0", 3)
        
        # Actuator (Execution)
        act_frame = ttk.LabelFrame(container, text="⚙️ Actuator Holon (Execution)", padding=15)
        act_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.act_last_ord = self._metric(act_frame, "Last Order:", "NONE", 0)
        
        # Brains
        brain_frame = ttk.LabelFrame(container, text="🧠 Brain Holons (Strategy & RL)", padding=15)
        brain_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        
        self.ag_regime = self._metric(brain_frame, "Market Regime:", "UNKNOWN", 0)
        self.ag_entropy = self._metric(brain_frame, "Entropy Score:", "0.0000", 1)
        self.ag_model = self._metric(brain_frame, "Strategy Model:", "-", 2)
        self.ag_kalman = self._metric(brain_frame, "Kalman Active:", "-", 3)
        self.ag_ppo_conv = self._metric(brain_frame, "PPO Conviction:", "-", 4)
        self.ag_ppo_reward = self._metric(brain_frame, "PPO Reward:", "-", 5)
        self.ag_lstm_prob = self._metric(brain_frame, "LSTM Prob:", "-", 6)
        self.ag_xgb_prob = self._metric(brain_frame, "XGB Prob:", "-", 7)
        
        # Performance
        perf_frame = ttk.LabelFrame(container, text="📈 Session Performance", padding=15)
        perf_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        
        self.perf_winrate = self._metric(perf_frame, "Win Rate:", "-", 0)
        self.perf_pnl = self._metric(perf_frame, "Realized PnL:", "-", 1)
        self.perf_omega = self._metric(perf_frame, "Omega Ratio:", "-", 2)
        
        # Phase 12: Risk Management
        phase12_frame = ttk.LabelFrame(container, text="🛡️ Phase 12: Risk Management", padding=15)
        phase12_frame.grid(row=2, column=1, sticky="nsew", padx=10, pady=10)
        
        self.p12_kelly = self._metric(phase12_frame, "Kelly Fraction:", "-", 0)
        self.p12_minimax = self._metric(phase12_frame, "Max Risk:", "-", 1)
        self.p12_vol_scalar = self._metric(phase12_frame, "Vol Scalar:", "-", 2)
        self.p12_principal = self._metric(phase12_frame, "Principal:", "$10.00", 3)
        self.p12_exposure = self._metric(phase12_frame, "Total Exposure:", "$0.00", 4)
        self.p12_margin = self._metric(phase12_frame, "Used Margin:", "$0.00", 5)
        self.p12_actual_lev = self._metric(phase12_frame, "Actual Leverage:", "0.00x", 6)
        
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)

    # ========================== TAB 3: CONFIG ==========================
    def _setup_config_tab(self):
        f = ttk.Frame(self.tab_config)
        f.pack(fill=tk.BOTH, expand=True, padx=50, pady=20)
        
        ttk.Label(f, text="System Configuration", style="Header.TLabel").pack(anchor="w", pady=10)
        
        # Symbol
        r1 = ttk.Frame(f); r1.pack(fill=tk.X, pady=5)
        ttk.Label(r1, text="Trading Pair:", width=20).pack(side=tk.LEFT)
        self.symbol_cb = ttk.Combobox(r1, textvariable=self.conf_symbol, values=["XRP/USDT", "ADA/USDT", "MATIC/USDT", "DOGE/USDT"])
        self.symbol_cb.pack(side=tk.LEFT)
        self.symbol_cb.current(0)
        
        # Allocation Slider
        r2 = ttk.Frame(f); r2.pack(fill=tk.X, pady=5)
        ttk.Label(r2, text="Max Allocation %:", width=20).pack(side=tk.LEFT)
        s = ttk.Scale(r2, from_=0.01, to=1.0, variable=self.conf_alloc, orient=tk.HORIZONTAL, length=200)
        s.pack(side=tk.LEFT)
        lbl = ttk.Label(r2, text="0.10")
        lbl.pack(side=tk.LEFT, padx=5)
        # update label on slide
        def update_alloc_lbl(val):
            lbl.config(text=f"{float(val):.2f}")
        s.configure(command=update_alloc_lbl)
        
        # Leverage
        r3 = ttk.Frame(f); r3.pack(fill=tk.X, pady=5)
        ttk.Label(r3, text="Leverage Cap (x):", width=20).pack(side=tk.LEFT)
        ttk.Entry(r3, textvariable=self.conf_leverage).pack(side=tk.LEFT)
        
        # Timeframe
        r4 = ttk.Frame(f); r4.pack(fill=tk.X, pady=5)
        ttk.Label(r4, text="Timeframe:", width=20).pack(side=tk.LEFT)
        self.tf_cb = ttk.Combobox(r4, textvariable=self.conf_timeframe, values=["1m", "5m", "15m", "1h", "4h", "1d"])
        self.tf_cb.pack(side=tk.LEFT)
        self.tf_cb.current(3) # Default to 1h
        
        ttk.Label(f, text="Changes apply on next Bot Start.", foreground="gray").pack(anchor="w", pady=20)

    # ========================== TAB 4: BACKTEST ==========================
    def _setup_backtest_tab(self):
         # Controls
        control_frame = ttk.Frame(self.tab_backtest, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        self.bt_start_btn = ttk.Button(control_frame, text="RUN SIMULATION", command=self.start_backtest)
        self.bt_start_btn.pack(side=tk.LEFT)
        self.bt_progress = ttk.Progressbar(control_frame, mode='determinate', length=300)
        self.bt_progress.pack(side=tk.LEFT, padx=20)

        # Results
        res_frame = ttk.Frame(self.tab_backtest, padding=10)
        res_frame.pack(side=tk.TOP, fill=tk.X)
        self.bt_roi = self._metric(res_frame, "ROI:", "0.00%", 0)
        self.bt_pnl = self._metric(res_frame, "PnL:", "$0.00", 1)
        
        # Chart
        self.chart_frame = ttk.LabelFrame(self.tab_backtest, text="Equity Curve")
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self._setup_chart()

    # ========================== LOGIC ==========================
    def _metric(self, parent, text, default, row):
        # Allow placing in grid or pack depending on parent layout?
        # Assuming parent uses grid for simplicity in most frames
        f = ttk.Frame(parent)
        f.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Label(f, text=text, width=15, anchor="w").pack(side=tk.LEFT)
        l = ttk.Label(f, text=default, style="Data.TLabel", foreground="#333")
        l.pack(side=tk.LEFT)
        return l

    def _setup_chart(self):
        self.fig = Figure(figsize=(5, 6), dpi=100) # Taller for 2 plots
        self.ax = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212, sharex=self.ax)
        
        self.ax.set_title("Equity Curve")
        self.ax.grid(True)
        self.ax2.set_title("Drawdown %")
        self.ax2.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def update_pie_chart(self, assets):
        """Update Live Asset Allocation Pie."""
        self.ax_pie.clear()
        
        if not assets:
            self.ax_pie.text(0.5, 0.5, "No Data", ha='center')
            self.canvas_pie.draw()
            return
            
        labels = list(assets.keys())
        values = list(assets.values())
        
        # Filter small values
        clean_labels = []
        clean_values = []
        for l, v in zip(labels, values):
            if v > 0.01:
                clean_labels.append(l)
                clean_values.append(v)
        
        if not clean_values:
             self.ax_pie.text(0.5, 0.5, "Empty", ha='center')
        else:
            # Enhanced Aesthetics (Phase 30: Glassmorphism/Vibrant)
            colors = plt.cm.Paired(np.linspace(0, 1, len(clean_values)))
            
            wedges, texts, autotexts = self.ax_pie.pie(
                clean_values, labels=clean_labels, autopct='%1.0f%%', 
                startangle=90, colors=colors, pctdistance=0.85, 
                textprops={'fontsize': 7, 'color': 'black'}
            )
            
            # Donut style
            centre_circle = plt.Circle((0,0), 0.70, fc='white')
            self.ax_pie.add_artist(centre_circle)
            
            self.ax_pie.axis('equal')
            
        self.canvas_pie.draw()

    def update_chart(self, history):
        if not history: return
        dates = [h[0] for h in history]
        values = [h[1] for h in history]
        
        # 1. Equity
        self.ax.clear()
        self.ax.plot(dates, values, color='blue', label='Equity')
        self.ax.set_title("Equity Curve")
        self.ax.grid(True)
        self.ax.legend()
        
        # 2. Drawdown
        import pandas as pd
        s = pd.Series(values)
        cummax = s.cummax()
        drawdown = (s - cummax) / cummax
        
        self.ax2.clear()
        self.ax2.fill_between(dates, drawdown, color='red', alpha=0.3, label='Drawdown')
        self.ax2.set_title("Drawdown %")
        self.ax2.grid(True)
        
        self.fig.autofmt_xdate()
        self.canvas.draw()

    def start_live_bot(self):
        if self.is_running_live: return
        
        # Gather Config
        cfg = {
            'symbol': self.conf_symbol.get(),
            'timeframe': self.conf_timeframe.get(),
            'max_allocation': self.conf_alloc.get(),
            'leverage_cap': self.conf_leverage.get()
        }
        
        self.gui_stop_event.clear()
        self.bot_thread = threading.Thread(target=run_bot, args=(self.gui_stop_event, self.gui_queue, cfg))
        self.bot_thread.daemon = True
        self.bot_thread.start()
        
        self.is_running_live = True
        self.status_var.set("LIVE TRADING ACTIVE")
        self.status_label.config(foreground="green")
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.gov_alloc.config(text=f"{cfg['max_allocation']*100:.1f}%")
        self.gov_lev.config(text=f"{cfg['leverage_cap']}x")

    def stop_bot(self):
        if not self.is_running_live: return
        self.gui_stop_event.set()
        self.is_running_live = False
        self.status_var.set("STOPPING...")
        # Note: Actual stop happens when thread sees event
        self.root.after(2000, lambda: self._finalize_stop())
        
    def _finalize_stop(self):
        self.status_var.set("STOPPED")
        self.status_label.config(foreground="red")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def log(self, msg):
        self.log_text.insert(tk.END, msg)
        
        # Apply Highlight Logic to the last few lines
        import re
        
        # Patterns for Syntax Highlighting
        patterns = {
             r"(BUY|LONG|PROFIT|WIN|EXECUTE)": "POSITIVE",
             r"(SELL|SHORT|LOSS|STOP-LOSS|REJECTED|HALT)": "NEGATIVE",
             r"(WARNING|RISK|Drawdown)": "WARNING",
             r"(EXIT|COVER|CLOSE)": "INFO",
             r"(\[.*?\])": "DIM" # Timestamps/Agent Names
        }
        
        # Iterate line by line for the Last 5 lines to catch any split messages
        count = int(self.log_text.index('end-1c').split('.')[0])
        check_rows = 5
        start_line = max(1, count - check_rows)
        
        for i in range(start_line, count + 1):
            line_idx = f"{i}.0"
            line_end = f"{i}.end"
            line_text = self.log_text.get(line_idx, line_end)
            
            for pat, tag in patterns.items():
                for match in re.finditer(pat, line_text, re.IGNORECASE):
                    start = f"{i}.{match.start()}"
                    end = f"{i}.{match.end()}"
                    self.log_text.tag_add(tag, start, end)

        self.log_text.see(tk.END)

    def process_queue(self):
        try:
            while True:
                msg = self.gui_queue.get_nowait()
                mtype = msg.get("type")
                
                if mtype == 'log':
                    self.log(msg.get('message', ''))
                    
                elif mtype == 'summary':
                    # Update Treeview
                    data = msg.get('data', []) # List of dicts
                    # Clear existing
                    for item in self.tree.get_children():
                        self.tree.delete(item)
                    
                    # Insert new rows
                    for row in data:
                        values = (
                            row.get('Symbol'),
                            row.get('Price'),
                            row.get('Regime', '?'),
                            row.get('Entropy', '0.00'),
                            row.get('RSI', '-'),
                            row.get('LSTM', '0.50'),
                            row.get('XGB', '0.50'),
                            row.get('PnL'),
                            row.get('Action')
                        )
                        self.tree.insert('', tk.END, values=values)
                        
                elif mtype == "backtest_result":
                    res = msg.get("data", {})
                    self.bt_roi.config(text=f"{res.get('roi'):.2f}%")
                    self.bt_pnl.config(text=f"${res.get('pnl'):.2f}")
                    self.bt_progress['value'] = 100
                    if 'history' in res:
                         self.update_chart(res['history'])
                         
                elif mtype == 'agent_status':
                    data = msg.get('data', {})
                    # Update Agent Tab Labels
                    self.gov_status.config(text=data.get('gov_state', 'ERR'))
                    self.gov_alloc.config(text=data.get('gov_alloc', '-'))
                    self.gov_lev.config(text=data.get('gov_lev', '-'))
                    self.gov_trends.config(text=data.get('gov_trends', '0'))
                    
                    self.ag_regime.config(text=data.get('regime', '?'))
                    self.ag_entropy.config(text=data.get('entropy', '0.0'))
                    self.ag_model.config(text=data.get('strat_model', '-'))
                    self.ag_kalman.config(text=data.get('kalman_active', '-'))
                    self.ag_ppo_conv.config(text=data.get('ppo_conv', '0.50'))
                    self.ag_ppo_reward.config(text=data.get('ppo_reward', '0.00'))
                    self.ag_lstm_prob.config(text=data.get('lstm_prob', '0.50'))
                    self.ag_xgb_prob.config(text=data.get('xgb_prob', '0.50'))
                    
                    # Update Last Order
                    self.act_last_ord.config(text=data.get('last_order', 'NONE'))
                    
                    # Update Performance
                    self.perf_winrate.config(text=data.get('win_rate', '-'))
                    self.perf_pnl.config(text=data.get('pnl', '-'))
                    self.perf_omega.config(text=data.get('omega', '-'))
                    
                    # Update Risk Metrics
                    self.p12_exposure.config(text=data.get('exposure', '$0.00'))
                    self.p12_margin.config(text=data.get('margin', '$0.00'))
                    self.p12_actual_lev.config(text=data.get('actual_lev', '0.00x'))
                    
                    # Update Pie
                    holdings = data.get('holdings')
                    if holdings:
                         self.update_pie_chart(holdings)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)
            
    # Backtest wrapper
    def start_backtest(self):
        if self.is_running_backtest: return
        self.bt_progress['value'] = 0
        self.is_running_backtest = True
        
        # Pass configuration to backtest
        symbol = self.conf_symbol.get()
        
        def run_and_reset():
            run_backtest(self.status_queue, symbol=symbol)
            self.is_running_backtest = False  # Reset flag on completion
        
        t = threading.Thread(target=run_and_reset)
        t.daemon = True
        t.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = HolonicDashboard(root)
    root.mainloop()
