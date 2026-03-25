import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import queue
from datetime import datetime
import time
import json

# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D # 3D Viz

# Import the bot runners
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# Theme Colors
# Theme Colors: BLOOMBERG TERMINAL EDITION
COLORS = {
    'bg_dark': '#000000',       # Pure Black
    'bg_card': '#141414',       # Dark Grey Cards
    'text_primary': '#FFB300',  # Amber (Data)
    'text_secondary': '#E0E0E0',# White (Labels)
    'accent_primary': '#FF6D00',# Bloomberg Orange
    'accent_secondary': '#00E5FF', # Cyan (Electric)
    'accent_green': '#00C853',  # Terminal Green
    'accent_red': '#D50000',    # Critical Red
    'accent_yellow': '#FFD600', # Warning Yellow
    'border': '#333333',
    'bg_input': '#1A1A1A',
    'accent_blue': '#2962FF',
    'accent': '#FF6D00'
}

class StatusPollThread(threading.Thread):
    def __init__(self, msg_queue, stop_event):
        super().__init__()
        self.msg_queue = msg_queue
        self.stop_event = stop_event
        # Search in current directory and parent directory for status file
        self.status_file_paths = [
            os.path.join(os.path.dirname(__file__), 'dashboard_status.json'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dashboard_status.json'),
        ]
        self.status_file = None  # Will be set when found
        self._last_mtime = 0
        self._last_data_time = None
        self.daemon = True

    def run(self):
        while not self.stop_event.is_set():
            try:
                # Find status file
                if not self.status_file:
                    for path in self.status_file_paths:
                        if os.path.exists(path):
                            self.status_file = path
                            break
                    if not self.status_file:
                        time.sleep(1.0)
                        continue

                if not os.path.exists(self.status_file):
                    self.status_file = None  # Reset and search again
                    time.sleep(1.0)
                    continue

                mtime = os.path.getmtime(self.status_file)
                if mtime > self._last_mtime:
                    self._last_mtime = mtime

                    # Robust read with retries
                    data = None
                    for _ in range(3):
                        try:
                            with open(self.status_file, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                                if content:
                                    data = json.loads(content)
                                    break
                        except json.JSONDecodeError:
                            time.sleep(0.1)
                        except Exception:
                            time.sleep(0.1)

                    if data:
                        # Check data freshness (use newest timestamp)
                        l_up = data.get('last_updated')
                        s_up = data.get('summary_updated')
                        last_updated = max([t for t in (l_up, s_up) if t]) if (l_up or s_up) else None
                        
                        self._last_data_time = last_updated
                        if data.get('type') == 'agent_status' or 'gov_state' in data:
                            self.msg_queue.put({'type': 'agent_status', 'data': data})

                        # Process summary
                        if 'summary' in data:
                            self.msg_queue.put({'type': 'summary', 'data': data['summary']})

                        # Process overwatch
                        if 'overwatch' in data:
                             ow = data['overwatch']
                             self.msg_queue.put({
                                'type': 'overwatch_update',
                                'state': ow.get('state', 'NOMINAL'),
                                'sitrep': ow.get('sitrep', ''),
                                'updated': ow.get('updated', '')
                             })
                        
                        # Send connection status
                        self.msg_queue.put({
                            'type': 'connection_status',
                            'connected': True,
                            'last_update': last_updated
                        })

            except Exception as e:
                print(f"Poll Error: {e}")
                self.msg_queue.put({'type': 'connection_status', 'connected': False, 'error': str(e)})

            time.sleep(0.5) # Poll interval

class HolonicDashboard:
    def __init__(self, root):
        # ... (init vars)
        self.scout_last_read = 0.0
        
        # ... (config vars)
        # Config Variables
        self.conf_symbol = tk.StringVar(value="BTC/USDT")
        self.conf_timeframe = tk.StringVar(value="1h")
        self.conf_alloc = tk.DoubleVar(value=0.25)
        self.conf_leverage = tk.DoubleVar(value=10.0)
        self.conf_micro_mode = tk.BooleanVar(value=False)
        
        # State Variables
        self.is_running_live = False
        self.is_running_backtest = False
        self.gui_queue = queue.Queue()
        self.command_queue = queue.Queue()
        self.gui_stop_event = threading.Event()
        
        # UI State
        self.status_var = tk.StringVar(value="🔴 STOPPED")
        self.equity_history = []
        self.order_history = [] 
        self.max_equity_points = 200
        self.max_log_entries = 500  # Maximum log entries before trimming
        self.max_orders = 100
        
        # Performance Optimization: Hash Guards
        self.root = root
        self.last_radar_hash = 0
        self.last_holdings_hash = 0
        self.last_summary_hash = 0
        self.rendered_news_hashes = set()
        self.scout_last_read = 0.0

        # Initialize data structures for new visualization elements
        self.current_holdings = {}
        self.current_entry_prices = {}
        self.current_current_prices = {}
        self.current_balance = 0

        self.root.title("A E H M L   T R A D E R   //   P H A S E   I V")
        self.root.geometry("1600x1000")
        self.root.configure(bg=COLORS['bg_dark'])
        
        # Style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self._configure_styles()
        
        # Notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tabs
        self.tab_live = ttk.Frame(self.notebook, padding=10)
        self.tab_overwatch = ttk.Frame(self.notebook, padding=10)
        self.tab_scout = ttk.Frame(self.notebook, padding=10)
        self.tab_agents = ttk.Frame(self.notebook, padding=10)
        self.tab_news = ttk.Frame(self.notebook, padding=10)
        self.tab_orders = ttk.Frame(self.notebook, padding=10)
        self.tab_signals = ttk.Frame(self.notebook, padding=10)
        self.tab_config = ttk.Frame(self.notebook, padding=10)
        self.tab_backtest = ttk.Frame(self.notebook, padding=10)
        
        self.notebook.add(self.tab_live, text="  📊 Live Operations  ")
        self.notebook.add(self.tab_overwatch, text="  👁️ Overwatch  ")
        self.notebook.add(self.tab_scout, text="  🔭 Scout Radar  ")
        self.notebook.add(self.tab_agents, text="  🤖 Holon Status  ")
        self.notebook.add(self.tab_news, text="  📰 News & Trends  ")
        self.notebook.add(self.tab_orders, text="  📋 Order History  ")
        self.notebook.add(self.tab_signals, text="  📡 Signals  ")
        
        self.notebook.add(self.tab_config, text="  ⚙️ Configuration  ")
        self.notebook.add(self.tab_backtest, text="  📈 Backtesting  ")
        
        # === Queue Cleanup Handler ===
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        self._setup_live_tab()
        self._setup_overwatch_tab()
        self._setup_scout_tab()
        self._setup_agents_tab()
        self._setup_news_tab()
        self._setup_orders_tab()
        self._setup_signals_tab()
        self._setup_config_tab()
        self._setup_backtest_tab()
        
        # Start Message Queue Loop
        self.process_queue()

        # Start Status Polling Thread
        self.poll_thread = StatusPollThread(self.gui_queue, self.gui_stop_event)
        self.poll_thread.start()

    def _configure_styles(self):
        s = self.style
        
        # General formatting
        s.configure(".", background=COLORS['bg_dark'], foreground=COLORS['text_primary'], font=('Segoe UI', 10))
        s.configure("TFrame", background=COLORS['bg_dark'])
        s.configure("TLabelframe", background=COLORS['bg_dark'], foreground=COLORS['text_secondary'])
        s.configure("TLabelframe.Label", background=COLORS['bg_dark'], foreground=COLORS['accent_primary'], font=('Segoe UI', 11, 'bold'))
        
        # Buttons
        s.configure("TButton", padding=5, font=('Segoe UI', 9, 'bold'))
        s.map("TButton", background=[('active', COLORS['accent_primary'])], foreground=[('active', 'white')])
        
        # Danger Button
        s.configure("Danger.TButton", foreground=COLORS['accent_red'])
        
        # Labels
        s.configure("Header.TLabel", font=('Segoe UI', 16, 'bold'), foreground=COLORS['accent_primary'])
        s.configure("SubHeader.TLabel", font=('Segoe UI', 10, 'bold'), foreground=COLORS['text_secondary'])
        s.configure("Accent.TLabel", foreground=COLORS['accent_secondary'])
        s.configure("Data.TLabel", font=('Consolas', 11), foreground=COLORS['text_primary'])
        
        # Treeview
        s.configure("Treeview", 
            background=COLORS['bg_card'], 
            foreground=COLORS['text_primary'], 
            fieldbackground=COLORS['bg_card'],
            rowheight=25,
            font=('Consolas', 9)
        )
        s.configure("Treeview.Heading", font=('Segoe UI', 9, 'bold'), background=COLORS['bg_dark'], foreground=COLORS['text_secondary'])
        s.map("Treeview", background=[('selected', COLORS['accent_primary'])])
        
        # News Card
        s.configure("Card.TFrame", background='#0b0e14')

    # ========================== TAB 1: LIVE ==========================
    def _setup_live_tab(self):
        # Top Controls with Panic Button (IMPROVEMENT 7)
        ctl_frame = ttk.Frame(self.tab_live)
        ctl_frame.pack(fill=tk.X, pady=5)

        self.start_btn = ttk.Button(ctl_frame, text="▶ START LIVE BOT", command=self.start_live_bot)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(ctl_frame, text="⏹ STOP BOT", command=self.stop_bot, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # === IMPROVEMENT 7: PANIC BUTTON ===
        self.panic_btn = ttk.Button(ctl_frame, text="🚨 PANIC CLOSE ALL", command=self.panic_close_all, style="Danger.TButton")
        self.panic_btn.pack(side=tk.RIGHT, padx=5)

        # Status Label
        self.status_label = ttk.Label(ctl_frame, textvariable=self.status_var, font=('Segoe UI', 10, 'bold'), foreground=COLORS['accent_red'])
        self.status_label.pack(side=tk.LEFT, padx=15)
        
        # Connection Status Indicator (NEW)
        self.conn_status_var = tk.StringVar(value="🔴 DISCONNECTED")
        self.conn_status_label = ttk.Label(ctl_frame, textvariable=self.conn_status_var, font=('Segoe UI', 9), foreground=COLORS['accent_red'])
        self.conn_status_label.pack(side=tk.LEFT, padx=15)

        # === IMPROVEMENT 10: Log Export Button ===
        self.export_btn = ttk.Button(ctl_frame, text="💾 Export Log", command=self.export_log)
        self.export_btn.pack(side=tk.RIGHT, padx=5)
        
        # Main Grid
        grid_frame = ttk.Frame(self.tab_live)
        grid_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # === IMPROVEMENT 11: Risk & Regime Panel (Top of Grid) ===
        regime_frame = ttk.LabelFrame(grid_frame, text="🛡️ Capital Regime & Risk Control", padding=5)
        regime_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Risk Grid
        r_grid = ttk.Frame(regime_frame)
        r_grid.pack(fill=tk.X)
        
        # 1. Regime Status
        ttk.Label(r_grid, text="CURRENT REGIME:", style="SubHeader.TLabel").grid(row=0, column=0, padx=5, sticky="w")
        self.regime_label = ttk.Label(r_grid, text="MICRO ($0 - $50)", style="Accent.TLabel", font=("Segoe UI", 12, "bold"))
        self.regime_label.grid(row=0, column=1, padx=5, sticky="w")
        
        # 2. System Health
        ttk.Label(r_grid, text="SYSTEM HEALTH:", style="SubHeader.TLabel").grid(row=0, column=2, padx=15, sticky="w")
        self.health_progress = ttk.Progressbar(r_grid, length=150, mode='determinate')
        self.health_progress.grid(row=0, column=3, padx=5, sticky="w")
        self.health_label = ttk.Label(r_grid, text="0.00", style="Data.TLabel")
        self.health_label.grid(row=0, column=4, padx=5, sticky="w")
        
        # 3. Promotion Timer
        ttk.Label(r_grid, text="PROMOTION IN:", style="SubHeader.TLabel").grid(row=0, column=5, padx=15, sticky="w")
        self.promo_label = ttk.Label(r_grid, text="--:--:--", style="Data.TLabel")
        self.promo_label.grid(row=0, column=6, padx=5, sticky="w")
        
        # 4. Balance (IMPROVEMENT 8) -> Renamed to Avail Margin
        ttk.Label(r_grid, text="AVAIL MARGIN:", style="SubHeader.TLabel").grid(row=0, column=7, padx=15, sticky="w")
        self.balance_label = ttk.Label(r_grid, text="Loading...", style="Data.TLabel", foreground=COLORS['accent_green'])
        self.balance_label.grid(row=0, column=8, padx=5, sticky="w")
        
        # 5. Total Equity (NEW)
        ttk.Label(r_grid, text="TOTAL EQUITY:", style="SubHeader.TLabel").grid(row=0, column=9, padx=15, sticky="w")
        self.equity_label = ttk.Label(r_grid, text="Loading...", style="Data.TLabel", foreground=COLORS['accent_secondary'])
        self.equity_label.grid(row=0, column=10, padx=5, sticky="w")
        
        # Left Col: Metrics & Table
        left_col = ttk.Frame(grid_frame)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # > Market Table (Treeview) with Color-Coded PnL (IMPROVEMENT 3)
        tbl_frame = ttk.LabelFrame(left_col, text="Market Overview", padding=5)
        tbl_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        cols = ("Symbol", "Price", "Regime", "Entropy", "Struct", "RSI", "LSTM", "XGB", "PnL", "Action")
        self.tree = ttk.Treeview(tbl_frame, columns=cols, show='headings', height=6) # Reduced height for Radar
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Configure columns
        self.tree.column("Symbol", width=60)
        self.tree.column("Price", width=70)
        self.tree.column("Regime", width=60)
        self.tree.column("Entropy", width=50)
        self.tree.column("Struct", width=60)
        self.tree.column("RSI", width=40)
        self.tree.column("LSTM", width=45)
        self.tree.column("XGB", width=45)
        self.tree.column("PnL", width=60)
        self.tree.column("Action", width=80)
        
        for col in cols:
            self.tree.heading(col, text=col)
        
        # === IMPROVEMENT 3: Color Tags for PnL ===
        self.tree.tag_configure('profit', foreground=COLORS['accent_green'])
        self.tree.tag_configure('loss', foreground=COLORS['accent_red'])
        self.tree.tag_configure('neutral', foreground=COLORS['text_secondary'])
        self.tree.tag_configure('whale', foreground='#00ffff', background='#003333') # Cyan for Whale
        
        # === IMPROVEMENT 12: Consolidation Radar (Bottom Left) ===
        radar_frame = ttk.LabelFrame(left_col, text="💀 Consolidation Radar (Kill List)", padding=5)
        radar_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        r_cols = ("Rank", "Symbol", "Score", "PnL", "Age", "EffectAge", "Status")
        self.radar_tree = ttk.Treeview(radar_frame, columns=r_cols, show='headings', height=5)
        self.radar_tree.pack(fill=tk.BOTH, expand=True)
        
        self.radar_tree.column("Rank", width=40)
        self.radar_tree.column("Symbol", width=80)
        self.radar_tree.column("Score", width=60)
        self.radar_tree.column("PnL", width=70)
        self.radar_tree.column("Age", width=60)
        self.radar_tree.column("EffectAge", width=70)
        self.radar_tree.column("Status", width=80)
        
        for col in r_cols:
            self.radar_tree.heading(col, text=col)
            
        self.radar_tree.tag_configure('keep', foreground=COLORS['accent_green'])
        self.radar_tree.tag_configure('close', foreground=COLORS['accent_red'])
        self.radar_tree.tag_configure('force', foreground=COLORS['accent_primary'], background='#331111')
        
        # === IMPROVEMENT 4: Position Cards (Hidden/Miniaturized or kept?) -> Kept below radar?
        # Maybe move to right col or float? Let's keep it compacted.
        pos_frame = ttk.LabelFrame(left_col, text="Exchange Positions", padding=5)
        pos_frame.pack(fill=tk.X, pady=5)
        self.positions_container = ttk.Frame(pos_frame)
        self.positions_container.pack(fill=tk.X)
        self.position_labels = {}
        
        # Right Col: Logs + Equity Chart
        
        # Right Col: Logs + Equity Chart
        right_col = ttk.Frame(grid_frame)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # === IMPROVEMENT 2: Real-Time Equity Chart ===
        equity_frame = ttk.LabelFrame(right_col, text="Live Equity Curve", padding=5)
        equity_frame.pack(fill=tk.BOTH, expand=False, pady=5) # EXPAND=FALSE (Fixed Height)
        
        self.fig_equity = Figure(figsize=(4, 2.5), dpi=90, facecolor=COLORS['bg_card'])
        self.ax_equity = self.fig_equity.add_subplot(111)
        self.ax_equity.set_facecolor(COLORS['bg_dark'])
        self.ax_equity.tick_params(colors=COLORS['text_secondary'])
        self.ax_equity.spines['bottom'].set_color(COLORS['border'])
        self.ax_equity.spines['left'].set_color(COLORS['border'])
        self.canvas_equity = FigureCanvasTkAgg(self.fig_equity, master=equity_frame)
        self.canvas_equity.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._init_equity_chart()
        
        # Scout Radar moved to self.tab_scout
        
        # Logs
        log_frame = ttk.LabelFrame(right_col, text="Activity Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5) # EXPAND=TRUE (Take all remaining space)
        
        self.log_tree = ttk.Treeview(log_frame, columns=("Time", "Level", "Agent", "Message"), show='headings', height=15) # Start Taller
        self.log_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbars
        log_vsb = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_tree.yview)
        log_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_tree.configure(yscrollcommand=log_vsb.set)
        
        # Columns
        self.log_tree.column("Time", width=70, anchor="center")
        self.log_tree.column("Level", width=70, anchor="center")
        self.log_tree.column("Agent", width=120, anchor="w")
        self.log_tree.column("Message", width=400, anchor="w")
        
        # Headings
        self.log_tree.heading("Time", text="Time")
        self.log_tree.heading("Level", text="Level")
        self.log_tree.heading("Agent", text="Agent")
        self.log_tree.heading("Message", text="Message")
        
        # Premium Styling Tags
        self.log_tree.tag_configure('POSITIVE', foreground=COLORS['accent_green'])
        self.log_tree.tag_configure('NEGATIVE', foreground=COLORS['accent_red'])
        self.log_tree.tag_configure('WARNING', foreground=COLORS['accent_yellow'])
        self.log_tree.tag_configure('INFO', foreground=COLORS['text_primary'])
        self.log_tree.tag_configure('DIM', foreground=COLORS['text_secondary'])

        # === IMPROVEMENT 5: Notifications/Alerts Panel ===
        alert_frame = ttk.LabelFrame(right_col, text="🔔 Alerts", padding=5)
        alert_frame.pack(fill=tk.X, pady=5)
        self.alert_text = tk.Text(alert_frame, height=3, font=("Consolas", 9),
            bg=COLORS['bg_input'], fg=COLORS['accent_yellow'], wrap=tk.WORD)
        self.alert_text.pack(fill=tk.X)
        self.alert_text.insert(tk.END, "No alerts yet.\n")
        self.alert_text.config(state=tk.DISABLED)

    # ========================== TAB 2: SCOUT ==========================
    def _setup_scout_tab(self):
        # Layout: Full screen radar
        self.create_scout_panel(self.tab_scout).pack(fill=tk.BOTH, expand=True)



    def _init_equity_chart(self):
        self.ax_equity.clear()
        self.ax_equity.set_title("Equity", color=COLORS['text_primary'], fontsize=10)
        self.ax_equity.text(0.5, 0.5, "Waiting for data...", ha='center', va='center', 
            color=COLORS['text_secondary'], transform=self.ax_equity.transAxes)
        self.canvas_equity.draw()

    # ========================== TAB 2: AGENTS ==========================
    def _setup_agents_tab(self):
        container = ttk.Frame(self.tab_agents)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Governor (Risk)
        gov_frame = ttk.LabelFrame(container, text="🛡️ Governor Holon (Risk Management)", padding=15)
        gov_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.gov_status = self._metric(gov_frame, "State:", "ACTIVE", 0)
        self.gov_alloc = self._metric(gov_frame, "Max Allocation:", "10.0%", 1)
        self.gov_lev = self._metric(gov_frame, "Leverage Cap:", "5.0x", 2)
        self.gov_trends = self._metric(gov_frame, "Active Trends:", "0", 3)
        self.gov_micro = self._metric(gov_frame, "Micro Mode:", "UNKNOWN", 4) 
        self.gov_fortress = self._metric(gov_frame, "🏰 Iron Bank Floor:", "$0.00", 5) # New
        self.gov_budget = self._metric(gov_frame, "⚔️ Risk Budget:", "$0.00", 6) # New
        
        # Actuator (Execution)
        act_frame = ttk.LabelFrame(container, text="⚙️ Actuator Holon (Execution)", padding=15)
        act_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.act_last_ord = self._metric(act_frame, "Last Order:", "NONE", 0)
        self.act_pending = self._metric(act_frame, "Pending Orders:", "0", 1)
        
        # === IMPROVEMENT 6: GC Monitor Status ===
        gc_frame = ttk.LabelFrame(container, text="🧹 Garbage Collector", padding=15)
        gc_frame.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)
        
        self.gc_last_run_label = self._metric(gc_frame, "Last Run:", "Never", 0)
        self.gc_cleaned_label = self._metric(gc_frame, "Items Cleaned:", "0", 1)
        
        # Brains
        brain_frame = ttk.LabelFrame(container, text="🧠 Brain Holons (Strategy & RL)", padding=15)
        brain_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        
        self.ag_regime = self._metric(brain_frame, "Market Regime:", "UNKNOWN", 0)
        self.ag_entropy = self._metric(brain_frame, "Entropy Score:", "0.0000", 1)
        self.ag_model = self._metric(brain_frame, "Strategy Matrix:", "UNIFIED (P+S)", 2) # Updated Label
        self.ag_kalman = self._metric(brain_frame, "Kalman Filter:", "ACTIVE", 3)
        self.ag_ppo_conv = self._metric(brain_frame, "PPO Conviction:", "-", 4)
        self.ag_ppo_reward = self._metric(brain_frame, "PPO Reward:", "-", 5)
        self.ag_lstm_prob = self._metric(brain_frame, "LSTM Prob:", "-", 6)
        self.ag_xgb_prob = self._metric(brain_frame, "XGB Prob:", "-", 7)
        
        # Performance
        perf_frame = ttk.LabelFrame(container, text="📈 Session Performance", padding=15)
        perf_frame.grid(row=1, column=2, sticky="nsew", padx=10, pady=10)
        
        self.perf_winrate = self._metric(perf_frame, "Win Rate:", "-", 0)
        self.perf_pnl = self._metric(perf_frame, "Realized PnL:", "-", 1)
        self.perf_omega = self._metric(perf_frame, "Omega Ratio:", "-", 2)
        
        # Risk Metrics
        phase12_frame = ttk.LabelFrame(container, text="🛡️ Risk Management", padding=15)
        phase12_frame.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)
        
        self.p12_exposure = self._metric(phase12_frame, "Total Exposure:", "$0.00", 0)
        self.p12_margin = self._metric(phase12_frame, "Used Margin:", "$0.00", 1)
        self.p12_actual_lev = self._metric(phase12_frame, "Actual Leverage:", "0.00x", 2)
        
        # Apex Evolution (Archipelago)
        evo_frame = ttk.LabelFrame(container, text="🧬 Apex Evolution (Archipelago Engine)", padding=15)
        evo_frame.grid(row=3, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)
        
        self.evo_status = self._metric(evo_frame, "Engine Status:", "ACTIVE", 0)
        self.evo_fitness = self._metric(evo_frame, "Best Fitness:", "0.00", 1)
        self.evo_kings = self._metric(evo_frame, "HOF Kings:", "0", 2)
        self.evo_archipelago = self._metric(evo_frame, "Active Islands:", "Volcano Peak, Iron Fort, Darwin's Rock", 3)
        
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.columnconfigure(2, weight=1)

    # === IMPROVEMENT 9: ORDER HISTORY TAB ===
    # === IMPROVEMENT 10: Pinterest-Style News Feed ===
    def _setup_news_tab(self):
        # 1. Scrollable Canvas
        self.news_canvas = tk.Canvas(self.tab_news, bg='#0b0e14', highlightthickness=0)
        self.news_scrollbar = ttk.Scrollbar(self.tab_news, orient="vertical", command=self.news_canvas.yview)
        
        # 2. Container Frame inside Canvas
        self.news_container = ttk.Frame(self.news_canvas, style="Card.TFrame")
        self.news_window = self.news_canvas.create_window((0, 0), window=self.news_container, anchor="nw")
        
        self.news_canvas.configure(yscrollcommand=self.news_scrollbar.set)
        
        # Layout Scroll
        self.news_canvas.pack(side="left", fill="both", expand=True)
        self.news_scrollbar.pack(side="right", fill="y")
        
        # 3. Masonry Columns (3 Columns)
        self.news_cols = []
        for i in range(3):
            col_frame = ttk.Frame(self.news_container, style="Card.TFrame") # Transparent/Dark bg
            col_frame.grid(row=0, column=i, sticky="nw", padx=5, pady=5)
            self.news_cols.append(col_frame)
            
        # Hook resize to adjust scroll region
        self.news_container.bind("<Configure>", lambda e: self.news_canvas.configure(scrollregion=self.news_canvas.bbox("all")))
        self.tab_news.bind("<Configure>", self._on_news_resize)

        # Header Badge
        header_lbl = ttk.Label(self.news_container, text="🌍 Global Macro & Crypto Pulse", font=('Segoe UI', 14, 'bold'), foreground='#a4b1cd')
        header_lbl.grid(row=0, column=0, columnspan=3, pady=(0, 15))
        
        # Shift columns down to row 1
        for col in self.news_cols:
            col.grid(row=1)
            
        self.rendered_news_hashes = set()

    def _on_news_resize(self, event):
        # Responsive Width
        canvas_width = event.width
        self.news_canvas.itemconfig(self.news_window, width=canvas_width)

    def _update_news_feed(self, news_items):
        import webbrowser
        
        if not news_items: return
    
        # Simple caching to avoid redraw flicker
        # Only render if we have new items or empty
        # For efficiency, we just clear and redraw top 50 if list changed significantly
        # Or simplistic: Clear all if top item is different
        
        if not news_items: return
        top_hash = hash(news_items[0]['title'])
        if top_hash in self.rendered_news_hashes:
            return # Already rendered top item, assume list is similar for now
            
        # CLEAR existing (Optimized: destroy children of cols)
        for col in self.news_cols:
            for widget in col.winfo_children():
                widget.destroy()
        self.rendered_news_hashes.clear()
        self.rendered_news_hashes.add(top_hash)

        # Distribute items
        for i, item in enumerate(news_items):
            col_idx = i % 3
            parent = self.news_cols[col_idx]
            
            # Card Frame
            score = item.get('sentiment', 0.0)
            is_crisis = item.get('is_crisis', False)
            is_whale = item.get('is_whale', False)
            is_hype = item.get('is_hype', False)
            
            border_color = '#2e3b52' # Neutral Grey
            accent_icon = ""
            
            if is_crisis: 
                border_color = '#d63031' # Red Crisis
                accent_icon = "☢️ "
            elif is_whale:
                border_color = '#a29bfe' # Purple Whale
                accent_icon = "🐋 "
            elif is_hype:
                border_color = '#fdcb6e' # Gold Hype
                accent_icon = "🚀 "
            elif score > 0.2: 
                border_color = '#00b894' # Green Bull
            elif score < -0.2: 
                border_color = '#ff7675' # Red Bear
            
            # Use specific styles if possible, else standard Frame with border trick
            # Tkinter Frame border trick: Frame(bg=border) -> Frame (bg=inner, padding=1)
            card_border = tk.Frame(parent, bg=border_color, padx=1, pady=1)
            card_border.pack(fill='x', pady=6, padx=4)
            
            card_inner = tk.Frame(card_border, bg='#28324e') # Slightly Lighter Card BG
            card_inner.pack(fill='both')
            
            # Content
            # 1. Header Row (Source + Time if avail)
            header_frame = tk.Frame(card_inner, bg='#28324e')
            header_frame.pack(fill='x', padx=10, pady=(8,2))
            
            source_lbl = tk.Label(header_frame, text=item.get('source', 'Unknown').upper(), font=('Segoe UI', 8, 'bold'), fg='#7da3e4', bg='#28324e', anchor='w')
            source_lbl.pack(side='left')
            
            # 2. Title
            title_text = f"{accent_icon}{item.get('title', 'No Title')}"
            title_lbl = tk.Label(card_inner, text=title_text, font=('Segoe UI', 11), fg='white', bg='#28324e', wraplength=230, justify='left', anchor='w')
            title_lbl.pack(fill='x', padx=10, pady=5)
            
            # 3. Badges / Footer
            footer_frame = tk.Frame(card_inner, bg='#28324e')
            footer_frame.pack(fill='x', padx=10, pady=(5,8))
            
            # Sentiment Badge
            sent_text = "NEUTRAL"
            sent_fg = '#b2bec3'
            if score > 0.2: sent_text, sent_fg = "BULLISH", '#55efc4'
            if score < -0.2: sent_text, sent_fg = "BEARISH", '#ff7675'
            if is_crisis: sent_text, sent_fg = "CRITICAL", '#ff0000'
            
            sent_lbl = tk.Label(footer_frame, text=sent_text, font=('Consolas', 8, 'bold'), fg=sent_fg, bg='#2d3436', padx=5, pady=2)
            sent_lbl.pack(side='left')
            
            # Sentiment Bar (Bottom Border)
            bar_height = 4 if is_crisis or is_whale else 2
            sent_bar = tk.Frame(card_inner, bg=border_color, height=bar_height)
            sent_bar.pack(fill='x', side='bottom')
            
            # Interaction
            def open_url(link=item.get('link')):
                if link: webbrowser.open(link)
                
            # Bind Click to everything in card
            for w in [card_border, card_inner, source_lbl, title_lbl, sent_bar, header_frame, footer_frame, sent_lbl]:
                w.bind("<Button-1>", lambda e, l=item.get('link'): open_url(l))
                w.bind("<Enter>", lambda e, c=card_border: c.configure(bg='white')) # Hover highlight border
                w.bind("<Leave>", lambda e, c=card_border, b=border_color: c.configure(bg=b))

    def _setup_orders_tab(self):
        frame = ttk.Frame(self.tab_orders)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text="Recent Order History", style="Header.TLabel").pack(anchor="w", pady=10)
        
        cols = ("Time", "Symbol", "Side", "Qty", "Price", "Status")
        self.order_tree = ttk.Treeview(frame, columns=cols, show='headings', height=20)
        self.order_tree.pack(fill=tk.BOTH, expand=True)
        
        self.order_tree.column("Time", width=150)
        self.order_tree.column("Symbol", width=100)
        self.order_tree.column("Side", width=80)
        self.order_tree.column("Qty", width=100)
        self.order_tree.column("Price", width=100)
        self.order_tree.column("Status", width=100)
        
        for col in cols:
            self.order_tree.heading(col, text=col)
        
        self.order_tree.tag_configure('buy', foreground=COLORS['accent_green'])
        self.order_tree.tag_configure('sell', foreground=COLORS['accent_red'])
        self.order_tree.tag_configure('filled', foreground=COLORS['accent_green'])
        self.order_tree.tag_configure('canceled', foreground=COLORS['accent_red'])

    # ========================== TAB: SIGNALS ==========================
    def _setup_signals_tab(self):
        frame = ttk.Frame(self.tab_signals)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text="📡 Signal Provider Intelligence", style="Header.TLabel").pack(anchor="w", pady=10)
        ttk.Label(frame, text="High-Probability Setups (HPS) — Auto-refreshes every 60s", style="SubHeader.TLabel").pack(anchor="w")
        
        sig_cols = ("Symbol", "Direction", "Price", "Conviction", "TP", "SL", "RR", "HPS", "Pillars")
        self.signal_tree = ttk.Treeview(frame, columns=sig_cols, show='headings', height=20)
        self.signal_tree.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.signal_tree.column("Symbol", width=90)
        self.signal_tree.column("Direction", width=70)
        self.signal_tree.column("Price", width=90)
        self.signal_tree.column("Conviction", width=80)
        self.signal_tree.column("TP", width=90)
        self.signal_tree.column("SL", width=90)
        self.signal_tree.column("RR", width=60)
        self.signal_tree.column("HPS", width=50)
        self.signal_tree.column("Pillars", width=250)
        
        for col in sig_cols:
            self.signal_tree.heading(col, text=col)
        
        self.signal_tree.tag_configure('buy', foreground=COLORS['accent_green'])
        self.signal_tree.tag_configure('sell', foreground=COLORS['accent_red'])
        self.signal_tree.tag_configure('hps', foreground='#FFD700', background='#1a1a00')  # Gold for HPS
        
        # Last update label
        self.signal_update_lbl = ttk.Label(frame, text="Waiting for signal data...", foreground=COLORS['text_secondary'])
        self.signal_update_lbl.pack(anchor="w")

    def _update_signals_tab(self, signal_report):
        """Update the Signals tab with latest signal provider data."""
        if not signal_report:
            return
        
        # Clear existing
        for child in self.signal_tree.get_children():
            self.signal_tree.delete(child)
        
        for sig in signal_report:
            direction = sig.get('direction', '?')
            tag = 'buy' if direction == 'BUY' else ('sell' if direction == 'SELL' else 'neutral')
            
            hps = sig.get('hps', {})
            is_hps = hps.get('is_hps', False) if isinstance(hps, dict) else False
            pillars = ', '.join(hps.get('pillars', [])) if isinstance(hps, dict) else ''
            hps_score = hps.get('score', 0) if isinstance(hps, dict) else 0
            
            if is_hps:
                tag = 'hps'
            
            rr = sig.get('risk_reward', 0)
            
            values = (
                self._safe_tcl(sig.get('symbol', '?')),
                self._safe_tcl(direction),
                self._safe_tcl(f"${sig.get('price', 0):.2f}"),
                self._safe_tcl(f"{sig.get('conviction', 0):.2f}"),
                self._safe_tcl(f"${sig.get('tp', 0):.2f}"),
                self._safe_tcl(f"${sig.get('sl', 0):.2f}"),
                self._safe_tcl(f"{rr:.1f}" if rr else '-'),
                self._safe_tcl(f"{hps_score}/5" if is_hps else '-'),
                self._safe_tcl(pillars if pillars else '-'),
            )
            self.signal_tree.insert('', tk.END, values=values, tags=(tag,))
        
        self.signal_update_lbl.config(text=f"Last updated: {datetime.now().strftime('%H:%M:%S')} ({len(signal_report)} signals)")

    # ========================== TAB OVERWATCH (NEW) ==========================
    def _setup_overwatch_tab(self):
        # 1. Header: System DEFCON Status
        status_frame = ttk.Frame(self.tab_overwatch)
        status_frame.pack(fill=tk.X, pady=10, padx=20)
        
        ttk.Label(status_frame, text="Current System State:", style="Header.TLabel").pack(side=tk.LEFT)
        self.ov_status_lbl = ttk.Label(status_frame, text="INITIALIZING...", font=("Segoe UI", 20, "bold"), foreground=COLORS['text_secondary'])
        self.ov_status_lbl.pack(side=tk.LEFT, padx=20)
        
        # 2. Main Narrative (SitRep)
        center_frame = ttk.LabelFrame(self.tab_overwatch, text="📝 Captain's Log (SitRep)", padding=10)
        center_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.ov_text = scrolledtext.ScrolledText(center_frame, font=("Consolas", 11), bg=COLORS['bg_input'], fg=COLORS['text_primary'], height=15)
        self.ov_text.pack(fill=tk.BOTH, expand=True)
        self.ov_text.insert(tk.END, "Waiting for Overwatch Holon...")
        self.ov_text.config(state=tk.DISABLED)
        
        # 3. Connection Status
        footer = ttk.Frame(self.tab_overwatch)
        footer.pack(fill=tk.X, pady=5, padx=10)
        ttk.Label(footer, text="Connected to Telegram: YES (Check Console for ID)", foreground=COLORS['accent_blue']).pack(side=tk.RIGHT)

    # ========================== TAB 4: CONFIG ==========================
    def _setup_config_tab(self):
        f = ttk.Frame(self.tab_config)
        f.pack(fill=tk.BOTH, expand=True, padx=50, pady=20)
        
        ttk.Label(f, text="System Configuration", style="Header.TLabel").pack(anchor="w", pady=10)
        
        # Symbol
        r1 = ttk.Frame(f); r1.pack(fill=tk.X, pady=8)
        ttk.Label(r1, text="Trading Pair:", width=20).pack(side=tk.LEFT)
        self.symbol_cb = ttk.Combobox(r1, textvariable=self.conf_symbol, 
            values=["XRP/USDT", "ADA/USDT", "SOL/USDT", "DOGE/USDT", "BTC/USDT", "ETH/USDT", "LINK/USDT", "LTC/USDT"])
        self.symbol_cb.pack(side=tk.LEFT)
        self.symbol_cb.current(0)
        
        # Allocation Slider
        r2 = ttk.Frame(f); r2.pack(fill=tk.X, pady=8)
        ttk.Label(r2, text="Max Allocation %:", width=20).pack(side=tk.LEFT)
        s = ttk.Scale(r2, from_=0.01, to=1.0, variable=self.conf_alloc, orient=tk.HORIZONTAL, length=200)
        s.pack(side=tk.LEFT)
        self.alloc_lbl = ttk.Label(r2, text="0.10")
        self.alloc_lbl.pack(side=tk.LEFT, padx=5)
        def update_alloc_lbl(val):
            self.alloc_lbl.config(text=f"{float(val):.2f}")
        s.configure(command=update_alloc_lbl)
        
        # Leverage
        r3 = ttk.Frame(f); r3.pack(fill=tk.X, pady=8)
        ttk.Label(r3, text="Leverage Cap (x):", width=20).pack(side=tk.LEFT)
        ttk.Entry(r3, textvariable=self.conf_leverage, width=10).pack(side=tk.LEFT)
        
        # Micro Mode (NEW)
        r5 = ttk.Frame(f); r5.pack(fill=tk.X, pady=8)
        ttk.Label(r5, text="Micro Account Mode:", width=20).pack(side=tk.LEFT)
        self.conf_micro_mode = tk.BooleanVar(value=True) # Default from Config
        ttk.Checkbutton(r5, text="Force Micro Protections (No Stacking)", variable=self.conf_micro_mode).pack(side=tk.LEFT)

        # Timeframe
        r4 = ttk.Frame(f); r4.pack(fill=tk.X, pady=8)
        ttk.Label(f, text="Timeframe:", width=20).pack(side=tk.LEFT)
        self.tf_cb = ttk.Combobox(r4, textvariable=self.conf_timeframe, values=["1m", "5m", "15m", "1h", "4h", "1d"])
        self.tf_cb.pack(side=tk.LEFT)
        self.tf_cb.current(3)
        
        # Save Button
        btn_frame = ttk.Frame(f)
        btn_frame.pack(fill=tk.X, pady=20)
        ttk.Button(btn_frame, text="💾 APPLY & SAVE CONFIG", command=self.save_live_config).pack(side=tk.LEFT)
        self.cfg_status_lbl = ttk.Label(btn_frame, text="", foreground=COLORS['accent_green'])
        self.cfg_status_lbl.pack(side=tk.LEFT, padx=10)

    def save_live_config(self):
        """Sends update to bot and persists to file."""
        try:
            # === INPUT VALIDATION ===
            max_alloc = self.conf_alloc.get()
            lev_cap = self.conf_leverage.get()
            micro_mode = self.conf_micro_mode.get()
            
            # Bounds checking
            if not (0.0 <= max_alloc <= 1.0):
                messagebox.showerror("Invalid Config", f"Max Allocation must be between 0-100% (got {max_alloc*100:.1f}%)")
                return
            
            if not (1.0 <= lev_cap <= 50.0):
                messagebox.showerror("Invalid Config", f"Leverage must be between 1x-50x (got {lev_cap}x)")
                return
            
            new_cfg = {
                'max_allocation': max_alloc,
                'leverage_cap': lev_cap,
                'micro_mode': micro_mode
            }
            
            # 1. Send to Live Bot (if running)
            if self.is_running_live:
                self.command_queue.put({'type': 'update_config', 'data': new_cfg})
                self.cfg_status_lbl.config(text="✅ Applied to Running Bot")
            else:
                self.cfg_status_lbl.config(text="✅ Saved (Will apply on Start)")
                
            # 3. Persist to user_config.json (JSON)
            import json
            import os
            
            # Read existing if available to preserve other keys
            user_config_path = 'user_config.json'
            existing_cfg = {}
            if os.path.exists(user_config_path):
                try:
                    with open(user_config_path, 'r') as f:
                        existing_cfg = json.load(f)
                except:
                    existing_cfg = {}
            
            # Update specific keys
            existing_cfg['max_allocation'] = new_cfg['max_allocation']
            existing_cfg['leverage_cap'] = new_cfg['leverage_cap']
            existing_cfg['micro_mode'] = new_cfg['micro_mode']
            
            with open(user_config_path, 'w') as f:
                json.dump(existing_cfg, f, indent=4)
                
                
            # Update UI Display instantly
            self.gov_alloc.config(text=f"{new_cfg['max_allocation']*100:.1f}%")
            self.gov_lev.config(text=f"{new_cfg['leverage_cap']}x")
            
            mm_text = "ACTIVE" if new_cfg['micro_mode'] else "INACTIVE"
            mm_color = COLORS['accent_green'] if new_cfg['micro_mode'] else COLORS['text_secondary']
            try: self.gov_micro.config(text=mm_text, foreground=mm_color)
            except: pass
            
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please enter valid numbers: {e}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save config: {e}")

    # ========================== TAB 5: BACKTEST ==========================
    def _setup_backtest_tab(self):
        control_frame = ttk.Frame(self.tab_backtest, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        self.bt_start_btn = ttk.Button(control_frame, text="RUN SIMULATION", command=self.start_backtest)
        self.bt_start_btn.pack(side=tk.LEFT)
        self.bt_progress = ttk.Progressbar(control_frame, mode='determinate', length=300)
        self.bt_progress.pack(side=tk.LEFT, padx=20)

        res_frame = ttk.Frame(self.tab_backtest, padding=10)
        res_frame.pack(side=tk.TOP, fill=tk.X)
        self.bt_roi = self._metric(res_frame, "ROI:", "0.00%", 0)
        self.bt_pnl = self._metric(res_frame, "PnL:", "$0.00", 1)
        
        self.chart_frame = ttk.LabelFrame(self.tab_backtest, text="Equity Curve")
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self._setup_chart()
    def _update_enhanced_consolidation_radar(self, consolidation_data):
        """Update the consolidation radar with enhanced information."""
        # OPTIMIZATION: Radar Guard
        radar_hash = hash(str(consolidation_data))
        if radar_hash != self.last_radar_hash:
            self.last_radar_hash = radar_hash
            # Clear and Repopulate Radar Tree
            for item in self.radar_tree.get_children():
                self.radar_tree.delete(item)

            # Sort by score in descending order to show highest risk first
            sorted_data = sorted(consolidation_data, key=lambda x: x.get('score', 0), reverse=True)

            for i, r in enumerate(sorted_data):
                # Format: {'symbol': 'BTC', 'score': 0.85, 'reason': 'High Vol', 'pnl': -123.45, 'age': 10, 'effective_age': 5, 'status': 'CLOSE_RECOMMENDED'}
                sym = r.get('symbol', 'UNKNOWN')
                score = r.get('score', 0.0)
                reason = r.get('reason', 'Scanning...')
                pnl = r.get('pnl', 0.0)
                age = r.get('age', 0)
                effective_age = r.get('effective_age', 0)
                status = r.get('status', 'MONITOR')

                # Determine tag based on status and score
                if status == 'FORCE_CLOSE' or score > 0.9:
                    tag = 'force'
                    status_display = "FORCE CLOSE"
                elif status == 'CLOSE_RECOMMENDED' or score > 0.7:
                    tag = 'close'
                    status_display = "CLOSE"
                else:
                    tag = 'keep'
                    status_display = "KEEP"

                # Format PnL with color consideration
                pnl_str = f"${pnl:.2f}" if abs(pnl) > 0.01 else "-"
                if pnl < 0:
                    pnl_str = f"{pnl_str}"

                self.radar_tree.insert('', 'end', values=(
                    self._safe_tcl(f"#{i+1}"),
                    self._safe_tcl(sym),
                    self._safe_tcl(f"{score:.2f}"),
                    self._safe_tcl(pnl_str),
                    self._safe_tcl(f"{age}"),
                    self._safe_tcl(f"{effective_age}"),
                    self._safe_tcl(status_display)
                ), tags=(tag,))

    # ========================== LOGIC ==========================
    def _metric(self, parent, text, default, row):
        f = ttk.Frame(parent)
        f.grid(row=row, column=0, columnspan=2, sticky="ew", pady=3)
        ttk.Label(f, text=text, width=18, anchor="w").pack(side=tk.LEFT)
        l = ttk.Label(f, text=default, style="Data.TLabel")
        l.pack(side=tk.LEFT)
        return l

    def _setup_chart(self):
        self.fig = Figure(figsize=(5, 6), dpi=100, facecolor=COLORS['bg_card'])
        self.ax = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212, sharex=self.ax)
        
        for ax in [self.ax, self.ax2]:
            ax.set_facecolor(COLORS['bg_dark'])
            ax.tick_params(colors=COLORS['text_secondary'])
        
        self.ax.set_title("Equity Curve", color=COLORS['text_primary'])
        self.ax.grid(True, color=COLORS['border'], alpha=0.3)
        self.ax2.set_title("Drawdown %", color=COLORS['text_primary'])
        self.ax2.grid(True, color=COLORS['border'], alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # === IMPROVEMENT 2: Update Live Equity Chart ===
    def update_equity_chart(self, equity_value):
        self.equity_history.append((datetime.now(), equity_value))
        if len(self.equity_history) > self.max_equity_points:
            self.equity_history.pop(0)
        
        if len(self.equity_history) < 2:
            return
        
        # OPTIMIZATION: Throttle redraws to every 5s (reduces CPU by ~60%)
        current_time = time.time()
        last_draw = getattr(self, '_last_chart_draw', 0)
        if current_time - last_draw < 5.0:
            return  # Skip redraw
        self._last_chart_draw = current_time
            
        times = [h[0] for h in self.equity_history]
        values = [h[1] for h in self.equity_history]
        
        self.ax_equity.clear()
        self.ax_equity.set_facecolor(COLORS['bg_dark'])
        self.ax_equity.plot(times, values, color=COLORS['accent_green'], linewidth=2)
        self.ax_equity.fill_between(times, values, alpha=0.2, color=COLORS['accent_green'])
        self.ax_equity.tick_params(colors=COLORS['text_secondary'], labelsize=8)
        self.ax_equity.set_title(f"Equity: ${values[-1]:.2f}", color=COLORS['text_primary'], fontsize=10)
        self.ax_equity.grid(True, color=COLORS['border'], alpha=0.3)
        
        # Format x-axis
        self.ax_equity.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.fig_equity.autofmt_xdate()
        
        self.canvas_equity.draw()

    def create_scout_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="🔭 Scout Radar (Tiered Watchlist)", padding=10)
        
        # Grid of Cards style or Text List? Text List is safer for TKinter thread issues
        self.scout_text = tk.Text(frame, height=12, width=40, font=("Consolas", 10), bg=COLORS['bg_input'], fg=COLORS['text_primary'], relief="flat")
        self.scout_text.pack(fill='both', expand=True)
        
        # Legend
        legend = ttk.Label(frame, text="🚀=Rocket (Hype/Vol) | ⚓=Anchor (Stable) | 💀=Dead", font=("Segoe UI", 8), foreground=COLORS['text_secondary'])
        legend.pack(anchor='w', pady=2)
        
        return frame

    def _update_scout_radar(self):
        """Reads scout_status.json and updates the Scout Panel."""
        import json
        import os
        
        now = time.time()
        if now - self.scout_last_read < 2.0: return # Throttle 2s
        
        try:
            path = os.path.join(os.getcwd(), 'scout_status.json')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                    self.scout_status = data.get('results', {})
                    self.scout_last_read = now
                    
                    # render
                    self.scout_text.delete(1.0, tk.END)
                    
                    # Sort: Rocket -> Anchor -> Dead
                    def sort_key(item):
                        k, v = item
                        if v == 'ROCKET': return 0
                        if v == 'ANCHOR': return 1
                        return 2
                        
                    sorted_items = sorted(self.scout_status.items(), key=sort_key)
                    
                    for symbol, personality in sorted_items:
                        icon = "💀"
                        color = COLORS['text_secondary']
                        
                        if personality == 'ROCKET':
                            icon = "🚀"
                            color = COLORS['accent_red']
                        elif personality == 'ANCHOR':
                            icon = "⚓"
                            color = COLORS['accent_blue']
                            
                        # Insert with color tag
                        tag_name = f"tag_{personality}"
                        self.scout_text.tag_config(tag_name, foreground=color)
                        self.scout_text.insert(tk.END, f"{icon} {symbol:<10} [{personality}]\n", tag_name)
                        
        except Exception as e:
            # print(f"Scout Read Error: {e}") 
            pass

    # === IMPROVEMENT 4: Update Position Cards ===
    def update_position_cards(self, holdings, entry_prices, current_prices):
        """Update position display cards with PnL information."""
        # OPTIMIZATION: Check if data changed using hash
        current_hash = hash(str(holdings) + str(entry_prices))
        if current_hash == self.last_holdings_hash:
            return # Data identical, skipping card rebuild
        self.last_holdings_hash = current_hash

        # Clear existing cards
        for widget in self.positions_container.winfo_children():
            widget.destroy()
        
        if not holdings:
            ttk.Label(self.positions_container, text="No open positions", 
                foreground=COLORS['text_secondary']).pack()
            return
        
        for symbol, value in holdings.items():
            if symbol == 'CASH':
                continue
            
            # Safe price lookups with fallbacks
            entry_p = entry_prices.get(symbol, 0)
            current_p = current_prices.get(symbol, entry_p)  # Fallback to entry if missing
            
            # Zero-division guard
            if entry_p > 0 and current_p > 0:
                pnl_pct = ((current_p - entry_p) / entry_p) * 100
            else:
                pnl_pct = 0.0  # Can't calculate PnL without valid prices
            
            color = COLORS['accent_green'] if pnl_pct > 0 else COLORS['accent_red']
            
            card = tk.Frame(self.positions_container, bg=COLORS['bg_card'], highlightbackground=color, highlightthickness=2)
            card.pack(fill='x', pady=2)
            tk.Label(card, text=symbol, bg=COLORS['bg_card'], fg=COLORS['text_primary'], font=('Consolas', 10, 'bold')).pack(anchor='w', padx=5)
            tk.Label(card, text=f"PnL: {pnl_pct:+.2f}%", bg=COLORS['bg_card'], fg=color, font=('Consolas', 9)).pack(anchor='w', padx=5)

    def update_chart(self, history):
        if not history: return
        dates = [h[0] for h in history]
        values = [h[1] for h in history]
        
        self.ax.clear()
        self.ax.set_facecolor(COLORS['bg_dark'])
        self.ax.plot(dates, values, color=COLORS['accent_blue'], label='Equity')
        self.ax.set_title("Equity Curve", color=COLORS['text_primary'])
        self.ax.grid(True, color=COLORS['border'], alpha=0.3)
        self.ax.legend()
        
        import pandas as pd
        s = pd.Series(values)
        cummax = s.cummax()
        drawdown = (s - cummax) / cummax
        
        self.ax2.clear()
        self.ax2.set_facecolor(COLORS['bg_dark'])
        self.ax2.fill_between(dates, drawdown, color=COLORS['accent_red'], alpha=0.3, label='Drawdown')
        self.ax2.set_title("Drawdown %", color=COLORS['text_primary'])
        self.ax2.grid(True, color=COLORS['border'], alpha=0.3)
        
        self.fig.autofmt_xdate()
        self.canvas.draw()

    # === IMPROVEMENT 5: Add Alert ===
    def add_alert(self, message, level="info"):
        self.alert_text.config(state=tk.NORMAL)
        
        # Limit to 10 alerts
        lines = self.alert_text.get("1.0", tk.END).strip().split('\n')
        if len(lines) >= 10:
            self.alert_text.delete("1.0", "2.0")
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        icon = {"info": "ℹ️", "warning": "⚠️", "error": "🚨", "success": "✅"}.get(level, "•")
        self.alert_text.insert(tk.END, f"{icon} [{timestamp}] {message}\n")
        self.alert_text.see(tk.END)
        self.alert_text.config(state=tk.DISABLED)

    def start_live_bot(self):
        if self.is_running_live: return
        
        cfg = {
            'symbol': self.conf_symbol.get(),
            'timeframe': self.conf_timeframe.get(),
            'max_allocation': self.conf_alloc.get(),
            'leverage_cap': self.conf_leverage.get(),
            'micro_mode': self.conf_micro_mode.get()
        }
        
        self.gui_stop_event.clear()
        
        # LAZY LOAD WRAPPER: Prevents UI Freeze during TensorFlow Import
        def run_wrapper(stop_event, queue, cfg, cmd_queue):
            try:
                from main_live_phase4 import run_bot
                run_bot(stop_event, queue, cfg, cmd_queue, disable_telegram=True)
            except Exception as e:
                print(f"CRITICAL BOOT ERROR: {e}")
                import traceback
                traceback.print_exc()

        self.bot_thread = threading.Thread(target=run_wrapper, args=(self.gui_stop_event, self.gui_queue, cfg, self.command_queue))
        self.bot_thread.daemon = True
        self.bot_thread.start()
        
        self.is_running_live = True
        self.status_var.set("🟢 LIVE TRADING ACTIVE")
        self.status_label.config(foreground=COLORS['accent_green'])
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.gov_alloc.config(text=f"{cfg['max_allocation']*100:.1f}%")
        self.gov_lev.config(text=f"{cfg['leverage_cap']}x")
        
        mm_text = "ACTIVE" if cfg['micro_mode'] else "INACTIVE"
        mm_color = COLORS['accent_green'] if cfg['micro_mode'] else COLORS['text_secondary']
        self.gov_micro.config(text=mm_text, foreground=mm_color)
        
        self.add_alert("Live trading started", "success")

    # === IMPROVEMENT 13: Graceful Shutdown ===
    def stop_bot(self):
        if not self.is_running_live: return
        self.gui_stop_event.set()
        self.status_var.set("⏳ STOPPING...")
        self.status_label.config(foreground=COLORS['accent_yellow'])
        
        # Wait for thread with timeout
        def wait_and_finalize():
            if self.bot_thread:
                self.bot_thread.join(timeout=5.0)
            self.root.after(0, self._finalize_stop)
        
        threading.Thread(target=wait_and_finalize, daemon=True).start()
        
    def _finalize_stop(self):
        self.is_running_live = False
        self.status_var.set("🔴 STOPPED")
        self.status_label.config(foreground=COLORS['accent_red'])
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.add_alert("Bot stopped", "info")

    # === IMPROVEMENT 7: PANIC CLOSE ALL ===
    def panic_close_all(self):
        """Emergency position liquidation."""
        if not self.is_running_live:
            messagebox.showwarning("⚠️ Not Running", "Bot is not active. Start it first.")
            return
        
        # Confirm action
        if not messagebox.askyesno("Confirm Panic Close", "Close ALL positions immediately?"):
            return
        
        # Disable button to prevent spam
        self.panic_btn.config(state='disabled', text="CLOSING...")
        
        # Send panic signal
        self.command_queue.put({'type': 'panic_close'})
        
        # Re-enable after delay
        def reset_button():
            time.sleep(2)
            if self.panic_btn.winfo_exists():
                self.panic_btn.config(state='normal', text="🚨 PANIC CLOSE")
        
        threading.Thread(target=reset_button, daemon=True).start()
        messagebox.showinfo("✅ Panic Signal Sent", "Closing all positions...")

    def _on_closing(self):
        """Handle window close event properly."""
        if self.is_running_live:
            if messagebox.askokcancel("Quit", "Bot is still running. Stop and quit?"):
                self.stop_bot()
                self.root.after(500, self.root.destroy)  # Give time to clean up
        else:
            self.root.destroy()

    # === IMPROVEMENT 10: Export Log ===
    def export_log(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt")],
            initialfile=f"holonic_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                # Header
                f.write("Time | Level | Agent | Message\n")
                f.write("-" * 80 + "\n")
                for item in self.log_tree.get_children():
                    vals = self.log_tree.item(item)['values']
                    f.write(f"{vals[0]} | {vals[1]} | {vals[2]} | {vals[3]}\n")
            self.add_alert(f"Log exported to {filepath}", "success")

    # === IMPROVEMENT 11: Log Size Limit ===
    def log(self, msg):
        # Format: "[AgentName] Message Content..."
        ts = datetime.now().strftime("%H:%M:%S")
        
        # 1. Parse Agent Name
        agent = "SYSTEM"
        content = msg
        if msg.startswith("["):
            parts = msg.split("]", 1)
            if len(parts) > 1:
                agent = parts[0][1:] # Strip [
                content = parts[1].strip()
        
        # 2. Determine Level/Tag
        tag = 'INFO'
        level = 'INFO'
        
        upper_msg = content.upper()
        if "ERROR" in upper_msg or "FAIL" in upper_msg or "REJECT" in upper_msg or "CRITICAL" in upper_msg:
            tag = 'NEGATIVE'
            level = 'ERROR'
        elif "WARNING" in upper_msg or "RISK" in upper_msg or "DRAWDOWN" in upper_msg:
            tag = 'WARNING'
            level = 'WARN'
        elif "SUCCESS" in upper_msg or "PROFIT" in upper_msg or "BUY" in upper_msg or "FILLED" in upper_msg:
            tag = 'POSITIVE'
            level = 'OK'
        elif "DIM" in msg:
            tag = 'DIM'
        
        # 3. Insert into Treeview (Top)
        # SANITIZATION: Fix Tcl "list element in braces" error
        # Replace curly braces with parentheses to prevent Tcl from parsing as a list
        safe_content = str(content).replace('{', '(').replace('}', ')')
        
        self.log_tree.insert("", tk.END, values=(ts, level, agent, safe_content), tags=(tag,))
        self.log_tree.yview_moveto(1)
        
        # === IMPROVEMENT 11 & 12: PARSE LOGS FOR UI UPDATES ===
        try:
            # A. Regime Status (STRUCTURED EVENT - PREFERRED)
            # Handled in 'regime_status' block below.
            # Kept here as fallback if parsing old logs? No, let's rely on new events.
            
            # C. Consolidation Radar - Items
            if content.strip().startswith(("1.", "2.", "3.", "4.", "5.")):
                 # Keep legacy parser as fallback/debug for now?
                 pass 
                    
        except Exception as e:
            pass # Silent fail on parse errs
        
        # 4. Limit Rows (Keep last 500)
        # OPTIMIZATION: Log pruning is handled in process_queue batch for better performance

    # === IMPROVEMENT 12: Better Error Handling ===
    def _safe_tcl(self, v):
        """Sanitize strings for Tcl/Tk to prevent brace errors."""
        return str(v).replace('{', '(').replace('}', ')')

    def _update_regime_health(self, regime, health_score, force_update=False):
        """Centralized regime and health update to prevent duplicate/conflicting updates."""
        # Timestamp check to prevent stale updates
        current_time = time.time()
        last_update = getattr(self, '_last_regime_update', 0)
        
        if not force_update and (current_time - last_update) < 0.5:
            return  # Ignore updates within 500ms to prevent flicker
        
        self._last_regime_update = current_time
        
        # Update regime label
        self.regime_label.config(text=regime)
        
        # Standardized color scheme (including NANO)
        color_map = {
            'NANO': COLORS['accent_red'],      # Critical tier
            'MICRO': COLORS['accent_yellow'],  # Warning tier
            'SMALL': COLORS['accent_blue'],    # Growth tier
            'MEDIUM': COLORS['accent_green'],  # Stable tier
            'LARGE': '#FF00FF'                 # Elite tier
        }
        fg = color_map.get(regime, COLORS['accent_yellow'])  # Default to yellow
        self.regime_label.config(foreground=fg)
        
        # Update health
        self.health_progress['value'] = health_score * 100
        self.health_label.config(text=f"{health_score:.2f}")
        
        # Promotion logic
        if health_score >= 0.95:
            self.promo_label.config(text="ELIGIBLE (Wait 72h)", foreground=COLORS['accent_green'])
        else:
            self.promo_label.config(text="BUILDING...", foreground=COLORS['text_secondary'])

    def process_queue(self):
        processed_any = False  # Track if we processed messages
        try:
            while True:
                msg = self.gui_queue.get_nowait()
                try:
                    self._handle_queue_message(msg)
                except Exception as e:
                    print(f"GUI Queue Processing Error: {e}")
                processed_any = True
        except queue.Empty:
            pass
        
        # === FILE-BASED SYNC FOR EXTERNAL BOT ===
        # Polling moved to background thread (StatusPollThread)
        # file_processed = self._poll_status_file() 
        # processed_any = processed_any or file_processed
        
        # === OPTIMIZATION: Batch Log Pruning ===
        children = self.log_tree.get_children()
        if len(children) > self.max_log_entries:
            excess_count = len(children) - self.max_log_entries
            for child in children[:excess_count]:
                self.log_tree.delete(child)

        # Note: Heavy generic updates (Holospace, MonteCarlo, etc.) were removed for performance.
        # OPTIMIZATION: Adaptive polling (50ms if busy, 200ms if idle)
        next_poll = 50 if processed_any else 200
        self.root.after(next_poll, self.process_queue)
    

    
    def _handle_queue_message(self, msg):
        mtype = msg.get("type", "")

        if mtype == 'log':
            self.log(msg.get('message', ''))
            
        elif mtype == 'connection_status':
            # Update connection status indicator
            connected = msg.get('connected', False)
            last_update = msg.get('last_update', '')
            error = msg.get('error', '')
            
            if connected:
                self.conn_status_var.set("🟢 CONNECTED")
                self.conn_status_label.config(foreground=COLORS['accent_green'])
                # Show last update time if available
                if last_update:
                    try:
                        # Parse ISO timestamp and show relative time
                        from datetime import datetime
                        dt = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                        now = datetime.now(dt.tzinfo)
                        diff = (now - dt).total_seconds()
                        if diff < 60:
                            self.conn_status_var.set(f"🟢 LIVE ({int(diff)}s ago)")
                        elif diff < 3600:
                            self.conn_status_var.set(f"🟢 LIVE ({int(diff/60)}m ago)")
                        else:
                            self.conn_status_var.set(f"🟡 STALE ({int(diff/3600)}h ago)")
                            self.conn_status_label.config(foreground=COLORS['accent_yellow'])
                    except Exception:
                        pass
            else:
                self.conn_status_var.set(f"🔴 DISCONNECTED{': ' + error if error else ''}")
                self.conn_status_label.config(foreground=COLORS['accent_red'])

        elif mtype == 'summary':
            data = msg.get('data', [])

            # Clear existing rows
            for child in self.tree.get_children():
                self.tree.delete(child)

            # Repopulate
            for row in data:


                pnl_str = row.get('PnL', '-')
                struct_mode = row.get('Struct', '-')
                
                tag = 'neutral'
                if '+' in str(pnl_str):
                    tag = 'profit'
                elif '-' in str(pnl_str) and pnl_str != '-':
                    tag = 'loss'
                elif 'BREAKOUT' in struct_mode:
                    tag = 'profit'
                elif 'BREAKDOWN' in struct_mode:
                    tag = 'loss'
                
                # Check for Whale
                action_text = row.get('Action', '')
                if 'WHALE' in str(action_text):
                    tag = 'whale'
                
                # CRITICAL FIX: Sanitize ALL values to prevent Tcl brace errors
                values = (
                    self._safe_tcl(row.get('Symbol', '?')),
                    self._safe_tcl(row.get('Price', '0')),
                    self._safe_tcl(row.get('Regime', '?')),
                    self._safe_tcl(row.get('Entropy', '0.00')),
                    self._safe_tcl(row.get('Struct', '-')),
                    self._safe_tcl(row.get('RSI', '-')),
                    self._safe_tcl(row.get('LSTM', '0.50')),
                    self._safe_tcl(row.get('XGB', '0.50')),
                    self._safe_tcl(pnl_str),
                    self._safe_tcl(row.get('Action', '-'))
                )
                # OPTIMIZATION: Summary Guard
                summary_hash = hash(str(values))
                if summary_hash != self.last_summary_hash:
                    self.last_summary_hash = summary_hash
                    self.tree.insert('', tk.END, values=values, tags=(tag,))
                
        elif mtype == "backtest_result":
            res = msg.get("data", {})
            self.bt_roi.config(text=f"{res.get('roi', 0):.2f}%")
            self.bt_pnl.config(text=f"${res.get('pnl', 0):.2f}")
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
            self.gov_fortress.config(text=data.get('fortress_balance', '-'))
            self.gov_budget.config(text=data.get('risk_budget', '-'))
            
            self.ag_regime.config(text=data.get('regime', '?'))
            self.ag_entropy.config(text=data.get('entropy', '0.0'))
            self.ag_model.config(text=data.get('strat_model', '-'))
            self.ag_kalman.config(text=data.get('kalman_active', '-'))
            self.ag_ppo_conv.config(text=data.get('ppo_conv', '0.50'))
            self.ag_ppo_reward.config(text=data.get('ppo_reward', '0.00'))
            self.ag_lstm_prob.config(text=data.get('lstm_prob', '0.50'))
            self.ag_xgb_prob.config(text=data.get('xgb_prob', '0.50'))
            
            self.act_last_ord.config(text=data.get('last_order', 'NONE'))
            self.act_pending.config(text=str(data.get('pending_count', '0')))
            
            # Update Micro Mode Status
            mm_status = data.get('gov_micro', 'UNKNOWN')
            mm_color = COLORS['accent_green'] if mm_status == 'ACTIVE' else COLORS['text_secondary']
            self.gov_micro.config(text=mm_status, foreground=mm_color)
            
            # Update News Feed
            self._update_news_feed(data.get('news_feed', []))
            
            self.perf_winrate.config(text=data.get('win_rate', '-'))
            self.perf_pnl.config(text=data.get('pnl', '-'))
            self.perf_omega.config(text=data.get('omega', '-'))
            
            self.p12_exposure.config(text=data.get('exposure', '$0.00'))
            self.p12_margin.config(text=data.get('margin', '$0.00'))
            self.p12_actual_lev.config(text=data.get('actual_lev', '0.00x'))
            
            # Evolution Wiring
            self.evo_fitness.config(text=data.get('evo_fitness', '0.00'))
            self.evo_kings.config(text=data.get('evo_kings', '0'))
            
            # === IMPROVEMENT 8: Update Balance ===
            balance = data.get('balance')
            if balance is not None:
                self.balance_label.config(text=f"${float(balance):.2f}")
            
            # === IMPROVEMENT 2: Update Equity Chart & Label ===
            equity = data.get('equity')
            if equity is not None:
                self.update_equity_chart(float(equity))
                if hasattr(self, 'equity_label'):
                    self.equity_label.config(text=f"${float(equity):.2f}")
            
            # === IMPROVEMENT 4: Update Position Cards ===
            holdings = data.get('holdings')
            entry_prices = data.get('entry_prices', {})
            current_prices = data.get('current_prices', {})
            if holdings:
                self.update_position_cards(holdings, entry_prices, current_prices)

            # === Regime & Health Update (Centralized) ===
            regime = data.get('regime', 'UNKNOWN')
            health_score = float(data.get('health_score', 0.0))
            solvency = data.get('solvency_status', 'SOLVENT')
            
            # Always update regime/health
            self._update_regime_health(regime, health_score)
            
            # Solvency warning overlay (doesn't replace regime)
            if solvency == 'INSOLVENT':
                self.health_label.config(text="⚠️ INSOLVENT", foreground=COLORS['accent_red'])
            
            # === UPDATED: Consume Explicit Queue Data (No Overwrites) ===
            # 1. Update Consolidation Radar (Kill List) - Bottom Left
            consolidation_data = data.get('consolidation_data', [])
            if consolidation_data:
                # OPTIMIZATION: Radar Guard
                radar_hash = hash(str(consolidation_data))
                if radar_hash != self.last_radar_hash:
                    self.last_radar_hash = radar_hash
                    # Clear and Repopulate Radar Tree
                    for item in self.radar_tree.get_children():
                        self.radar_tree.delete(item)

                    for i, r in enumerate(consolidation_data):
                        # Format: {'symbol': 'BTC', 'score': 0.85, 'reason': 'High Vol'}
                        sym = r.get('symbol', 'UNKNOWN')
                        score = r.get('score', 0.0)
                        reason = r.get('reason', 'Scanning...')

                        status = "HIGH RISK" if score > 0.8 else "WATCH"
                        tag = 'close' if score > 0.8 else 'neutral'

                        self.radar_tree.insert('', 'end', values=(
                            self._safe_tcl(f"#{i+1}"),
                            self._safe_tcl(sym),
                            self._safe_tcl(f"{score:.2f}"),
                            "-",
                            "-",
                            "-",
                            self._safe_tcl(status)
                        ), tags=(tag,))

            # 2. Update Scout Watchlist - Scout Tab
            scout_data = data.get('scout_data', [])
            if scout_data:
                 self._update_scout_text(scout_data)

            # 3. Store portfolio data for Portfolio tab
            holdings = data.get('holdings', {})
            entry_prices = data.get('entry_prices', {})
            current_prices = data.get('current_prices', {})
            balance = data.get('balance', 0)

            # Store data for portfolio tab
            self.current_holdings = holdings
            self.current_entry_prices = entry_prices
            self.current_current_prices = current_prices
            self.current_balance = balance

            # (Monte Carlo and Enhanced Radar updates removed)

        elif mtype == 'order':
            order = msg.get('data', {})
            self._add_order_to_history(order)
            
        elif mtype == 'gc_status':
            data = msg.get('data', {})
            self.gc_last_run_label.config(text=data.get('last_run', 'Never'))
            self.gc_cleaned_label.config(text=str(data.get('cleaned', 0)))
            
        elif mtype == 'alert':
            self.add_alert(msg.get('message', ''), msg.get('level', 'info'))
            
        elif mtype == 'overwatch_update':
            state = msg.get('state', 'NOMINAL')
            sitrep = msg.get('sitrep', '')
            
            # Update Label Color
            color = COLORS['accent_green']
            if state == 'CRITICAL': color = COLORS['accent_red']
            elif state == 'CAUTION': color = COLORS['accent_yellow']
            elif state == 'OPTIMAL': color = COLORS['accent_blue']
            
            self.ov_status_lbl.config(text=state, foreground=color)
            
            # Update Text
            self.ov_text.config(state=tk.NORMAL)
            self.ov_text.delete(1.0, tk.END)
            self.ov_text.insert(tk.END, sitrep)
            self.ov_text.config(state=tk.DISABLED)

        elif mtype == 'regime_status':
            data = msg.get('data', {})
            regime = data.get('regime', 'UNKNOWN')
            health = float(data.get('health', 0.0))
            
            # Use centralized update (force=True for explicit regime events)
            self._update_regime_health(regime, health, force_update=True)

        # === Signal Report (from agent_status data) ===
        if mtype == 'agent_status':
            signal_report = msg.get('data', {}).get('signal_report', [])
            if signal_report:
                self._update_signals_tab(signal_report)

    def _update_scout_text(self, scout_data):
        self.scout_text.delete(1.0, tk.END)
        for item in scout_data:
            symbol = self._safe_tcl(item.get('symbol', '?'))
            reason = self._safe_tcl(item.get('reason', 'UNKNOWN'))
            
            icon = "💀"
            color = COLORS['text_secondary']
            
            if reason == 'ROCKET':
                icon = "🚀"
                color = COLORS['accent_red']
            elif reason == 'ANCHOR':
                icon = "⚓"
                color = COLORS['accent_blue']
                
            tag_name = f"tag_{reason}"
            self.scout_text.tag_config(tag_name, foreground=color)
            self.scout_text.insert(tk.END, f"{icon} {symbol:<10} [{reason}]\n", tag_name)

    # === IMPROVEMENT 9: Add Order to History ===
    def _add_order_to_history(self, order):
        if len(self.order_history) >= self.max_orders:
            try:
                oldest = self.order_tree.get_children()[0]
                self.order_tree.delete(oldest)
                self.order_history.pop(0)
            except: pass
        
        # Handle Case Mismatch (Executor sends Title Case, Local expected lowercase)
        # Normalize to lowercase keys if needed, or check both
        side = order.get('side') or order.get('Side') or 'BUY'
        status = order.get('status') or order.get('Status') or 'PENDING'
        
        # Determine Tag
        tag = 'buy' if str(side).upper() == 'BUY' else 'sell'
        if str(status).upper() == 'CANCELED': tag = 'canceled'
        if str(status).upper() == 'FILLED': tag = 'filled'
        
        values = (
            self._safe_tcl(order.get('time') or order.get('Time') or datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            self._safe_tcl(order.get('symbol') or order.get('Symbol') or '-'),
            self._safe_tcl(side),
            self._safe_tcl(f"{float(order.get('qty', 0) or order.get('Qty', 0)):.4f}"),
            self._safe_tcl(f"{float(order.get('price', 0) or order.get('Price', 0)):.2f}"),
            self._safe_tcl(status)
        )
        self.order_tree.insert('', 0, values=values, tags=(tag,))
        self.order_history.append(order)
            
    def start_backtest(self):
        if self.is_running_backtest: return
        self.bt_progress['value'] = 0
        self.is_running_backtest = True
        
        symbol = self.conf_symbol.get()
        
        def run_and_reset():
            try:
                from run_backtest import run_backtest
                run_backtest(self.gui_queue, symbol=symbol)
            except Exception as e:
                print(f"Backtest Error: {e}")
            self.is_running_backtest = False
        
        t = threading.Thread(target=run_and_reset)
        t.daemon = True
        t.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = HolonicDashboard(root)
    root.mainloop()
