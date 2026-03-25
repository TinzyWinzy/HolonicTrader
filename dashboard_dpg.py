import dearpygui.dearpygui as dpg
import threading
import queue
import time
from datetime import datetime
import sys
import os

# Adhering to the project structure
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.append(os.path.dirname(__file__)) # Add self to path for sibling imports

# from main_live_phase4 import run_bot # (Import inside method to avoid circular deps)

# ==============================================================================
# CONFIG & COLORS
# ==============================================================================
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
}

def hex_to_rgba(hex_code, alpha=255):
    """Converts hex string to list of ints [r, g, b, a] for DPG."""
    hex_code = hex_code.lstrip('#')
    return [int(hex_code[i:i+2], 16) for i in (0, 2, 4)] + [alpha]

class HolonicDashboardDPG:
    def __init__(self):
        self.title = "A E H M L   T R A D E R   //   P H A S E   I V   (G P U   E D I T I O N)"
        self.width = 1600
        self.height = 1000
        
        # State
        self.is_running = False
        self.gui_queue = queue.Queue()
        self.command_queue = queue.Queue()
        self.gui_stop_event = threading.Event()
        self.bot_thread = None
        
        # Data Buffers (for plotting)
        self.equity_data_x = []
        self.equity_data_y = []
        self.max_chart_points = 1000
        # (Holospace removed for performance)
        
        # IDs for dynamic updates
        self.id_regime_text = 0
        self.id_market_table = 0
        self.id_log_table = 0
        self.id_equity_series = 0
        self.id_status_text = 0
        self.id_health_bar = 0
        self.id_x_axis = 0
        self.id_y_axis = 0
        
        # AGENTS TAB IDs
        self.ids_gov = {} # state, alloc, lev, trends, micro, castle, budget
        self.ids_act = {} # last, pending
        self.ids_brain = {} # regime, entropy, model, kalman, ppo_c, ppo_r, lstm, xgb
        self.ids_perf = {} # win, pnl, omega
        self.ids_risk = {} # exposure, margin, lev
        
        self.log_buffer = []

    def setup_theme(self):
        """Applies the Bloomberg Terminal aesthetic."""
        with dpg.theme() as global_theme:
            
            # 1. GLOBAL STYLES (Window, Text, Buttons)
            with dpg.theme_component(dpg.mvAll):
                # Window & Background
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, hex_to_rgba(COLORS['bg_dark']))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, hex_to_rgba(COLORS['bg_card']))
                dpg.add_theme_color(dpg.mvThemeCol_Border, hex_to_rgba(COLORS['border']))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, hex_to_rgba(COLORS['bg_card']))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg, hex_to_rgba(COLORS['bg_card']))
                
                # Text
                dpg.add_theme_color(dpg.mvThemeCol_Text, hex_to_rgba(COLORS['text_secondary']))
                
                # Buttons
                dpg.add_theme_color(dpg.mvThemeCol_Button, hex_to_rgba(COLORS['bg_input']))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, hex_to_rgba(COLORS['accent_primary'], 150))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, hex_to_rgba(COLORS['accent_primary']))
                
                # Tables
                dpg.add_theme_color(dpg.mvThemeCol_TableHeaderBg, hex_to_rgba(COLORS['bg_input']))
                
                # Styles
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 8)

            # 2. PLOT STYLES (Specific to dpg.mvPlot)
            with dpg.theme_component(dpg.mvPlot):
                dpg.add_theme_color(dpg.mvPlotCol_PlotBg, hex_to_rgba(COLORS['bg_dark']))
                dpg.add_theme_color(dpg.mvPlotCol_Line, hex_to_rgba(COLORS['accent_secondary']))
                dpg.add_theme_color(dpg.mvPlotCol_FrameBg, hex_to_rgba(COLORS['bg_dark']))
                
        dpg.bind_theme(global_theme)

    def build_layout(self):
        with dpg.window(tag="Primary Window"):
            # 1. Top Control Bar
            with dpg.group(horizontal=True):
                dpg.add_button(label="▶ START LIVE BOT", callback=self.start_live_bot, width=150, height=30)
                dpg.add_button(label="⏹ STOP BOT", callback=self.stop_bot, width=120, height=30)
                dpg.add_spacer(width=20)
                self.id_status_text = dpg.add_text("🔴 STOPPED", color=hex_to_rgba(COLORS['accent_red']))
                
                dpg.add_spacer(width=50)
                # Panic Button (Right Aligned - approximate using hefty spacer for now or layout tricks)
                # DPG doesn't have "Pack right", keeping it simple flow
                dpg.add_button(label="🚨 PANIC CLOSE ALL", callback=self.panic_close, width=150, height=30)

            dpg.add_separator()
            
            # 2. Tab Bar
            with dpg.tab_bar():
                
                # === TAB 1: LIVE OPERATIONS ===
                with dpg.tab(label="  📊 Live Operations  "):
                    
                    # Row 1: Regime & Health (Custom Header)
                    with dpg.group(horizontal=True):
                        dpg.add_text("CURRENT REGIME:")
                        self.id_regime_text = dpg.add_text("MICRO", color=hex_to_rgba(COLORS['accent_yellow']))
                        dpg.add_spacer(width=20)
                        dpg.add_text("SYSTEM HEALTH:")
                        self.id_health_bar = dpg.add_progress_bar(default_value=0.0, width=200, overlay="0.00", tag="health_bar")
                    
                    dpg.add_spacer(height=5)
                    
                    # Row 2: Split View (Market Table | Logs + Charts)
                    with dpg.group(horizontal=True):
                        
                        # Left Column: Market Table & Radar (40% width)
                        with dpg.child_window(width=600, border=False):
                            dpg.add_text("Market Overview", color=hex_to_rgba(COLORS['accent_primary']))
                            
                            # Market Table
                            with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, 
                                           borders_outerH=True, borders_innerV=True, row_background=True, tag="market_table_tag"):
                                dpg.add_table_column(label="Symbol", width_stretch=True, init_width_or_weight=1.0)
                                dpg.add_table_column(label="Price", init_width_or_weight=1.0)
                                dpg.add_table_column(label="Regime", init_width_or_weight=0.8)
                                dpg.add_table_column(label="Struct", init_width_or_weight=0.8)
                                dpg.add_table_column(label="PnL", init_width_or_weight=1.0)
                                dpg.add_table_column(label="Action", init_width_or_weight=1.2)
                                
                                # We will populate rows dynamically
                            
                            dpg.add_spacer(height=10)
                            dpg.add_text("💀 Consolidation Radar (Kill List)", color=hex_to_rgba(COLORS['accent_red']))
                            
                            with dpg.table(header_row=True, resizable=True, tag="radar_table_tag", height=200):
                                dpg.add_table_column(label="Rank")
                                dpg.add_table_column(label="Symbol")
                                dpg.add_table_column(label="Score")
                                dpg.add_table_column(label="Status")

                        # Right Column: Equity Chart & Logs (Remaining width)
                        with dpg.child_window(border=False):
                            
                            # TOP: Equity Chart
                            dpg.add_text("Live Equity Curve", color=hex_to_rgba(COLORS['accent_green']))
                            with dpg.plot(height=300, width=-1, no_menus=True):
                                dpg.add_plot_legend()
                                # X Axis
                                self.id_x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Time")
                                
                                # Y Axis & Series
                                self.id_y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Equity ($)")
                                self.id_equity_series = dpg.add_line_series([], [], label="Total Equity", parent=self.id_y_axis)

                            dpg.add_spacer(height=10)
                            
                            # BOTTOM: Logs
                            dpg.add_text("Activity Log", color=hex_to_rgba(COLORS['text_primary']))
                            with dpg.table(header_row=True, scrollY=True, height=-1, 
                                           policy=dpg.mvTable_SizingFixedFit, row_background=True, tag="log_table_tag"):
                                dpg.add_table_column(label="Time", width_fixed=True, init_width_or_weight=80)
                                dpg.add_table_column(label="Level", width_fixed=True, init_width_or_weight=70)
                                dpg.add_table_column(label="Agent", width_fixed=True, init_width_or_weight=100)
                                dpg.add_table_column(label="Message", width_stretch=True)

                # === TAB 2: AGENTS & STATUS ===
                with dpg.tab(label="  🤖 Agents & Status  "):
                    dpg.add_spacer(height=5)
                    
                    # Row 1: Governor | Actuator
                    with dpg.group(horizontal=True):
                        # Governor (Risk)
                        with dpg.child_window(width=500, height=200, border=True):
                            dpg.add_text("🛡️ Governor Holon (Risk)", color=hex_to_rgba(COLORS['accent_primary']))
                            dpg.add_separator()
                            with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp):
                                dpg.add_table_column(); dpg.add_table_column()
                                with dpg.table_row(): 
                                    dpg.add_text("State:"); self.ids_gov['state'] = dpg.add_text("-", color=hex_to_rgba(COLORS['text_primary']))
                                with dpg.table_row():
                                    dpg.add_text("Max Alloc:"); self.ids_gov['alloc'] = dpg.add_text("-")
                                with dpg.table_row():
                                    dpg.add_text("Leverage Cap:"); self.ids_gov['lev'] = dpg.add_text("-")
                                with dpg.table_row():
                                    dpg.add_text("Active Trends:"); self.ids_gov['trends'] = dpg.add_text("-")
                                with dpg.table_row():
                                    dpg.add_text("Micro Mode:"); self.ids_gov['micro'] = dpg.add_text("-")
                                with dpg.table_row():
                                    dpg.add_text("🏰 Iron Fortress:"); self.ids_gov['castle'] = dpg.add_text("-")
                                with dpg.table_row():
                                    dpg.add_text("⚔️ Risk Budget:"); self.ids_gov['budget'] = dpg.add_text("-")

                        # Actuator (Execution)
                        with dpg.child_window(width=400, height=200, border=True):
                            dpg.add_text("⚙️ Actuator Holon (Exec)", color=hex_to_rgba(COLORS['accent_secondary']))
                            dpg.add_separator()
                            with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp):
                                dpg.add_table_column(); dpg.add_table_column()
                                with dpg.table_row(): 
                                    dpg.add_text("Last Order:"); self.ids_act['last'] = dpg.add_text("-", wrap=300)
                                with dpg.table_row(): 
                                    dpg.add_text("Pending Orders:"); self.ids_act['pending'] = dpg.add_text("-")

                    dpg.add_spacer(height=10)

                    # Row 2: Brains | Performance | Risk
                    with dpg.group(horizontal=True):
                        
                        # Brains
                        with dpg.child_window(width=500, height=250, border=True):
                            dpg.add_text("🧠 Brain Holons (Strategy)", color=hex_to_rgba(COLORS['accent_secondary']))
                            dpg.add_separator()
                            with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp):
                                dpg.add_table_column(); dpg.add_table_column()
                                with dpg.table_row(): dpg.add_text("Regime:"); self.ids_brain['regime'] = dpg.add_text("-")
                                with dpg.table_row(): dpg.add_text("Entropy:"); self.ids_brain['entropy'] = dpg.add_text("-")
                                with dpg.table_row(): dpg.add_text("Strategy:"); self.ids_brain['model'] = dpg.add_text("-")
                                with dpg.table_row(): dpg.add_text("Kalman:"); self.ids_brain['kalman'] = dpg.add_text("-")
                                with dpg.table_row(): dpg.add_text("PPO Conviction:"); self.ids_brain['ppo_c'] = dpg.add_text("-")
                                with dpg.table_row(): dpg.add_text("PPO Reward:"); self.ids_brain['ppo_r'] = dpg.add_text("-")
                                with dpg.table_row(): dpg.add_text("LSTM Prob:"); self.ids_brain['lstm'] = dpg.add_text("-")
                                with dpg.table_row(): dpg.add_text("XGB Prob:"); self.ids_brain['xgb'] = dpg.add_text("-")

                        # Performance
                        with dpg.child_window(width=300, height=250, border=True):
                            dpg.add_text("📈 Performance", color=hex_to_rgba(COLORS['accent_green']))
                            dpg.add_separator()
                            with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp):
                                dpg.add_table_column(); dpg.add_table_column()
                                with dpg.table_row(): dpg.add_text("Win Rate:"); self.ids_perf['win'] = dpg.add_text("-")
                                with dpg.table_row(): dpg.add_text("Realized PnL:"); self.ids_perf['pnl'] = dpg.add_text("-")
                                with dpg.table_row(): dpg.add_text("Omega Ratio:"); self.ids_perf['omega'] = dpg.add_text("-")

                        # Risk Stats
                        with dpg.child_window(width=300, height=250, border=True):
                            dpg.add_text("🛡️ Risk Metrics", color=hex_to_rgba(COLORS['accent_red']))
                            dpg.add_separator()
                            with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchProp):
                                dpg.add_table_column(); dpg.add_table_column()
                                with dpg.table_row(): dpg.add_text("Total Exposure:"); self.ids_risk['exposure'] = dpg.add_text("-")
                                with dpg.table_row(): dpg.add_text("Used Margin:"); self.ids_risk['margin'] = dpg.add_text("-")
                                with dpg.table_row(): dpg.add_text("Real Leverage:"); self.ids_risk['lev'] = dpg.add_text("-")

                # === TAB 3: SCOUT RADAR ===
                with dpg.tab(label="  🔭 Scout Radar  "):
                     dpg.add_text("Physics-Based Market Scanner (Entropy & Regime)", color=hex_to_rgba(COLORS['accent_secondary']))
                     
                     with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, 
                                    borders_outerH=True, borders_innerV=True, row_background=True, tag="scout_table_tag", height=-1):
                                dpg.add_table_column(label="Symbol")
                                dpg.add_table_column(label="Regime/State")
                                dpg.add_table_column(label="Entropy Score")
                                dpg.add_table_column(label="Status")

    def process_queue(self):
        """Reads from queues and updates DPG items."""
        try:
            while True:
                msg = self.gui_queue.get_nowait()
                self.handle_message(msg)
        except queue.Empty:
            pass
        
        # Project
        proj = self.holospace.project(self.market_points_3d, w, h)
        
        # Draw Lines (Axes) - simple
        origin = self.holospace.project([(0,0,0, [255,255,255,255])], w, h)[0]
        x_axis = self.holospace.project([(2,0,0, [255,0,0,255])], w, h)[0]
        y_axis = self.holospace.project([(0,2,0, [0,255,0,255])], w, h)[0]
        z_axis = self.holospace.project([(0,0,2, [0,0,255,255])], w, h)[0]
        
        dpg.draw_line((origin['x'], origin['y']), (x_axis['x'], x_axis['y']), color=[255,50,50,255], thickness=2, parent="holospace_canvas")
        dpg.draw_line((origin['x'], origin['y']), (y_axis['x'], y_axis['y']), color=[50,255,50,255], thickness=2, parent="holospace_canvas")
        dpg.draw_line((origin['x'], origin['y']), (z_axis['x'], z_axis['y']), color=[50,50,255,255], thickness=2, parent="holospace_canvas")

        # Draw Points
        for p in proj:
            dpg.draw_circle((p['x'], p['y']), p['size'], color=p['color'], fill=p['color'], parent="holospace_canvas")

    def handle_message(self, msg):
        """Dispatches queue messages to UI updates."""
        msg_type = msg.get('type')
        data = msg.get('data')

        if msg_type == 'log':
            self.log(msg.get('message', ''))

        elif msg_type == 'agent_status':
            # Update Header Info
            dpg.set_value(self.id_regime_text, f"REGIME: {data.get('regime', 'UNKNOWN')}")
            
            # Health Bar
            health = float(data.get('health_score', 0))
            if dpg.does_item_exist("health_bar"):
                dpg.set_value("health_bar", health)
            
            # Status Text
            dpg.set_value(self.id_status_text, f"Equity: ${data.get('equity', 0):.2f} | Margin: {data.get('margin', '$0')}")

            # Update Equity Chart
            eq_val = data.get('equity', 0)
            if eq_val and eq_val > 0:
                self.equity_data_y.append(float(eq_val))
                self.equity_data_x.append(len(self.equity_data_y))
                
                if len(self.equity_data_y) > self.max_chart_points:
                    self.equity_data_y.pop(0)
                    self.equity_data_x.pop(0)
                
                dpg.set_value(self.id_equity_series, [list(self.equity_data_x), list(self.equity_data_y)])
                dpg.fit_axis_data(self.id_y_axis)
                dpg.fit_axis_data(self.id_x_axis)
            
            # Consolidation Radar
            c_data = data.get('consolidation_data', [])
            if c_data and dpg.does_item_exist("radar_table_tag"):
                dpg.delete_item("radar_table_tag", children_only=True)
                for i, r in enumerate(c_data):
                    with dpg.table_row(parent="radar_table_tag"):
                        dpg.add_text(f"#{i+1}")
                        dpg.add_text(r.get('symbol', '?'))
                        dpg.add_text(f"{r.get('score', 0):.2f}")
                        stat = "WATCH" if r.get('score', 0) < 0.8 else "KILL"
                        col = COLORS['accent_yellow'] if stat == "WATCH" else COLORS['accent_red']
                        dpg.add_text(stat, color=hex_to_rgba(col))
            
            # Update Agents Tab
            if self.ids_gov: # Check if initialized
                dpg.set_value(self.ids_gov['state'], data.get('gov_state', '-'))
                dpg.set_value(self.ids_gov['alloc'], data.get('gov_alloc', '-'))
                dpg.set_value(self.ids_gov['lev'], data.get('gov_lev', '-'))
                dpg.set_value(self.ids_gov['trends'], data.get('gov_trends', '-'))
                dpg.set_value(self.ids_gov['micro'], data.get('gov_micro', '-'))
                dpg.set_value(self.ids_gov['castle'], data.get('fortress_balance', '-'))
                dpg.set_value(self.ids_gov['budget'], data.get('risk_budget', '-'))
            
            if self.ids_act:
                # Format Order Dict
                last_o = data.get('last_order', 'NONE')
                if isinstance(last_o, dict):
                    # "BTC/USDT BUY 1.5 @ $50000"
                    last_o_str = f"{last_o.get('symbol','?')} {last_o.get('side','?')} {last_o.get('qty',0)} @ ${last_o.get('price',0)}"
                else:
                    last_o_str = str(last_o)
                dpg.set_value(self.ids_act['last'], last_o_str)
                dpg.set_value(self.ids_act['pending'], str(data.get('pending_count', '0')))

            if self.ids_brain:
                dpg.set_value(self.ids_brain['regime'], data.get('regime', '-'))
                dpg.set_value(self.ids_brain['entropy'], data.get('entropy', '-'))
                dpg.set_value(self.ids_brain['model'], data.get('strat_model', '-'))
                dpg.set_value(self.ids_brain['kalman'], data.get('kalman_active', '-'))
                dpg.set_value(self.ids_brain['ppo_c'], data.get('ppo_conv', '-'))
                dpg.set_value(self.ids_brain['ppo_r'], data.get('ppo_reward', '-'))
                dpg.set_value(self.ids_brain['lstm'], data.get('lstm_prob', '-'))
                dpg.set_value(self.ids_brain['xgb'], data.get('xgb_prob', '-'))
            
            if self.ids_perf:
                dpg.set_value(self.ids_perf['win'], data.get('win_rate', '-'))
                dpg.set_value(self.ids_perf['pnl'], data.get('pnl', '-'))
                dpg.set_value(self.ids_perf['omega'], data.get('omega', '-'))

            if self.ids_risk:
                dpg.set_value(self.ids_risk['exposure'], data.get('exposure', '-'))
                dpg.set_value(self.ids_risk['margin'], data.get('margin', '-'))
                dpg.set_value(self.ids_risk['lev'], data.get('actual_lev', '-'))
            
            # Update Scout Radar
            s_data = data.get('scout_data', [])
            if s_data and dpg.does_item_exist("scout_table_tag"):
                dpg.delete_item("scout_table_tag", children_only=True)
                for item in s_data:
                    # Item: {'symbol': s, 'score': val, 'reason': {'regime':..., 'entropy':...}}
                    sym = item.get('symbol', '?')
                    reason = item.get('reason', {})
                    reg = reason.get('regime', 'UNKNOWN') if isinstance(reason, dict) else str(reason)
                    ent = f"{reason.get('entropy', 0):.4f}" if isinstance(reason, dict) else "?"
                    
                    with dpg.table_row(parent="scout_table_tag"):
                        dpg.add_text(sym)
                        
                        rc = COLORS['accent_red'] if 'CHAOS' in reg else COLORS['accent_green']
                        dpg.add_text(reg, color=hex_to_rgba(rc))
                        
                        dpg.add_text(ent)
                        
                        stat = "ACTIVE" if 'CHAOS' not in reg else "IGNORED"
                        dpg.add_text(stat)


        elif msg_type == 'summary':
            # Market Table Update
            if dpg.does_item_exist("market_table_tag"):
                dpg.delete_item("market_table_tag", children_only=True)
                
                new_3d_points = []
                
                for row in data:
                    pnl = str(row.get('PnL', '-'))
                    text_col = COLORS['text_secondary']
                    if '+' in pnl: text_col = COLORS['accent_green']
                    elif '-' in pnl and pnl != '-': text_col = COLORS['accent_red']
                    
                    with dpg.table_row(parent="market_table_tag"):
                        dpg.add_text(row.get('Symbol', '?'))
                        dpg.add_text(str(row.get('Price', '0')))
                        
                        reg = row.get('Regime', '-')
                        reg_col = COLORS['accent_red'] if 'CHAOS' in reg else COLORS['accent_green']
                        dpg.add_text(reg, color=hex_to_rgba(reg_col))
                        
                        dpg.add_text(row.get('Struct', '-'))
                        dpg.add_text(pnl, color=hex_to_rgba(text_col))
                        dpg.add_text(row.get('Action', '-'))

    def log(self, text):
        ts = datetime.now().strftime("%H:%M:%S")
        
        # Parse minimal info
        level = "INFO"
        if "ERROR" in text or "FAIL" in text: level = "ERR"
        elif "SUCCESS" in text: level = "OK"
        
        # Color
        col = hex_to_rgba(COLORS['text_secondary'])
        if level == "ERR": col = hex_to_rgba(COLORS['accent_red'])
        elif level == "OK": col = hex_to_rgba(COLORS['accent_green'])

        # Add row to log table
        with dpg.table_row(parent="log_table_tag"):
            dpg.add_text(ts)
            dpg.add_text(level, color=col)
            dpg.add_text("SYS") # Agent parsing later
            dpg.add_text(text, color=col)
            pass

    # ========================== COMMANDS ==========================
    def start_live_bot(self):
        if self.is_running: return
        
        self.is_running = True
        dpg.set_value(self.id_status_text, "🟢 LIVE TRADING")
        dpg.configure_item(self.id_status_text, color=hex_to_rgba(COLORS['accent_green']))
        
        # Start Thread
        self.gui_stop_event.clear()
        
        # Allow main_live_phase4 to use config.py defaults by passing None
        cfg = None
        
        def run_wrapper():
            try:
                from main_live_phase4 import run_bot
                run_bot(self.gui_stop_event, self.gui_queue, cfg, self.command_queue)
            except Exception as e:
                print(f"Bot Error: {e}")
                self.gui_queue.put({'type': 'log', 'message': f"CRITICAL FAULT: {e}"})

        self.bot_thread = threading.Thread(target=run_wrapper, daemon=True)
        self.bot_thread.start()

    def stop_bot(self):
        if not self.is_running: return
        self.gui_stop_event.set()
        self.is_running = False
        dpg.set_value(self.id_status_text, "🔴 STOPPED")
        dpg.configure_item(self.id_status_text, color=hex_to_rgba(COLORS['accent_red']))

    def panic_close(self):
        self.command_queue.put({'type': 'panic_close'})
        self.log("PANIC SIGNAL SENT!")

    def run(self):
        dpg.create_context()
        self.setup_theme()
        self.build_layout()
        
        dpg.create_viewport(title=self.title, width=self.width, height=self.height)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Primary Window", True)

        # Main Loop
        while dpg.is_dearpygui_running():
            self.process_queue()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()
        if self.is_running:
            self.stop_bot()

if __name__ == "__main__":
    app = HolonicDashboardDPG()
    app.run()
