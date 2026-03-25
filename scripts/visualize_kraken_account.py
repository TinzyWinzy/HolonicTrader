import sys
import os
import time
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())
# Also try adding HolonicTrader subdir if running from root
if os.path.isdir('HolonicTrader'):
    sys.path.append(os.path.join(os.getcwd(), 'HolonicTrader'))

try:
    import config
    from HolonicTrader.agent_kraken import KrakenHolon
except ImportError:
    print("❌ Failed to import HolonicTrader modules. Make sure you are in the project root.")
    sys.exit(1)

# Try importing rich for beautiful output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️ 'rich' library not found. Falling back to standard output.")

def main():
    print("🚀 Initializing Kraken Holon...")
    kraken = KrakenHolon()
    
    print("📡 Fetching Kraken Data...")
    
    # 1. Platform Info
    platform_info = kraken.get_platform_info()
    
    # 2. Equity Truth
    equity_truth = kraken.get_equity_truth()
    
    # 3. Detect Ghost Positions (Pass empty dict to see ALL positions on exchange)
    # The 'ghosts' key will contain all positions not in our empty ledger
    positions_report = kraken.detect_ghost_positions(internal_held_assets={})
    all_positions = positions_report.get('ghosts', {})
    
    # 4. Environment Status (Active symbols)
    active_symbols = list(all_positions.keys())
    # If no positions, just check majors
    if not active_symbols:
        active_symbols = ['BTC/USDT', 'ETH/USDT']
    
    env_status = kraken.monitor_execution_environment(active_symbols)
    
    if RICH_AVAILABLE:
        display_rich_dashboard(platform_info, equity_truth, all_positions, env_status)
    else:
        display_standard_output(platform_info, equity_truth, all_positions, env_status)

def display_rich_dashboard(platform, equity, positions, env):
    console = Console()
    
    # --- Header ---
    console.print(Panel.fit("🐙 [bold purple]KRAKEN COMMAND CENTER[/bold purple]", style="bold white"))
    console.print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # --- System Status ---
    status_color = "green" if platform['status'] == 'HEALTHY' else "red"
    console.print(f"System Status: [{status_color}]{platform['status']}[/]")
    
    env_color = "green" if env.get('status') == 'STABLE' else "red"
    console.print(f"Execution Env: [{env_color}]{env.get('status', 'UNKNOWN')}[/]\n")

    # --- Account Overview Table ---
    grid = Table.grid(expand=True)
    grid.add_column()
    grid.add_column()
    
    # Health Table
    health_table = Table(title="🏥 Account Health", box=box.ROUNDED)
    health_table.add_column("Metric", style="cyan")
    health_table.add_column("Value", style="bold white")
    
    health = platform.get('account_health', {})
    health_table.add_row("Margin Equity", f"${health.get('equity', 0):.2f}")
    health_table.add_row("Used Margin", f"${health.get('used_margin', 0):.2f}")
    health_table.add_row("Available Margin", f"${health.get('available', 0):.2f}")
    
    liq_dist = health.get('liquidation_distance', 0)
    liq_color = "green" if liq_dist > 0.4 else "yellow" if liq_dist > 0.2 else "red"
    health_table.add_row("Liquidation Dist", f"[{liq_color}]{liq_dist*100:.1f}%[/]")
    
    margin_level = health.get('margin_level', 999)
    health_table.add_row("Margin Level", f"{margin_level:.2f}")

    # Truth Table
    truth_table = Table(title="👁️ Equity Truth (Collateral)", box=box.ROUNDED)
    truth_table.add_column("Metric", style="magenta")
    truth_table.add_column("Value", style="bold white")
    
    truth_table.add_row("Collateral Value", f"${equity.get('collateral', 0):.2f}")
    truth_table.add_row("Unrealized PnL", f"${equity.get('unrealized_pnl', 0):.2f}")
    truth_table.add_row("Total Equity", f"${equity.get('equity', 0):.2f}")
    
    grid.add_row(health_table, truth_table)
    console.print(grid)
    console.print("\n")

    # --- Positions Table ---
    pos_table = Table(title="📊 Open Positions (On Exchange)", box=box.ROUNDED, expand=True)
    pos_table.add_column("Symbol", style="cyan")
    pos_table.add_column("Type", style="bold")
    pos_table.add_column("Size", style="white")
    
    if positions:
        for sym, qty in positions.items():
            side = "[green]LONG[/]" if qty > 0 else "[red]SHORT[/]"
            pos_table.add_row(sym, side, f"{abs(qty)}")
    else:
        pos_table.add_row("No Open Positions", "-", "-")
        
    console.print(pos_table)
    
    # --- Funding Intel ---
    market_intel = platform.get('market_intel', {})
    funding_data = market_intel.get('funding', {})
    
    if funding_data:
        fund_table = Table(title="💸 Funding Rates", box=box.SIMPLE)
        fund_table.add_column("Symbol")
        fund_table.add_column("Rate")
        fund_table.add_column("APY")
        
        for sym, data in funding_data.items():
             rate = data.get('rate', 0)
             apy = data.get('apy', 0)
             color = "green" if rate > 0 else "red"
             fund_table.add_row(sym, f"[{color}]{rate:.6f}[/]", f"{apy:.2f}%")
        
        console.print(fund_table)

def display_standard_output(platform, equity, positions, env):
    print("\n--- KRAKEN COMMAND CENTER ---")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Status: {platform['status']}")
    print(f"Env: {env.get('status', 'UNKNOWN')}")
    
    print("\n--- Account Health ---")
    health = platform.get('account_health', {})
    for k, v in health.items():
        print(f"{k}: {v}")
        
    print("\n--- Equity Truth ---")
    for k, v in equity.items():
        print(f"{k}: {v}")
        
    print("\n--- Open Positions ---")
    if positions:
        for sym, qty in positions.items():
            print(f"{sym}: {qty}")
    else:
        print("No Open Positions")

if __name__ == "__main__":
    main()
