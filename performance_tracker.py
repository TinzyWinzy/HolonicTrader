
import sqlite3
import pandas as pd
import json
import math
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

DB_PATH = "holonic_trader.db"
console = Console()

def calculate_omega_ratio(returns: list, threshold: float = 0.0) -> float:
    """
    Calculate Omega Ratio.
    Omega(L) = Sum(Gains - L) / Sum(L - Losses)
    """
    if not returns:
        return 0.0
        
    gains = [r - threshold for r in returns if r > threshold]
    losses = [threshold - r for r in returns if r < threshold]
    
    sum_gains = sum(gains)
    sum_losses = sum(losses)
    
    if sum_losses == 0:
        return 100.0 if sum_gains > 0 else 0.0 
        
    return sum_gains / sum_losses

def get_performance_data():
    """
    Fetch high-fidelity performance metrics from the holonic database.
    """
    data = {
        'total_trades': 0,
        'win_rate': 0.0,
        'realized_pnl': 0.0,
        'avg_pnl': 0.0,
        'profit_factor': 0.0,
        'expectancy': 0.0,
        'omega_ratio': 0.0,
        'best_trade': 0.0,
        'worst_trade': 0.0,
        'portfolio_usd': 0.0,
        'equity': 0.0,
        'held_assets': {},
        'recent_trades': []
    }
    
    conn = sqlite3.connect(DB_PATH)
    try:
        # 1. TRADES ANALYSIS
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY id ASC", conn)
        if not df.empty:
            # Exits are rows where cost_usd is 0 (Margin Release) vs Entries which have cost > 0
            # This correctly captures Breakevens (PnL=0) which simple PnL filtering misses.
            exits = df[df['cost_usd'] <= 1e-9].copy()
            
            total_trades = len(exits)
            data['total_trades'] = total_trades
            data['realized_pnl'] = exits['pnl'].sum()
            
            if total_trades > 0:
                winning_trades = exits[exits['pnl'] > 0]
                losing_trades = exits[exits['pnl'] < 0]
                
                data['win_rate'] = (len(winning_trades) / total_trades) * 100
                data['avg_pnl'] = exits['pnl'].mean()
                data['best_trade'] = exits['pnl'].max()
                data['worst_trade'] = exits['pnl'].min()
                
                # Profit Factor = Gross Profit / Gross Loss
                gross_profit = winning_trades['pnl'].sum()
                gross_loss = abs(losing_trades['pnl'].sum())
                data['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else (100.0 if gross_profit > 0 else 0.0)
                
                # Expectancy = (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
                avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
                avg_loss = abs(losing_trades['pnl'].mean()) if not losing_trades.empty else 0
                win_prob = len(winning_trades) / total_trades
                loss_prob = len(losing_trades) / total_trades
                data['expectancy'] = (win_prob * avg_win) - (loss_prob * avg_loss)
                
                pnl_list = exits['pnl'].tolist()
                data['omega_ratio'] = calculate_omega_ratio(pnl_list, threshold=0.0)
            
            # Recent Activity
            recent = df.tail(50).copy()  # Expanded to 50
            data['recent_trades'] = recent.to_dict(orient='records')

        # 2. PORTFOLIO & EQUITY
        port = pd.read_sql_query("SELECT * FROM portfolio", conn)
        if not port.empty:
            data['portfolio_usd'] = port.iloc[0]['balance_usd']
            data['held_assets'] = json.loads(port.iloc[0].get('held_assets', '{}'))
            # Store metadata too if we want to show side precisely
            data['meta'] = json.loads(port.iloc[0].get('position_metadata', '{}'))
            data['equity'] = data['portfolio_usd'] 

    except Exception as e:
        console.print(f"[bold red]Performance Tracker Error:[/bold red] {e}")
    finally:
        conn.close()
        
    return data

def render_performance_report():
    data = get_performance_data()
    
    # 1. Header & Summary Panel
    pnl_color = "green" if data['realized_pnl'] >= 0 else "red"
    summary_text = Text()
    summary_text.append(f"Total PnL: ", style="bold white")
    summary_text.append(f"${data['realized_pnl']:,.2f}", style=f"bold {pnl_color}")
    summary_text.append(f" | Win Rate: ", style="bold white")
    summary_text.append(f"{data['win_rate']:.1f}%", style="bold cyan")
    summary_text.append(f" | Balance: ", style="bold white")
    summary_text.append(f"${data['portfolio_usd']:,.2f}", style="bold green")
    
    console.print(Panel(summary_text, title="[bold gold1]AEHML MONOLITH-V5 PERFORMANCE[/bold gold1]", border_style="bright_blue", box=box.DOUBLE))

    # 2. Advanced Metrics Table
    metrics_table = Table(title="[bold]Advanced Analytics[/bold]", box=box.ROUNDED, header_style="bold magenta")
    metrics_table.add_column("Metric", style="dim")
    metrics_table.add_column("Value", justify="right")
    
    metrics_table.add_row("Profit Factor", f"{data['profit_factor']:.2f}")
    metrics_table.add_row("Expectancy", f"${data['expectancy']:.2f}")
    metrics_table.add_row("Omega Ratio", f"{data['omega_ratio']:.4f}")
    metrics_table.add_row("Best Trade", f"${data['best_trade']:.2f}", style="green")
    metrics_table.add_row("Worst Trade", f"${data['worst_trade']:.2f}", style="red")
    metrics_table.add_row("Total Trades", str(data['total_trades']))
    
    # 3. Active Positions Table
    positions_table = Table(title="[bold]Active Positions[/bold]", box=box.ROUNDED, header_style="bold yellow")
    positions_table.add_column("Asset", style="bold")
    positions_table.add_column("Quantity", justify="right")
    positions_table.add_column("Side", justify="center")
    
    has_pos = False
    for sym, qty in data['held_assets'].items():
        if abs(qty) > 1e-8:
            has_pos = True
            # Use metadata for side if possible
            meta = data.get('meta', {}).get(sym, {})
            direction_meta = meta.get('direction', '')
            if direction_meta == 'BUY' or qty > 0:
                side = "[bold green]LONG[/bold green]"
            else:
                side = "[bold red]SHORT[/bold red]"
            positions_table.add_row(sym, f"{abs(qty):.4f}", side)
    
    if not has_pos:
        positions_table.add_row("NONE", "0.0000", "-")

    # Layout: Metrics and Positions side-by-side
    from rich.columns import Columns
    console.print(Columns([metrics_table, positions_table]))

    # 4. Recent Activity Table
    activity_table = Table(title=f"[bold]Recent Activity (Last {len(data['recent_trades'])})[/bold]", box=box.SIMPLE_HEAVY, header_style="bold cyan")
    activity_table.add_column("Timestamp", style="dim", width=20)
    activity_table.add_column("Symbol")
    activity_table.add_column("Type", justify="center")
    activity_table.add_column("Price", justify="right")
    activity_table.add_column("PnL ($)", justify="right")
    
    for t in reversed(data['recent_trades']):
        pnl_val = t.get('pnl', 0.0)
        p_color = "green" if pnl_val > 0 else ("red" if pnl_val < 0 else "white")
        pnl_str = f"${pnl_val:+.2f}" if pnl_val != 0 else "-"
        
        # Determine precise type
        raw_dir = t['direction']
        if pnl_val != 0:
            type_label = "L-EXIT" if raw_dir == 'SELL' else "S-COVER"
        else:
            type_label = "L-ENTRY" if raw_dir == 'BUY' else "S-ENTRY"
            
        activity_table.add_row(
            t['timestamp'][:19].replace('T', ' '), # Full Date + Time (YYYY-MM-DD HH:MM:SS)
            t['symbol'],
            type_label,
            f"{t['price']:.4f}",
            Text(pnl_str, style=p_color)
        )
    
    console.print(activity_table)

if __name__ == "__main__":
    render_performance_report()
