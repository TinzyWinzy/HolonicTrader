"""
manage_sol_positions.py
SOL Position Management Script

Purpose: Help manage the 84 accumulated SOL/USDT positions

Options:
1. --summary     : Show current SOL position overview
2. --close N     : Close N oldest SOL positions
3. --close-pct N : Close N% of SOL positions
4. --set-stop    : Set stop-loss at specified price level
5. --consolidate : Close all and reopen as single position (if exchange supports)

Usage:
    python -m HolonicTrader.manage_sol_positions --summary
    python -m HolonicTrader.manage_sol_positions --close 42
    python -m HolonicTrader.manage_sol_positions --close-pct 50
    python -m HolonicTrader.manage_sol_positions --set-stop 75.00
"""

import sqlite3
import sys
import os
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt
from rich import box

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

console = Console()

DB_PATH = "holonic_trader.db"

# Kraken Futures symbol mapping
SOL_SYMBOL = 'SOL/USD:USD'  # Kraken Futures perpetual


def get_sol_positions(db_path=DB_PATH):
    """Fetch all SOL positions from database."""
    conn = sqlite3.connect(db_path)
    try:
        query = """
            SELECT 
                id, symbol, direction, quantity, price, cost_usd,
                unrealized_pnl, unrealized_pnl_percent,
                mfe, mae, timestamp
            FROM trades 
            WHERE symbol = 'SOL/USDT' AND cost_usd > 1e-9
            ORDER BY timestamp ASC
        """
        cursor = conn.execute(query)
        columns = [desc[0] for desc in cursor.description]
        positions = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return positions
    except Exception as e:
        console.print(f"[red]Error fetching positions:[/red] {e}")
        return []
    finally:
        conn.close()


def get_current_sol_price():
    """Fetch current SOL price from Kraken Futures."""
    if not HAS_CCXT:
        return None
    
    try:
        exchange = ccxt.krakenfutures({'enableRateLimit': True})
        ticker = exchange.fetch_ticker(SOL_SYMBOL)
        return ticker.get('last', 0)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not fetch live price: {e}[/yellow]")
        return None


def show_summary(positions):
    """Display SOL position summary."""
    if not positions:
        console.print("[bold yellow]No SOL positions found in database.[/bold yellow]\n")
        return
    
    # Fetch live price
    live_price = get_current_sol_price()
    
    # Calculate statistics
    total_margin = sum(p['cost_usd'] for p in positions)
    total_quantity = sum(p['quantity'] for p in positions)
    avg_entry = total_quantity / len(positions) if positions else 0  # Weighted avg
    weighted_entry = sum(p['price'] * p['quantity'] for p in positions) / total_quantity if total_quantity > 0 else 0
    
    # Calculate live PnL
    if live_price and live_price > 0:
        total_value = total_quantity * live_price
        live_pnl = total_value - total_margin
        live_pnl_pct = (live_pnl / total_margin * 100) if total_margin > 0 else 0
    else:
        live_pnl = sum(p['unrealized_pnl'] for p in positions)
        live_pnl_pct = (live_pnl / total_margin * 100) if total_margin > 0 else 0
    
    # Display summary panel
    summary_text = f"""[bold]SOL/USDT Position Summary[/bold]

Total Positions: [cyan]{len(positions)}[/cyan]
Total Margin: [cyan]${total_margin:.2f}[/cyan]
Total Quantity: [cyan]{total_quantity:.4f} SOL[/cyan]
Weighted Avg Entry: [cyan]${weighted_entry:.2f}[/cyan]
"""
    
    if live_price:
        summary_text += f"\nCurrent Price: [cyan]${live_price:.2f}[/cyan]"
        pnl_color = "green" if live_pnl >= 0 else "red"
        summary_text += f"\nLive PnL: [{pnl_color}]${live_pnl:+.2f} ({live_pnl_pct:+.2f}%)[/{pnl_color}]"
    else:
        summary_text += f"\nUnrealized PnL: ${live_pnl:+.2f} ({live_pnl_pct:+.2f}%)"
    
    console.print(Panel(
        summary_text,
        title="[bold gold1]SOL POSITION OVERVIEW[/bold gold1]",
        border_style="bright_cyan",
        box=box.DOUBLE
    ))
    
    # Show distribution table
    table = Table(title="Position Distribution by Entry Price", box=box.ROUNDED)
    table.add_column("Entry Price Range", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Total Margin", justify="right")
    table.add_column("Qty", justify="right")
    
    # Group by price ranges
    ranges = [
        ("< $78", 0, 78),
        ("$78 - $80", 78, 80),
        ("$80 - $82", 80, 82),
        ("$82 - $84", 82, 84),
        ("$84 - $86", 84, 86),
        ("> $86", 86, 999)
    ]
    
    for label, low, high in ranges:
        range_positions = [p for p in positions if low <= p['price'] < high]
        if range_positions:
            count = len(range_positions)
            margin = sum(p['cost_usd'] for p in range_positions)
            qty = sum(p['quantity'] for p in range_positions)
            table.add_row(label, f"{count}", f"${margin:.2f}", f"{qty:.4f}")
    
    console.print(table)
    console.print("\n")


def close_positions(n_positions, db_path=DB_PATH):
    """Mark N oldest SOL positions as closed in database."""
    positions = get_sol_positions(db_path)
    
    if not positions:
        console.print("[bold yellow]No SOL positions to close.[/bold yellow]\n")
        return
    
    if n_positions >= len(positions):
        console.print(f"[yellow]Warning: Requested to close {n_positions} positions, but only {len(positions)} exist.[/yellow]\n")
        n_positions = len(positions)
    
    # Get oldest N positions
    to_close = positions[:n_positions]
    
    console.print(Panel(
        f"[bold]Closing {n_positions} SOL Positions[/bold]\n\n"
        f"Total Margin to Close: ${sum(p['cost_usd'] for p in to_close):.2f}\n"
        f"Total Quantity: {sum(p['quantity'] for p in to_close):.4f} SOL\n"
        f"Average Entry: ${sum(p['price'] * p['quantity'] for p in to_close) / sum(p['quantity'] for p in to_close):.2f}",
        title="[bold yellow]CONFIRMATION REQUIRED[/bold yellow]",
        border_style="bright_yellow",
        box=box.ROUNDED
    ))
    
    if not Confirm.ask("\nProceed with closing these positions?"):
        console.print("[bold]Operation cancelled.[/bold]\n")
        return
    
    # In a real scenario, this would execute market orders on the exchange
    # For now, we'll just log what would happen
    
    conn = sqlite3.connect(db_path)
    try:
        for pos in to_close:
            console.print(f"  [dim]Would close: ID {pos['id']} - {pos['quantity']:.4f} SOL @ ${pos['price']:.2f}[/dim]")
            
            # NOTE: In production, you would:
            # 1. Execute market sell order on Kraken
            # 2. Update database with exit record
            # 3. Record realized PnL
        
        console.print(f"\n[bold green]✓ Would close {n_positions} SOL positions[/bold green]")
        console.print("[yellow]Note: This is a simulation. To actually close, use the trading bot or exchange directly.[/yellow]\n")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]\n")
    finally:
        conn.close()


def close_percentage(pct, db_path=DB_PATH):
    """Close N% of SOL positions."""
    positions = get_sol_positions(db_path)
    
    if not positions:
        console.print("[bold yellow]No SOL positions to close.[/bold yellow]\n")
        return
    
    n_positions = max(1, int(len(positions) * pct / 100))
    close_positions(n_positions, db_path)


def set_stop_loss(stop_price, db_path=DB_PATH):
    """Set stop-loss for all SOL positions."""
    positions = get_sol_positions(db_path)
    
    if not positions:
        console.print("[bold yellow]No SOL positions found.[/bold yellow]\n")
        return
    
    current_price = get_current_sol_price()
    
    if current_price and current_price > 0:
        distance = (current_price - stop_price) / current_price * 100
        console.print(Panel(
            f"[bold]Stop-Loss Configuration[/bold]\n\n"
            f"Current Price: [cyan]${current_price:.2f}[/cyan]\n"
            f"Stop Price: [yellow]${stop_price:.2f}[/yellow]\n"
            f"Distance: [yellow]{distance:.2f}%[/yellow]\n\n"
            f"Positions Affected: [cyan]{len(positions)}[/cyan]",
            title="[bold]STOP-LOSS SETUP[/bold]",
            border_style="bright_red",
            box=box.ROUNDED
        ))
        
        # In production, this would place stop-market orders on the exchange
        console.print("\n[yellow]Note: To actually set stop-loss, use the exchange directly or update the bot's position metadata.[/yellow]\n")
        console.print("Recommended approach:")
        console.print("  1. Place stop-market order for total quantity at exchange")
        console.print(f"  2. Or close {len(positions)} positions and reopen with SL=${stop_price:.2f}\n")
    else:
        console.print("[red]Could not fetch current price. Stop-loss not set.[/red]\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        console.print(Panel(
            "[bold]SOL Position Management Script[/bold]\n\n"
            "Usage:\n"
            "  python -m HolonicTrader.manage_sol_positions --summary\n"
            "  python -m HolonicTrader.manage_sol_positions --close N\n"
            "  python -m HolonicTrader.manage_sol_positions --close-pct N\n"
            "  python -m HolonicTrader.manage_sol_positions --set-stop PRICE\n\n"
            "[yellow]Run with --help for more options[/yellow]",
            title="[bold gold1]📊 SOL MANAGEMENT TOOL[/bold gold1]",
            border_style="bright_cyan",
            box=box.DOUBLE
        ))
        return
    
    # Parse arguments
    if '--summary' in sys.argv:
        positions = get_sol_positions()
        show_summary(positions)
    
    elif '--close' in sys.argv:
        try:
            idx = sys.argv.index('--close')
            n = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 0
            if n <= 0:
                raise ValueError()
            close_positions(n)
        except (ValueError, IndexError):
            console.print("[red]Invalid argument. Use: --close N (where N is a positive integer)[/red]\n")
    
    elif '--close-pct' in sys.argv:
        try:
            idx = sys.argv.index('--close-pct')
            pct = float(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 0
            if pct <= 0 or pct > 100:
                raise ValueError()
            close_percentage(pct)
        except (ValueError, IndexError):
            console.print("[red]Invalid argument. Use: --close-pct N (where N is 0-100)[/red]\n")
    
    elif '--set-stop' in sys.argv:
        try:
            idx = sys.argv.index('--set-stop')
            price = float(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 0
            if price <= 0:
                raise ValueError()
            set_stop_loss(price)
        except (ValueError, IndexError):
            console.print("[red]Invalid argument. Use: --set-stop PRICE (e.g., --set-stop 75.00)[/red]\n")
    
    elif '--help' in sys.argv or '-h' in sys.argv:
        console.print("""
[bold]SOL Position Management - Help[/bold]

[bold]--summary[/bold]       Show current SOL position overview with PnL
[bold]--close N[/bold]       Close N oldest SOL positions
[bold]--close-pct N[/bold]   Close N% of SOL positions (rounded down)
[bold]--set-stop P[/bold]    Set stop-loss at price P
[bold]--help[/bold]          Show this help message

[bold]Examples:[/bold]
  python -m HolonicTrader.manage_sol_positions --summary
  python -m HolonicTrader.manage_sol_positions --close 42
  python -m HolonicTrader.manage_sol_positions --close-pct 50
  python -m HolonicTrader.manage_sol_positions --set-stop 75.00

[bold]Recommendations:[/bold]
  - Start with --summary to see current state
  - Consider closing 50% to reduce concentration risk
  - Set stop-loss at $75 to limit total loss to ~$170
  - Monitor for recovery to $82-84 for partial exit
""")
    
    else:
        console.print("[red]Unknown argument. Use --help for usage information.[/red]\n")


if __name__ == "__main__":
    main()
