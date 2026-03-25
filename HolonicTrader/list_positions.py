"""
list_positions.py
List all open positions with current market prices and live PnL.

Usage:
    python -m HolonicTrader.list_positions [--detail]
"""

import sqlite3
import pandas as pd
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
import sys

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

console = Console()

# Kraken Futures symbols mapping
SYMBOL_MAP = {
    'BTC/USDT': 'XBT/USDT:USDT',
    'SOL/USDT': 'SOL/USDT:USDT',
    'ETH/USDT': 'ETH/USDT:USDT',
}


def fetch_current_prices():
    """Fetch current market prices from exchange."""
    if not HAS_CCXT:
        console.print("[yellow]CCXT not available. Using database prices only.[/yellow]\n")
        return {}
    
    try:
        # Use same initialization as agent_actuator
        exchange = ccxt.krakenfutures({
            'enableRateLimit': True,
        })
        exchange.load_markets()
        
        prices = {}
        
        # Fallback: fetch individual tickers (more reliable than fetch_tickers)
        # Note: Kraken Futures uses USD not USDT, and BTC not XBT
        symbol_map = {
            'BTC/USDT': 'BTC/USD:USD',
            'SOL/USDT': 'SOL/USD:USD', 
            'ETH/USDT': 'ETH/USD:USD'
        }
        
        console.print("[cyan]Fetching individual tickers...[/cyan]\n")
        
        for internal_sym, exec_sym in symbol_map.items():
            try:
                ticker = exchange.fetch_ticker(exec_sym)
                last_price = ticker.get('last')
                if last_price and last_price > 0:
                    prices[internal_sym] = float(last_price)
                    console.print(f"[green]✓[/green] {internal_sym}: ${last_price:,.2f}")
                else:
                    # Try mark price as fallback
                    mark = ticker.get('mark', ticker.get('close', 0))
                    if mark and mark > 0:
                        prices[internal_sym] = float(mark)
                        console.print(f"[yellow]⚠[/yellow] {internal_sym}: ${mark:,.2f} (mark price)")
                    else:
                        console.print(f"[red]✗[/red] {internal_sym}: No valid price")
            except Exception as e2:
                console.print(f"[red]✗[/red] {internal_sym}: {str(e2)[:60]}")
        
        return {k: v for k, v in prices.items() if v and v > 0}
    except Exception as e:
        console.print(f"[red]Error fetching prices:[/red] {e}")
        return {}


def list_positions(db_path: str = "holonic_trader.db", show_detail: bool = False, live_prices: dict = None):
    """
    List all open positions with PnL analysis.
    
    Args:
        db_path: Path to database
        show_detail: Whether to show all positions individually
        live_prices: Dict of current market prices
    """
    conn = sqlite3.connect(db_path)
    try:
        # Fetch open positions (entries with cost > 0)
        query = """
            SELECT 
                symbol, direction, quantity, price, cost_usd,
                unrealized_pnl, unrealized_pnl_percent,
                mfe, mae, timestamp
            FROM trades 
            WHERE cost_usd > 1e-9
            ORDER BY symbol, timestamp ASC
        """
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            console.print("[bold yellow]No open positions found.[/bold yellow]\n")
            return
        
        # Calculate live PnL if prices available
        if live_prices:
            df['current_price'] = df['symbol'].map(live_prices)
            df['live_pnl'] = df.apply(
                lambda row: (row['current_price'] - row['price']) * row['quantity'] 
                if row['direction'] == 'BUY' and row['current_price'] 
                else (row['price'] - row['current_price']) * abs(row['quantity'])
                if row['direction'] == 'SELL' and row['current_price']
                else 0.0,
                axis=1
            )
            df['live_pnl_pct'] = df['live_pnl'] / df['cost_usd'] * 100
        
        # Group by symbol for summary
        symbol_groups = df.groupby('symbol')
        
        # Header
        console.print(Panel(
            f"[bold]OPEN POSITIONS SUMMARY[/bold]\n"
            f"Total Positions: {len(df)} | Symbols: {len(symbol_groups)} | "
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            title="[bold gold1]POSITION OVERVIEW[/bold gold1]",
            border_style="bright_cyan",
            box=box.DOUBLE
        ))
        console.print("\n")
        
        # Summary table by symbol
        summary_table = Table(title="Positions by Symbol", box=box.ROUNDED, header_style="bold magenta")
        summary_table.add_column("Symbol", style="cyan")
        summary_table.add_column("Count", justify="right")
        summary_table.add_column("Total Margin", justify="right")
        summary_table.add_column("Avg Entry", justify="right")
        
        if live_prices:
            summary_table.add_column("Current Price", justify="right")
            summary_table.add_column("Live PnL", justify="right")
            summary_table.add_column("Live %", justify="right")
        else:
            summary_table.add_column("Unr PnL", justify="right")
            summary_table.add_column("Unr %", justify="right")
        
        summary_table.add_column("Direction", justify="center")
        
        total_margin = 0
        total_pnl = 0
        
        for symbol, group in symbol_groups:
            count = len(group)
            margin = group['cost_usd'].sum()
            avg_entry = (group['price'] * group['quantity']).sum() / group['quantity'].sum()
            
            if live_prices and symbol in live_prices:
                current_price = live_prices.get(symbol)
                if not current_price or current_price <= 0:
                    # Skip live calculation if price invalid
                    unr_pnl = group['unrealized_pnl'].sum()
                    unr_pct = group['unrealized_pnl_percent'].mean()
                    direction = group['direction'].iloc[0]
                    pnl_style = "green" if unr_pnl >= 0 else "red"
                    
                    summary_table.add_row(
                        symbol,
                        f"{count}",
                        f"${margin:.2f}",
                        f"${avg_entry:.4f}",
                        f"[{pnl_style}]${unr_pnl:+.2f}[/{pnl_style}]",
                        f"[{pnl_style}]{unr_pct:+.2f}%[/{pnl_style}]",
                        "[green]LONG[/green]" if direction == 'BUY' else "[red]SHORT[/red]"
                    )
                    total_margin += margin
                    total_pnl += unr_pnl
                    continue
                
                total_qty = group['quantity'].sum()
                direction = group['direction'].iloc[0]
                
                # Calculate PnL for group
                if direction == 'BUY':
                    pnl = (current_price - avg_entry) * total_qty
                else:
                    pnl = (avg_entry - current_price) * total_qty
                
                pnl_pct = pnl / margin * 100 if margin > 0 else 0
                
                pnl_style = "green" if pnl >= 0 else "red"
                price_change = current_price - avg_entry
                price_pct = price_change / avg_entry * 100 if avg_entry > 0 else 0
                price_style = "green" if price_change >= 0 else "red"
                
                summary_table.add_row(
                    symbol,
                    f"{count}",
                    f"${margin:.2f}",
                    f"${avg_entry:.4f}",
                    f"[{price_style}]${current_price:,.2f}[/{price_style}]",
                    f"[{pnl_style}]${pnl:+.2f}[/{pnl_style}]",
                    f"[{pnl_style}]{pnl_pct:+.2f}%[/{pnl_style}]",
                    "[green]LONG[/green]" if direction == 'BUY' else "[red]SHORT[/red]"
                )
            else:
                unr_pnl = group['unrealized_pnl'].sum()
                unr_pct = group['unrealized_pnl_percent'].mean()
                direction = group['direction'].iloc[0]
                pnl_style = "green" if unr_pnl >= 0 else "red"
                
                summary_table.add_row(
                    symbol,
                    f"{count}",
                    f"${margin:.2f}",
                    f"${avg_entry:.4f}",
                    f"[{pnl_style}]${unr_pnl:+.2f}[/{pnl_style}]",
                    f"[{pnl_style}]{unr_pct:+.2f}%[/{pnl_style}]",
                    "[green]LONG[/green]" if direction == 'BUY' else "[red]SHORT[/red]"
                )
            
            total_margin += margin
            if live_prices and symbol in live_prices and live_prices.get(symbol, 0) > 0:
                total_pnl += pnl
            else:
                total_pnl += unr_pnl
        
        console.print(summary_table)
        
        # Overall summary
        console.print("\n")
        overall_pnl_style = "green" if total_pnl >= 0 else "red"
        total_return_pct = (total_pnl / total_margin * 100) if total_margin > 0 else 0
        
        summary_text = Text()
        summary_text.append("Total Margin: ", style="bold white")
        summary_text.append(f"${total_margin:.2f}", style="bold cyan")
        
        if live_prices:
            summary_text.append("  |  Live PnL: ", style="bold white")
            summary_text.append(f"${total_pnl:+.2f}", style=f"bold {overall_pnl_style}")
            summary_text.append("  |  Return: ", style="bold white")
            summary_text.append(f"{total_return_pct:+.2f}%", style=f"bold {overall_pnl_style}")
        else:
            summary_text.append("  |  Unrealized PnL: ", style="bold white")
            summary_text.append(f"${total_pnl:+.2f}", style=f"bold {overall_pnl_style}")
            summary_text.append("  |  Return: ", style="bold white")
            summary_text.append(f"{total_return_pct:+.2f}%", style=f"bold {overall_pnl_style}")
        
        console.print(Panel(summary_text, border_style="bright_green"))
        
        # Detailed view
        if show_detail or len(df) <= 20:
            console.print("\n")
            detail_table = Table(title="All Positions", box=box.SIMPLE)
            detail_table.add_column("Symbol", style="cyan")
            detail_table.add_column("Dir", justify="center")
            detail_table.add_column("Quantity", justify="right")
            detail_table.add_column("Entry", justify="right")
            detail_table.add_column("Margin", justify="right")
            
            if live_prices:
                detail_table.add_column("Current", justify="right")
                detail_table.add_column("Live PnL", justify="right")
                detail_table.add_column("Live %", justify="right")
            else:
                detail_table.add_column("Unr PnL", justify="right")
                detail_table.add_column("Unr %", justify="right")
            
            detail_table.add_column("MFE", justify="right")
            detail_table.add_column("MAE", justify="right")
            
            for _, row in df.iterrows():
                if live_prices and row['symbol'] in live_prices:
                    current = row['current_price']
                    pnl = row['live_pnl']
                    pnl_pct = row['live_pnl_pct']
                else:
                    current = row['price']
                    pnl = row['unrealized_pnl']
                    pnl_pct = row['unrealized_pnl_percent']
                
                pnl_style = "green" if pnl >= 0 else "red"
                
                detail_table.add_row(
                    row['symbol'],
                    "[green]BUY[/green]" if row['direction'] == 'BUY' else "[red]SELL[/red]",
                    f"{row['quantity']:.4f}",
                    f"${row['price']:.4f}",
                    f"${row['cost_usd']:.2f}",
                    f"${current:,.2f}" if live_prices else f"[{pnl_style}]${pnl:+.2f}[/{pnl_style}]",
                    f"[{pnl_style}]{pnl:+.2f}[/{pnl_style}]" if live_prices else f"[{pnl_style}]{pnl_pct:+.2f}%[/{pnl_style}]",
                    f"[{pnl_style}]{pnl_pct:+.2f}%[/{pnl_style}]" if live_prices else "",
                    f"{row['mfe']:+.2f}%",
                    f"{row['mae']:+.2f}%"
                )
            
            console.print(detail_table)
        else:
            console.print(f"\n[dim]Showing summary only ({len(df)} positions total). Use --detail flag for full list.[/dim]")
        
        console.print("\n")
        
    except Exception as e:
        console.print(f"[red]Error listing positions:[/red] {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()


if __name__ == "__main__":
    db_path = "holonic_trader.db"
    show_detail = "--detail" in sys.argv
    
    console.print("\n[bold cyan]Fetching current market prices...[/bold cyan]\n")
    prices = fetch_current_prices()
    
    if prices:
        console.print(f"\n[bold green]✓ Live prices fetched for {len(prices)} symbols[/bold green]\n")
    else:
        console.print("\n[yellow]⚠ Using database prices only (no live updates)[/yellow]\n")
    
    list_positions(db_path, show_detail=show_detail, live_prices=prices)
