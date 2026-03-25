"""
Performance Tracker - AEGIS QUANTSEC Trading Analytics

Provides comprehensive performance metrics and analytics for the HolonicTrader system.

Features:
- Trade statistics (win rate, PnL, expectancy)
- Risk metrics (Sharpe, Omega, max drawdown)
- Portfolio tracking
- Rich console reporting

Addresses: M-02 Module Import Failures

Author: AEGIS QuantSec v1.0
Date: 2026-03-15
"""

import sqlite3
import pandas as pd
import json
import math
from datetime import datetime
from typing import Dict, List, Any, Optional
import threading

# Optional: Rich console for beautiful reports
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("[PerformanceTracker] ℹ️ Rich not installed. Install with: pip install rich")

DB_PATH = "holonic_trader.db"

if RICH_AVAILABLE:
    console = Console()


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def calculate_omega_ratio(returns: list, threshold: float = 0.0) -> float:
    """
    Calculate Omega Ratio.
    Omega(L) = Sum(Gains - L) / Sum(L - Losses)
    
    Higher is better. >1.0 indicates positive expectancy.
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


def calculate_sharpe_ratio(returns: list, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe Ratio.
    Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
    
    Higher is better. >1.0 is good, >2.0 is excellent.
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    import numpy as np
    returns_array = np.array(returns)
    mean_return = returns_array.mean()
    std_dev = returns_array.std()
    
    if std_dev == 0:
        return 0.0
    
    return (mean_return - risk_free_rate) / std_dev


def calculate_max_drawdown(equity_curve: list) -> float:
    """
    Calculate Maximum Drawdown.
    The largest peak-to-trough decline in portfolio value.
    
    Lower is better. Expressed as positive percentage.
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0
    
    peak = equity_curve[0]
    max_dd = 0.0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        
        drawdown = (peak - value) / peak if peak > 0 else 0
        max_dd = max(max_dd, drawdown)
    
    return max_dd


def get_performance_data() -> Dict[str, Any]:
    """
    Fetch high-fidelity performance metrics from the holonic database.
    
    Returns:
        Dictionary with comprehensive performance data
    """
    data = {
        'total_trades': 0,
        'win_rate': 0.0,
        'realized_pnl': 0.0,
        'avg_pnl': 0.0,
        'profit_factor': 0.0,
        'expectancy': 0.0,
        'omega_ratio': 0.0,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0,
        'best_trade': 0.0,
        'worst_trade': 0.0,
        'portfolio_usd': 0.0,
        'equity': 0.0,
        'held_assets': {},
        'recent_trades': [],
        'equity_curve': []
    }
    
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # 1. TRADES ANALYSIS
        try:
            df = pd.read_sql_query("SELECT * FROM trades ORDER BY id ASC", conn)
            if not df.empty:
                # Exits are rows where cost_usd is 0 (Margin Release)
                exits = df[df['cost_usd'] <= 1e-9].copy()

                total_trades = len(exits)
                data['total_trades'] = total_trades
                data['realized_pnl'] = float(exits['pnl'].sum())

                if total_trades > 0:
                    winning_trades = exits[exits['pnl'] > 0]
                    losing_trades = exits[exits['pnl'] < 0]

                    data['win_rate'] = (len(winning_trades) / total_trades) * 100
                    data['avg_pnl'] = float(exits['pnl'].mean())
                    data['best_trade'] = float(exits['pnl'].max())
                    data['worst_trade'] = float(exits['pnl'].min())

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

                    # Omega Ratio
                    pnl_list = exits['pnl'].tolist()
                    data['omega_ratio'] = calculate_omega_ratio(pnl_list, threshold=0.0)
                    
                    # Sharpe Ratio (annualized, assuming daily returns)
                    data['sharpe_ratio'] = calculate_sharpe_ratio(pnl_list) * math.sqrt(365)

                # Recent Activity
                recent = df.tail(50).copy()
                data['recent_trades'] = recent.to_dict(orient='records')
                
                # Build equity curve from cumulative PnL
                if not exits.empty:
                    data['equity_curve'] = exits['pnl'].cumsum().tolist()
                    data['max_drawdown'] = calculate_max_drawdown(data['equity_curve'])
        except Exception as e:
            print(f"[PerformanceTracker] Error reading trades: {e}")
        
        # 2. PORTFOLIO & EQUITY
        try:
            port = pd.read_sql_query("SELECT * FROM portfolio", conn)
            if not port.empty:
                data['portfolio_usd'] = float(port.iloc[0]['balance_usd'])
                data['held_assets'] = json.loads(port.iloc[0].get('held_assets', '{}'))
        except Exception as e:
            print(f"[PerformanceTracker] Error reading portfolio: {e}")
        
        conn.close()
        
    except Exception as e:
        print(f"[PerformanceTracker] Database error: {e}")
    
    return data


# =============================================================================
# RICH CONSOLE REPORTING
# =============================================================================

def render_performance_report(data: Optional[Dict] = None) -> str:
    """
    Render a beautiful performance report using Rich library.
    
    Args:
        data: Performance data dict. If None, fetches from DB.
        
    Returns:
        Rendered report string
    """
    if not RICH_AVAILABLE:
        return "Rich library not installed. Install with: pip install rich"
    
    if data is None:
        data = get_performance_data()
    
    # Create summary panel
    summary_text = Text()
    summary_text.append("Total Trades: ", style="bold")
    summary_text.append(f"{data['total_trades']}\n", style="cyan")
    summary_text.append("Win Rate: ", style="bold")
    summary_text.append(f"{data['win_rate']:.1f}%\n", style="green" if data['win_rate'] > 50 else "red")
    summary_text.append("Realized PnL: ", style="bold")
    pnl_color = "green" if data['realized_pnl'] > 0 else "red"
    summary_text.append(f"${data['realized_pnl']:.2f}\n", style=pnl_color)
    summary_text.append("Profit Factor: ", style="bold")
    summary_text.append(f"{data['profit_factor']:.2f}\n", style="green" if data['profit_factor'] > 1.5 else "yellow")
    
    summary_panel = Panel(
        summary_text,
        title="[bold]Trading Summary[/bold]",
        border_style="blue"
    )
    
    # Create metrics table
    metrics_table = Table(title="Performance Metrics", box=box.ROUNDED)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", justify="right")
    metrics_table.add_column("Rating", justify="right")
    
    # Add rows with ratings
    def rate_omega(value):
        if value > 2.0: return "[green]Excellent"
        if value > 1.5: return "[green]Good"
        if value > 1.0: return "[yellow]Average"
        return "[red]Poor"
    
    def rate_sharpe(value):
        if value > 2.0: return "[green]Excellent"
        if value > 1.0: return "[green]Good"
        if value > 0.5: return "[yellow]Average"
        return "[red]Poor"
    
    def rate_drawdown(value):
        if value < 0.1: return "[green]Excellent"
        if value < 0.2: return "[green]Good"
        if value < 0.3: return "[yellow]Average"
        return "[red]Poor"
    
    metrics_table.add_row("Omega Ratio", f"{data['omega_ratio']:.2f}", rate_omega(data['omega_ratio']))
    metrics_table.add_row("Sharpe Ratio", f"{data['sharpe_ratio']:.2f}", rate_sharpe(data['sharpe_ratio']))
    metrics_table.add_row("Max Drawdown", f"{data['max_drawdown']*100:.1f}%", rate_drawdown(data['max_drawdown']))
    metrics_table.add_row("Expectancy", f"${data['expectancy']:.2f}", "[green]Positive" if data['expectancy'] > 0 else "[red]Negative")
    metrics_table.add_row("Avg PnL", f"${data['avg_pnl']:.2f}", "[green]" if data['avg_pnl'] > 0 else "[red]")
    metrics_table.add_row("Best Trade", f"${data['best_trade']:.2f}", "[green]")
    metrics_table.add_row("Worst Trade", f"${data['worst_trade']:.2f}", "[red]")
    
    # Create portfolio table
    portfolio_table = Table(title="Portfolio", box=box.ROUNDED)
    portfolio_table.add_column("Asset", style="cyan")
    portfolio_table.add_column("Quantity", justify="right")
    portfolio_table.add_column("Value (USD)", justify="right")
    
    total_value = data['portfolio_usd']
    for symbol, qty in data.get('held_assets', {}).items():
        if qty > 0:
            portfolio_table.add_row(symbol, f"{qty:.4f}", "[cyan]Active")
    
    portfolio_table.add_row("TOTAL", "", f"${total_value:.2f}", style="bold")
    
    # Render to string
    from io import StringIO
    output = StringIO()
    console.file = output
    
    console.print(summary_panel)
    console.print("\n")
    console.print(metrics_table)
    console.print("\n")
    console.print(portfolio_table)
    
    result = output.getvalue()
    console.file = None  # Reset
    
    return result


# =============================================================================
# DATABASE MANAGER (Compatibility Layer)
# =============================================================================

class DatabaseManager:
    """
    Simplified database manager for compatibility.
    Provides basic operations and trade logging.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_tables()

    def _init_tables(self):
        """Initialize database tables if they don't exist."""
        with self._lock:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Create trades table for PnL tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    size REAL NOT NULL,
                    pnl REAL DEFAULT 0,
                    pnl_percent REAL DEFAULT 0,
                    cost_usd REAL DEFAULT 0,
                    exit_reason TEXT,
                    strategy TEXT DEFAULT 'DIRECTIONAL',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    closed_at TIMESTAMP
                )
            ''')

            # Create portfolio table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    balance_usd REAL NOT NULL,
                    held_assets TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()
            conn.close()

    def save_trade(self, trade_record: Dict[str, Any]):
        """Save a trade record to the database."""
        with self._lock:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO trades (symbol, direction, entry_price, exit_price, size, pnl, pnl_percent, cost_usd, exit_reason, strategy, closed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_record.get('symbol', ''),
                trade_record.get('direction', ''),
                trade_record.get('entry_price', 0),
                trade_record.get('exit_price'),
                trade_record.get('size', 0),
                trade_record.get('pnl', 0),
                trade_record.get('pnl_percent', 0),
                trade_record.get('cost_usd', 0),
                trade_record.get('exit_reason', ''),
                trade_record.get('strategy', 'DIRECTIONAL'),
                trade_record.get('closed_at', datetime.utcnow().isoformat())
            ))

            conn.commit()
            conn.close()

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute query and return results as list of dicts."""
        with self._lock:
            conn = self.get_connection()
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            try:
                cursor.execute(query, params)
                results = [dict(row) for row in cursor.fetchall()]
                return results
            finally:
                conn.close()
    
    def execute_write(self, query: str, params: tuple = ()) -> int:
        """Execute write query and return rows affected."""
        with self._lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute(query, params)
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_performance_db: Optional[DatabaseManager] = None


def get_performance_database() -> DatabaseManager:
    """Get global database manager instance."""
    global _performance_db
    if _performance_db is None:
        _performance_db = DatabaseManager()
    return _performance_db
