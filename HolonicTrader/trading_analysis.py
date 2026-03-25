"""
trading_analysis.py
Comprehensive Trading System Analysis Module

Provides analysis for:
1. Execution Quality - Order fills and slippage tracking
2. Position Management - Exit logic effectiveness monitoring
3. Performance Metrics - Sharpe ratio and drawdown analysis

Usage:
    from HolonicTrader.trading_analysis import (
        ExecutionQualityAnalyzer,
        PositionManagementAnalyzer,
        PerformanceMetricsAnalyzer,
        TradingAnalysisDashboard
    )
    
    # Run all analyses
    dashboard = TradingAnalysisDashboard()
    dashboard.generate_report()
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.columns import Columns
import json
import math
import sys
import os

# Fix Windows console encoding for Unicode support
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7 workaround
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

console = Console()

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ExecutionMetrics:
    """Metrics related to order execution quality."""
    total_orders: int = 0
    filled_orders: int = 0
    partial_fills: int = 0
    failed_orders: int = 0
    fill_rate: float = 0.0
    avg_slippage_pct: float = 0.0
    max_slippage_pct: float = 0.0
    slippage_std: float = 0.0
    avg_fill_time_sec: float = 0.0
    price_improvement_count: int = 0
    price_worsening_count: int = 0
    high_impact_trades: int = 0
    liquidity_rejections: int = 0


@dataclass
class ExitEffectivenessMetrics:
    """Metrics related to exit logic effectiveness."""
    total_exits: int = 0
    tp_hits: int = 0
    sl_hits: int = 0
    thesis_exits: int = 0
    manual_exits: int = 0
    tp_hit_rate: float = 0.0
    sl_hit_rate: float = 0.0
    avg_exit_efficiency: float = 0.0  # Actual exit price / Optimal exit price
    premature_exits: int = 0  # Exits that would have been winners if held
    late_exits: int = 0  # Exits that gave back >50% of max profit
    avg_holding_period_mins: float = 0.0
    mfe_realization_rate: float = 0.0  # How much of MFE was captured


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    total_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_usd: float = 0.0
    max_drawdown_duration_days: int = 0
    current_drawdown_pct: float = 0.0
    avg_drawdown_pct: float = 0.0
    drawdown_frequency: int = 0
    omega_ratio: float = 0.0
    tail_ratio: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    var_95: float = 0.0  # Value at Risk (95%)
    cvar_95: float = 0.0  # Conditional VaR (95%)


# ============================================================================
# EXECUTION QUALITY ANALYZER
# ============================================================================

class ExecutionQualityAnalyzer:
    """
    Analyzes order execution quality including fills and slippage.
    
    Key Metrics:
    - Fill Rate: Percentage of orders successfully executed
    - Slippage: Difference between expected and actual fill price
    - Price Improvement: Orders filled better than expected
    - Liquidity Impact: Orders rejected due to insufficient liquidity
    """
    
    def __init__(self, db_path: str = "holonic_trader.db"):
        self.db_path = db_path
        self.metrics = ExecutionMetrics()
        self.slippage_history: List[Dict] = []
        self.fill_history: List[Dict] = []
    
    def fetch_order_data(self) -> pd.DataFrame:
        """Fetch order data from database."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Fetch trades with execution details
            # Note: Using 'quantity' column (not 'size') per actual schema
            # metadata column may not exist in all schemas
            query = """
                SELECT 
                    id, symbol, timestamp, direction, price,
                    quantity, pnl, cost_usd, pnl_percent,
                    mfe, mae
                FROM trades 
                ORDER BY id ASC
            """
            df = pd.read_sql_query(query, conn)
            
            # Rename quantity to size for internal consistency
            if 'quantity' in df.columns:
                df['size'] = df['quantity']
            
            # Parse metadata for execution details (if column exists)
            if 'metadata' in df.columns:
                df['meta_dict'] = df['metadata'].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else {}
                )
                df['expected_price'] = df['meta_dict'].apply(
                    lambda x: x.get('expected_price', x.get('price', 0.0)) if isinstance(x, dict) else 0.0
                )
                df['slippage_pct'] = df.apply(
                    lambda row: abs(row['price'] - row['expected_price']) / row['expected_price'] * 100 
                    if row['expected_price'] > 0 else 0.0,
                    axis=1
                )
                df['price_improvement'] = df.apply(
                    lambda row: (row['expected_price'] - row['price']) / row['expected_price'] 
                    if row['direction'] == 'BUY' and row['expected_price'] > 0 
                    else (row['price'] - row['expected_price']) / row['expected_price']
                    if row['direction'] == 'SELL' and row['expected_price'] > 0 
                    else 0.0,
                    axis=1
                )
            else:
                # No metadata column - use available data
                df['expected_price'] = df['price']
                df['slippage_pct'] = 0.0
                df['price_improvement'] = 0.0
            
            return df
        except Exception as e:
            console.print(f"[red]Error fetching order data:[/red] {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def analyze_fills(self, df: pd.DataFrame) -> ExecutionMetrics:
        """Analyze order fill statistics."""
        if df.empty:
            return ExecutionMetrics()
        
        metrics = ExecutionMetrics()
        metrics.total_orders = len(df)
        
        # Identify fills vs failed orders
        # Filled orders have cost_usd > 0 (entries) or cost_usd = 0 with pnl != 0 (exits)
        entries = df[df['cost_usd'] > 1e-9]
        exits = df[(df['cost_usd'] <= 1e-9) & (df['pnl'] != 0)]
        
        metrics.filled_orders = len(entries) + len(exits)
        metrics.fill_rate = metrics.filled_orders / metrics.total_orders * 100 if metrics.total_orders > 0 else 0.0
        
        # Slippage analysis
        if 'slippage_pct' in df.columns:
            slippage_values = df['slippage_pct'].abs()
            metrics.avg_slippage_pct = slippage_values.mean()
            metrics.max_slippage_pct = slippage_values.max()
            metrics.slippage_std = slippage_values.std() if len(slippage_values) > 1 else 0.0
            
            # Store slippage history
            self.slippage_history = df[['symbol', 'timestamp', 'slippage_pct', 'direction']].to_dict('records')
        
        # Price improvement analysis
        if 'price_improvement' in df.columns:
            metrics.price_improvement_count = len(df[df['price_improvement'] > 0])
            metrics.price_worsening_count = len(df[df['price_improvement'] < 0])
        
        # High impact trades (slippage > 1%)
        metrics.high_impact_trades = len(df[df['slippage_pct'] > 1.0]) if 'slippage_pct' in df.columns else 0
        
        self.metrics = metrics
        return metrics
    
    def analyze_liquidity(self, df: pd.DataFrame) -> Dict[str, int]:
        """Analyze liquidity-related rejections."""
        liquidity_stats = {
            'liquidity_rejections': 0,
            'high_impact_warnings': 0,
            'partial_fills': 0
        }
        
        if 'meta_dict' in df.columns:
            for meta in df['meta_dict']:
                if isinstance(meta, dict):
                    if meta.get('liquidity_rejected', False):
                        liquidity_stats['liquidity_rejections'] += 1
                    if meta.get('high_impact', False):
                        liquidity_stats['high_impact_warnings'] += 1
                    if meta.get('partial_fill', False):
                        liquidity_stats['partial_fills'] += 1
        
        self.metrics.liquidity_rejections = liquidity_stats['liquidity_rejections']
        self.metrics.partial_fills = liquidity_stats['partial_fills']
        
        return liquidity_stats
    
    def generate_report(self) -> str:
        """Generate execution quality report."""
        df = self.fetch_order_data()
        if df.empty:
            return "No order data available for analysis."
        
        self.analyze_fills(df)
        liquidity_stats = self.analyze_liquidity(df)
        m = self.metrics
        
        # Create summary panel
        summary = Text()
        summary.append(f"Total Orders: ", style="bold white")
        summary.append(f"{m.total_orders}", style="bold cyan")
        summary.append(f" | Fill Rate: ", style="bold white")
        fill_color = "green" if m.fill_rate >= 95 else "yellow" if m.fill_rate >= 80 else "red"
        summary.append(f"{m.fill_rate:.1f}%", style=f"bold {fill_color}")
        summary.append(f" | Avg Slippage: ", style="bold white")
        slip_color = "green" if m.avg_slippage_pct < 0.1 else "yellow" if m.avg_slippage_pct < 0.5 else "red"
        summary.append(f"{m.avg_slippage_pct:.3f}%", style=f"bold {slip_color}")
        
        console.print(Panel(
            summary,
            title="[bold gold1]EXECUTION QUALITY SUMMARY[/bold gold1]",
            border_style="bright_blue",
            box=box.DOUBLE
        ))
        
        # Detailed metrics table
        table = Table(title="Execution Metrics", box=box.ROUNDED, header_style="bold magenta")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        table.add_column("Status", justify="center")
        
        def status_icon(value: float, good: float, bad: float, higher_better: bool = True) -> str:
            if higher_better:
                return "✅" if value >= good else "⚠️" if value >= bad else "❌"
            else:
                return "✅" if value <= good else "⚠️" if value <= bad else "❌"
        
        table.add_row(
            "Fill Rate", f"{m.fill_rate:.1f}%",
            status_icon(m.fill_rate, 95, 80)
        )
        table.add_row(
            "Avg Slippage", f"{m.avg_slippage_pct:.3f}%",
            status_icon(m.avg_slippage_pct, 0.1, 0.5, higher_better=False)
        )
        table.add_row(
            "Max Slippage", f"{m.max_slippage_pct:.3f}%",
            status_icon(m.max_slippage_pct, 0.5, 1.0, higher_better=False)
        )
        table.add_row(
            "Slippage Std Dev", f"{m.slippage_std:.3f}%",
            status_icon(m.slippage_std, 0.2, 0.5, higher_better=False)
        )
        table.add_row(
            "Price Improvements", f"{m.price_improvement_count}",
            "✅" if m.price_improvement_count > m.price_worsening_count else "⚠️"
        )
        table.add_row(
            "Price Worsening", f"{m.price_worsening_count}",
            "✅" if m.price_worsening_count < m.price_improvement_count else "⚠️"
        )
        table.add_row(
            "High Impact Trades", f"{m.high_impact_trades}",
            status_icon(m.high_impact_trades, 0, 5, higher_better=False)
        )
        table.add_row(
            "Liquidity Rejections", f"{m.liquidity_rejections}",
            status_icon(m.liquidity_rejections, 0, 3, higher_better=False)
        )
        
        console.print(table)
        
        # Slippage distribution
        if self.slippage_history:
            slippage_df = pd.DataFrame(self.slippage_history)
            if not slippage_df.empty:
                dist_table = Table(title="Slippage Distribution", box=box.SIMPLE)
                dist_table.add_column("Range", style="cyan")
                dist_table.add_column("Count", justify="right")
                dist_table.add_column("Percentage", justify="right")
                
                total = len(slippage_df)
                ranges = [
                    ("0.00% - 0.10%", 0.0, 0.1),
                    ("0.10% - 0.25%", 0.1, 0.25),
                    ("0.25% - 0.50%", 0.25, 0.5),
                    ("0.50% - 1.00%", 0.5, 1.0),
                    ("> 1.00%", 1.0, float('inf'))
                ]
                
                for label, low, high in ranges:
                    count = len(slippage_df[(slippage_df['slippage_pct'] >= low) & 
                                           (slippage_df['slippage_pct'] < high)])
                    pct = count / total * 100 if total > 0 else 0
                    dist_table.add_row(label, f"{count}", f"{pct:.1f}%")
                
                console.print(dist_table)
        
        return "Execution Quality Analysis Complete"


# ============================================================================
# POSITION MANAGEMENT ANALYZER
# ============================================================================

class PositionManagementAnalyzer:
    """
    Analyzes position management and exit logic effectiveness.
    
    Key Metrics:
    - TP/SL Hit Rates: Percentage of exits at target levels
    - Exit Efficiency: How close actual exits were to optimal
    - MFE/MAE Analysis: Maximum Favorable/Adverse Excursion
    - Holding Period Analysis: Time-based performance
    """
    
    def __init__(self, db_path: str = "holonic_trader.db"):
        self.db_path = db_path
        self.metrics = ExitEffectivenessMetrics()
        self.exit_history: List[Dict] = []
        self.mfe_mae_data: List[Dict] = []
    
    def fetch_exit_data(self) -> pd.DataFrame:
        """Fetch exit data from database."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Fetch exits (rows with cost_usd = 0 and pnl != 0)
            # Note: Using 'quantity' column (not 'size') per actual schema
            query = """
                SELECT 
                    id, symbol, timestamp, direction, price,
                    quantity, pnl, cost_usd, pnl_percent,
                    mfe, mae
                FROM trades 
                WHERE cost_usd <= 1e-9 AND pnl != 0
                ORDER BY id ASC
            """
            df = pd.read_sql_query(query, conn)
            
            # Rename quantity to size for internal consistency
            if 'quantity' in df.columns:
                df['size'] = df['quantity']
            
            if not df.empty:
                # Parse metadata if column exists
                if 'metadata' in df.columns:
                    df['meta_dict'] = df['metadata'].apply(
                        lambda x: json.loads(x) if isinstance(x, str) else {}
                    )
                    
                    # Extract exit reason
                    df['exit_reason'] = df['meta_dict'].apply(
                        lambda x: x.get('reason', 'UNKNOWN') if isinstance(x, dict) else 'UNKNOWN'
                    )
                    
                    # Extract entry price for comparison
                    df['entry_price'] = df['meta_dict'].apply(
                        lambda x: x.get('entry_price', 0.0) if isinstance(x, dict) else 0.0
                    )
                    
                    # Extract TP/SL levels
                    df['take_profit'] = df['meta_dict'].apply(
                        lambda x: x.get('take_profit', 0.0) if isinstance(x, dict) else 0.0
                    )
                    df['stop_loss'] = df['meta_dict'].apply(
                        lambda x: x.get('stop_loss', 0.0) if isinstance(x, dict) else 0.0
                    )
                else:
                    # No metadata - use defaults
                    df['exit_reason'] = 'UNKNOWN'
                    df['entry_price'] = 0.0
                    df['take_profit'] = 0.0
                    df['stop_loss'] = 0.0
            
            return df
        except Exception as e:
            console.print(f"[red]Error fetching exit data:[/red] {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def analyze_exits(self, df: pd.DataFrame) -> ExitEffectivenessMetrics:
        """Analyze exit effectiveness."""
        if df.empty:
            return ExitEffectivenessMetrics()
        
        metrics = ExitEffectivenessMetrics()
        metrics.total_exits = len(df)
        
        # Categorize exits by reason
        if 'exit_reason' in df.columns:
            metrics.tp_hits = len(df[df['exit_reason'] == 'TAKE_PROFIT'])
            metrics.sl_hits = len(df[df['exit_reason'] == 'STOP_LOSS'])
            metrics.thesis_exits = len(df[df['exit_reason'] == 'Thesis'])
            metrics.manual_exits = len(df[df['exit_reason'].isin(['MANUAL', 'Strat'])])
            
            metrics.tp_hit_rate = metrics.tp_hits / metrics.total_exits * 100
            metrics.sl_hit_rate = metrics.sl_hits / metrics.total_exits * 100
        
        # MFE/MAE Analysis
        if 'mfe' in df.columns and 'mae' in df.columns:
            # MFE Realization Rate: How much of max profit was captured
            profitable = df[df['pnl'] > 0]
            if not profitable.empty and 'mfe' in profitable.columns:
                # For longs: MFE = (max_price - entry) / entry
                # Realized = (exit - entry) / entry
                # Capture rate = Realized / MFE
                mfe_positive = profitable[profitable['mfe'] > 0]
                if not mfe_positive.empty:
                    # Approximate capture rate
                    avg_mfe_pct = mfe_positive['mfe'].mean() * 100
                    avg_pnl_pct = mfe_positive['pnl'].mean()  # Already in USD
                    # Normalize by average position size (approximate)
                    avg_size = mfe_positive['size'].mean() if 'size' in mfe_positive.columns else 1.0
                    avg_entry = mfe_positive['entry_price'].mean() if 'entry_price' in mfe_positive.columns else 1.0
                    avg_position_value = avg_size * avg_entry
                    avg_pnl_pct_normalized = (avg_pnl_pct / avg_position_value * 100) if avg_position_value > 0 else 0
                    
                    metrics.mfe_realization_rate = min(100, avg_pnl_pct_normalized / avg_mfe_pct * 100) if avg_mfe_pct > 0 else 100.0
            
            # Identify premature exits (negative PnL but MFE was positive and large)
            losing_trades = df[df['pnl'] < 0]
            if not losing_trades.empty and 'mfe' in losing_trades.columns:
                # Trades that went positive but closed negative
                metrics.premature_exits = len(losing_trades[losing_trades['mfe'] > 0.01])  # >1% MFE
            
            # Identify late exits (gave back >50% of MFE)
            winning_trades = df[df['pnl'] > 0]
            if not winning_trades.empty:
                # Where PnL < 50% of MFE
                late_mask = (winning_trades['pnl'] < winning_trades['mfe'] * 0.5)
                metrics.late_exits = len(winning_trades[late_mask])
        
        # Holding period analysis
        if 'timestamp' in df.columns and len(df) > 1:
            # Calculate time between entries and exits (approximate)
            timestamps = pd.to_datetime(df['timestamp'])
            holding_periods = timestamps.diff().dropna()
            metrics.avg_holding_period_mins = holding_periods.mean().total_seconds() / 60 if len(holding_periods) > 0 else 0.0
        
        # Exit efficiency (actual vs optimal)
        if 'entry_price' in df.columns and 'take_profit' in df.columns:
            efficiency_scores = []
            for _, row in df.iterrows():
                if row['entry_price'] > 0 and row['take_profit'] > 0:
                    if row['direction'] == 'SELL':  # Long exit
                        optimal = row['take_profit']
                        actual = row['price']
                        # Efficiency = how close to TP
                        entry_to_tp = optimal - row['entry_price']
                        entry_to_actual = actual - row['entry_price']
                        if entry_to_tp > 0:
                            eff = min(1.0, entry_to_actual / entry_to_tp)
                            efficiency_scores.append(eff)
            
            if efficiency_scores:
                metrics.avg_exit_efficiency = np.mean(efficiency_scores) * 100
        
        self.metrics = metrics
        self.exit_history = df.to_dict('records')
        return metrics
    
    def generate_report(self) -> str:
        """Generate position management report."""
        df = self.fetch_exit_data()
        if df.empty:
            return "No exit data available for analysis."
        
        self.analyze_exits(df)
        m = self.metrics
        
        # Summary panel
        summary = Text()
        summary.append(f"Total Exits: ", style="bold white")
        summary.append(f"{m.total_exits}", style="bold cyan")
        summary.append(f" | TP Hit Rate: ", style="bold white")
        tp_color = "green" if m.tp_hit_rate >= 40 else "yellow" if m.tp_hit_rate >= 25 else "red"
        summary.append(f"{m.tp_hit_rate:.1f}%", style=f"bold {tp_color}")
        summary.append(f" | SL Hit Rate: ", style="bold white")
        sl_color = "green" if m.sl_hit_rate <= 30 else "yellow" if m.sl_hit_rate <= 45 else "red"
        summary.append(f"{m.sl_hit_rate:.1f}%", style=f"bold {sl_color}")
        summary.append(f" | Exit Efficiency: ", style="bold white")
        eff_color = "green" if m.avg_exit_efficiency >= 80 else "yellow" if m.avg_exit_efficiency >= 60 else "red"
        summary.append(f"{m.avg_exit_efficiency:.1f}%", style=f"bold {eff_color}")
        
        console.print(Panel(
            summary,
            title="[bold gold1]POSITION MANAGEMENT SUMMARY[/bold gold1]",
            border_style="bright_green",
            box=box.DOUBLE
        ))
        
        # Exit breakdown table
        table = Table(title="Exit Breakdown", box=box.ROUNDED, header_style="bold magenta")
        table.add_column("Exit Type", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Percentage", justify="right")
        table.add_column("Quality", justify="center")
        
        exits = [
            ("Take Profit", m.tp_hits, m.tp_hit_rate, "✅" if m.tp_hit_rate >= 40 else "⚠️"),
            ("Stop Loss", m.sl_hits, m.sl_hit_rate, "✅" if m.sl_hit_rate <= 30 else "⚠️"),
            ("Thesis Invalid", m.thesis_exits, m.thesis_exits / m.total_exits * 100 if m.total_exits > 0 else 0, "ℹ️"),
            ("Manual/Strat", m.manual_exits, m.manual_exits / m.total_exits * 100 if m.total_exits > 0 else 0, "ℹ️")
        ]
        
        for label, count, pct, quality in exits:
            table.add_row(label, f"{count}", f"{pct:.1f}%", quality)
        
        console.print(table)
        
        # Effectiveness metrics
        eff_table = Table(title="Exit Effectiveness Metrics", box=box.ROUNDED)
        eff_table.add_column("Metric", style="dim")
        eff_table.add_column("Value", justify="right")
        eff_table.add_column("Assessment", justify="center")
        
        eff_table.add_row(
            "Avg Exit Efficiency", f"{m.avg_exit_efficiency:.1f}%",
            "✅" if m.avg_exit_efficiency >= 80 else "⚠️" if m.avg_exit_efficiency >= 60 else "❌"
        )
        eff_table.add_row(
            "MFE Capture Rate", f"{m.mfe_realization_rate:.1f}%",
            "✅" if m.mfe_realization_rate >= 70 else "⚠️" if m.mfe_realization_rate >= 50 else "❌"
        )
        eff_table.add_row(
            "Premature Exits", f"{m.premature_exits}",
            "✅" if m.premature_exits == 0 else "⚠️" if m.premature_exits <= 3 else "❌"
        )
        eff_table.add_row(
            "Late Exits", f"{m.late_exits}",
            "✅" if m.late_exits == 0 else "⚠️" if m.late_exits <= 3 else "❌"
        )
        eff_table.add_row(
            "Avg Holding Period", f"{m.avg_holding_period_mins:.1f} mins",
            "ℹ️"
        )
        
        console.print(eff_table)
        
        return "Position Management Analysis Complete"


# ============================================================================
# PERFORMANCE METRICS ANALYZER
# ============================================================================

class PerformanceMetricsAnalyzer:
    """
    Analyzes comprehensive performance metrics.
    
    Key Metrics:
    - Risk-Adjusted Returns: Sharpe, Sortino, Calmar ratios
    - Drawdown Analysis: Max, average, frequency, duration
    - Return Distribution: Skewness, kurtosis, VaR, CVaR
    - Consistency Metrics: Win rate, profit factor, omega ratio
    """
    
    def __init__(self, db_path: str = "holonic_trader.db"):
        self.db_path = db_path
        self.metrics = PerformanceMetrics()
        self.equity_curve: List[float] = []
        self.returns: List[float] = []
        self.drawdowns: List[float] = []
    
    def fetch_performance_data(self) -> pd.DataFrame:
        """Fetch trade data for performance analysis."""
        conn = sqlite3.connect(self.db_path)
        try:
            query = """
                SELECT 
                    id, symbol, timestamp, direction, price,
                    quantity, pnl, cost_usd, pnl_percent,
                    mfe, mae, unrealized_pnl, unrealized_pnl_percent
                FROM trades 
                ORDER BY id ASC
            """
            df = pd.read_sql_query(query, conn)
            
            if not df.empty:
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Rename quantity to size for consistency
                if 'quantity' in df.columns:
                    df['size'] = df['quantity']
                
                # Identify exits (cost_usd = 0 with realized pnl) vs entries
                # In this schema, entries have cost_usd > 0 and may have unrealized_pnl
                df['is_exit'] = (df['cost_usd'] <= 1e-9) & (df['pnl'] != 0)
                df['is_entry'] = df['cost_usd'] > 1e-9
                
                # For entries, use unrealized_pnl if pnl is 0
                df['effective_pnl'] = df.apply(
                    lambda row: row['unrealized_pnl'] if row['is_entry'] and 'unrealized_pnl' in row and row['pnl'] == 0 else row['pnl'],
                    axis=1
                )
                
                # Calculate returns (PnL as percentage of position value)
                df['position_value'] = df['price'] * df['size']
                df['return_pct'] = df.apply(
                    lambda row: row['effective_pnl'] / row['position_value'] * 100 
                    if row['position_value'] > 0 else 0.0,
                    axis=1
                )
            
            return df
        except Exception as e:
            console.print(f"[red]Error fetching performance data:[/red] {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def calculate_risk_metrics(self, df: pd.DataFrame) -> PerformanceMetrics:
        """Calculate comprehensive risk and performance metrics."""
        if df.empty:
            return PerformanceMetrics()
        
        metrics = PerformanceMetrics()
        
        # Use all trades for metrics (including unrealized)
        # This gives a "current state" view rather than just closed trades
        metrics.total_trades = len(df)
        metrics.total_pnl = df['effective_pnl'].sum() if 'effective_pnl' in df.columns else df['pnl'].sum()
        metrics.avg_pnl = df['effective_pnl'].mean() if 'effective_pnl' in df.columns else df['pnl'].mean()
        
        # Win/Loss analysis
        pnl_col = 'effective_pnl' if 'effective_pnl' in df.columns else 'pnl'
        winning = df[df[pnl_col] > 0]
        losing = df[df[pnl_col] < 0]
        
        metrics.win_rate = len(winning) / metrics.total_trades * 100 if metrics.total_trades > 0 else 0.0
        
        gross_profit = winning[pnl_col].sum() if not winning.empty else 0
        gross_loss = abs(losing[pnl_col].sum()) if not losing.empty else 0
        metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else (100.0 if gross_profit > 0 else 0.0)
        
        # Returns series for risk calculations
        self.returns = df['return_pct'].tolist() if 'return_pct' in df.columns else []
        
        if len(self.returns) > 1:
            returns_arr = np.array(self.returns)
            
            # Sharpe Ratio (annualized, assuming 252 trading days)
            mean_return = np.mean(returns_arr)
            std_return = np.std(returns_arr, ddof=1)
            metrics.sharpe_ratio = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0
            
            # Sortino Ratio (downside deviation)
            downside_returns = returns_arr[returns_arr < 0]
            downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0.0
            metrics.sortino_ratio = (mean_return / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0
            
            # Omega Ratio
            metrics.omega_ratio = self._calculate_omega_ratio(returns_arr)
            
            # Return distribution
            metrics.skewness = float(pd.Series(returns_arr).skew())
            metrics.kurtosis = float(pd.Series(returns_arr).kurtosis())
            
            # VaR and CVaR (95%)
            metrics.var_95 = np.percentile(returns_arr, 5)
            metrics.cvar_95 = returns_arr[returns_arr <= metrics.var_95].mean() if len(returns_arr[returns_arr <= metrics.var_95]) > 0 else metrics.var_95
            
            # Tail Ratio (95th percentile / 5th percentile)
            p95 = np.percentile(returns_arr, 95)
            p5 = np.percentile(returns_arr, 5)
            metrics.tail_ratio = abs(p95 / p5) if p5 != 0 else 0.0
        
        # Drawdown analysis (requires equity curve)
        self._calculate_drawdowns(df, metrics)
        
        # Calmar Ratio (annual return / max drawdown)
        if metrics.max_drawdown_pct > 0:
            # Approximate annualized return
            if len(df) > 0:
                total_return = metrics.total_pnl
                # Assume average holding period and extrapolate
                metrics.calmar_ratio = (total_return / metrics.max_drawdown_pct) if metrics.max_drawdown_pct > 0 else 0.0
        
        self.metrics = metrics
        return metrics
    
    def _calculate_omega_ratio(self, returns: np.ndarray, threshold: float = 0.0) -> float:
        """Calculate Omega Ratio."""
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]
        
        sum_gains = np.sum(gains)
        sum_losses = np.sum(losses)
        
        if sum_losses == 0:
            return 100.0 if sum_gains > 0 else 0.0
        
        return sum_gains / sum_losses
    
    def _calculate_drawdowns(self, df: pd.DataFrame, metrics: PerformanceMetrics):
        """Calculate drawdown metrics from equity curve."""
        # Build equity curve from cumulative PnL
        pnl_series = df['pnl'].cumsum()
        
        if pnl_series.empty:
            return
        
        # Add initial capital (assume 100 for percentage calculations)
        initial_capital = 100.0
        equity_curve = initial_capital + pnl_series
        
        self.equity_curve = equity_curve.tolist()
        
        # Calculate drawdowns
        running_max = equity_curve.expanding().max()
        drawdown_series = (equity_curve - running_max) / running_max * 100
        
        self.drawdowns = drawdown_series.tolist()
        
        # Max drawdown
        metrics.max_drawdown_pct = abs(drawdown_series.min())
        metrics.max_drawdown_usd = abs(pnl_series.min()) if pnl_series.min() < 0 else 0.0
        
        # Current drawdown
        metrics.current_drawdown_pct = abs(drawdown_series.iloc[-1]) if drawdown_series.iloc[-1] < 0 else 0.0
        
        # Average drawdown (of significant DDs > 1%)
        significant_dd = drawdown_series[drawdown_series < -1.0]
        metrics.avg_drawdown_pct = abs(significant_dd.mean()) if len(significant_dd) > 0 else 0.0
        
        # Drawdown frequency (number of distinct drawdown periods)
        metrics.drawdown_frequency = len(significant_dd)
        
        # Max drawdown duration (approximate)
        in_drawdown = drawdown_series < -1.0
        dd_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    dd_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            dd_periods.append(current_period)
        
        metrics.max_drawdown_duration_days = max(dd_periods) if dd_periods else 0
    
    def generate_report(self) -> str:
        """Generate performance metrics report."""
        df = self.fetch_performance_data()
        if df.empty:
            return "No performance data available for analysis."
        
        self.calculate_risk_metrics(df)
        m = self.metrics
        
        # Summary panel
        summary = Text()
        summary.append(f"Total PnL: ", style="bold white")
        pnl_color = "green" if m.total_pnl >= 0 else "red"
        summary.append(f"${m.total_pnl:+,.2f}", style=f"bold {pnl_color}")
        summary.append(f" | Win Rate: ", style="bold white")
        summary.append(f"{m.win_rate:.1f}%", style="bold cyan")
        summary.append(f" | Sharpe: ", style="bold white")
        sharpe_color = "green" if m.sharpe_ratio >= 1.5 else "yellow" if m.sharpe_ratio >= 0.5 else "red"
        summary.append(f"{m.sharpe_ratio:.2f}", style=f"bold {sharpe_color}")
        summary.append(f" | Max DD: ", style="bold white")
        dd_color = "green" if m.max_drawdown_pct <= 10 else "yellow" if m.max_drawdown_pct <= 20 else "red"
        summary.append(f"{m.max_drawdown_pct:.1f}%", style=f"bold {dd_color}")
        
        console.print(Panel(
            summary,
            title="[bold gold1]PERFORMANCE METRICS SUMMARY[/bold gold1]",
            border_style="bright_magenta",
            box=box.DOUBLE
        ))
        
        # Risk-adjusted returns table
        risk_table = Table(title="Risk-Adjusted Returns", box=box.ROUNDED, header_style="bold magenta")
        risk_table.add_column("Metric", style="dim")
        risk_table.add_column("Value", justify="right")
        risk_table.add_column("Assessment", justify="center")
        
        risk_table.add_row(
            "Sharpe Ratio", f"{m.sharpe_ratio:.2f}",
            "✅" if m.sharpe_ratio >= 1.5 else "⚠️" if m.sharpe_ratio >= 0.5 else "❌"
        )
        risk_table.add_row(
            "Sortino Ratio", f"{m.sortino_ratio:.2f}",
            "✅" if m.sortino_ratio >= 2.0 else "⚠️" if m.sortino_ratio >= 1.0 else "❌"
        )
        risk_table.add_row(
            "Calmar Ratio", f"{m.calmar_ratio:.2f}",
            "✅" if m.calmar_ratio >= 3.0 else "⚠️" if m.calmar_ratio >= 1.0 else "❌"
        )
        risk_table.add_row(
            "Omega Ratio", f"{m.omega_ratio:.2f}",
            "✅" if m.omega_ratio >= 1.5 else "⚠️" if m.omega_ratio >= 1.0 else "❌"
        )
        risk_table.add_row(
            "Tail Ratio", f"{m.tail_ratio:.2f}",
            "✅" if m.tail_ratio >= 2.0 else "⚠️" if m.tail_ratio >= 1.0 else "❌"
        )
        
        console.print(risk_table)
        
        # Drawdown analysis table
        dd_table = Table(title="Drawdown Analysis", box=box.ROUNDED)
        dd_table.add_column("Metric", style="dim")
        dd_table.add_column("Value", justify="right")
        dd_table.add_column("Risk Level", justify="center")
        
        dd_table.add_row(
            "Max Drawdown", f"{m.max_drawdown_pct:.1f}% (${m.max_drawdown_usd:.2f})",
            "✅" if m.max_drawdown_pct <= 10 else "⚠️" if m.max_drawdown_pct <= 20 else "❌"
        )
        dd_table.add_row(
            "Current Drawdown", f"{m.current_drawdown_pct:.1f}%",
            "✅" if m.current_drawdown_pct <= 5 else "⚠️" if m.current_drawdown_pct <= 10 else "❌"
        )
        dd_table.add_row(
            "Avg Drawdown", f"{m.avg_drawdown_pct:.1f}%",
            "ℹ️"
        )
        dd_table.add_row(
            "DD Frequency", f"{m.drawdown_frequency} events",
            "ℹ️"
        )
        dd_table.add_row(
            "Max DD Duration", f"{m.max_drawdown_duration_days} trades",
            "ℹ️"
        )
        
        console.print(dd_table)
        
        # Return distribution table
        dist_table = Table(title="Return Distribution", box=box.SIMPLE)
        dist_table.add_column("Metric", style="dim")
        dist_table.add_column("Value", justify="right")
        dist_table.add_column("Interpretation", justify="left")
        
        dist_table.add_row("Skewness", f"{m.skewness:.2f}", 
                          "Positive" if m.skewness > 0.5 else "Negative" if m.skewness < -0.5 else "Neutral")
        dist_table.add_row("Kurtosis", f"{m.kurtosis:.2f}",
                          "Fat tails" if m.kurtosis > 0 else "Normal" if m.kurtosis > -1 else "Thin tails")
        dist_table.add_row("VaR (95%)", f"{m.var_95:.2f}%",
                          "Worst 5% of returns below this")
        dist_table.add_row("CVaR (95%)", f"{m.cvar_95:.2f}%",
                          "Average of worst 5% of returns")
        
        console.print(dist_table)
        
        return "Performance Metrics Analysis Complete"


# ============================================================================
# UNIFIED DASHBOARD
# ============================================================================

class TradingAnalysisDashboard:
    """
    Unified dashboard combining all three analysis modules.
    
    Provides comprehensive analysis of:
    1. Execution Quality
    2. Position Management
    3. Performance Metrics
    """
    
    def __init__(self, db_path: str = "holonic_trader.db"):
        self.db_path = db_path
        self.execution_analyzer = ExecutionQualityAnalyzer(db_path)
        self.position_analyzer = PositionManagementAnalyzer(db_path)
        self.performance_analyzer = PerformanceMetricsAnalyzer(db_path)
    
    def generate_report(self):
        """Generate comprehensive trading analysis report."""
        console.print("\n")
        console.print(Panel(
            "[bold]HOLONIC TRADER - COMPREHENSIVE ANALYSIS REPORT[/bold]\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            title="[bold gold1]🤖 AEHML TRADING ANALYSIS[/bold gold1]",
            border_style="bright_blue",
            box=box.DOUBLE
        ))
        console.print("\n")
        
        # Section 1: Execution Quality
        console.print(Panel(
            "[bold]Analyzing order execution quality, fills, and slippage...[/bold]",
            border_style="cyan"
        ))
        self.execution_analyzer.generate_report()
        console.print("\n")
        
        # Section 2: Position Management
        console.print(Panel(
            "[bold]Analyzing position management and exit effectiveness...[/bold]",
            border_style="green"
        ))
        self.position_analyzer.generate_report()
        console.print("\n")
        
        # Section 3: Performance Metrics
        console.print(Panel(
            "[bold]Analyzing performance metrics, Sharpe ratio, and drawdowns...[/bold]",
            border_style="magenta"
        ))
        self.performance_analyzer.generate_report()
        console.print("\n")
        
        # Summary and Recommendations
        self._generate_recommendations()
    
    def _generate_recommendations(self):
        """Generate actionable recommendations based on analysis."""
        console.print(Panel(
            "[bold]ACTIONABLE RECOMMENDATIONS[/bold]",
            title="[bold gold1]TRADING IMPROVEMENT INSIGHTS[/bold gold1]",
            border_style="bright_yellow",
            box=box.DOUBLE
        ))
        
        recommendations = []
        
        # Check data availability
        exec_metrics = self.execution_analyzer.metrics
        pos_metrics = self.position_analyzer.metrics
        perf_metrics = self.performance_analyzer.metrics
        
        # Determine if we have realized trades or just unrealized
        has_realized_trades = pos_metrics.total_exits > 0
        total_trades = perf_metrics.total_trades
        
        if not has_realized_trades:
            # Session in progress - no exits yet
            console.print("[bold cyan]Session Status:[/bold cyan] All positions currently open (unrealized PnL)\n")
            
            # Only show execution-related recommendations
            if exec_metrics.fill_rate < 90 and total_trades > 10:
                recommendations.append(
                    "[yellow]EXECUTION[/yellow]: Fill rate below 90%. Consider using more aggressive order types "
                    "or improving liquidity checks."
                )
            if exec_metrics.avg_slippage_pct > 0.5 and total_trades > 10:
                recommendations.append(
                    "[yellow]EXECUTION[/yellow]: High average slippage (>0.5%). Review order sizing relative to "
                    "market depth or implement better timing algorithms."
                )
            if exec_metrics.liquidity_rejections > 0:
                recommendations.append(
                    f"[yellow]EXECUTION[/yellow]: {exec_metrics.liquidity_rejections} orders rejected due to liquidity. "
                    "Consider reducing position sizes for illiquid assets."
                )
            
            # Show unrealized performance snapshot
            if perf_metrics.total_pnl != 0:
                pnl_status = "positive" if perf_metrics.total_pnl > 0 else "negative"
                console.print(f"[bold]Current Unrealized PnL:[/bold] ${perf_metrics.total_pnl:+.2f} ({pnl_status})")
                console.print(f"[bold]Open Positions:[/bold] {total_trades}\n")
            
            if not recommendations:
                console.print("[bold green]Execution operating normally. Monitor positions for exit analysis.[/bold green]")
        else:
            # Have realized trades - show full recommendations
            
            # Execution recommendations
            if exec_metrics.fill_rate < 90:
                recommendations.append(
                    "[yellow]EXECUTION[/yellow]: Fill rate below 90%. Consider using more aggressive order types "
                    "or improving liquidity checks."
                )
            if exec_metrics.avg_slippage_pct > 0.5:
                recommendations.append(
                    "[yellow]EXECUTION[/yellow]: High average slippage (>0.5%). Review order sizing relative to "
                    "market depth or implement better timing algorithms."
                )
            if exec_metrics.liquidity_rejections > 0:
                recommendations.append(
                    f"[yellow]EXECUTION[/yellow]: {exec_metrics.liquidity_rejections} orders rejected due to liquidity. "
                    "Consider reducing position sizes for illiquid assets."
                )
            
            # Position management recommendations
            if pos_metrics.tp_hit_rate < 30:
                recommendations.append(
                    "[green]POSITION MGMT[/green]: Low TP hit rate (<30%). Take-profit levels may be too ambitious. "
                    "Consider more realistic targets or trailing stops."
                )
            if pos_metrics.sl_hit_rate > 40:
                recommendations.append(
                    "[green]POSITION MGMT[/green]: High SL hit rate (>40%). Stop-loss levels may be too tight. "
                    "Consider wider stops or better entry timing."
                )
            if pos_metrics.premature_exits > 5:
                recommendations.append(
                    f"[green]POSITION MGMT[/green]: {pos_metrics.premature_exits} premature exits detected. "
                    "Review exit logic to avoid closing positions before thesis plays out."
                )
            if pos_metrics.late_exits > 5:
                recommendations.append(
                    f"[green]POSITION MGMT[/green]: {pos_metrics.late_exits} late exits detected (gave back >50% profits). "
                    "Implement tighter trailing stops or profit protection mechanisms."
                )
            
            # Performance recommendations (only with sufficient data)
            if perf_metrics.sharpe_ratio < 0.5 and pos_metrics.total_exits > 10:
                recommendations.append(
                    "[magenta]PERFORMANCE[/magenta]: Low Sharpe ratio (<0.5). Risk-adjusted returns need improvement. "
                    "Consider reducing position sizes or improving entry/exit timing."
                )
            if perf_metrics.max_drawdown_pct > 20:
                recommendations.append(
                    f"[magenta]PERFORMANCE[/magenta]: Maximum drawdown ({perf_metrics.max_drawdown_pct:.1f}%) exceeds 20%. "
                    "Implement stricter risk limits or reduce leverage."
                )
            if perf_metrics.win_rate < 40 and pos_metrics.total_exits > 10:
                recommendations.append(
                    "[magenta]PERFORMANCE[/magenta]: Low win rate (<40%). Focus on higher-quality setups or "
                    "improve signal filtering."
                )
            if perf_metrics.profit_factor < 1.5 and pos_metrics.total_exits > 10:
                recommendations.append(
                    "[magenta]PERFORMANCE[/magenta]: Profit factor below 1.5. Work on improving risk/reward ratios "
                    "or reducing loss magnitude."
                )
        
        # Display recommendations
        if recommendations:
            console.print("\n")
            for rec in recommendations:
                console.print(f"• {rec}")
        else:
            if has_realized_trades:
                console.print("[bold green]All metrics within acceptable ranges. System performing well![/bold green]")
        
        console.print("\n")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else "holonic_trader.db"
    
    # Check for --positions flag
    if len(sys.argv) > 2 and sys.argv[2] == '--positions':
        console.print(f"\n[bold cyan]Loading positions from:[/bold cyan] {db_path}\n")
        list_positions(db_path, show_summary=True)
    else:
        console.print(f"\n[bold cyan]Loading database:[/bold cyan] {db_path}\n")
        dashboard = TradingAnalysisDashboard(db_path)
        dashboard.generate_report()


# ============================================================================
# POSITION LISTING UTILITY
# ============================================================================

def list_positions(db_path: str = "holonic_trader.db", show_summary: bool = True):
    """
    List all open positions with PnL analysis.
    
    Args:
        db_path: Path to database
        show_summary: Whether to show summary statistics
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
        
        # Group by symbol for summary
        symbol_groups = df.groupby('symbol')
        
        if show_summary:
            console.print(Panel(
                f"[bold]OPEN POSITIONS SUMMARY[/bold]\n"
                f"Total Positions: {len(df)} | Symbols: {len(symbol_groups)}",
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
        summary_table.add_column("Unr PnL", justify="right")
        summary_table.add_column("Unr %", justify="right")
        summary_table.add_column("Direction", justify="center")
        
        total_margin = 0
        total_unrealized = 0
        
        for symbol, group in symbol_groups:
            count = len(group)
            margin = group['cost_usd'].sum()
            avg_entry = (group['price'] * group['quantity']).sum() / group['quantity'].sum()
            unr_pnl = group['unrealized_pnl'].sum()
            unr_pct = group['unrealized_pnl_percent'].mean()
            direction = group['direction'].iloc[0]
            
            total_margin += margin
            total_unrealized += unr_pnl
            
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
        
        console.print(summary_table)
        
        # Overall summary
        console.print("\n")
        overall_pnl_style = "green" if total_unrealized >= 0 else "red"
        
        summary_text = Text()
        summary_text.append("Total Margin: ", style="bold white")
        summary_text.append(f"${total_margin:.2f}", style="bold cyan")
        summary_text.append("  |  Total Unrealized PnL: ", style="bold white")
        summary_text.append(f"${total_unrealized:+.2f}", style=f"bold {overall_pnl_style}")
        summary_text.append("  |  Return: ", style="bold white")
        total_return_pct = (total_unrealized / total_margin * 100) if total_margin > 0 else 0
        summary_text.append(f"{total_return_pct:+.2f}%", style=f"bold {overall_pnl_style}")
        
        console.print(Panel(summary_text, border_style="bright_green"))
        console.print("\n")
        
        # Detailed view (optional - show first 20)
        if len(df) <= 20:
            detail_table = Table(title="All Positions", box=box.SIMPLE)
            detail_table.add_column("Symbol", style="cyan")
            detail_table.add_column("Dir", justify="center")
            detail_table.add_column("Quantity", justify="right")
            detail_table.add_column("Entry", justify="right")
            detail_table.add_column("Margin", justify="right")
            detail_table.add_column("Unr PnL", justify="right")
            detail_table.add_column("Unr %", justify="right")
            detail_table.add_column("MFE", justify="right")
            detail_table.add_column("MAE", justify="right")
            
            for _, row in df.iterrows():
                pnl_style = "green" if row['unrealized_pnl'] >= 0 else "red"
                detail_table.add_row(
                    row['symbol'],
                    "[green]BUY[/green]" if row['direction'] == 'BUY' else "[red]SELL[/red]",
                    f"{row['quantity']:.4f}",
                    f"${row['price']:.4f}",
                    f"${row['cost_usd']:.2f}",
                    f"[{pnl_style}]${row['unrealized_pnl']:+.2f}[/{pnl_style}]",
                    f"[{pnl_style}]{row['unrealized_pnl_percent']:+.2f}%[/{pnl_style}]",
                    f"{row['mfe']:+.2f}%",
                    f"{row['mae']:+.2f}%"
                )
            
            console.print(detail_table)
        else:
            console.print(f"[dim]Showing summary only ({len(df)} positions total). Use --detail flag for full list.[/dim]\n")
        
    except Exception as e:
        console.print(f"[red]Error listing positions:[/red] {e}")
    finally:
        conn.close()
