#!/usr/bin/env python3
"""
Historical Backtest Engine - Test Fixes Against Past Data

Replays historical trades from logs/database and calculates:
1. What would have happened with ORIGINAL gates
2. What would have happened with FIXED gates (unified + dynamic conviction)
3. Expectancy improvement from fixes

Usage:
    python HolonicTrader\HolonicTrader\backtest_historical_fixes.py
"""

import sqlite3
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

# Add parent to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unified_gates import UnifiedGateSystem, GateScore


@dataclass
class HistoricalTrade:
    """Historical trade record"""
    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime]
    exit_price: Optional[float]
    pnl_usd: float
    pnl_percent: float
    exit_reason: str
    confluence_count: int = 1
    conviction: float = 0.5
    strategy: str = 'DIP'
    structure_zone: str = 'NEUTRAL'
    regime: str = 'TRANSITION'


class HistoricalBacktestEngine:
    """
    Backtest engine for historical trades
    
    Replays trades and applies both original and fixed gate logic
    """
    
    def __init__(self, db_path: str = 'holonic_trader.db'):
        self.db_path = db_path
        self.trades = []
        self.load_trades_from_db()
        self.load_trades_from_logs()
        
        # Gate system for fixed version
        self.gates = UnifiedGateSystem()
    
    def load_trades_from_db(self):
        """Load trades from database"""
        if not os.path.exists(self.db_path):
            print(f"Database not found: {self.db_path}")
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if trades table exists and has data
            cursor.execute("SELECT COUNT(*) FROM trades")
            count = cursor.fetchone()[0]
            
            if count > 0:
                cursor.execute("""
                    SELECT symbol, direction, entry_time, entry_price, 
                           exit_time, exit_price, pnl_usd, pnl_percent, exit_reason
                    FROM trades 
                    WHERE exit_time IS NOT NULL
                    ORDER BY exit_time DESC
                    LIMIT 100
                """)
                
                for row in cursor.fetchall():
                    trade = HistoricalTrade(
                        symbol=row[0],
                        direction=row[1],
                        entry_time=datetime.fromisoformat(row[2]) if row[2] else None,
                        entry_price=row[3],
                        exit_time=datetime.fromisoformat(row[4]) if row[4] else None,
                        exit_price=row[5],
                        pnl_usd=row[6],
                        pnl_percent=row[7],
                        exit_reason=row[8] if row[8] else 'UNKNOWN'
                    )
                    self.trades.append(trade)
                
                print(f"Loaded {len(self.trades)} trades from database")
            
            conn.close()
        except Exception as e:
            print(f"Error loading trades from DB: {e}")
    
    def load_trades_from_logs(self):
        """Load trades from log files"""
        log_dir = Path('HolonicTrader')
        log_files = list(log_dir.glob('live_trading_session_*.log'))
        
        if not log_files:
            print("No log files found")
            return
        
        # Parse trade logs
        trade_pattern = re.compile(
            r'Trade Logged: (\S+) Profit: ([-.\d]+)%.*?(?:\[(\$[-.\d]+)\])?'
        )
        
        trades_from_logs = set()
        
        for log_file in log_files[:10]:  # Last 10 log files
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        match = trade_pattern.search(line)
                        if match:
                            symbol = match.group(1)
                            pnl_pct = float(match.group(2))
                            pnl_usd = float(match.group(3).replace('$', '')) if match.group(3) else pnl_pct * 10  # Estimate
                            
                            # Create unique key to avoid duplicates
                            trade_key = f"{symbol}_{pnl_pct}_{log_file.name}"
                            if trade_key not in trades_from_logs:
                                trades_from_logs.add(trade_key)
                                
                                trade = HistoricalTrade(
                                    symbol=symbol,
                                    direction='BUY',  # Assume BUY for now
                                    entry_time=None,
                                    entry_price=0,
                                    exit_time=None,
                                    exit_price=0,
                                    pnl_usd=pnl_usd,
                                    pnl_percent=pnl_pct,
                                    exit_reason='LOG_EXTRACTED'
                                )
                                self.trades.append(trade)
            except Exception as e:
                print(f"Error reading {log_file}: {e}")
        
        print(f"Loaded additional trades from logs (total: {len(self.trades)})")
    
    def simulate_original_gates(self, trade: HistoricalTrade) -> Tuple[bool, str]:
        """
        Simulate ORIGINAL gate logic (before fixes)
        
        Returns: (passed, veto_reason)
        """
        # Original Gate 1: Confluence >= 2
        if trade.confluence_count < 2:
            return False, f"CONFLUENCE: {trade.confluence_count} < 2"
        
        # Original Gate 2: Static conviction floor
        symbol_floors = {
            'SHIB/USDT': 0.53,
            'DOT/USDT': 0.58,
            'WIF/USDT': 0.55,
            'LDO/USDT': 0.50,
            'ETH/USDT': 0.45,
            'TAO/USDT': 0.45,
            'BNB/USDT': 0.45,
        }
        static_floor = symbol_floors.get(trade.symbol, 0.50)
        
        if trade.conviction < static_floor:
            return False, f"CONVICTION: {trade.conviction:.2f} < {static_floor:.2f}"
        
        # Original Gate 3: Cost filter ($0.20 min)
        if trade.pnl_usd < 0.20:
            return False, f"COST: ${trade.pnl_usd:.2f} < $0.20"
        
        return True, "PASS"
    
    def simulate_fixed_gates_v2(self, trade: HistoricalTrade, asset_stats: Dict) -> Tuple[bool, str, List[GateScore]]:
        """
        Simulate FIXED gate logic V2 (with blacklist + win rate filter)

        Returns: (passed, reason, gate_scores)
        """
        # Check chronic loser blacklist FIRST (fast fail)
        if trade.symbol in self.gates.CHRONIC_LOSER_BLACKLIST:
            return False, f"BLACKLISTED: {trade.symbol} is chronic loser", []

        # Check win rate filter - MORE LENIENT
        if trade.symbol in asset_stats:
            stats = asset_stats[trade.symbol]
            if stats.get('total_trades', 0) >= 3:
                win_rate = stats.get('win_rate', 0.5)
                if win_rate < 0.25:  # Only block if <25% (was 35%)
                    return False, f"WIN_RATE: {trade.symbol} WR {win_rate:.0%} < 25%", []

        # Create mock signal object
        class MockSignal:
            def __init__(self, trade):
                self.symbol = trade.symbol
                self.direction = trade.direction
                self.conviction = trade.conviction
                self.metadata = {
                    'confirmation_score': trade.confluence_count,
                    'strategy': trade.strategy,
                    'recent_win_rate': 0.50,  # Assume neutral
                    'orion': {'path': 'NEUTRAL'},
                }

        signal = MockSignal(trade)
        structure = {'sls_zone': trade.structure_zone}

        # Run unified gates
        all_passed, scores = self.gates.check_all_gates(
            symbol=trade.symbol,
            signal=signal,
            structure=structure,
            regime=trade.regime,
            portfolio_state={'health': 'GOOD'},
        )

        if all_passed:
            avg_score = sum(s.score for s in scores) / len(scores)
            return True, f"PASS (avg: {avg_score:.1f})", scores
        else:
            failed_gates = [s.gate_name for s in scores if not s.passed]
            return False, f"FAIL: {', '.join(failed_gates)}", scores
    
    def run_backtest_v2(self, asset_stats: Dict = None) -> Dict:
        """
        Run full backtest comparing original vs fixed gates V2 (with blacklist + win rate)

        Returns: Performance comparison report
        """
        print("=" * 70)
        print("HISTORICAL BACKTEST V2 - WITH BLACKLIST + WIN RATE FILTER")
        print("=" * 70)
        print(f"\nTotal historical trades: {len(self.trades)}\n")

        if asset_stats:
            print(f"Asset stats loaded for {len(asset_stats)} symbols")
            print(f"Chronic loser blacklist: {list(self.gates.CHRONIC_LOSER_BLACKLIST.keys())}\n")

        # Track results
        original_approved = []
        fixed_v2_approved = []

        veto_comparison = defaultdict(lambda: {'original': 0, 'fixed_v2': 0})

        for trade in self.trades:
            # Simulate original gates
            orig_passed, orig_reason = self.simulate_original_gates(trade)

            # Simulate fixed gates V2
            fixed_passed, fixed_reason, _ = self.simulate_fixed_gates_v2(trade, asset_stats or {})

            if orig_passed:
                original_approved.append(trade)
            else:
                # Track veto reason
                if 'CONFLUENCE' in orig_reason:
                    veto_comparison['CONFLUENCE']['original'] += 1
                elif 'CONVICTION' in orig_reason:
                    veto_comparison['CONVICTION']['original'] += 1
                elif 'COST' in orig_reason:
                    veto_comparison['COST']['original'] += 1

            if fixed_passed:
                fixed_v2_approved.append(trade)
            else:
                # Track veto reason
                if 'BLACKLIST' in fixed_reason:
                    veto_comparison['BLACKLIST']['fixed_v2'] += 1
                elif 'WIN_RATE' in fixed_reason:
                    veto_comparison['WIN_RATE']['fixed_v2'] += 1
                elif 'QUALITY' in fixed_reason:
                    veto_comparison['QUALITY']['fixed_v2'] += 1
                elif 'ALIGNMENT' in fixed_reason:
                    veto_comparison['ALIGNMENT']['fixed_v2'] += 1
                elif 'RISK' in fixed_reason:
                    veto_comparison['RISK']['fixed_v2'] += 1

        # Calculate performance metrics
        original_metrics = self._calculate_metrics(original_approved, "ORIGINAL")
        fixed_v2_metrics = self._calculate_metrics(fixed_v2_approved, "FIXED_V2")

        # Generate report
        report = {
            'summary': {
                'total_trades': len(self.trades),
                'original_approved': len(original_approved),
                'fixed_v2_approved': len(fixed_v2_approved),
                'original_approval_rate': len(original_approved) / max(len(self.trades), 1),
                'fixed_v2_approval_rate': len(fixed_v2_approved) / max(len(self.trades), 1),
            },
            'original_performance': original_metrics,
            'fixed_v2_performance': fixed_v2_metrics,
            'improvement': {
                'approval_rate_change': (
                    fixed_v2_metrics['approval_rate'] - original_metrics['approval_rate']
                ) * 100,
                'expectancy_change': (
                    fixed_v2_metrics['expectancy'] - original_metrics['expectancy']
                ),
                'total_pnl_change': (
                    fixed_v2_metrics['total_pnl'] - original_metrics['total_pnl']
                ),
            },
            'veto_comparison': dict(veto_comparison),
        }

        return report
    
    def _calculate_metrics(self, approved_trades: List[HistoricalTrade], label: str) -> Dict:
        """Calculate performance metrics for approved trades"""
        if not approved_trades:
            return {
                'total_trades': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'win_rate': 0,
                'expectancy': 0,
                'approval_rate': 0,
            }
        
        total_pnl = sum(t.pnl_usd for t in approved_trades)
        wins = [t for t in approved_trades if t.pnl_usd > 0]
        losses = [t for t in approved_trades if t.pnl_usd <= 0]
        
        win_rate = len(wins) / len(approved_trades)
        avg_win = sum(t.pnl_usd for t in wins) / max(len(wins), 1)
        avg_loss = sum(t.pnl_usd for t in losses) / max(len(losses), 1)
        
        # Expectancy = (Win% × Avg Win) - (Loss% × Avg Loss)
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        
        return {
            'total_trades': len(approved_trades),
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(approved_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'approval_rate': len(approved_trades) / max(len(self.trades), 1),
        }
    
    def print_report(self, report: Dict):
        """Print formatted backtest report"""
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)

        # Summary
        print("\n[SUMMARY]")
        print("-" * 70)
        summary = report['summary']
        print(f"Total Historical Trades: {summary['total_trades']}")
        print(f"Original System Approved: {summary['original_approved']} ({summary['original_approval_rate']*100:.1f}%)")
        print(f"Fixed V2 System Approved: {summary['fixed_v2_approved']} ({summary['fixed_v2_approval_rate']*100:.1f}%)")
        print(f"Approval Rate Change:     {report['improvement']['approval_rate_change']:+.1f} percentage points")

        # Performance comparison
        print("\n[PERFORMANCE COMPARISON]")
        print("-" * 70)

        orig = report['original_performance']
        fixed_v2 = report['fixed_v2_performance']
        
        print(f"{'Metric':<25} {'Original':>15} {'Fixed V2':>15} {'Change':>15}")
        print("-" * 70)
        print(f"{'Total Trades':<25} {orig['total_trades']:>15} {fixed_v2['total_trades']:>15} {fixed_v2['total_trades']-orig['total_trades']:>+15}")
        print(f"{'Total PnL ($)':<25} {orig['total_pnl']:>15.2f} {fixed_v2['total_pnl']:>15.2f} {report['improvement']['total_pnl_change']:>+15.2f}")
        print(f"{'Avg PnL ($)':<25} {orig['avg_pnl']:>15.2f} {fixed_v2['avg_pnl']:>15.2f} {fixed_v2['avg_pnl']-orig['avg_pnl']:>+15.2f}")
        print(f"{'Win Rate':<25} {orig['win_rate']*100:>14.1f}% {fixed_v2['win_rate']*100:>14.1f}% {(fixed_v2['win_rate']-orig['win_rate'])*100:>+14.1f}%")
        print(f"{'Expectancy ($/trade)':<25} {orig['expectancy']:>15.2f} {fixed_v2['expectancy']:>15.2f} {report['improvement']['expectancy_change']:>+15.2f}")

        # Veto comparison
        print("\n[VETO COMPARISON]")
        print("-" * 70)
        print(f"{'Gate Type':<20} {'Original Vetoes':>20} {'Fixed V2 Vetoes':>20}")
        print("-" * 70)

        for gate_type, counts in report['veto_comparison'].items():
            print(f"{gate_type:<20} {counts['original']:>20} {counts['fixed_v2']:>20}")
        
        # Key findings
        print("\n[KEY FINDINGS]")
        print("-" * 70)

        if report['improvement']['approval_rate_change'] > 5:
            print(f"[PASS] Approval rate increased by {report['improvement']['approval_rate_change']:.1f}%")

        if report['improvement']['expectancy_change'] > 0:
            print(f"[PASS] Expectancy improved by ${report['improvement']['expectancy_change']:.2f}/trade")

        if report['improvement']['total_pnl_change'] > 0:
            print(f"[PASS] Total PnL improved by ${report['improvement']['total_pnl_change']:.2f}")

        # Check which gates were blocking most trades
        max_original_veto = max(
            report['veto_comparison'].items(),
            key=lambda x: x[1]['original'],
            default=(None, {'original': 0})
        )
        if max_original_veto[0]:
            print(f"[WARN] Main blocker in original: {max_original_veto[0]} ({max_original_veto[1]['original']} trades)")
        
        print("\n" + "=" * 70)


def main():
    """Main entry point"""
    print("=" * 70)
    print("HISTORICAL FIX BACKTEST ENGINE V2")
    print("=" * 70)

    # Create backtest engine
    engine = HistoricalBacktestEngine(db_path='holonic_trader.db')

    # Load asset performance stats
    asset_stats = {}
    stats_path = 'asset_performance_stats.json'
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            asset_stats = json.load(f)
        print(f"Loaded asset stats for {len(asset_stats)} symbols")

    if len(engine.trades) == 0:
        print("\n[WARN] No historical trades found in database or logs.")
        print("Creating synthetic trades for demonstration...")
        
        # Create synthetic trades based on log patterns we've seen
        engine.trades = [
            HistoricalTrade('SHIB/USDT', 'BUY', None, 0, None, 0, 0.50, 0.5, 'TEST', 2, 0.50, 'DIP', 'SUPPORT', 'LOW_VOL_MEAN_REVERT'),
            HistoricalTrade('SHIB/USDT', 'BUY', None, 0, None, 0, -0.30, -0.3, 'TEST', 1, 0.48, 'DIP', 'NEUTRAL', 'TRANSITION'),
            HistoricalTrade('DOT/USDT', 'BUY', None, 0, None, 0, -0.66, -0.66, 'TEST', 1, 0.52, 'DIP', 'NEUTRAL', 'TRANSITION'),
            HistoricalTrade('DOT/USDT', 'BUY', None, 0, None, 0, -0.07, -0.07, 'TEST', 1, 0.50, 'DIP', 'SUPPORT', 'LOW_VOL_MEAN_REVERT'),
            HistoricalTrade('ETH/USDT', 'BUY', None, 0, None, 0, 0.80, 0.8, 'TEST', 2, 0.55, 'DIP', 'SUPPORT', 'LOW_VOL_MEAN_REVERT'),
            HistoricalTrade('TAO/USDT', 'BUY', None, 0, None, 0, 0.48, 0.48, 'TEST', 1, 0.50, 'DIP', 'SUPPORT', 'LOW_VOL_MEAN_REVERT'),
            HistoricalTrade('WIF/USDT', 'BUY', None, 0, None, 0, -0.52, -0.52, 'TEST', 1, 0.45, 'DIP', 'NEUTRAL', 'TRANSITION'),
            HistoricalTrade('LDO/USDT', 'BUY', None, 0, None, 0, -0.14, -0.14, 'TEST', 1, 0.48, 'DIP', 'SUPPORT', 'LOW_VOL_MEAN_REVERT'),
        ]
        print(f"Created {len(engine.trades)} synthetic trades")

    # Run backtest V2 (with blacklist + win rate filter)
    report = engine.run_backtest_v2(asset_stats)

    # Print report
    engine.print_report(report)
    
    # Save report to file
    report_path = 'backtest_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[INFO] Full report saved to: {report_path}")

    return report


if __name__ == '__main__':
    main()
