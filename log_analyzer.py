#!/usr/bin/env python3
"""
Log Analyzer for HolonicTrader
Analyzes trading logs to identify performance issues and loss patterns
"""

import re
import os
import sys
from datetime import datetime
from collections import defaultdict, Counter
import statistics

def analyze_log_performance(log_file_path):
    """Analyze a trading log file for performance metrics"""
    print(f"Analyzing log file: {log_file_path}")
    
    if not os.path.exists(log_file_path):
        print(f"Error: Log file {log_file_path} does not exist")
        return
    
    # Initialize metrics
    metrics = {
        'total_lines': 0,
        'trades': [],
        'positions': {},
        'errors': [],
        'warnings': [],
        'balance_changes': [],
        'pnl_snapshots': [],
        'entry_signals': [],
        'exit_signals': [],
        'system_states': [],
        'hygiene_events': [],
        'monte_carlo_events': []
    }
    
    # Patterns for extracting data
    patterns = {
        'balance_sync': r'\[ExecutorAgent\] SYNC SUCCESS: Real Equity \$(\d+\.\d+)',
        'trade_entry': r'\[ExecutorAgent\] EXECUTED: ([A-Z0-9/]+) ([A-Z]+) ([\d.-]+) @ ([\d.,]+)',
        'trade_exit': r'\[ExecutorAgent\] CLOSED: ([A-Z0-9/]+) ([A-Z]+) ([\d.-]+) @ ([\d.,]+)',
        'pnl_realized': r'\[ExecutorAgent\] REALIZED PnL: \$([+-]?[\d.]+)',
        'position_opened': r'\[GovernorAgent\] Position OPENED: ([A-Z0-9/]+) ([A-Z]+) @ ([\d.]+)',
        'position_closed': r'\[GovernorAgent\] Position CLOSED: ([A-Z0-9/]+)',
        'hygiene_recycle': r'\[GovernorAgent\] ☣️ HYGIENE RECYCLE: ([A-Z0-9/]+) (.+)',
        'monte_carlo_exit': r'\[TraderNexus\] 🎲 MONTE CARLO EXIT: ([A-Z0-9/]+) - (.+)',
        'error': r'(ERROR|CRITICAL|FATAL).*',
        'warning': r'(WARNING|WARN).*',
        'system_state': r'\[GovernorAgent\] State: (\w+)',
        'drawdown': r'\[GovernorAgent\] Drawdown: ([\d.]+)%',
        'balance_snapshot': r'\[GovernorAgent\] Balance: \$(\d+\.\d+)',
        'veto': r'\[GovernorAgent\] .* VETO: (.*)',
        'solvent_check': r'\[GovernorAgent\] SOLVENCY (CHECK|VETO): (.*)'
    }
    
    # Compile regex patterns
    compiled_patterns = {k: re.compile(v, re.IGNORECASE) for k, v in patterns.items()}
    
    # Read and analyze log file
    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            metrics['total_lines'] += 1
            
            # Check for various patterns
            for pattern_name, pattern in compiled_patterns.items():
                match = pattern.search(line)
                if match:
                    if pattern_name == 'balance_sync':
                        balance = float(match.group(1))
                        metrics['balance_changes'].append(('SYNC', balance, line_num))
                    elif pattern_name == 'trade_entry':
                        symbol, direction, qty, price = match.groups()
                        metrics['trades'].append({
                            'type': 'ENTRY',
                            'symbol': symbol,
                            'direction': direction,
                            'quantity': float(qty),
                            'price': float(price.replace(',', '')),
                            'line': line_num
                        })
                    elif pattern_name == 'trade_exit':
                        symbol, direction, qty, price = match.groups()
                        metrics['trades'].append({
                            'type': 'EXIT',
                            'symbol': symbol,
                            'direction': direction,
                            'quantity': float(qty),
                            'price': float(price.replace(',', '')),
                            'line': line_num
                        })
                    elif pattern_name == 'pnl_realized':
                        pnl = float(match.group(1))
                        metrics['pnl_snapshots'].append(pnl)
                    elif pattern_name == 'hygiene_recycle':
                        symbol, reason = match.groups()
                        metrics['hygiene_events'].append({
                            'symbol': symbol,
                            'reason': reason,
                            'line': line_num
                        })
                    elif pattern_name == 'monte_carlo_exit':
                        symbol, reason = match.groups()
                        metrics['monte_carlo_events'].append({
                            'symbol': symbol,
                            'reason': reason,
                            'line': line_num
                        })
                    elif pattern_name == 'error':
                        metrics['errors'].append((match.group(0), line_num))
                    elif pattern_name == 'warning':
                        metrics['warnings'].append((match.group(0), line_num))
    
    return metrics

def calculate_performance_metrics(metrics):
    """Calculate performance metrics from parsed log data"""
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS RESULTS")
    print("="*60)

    print(f"Total Lines Analyzed: {metrics['total_lines']:,}")
    print(f"Total Trades Processed: {len(metrics['trades']):,}")
    print(f"Realized PnL Events: {len(metrics['pnl_snapshots']):,}")
    print(f"Hygiene Events: {len(metrics['hygiene_events']):,}")
    print(f"Monte Carlo Events: {len(metrics['monte_carlo_events']):,}")
    print(f"Errors Found: {len(metrics['errors']):,}")
    print(f"Warnings Found: {len(metrics['warnings']):,}")

    # Initialize variables to avoid UnboundLocalError
    avg_pnl = 0
    win_rate = 0
    avg_win = 0
    avg_loss = 0
    winning_trades = []
    losing_trades = []
    total_pnl = 0
    pnl_std = 0
    max_gain = 0
    max_loss = 0
    hygiene_by_reason = {}

    # Calculate PnL statistics
    if metrics['pnl_snapshots']:
        total_pnl = sum(metrics['pnl_snapshots'])
        avg_pnl = statistics.mean(metrics['pnl_snapshots'])
        pnl_std = statistics.stdev(metrics['pnl_snapshots']) if len(metrics['pnl_snapshots']) > 1 else 0
        max_gain = max(metrics['pnl_snapshots']) if metrics['pnl_snapshots'] else 0
        max_loss = min(metrics['pnl_snapshots']) if metrics['pnl_snapshots'] else 0

        winning_trades = [pnl for pnl in metrics['pnl_snapshots'] if pnl > 0]
        losing_trades = [pnl for pnl in metrics['pnl_snapshots'] if pnl < 0]

        win_rate = len(winning_trades) / len(metrics['pnl_snapshots']) if metrics['pnl_snapshots'] else 0
        avg_win = statistics.mean(winning_trades) if winning_trades else 0
        avg_loss = statistics.mean(losing_trades) if losing_trades else 0
        
        print(f"\nPnL Statistics:")
        print(f"  Total PnL: ${total_pnl:.2f}")
        print(f"  Average PnL: ${avg_pnl:.2f}")
        print(f"  PnL Std Dev: ${pnl_std:.2f}")
        print(f"  Largest Gain: ${max_gain:.2f}")
        print(f"  Largest Loss: ${max_loss:.2f}")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Avg Win: ${avg_win:.2f}")
        print(f"  Avg Loss: ${avg_loss:.2f}")
        print(f"  Profit Factor: {abs(avg_win/avg_loss) if avg_loss != 0 else float('inf'):.2f}")
    
    # Analyze hygiene events
    if metrics['hygiene_events']:
        print(f"\nHygiene Event Analysis:")
        hygiene_by_reason = Counter([event['reason'].split(':')[0] for event in metrics['hygiene_events']])
        for reason, count in hygiene_by_reason.most_common():
            print(f"  {reason}: {count} events")
    
    # Analyze Monte Carlo events
    if metrics['monte_carlo_events']:
        print(f"\nMonte Carlo Event Analysis:")
        mc_by_reason = Counter([event['reason'].split(':')[0] for event in metrics['monte_carlo_events']])
        for reason, count in mc_by_reason.most_common():
            print(f"  {reason}: {count} events")
    
    # Analyze errors and warnings
    if metrics['errors']:
        print(f"\nTop 10 Errors:")
        error_counts = Counter([err[0] for err in metrics['errors']])
        for error, count in error_counts.most_common(10):
            print(f"  [{count}] {error[:100]}...")
    
    if metrics['warnings']:
        print(f"\nTop 10 Warnings:")
        warning_counts = Counter([warn[0] for warn in metrics['warnings']])
        for warning, count in warning_counts.most_common(10):
            print(f"  [{count}] {warning[:100]}...")
    
    # Identify potential issues
    print(f"\n" + "="*60)
    print("POTENTIAL ISSUES IDENTIFIED")
    print("="*60)
    
    issues_found = []
    
    if metrics['pnl_snapshots'] and avg_pnl < 0:
        issues_found.append(f"NEGATIVE AVERAGE PnL: ${avg_pnl:.2f}")
    
    if win_rate < 0.4:  # Less than 40% win rate
        issues_found.append(f"LOW WIN RATE: {win_rate:.2%}")
    
    if len(losing_trades) > len(winning_trades) * 2:  # More than 2x as many losing trades
        issues_found.append(f"HIGH RATIO OF LOSING TRADES: {len(losing_trades)} losing vs {len(winning_trades)} winning")
    
    if metrics['errors']:
        issues_found.append(f"SIGNIFICANT ERRORS PRESENT: {len(metrics['errors'])} errors found")
    
    if hygiene_by_reason.get('TOXIC_FUNDING', 0) > 5:  # More than 5 toxic funding events
        issues_found.append(f"HIGH TOXIC FUNDING EVENTS: {hygiene_by_reason['TOXIC_FUNDING']} events")
    
    if not metrics['monte_carlo_events']:
        issues_found.append("MONTE CARLO SYSTEM NOT ACTIVE: No Monte Carlo events found")
    
    for i, issue in enumerate(issues_found, 1):
        print(f"{i}. {issue}")
    
    if not issues_found:
        print("No major issues identified in this analysis.")
    
    return issues_found

def analyze_latest_logs():
    """Analyze the most recent log files"""
    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_files = []
    
    # Look for log files in the current directory and parent directory
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith('.log') and 'live_trading_session' in file:
                log_files.append(os.path.join(root, file))
    
    # Also check parent directory
    parent_dir = os.path.dirname(log_dir)
    for file in os.listdir(parent_dir):
        if file.endswith('.log') and 'live_trading_session' in file:
            log_files.append(os.path.join(parent_dir, file))
    
    if not log_files:
        print("No log files found in current or parent directory")
        return
    
    # Sort by modification time to get the most recent
    log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    print(f"Found {len(log_files)} log files, analyzing the most recent:")
    for i, log_file in enumerate(log_files[:3]):  # Analyze top 3 most recent
        print(f"\nAnalyzing: {os.path.basename(log_file)}")
        print("-" * 80)
        
        metrics = analyze_log_performance(log_file)
        issues = calculate_performance_metrics(metrics)
        
        if i < 2:  # Only analyze Monte Carlo for first 2 files
            print(f"\nDetailed Monte Carlo Analysis for {os.path.basename(log_file)}:")
            if metrics['monte_carlo_events']:
                print(f"  Monte Carlo events found: {len(metrics['monte_carlo_events'])}")
                for event in metrics['monte_carlo_events'][:5]:  # Show first 5
                    print(f"    Line {event['line']}: {event['symbol']} - {event['reason']}")
            else:
                print(f"  No Monte Carlo events found - system may not be active")

if __name__ == "__main__":
    print("HolonicTrader Log Analyzer")
    print("="*60)
    
    if len(sys.argv) > 1:
        # Analyze specific log file
        log_file = sys.argv[1]
        if os.path.exists(log_file):
            metrics = analyze_log_performance(log_file)
            calculate_performance_metrics(metrics)
        else:
            print(f"File {log_file} not found")
    else:
        # Analyze latest log files
        analyze_latest_logs()