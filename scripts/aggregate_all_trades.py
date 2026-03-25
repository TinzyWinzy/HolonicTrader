"""
Aggregate all trade data from quant_ops_reports and execution_integrity log.
Creates a comprehensive trade dataset for ML training.
"""
import os
import json
import glob
import pandas as pd
from datetime import datetime

# --- Load quant_ops_reports ---
print("=" * 70)
print("LOADING QUANT_OPS_REPORTS")
print("=" * 70)

report_files = sorted(glob.glob('quant_ops_reports/cycle_*.json'))
print(f"Found {len(report_files)} report files")

all_trades = []
for rf in report_files[-20:]:  # Last 20 cycles
    with open(rf, 'r') as f:
        data = json.load(f)
    
    # Extract Atlas performance data
    if 'reports' in data and 'atlas' in data['reports']:
        atlas = data['reports']['atlas']
        if 'system_state_summary' in atlas:
            state = atlas['system_state_summary']
            all_trades.append({
                'cycle_id': data.get('cycle_id'),
                'timestamp': data.get('timestamp'),
                'equity': state.get('equity'),
                'expectancy': state.get('expectancy'),
                'win_rate': state.get('win_rate'),
                'total_trades': state.get('total_trades'),
                'max_drawdown': state.get('max_drawdown'),
            })
    
    # Extract Chronos loss attribution
    if 'reports' in data and 'chronos' in data['reports']:
        chronos = data['reports']['chronos']
        if 'loss_attribution' in chronos:
            for loss in chronos['loss_attribution']:
                if 'evidence' in loss:
                    for ev in loss['evidence']:
                        # Parse "DOT/USDT: -86.26% (STOP_TOO_LOOSE)"
                        parts = ev.split(': ')
                        if len(parts) >= 2:
                            symbol = parts[0]
                            rest = parts[1]
                            pnl_pct = rest.split('%')[0] if '%' in rest else None
                            reason = rest.split('(')[1].rstrip(')') if '(' in rest else None
                            all_trades.append({
                                'cycle_id': data.get('cycle_id'),
                                'timestamp': data.get('timestamp'),
                                'symbol': symbol,
                                'pnl_percent': float(pnl_pct) if pnl_pct else None,
                                'loss_category': loss.get('category'),
                                'loss_reason': reason,
                            })

print(f"Extracted {len(all_trades)} trade records from quant_ops_reports")

# --- Load execution_integrity log ---
print("\n" + "=" * 70)
print("LOADING EXECUTION_INTEGRITY LOG")
print("=" * 70)

integrity_path = 'logs/execution_integrity.json'
if os.path.exists(integrity_path):
    with open(integrity_path, 'r') as f:
        integrity = json.load(f)
    
    entries = integrity.get('entries', [])
    print(f"Found {len(entries)} integrity log entries")
    
    # Filter TRADE events
    trade_events = [e for e in entries if e.get('event_type') == 'TRADE']
    print(f"Found {len(trade_events)} TRADE events")
    
    for te in trade_events[:5]:
        print(f"  {te.get('timestamp')}: {te.get('event_type')} {te.get('symbol')} - {te.get('data', {})}")
else:
    print("No execution_integrity.json found")
    trade_events = []

# --- Load live trading logs for actual ENTRY/EXIT ---
print("\n" + "=" * 70)
print("PARSING LIVE TRADING LOGS")
print("=" * 70)

log_files = sorted(glob.glob('live_trading_session_*.log'))
print(f"Found {len(log_files)} log files")

entry_events = []
exit_events = []

for log_file in log_files[-5:]:  # Last 5 logs
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            
            # Parse ENTRY from TraderNexus
            if 'EXECUTING ENTRY' in line and 'TraderNexus' in line:
                # [2026-03-21 02:44:30] [TraderNexus] 🎯 EXECUTING ENTRY (Attempt 1/2): XRP/USDT (Qty: 10.3941, Lev: 2.0x)
                try:
                    ts_part = line.split('] [')[0].lstrip('[')
                    rest = line.split('EXECUTING ENTRY')[1]
                    symbol = rest.split(':')[1].split('(')[0].strip()
                    qty_part = rest.split('Qty:')[1].split(',')[0].strip() if 'Qty:' in rest else None
                    lev_part = rest.split('Lev:')[1].split(')')[0].strip() if 'Lev:' in rest else None
                    entry_events.append({
                        'timestamp': ts_part,
                        'symbol': symbol,
                        'qty': float(qty_part) if qty_part else None,
                        'leverage': float(lev_part.replace('x', '')) if lev_part else None,
                        'session_file': os.path.basename(log_file),
                        'raw': line,
                    })
                except Exception as e:
                    pass
            
            # Parse EXIT from GovernorAgent "Trade Logged"
            if 'Trade Logged' in line and 'GovernorAgent' in line:
                # [2026-03-21 02:57:32] [GovernorAgent] 🟢 Trade Logged: TAO/USDT Profit: 0.48%. Resetting loss streak.
                try:
                    ts_part = line.split('] [')[0].lstrip('[')
                    rest = line.split('Trade Logged:')[1].strip()
                    symbol = rest.split(' ')[0].strip()
                    pnl_part = rest.split('Profit:')[1].split('%')[0].strip() if 'Profit:' in rest else None
                    exit_events.append({
                        'timestamp': ts_part,
                        'symbol': symbol,
                        'pnl_percent': float(pnl_part) if pnl_part else None,
                        'session_file': os.path.basename(log_file),
                        'raw': line,
                    })
                except Exception as e:
                    pass

print(f"Found {len(entry_events)} ENTRY events")
print(f"Found {len(exit_events)} EXIT events")

if entry_events:
    print("\nSample ENTRY events:")
    for e in entry_events[:3]:
        print(f"  {e['timestamp']}: {e['symbol']} qty={e['qty']} lev={e['leverage']}")

if exit_events:
    print("\nSample EXIT events:")
    for e in exit_events[:3]:
        print(f"  {e['timestamp']}: {e['symbol']} pnl={e['pnl_percent']}%")

# --- Save aggregated data ---
print("\n" + "=" * 70)
print("SAVING AGGREGATED DATA")
print("=" * 70)

os.makedirs('datasets', exist_ok=True)

# Save entry events
if entry_events:
    entry_df = pd.DataFrame(entry_events)
    entry_df.to_parquet('datasets/entry_events.parquet', index=False)
    print(f"Saved {len(entry_df)} entry events to datasets/entry_events.parquet")

# Save exit events
if exit_events:
    exit_df = pd.DataFrame(exit_events)
    exit_df.to_parquet('datasets/exit_events.parquet', index=False)
    print(f"Saved {len(exit_df)} exit events to datasets/exit_events.parquet")

# Save quant_ops trades
if all_trades:
    trades_df = pd.DataFrame(all_trades)
    trades_df.to_parquet('datasets/quant_ops_trades.parquet', index=False)
    print(f"Saved {len(trades_df)} trade records to datasets/quant_ops_trades.parquet")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Entry events: {len(entry_events)}")
print(f"Exit events: {len(exit_events)}")
print(f"Quant ops trades: {len(all_trades)}")
print(f"Integrity trade events: {len(trade_events)}")
