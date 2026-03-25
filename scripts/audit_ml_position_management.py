"""
ML Position Management Audit

Traces how ML manages positions from entry → monitoring → exit
"""
import glob
import re
from collections import defaultdict

print("=" * 70)
print("ML POSITION MANAGEMENT AUDIT")
print("=" * 70)

# Find recent logs
log_files = sorted(glob.glob('live_trading_session_*.log'))[-3:]

positions = defaultdict(lambda: {
    'entry_time': None,
    'entry_price': None,
    'exit_time': None,
    'exit_price': None,
    'ml_predictions': [],
    'ml_exits': [],
    'final_pnl': None
})

for log_file in log_files:
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Parse ENTRY
            if 'EXECUTING ENTRY' in line:
                match = re.search(r'EXECUTING ENTRY.*?: (\S+) \(Qty: ([\d.]+), Lev: ([\d.]+)x\) @ ([\d.]+)', line)
                if match:
                    symbol = match.group(1)
                    positions[symbol]['entry_time'] = line.split(']')[0].lstrip('[')
                    positions[symbol]['entry_price'] = float(match.group(4))
                    positions[symbol]['entry_qty'] = float(match.group(2))
                    positions[symbol]['entry_lev'] = float(match.group(3))
            
            # Parse ML EXIT recommendations
            if '🤖 ML EXIT:' in line:
                match = re.search(r'🤖 ML EXIT: (\S+) - (\w+) \((.+?)\)', line)
                if match:
                    symbol = match.group(1)
                    positions[symbol]['ml_exits'].append({
                        'time': line.split(']')[0].lstrip('['),
                        'recommendation': match.group(2),
                        'reason': match.group(3)
                    })
            
            # Parse EXIT/Realized
            if '📉 EXIT' in line and 'Realized' in line:
                match = re.search(r'📉 EXIT (\S+): Realized \$([-.\d]+)', line)
                if match:
                    symbol = match.group(1)
                    positions[symbol]['exit_time'] = line.split(']')[0].lstrip('[')
                    positions[symbol]['final_pnl_usd'] = float(match.group(2))
            
            # Parse Governor Trade Logged
            if 'Trade Logged' in line and 'Governor' in line:
                match = re.search(r'Trade Logged: (\S+) Profit: ([-.\d]+)%', line)
                if match:
                    symbol = match.group(1)
                    positions[symbol]['final_pnl_pct'] = float(match.group(2))

print("\n" + "=" * 70)
print("POSITION LIFECYCLE ANALYSIS")
print("=" * 70)

for symbol, data in sorted(positions.items(), key=lambda x: x[1]['entry_time'] or ''):
    if not data['entry_time']:
        continue
    
    print(f"\n{symbol}")
    print(f"  Entry: {data['entry_time']} @ ${data['entry_price']}")
    print(f"  Exit:  {data['exit_time'] or 'OPEN'} @ {data['exit_price'] or 'N/A'}")
    print(f"  PnL:   {data['final_pnl_pct'] or 'N/A':+.2f}% ({data.get('final_pnl_usd', 'N/A')})")
    
    # ML Exit recommendations
    if data['ml_exits']:
        print(f"  ML Exit Checks: {len(data['ml_exits'])} recommendations")
        
        # Show unique recommendations
        rec_types = defaultdict(int)
        for rec in data['ml_exits']:
            rec_types[rec['recommendation']] += 1
        
        for rec_type, count in sorted(rec_types.items()):
            print(f"    - {rec_type}: {count} times")
        
        # Show last recommendation
        if data['ml_exits']:
            last = data['ml_exits'][-1]
            print(f"    Last: {last['recommendation']} - {last['reason']}")
    else:
        print(f"  ML Exit Checks: None recorded")

print("\n" + "=" * 70)
print("ML EXIT EFFECTIVENESS")
print("=" * 70)

# Analyze ML exit behavior
all_recs = []
for symbol, data in positions.items():
    for rec in data['ml_exits']:
        all_recs.append(rec['recommendation'])

if all_recs:
    from collections import Counter
    rec_counts = Counter(all_recs)
    
    print("\nRecommendation Distribution:")
    for rec, count in rec_counts.most_common():
        pct = count / len(all_recs) * 100
        print(f"  {rec}: {count} ({pct:.1f}%)")
    
    # Check if CUT_LOSS was ever recommended
    cut_loss_count = rec_counts.get('CUT_LOSS', 0)
    take_profit_count = rec_counts.get('TAKE_PROFIT', 0)
    hold_count = rec_counts.get('HOLD', 0)
    
    print(f"\nAction Summary:")
    print(f"  HOLD recommendations: {hold_count}")
    print(f"  CUT_LOSS recommendations: {cut_loss_count}")
    print(f"  TAKE_PROFIT recommendations: {take_profit_count}")
    
    if hold_count > 0 and cut_loss_count == 0:
        print(f"\n⚠️  WARNING: Only HOLD recommendations - thresholds may be too conservative")
    elif cut_loss_count > 0:
        print(f"\n✅ CUT_LOSS actively recommending exits")

print("\n" + "=" * 70)
print("POSITION DURATION ANALYSIS")
print("=" * 70)

# Calculate position durations
from datetime import datetime

closed_positions = [p for p in positions.values() if p['exit_time']]

if closed_positions:
    durations = []
    for pos in closed_positions:
        try:
            entry = datetime.fromisoformat(pos['entry_time'].replace('Z', '+00:00'))
            exit = datetime.fromisoformat(pos['exit_time'].replace('Z', '+00:00'))
            duration = (exit - entry).total_seconds() / 60  # minutes
            durations.append(duration)
        except:
            pass
    
    if durations:
        avg_duration = sum(durations) / len(durations)
        print(f"\nClosed Positions: {len(closed_positions)}")
        print(f"  Average Duration: {avg_duration:.1f} minutes")
        print(f"  Shortest: {min(durations):.1f} min")
        print(f"  Longest: {max(durations):.1f} min")

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

print("\nML Position Management Assessment:")

# Check for issues
issues = []

# Issue 1: No CUT_LOSS recommendations
if all_recs and 'CUT_LOSS' not in all_recs:
    issues.append("❌ No CUT_LOSS recommendations - exit thresholds too conservative")

# Issue 2: Only HOLD
if all_recs and all(r == 'HOLD' for r in all_recs):
    issues.append("❌ 100% HOLD - ML exit not actively managing risk")

# Issue 3: No exit data
if len(closed_positions) == 0:
    issues.append("⚠️  No closed positions to analyze")

if issues:
    print("\nIssues Found:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("\n✅ ML exit management appears functional")

print("\n" + "=" * 70)
