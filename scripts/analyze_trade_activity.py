"""
Trade Activity Analysis - Find why system is not trading
"""
import glob
import re
from collections import Counter

# Find most recent logs
log_files = sorted(glob.glob('live_trading_session_*.log'))[-5:]
print('=' * 70)
print('TRADE ACTIVITY ANALYSIS')
print('=' * 70)
print(f'\nAnalyzing {len(log_files)} recent log files...')

# Count events
entry_count = 0
exit_count = 0
veto_count = 0
signal_count = 0

# Track veto reasons
veto_reasons = Counter()
ml_confidence = []

for log_file in log_files:
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Count signals
            if 'VOLATILITY SQUEEZE' in line or 'EntryOracle' in line:
                signal_count += 1
            
            # Count entries
            if 'EXECUTING ENTRY' in line:
                entry_count += 1
            
            # Count exits
            if 'EXIT' in line and 'Realized' in line:
                exit_count += 1
            
            # Count vetoes
            if 'VETO' in line or 'REJECT' in line:
                veto_count += 1
                # Extract reason
                if 'GOV_VETO' in line:
                    veto_reasons['GOV_VETO'] += 1
                if 'ML_' in line:
                    veto_reasons['ML_FILTER'] += 1
                if 'BLACKLIST' in line:
                    veto_reasons['BLACKLIST'] += 1
                if 'QTY_TOO_SMALL' in line or 'MinContract' in line:
                    veto_reasons['CONTRACT_SIZE'] += 1
            
            # ML confidence
            match = re.search(r'(\\d+\\.\\d+)%', line)
            if match and ('ML' in line or 'win_prob' in line.lower()):
                ml_confidence.append(float(match.group(1)))

print(f'\n' + '=' * 70)
print('ACTIVITY SUMMARY')
print('=' * 70)
print(f'Signals Generated: {signal_count}')
print(f'Entries Executed:  {entry_count}')
print(f'Exits Completed:   {exit_count}')
print(f'Vetoes/Rejections: {veto_count}')

if signal_count > 0 and entry_count > 0:
    entry_rate = entry_count / signal_count * 100
    print(f'\nEntry Rate: {entry_rate:.1f}% ({entry_count}/{signal_count})')

print(f'\n' + '=' * 70)
print('VETO BREAKDOWN')
print('=' * 70)
for reason, count in veto_reasons.most_common():
    pct = count / veto_count * 100 if veto_count > 0 else 0
    print(f'{reason:<20} {count:>5} ({pct:>5.1f}%)')

if ml_confidence:
    print(f'\n' + '=' * 70)
    print('ML CONFIDENCE DISTRIBUTION')
    print('=' * 70)
    avg_conf = sum(ml_confidence) / len(ml_confidence)
    print(f'Average: {avg_conf:.1f}%')
    print(f'Min: {min(ml_confidence):.1f}%')
    print(f'Max: {max(ml_confidence):.1f}%')

print(f'\n' + '=' * 70)
print('ROOT CAUSE ANALYSIS')
print('=' * 70)

if signal_count > 0 and entry_count == 0:
    print('\n🚨 NO TRADES EXECUTED DESPITE SIGNALS')
    print('\nLikely Causes:')
    if veto_reasons.get('ML_FILTER', 0) > 0:
        print(f'  1. ML Filter rejecting {veto_reasons["ML_FILTER"]} signals')
    if veto_reasons.get('CONTRACT_SIZE', 0) > 0:
        print(f'  2. Contract Size issues ({veto_reasons["CONTRACT_SIZE"]} rejections)')
    if veto_reasons.get('GOV_VETO', 0) > 0:
        print(f'  3. Governor Veto ({veto_reasons["GOV_VETO"]} rejections)')
    if ml_confidence and avg_conf < 45:
        print(f'  4. Low ML Confidence (avg {avg_conf:.1f}%)')
    
    print('\nRecommendations:')
    print('  1. Check ML model accuracy - may need retraining')
    print('  2. Verify contract size thresholds')
    print('  3. Review Governor risk parameters')
    print('  4. Check if signals match current market regime')
elif entry_count > 0:
    print(f'\n✅ System IS trading ({entry_count} entries)')
    if exit_count > 0:
        print(f'   {exit_count} exits completed')
    else:
        print('   ⚠️ No exits yet - positions still open')
else:
    print('\n⚠️ NO SIGNALS GENERATED')
    print('\nLikely Causes:')
    print('  1. Market conditions not matching strategy')
    print('  2. Oracle/Signal Provider not active')
    print('  3. VIX/macro filters too strict')
    print('  4. Data feed issues')
