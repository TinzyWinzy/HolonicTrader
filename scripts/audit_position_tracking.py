"""
Position Tracking Audit - Find discrepancy between logs and exchange

This script finds where "14 positions" is coming from when exchange shows 0
"""
import glob
import re

print('=' * 70)
print('POSITION TRACKING AUDIT')
print('=' * 70)

# Find most recent logs
log_files = sorted(glob.glob('live_trading_session_*.log'))[-5:]

entries = []
exits = []
position_opens = []
position_closes = []

for log_file in log_files:
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
        # Count EXECUTING ENTRY
        entry_matches = re.findall(r'EXECUTING ENTRY.*?: (\S+)', content)
        for symbol in entry_matches:
            entries.append(symbol)
        
        # Count EXIT Realized
        exit_matches = re.findall(r'📉 EXIT (\S+):', content)
        for symbol in exit_matches:
            exits.append(symbol)
        
        # Count POSITION OPENED
        open_matches = re.findall(r'Position OPENED: (\S+)', content)
        for symbol in open_matches:
            position_opens.append(symbol)
        
        # Count position closed
        close_matches = re.findall(r'Position CLOSED|📉 EXIT', content)
        position_closes.extend(close_matches)

print(f'\nLog Analysis:')
print(f'  "EXECUTING ENTRY" count: {len(entries)}')
print(f'  "📉 EXIT" count: {len(exits)}')
print(f'  "Position OPENED" count: {len(position_opens)}')
print(f'  "Position CLOSED/EXIT" count: {len(position_closes)}')

net_positions = len(position_opens) - len(position_closes)
print(f'\n  Net Positions (Open - Close): {net_positions}')

if len(entries) > 0 and len(exits) == 0 and len(position_opens) == 0:
    print(f'\n' + '=' * 70)
    print('🚨 CRITICAL FINDING')
    print('=' * 70)
    print()
    print('Entries are being LOGGED but NOT EXECUTED!')
    print()
    print('Evidence:')
    print(f'  - {len(entries)} "EXECUTING ENTRY" messages in logs')
    print(f'  - 0 "Position OPENED" confirmations')
    print(f'  - 0 positions on exchange')
    print()
    print('This means:')
    print('  1. System GENERATES entry signals')
    print('  2. System LOGS "EXECUTING ENTRY"')
    print('  3. But orders are REJECTED or NOT sent to exchange')
    print()
    print('LIKELY CAUSES:')
    print('  1. Governor veto AFTER "EXECUTING ENTRY" log')
    print('  2. Actuator order rejection')
    print('  3. Exchange API rejection')
    print('  4. Risk checks blocking execution')
    print()
    print('SEARCH FOR THESE LOGS:')
    print('  - "REJECT" or "VETO"')
    print('  - "Order Failed" or "Rejected"')
    print('  - "Insufficient" or "Error"')
    print()
    
    # Search for rejection messages
    print('Scanning for rejection messages...')
    rejects = []
    for log_file in log_files:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if 'REJECT' in line or 'VETO' in line or 'Failed' in line or 'Error' in line:
                    if 'ENTRY' in line or 'order' in line.lower():
                        rejects.append(line.strip()[:200])
    
    if rejects:
        print(f'\nFound {len(rejects)} rejection messages:')
        for r in rejects[:10]:
            print(f'  {r}')
    else:
        print('\nNo obvious rejection messages found')
        print('Check ActuatorAgent logs for order status')

elif len(position_opens) > 0:
    print(f'\n' + '=' * 70)
    print('POSITIONS WERE OPENED')
    print('=' * 70)
    print()
    print(f'{len(position_opens)} positions were opened according to logs')
    print(f'{len(position_closes)} positions were closed')
    print(f'Net open: {net_positions}')
    print()
    print('If exchange shows 0 positions but logs show opens:')
    print('  1. Position tracking out of sync')
    print('  2. Paper/simulated trades in logs')
    print('  3. Exchange sync issue')
else:
    print(f'\n' + '=' * 70)
    print('NO POSITIONS IN LOGS')
    print('=' * 70)
    print()
    print('Logs confirm: No positions should be open')
    print('The "14 positions" must be from old/incorrect analysis')
