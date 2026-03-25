"""
Simple trade-event extractor for HolonicTrader logs.
Parses `live_trading_session_*.log` files and extracts ENTRY, EXIT, STOP placement and FILL events.
Outputs `datasets/raw_trades_snapshot.csv` and attempts parquet `datasets/raw_trades_snapshot.parquet`.

Updated regex patterns to match actual log formats:
- ENTRY: [2026-03-21 01:27:41] [TraderNexus] 🎯 EXECUTING ENTRY (Attempt 1/2): DOT/USDT (Qty: 9.9536, Lev: 2.0x)
- EXIT: [2026-03-21 01:59:39] [GovernorAgent] 📉 Trade Logged: DOT/USDT Profit: -0.46%. Consecutive Losses: 1/2
"""
import re
import os
import glob
import csv
from datetime import datetime
import pandas as pd

LOG_GLOB = 'live_trading_session_*.log'
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'datasets')
os.makedirs(OUT_DIR, exist_ok=True)

# ENTRY pattern: [2026-03-21 01:27:41] [TraderNexus] 🎯 EXECUTING ENTRY (Attempt 1/2): DOT/USDT (Qty: 9.9536, Lev: 2.0x)
entry_re = re.compile(r"\[(?P<ts>[^\]]+)\].*EXECUTING ENTRY.*?:\s*(?P<symbol>\S+)\s*\(Qty:\s*(?P<qty>[\d\.eE+-]+),\s*Lev:\s*(?P<lev>[\d\.]+)x\)")

# EXIT pattern: [2026-03-21 01:59:39] [GovernorAgent] 📉 Trade Logged: DOT/USDT Profit: -0.46%. Consecutive Losses: 1/2
exit_re = re.compile(r"\[(?P<ts>[^\]]+)\].*Trade Logged:\s*(?P<symbol>\S+)\s*Profit:\s*(?P<pnl>[-\d\.]+)%")

# STOP placement pattern: [2026-03-20 21:20:48] [ActuatorAgent] 🛡️ PLACING STOP LOSS: SELL 0.03 AAVE/USDT @ 108.39
stop_place_re = re.compile(r"\[(?P<ts>[^\]]+)\].*(PLACC?ING|PLACING) STOP LOSS:\s*(?P<side>\w+)\s*(?P<qty>[\d\.eE+-]+)\s*(?P<symbol>\S+)\s*@\s*(?P<price>[\d\.eE+-]+)")

# Fallback to Actuator-formatted messages (STOP LOSS PLACED / STOP LOSS ACTIVE)
act_stop_re = re.compile(r"\[(?P<ts>[^\]]+)\].*ActuatorAgent\].*(STOP LOSS PLACED|STOP LOSS ACTIVE|STOP PLACED):\s*(?P<rest>.*)")

# Additional patterns: fills / order results
fill_re = re.compile(r"\[(?P<ts>[^\]]+)\].*(FILLED|filled|FILL)\s*(?P<symbol>\S+).*?qty[:=]?\s*(?P<filled>[\d\.eE+-]+).*?price[:=]?\s*(?P<price>[\d\.eE+-]+).*?order[_ ]?id[:=]?\s*(?P<orderid>[A-Za-z0-9\-]+)", re.IGNORECASE)
order_result_re = re.compile(r"\[(?P<ts>[^\]]+)\].*(order_id|order id|order)[:=\s]*?(?P<orderid>[A-Za-z0-9\-]+).*?filled[:=]?\s*(?P<filled>[\d\.eE+-]+).*?avg[_ ]?fill[_ ]?price[:=]?\s*(?P<avg>[\d\.eE+-]+).*?fee[:=]?\s*(?P<fee>[\d\.eE+-]+)", re.IGNORECASE)
fee_re = re.compile(r"fee[:=]?\s*(?P<fee>[\d\.eE+-]+).*(currency[:=]?\s*(?P<cur>\w+))?", re.IGNORECASE)

# EXIT with price: [2026-03-20 21:19:43] [ExecutorAgent] 📉 EXIT AAVE/USDT: Realized $-0.05
exit_price_re = re.compile(r"\[(?P<ts>[^\]]+)\].*EXIT\s*(?P<symbol>\S+):\s*Realized\s*\$(?P<pnl>[-\d\.]+)")

# ENTRY with price (alternative format)
entry_price_re = re.compile(r"\[(?P<ts>[^\]]+)\].*ENTRY\s*(?P<symbol>\S+):\s*(?P<qty>[\d\.eE+-]+)\s*@\s*(?P<price>[\d\.eE+-]+)")

rows = []
files = sorted(glob.glob(LOG_GLOB))
if not files:
    print('No log files matching', LOG_GLOB)

for f in files:
    with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
        for ln in fh:
            ln = ln.strip('\n')
            
            # Try ENTRY pattern (TraderNexus format)
            m = entry_re.search(ln)
            if m:
                rows.append({
                    'session_file': os.path.basename(f),
                    'timestamp': m.group('ts'),
                    'event': 'ENTRY',
                    'symbol': m.group('symbol').strip(),
                    'direction': None,
                    'qty': float(m.group('qty')),
                    'leverage': float(m.group('lev')) if m.group('lev') else None,
                    'price': None,  # Price not in this format
                    'pnl_percent': None,
                    'order_id': None,
                    'fee_usd': None,
                    'raw': ln
                })
                continue
            
            # Try ENTRY with price (alternative format)
            m = entry_price_re.search(ln)
            if m:
                rows.append({
                    'session_file': os.path.basename(f),
                    'timestamp': m.group('ts'),
                    'event': 'ENTRY',
                    'symbol': m.group('symbol').strip(),
                    'direction': None,
                    'qty': float(m.group('qty')),
                    'leverage': None,
                    'price': float(m.group('price')),
                    'pnl_percent': None,
                    'order_id': None,
                    'fee_usd': None,
                    'raw': ln
                })
                continue
            
            # Try EXIT pattern (GovernorAgent Trade Logged)
            m = exit_re.search(ln)
            if m:
                rows.append({
                    'session_file': os.path.basename(f),
                    'timestamp': m.group('ts'),
                    'event': 'EXIT',
                    'symbol': m.group('symbol').strip(),
                    'direction': None,
                    'qty': None,
                    'leverage': None,
                    'price': None,
                    'pnl_percent': float(m.group('pnl')),
                    'order_id': None,
                    'fee_usd': None,
                    'raw': ln
                })
                continue
            
            # Try EXIT with price (Executor format)
            m = exit_price_re.search(ln)
            if m:
                rows.append({
                    'session_file': os.path.basename(f),
                    'timestamp': m.group('ts'),
                    'event': 'EXIT',
                    'symbol': m.group('symbol').strip(),
                    'direction': None,
                    'qty': None,
                    'leverage': None,
                    'price': None,
                    'pnl_percent': float(m.group('pnl')),
                    'order_id': None,
                    'fee_usd': None,
                    'raw': ln
                })
                continue
            
            # Try STOP placement
            m = stop_place_re.search(ln)
            if m:
                rows.append({
                    'session_file': os.path.basename(f),
                    'timestamp': m.group('ts'),
                    'event': 'STOP_PLACED',
                    'symbol': m.group('symbol').strip(),
                    'direction': m.group('side').strip(),
                    'qty': float(m.group('qty')),
                    'leverage': None,
                    'price': float(m.group('price')),
                    'pnl_percent': None,
                    'order_id': None,
                    'fee_usd': None,
                    'raw': ln
                })
                continue
            
            # Try Actuator fallback
            m = act_stop_re.search(ln)
            if m:
                rest = m.group('rest')
                small = re.search(r"(?P<side>BUY|SELL)\s(?P<qty>[\d\.eE+-]+)\s(?P<symbol>\S+)\s@\s(?P<price>[\d\.eE+-]+)", rest)
                if small:
                    rows.append({
                        'session_file': os.path.basename(f),
                        'timestamp': m.group('ts'),
                        'event': 'STOP_PLACED',
                        'symbol': small.group('symbol').strip(),
                        'direction': small.group('side').strip(),
                        'qty': float(small.group('qty')),
                        'leverage': None,
                        'price': float(small.group('price')),
                        'pnl_percent': None,
                        'order_id': None,
                        'fee_usd': None,
                        'raw': ln
                    })
                continue
            
            # Try FILL pattern
            m = fill_re.search(ln)
            if m:
                rows.append({
                    'session_file': os.path.basename(f),
                    'timestamp': m.group('ts'),
                    'event': 'FILL',
                    'symbol': m.group('symbol').strip(),
                    'direction': None,
                    'qty': float(m.group('filled')),
                    'leverage': None,
                    'price': float(m.group('price')),
                    'pnl_percent': None,
                    'order_id': m.group('orderid'),
                    'fee_usd': None,
                    'raw': ln
                })
                continue
            
            # Try order result pattern
            m = order_result_re.search(ln)
            if m:
                rows.append({
                    'session_file': os.path.basename(f),
                    'timestamp': m.group('ts'),
                    'event': 'ORDER_RESULT',
                    'symbol': None,
                    'direction': None,
                    'qty': float(m.group('filled')),
                    'leverage': None,
                    'price': float(m.group('avg')),
                    'pnl_percent': None,
                    'order_id': m.group('orderid'),
                    'fee_usd': float(m.group('fee')),
                    'raw': ln
                })
                continue
            
            # Try fee pattern
            m = fee_re.search(ln)
            if m:
                rows.append({
                    'session_file': os.path.basename(f),
                    'timestamp': datetime.now().isoformat(),
                    'event': 'FEE_REPORTED',
                    'symbol': None,
                    'direction': None,
                    'qty': None,
                    'leverage': None,
                    'price': None,
                    'pnl_percent': None,
                    'order_id': None,
                    'fee_usd': float(m.group('fee')),
                    'raw': ln
                })
                continue

# Write CSV
out_csv = os.path.join(OUT_DIR, 'raw_trades_snapshot.csv')
keys = ['session_file','timestamp','event','symbol','direction','qty','leverage','price','pnl_percent','order_id','fee_usd','raw']
with open(out_csv, 'w', newline='', encoding='utf-8') as of:
    writer = csv.DictWriter(of, fieldnames=keys)
    writer.writeheader()
    for r in rows:
        # ensure all keys present
        out = {k: r.get(k, None) for k in keys}
        writer.writerow(out)

print(f'Wrote {len(rows)} events to {out_csv}')

# Print summary
event_counts = {}
for r in rows:
    evt = r.get('event', 'UNKNOWN')
    event_counts[evt] = event_counts.get(evt, 0) + 1
print(f'\nEvent summary:')
for evt, cnt in sorted(event_counts.items()):
    print(f'  {evt}: {cnt}')

# Try parquet
try:
    df = pd.DataFrame(rows)
    out_parquet = os.path.join(OUT_DIR, 'raw_trades_snapshot.parquet')
    df.to_parquet(out_parquet, index=False)
    print(f'\nWrote parquet: {out_parquet}')
    
    # Show sample
    print(f'\nSample rows:')
    print(df[['event', 'symbol', 'timestamp', 'qty', 'price', 'pnl_percent']].head(10).to_string())
except Exception as e:
    print('Parquet write failed (pyarrow may be missing):', e)
    print('CSV is available at', out_csv)
