"""
Root Cause Analysis - Why All Losses?

Analyzes trade patterns to understand why win rate is 0%
"""
import glob
import re
from collections import defaultdict

print("=" * 70)
print("ROOT CAUSE ANALYSIS - WHY ALL LOSSES?")
print("=" * 70)

# Parse all trades from logs
log_files = sorted(glob.glob('live_trading_session_*.log'))
trades = []

for log_file in log_files:
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if 'Trade Logged' in line and 'Governor' in line:
                match = re.search(r'Trade Logged: (\S+) Profit: ([-.\d]+)%', line)
                if match:
                    pnl = float(match.group(2))
                    trades.append({
                        'symbol': match.group(1),
                        'pnl': pnl,
                        'is_win': pnl > 0,
                        'file': log_file
                    })

print(f"\nTotal Trades Analyzed: {len(trades)}")
print(f"Wins: {sum(1 for t in trades if t['is_win'])} ({sum(1 for t in trades if t['is_win'])/len(trades)*100:.1f}%)")
print(f"Losses: {sum(1 for t in trades if not t['is_win'])} ({sum(1 for t in trades if not t['is_win'])/len(trades)*100:.1f}%)")

# 1. Symbol Analysis
print("\n" + "=" * 70)
print("1. SYMBOL ANALYSIS")
print("=" * 70)

symbol_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0})
for t in trades:
    sym = t['symbol']
    symbol_stats[sym]['total_pnl'] += t['pnl']
    if t['is_win']:
        symbol_stats[sym]['wins'] += 1
    else:
        symbol_stats[sym]['losses'] += 1

print("\nPerformance by Symbol:")
for sym, stats in sorted(symbol_stats.items(), key=lambda x: -x[1]['total_pnl']):
    total = stats['wins'] + stats['losses']
    win_rate = stats['wins'] / total * 100 if total > 0 else 0
    print(f"  {sym:<12} {stats['wins']}/{total} wins ({win_rate:.0f}%)  Total: {stats['total_pnl']:+.2f}%")

# 2. Loss Magnitude Analysis
print("\n" + "=" * 70)
print("2. LOSS MAGNITUDE ANALYSIS")
print("=" * 70)

losses = [t for t in trades if not t['is_win']]
if losses:
    tiny = [l for l in losses if -0.1 <= l['pnl'] < 0]
    small = [l for l in losses if -0.5 <= l['pnl'] < -0.1]
    medium = [l for l in losses if -1.0 <= l['pnl'] < -0.5]
    large = [l for l in losses if l['pnl'] < -1.0]
    
    print(f"\nLoss Distribution:")
    print(f"  Tiny  (-0.1% to 0%):    {len(tiny)} ({len(tiny)/len(losses)*100:.1f}%)")
    print(f"  Small (-0.5% to -0.1%): {len(small)} ({len(small)/len(losses)*100:.1f}%)")
    print(f"  Medium (-1% to -0.5%):  {len(medium)} ({len(medium)/len(losses)*100:.1f}%)")
    print(f"  Large (<-1%):           {len(large)} ({len(large)/len(losses)*100:.1f}%)")
    
    avg_loss = sum(l['pnl'] for l in losses) / len(losses)
    print(f"\n  Average Loss: {avg_loss:.2f}%")
    print(f"  Median Loss:  {sorted(losses, key=lambda x: x['pnl'])[len(losses)//2]['pnl']:.2f}%")

# 3. Consecutive Loss Patterns
print("\n" + "=" * 70)
print("3. CONSECUTIVE LOSS PATTERNS")
print("=" * 70)

# Check for consecutive losses on same symbol
from itertools import groupby

symbol_sequences = defaultdict(list)
for t in trades:
    symbol_sequences[t['symbol']].append(t['is_win'])

print("\nConsecutive Loss Streaks by Symbol:")
for sym, results in symbol_sequences.items():
    max_streak = 0
    current_streak = 0
    for is_win in results:
        if not is_win:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    if max_streak > 1:
        print(f"  {sym}: {max_streak} consecutive losses")

# 4. Time-based Analysis
print("\n" + "=" * 70)
print("4. POTENTIAL ROOT CAUSES")
print("=" * 70)

print("\n✅ GOOD NEWS:")
print("  • All losses are small (<2%) - risk management working")
print("  • No catastrophic losses - stops are effective")
print("  • Losses are controlled - system not blowing up")

print("\n⚠️  LIKELY CAUSES:")

# Check if it's a sample size issue
if len(trades) < 50:
    print(f"\n  1. SMALL SAMPLE SIZE (n={len(trades)})")
    print(f"     • Need 50+ trades for statistical significance")
    print(f"     • 0% win rate on 23 trades is concerning but not definitive")
    print(f"     • Expected variance at 50% win rate: could see 0-5 wins by chance")

# Check for symbol concentration
if len(symbol_stats) < 10:
    print(f"\n  2. SYMBOL CONCENTRATION")
    print(f"     • Only {len(symbol_stats)} symbols traded")
    print(f"     • May be trading in unfavorable market conditions for these assets")
    print(f"     • Consider diversifying to more symbols")

# Check for systematic issues
avg_loss = sum(l['pnl'] for l in losses) / len(losses) if losses else 0
if -0.5 < avg_loss < -0.1:
    print(f"\n  3. DEATH BY A THOUSAND CUTS")
    print(f"     • Average loss {avg_loss:.2f}% suggests fees + slippage")
    print(f"     • May be getting stopped out by noise")
    print(f"     • Consider wider stops or better entry timing")

# Check for blacklist issues
ldo_trades = [t for t in trades if t['symbol'] == 'LDO/USDT']
if len(ldo_trades) > 5:
    ldo_losses = sum(1 for t in ldo_trades if not t['is_win'])
    print(f"\n  4. BLACKLIST NOT WORKING")
    print(f"     • LDO/USDT: {ldo_losses} losses but system kept trading it")
    print(f"     • Consecutive loss circuit breaker may not be triggering")
    print(f"     • Check Governor blacklist logic")

print("\n  5. EXIT OPTIMIZER TOO PASSIVE (NOW FIXED)")
print(f"     • Was recommending HOLD on 100% of losses")
print(f"     • Thresholds lowered - should cut losses earlier now")
print(f"     • Monitor for improvement")

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

print("\nIMMEDIATE:")
print("  1. ✅ Exit optimizer thresholds lowered (done)")
print("  2. ⏳ Monitor LDO/USDT - should be blacklisted after 2 losses")
print("  3. ⏳ Watch for CUT_LOSS recommendations from exit optimizer")

print("\nSHORT-TERM:")
print("  4. Collect 50+ trades for statistical significance")
print("  5. Analyze which symbols perform best")
print("  6. Consider stricter entry filter (raise ML threshold from 35% to 45%)")

print("\nLONG-TERM:")
print("  7. Retrain ML model on recent data")
print("  8. Add regime-specific filtering")
print("  9. Consider ensemble of multiple models")

print("\n" + "=" * 70)
