#!/usr/bin/env python3
"""
Winner DNA Analysis — Phase C1
Analyzes trading database to identify repeatable winning patterns.

Usage:
    python analyze_winners.py [--min-trades 10] [--json]

Output: Console report + winner_dna_report.json
"""

import sqlite3
import json
import sys
from datetime import datetime
from collections import defaultdict

DB_PATH = "holonic_trader.db"

def load_trades(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Get column names to handle missing metadata columns gracefully
    cur.execute("PRAGMA table_info(trades)")
    columns = {row['name'] for row in cur.fetchall()}
    
    # Build SELECT with available columns
    base_cols = ['id', 'symbol', 'direction', 'quantity', 'price', 'cost_usd',
                 'timestamp', 'pnl', 'pnl_percent']
    optional_cols = ['mfe', 'mae', 'exit_reason', 'strategy_type', 'entropy_score',
                     'regime', 'conviction', 'quality_score', 'is_whitelisted']
    
    select_cols = base_cols + [c for c in optional_cols if c in columns]
    query = f"SELECT {', '.join(select_cols)} FROM trades WHERE pnl IS NOT NULL ORDER BY timestamp"
    
    rows = cur.fetchall() if cur.execute(query) else []
    trades = [dict(r) for r in cur.fetchall()]
    
    # Re-execute properly
    cur.execute(query)
    trades = [dict(r) for r in cur.fetchall()]
    conn.close()
    return trades


def analyze_by_symbol(trades):
    """Per-symbol performance breakdown."""
    symbols = defaultdict(lambda: {
        'trades': 0, 'wins': 0, 'losses': 0,
        'total_pnl': 0.0, 'pnl_list': [],
        'win_pnl': 0.0, 'loss_pnl': 0.0,
        'hold_durations': [], 'mfe_list': [], 'mae_list': [],
        'directions': defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0}),
        'streaks': {'current': 0, 'max_win': 0, 'max_loss': 0},
    })
    
    for t in trades:
        sym = t['symbol']
        s = symbols[sym]
        pnl = t.get('pnl', 0) or 0
        
        s['trades'] += 1
        s['total_pnl'] += pnl
        s['pnl_list'].append(pnl)
        
        is_win = pnl > 0
        if is_win:
            s['wins'] += 1
            s['win_pnl'] += pnl
        else:
            s['losses'] += 1
            s['loss_pnl'] += abs(pnl)
        
        # Streak tracking
        if is_win:
            if s['streaks']['current'] >= 0:
                s['streaks']['current'] += 1
            else:
                s['streaks']['current'] = 1
            s['streaks']['max_win'] = max(s['streaks']['max_win'], s['streaks']['current'])
        else:
            if s['streaks']['current'] <= 0:
                s['streaks']['current'] -= 1
            else:
                s['streaks']['current'] = -1
            s['streaks']['max_loss'] = max(s['streaks']['max_loss'], abs(s['streaks']['current']))
        
        # Direction
        d = t.get('direction', 'UNKNOWN')
        s['directions'][d]['trades'] += 1
        s['directions'][d]['pnl'] += pnl
        if is_win:
            s['directions'][d]['wins'] += 1
        
        # MFE/MAE
        if t.get('mfe') is not None:
            s['mfe_list'].append(t['mfe'])
        if t.get('mae') is not None:
            s['mae_list'].append(t['mae'])
    
    # Compute derived metrics
    results = {}
    for sym, s in sorted(symbols.items(), key=lambda x: x[1]['total_pnl'], reverse=True):
        win_rate = s['wins'] / s['trades'] if s['trades'] > 0 else 0
        expectancy = s['total_pnl'] / s['trades'] if s['trades'] > 0 else 0
        profit_factor = s['win_pnl'] / s['loss_pnl'] if s['loss_pnl'] > 0 else float('inf')
        avg_win = s['win_pnl'] / s['wins'] if s['wins'] > 0 else 0
        avg_loss = s['loss_pnl'] / s['losses'] if s['losses'] > 0 else 0
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        results[sym] = {
            'trades': s['trades'],
            'wins': s['wins'],
            'losses': s['losses'],
            'win_rate': round(win_rate, 4),
            'total_pnl': round(s['total_pnl'], 4),
            'expectancy': round(expectancy, 4),
            'profit_factor': round(min(profit_factor, 99.9), 2),
            'avg_win': round(avg_win, 4),
            'avg_loss': round(avg_loss, 4),
            'payoff_ratio': round(min(payoff_ratio, 99.9), 2),
            'max_win_streak': s['streaks']['max_win'],
            'max_loss_streak': s['streaks']['max_loss'],
            'directions': {d: dict(v) for d, v in s['directions'].items()},
        }
        
        if s['mfe_list']:
            avg_mfe = sum(s['mfe_list']) / len(s['mfe_list'])
            avg_mae = sum(s['mae_list']) / len(s['mae_list']) if s['mae_list'] else 0
            results[sym]['avg_mfe'] = round(avg_mfe, 4)
            results[sym]['avg_mae'] = round(avg_mae, 4)
            # Edge ratio: avg MFE / avg MAE — higher = better trade management
            results[sym]['edge_ratio'] = round(avg_mfe / abs(avg_mae), 2) if avg_mae != 0 else 0
    
    return results


def analyze_by_time(trades):
    """Time-of-day and day-of-week patterns."""
    hourly = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0})
    daily = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0})
    
    for t in trades:
        ts = t.get('timestamp', '')
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00')) if 'T' in ts else datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            continue
        
        hour = dt.hour
        day = dt.strftime('%A')
        pnl = t.get('pnl', 0) or 0
        is_win = pnl > 0
        
        hourly[hour]['trades'] += 1
        hourly[hour]['pnl'] += pnl
        if is_win:
            hourly[hour]['wins'] += 1
        
        daily[day]['trades'] += 1
        daily[day]['pnl'] += pnl
        if is_win:
            daily[day]['wins'] += 1
    
    return {
        'hourly': {h: {**v, 'win_rate': round(v['wins']/v['trades'], 3) if v['trades'] > 0 else 0, 'pnl': round(v['pnl'], 4)}
                   for h, v in sorted(hourly.items())},
        'daily': {d: {**v, 'win_rate': round(v['wins']/v['trades'], 3) if v['trades'] > 0 else 0, 'pnl': round(v['pnl'], 4)}
                  for d, v in daily.items()},
    }


def analyze_metadata_patterns(trades):
    """Analyze new metadata columns (when populated)."""
    by_exit = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0})
    by_strategy = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0})
    by_regime = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0})
    
    entropy_buckets = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0})
    conviction_buckets = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0})
    
    metadata_present = 0
    
    for t in trades:
        pnl = t.get('pnl', 0) or 0
        is_win = pnl > 0
        
        # Exit reason
        er = t.get('exit_reason')
        if er:
            metadata_present += 1
            by_exit[er]['trades'] += 1
            by_exit[er]['pnl'] += pnl
            if is_win: by_exit[er]['wins'] += 1
        
        # Strategy
        st = t.get('strategy_type')
        if st:
            by_strategy[st]['trades'] += 1
            by_strategy[st]['pnl'] += pnl
            if is_win: by_strategy[st]['wins'] += 1
        
        # Regime
        reg = t.get('regime')
        if reg:
            by_regime[reg]['trades'] += 1
            by_regime[reg]['pnl'] += pnl
            if is_win: by_regime[reg]['wins'] += 1
        
        # Entropy buckets (0-0.5, 0.5-1.0, 1.0-1.5, 1.5-2.0, 2.0+)
        ent = t.get('entropy_score')
        if ent is not None:
            bucket = f"{int(ent * 2) / 2:.1f}-{int(ent * 2) / 2 + 0.5:.1f}"
            entropy_buckets[bucket]['trades'] += 1
            entropy_buckets[bucket]['pnl'] += pnl
            if is_win: entropy_buckets[bucket]['wins'] += 1
        
        # Conviction buckets (0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0)
        conv = t.get('conviction')
        if conv is not None:
            bucket = f"{int(conv * 5) / 5:.1f}-{int(conv * 5) / 5 + 0.2:.1f}"
            conviction_buckets[bucket]['trades'] += 1
            conviction_buckets[bucket]['pnl'] += pnl
            if is_win: conviction_buckets[bucket]['wins'] += 1
    
    def enrich(d):
        return {k: {**v, 'win_rate': round(v['wins']/v['trades'], 3) if v['trades'] > 0 else 0,
                     'expectancy': round(v['pnl']/v['trades'], 4) if v['trades'] > 0 else 0,
                     'pnl': round(v['pnl'], 4)}
                for k, v in sorted(d.items(), key=lambda x: x[1]['pnl'], reverse=True)}
    
    return {
        'metadata_trades_populated': metadata_present,
        'by_exit_reason': enrich(by_exit),
        'by_strategy': enrich(by_strategy),
        'by_regime': enrich(by_regime),
        'by_entropy_bucket': enrich(entropy_buckets),
        'by_conviction_bucket': enrich(conviction_buckets),
    }


def print_report(symbol_stats, time_stats, meta_stats, total_trades):
    """Print formatted console report."""
    print("=" * 70)
    print("  WINNER DNA ANALYSIS — HolonicTrader")
    print(f"  {total_trades} trades analyzed | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    
    # === Per-Symbol ===
    print("\n📊 PER-SYMBOL PERFORMANCE (sorted by PnL)")
    print("-" * 70)
    print(f"{'Symbol':<14} {'Trades':>6} {'WR':>6} {'PnL':>10} {'Exp':>8} {'PF':>6} {'Payoff':>6} {'Streak':>7}")
    print("-" * 70)
    
    for sym, s in symbol_stats.items():
        wr_str = f"{s['win_rate']:.1%}"
        pnl_str = f"${s['total_pnl']:+.2f}"
        exp_str = f"${s['expectancy']:+.4f}"
        pf_str = f"{s['profit_factor']:.1f}"
        pay_str = f"{s['payoff_ratio']:.1f}"
        streak = f"+{s['max_win_streak']}/-{s['max_loss_streak']}"
        tag = " ⭐" if s['win_rate'] > 0.6 and s['total_pnl'] > 0 else ""
        tag += " ❌" if s['total_pnl'] < 0 and s['trades'] > 10 else ""
        print(f"{sym:<14} {s['trades']:>6} {wr_str:>6} {pnl_str:>10} {exp_str:>8} {pf_str:>6} {pay_str:>6} {streak:>7}{tag}")
    
    # MFE/MAE
    mfe_symbols = {s: d for s, d in symbol_stats.items() if 'avg_mfe' in d}
    if mfe_symbols:
        print("\n📈 MFE/MAE EDGE ANALYSIS")
        print("-" * 50)
        print(f"{'Symbol':<14} {'Avg MFE':>10} {'Avg MAE':>10} {'Edge Ratio':>10}")
        print("-" * 50)
        for sym, s in sorted(mfe_symbols.items(), key=lambda x: x[1].get('edge_ratio', 0), reverse=True):
            print(f"{sym:<14} {s['avg_mfe']:>10.4f} {s['avg_mae']:>10.4f} {s.get('edge_ratio', 0):>10.2f}")
    
    # === Direction Bias ===
    print("\n🔄 DIRECTION BIAS (per symbol)")
    print("-" * 60)
    for sym, s in symbol_stats.items():
        dirs = s.get('directions', {})
        if len(dirs) > 1:
            parts = []
            for d, dv in dirs.items():
                wr = dv['wins'] / dv['trades'] if dv['trades'] > 0 else 0
                parts.append(f"{d}: {dv['trades']}t {wr:.0%}WR ${dv['pnl']:+.2f}")
            print(f"  {sym:<14} | {' | '.join(parts)}")
    
    # === Time Patterns ===
    hourly = time_stats.get('hourly', {})
    if hourly:
        print("\n⏰ HOURLY PERFORMANCE (UTC)")
        print("-" * 50)
        best_hours = sorted(hourly.items(), key=lambda x: x[1]['pnl'], reverse=True)[:5]
        worst_hours = sorted(hourly.items(), key=lambda x: x[1]['pnl'])[:3]
        print("  Best hours: ", ", ".join(f"{h}:00 (${d['pnl']:+.2f}, {d['win_rate']:.0%}WR, {d['trades']}t)" for h, d in best_hours))
        print("  Worst hours:", ", ".join(f"{h}:00 (${d['pnl']:+.2f}, {d['win_rate']:.0%}WR, {d['trades']}t)" for h, d in worst_hours))
    
    daily = time_stats.get('daily', {})
    if daily:
        print("\n📅 DAILY PERFORMANCE")
        for day, d in sorted(daily.items(), key=lambda x: x[1]['pnl'], reverse=True):
            print(f"  {day:<10} {d['trades']:>4}t  {d['win_rate']:.0%}WR  ${d['pnl']:+.2f}")
    
    # === Metadata Patterns ===
    if meta_stats['metadata_trades_populated'] > 0:
        print(f"\n🧬 METADATA PATTERNS ({meta_stats['metadata_trades_populated']} trades with metadata)")
        
        for label, key in [("Exit Reason", "by_exit_reason"), ("Strategy", "by_strategy"),
                           ("Regime", "by_regime"), ("Entropy", "by_entropy_bucket"),
                           ("Conviction", "by_conviction_bucket")]:
            data = meta_stats.get(key, {})
            if data:
                print(f"\n  {label}:")
                for k, v in data.items():
                    print(f"    {k:<20} {v['trades']:>4}t  {v['win_rate']:.0%}WR  ${v['pnl']:+.2f}  Exp=${v['expectancy']:+.4f}")
    else:
        print("\n🧬 METADATA: No metadata columns populated yet — will fill as new trades execute.")
    
    # === Summary ===
    total_pnl = sum(s['total_pnl'] for s in symbol_stats.values())
    total_wins = sum(s['wins'] for s in symbol_stats.values())
    total_losses = sum(s['losses'] for s in symbol_stats.values())
    total = total_wins + total_losses
    
    winners = {s: d for s, d in symbol_stats.items() if d['total_pnl'] > 0 and d['trades'] >= 10}
    bleeders = {s: d for s, d in symbol_stats.items() if d['total_pnl'] < 0 and d['trades'] >= 10}
    
    print("\n" + "=" * 70)
    print("  WINNING FORMULA SUMMARY")
    print("=" * 70)
    print(f"  Total PnL: ${total_pnl:+.2f} across {total} trades")
    print(f"  Overall WR: {total_wins/total:.1%}" if total > 0 else "  No trades")
    winner_str = ', '.join(f"{s} (${d['total_pnl']:+.0f})" for s, d in winners.items())
    bleeder_str = ', '.join(f"{s} (${d['total_pnl']:+.0f})" for s, d in bleeders.items())
    print(f"  Winners ({len(winners)}): {winner_str}")
    print(f"  Bleeders ({len(bleeders)}): {bleeder_str}")
    
    if winners:
        avg_winner_wr = sum(d['win_rate'] for d in winners.values()) / len(winners)
        avg_winner_pf = sum(d['profit_factor'] for d in winners.values()) / len(winners)
        print(f"\n  🏆 Winner DNA: Avg WR={avg_winner_wr:.1%}, Avg PF={avg_winner_pf:.1f}")
    print("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Winner DNA Analysis")
    parser.add_argument('--min-trades', type=int, default=1, help='Min trades to include symbol')
    parser.add_argument('--json', action='store_true', help='Output JSON only')
    parser.add_argument('--db', type=str, default=DB_PATH, help='Database path')
    args = parser.parse_args()
    
    trades = load_trades(args.db)
    if not trades:
        print("No trades found in database.")
        sys.exit(1)
    
    symbol_stats = analyze_by_symbol(trades)
    time_stats = analyze_by_time(trades)
    meta_stats = analyze_metadata_patterns(trades)
    
    # Filter by min trades
    if args.min_trades > 1:
        symbol_stats = {s: d for s, d in symbol_stats.items() if d['trades'] >= args.min_trades}
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'total_trades': len(trades),
        'symbol_performance': symbol_stats,
        'time_patterns': time_stats,
        'metadata_patterns': meta_stats,
    }
    
    # Save JSON
    with open('winner_dna_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(symbol_stats, time_stats, meta_stats, len(trades))
        print(f"\n📁 Full report saved to winner_dna_report.json")


if __name__ == '__main__':
    main()
