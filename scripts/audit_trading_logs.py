"""
Trading Log Audit - Loss Analysis & System Reactivity

Analyzes recent trading logs to understand:
1. Loss patterns and root causes
2. System response to losses
3. ML prediction accuracy
4. Exit timing effectiveness
"""
import glob
import re
from datetime import datetime
from collections import defaultdict

print("=" * 70)
print("TRADING LOG AUDIT - LOSS ANALYSIS & SYSTEM REACTIVITY")
print("=" * 70)

# Find all trading logs
log_files = sorted(glob.glob('live_trading_session_*.log'))
print(f"\nFound {len(log_files)} trading session logs")

# Parse trades from logs
trades = []
exits = []
ml_predictions = []
ml_exits = []
atlas_decisions = []
bridge_overrides = []

for log_file in log_files[-5:]:  # Last 5 sessions
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            
            # Parse ML predictions
            if '🤖 ML HIGH CONFIDENCE' in line or '🤖 ML MODERATE' in line or '🤖 ML LOW CONFIDENCE' in line:
                match = re.search(r'🤖 ML (\w+ CONFIDENCE|VERY LOW): (\d+\.\d+)% win prob', line)
                if match:
                    ml_predictions.append({
                        'timestamp': line.split(']')[0].lstrip('['),
                        'confidence': match.group(1),
                        'win_prob': float(match.group(2)),
                        'log': line[:200]
                    })
            
            # Parse ML-Atlas Bridge decisions
            if '🤖🗺️ ML-ATLAS BRIDGE' in line:
                bridge_overrides.append({
                    'timestamp': line.split(']')[0].lstrip('['),
                    'log': line[:200]
                })
            
            # Parse ML Exit recommendations
            if '🤖 ML EXIT:' in line:
                match = re.search(r'🤖 ML EXIT: (\S+) - (\w+) \((.+?)\)', line)
                if match:
                    ml_exits.append({
                        'timestamp': line.split(']')[0].lstrip('['),
                        'symbol': match.group(1),
                        'recommendation': match.group(2),
                        'reason': match.group(3),
                        'log': line[:200]
                    })
            
            # Parse Atlas decisions
            if '[Atlas] Trade APPROVED' in line or '[Atlas] VETO' in line:
                atlas_decisions.append({
                    'timestamp': line.split(']')[0].lstrip('['),
                    'log': line[:200]
                })
            
            # Parse EXIT trades
            if '📉 EXIT' in line or 'Realized' in line:
                match = re.search(r'📉 EXIT (\S+): Realized \$([-.\d]+)', line)
                if match:
                    exits.append({
                        'timestamp': line.split(']')[0].lstrip('['),
                        'symbol': match.group(1),
                        'pnl_usd': float(match.group(2)),
                        'log': line[:200]
                    })
            
            # Parse Trade Logged (Governor)
            if 'Trade Logged' in line and 'Governor' in line:
                match = re.search(r'Trade Logged: (\S+) Profit: ([-.\d]+)%', line)
                if match:
                    trades.append({
                        'timestamp': line.split(']')[0].lstrip('['),
                        'symbol': match.group(1),
                        'pnl_percent': float(match.group(2)),
                        'is_loss': float(match.group(2)) < 0,
                        'log': line[:200]
                    })

print("\n" + "=" * 70)
print("TRADE ANALYSIS")
print("=" * 70)

print(f"\nTotal Trades Logged: {len(trades)}")
if trades:
    wins = [t for t in trades if not t['is_loss']]
    losses = [t for t in trades if t['is_loss']]
    
    print(f"  Wins: {len(wins)} ({len(wins)/len(trades)*100:.1f}%)")
    print(f"  Losses: {len(losses)} ({len(losses)/len(trades)*100:.1f}%)")
    
    if losses:
        print(f"\nLoss Breakdown:")
        small_losses = [l for l in losses if -2 <= l['pnl_percent'] < 0]
        medium_losses = [l for l in losses if -5 <= l['pnl_percent'] < -2]
        large_losses = [l for l in losses if l['pnl_percent'] < -5]
        
        print(f"  Small (-2% to 0%): {len(small_losses)} ({len(small_losses)/len(losses)*100:.1f}%)")
        print(f"  Medium (-5% to -2%): {len(medium_losses)} ({len(medium_losses)/len(losses)*100:.1f}%)")
        print(f"  Large (<-5%): {len(large_losses)} ({len(large_losses)/len(losses)*100:.1f}%)")
        
        print(f"\nLoss Statistics:")
        if losses:
            avg_loss = sum(l['pnl_percent'] for l in losses) / len(losses)
            max_loss = min(l['pnl_percent'] for l in losses)
            print(f"  Average Loss: {avg_loss:.2f}%")
            print(f"  Max Loss: {max_loss:.2f}%")
            print(f"  Total Loss: {sum(l['pnl_percent'] for l in losses):.2f}%")

print("\n" + "=" * 70)
print("ML PREDICTION ANALYSIS")
print("=" * 70)

print(f"\nML Predictions Made: {len(ml_predictions)}")
if ml_predictions:
    high_conf = [p for p in ml_predictions if 'HIGH' in p['confidence']]
    mod_conf = [p for p in ml_predictions if 'MODERATE' in p['confidence']]
    low_conf = [p for p in ml_predictions if 'LOW' in p['confidence'] or 'VERY LOW' in p['confidence']]
    
    print(f"  High Confidence (>60%): {len(high_conf)} ({len(high_conf)/len(ml_predictions)*100:.1f}%)")
    print(f"  Moderate (50-60%): {len(mod_conf)} ({len(mod_conf)/len(ml_predictions)*100:.1f}%)")
    print(f"  Low (<50%): {len(low_conf)} ({len(low_conf)/len(ml_predictions)*100:.1f}%)")
    
    print(f"\nWin Probability Distribution:")
    avg_win_prob = sum(p['win_prob'] for p in ml_predictions) / len(ml_predictions)
    print(f"  Average Predicted Win Prob: {avg_win_prob:.1f}%")
    print(f"  Min: {min(p['win_prob'] for p in ml_predictions):.1f}%")
    print(f"  Max: {max(p['win_prob'] for p in ml_predictions):.1f}%")

print("\n" + "=" * 70)
print("ML-ATLAS BRIDGE ANALYSIS")
print("=" * 70)

print(f"\nBridge Activations: {len(bridge_overrides)}")
if bridge_overrides:
    agreements = [b for b in bridge_overrides if 'Strong agreement' in b['log'] or 'Moderate agreement' in b['log']]
    overrides = [b for b in bridge_overrides if 'override' in b['log'].lower()]
    
    print(f"  Agreements: {len(agreements)}")
    print(f"  Overrides: {len(overrides)}")
    
    if overrides:
        print(f"\nOverride Examples:")
        for override in overrides[:3]:
            print(f"  {override['timestamp']}: {override['log'][:150]}")

print("\n" + "=" * 70)
print("ML EXIT OPTIMIZER ANALYSIS")
print("=" * 70)

print(f"\nExit Recommendations: {len(ml_exits)}")
if ml_exits:
    rec_counts = defaultdict(int)
    for e in ml_exits:
        rec_counts[e['recommendation']] += 1
    
    print(f"\nRecommendation Breakdown:")
    for rec, count in sorted(rec_counts.items()):
        print(f"  {rec}: {count} ({count/len(ml_exits)*100:.1f}%)")
    
    print(f"\nReason Categories:")
    reason_categories = defaultdict(int)
    for e in ml_exits:
        if 'loss' in e['reason'].lower():
            reason_categories['Loss-related'] += 1
        elif 'profit' in e['reason'].lower():
            reason_categories['Profit-taking'] += 1
        elif 'hold' in e['reason'].lower() or 'recent' in e['reason'].lower():
            reason_categories['Hold/Monitor'] += 1
        else:
            reason_categories['Other'] += 1
    
    for cat, count in sorted(reason_categories.items()):
        print(f"  {cat}: {count}")

print("\n" + "=" * 70)
print("ATLAS FILTER ANALYSIS")
print("=" * 70)

print(f"\nAtlas Decisions: {len(atlas_decisions)}")
if atlas_decisions:
    approvals = [a for a in atlas_decisions if 'APPROVED' in a['log']]
    vetos = [a for a in atlas_decisions if 'VETO' in a['log']]
    
    print(f"  Approvals: {len(approvals)}")
    print(f"  Vetos: {len(vetos)}")
    
    if vetos:
        print(f"\nVeto Reasons:")
        veto_reasons = defaultdict(int)
        for v in vetos:
            if 'INSUFFICIENT_VOLATILITY' in v['log']:
                veto_reasons['Low Volatility'] += 1
            elif 'BLACKLIST' in v['log']:
                veto_reasons['Blacklist'] += 1
            elif 'SPREAD' in v['log']:
                veto_reasons['Spread'] += 1
            else:
                veto_reasons['Other'] += 1
        
        for reason, count in sorted(veto_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

print("\n" + "=" * 70)
print("LOSS ROOT CAUSE ANALYSIS")
print("=" * 70)

if losses:
    print("\nLoss Patterns:")
    
    # Check if losses correlate with low ML confidence
    low_conf_losses = 0
    for loss in losses:
        # Check if there was a low confidence ML prediction around the same time
        for pred in ml_predictions:
            if pred['win_prob'] < 50:
                low_conf_losses += 1
                break
    
    print(f"  Losses with Low ML Confidence: {low_conf_losses} ({low_conf_losses/len(losses)*100:.1f}%)")
    
    # Check exit timing
    early_exits = [e for e in ml_exits if 'CUT_LOSS' in e['recommendation']]
    print(f"  ML Recommended Early Exits: {len(early_exits)}")
    
    print("\nRecommendations:")
    if low_conf_losses > 0:
        print(f"  ⚠️  {low_conf_losses} losses occurred with low ML confidence - consider stricter entry filter")
    
    if len(early_exits) < len(losses) * 0.5:
        print(f"  ⚠️  Exit optimizer may be too passive - consider lowering urgency threshold")
    
    if len(vetos) > 0 if 'vetos' in dir() else False:
        print(f"  ℹ️  Atlas vetoed {len(vetos)} trades - some may have been missed opportunities")

print("\n" + "=" * 70)
print("SYSTEM REACTIVITY SCORE")
print("=" * 70)

# Calculate reactivity score (0-100)
reactivity_components = {
    'ML Coverage': min(100, len(ml_predictions) / max(1, len(trades)) * 100),
    'Exit Monitoring': min(100, len(ml_exits) / max(1, len(trades)) * 100),
    'Bridge Activity': min(100, len(bridge_overrides) / max(1, len(trades)) * 100 * 5),  # Weight bridge higher
}

reactivity_score = sum(reactivity_components.values()) / len(reactivity_components)

print(f"\nReactivity Components:")
for component, score in reactivity_components.items():
    status = '✅' if score > 80 else '⚠️' if score > 50 else '❌'
    print(f"  {status} {component}: {score:.0f}%")

print(f"\n{'✅' if reactivity_score > 80 else '⚠️' if reactivity_score > 50 else '❌'} Overall Reactivity Score: {reactivity_score:.0f}/100")

if reactivity_score < 50:
    print("\n⚠️  System reactivity is low - ML components may not be fully integrated")
elif reactivity_score < 80:
    print("\n⚠️  System reactivity is moderate - some ML components need attention")
else:
    print("\n✅ System reactivity is excellent - ML components actively monitoring")

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
