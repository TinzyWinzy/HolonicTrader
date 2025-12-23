"""
Validate Ledger Logic - Check HALT/REDUCE Triggers
Analyzes the decision-making logic and ledger data
"""

import sqlite3
from datetime import datetime

conn = sqlite3.connect('holonic_trader.db')
cursor = conn.cursor()

print("=" * 70)
print("LEDGER LOGIC VALIDATION")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
print()

# 1. Current Ledger Distribution
print("1. CURRENT LEDGER DISTRIBUTION")
print("-" * 70)
cursor.execute("SELECT action, COUNT(*) FROM ledger GROUP BY action")
actions = cursor.fetchall()
total = sum(count for _, count in actions)

for action, count in actions:
    pct = (count / total) * 100
    print(f"  {action:10s}: {count:6d} ({pct:5.2f}%)")

print(f"  {'TOTAL':10s}: {total:6d}")
print()

# 2. Regime Distribution
print("2. REGIME DISTRIBUTION")
print("-" * 70)
cursor.execute("SELECT regime, COUNT(*) FROM ledger GROUP BY regime")
regimes = cursor.fetchall()

for regime, count in regimes:
    pct = (count / total) * 100
    print(f"  {regime:12s}: {count:6d} ({pct:5.2f}%)")
print()

# 3. Entropy Analysis by Action
print("3. ENTROPY ANALYSIS BY ACTION")
print("-" * 70)
cursor.execute("""
    SELECT action, 
           AVG(entropy_score) as avg_entropy,
           MIN(entropy_score) as min_entropy,
           MAX(entropy_score) as max_entropy
    FROM ledger 
    GROUP BY action
""")

results = cursor.fetchall()
for action, avg_ent, min_ent, max_ent in results:
    print(f"  {action}:")
    print(f"    Average: {avg_ent:.4f}")
    print(f"    Range:   {min_ent:.4f} - {max_ent:.4f}")
print()

# 4. Check Threshold Logic
print("4. THRESHOLD ANALYSIS")
print("-" * 70)
print("Current Thresholds (from Phase 10 recalibration):")
print("  ORDERED:    < 0.67")
print("  TRANSITION: 0.67 - 0.80")
print("  CHAOTIC:    > 0.80")
print()

# Check if HALT/REDUCE are triggered in correct regimes
cursor.execute("""
    SELECT regime, action, COUNT(*) 
    FROM ledger 
    WHERE action IN ('HALT', 'REDUCE')
    GROUP BY regime, action
""")

halt_reduce = cursor.fetchall()
if halt_reduce:
    print("HALT/REDUCE by Regime:")
    for regime, action, count in halt_reduce:
        print(f"  {regime:12s} -> {action:6s}: {count}")
else:
    print("⚠️  NO HALT/REDUCE ACTIONS FOUND")
print()

# 5. Entropy Distribution for HALT/REDUCE
print("5. ENTROPY VALUES FOR HALT/REDUCE ACTIONS")
print("-" * 70)
cursor.execute("""
    SELECT entropy_score, regime, action, timestamp
    FROM ledger
    WHERE action IN ('HALT', 'REDUCE')
    ORDER BY timestamp DESC
    LIMIT 10
""")

halt_reduce_entries = cursor.fetchall()
if halt_reduce_entries:
    print(f"{'Entropy':>10s} {'Regime':>12s} {'Action':>8s} {'Timestamp':>25s}")
    print("-" * 70)
    for entropy, regime, action, ts in halt_reduce_entries:
        print(f"{entropy:>10.4f} {regime:>12s} {action:>8s} {ts:>25s}")
else:
    print("⚠️  NO HALT/REDUCE ENTRIES IN RECENT HISTORY")
print()

# 6. Validate Decision Logic
print("6. DECISION LOGIC VALIDATION")
print("-" * 70)

# Get sample of high entropy values
cursor.execute("""
    SELECT entropy_score, regime, action
    FROM ledger
    WHERE entropy_score > 0.75
    ORDER BY entropy_score DESC
    LIMIT 5
""")

high_entropy = cursor.fetchall()
if high_entropy:
    print("High Entropy Samples (> 0.75):")
    for entropy, regime, action in high_entropy:
        expected_regime = "CHAOTIC" if entropy > 0.80 else "TRANSITION"
        regime_ok = "✓" if regime == expected_regime else "✗"
        print(f"  Entropy: {entropy:.4f} | Regime: {regime:12s} {regime_ok} | Action: {action}")
else:
    print("No high entropy values found (all < 0.75)")
print()

# 7. Check Executor Decision Logic
print("7. EXECUTOR DECISION MAPPING")
print("-" * 70)
print("Expected behavior (from agent_executor.py):")
print("  Autonomy > 0.6  -> EXECUTE")
print("  Autonomy < 0.4  -> HALT")
print("  0.4 <= Autonomy <= 0.6 -> REDUCE")
print()
print("Autonomy calculation:")
print("  autonomy = 1.0 / (1.0 + exp(5.0 * (entropy - 2.0)))")
print()

# Calculate autonomy for sample entropy values
import math
test_entropies = [0.5, 0.67, 0.75, 0.80, 0.90, 1.0, 1.5, 2.0]
print("Sample Autonomy Values:")
print(f"{'Entropy':>10s} {'Autonomy':>10s} {'Expected Action':>15s}")
print("-" * 40)
for ent in test_entropies:
    autonomy = 1.0 / (1.0 + math.exp(5.0 * (ent - 2.0)))
    if autonomy > 0.6:
        action = "EXECUTE"
    elif autonomy < 0.4:
        action = "HALT"
    else:
        action = "REDUCE"
    print(f"{ent:>10.2f} {autonomy:>10.4f} {action:>15s}")

print()
print("=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

# Summary
execute_pct = next((count/total*100 for action, count in actions if action == 'EXECUTE'), 0)
halt_pct = next((count/total*100 for action, count in actions if action == 'HALT'), 0)
reduce_pct = next((count/total*100 for action, count in actions if action == 'REDUCE'), 0)

if halt_pct < 1 and reduce_pct < 1:
    print("⚠️  WARNING: HALT/REDUCE triggers appear inactive")
    print(f"   HALT: {halt_pct:.2f}%, REDUCE: {reduce_pct:.2f}%")
    print()
    print("DIAGNOSIS:")
    print("  The sigmoid threshold (entropy=2.0) is too high for live data.")
    print("  Live entropy max is ~1.85, so autonomy never drops below 0.6")
    print()
    print("RECOMMENDATION:")
    print("  Adjust sigmoid threshold from 2.0 to 0.75 in agent_executor.py")
    print("  This will make the system more sensitive to entropy changes")
else:
    print("✓ HALT/REDUCE triggers are active")
    print(f"  HALT: {halt_pct:.2f}%, REDUCE: {reduce_pct:.2f}%")

conn.close()
