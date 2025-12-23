"""
Validate new thresholds against historical ledger data
"""

import sqlite3

conn = sqlite3.connect('holonic_trader.db')
cursor = conn.cursor()

# Get all entropy scores
cursor.execute("SELECT entropy_score FROM ledger")
scores = [row[0] for row in cursor.fetchall()]

# Apply new thresholds
ORDERED_THRESHOLD = 0.69
CHAOTIC_THRESHOLD = 0.81

ordered_count = sum(1 for s in scores if s < ORDERED_THRESHOLD)
chaotic_count = sum(1 for s in scores if s > CHAOTIC_THRESHOLD)
transition_count = sum(1 for s in scores if ORDERED_THRESHOLD <= s <= CHAOTIC_THRESHOLD)

total = len(scores)

print("=" * 60)
print("VALIDATION: New Thresholds vs Historical Data")
print("=" * 60)
print(f"\nTotal Samples: {total}")
print(f"\nNew Thresholds:")
print(f"  ORDERED:    < {ORDERED_THRESHOLD}")
print(f"  TRANSITION: {ORDERED_THRESHOLD} - {CHAOTIC_THRESHOLD}")
print(f"  CHAOTIC:    > {CHAOTIC_THRESHOLD}")
print(f"\nPredicted Distribution:")
print(f"  ORDERED:    {ordered_count} ({ordered_count/total*100:.1f}%)")
print(f"  TRANSITION: {transition_count} ({transition_count/total*100:.1f}%)")
print(f"  CHAOTIC:    {chaotic_count} ({chaotic_count/total*100:.1f}%)")
print(f"\nTarget Distribution:")
print(f"  ORDERED:    60%")
print(f"  TRANSITION: 30%")
print(f"  CHAOTIC:    10%")
print("\n" + "=" * 60)

# Check if close to target
ordered_pct = ordered_count/total*100
transition_pct = transition_count/total*100
chaotic_pct = chaotic_count/total*100

if 55 <= ordered_pct <= 65 and 25 <= transition_pct <= 35 and 5 <= chaotic_pct <= 15:
    print("✓ VALIDATION PASSED: Distribution matches target")
else:
    print("⚠️  WARNING: Distribution deviates from target")

conn.close()
