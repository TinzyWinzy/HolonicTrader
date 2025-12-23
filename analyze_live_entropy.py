"""
Analyze live entropy distribution from ledger
Calculate new thresholds for live trading
"""

import sqlite3

conn = sqlite3.connect('holonic_trader.db')
cursor = conn.cursor()

print("=" * 70)
print("LIVE ENTROPY DISTRIBUTION ANALYSIS")
print("=" * 70)
print()

# Get all entropy scores
cursor.execute("SELECT entropy_score FROM ledger ORDER BY entropy_score")
scores = [row[0] for row in cursor.fetchall()]

if not scores:
    print("No entropy data found!")
    conn.close()
    exit()

print(f"Total Samples: {len(scores)}")
print()

# Calculate statistics
import statistics

mean = statistics.mean(scores)
median = statistics.median(scores)
stdev = statistics.stdev(scores) if len(scores) > 1 else 0
min_score = min(scores)
max_score = max(scores)

print("STATISTICS:")
print(f"  Mean:   {mean:.4f}")
print(f"  Median: {median:.4f}")
print(f"  StdDev: {stdev:.4f}")
print(f"  Min:    {min_score:.4f}")
print(f"  Max:    {max_score:.4f}")
print()

# Calculate percentiles
def percentile(data, p):
    """Calculate percentile of sorted data"""
    k = (len(data) - 1) * p
    f = int(k)
    c = f + 1
    if c >= len(data):
        return data[-1]
    d0 = data[f] * (c - k)
    d1 = data[c] * (k - f)
    return d0 + d1

percentiles = [10, 25, 50, 60, 75, 90, 95]
print("PERCENTILES:")
for p in percentiles:
    value = percentile(scores, p / 100)
    print(f"  {p}th: {value:.4f}")
print()

# Recommended thresholds (60/30/10 split)
threshold_ordered = percentile(scores, 0.60)
threshold_chaotic = percentile(scores, 0.90)

print("=" * 70)
print("RECOMMENDED THRESHOLDS (60/30/10 split)")
print("=" * 70)
print(f"  ORDERED   (< 60th percentile): < {threshold_ordered:.4f}")
print(f"  TRANSITION (60th - 90th):      {threshold_ordered:.4f} - {threshold_chaotic:.4f}")
print(f"  CHAOTIC   (> 90th percentile): > {threshold_chaotic:.4f}")
print()

# Compare with current thresholds
print("CURRENT THRESHOLDS (from Phase 6 backtest):")
print("  ORDERED:    < 1.96")
print("  TRANSITION: 1.96 - 2.10")
print("  CHAOTIC:    > 2.10")
print()

print("=" * 70)
print("ANALYSIS:")
print("=" * 70)

if max_score < 1.96:
    print("⚠️  CRITICAL: All live entropy scores are below the ORDERED threshold!")
    print("    This explains why 100% of decisions were ORDERED.")
    print()
    print("    Live market entropy is LOWER than backtest data.")
    print("    Thresholds must be recalibrated for live conditions.")
else:
    print("✓ Live entropy scores span the threshold range.")

print()
print("=" * 70)

conn.close()
