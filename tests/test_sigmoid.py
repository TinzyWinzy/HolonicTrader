"""Test the new sigmoid threshold"""
import math

threshold = 0.75
k = 5.0

print("=" * 60)
print("SIGMOID THRESHOLD TEST (threshold=0.75)")
print("=" * 60)
print()

test_entropies = [0.5, 0.60, 0.67, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0, 1.5, 1.85]

print(f"{'Entropy':>10s} {'Autonomy':>10s} {'Action':>10s} {'Expected':>15s}")
print("-" * 60)

for ent in test_entropies:
    autonomy = 1.0 / (1.0 + math.exp(k * (ent - threshold)))
    
    if autonomy > 0.6:
        action = "EXECUTE"
    elif autonomy < 0.4:
        action = "HALT"
    else:
        action = "REDUCE"
    
    # Determine regime
    if ent < 0.67:
        regime = "ORDERED"
    elif ent > 0.80:
        regime = "CHAOTIC"
    else:
        regime = "TRANSITION"
    
    print(f"{ent:>10.2f} {autonomy:>10.4f} {action:>10s} {regime:>15s}")

print()
print("=" * 60)
print("EXPECTED DISTRIBUTION (based on live data):")
print("  ORDERED (< 0.67):    ~53% -> EXECUTE")
print("  TRANSITION (0.67-0.80): ~32% -> REDUCE")
print("  CHAOTIC (> 0.80):    ~15% -> HALT")
print("=" * 60)
