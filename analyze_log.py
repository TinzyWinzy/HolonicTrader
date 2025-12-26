import re

log_file = "live_trading_session_20251225_084933.log"
print(f"Analyzing {log_file}...")

with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

print(f"Total Lines: {len(lines)}")

keywords = ["Error", "Exception", "Traceback", "fair weather", "SOLVENCY", "PANIC", "nan"]

for i, line in enumerate(lines):
    # Check keywords
    for kw in keywords:
        if kw.lower() in line.lower():
            print(f"Line {i+1}: {line.strip()}")
            
    # Check for weird concatenation (e.g. timestamp in middle of line)
    # "[2025" occurring not at start
    if "[2025" in line and not line.strip().startswith("[2025"):
        print(f"Possible Corruption Line {i+1}: {line.strip()}")

print("\n--- Last 10 Lines ---")
for line in lines[-10:]:
    print(line.strip())
