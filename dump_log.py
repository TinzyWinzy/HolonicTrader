log_file = "live_trading_session_20251225_084933.log"
with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()
    
start = 500
end = 530
print(f"--- Lines {start} to {end} ---")
for i, line in enumerate(lines[start:end]):
    print(f"{start+i}: {line.strip()}")
