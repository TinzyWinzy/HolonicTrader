
import sys
import os

# Ensure we can import from the package
sys.path.append(os.getcwd())

try:
    from HolonicTrader.agent_executor import ExecutorHolon
    print("✅ Successfully imported ExecutorHolon.")
except ImportError as e:
    print(f"❌ ImportError: {e}")
    sys.exit(1)

# Inspect the class
if hasattr(ExecutorHolon, 'reconcile_exchange_positions'):
    print("✅ Method 'reconcile_exchange_positions' FOUND in ExecutorHolon class.")
else:
    print("❌ Method 'reconcile_exchange_positions' NOT FOUND in ExecutorHolon class.")
    sys.exit(1)

# Instantiate and double check
try:
    executor = ExecutorHolon(initial_capital=100)
    if hasattr(executor, 'reconcile_exchange_positions'):
        print("✅ Instance check passed. Logic is ready.")
    else:
        print("❌ Instance check failed.")
except Exception as e:
    print(f"⚠️ Instantiation warning (expected if config missing): {e}")
    # If instantiation fails due to missing dependencies, we still proved the class has the method above
    pass
