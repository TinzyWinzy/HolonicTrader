"""
test_signal_stack.py - Verify the Signal Engine Refactor
"""
import sys
import os
import logging

# Add root directory to path
sys.path.append(os.getcwd())

from HolonicTrader.stack_factory import create_signal_stack
import config

logging.basicConfig(level=logging.INFO)

def test_stack_initialization():
    print("Testing Holon Stack Initialization...")
    stack = create_signal_stack(include_overwatch=True)
    
    if not stack:
        print("❌ FAILED: Stack returned None")
        return False
        
    print(f"✅ Stack Initialized: {list(stack.holons.keys())}")
    
    # Test Report Generation (Mocking/Simulating if needed, but we check if method exists)
    if hasattr(stack.signal_provider, 'generate_signal_report'):
        print("✅ SignalProvider ready.")
    else:
        print("❌ FAILED: SignalProvider missing generate_signal_report")
        return False
        
    # Check Governor
    if stack.governor.balance > 0:
        print(f"✅ Governor balance synced: {stack.governor.balance}")
    else:
        print("❌ FAILED: Governor balance check failed")
        
    stack.close()
    return True

if __name__ == "__main__":
    if test_stack_initialization():
        print("\n🏆 Verification Successful!")
    else:
        print("\n❌ Verification Failed!")
        sys.exit(1)
