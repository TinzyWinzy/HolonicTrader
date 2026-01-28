
import sys
import os

# Go up one level to import HolonicTrader modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import config
    print("Config imported successfully.")
except Exception as e:
    print(f"Error importing config: {e}")
    sys.exit(1)

def test_nano_calc():
    print("Testing calculate_nano_position...")
    try:
        # Test 1: Sufficient
        res = config.calculate_nano_position(100.0, 'DOGE/USDT', 0.50)
        print(f"Test 1 (100.0 bal): {res}")
        
        # Test 2: Bump
        res = config.calculate_nano_position(60.0, 'DOGE/USDT', 0.50)
        print(f"Test 2 (60.0 bal): {res}")
        
        # Test 3: Insufficient
        res = config.calculate_nano_position(40.0, 'DOGE/USDT', 0.50)
        print(f"Test 3 (40.0 bal): {res}")
        
    except Exception as e:
        print(f"Error in test_nano_calc: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_nano_calc()
