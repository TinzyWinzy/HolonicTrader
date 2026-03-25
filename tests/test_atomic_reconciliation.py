import os
import sys
import threading
import time
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HolonicTrader.agent_executor import ExecutorHolon, Position, PositionState
from HolonicTrader.agent_governor import GovernorHolon

def run_test():
    print("Initializing test...")
    gov = GovernorHolon(initial_balance=1000.0)
    exec = ExecutorHolon(initial_capital=1000.0, governor=gov, db_manager=None, market=None)
    gov.executor = exec
    
    # 1. Simulate entry
    print("Simulating entry...")
    exec._update_state_post_fill(
        vk="BTC/USDT",
        fill={"symbol": "BTC/USDT", "filled_qty": 0.5, "price": 50000.0, "direction": "BUY"},
        leverage=1.0,
        strategy="DIRECTIONAL"
    )
    
    # Check governor tracking
    assert "BTC/USDT" in gov.last_specific_entry
    assert gov.last_specific_entry["BTC/USDT"] == 50000.0
    
    # 2. Simulate concurrent access
    errors = []
    
    def reader_thread():
        try:
            for _ in range(100):
                # The property should return a snapshot
                positions = gov.positions
                for sym, pos in positions.items():
                    pass # Just iterate to check for size change error
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)
            
    def writer_thread():
        try:
            for i in range(100):
                exec._update_state_post_fill(
                    vk="ETH/USDT",
                    fill={"symbol": "ETH/USDT", "filled_qty": 0.1, "price": 3000.0 + i, "direction": "BUY"},
                    leverage=1.0,
                    strategy="DIRECTIONAL"
                )
                if i % 5 == 0:
                    exec.purge_position("ETH/USDT")
        except Exception as e:
            errors.append(e)

    print("Running concurrency test...")
    t1 = threading.Thread(target=reader_thread)
    t2 = threading.Thread(target=writer_thread)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    if errors:
        print(f"Concurrency errors: {errors}")
        sys.exit(1)
    else:
        print("Concurrency test passed: no RuntimeError during iteration")
        
    # 3. Simulate exit
    print("Simulating full exit and checking governor state cleanup...")
    exec._update_state_post_fill(
        vk="BTC/USDT",
        fill={"symbol": "BTC/USDT", "filled_qty": 0.5, "price": 51000.0, "direction": "SELL"},
        leverage=1.0,
        strategy="DIRECTIONAL"
    )
    exec.purge_position("ETH/USDT")
    
    # After exit/purge, the positions map should be empty, and governor trackers should be cleaned
    assert "BTC/USDT" not in gov.positions
    assert "ETH/USDT" not in gov.positions
    
    print(f"Governor trackers after exit: {gov.last_specific_entry}")
    assert "BTC/USDT" not in gov.last_specific_entry
    assert "ETH/USDT" not in gov.last_specific_entry
    
    print("All tests passed successfully.")

if __name__ == '__main__':
    run_test()
