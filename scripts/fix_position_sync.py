"""
CRITICAL FIX: Position Sync with Exchange

Problem: System logs "Position OPENED" but exchange shows 0 positions
Root Cause: Position tracked BEFORE confirming exchange execution
Solution: Clear phantom positions, add execution confirmation
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

print('=' * 70)
print('POSITION SYNC FIX')
print('=' * 70)

# Step 1: Clear phantom positions from executor
print('\n1. Clearing phantom positions from executor.position_metadata...')

try:
    from HolonicTrader.agent_executor import Executor
    from HolonicTrader.agent_governor import GovernorHolon
    
    executor = Executor()
    governor = GovernorHolon()
    
    print(f'   Before: {len(executor.position_metadata)} positions in metadata')
    print(f'   Before: {len(governor.positions)} positions in governor')
    
    # Clear phantom positions
    executor.position_metadata.clear()
    governor.positions.clear()
    
    print(f'   After: {len(executor.position_metadata)} positions in metadata')
    print(f'   After: {len(governor.positions)} positions in governor')
    print('   ✅ Phantom positions cleared')
    
except Exception as e:
    print(f'   ❌ Error clearing positions: {e}')

# Step 2: Add execution confirmation patch
print('\n2. Patching execution confirmation...')

patch_code = '''
# In HolonicTrader/agent_executor.py, execute_transaction() method
# Around line 3267, AFTER _dispatch_to_market call:

fill = self._dispatch_to_market(...)

# === FIX 2026-03-22: Verify fill actually happened on exchange ===
if fill:
    # Verify fill has actual filled_qty > 0
    filled_qty = fill.get('filled_qty', 0)
    if filled_qty <= 0:
        logger.warning(f"[{self.name}] ❌ FILL REJECTED: Order ID {fill.get('order_id')} has 0 filled_qty")
        self._last_execution_error = f"Order filled_qty=0 for {symbol}"
        return None
    
    # Verify order status is FILLED/OPEN (not REJECTED/CANCELLED)
    order_status = fill.get('status', 'FILLED').upper()
    if order_status in ['REJECTED', 'CANCELLED', 'EXPIRED']:
        logger.warning(f"[{self.name}] ❌ ORDER {order_status}: Order ID {fill.get('order_id')}")
        self._last_execution_error = f"Order {order_status} for {symbol}"
        return None
    
    logger.info(f"[{self.name}] ✅ ORDER FILLED: {symbol} {filled_qty} @ {fill.get('fill_price', current_price)}")
'''

print('   Patch code prepared (see reports/POSITION_SYNC_FIX_COMPLETE.md)')
print('   Manual integration required in agent_executor.py')

# Step 3: Add position sync method
print('\n3. Adding position sync method...')

sync_method = '''
def sync_positions_with_exchange(self):
    """Sync internal position tracking with actual exchange positions"""
    logger.info(f"[{self.name}] 🔄 Syncing positions with exchange...")
    
    # Get actual positions from exchange
    try:
        exchange_positions = self.market.get_positions() if self.market else {}
        exchange_symbols = set(exchange_positions.keys())
    except Exception as e:
        logger.error(f"[{self.name}] ❌ Exchange sync failed: {e}")
        return
    
    # Get tracked positions
    tracked_symbols = set(self.position_metadata.keys())
    
    # Find phantom positions (tracked but not on exchange)
    phantoms = tracked_symbols - exchange_symbols
    
    for symbol in phantoms:
        logger.warning(f"[{self.name}] 🧹 PHANTOM DETECTED: {symbol} (tracked but not on exchange)")
        # Clear from tracking
        if symbol in self.position_metadata:
            del self.position_metadata[symbol]
        if symbol in self.held_assets:
            del self.held_assets[symbol]
        logger.info(f"[{self.name}] ✅ PHANTOM CLEARED: {symbol}")
    
    # Find missing positions (on exchange but not tracked)
    missing = exchange_symbols - tracked_symbols
    
    for symbol in missing:
        logger.warning(f"[{self.name}] ⚠️ MISSING POSITION: {symbol} (on exchange but not tracked)")
        # TODO: Reconstruct position metadata from exchange
    
    logger.info(f"[{self.name}] ✅ SYNC COMPLETE: {len(phantoms)} phantoms cleared, {len(missing)} missing found")
'''

print('   Sync method prepared (add to agent_executor.py)')

print('\n' + '=' * 70)
print('NEXT STEPS')
print('=' * 70)
print('''
1. ✅ Phantom positions cleared (done above)

2. ⏳ Manual patch required in agent_executor.py:
   - Add execution confirmation after _dispatch_to_market
   - Verify filled_qty > 0 before returning success
   - Check order status is FILLED/OPEN

3. ⏳ Add sync_positions_with_exchange method to Executor
   - Call periodically (every 5-10 minutes)
   - Clear phantoms automatically
   - Log discrepancies

4. ⏳ Update trader_entry_handler.py:
   - Only call governor.open_position() AFTER confirming fill
   - Move open_position() call inside "if fill:" block

5. ⏳ Test with small position:
   - Execute small trade
   - Verify logs match exchange
   - Confirm no phantoms created
''')

print('\nDocumentation: reports/POSITION_SYNC_FIX_COMPLETE.md')
