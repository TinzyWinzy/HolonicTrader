"""
AEGIS QUANTSEC - ExecutorHolon Integration Patch

Integrates LogIntegrityManager with existing ExecutorHolon AuditLedger
to provide tamper-evident execution logging.

Usage:
    # In your ExecutorHolon initialization:
    from HolonicTrader.log_integrity import LogIntegrityManager, ExecutorLedgerIntegrator
    
    # Create integrity manager
    self.integrity_manager = LogIntegrityManager(
        storage_path="logs/execution_integrity.json",
        auto_anchor_interval=100
    )
    
    # Wrap with integrator
    self.ledger_integrator = ExecutorLedgerIntegrator(
        executor_holon=self,
        integrity_manager=self.integrity_manager
    )
    
    # Use in trade execution
    self.ledger_integrator.log_position_open(...)
"""

import time
from typing import Optional, Any, Dict
from datetime import datetime, timezone


def patch_executor_holon(executor_holon):
    """
    Patch an existing ExecutorHolon instance with integrity logging.
    
    Args:
        executor_holon: ExecutorHolon instance to patch
    
    Returns:
        ExecutorLedgerIntegrator instance
    """
    from HolonicTrader.log_integrity import LogIntegrityManager, ExecutorLedgerIntegrator
    
    # Create integrity manager
    executor_holon.integrity_manager = LogIntegrityManager(
        storage_path="logs/execution_integrity.json",
        auto_anchor_interval=100,  # Anchor every 100 events
        enable_tamper_detection=True
    )
    
    # Create integrator
    executor_holon.ledger_integrator = ExecutorLedgerIntegrator(
        executor_holon=executor_holon,
        integrity_manager=executor_holon.integrity_manager
    )
    
    print(f"[{executor_holon.name}] 🔒 AEGIS Log Integrity Engine initialized")
    
    return executor_holon.ledger_integrator


def integrity_logged_trade_execution(original_execute_method):
    """
    Decorator for wrapping trade execution methods with integrity logging.
    
    Usage:
        @integrity_logged_trade_execution
        def execute_trade(self, signal):
            # Original execution logic
            ...
    """
    def wrapper(self, *args, **kwargs):
        # Extract signal info for logging
        signal = args[0] if args else kwargs.get('signal')
        
        if signal and hasattr(signal, 'symbol'):
            # Log pre-execution
            self.integrity_manager.log_event(
                event_type="EXECUTION_START",
                symbol=signal.symbol,
                data={
                    'action': signal.direction,
                    'size': signal.size,
                    'conviction': signal.conviction,
                    'timestamp': time.time()
                }
            )
        
        # Execute original method
        result = original_execute_method(self, *args, **kwargs)
        
        # Log post-execution
        if result:
            self.integrity_manager.log_trade(
                symbol=signal.symbol if signal else "UNKNOWN",
                action=result.get('action', 'TRADE'),
                quantity=result.get('quantity', 0),
                price=result.get('price', 0),
                order_id=result.get('order_id', 'unknown')
            )
        
        return result
    
    return wrapper


def integrity_verified_sync(original_sync_method):
    """
    Decorator for verifying integrity after exchange sync.
    
    Usage:
        @integrity_verified_sync
        def sync_with_exchange(self, mode='SOFT'):
            # Original sync logic
            ...
    """
    def wrapper(self, *args, **kwargs):
        # Execute original sync
        result = original_sync_method(self, *args, **kwargs)
        
        # Verify integrity post-sync
        if hasattr(self, 'integrity_manager'):
            is_valid, violations = self.integrity_manager.verify_integrity()
            
            if not is_valid:
                # Log critical integrity violation
                self.integrity_manager.log_error(
                    error_type="INTEGRITY_VIOLATION",
                    message=f"Detected {len(violations)} integrity violations after sync",
                    symbol="SYSTEM",
                    traceback=str([v.violation_type for v in violations])
                )
                
                # Trigger alert (if Telegram configured)
                if hasattr(self, 'telegram_bot') and self.telegram_bot:
                    from HolonicTrader.log_integrity import IntegrityAlertHandler
                    alert_handler = IntegrityAlertHandler(
                        integrity_manager=self.integrity_manager,
                        telegram_bot=self.telegram_bot,
                        chat_id=getattr(self, 'telegram_chat_id', '')
                    )
                    alert_handler.check_and_alert()
        
        return result
    
    return wrapper


# =============================================================================
# PATCH HELPER FOR EXISTING CODE
# =============================================================================

def apply_integrity_patches():
    """
    Apply all integrity patches to ExecutorHolon class.
    
    Call this once at application startup.
    """
    try:
        from HolonicTrader.agent_executor import ExecutorHolon
        
        # Store original methods
        ExecutorHolon._original_execute_trade = getattr(ExecutorHolon, 'execute_trade', None)
        ExecutorHolon._original_sync = getattr(ExecutorHolon, 'sync_with_exchange', None)
        
        # Note: Actual method patching requires the methods to exist
        # This is a template for manual integration
        
        print("[AEGIS] ✅ Integrity patch templates loaded")
        print("[AEGIS] ℹ️  Manual integration required - see patch_executor_holon()")
        
    except ImportError as e:
        print(f"[AEGIS] ⚠️ Could not import ExecutorHolon: {e}")


# =============================================================================
# STARTUP INTEGRATION EXAMPLE
# =============================================================================

def initialize_with_integrity(
    executor_holon,
    enable_telegram_alerts: bool = True,
    telegram_bot=None,
    chat_id: str = None
):
    """
    Complete initialization of integrity logging for ExecutorHolon.
    
    Args:
        executor_holon: ExecutorHolon instance
        enable_telegram_alerts: Whether to send Telegram alerts on violations
        telegram_bot: Telegram bot instance
        chat_id: Telegram chat ID
    
    Returns:
        ExecutorLedgerIntegrator instance
    """
    # Patch executor
    integrator = patch_executor_holon(executor_holon)
    
    # Setup Telegram alerts if enabled
    if enable_telegram_alerts and telegram_bot and chat_id:
        from HolonicTrader.log_integrity import IntegrityAlertHandler
        
        executor_holon.integrity_alert_handler = IntegrityAlertHandler(
            integrity_manager=executor_holon.integrity_manager,
            telegram_bot=telegram_bot,
            chat_id=chat_id
        )
        
        print(f"[{executor_holon.name}] 📱 Telegram integrity alerts enabled")
    
    return integrator


# =============================================================================
# PERIODIC INTEGRITY CHECK TASK
# =============================================================================

def create_integrity_check_task(executor_holon, check_interval_sec: int = 300):
    """
    Create a periodic task to check log integrity.
    
    Args:
        executor_holon: ExecutorHolon instance
        check_interval_sec: How often to check (default 5 minutes)
    
    Returns:
        Callable task function for scheduler
    """
    import threading
    
    def check_task():
        """Periodic integrity check."""
        while True:
            time.sleep(check_interval_sec)
            
            if not hasattr(executor_holon, 'integrity_manager'):
                continue
            
            try:
                # Verify integrity
                is_valid, violations = executor_holon.integrity_manager.verify_integrity()
                
                if not is_valid:
                    print(f"[AEGIS] 🚨 INTEGRITY CHECK FAILED: {len(violations)} violations")
                    
                    # Send alert if handler available
                    if hasattr(executor_holon, 'integrity_alert_handler'):
                        executor_holon.integrity_alert_handler.check_and_alert()
                
                # Create periodic anchor
                executor_holon.integrity_manager.create_anchor()
                
            except Exception as e:
                print(f"[AEGIS] ❌ Integrity check error: {e}")
    
    # Start background thread
    task_thread = threading.Thread(target=check_task, daemon=True)
    task_thread.start()
    
    print(f"[AEGIS] 🔄 Periodic integrity check started (interval: {check_interval_sec}s)")
    
    return task_thread


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

"""
Example integration in main_live_phase4.py:

```python
from HolonicTrader.log_integrity import initialize_with_integrity, create_integrity_check_task

# ... after creating executor_holon ...

# Initialize integrity logging
integrator = initialize_with_integrity(
    executor_holon=executor,
    enable_telegram_alerts=True,
    telegram_bot=telegram_bot,  # Your existing bot
    chat_id=TELEGRAM_CHAT_ID
)

# Start periodic checks
create_integrity_check_task(executor, check_interval_sec=300)

# Now all trade executions are logged with tamper-evident hashing
# Violations will trigger Telegram alerts
```
"""
