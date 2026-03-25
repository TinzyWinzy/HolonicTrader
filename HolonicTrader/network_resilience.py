import time
import random
import logging
from functools import wraps
from typing import Callable, Any, Type, Tuple, Optional

import ccxt

logger = logging.getLogger(__name__)

class CircuitOpenException(Exception):
    """Raised when a circuit breaker is open."""
    pass

class CircuitBreaker:
    """
    A simple Circuit Breaker that tracks failures.
    States:
    - CLOSED: Normal operation. Requests pass through.
    - OPEN: Threshold exceeded. Requests fail fast.
    - HALF-OPEN: Testing recovery after timeout.
    """
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = 'CLOSED'

    def record_success(self):
        self.failure_count = 0
        self.state = 'CLOSED'

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'

    def can_execute(self) -> bool:
        if self.state == 'CLOSED':
            return True
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = 'HALF-OPEN'
                return True # Allow exactly one test request
            return False
        if self.state == 'HALF-OPEN':
            # In half-open, we only allow one concurrent test request theoretically.
            # For simplicity in a multi-threaded app without complex atomic locks on the breaker state,
            # we'll just let it through and rely on the next success/failure to clamp it.
            return True
        return False

# Global registry of circuit breakers by endpoint key
_circuit_breakers = {}

def get_circuit_breaker(endpoint_key: str, threshold: int = 5, timeout: float = 30.0) -> CircuitBreaker:
    if endpoint_key not in _circuit_breakers:
        _circuit_breakers[endpoint_key] = CircuitBreaker(threshold, timeout)
    return _circuit_breakers[endpoint_key]

def with_retry(
    max_retries: int = 3, 
    base_delay: float = 1.0, 
    max_delay: float = 10.0, 
    exceptions: Tuple[Type[Exception], ...] = (ccxt.NetworkError, ccxt.ExchangeError)
):
    """
    Decorator for adding exponential backoff retries with jitter to functions.
    Specifically handles ccxt.RateLimitExceeded with heavier penalties.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1): # Attempt 0 is the first try
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    # Do not retry if we hit max
                    if attempt == max_retries:
                        break
                    
                    # Log the attempt
                    func_name = getattr(func, '__name__', 'unknown_func')
                    
                    # Calculate delay
                    if isinstance(e, ccxt.RateLimitExceeded) or isinstance(e, ccxt.DDoSProtection):
                        delay = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(2.0, 5.0)
                        logger.warning(f"[Retry] Rate limit hit on {func_name}. Attempt {attempt+1}/{max_retries}. Sleeping {delay:.2f}s...")
                    else:
                        delay = min(max_delay, base_delay * (1.5 ** attempt)) + random.uniform(0.1, 1.0)
                        logger.warning(f"[Retry] Transient error on {func_name}: {e}. Attempt {attempt+1}/{max_retries}. Sleeping {delay:.2f}s...")
                        
                    time.sleep(delay)
                    
            # If we exhausted retries, raise the last exception
            raise last_exception if last_exception else Exception("Retry failed without exception")
        return wrapper
    return decorator

def with_circuit_breaker(
    endpoint_key: str, 
    failure_threshold: int = 5, 
    recovery_timeout: float = 30.0,
    fallback_value: Any = None
):
    """
    Decorator that applies a circuit breaker to protect against cascading failures.
    If the circuit trips, it optionally returns a fallback_value or raises CircuitOpenException.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cb = get_circuit_breaker(endpoint_key, failure_threshold, recovery_timeout)
            
            if not cb.can_execute():
                logger.warning(f"[CircuitBreaker] {endpoint_key} is OPEN. Failing fast.")
                if fallback_value is not None:
                    return fallback_value
                raise CircuitOpenException(f"Circuit {endpoint_key} is open")
                
            try:
                result = func(*args, **kwargs)
                if cb.state in ('OPEN', 'HALF-OPEN'):
                    logger.info(f"[CircuitBreaker] {endpoint_key} recovered. Closing circuit.")
                cb.record_success()
                return result
            except Exception as e:
                # We record failure on ANY exception that bubbles up here.
                # If combined with @with_retry, those retries happen *inside* the circuit breaker.
                # The circuit breaker only trips when all retries fail.
                cb.record_failure()
                if cb.state == 'OPEN':
                    logger.error(f"[CircuitBreaker] {endpoint_key} tripped OPEN due to repeated failures! Last Error: {e}")
                raise e
        return wrapper
    return decorator
