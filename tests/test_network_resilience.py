import time
import pytest
import ccxt
import sys
import os
from unittest.mock import MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HolonicTrader.network_resilience import (
    with_retry, 
    with_circuit_breaker, 
    CircuitOpenException,
    get_circuit_breaker,
    _circuit_breakers
)

def test_retry_success():
    mock_func = MagicMock(return_value="Success")
    decorated_func = with_retry(max_retries=2, base_delay=0.1, max_delay=1.0)(mock_func)
    
    result = decorated_func()
    
    assert result == "Success"
    assert mock_func.call_count == 1

def test_retry_transient_failure_then_success():
    mock_func = MagicMock(side_effect=[ccxt.NetworkError("Timeout"), "Success"])
    decorated_func = with_retry(max_retries=2, base_delay=0.01, max_delay=0.1)(mock_func)
    
    result = decorated_func()
    
    assert result == "Success"
    assert mock_func.call_count == 2

def test_retry_max_retries_exceeded():
    mock_func = MagicMock(side_effect=ccxt.NetworkError("Timeout"))
    decorated_func = with_retry(max_retries=2, base_delay=0.01, max_delay=0.1)(mock_func)
    
    with pytest.raises(ccxt.NetworkError):
        decorated_func()
        
    assert mock_func.call_count == 3 # 1 initial + 2 retries

def test_circuit_breaker_tripping():
    # Clear global state for test
    _circuit_breakers.clear()
    
    # We set threshold to 2 for quick testing
    endpoint = "test_endpoint"
    mock_func = MagicMock(side_effect=ccxt.NetworkError("Timeout"))
    
    decorated_func = with_circuit_breaker(endpoint, failure_threshold=2, recovery_timeout=0.2)(mock_func)
    
    # 1. First failure
    with pytest.raises(ccxt.NetworkError):
        decorated_func()
        
    # 2. Second failure - Circuit trips OPEN
    with pytest.raises(ccxt.NetworkError):
        decorated_func()
        
    cb = get_circuit_breaker(endpoint)
    assert cb.state == 'OPEN'
    
    # 3. Third call should fail fast without invoking mock_func
    with pytest.raises(CircuitOpenException):
        decorated_func()
        
    assert mock_func.call_count == 2 # mock_func not called on 3rd attempt
    
    # Wait for recovery timeout
    time.sleep(0.3)
    
    # 4. Half-Open state test
    assert cb.state == 'OPEN' # The state doesn't change until requested
    assert cb.can_execute() == True
    assert cb.state == 'HALF-OPEN'
    
    # Let's make it succeed this time
    mock_func.side_effect = None
    mock_func.return_value = "Recovered"
    
    result = decorated_func()
    
    assert result == "Recovered"
    assert cb.state == 'CLOSED'
    assert mock_func.call_count == 3

def test_combined_decorators():
    _circuit_breakers.clear()
    endpoint = "combined_endpoint"
    
    # Here the retry triggers first, THEN circuit breaker trips if retry exhausts
    # So 1 call = 3 mock executions (initial + 2 retries)
    mock_func = MagicMock(side_effect=ccxt.NetworkError("Timeout"))
    
    @with_circuit_breaker(endpoint, failure_threshold=2, recovery_timeout=1.0)
    @with_retry(max_retries=2, base_delay=0.01, max_delay=0.1)
    def resilient_fetch():
        return mock_func()
        
    # First call - exhausted retries (3 calls to mock)
    with pytest.raises(ccxt.NetworkError):
        resilient_fetch()
        
    cb = get_circuit_breaker(endpoint)
    # Circuit Breaker saw 1 failure from the fully wrapped retry block
    assert cb.failure_count == 1
    assert cb.state == 'CLOSED'
    
    # Second call - exhausted retries again (3 more calls to mock)
    with pytest.raises(ccxt.NetworkError):
        resilient_fetch()

    # Circuit Breaker saw 2 failures. Threshold is 2. Trips OPEN.
    assert cb.state == 'OPEN'
    assert mock_func.call_count == 6 # 2 calls * 3 attempts
    
    with pytest.raises(CircuitOpenException):
        resilient_fetch()
        
    # Still 6 calls to mock, circuit blocked it
    assert mock_func.call_count == 6

if __name__ == "__main__":
    pytest.main([__file__])
