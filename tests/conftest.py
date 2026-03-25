"""
HolonicTrader Test Suite - Shared Fixtures
"""
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def mock_config():
    """Provides a mock config module for testing."""
    import config
    # Save original values
    original = {
        'STRATEGY_RSI_OVERSOLD': getattr(config, 'STRATEGY_RSI_OVERSOLD', 30),
        'STRATEGY_RSI_OVERBOUGHT': getattr(config, 'STRATEGY_RSI_OVERBOUGHT', 70),
        'SATELLITE_STOP_LOSS': getattr(config, 'SATELLITE_STOP_LOSS', 0.05),
        'SATELLITE_TAKE_PROFIT_1': getattr(config, 'SATELLITE_TAKE_PROFIT_1', 0.10),
    }
    yield config
    # Restore original values
    for key, value in original.items():
        setattr(config, key, value)

@pytest.fixture
def temp_genome_file(tmp_path):
    """Creates a temporary genome file for testing."""
    import json
    genome_path = tmp_path / "live_genome.json"
    
    def create_genome(genome_data, fitness=1.0):
        data = {
            'genome': genome_data,
            'fitness': fitness
        }
        with open(genome_path, 'w') as f:
            json.dump(data, f)
        return str(genome_path)
    
    return create_genome

@pytest.fixture
def mock_db_manager():
    """Provides a mock database manager."""
    class MockDBManager:
        def __init__(self):
            self.db_path = ':memory:'
        def close(self):
            pass
    return MockDBManager()

@pytest.fixture
def mock_market():
    """Provides a mock market holon."""
    class MockMarket:
        def get_balance(self):
            return 1000.0
    return MockMarket()
