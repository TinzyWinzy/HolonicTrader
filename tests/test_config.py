import config
import pytest

required_attrs = [
    'ACTIVE_WATCHLIST',
    'SCAVENGER_THRESHOLD',
    'MEMECOIN_ASSETS',
    'SATELLITE_ASSETS',
    'VOL_WINDOW_MAX_POSITIONS',
    'ACC_DRAWDOWN_LIMIT',
    'MICRO_MAX_LEVERAGE',
    'IMMUNE_MAX_LEVERAGE_RATIO',
    'VOL_WINDOW_SPREAD_THRESHOLD',
    'MIN_ORDER_VALUE',
    'REGIME_PERMISSIONS'
]


def test_config_integrity():
    missing = [attr for attr in required_attrs if not hasattr(config, attr)]
    assert not missing, f"Missing config attributes: {missing}"

