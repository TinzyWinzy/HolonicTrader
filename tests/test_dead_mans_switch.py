import pytest
from unittest.mock import MagicMock
from HolonicTrader.exceptions import DeadMansSwitchTriggered
from HolonicTrader.agent_executor import ExecutorHolon, Position
from datetime import datetime, timezone

def test_dead_mans_switch_massive_ghost():
    """Verify that a massive undocumented exchange position triggers the DMS during a SOFT sync."""
    mock_market = MagicMock()
    # Mock an exchange returning a massive position not in ledger
    # DMS check: quantity > balance_usd * 0.5, so with $1000 balance, need quantity > 500
    mock_market.fetch_positions.return_value = [
        {'symbol': 'BTC/USD:USD', 'side': 'long', 'contracts': 600.0, 'entryPrice': 60000.0, 'markPrice': 60000.0, 'leverage': 5.0} # 600 contracts > $500 threshold
    ]
    mock_market.get_equity.return_value = 1000.0
    mock_market.get_wallet_balance.return_value = 1000.0

    mock_db = MagicMock()
    mock_db.get_portfolio.return_value = {'balance_usd': 1000.0, 'held_assets': {}, 'position_metadata': {}}

    # DMS triggers during sync_with_exchange called in constructor
    with pytest.raises(DeadMansSwitchTriggered):
        ExecutorHolon(initial_capital=1000.0, governor=MagicMock(), market=mock_market, db_manager=mock_db, actuator=MagicMock())

def test_dead_mans_switch_massive_purge():
    """Verify that if a massive internal position suddenly vanishes from exchange, DMS triggers."""
    mock_market = MagicMock()
    # Exchange returns nothing!
    mock_market.fetch_positions.return_value = []
    mock_market.get_equity.return_value = 50000.0
    mock_market.get_wallet_balance.return_value = 50000.0

    # Mock an internal position that is massive (value > 50% of $50000 = $25000)
    # pos_value = quantity * entry_price = 1.0 * 40000 = $40000 > $25000 threshold
    pos_metadata = {
        'BTC/USDT': {
            'symbol': 'BTC/USDT',
            'direction': 'BUY',
            'quantity': 1.0,
            'entry_price': 40000.0,
            'entry_timestamp': datetime.now(timezone.utc).isoformat(),
            'leverage': 1.0,
            'strategy': 'test'
        }
    }

    mock_db = MagicMock()
    mock_db.get_portfolio.return_value = {'balance_usd': 50000.0, 'held_assets': {}, 'position_metadata': pos_metadata}

    # Position loaded from DB, then sync finds nothing on exchange -> DMS triggers
    with pytest.raises(DeadMansSwitchTriggered):
        ExecutorHolon(initial_capital=50000.0, governor=MagicMock(), market=mock_market, db_manager=mock_db, actuator=MagicMock())

if __name__ == "__main__":
    test_dead_mans_switch_massive_ghost()
    test_dead_mans_switch_massive_purge()
    print("All tests passed.")
