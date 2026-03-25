import time
import pytest
from HolonicTrader.agent_executor import ExecutorHolon, TradeSignal, TradeDecision
from HolonicTrader.holon_core import Disposition


class DummyMarket:
    def __init__(self, place_order_result=None, place_bracket_result=None):
        self.place_order_result = place_order_result
        self.place_bracket_result = place_bracket_result
        self.pending_orders = []

    def place_order(self, **kwargs):
        return self.place_order_result

    def place_bracket_order(self, **kwargs):
        return self.place_bracket_result


def test_dispatch_partial_fill(tmp_path, capsys):
    # Setup executor with dummy market returning partial fill
    dummy_fill = {'avg_fill_price': 1.23, 'filled': 6.0, 'fee': {'cost': 0.05}, 'id': 'ord123'}
    m = DummyMarket(place_order_result=dummy_fill)
    ex = ExecutorHolon(name='TestExecutor', initial_capital=100.0)
    ex.market = m

    res = ex._dispatch_to_market(symbol='TEST/USDT', direction='BUY', qty=10.0, price=1.2, leverage=1.0, is_exit=False)
    assert isinstance(res, dict)
    assert res.get('filled_qty') == 6.0
    assert res.get('avg_fill_price') == 1.23
    assert res.get('fee_usd') == 0.05


def test_place_bracket_orders_prefers_actuator(tmp_path):
    # Executor should prefer actuator.place_bracket_order over market
    bracket_response = {'sl': {'id': 'sl1'}, 'tp': {'id': 'tp1'}}

    ex = ExecutorHolon(name='TestExecutor2', initial_capital=100.0)

    class DummyActuator:
        def place_bracket_order(self, **kwargs):
            return bracket_response

    ex.actuator = DummyActuator()
    # market also has bracket but actuator should be preferred
    ex.market = DummyMarket(place_bracket_result={'market_bracket': True})

    res = ex.place_bracket_orders('ABC/USDT', 'BUY', 1.0, 0.9, 1.1, leverage=1.0)
    assert res == bracket_response
