"""
ActuatorHolon - Execution (Micro-Holon Architecture)

Objective: Minimize friction (Fees).
Mandate:
- Maker-Only Mode: NEVER use Market Orders.
- Logic: Place Limit Orders at Bid (for Long) or Ask (for Short).
- Post-Order: Monitor for fill. If not filled in 60 seconds, Cancel & Re-assess.

Note: Since this is a simulation/paper-trading env first, 
we simulate the "Wait for Fill" logic.
"""

import time
from typing import Any, Literal
from HolonicTrader.holon_core import Holon, Disposition

class ActuatorHolon(Holon):
    def __init__(self, name: str = "ActuatorAgent"):
        super().__init__(name=name, disposition=Disposition(autonomy=0.8, integration=0.2))
        self.pending_orders = []

    def place_limit_order(self, symbol: str, direction: Literal['BUY', 'SELL'], quantity: float, limit_price: float, mock_execution: bool = True):
        """
        Place a Limit Order (Maker).
        Returns the Order ID. The order is lazily tracked in self.pending_orders.
        """
        print(f"[{self.name}] PLACING LIMIT {direction} {quantity:.4f} {symbol} @ {limit_price:.4f}")
        
        order_id = f"ord_{int(time.time()*1000)}"
        order = {
            'id': order_id,
            'status': 'OPEN',
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'limit_price': limit_price,
            'timestamp': time.time()
        }
        self.pending_orders.append(order)
        return order # Return the OPEN order object
            
    def check_fills(self, candle_low: float, candle_high: float):
        """
        Simulate the fill logic based on OHLC data (for backtesting).
        """
        filled_orders = []
        remaining_orders = []
        
        for order in self.pending_orders:
            is_filled = False
            
            if order['direction'] == 'BUY':
                if candle_low <= order['limit_price']:
                    is_filled = True
            elif order['direction'] == 'SELL':
                if candle_high >= order['limit_price']:
                    is_filled = True
            
            if is_filled:
                order['status'] = 'FILLED'
                order['filled_qty'] = order['quantity']
                order['cost_usd'] = order['quantity'] * order['limit_price']
                filled_orders.append(order)
            else:
                remaining_orders.append(order)
                
        self.pending_orders = remaining_orders
        return filled_orders

    def check_fills_live(self, current_price: float):
        """
        Simulate the fill logic based on the real-time ticker (for live paper-trading).
        """
        filled_orders = []
        remaining_orders = []

        for order in self.pending_orders:
            is_filled = False
            
            if order['direction'] == 'BUY':
                if current_price <= order['limit_price']:
                    is_filled = True
            elif order['direction'] == 'SELL':
                if current_price >= order['limit_price']:
                    is_filled = True
            
            if is_filled:
                order['status'] = 'FILLED'
                order['filled_qty'] = order['quantity']
                order['cost_usd'] = order['quantity'] * order['limit_price']
                filled_orders.append(order)
                print(f"[{self.name}] LIVE FILL: {order['direction']} {order['quantity']} @ {order['limit_price']}")
            else:
                # Cancel if too old (e.g. 5 minutes)
                if time.time() - order['timestamp'] > 300:
                    print(f"[{self.name}] LIVE CANCEL: Order timed out.")
                else:
                    remaining_orders.append(order)

        self.pending_orders = remaining_orders
        return filled_orders

    def receive_message(self, sender: Any, content: Any) -> None:
        """Handle incoming messages."""
        if isinstance(content, Message):
            if content.type == 'PLACE_ORDER':
                # content.payload = {'symbol': ..., 'direction': ..., ...}
                pass
        else:
            pass
