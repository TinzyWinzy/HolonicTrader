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
import ccxt
import config
import os
from typing import Any, Literal
from HolonicTrader.holon_core import Holon, Disposition, Message

class ActuatorHolon(Holon):
    def __init__(self, name: str = "ActuatorAgent", exchange_id: str = 'kraken'):
        super().__init__(name=name, disposition=Disposition(autonomy=0.8, integration=0.2))
        self.pending_orders = []
        self.exchange_id = exchange_id
        
        # Initialize real exchange connection
        if hasattr(ccxt, exchange_id):
            self.exchange = getattr(ccxt, exchange_id)({
                'apiKey': config.API_KEY,
                'secret': config.API_SECRET,
                'enableRateLimit': True,
                # 'options': {'defaultType': 'margin'} # Or similar for futures/margin
            })
        else:
            raise ValueError(f"Exchange {exchange_id} not found in ccxt")

        # Kraken Symbol Mapping (Internal USDT -> Kraken USD)
        self.symbol_map = {
            'UNI/USDT': 'UNI/USD',
            'AAVE/USDT': 'AAVE/USD'
        }

    def place_limit_order(self, symbol: str, direction: Literal['BUY', 'SELL'], quantity: float, limit_price: float, margin: bool = True):
        """
        Place a Limit Order (Maker).
        Returns the Order object.
        """
        exec_symbol = self.symbol_map.get(symbol, symbol)
        side = 'buy' if direction == 'BUY' else 'sell'
        
        print(f"[{self.name}] 🚀 PLACING REAL LIMIT {direction} {quantity:.4f} {exec_symbol} @ {limit_price:.4f}")
        
        try:
            params = {'postOnly': True}
            if margin:
                # Isolated margin for shorting/leverage
                params['marginMode'] = 'isolated' 
                
            order = self.exchange.create_order(
                symbol=exec_symbol,
                type='limit',
                side=side,
                amount=quantity,
                price=limit_price,
                params=params
            )
            
            # Map back to our internal structure for tracking
            internal_order = {
                'id': order['id'],
                'status': 'OPEN',
                'symbol': symbol,
                'direction': direction,
                'quantity': quantity,
                'limit_price': limit_price,
                'timestamp': time.time()
            }
            self.pending_orders.append(internal_order)
            return internal_order
            
        except Exception as e:
            print(f"[{self.name}] ❌ Order Placement Failed: {e}")
            return None
            
    def check_fills(self, candle_low: float = None, candle_high: float = None):
        """
        Check if pending orders were filled. For live, we fetch from exchange.
        """
        filled_orders = []
        remaining_orders = []
        
        for order in self.pending_orders:
            try:
                # Fetch order status from exchange
                remote_order = self.exchange.fetch_order(order['id'], order['symbol'])
                
                if remote_order['status'] == 'closed':
                    order['status'] = 'FILLED'
                    order['filled_qty'] = remote_order['filled']
                    order['cost_usd'] = remote_order['cost']
                    filled_orders.append(order)
                    print(f"[{self.name}] ✅ FILL CONFIRMED: {order['id']}")
                elif remote_order['status'] == 'canceled':
                    print(f"[{self.name}] ⚠️ Order {order['id']} was CANCELED.")
                else:
                    # Timeout logic (5 mins)
                    if time.time() - order['timestamp'] > 300:
                        self.exchange.cancel_order(order['id'], order['symbol'])
                        print(f"[{self.name}] ⏱️ Order {order['id']} TIMEOUT. Canceled.")
                    else:
                        remaining_orders.append(order)
            except Exception as e:
                print(f"[{self.name}] Error checking fill for {order['id']}: {e}")
                remaining_orders.append(order)
                
        self.pending_orders = remaining_orders
        return filled_orders

    def receive_message(self, sender: Any, content: Any) -> None:
        """Handle incoming messages."""
        if isinstance(content, Message):
            if content.type == 'PLACE_ORDER':
                pass
        else:
            pass
