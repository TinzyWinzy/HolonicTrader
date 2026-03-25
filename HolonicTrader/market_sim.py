import time
import logging
import uuid
import config
from typing import Dict, Any, List, Optional, Literal
from HolonicTrader.holon_core import Disposition
from HolonicTrader.market_interface import MarketHolon

logger = logging.getLogger("HolonicTrader.SimulationMarket")

class SimulationMarketHolon(MarketHolon):
    """
    High-fidelity simulation holon that matches real market behavior.
    Tracks virtual balances, positions, and accounts for fees and slippage.
    """
    def __init__(self, initial_capital: float = None):
        super().__init__(name="SimulationMarket", disposition=Disposition(autonomy=0.9, integration=0.1))
        
        self.equity = initial_capital if initial_capital is not None else config.INITIAL_CAPITAL
        self.wallet_balance = self.equity
        self.available_margin = self.equity
        
        # symbol -> position data
        self.positions: Dict[str, Dict[str, Any]] = {}
        
        # pending orders: list of dicts (inherited from MarketHolon)
        self.pending_orders = []
        
        # Simulation parameters
        self.fee_rate = 0.0002 # 0.02% (Maker) or 0.05% (Taker)
        self.slippage_factor = 0.0001 # 0.01%
        
        self.symbol_map = config.KRAKEN_SYMBOL_MAP
        self._last_tick_cache: Dict[str, Dict[str, Any]] = {}

    def get_equity(self) -> float:
        # Equity = Wallet Balance + Unrealized PnL
        unrealized_pnl = 0.0
        for symbol, pos in self.positions.items():
            current_price = self._get_last_price(symbol)
            if current_price > 0:
                side_mult = 1 if pos['side'] == 'long' else -1
                pnl_pct = (current_price - pos['entryPrice']) / pos['entryPrice'] * side_mult
                unrealized_pnl += pnl_pct * pos['notional']
        
        return self.wallet_balance + unrealized_pnl

    def get_balance(self) -> float:
        # Available Margin = Equity - Used Margin
        used_margin = 0.0
        for pos in self.positions.values():
            used_margin += pos['notional'] / pos['leverage']
        
        return self.get_equity() - used_margin

    def get_buying_power(self, leverage: float = 5.0) -> float:
        return self.get_balance() * leverage

    def get_wallet_balance(self) -> float:
        return self.wallet_balance

    def place_order(self, 
                    symbol: str, 
                    direction: Literal['BUY', 'SELL'], 
                    quantity: float, 
                    price: float = 0.0, 
                    order_type: str = 'limit', 
                    leverage: float = 1.0, 
                    reduce_only: bool = False,
                    urgent: bool = False,
                    **kwargs) -> Optional[Dict[str, Any]]:
        
        order_id = f"sim-{uuid.uuid4().hex[:8]}"
        
        # Check if it's a market order or limit order
        if order_type == 'market':
            return self._execute_market_order(symbol, direction, quantity, leverage, reduce_only, order_id, price)
        else:
            # Add to open orders
            new_order = {
                'id': order_id,
                'symbol': symbol,
                'side': direction.lower(),
                'amount': quantity,
                'price': price,
                'type': 'limit',
                'status': 'open',
                'timestamp': time.time(),
                'leverage': leverage,
                'reduceOnly': reduce_only
            }
            self.pending_orders.append(new_order)
            logger.info(f"Placed SIM LIMIT order: {direction} {quantity} {symbol} @ {price}")
            return new_order

    def _execute_market_order(self, symbol, direction, quantity, leverage, reduce_only, order_id, fallback_price=0.0):
        current_price = self._get_last_price(symbol)
        if current_price <= 0:
            current_price = fallback_price
            
        if current_price <= 0:
            logger.error(f"Cannot execute market order for {symbol}: price unknown")
            return None
            
        # Add slippage for market orders
        side_mult = 1 if direction == 'BUY' else -1
        fill_price = current_price * (1 + self.slippage_factor * side_mult)
        
        # Calculate fee
        notional = quantity * fill_price
        fee = notional * 0.0005 # Taker fee
        
        self.wallet_balance -= fee
        
        self._update_position(symbol, direction, quantity, fill_price, leverage, reduce_only)
        
        logger.info(f"Executed SIM MARKET order: {direction} {quantity} {symbol} @ {fill_price:.4f} (Fee: ${fee:.4f})")
        
        return {
            'id': order_id,
            'status': 'closed',
            'symbol': symbol,
            'side': direction.lower(),
            'amount': quantity,
            'price': fill_price,
            'fee': fee
        }

    def _update_position(self, symbol, side, qty, price, leverage, reduce_only):
        side = side.lower()
        if symbol in self.positions:
            pos = self.positions[symbol]
            if pos['side'] == side:
                # Average up/down
                new_qty = pos['contracts'] + qty
                new_notional = (pos['contracts'] * pos['entryPrice']) + (qty * price)
                pos['entryPrice'] = new_notional / new_qty
                pos['contracts'] = new_qty
                pos['notional'] = new_qty * pos['entryPrice']
            else:
                # Reducing or Reversing
                if qty >= pos['contracts']:
                    # Realize PnL and potentially reverse
                    remaining_qty = qty - pos['contracts']
                    pnl = (price - pos['entryPrice']) / pos['entryPrice'] * pos['notional'] * (1 if pos['side'] == 'long' else -1)
                    self.wallet_balance += pnl
                    del self.positions[symbol]
                    if remaining_qty > 0 and not reduce_only:
                        self._update_position(symbol, side, remaining_qty, price, leverage, False)
                else:
                    # Partial reduction
                    realized_pnl = (price - pos['entryPrice']) / pos['entryPrice'] * (qty * pos['entryPrice']) * (1 if pos['side'] == 'long' else -1)
                    self.wallet_balance += realized_pnl
                    pos['contracts'] -= qty
                    pos['notional'] = pos['contracts'] * pos['entryPrice']
        else:
            if not reduce_only:
                self.positions[symbol] = {
                    'symbol': symbol,
                    'contracts': qty,
                    'entryPrice': price,
                    'side': side,
                    'notional': qty * price,
                    'leverage': leverage
                }

    def cancel_all_orders(self, symbol: str):
        original_count = len(self.pending_orders)
        self.pending_orders = [o for o in self.pending_orders if o['symbol'] != symbol]
        cancelled = original_count - len(self.pending_orders)
        if cancelled > 0:
            logger.info(f"Cancelled {cancelled} SIM orders for {symbol}")

    def fetch_positions(self) -> List[Dict[str, Any]]:
        return list(self.positions.values())

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        # In simulation, we need a way to get the real price to update internal state
        # For now, we'll return a mock or actual data if available via a callback
        return self._last_tick_cache.get(symbol, {'last': 0.0})

    def fetch_order_book(self, symbol: str, limit: int = 50) -> Dict[str, Any]:
        return {'bids': [[0, 0]], 'asks': [[0, 0]]} # Simple mock

    def _get_last_price(self, symbol: str) -> float:
        return self._last_tick_cache.get(symbol, {}).get('last', 0.0)

    def update_market_state(self, symbol: str, ticker: Dict[str, Any]):
        """Callback to update the simulation with fresh market data."""
        self._last_tick_cache[symbol] = ticker
        self._check_limit_orders(symbol, ticker)

    def _check_limit_orders(self, symbol: str, ticker: Dict[str, Any]):
        last_price = ticker.get('last', 0.0)
        if last_price <= 0: return

        # Check for fills
        for i in range(len(self.pending_orders) - 1, -1, -1):
            order = self.pending_orders[i]
            if order['symbol'] != symbol: continue
            
            filled = False
            if order['side'] == 'buy' and last_price <= order['price']:
                filled = True
            elif order['side'] == 'sell' and last_price >= order['price']:
                filled = True
                
            if filled:
                logger.info(f"SIM LIMIT ORDER FILLED: {order['side']} {order['amount']} {symbol} @ {order['price']}")
                # Account for maker fee
                fee = order['amount'] * order['price'] * self.fee_rate
                self.wallet_balance -= fee
                self._update_position(symbol, order['side'], order['amount'], order['price'], order['leverage'], order['reduceOnly'])
                self.pending_orders.pop(i)

    def receive_message(self, sender: Any, content: Any) -> None:
        if isinstance(content, dict) and content.get('type') == 'TICKER_UPDATE':
            self.update_market_state(content['symbol'], content['payload'])
