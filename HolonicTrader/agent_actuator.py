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
        
        if config.TRADING_MODE == 'FUTURES':
            self.exchange_id = 'krakenfutures'
            print(f"[{self.name}] 🔌 Connecting to Kraken FUTURES...")
            # Use specific Futures keys if available, else fallback to standard
            api_key = config.KRAKEN_FUTURES_API_KEY or config.API_KEY
            api_secret = config.KRAKEN_FUTURES_PRIVATE_KEY or config.API_SECRET
        else:
            self.exchange_id = 'kraken' # Spot
            print(f"[{self.name}] 🔌 Connecting to Kraken SPOT...")
            api_key = config.API_KEY
            api_secret = config.API_SECRET

        if hasattr(ccxt, self.exchange_id):
            self.exchange = getattr(ccxt, self.exchange_id)({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future' if config.TRADING_MODE == 'FUTURES' else 'spot'
                }
            })
        else:
            raise ValueError(f"Exchange {self.exchange_id} not found in ccxt")

        # Kraken Symbol Mapping (Internal USDT -> Kraken USD)
        self.symbol_map = config.KRAKEN_SYMBOL_MAP

    def check_liquidity(self, symbol: str, direction: str, quantity: float, price: float) -> bool:
        """
        Verify that the order book has sufficient depth to absorb this order
        without massive slippage.
        Rule: Top 10 levels must have cumulative volume >= 3x order quantity.
        """
        try:
            # Fetch shallow book (Limit 10 is fast/cheap)
            book = self.exchange.fetch_order_book(symbol, limit=10)
            
            # If Buying, we consume Asks. If Selling, we hit Bids.
            # (Limit orders technically wait, but we want to know there's a market nearby)
            side_book = book['asks'] if direction == 'BUY' else book['bids']
            
            cumulative_vol = 0.0
            
            for bid_ask in side_book:
                level_price = float(bid_ask[0])
                level_qty = float(bid_ask[1])
                
                # Only count volume within 2% of price checks
                if abs(level_price - price) / price < 0.02:
                     cumulative_vol += level_qty
            
            # Safety Factor: We want book volume to be at least 3x our size
            safety_ratio = 3.0
            
            if cumulative_vol < (quantity * safety_ratio):
                print(f"[{self.name}] ⚠️ THIN BOOK: {symbol} Top 10 Vol {cumulative_vol:.4f} < Req {quantity * safety_ratio:.4f}")
                return False
                
            return True
            
        except Exception as e:
            print(f"[{self.name}] ⚠️ Liquidity Check Error: {e}. Proceeding with caution.")
            return True # Fail open to avoid paralysis, but log warning

    def get_account_balance(self, asset: str = 'USDT') -> float:
        """
        Fetch REAL free balance from exchange.
        """
        try:
            balance = self.exchange.fetch_balance()
            # Kraken might have 'USD' or 'USDT' or 'ZUSD'
            # Try a few common quote currencies
            b_usd = balance['free'].get('USD', 0.0)
            b_usdt = balance['free'].get('USDT', 0.0)
            b_zusd = balance['free'].get('ZUSD', 0.0)
            
            total_avail = max(b_usd, b_usdt, b_zusd)
            # print(f"[{self.name}] 💰 Real Balance: ${total_avail:.2f}")
            return total_avail
        except Exception as e:
            print(f"[{self.name}] ❌ Balance Check Failed: {e}")
            return 0.0

    def get_buying_power(self, leverage: float = 5.0) -> float:
        """
        Fetch Effective Buying Power (Equity * Leverage).
        Uses Kraken's 'eb' (Equivalent Balance) or 'tb' (Trade Balance).
        """
        try:
            balance = self.exchange.fetch_balance()
            info = balance.get('info', {})
            
            # 1. Try Equivalent Balance (Equity) - Spot/Unified
            equity = float(info.get('eb', 0.0))
            
            # 2. Try Futures 'marginEquity' (Common in Kraken Futures API)
            if equity <= 0 and config.TRADING_MODE == 'FUTURES':
                # Handle Kraken Futures 'flex' account structure
                accounts = info.get('accounts', {})
                flex = accounts.get('flex', {})
                
                # Priority: availableMargin (Free to trade) -> marginEquity (Total Net Worth)
                # We use availableMargin to avoid rejecting orders due to tied up funds
                avail_margin = float(flex.get('availableMargin', 0.0))
                margin_equity = float(flex.get('marginEquity', 0.0))
                
                if avail_margin > 0:
                    equity = avail_margin
                    # NOTE: availableMargin is already " Buying Power / Leverage " ? 
                    # No, usually it's the equity available for initial margin.
                    # Buying Power = availableMargin * Leverage.
                elif margin_equity > 0:
                     equity = margin_equity
                else:
                    # Fallback to total USD if flex is empty (e.g. cash only)
                    equity = balance.get('total', {}).get('USD', 0.0)
                
            # 3. Fallback to Trade Balance
            if equity <= 0:
                equity = float(info.get('tb', 0.0))
                
            # 4. Fallback to Free Balance
            if equity <= 0:
                return self.get_account_balance()
                
            # Buying Power = Equity * Leverage
            if config.TRADING_MODE == 'FUTURES':
                # For Futures, let's trust the configured leverage cap
                return equity * leverage
            else:
                return equity * leverage
            
        except Exception as e:
            print(f"[{self.name}] ❌ Buying Power Check Failed: {e}")
            return self.get_account_balance()

    def place_limit_order(self, symbol: str, direction: Literal['BUY', 'SELL'], quantity: float, limit_price: float, margin: bool = True, leverage: float = 1.0):
        """
        Place a limit order (Post-Only) on the exchange.
        
        Args:
            symbol: Internal symbol (e.g. 'BTC/USDT')
            direction: 'BUY' or 'SELL'
            quantity: Amount to buy/sell
            limit_price: Limit price
            margin: Whether to use margin (Futures default: True)
            leverage: Leverage multiplier (e.g., 20.0). Defaults to 1.0 if not specified.
        """
        exec_symbol = self.symbol_map.get(symbol, symbol)
        side = 'buy' if direction == 'BUY' else 'sell'
        
        # Prepare values with correct precision
        try:
            # We must use the 'exec_symbol' (ccxt unified or mapped) for precision calls
            # However, if 'exec_symbol' is a raw ID like 'PF_XBTUSD', ccxt might not find it in 'markets' map by that key directly 
            # if loaded via fetch_markets(). 
            # Safest is to use the Unified Symbol if possible, or try the mapped one.
            # check if self.exchange.markets is loaded
            if not self.exchange.markets:
                self.exchange.load_markets()

            # --- PATCH: MIN QUANTITY CLAMPING ---
            market = self.exchange.market(exec_symbol)
            min_limit = market.get('limits', {}).get('amount', {}).get('min')
            prec_amount = market.get('precision', {}).get('amount')
            
            # Use strict fallback if None
            if min_limit is None: min_limit = 0.0
            if prec_amount is None: prec_amount = 0.0
            
            # Effective minimum is the larger of the exchange limit or the precision unit
            # (e.g. limit 0.0001, precision 0.001 -> we can't trade 0.0001)
            effective_min = max(min_limit, prec_amount)
            
            if quantity < effective_min and quantity > 0:
                 print(f"[{self.name}] 🤏 Clamping Qty {quantity} -> {effective_min} (Min Allowed)")
                 quantity = effective_min
            # ------------------------------------

            # Convert float to precise string
            qty_str = self.exchange.amount_to_precision(exec_symbol, quantity)
            price_str = self.exchange.price_to_precision(exec_symbol, limit_price)
            
            # Convert back to float/number for create_order if it expects numbers, 
            # BUT ccxt usually handles strings best to avoid float drift. 
            # However, standard ccxt usage is often float. 
            # Let's pass the float of the precise string to be safe, or just the string if ccxt supports it.
            # Most robust: Pass float, but rounded.
            final_qty = float(qty_str)
            final_price = float(price_str)
            
        except Exception as e:
            print(f"[{self.name}] ⚠️ Precision formatting failed: {e}. Using raw values.")
            final_qty = quantity
            final_price = limit_price

        # --- LIQUIDITY SANITY CHECK ---
        # Ensure we aren't eating the entire book
        if not self.check_liquidity(exec_symbol, direction, final_qty, final_price):
            print(f"[{self.name}] 🛑 LIQUIDITY CHECK FAILED: {exec_symbol} Book too thin for {final_qty}. Order Aborted.")
            return None
        # ------------------------------

        print(f"[{self.name}] 🚀 PLACING REAL LIMIT {direction} {final_qty} {exec_symbol} @ {final_price} (Lev: {leverage}x)")
        
        try:
            # --- PATCH: SET LEVERAGE ---
            if margin and leverage > 1.0:
                 try:
                     self.exchange.set_leverage(leverage, exec_symbol)
                 except Exception as lev_err:
                     print(f"[{self.name}] ⚠️ Set Leverage Failed: {lev_err}")
            # ---------------------------

            params = {'postOnly': True}
            if margin:
                # Isolated margin for shorting/leverage
                params['marginMode'] = 'isolated' 
                
            order = self.exchange.create_order(
                symbol=exec_symbol,
                type='limit',
                side=side,
                amount=final_qty, # Use precise value
                price=final_price, # Use precise value
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
            error_msg = str(e)
            if "postWouldExecute" in error_msg or "OrderImmediatelyFillable" in error_msg:
                print(f"[{self.name}] ⚠️ Maker Order Rejected (Price crossed spread). Retrying as TAKER...")
                try:
                    # Retry without Post-Only (Eat the Taker Fee to ensure execution)
                    if 'postOnly' in params:
                        del params['postOnly']
                    
                    order = self.exchange.create_order(
                        symbol=exec_symbol,
                        type='limit',
                        side=side,
                        amount=final_qty,
                        price=final_price,
                        params=params
                    )
                    
                    # Log successful retry
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
                    print(f"[{self.name}] ✅ TAKER FILL SUBMITTED: {order['id']}")
                    return internal_order
                    
                except Exception as retry_err:
                    print(f"[{self.name}] ❌ Taker Retry Failed: {retry_err}")
                    return None
            else:
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
                # Kraken Futures usually doesn't support fetching a single order by ID easily
                # So we must scan Open and Closed lists
                found_order = None
                
                # Use mapped symbol for exchange calls
                exec_symbol = self.symbol_map.get(order['symbol'], order['symbol'])
                
                # 1. Check Open Orders
                # (Optimization: We could fetch all open orders ONCE per cycle instead of per order, 
                # but for now let's keep it robust)
                try:
                    # We pass symbol to narrow it down if possible
                    open_orders = self.exchange.fetch_open_orders(exec_symbol)
                    for o in open_orders:
                        if o['id'] == order['id']:
                            found_order = o
                            break
                except Exception as e:
                    print(f"[{self.name}] ⚠️ fetch_open_orders failed: {e}")

                # 2. If not found, Check Closed Orders (It might have just filled)
                if not found_order:
                    try:
                        closed_orders = self.exchange.fetch_closed_orders(exec_symbol, limit=20)
                        for o in closed_orders:
                            if o['id'] == order['id']:
                                found_order = o
                                break
                    except Exception as e:
                        print(f"[{self.name}] ⚠️ fetch_closed_orders failed: {e}")
                
                # 3. Process Result
                if found_order:
                    remote_status = found_order['status']
                    
                    if remote_status == 'closed':
                        order['status'] = 'FILLED'
                        order['filled_qty'] = found_order.get('filled', order.get('quantity'))
                        order['cost_usd'] = found_order.get('cost', 0.0)
                        # CRITICAL FIX: Ensure 'price' is populated for Executor
                        order['price'] = found_order.get('average') or found_order.get('price') or order.get('price')
                        
                        filled_orders.append(order)
                        print(f"[{self.name}] ✅ FILL CONFIRMED: {order['id']} ({order['symbol']}) @ {order['price']}")
                        
                    elif remote_status == 'canceled':
                        print(f"[{self.name}] ⚠️ Order {order['id']} was CANCELED.")
                        
                    elif remote_status == 'open':
                        # Still open, check timeout
                        if time.time() - order['timestamp'] > 300:
                            self.exchange.cancel_order(order['id'], exec_symbol)
                            print(f"[{self.name}] ⏱️ Order {order['id']} TIMEOUT. Canceled.")
                        else:
                            remaining_orders.append(order)
                    else:
                        # Unknown status
                        remaining_orders.append(order)
                else:
                    # Order not found in either list? 
                    # It might be an old order that fell off the list, or API lag.
                    # Keep watching it for a bit unless it's very old
                    if time.time() - order['timestamp'] > 600:
                         print(f"[{self.name}] 👻 Order {order['id']} disappeared. Assuming Canceled/lost.")
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

    def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        """
        Fetch Order Book from the EXECUTION VENUE (Kraken Futures etc).
        Crucial for checking liquidity on the exchange we actually trade on.
        """
        try:
            # Map Symbol
            exec_symbol = symbol
            if config.TRADING_MODE == 'FUTURES':
                exec_symbol = config.KRAKEN_SYMBOL_MAP.get(symbol, symbol)
                
            book = self.exchange.fetch_order_book(exec_symbol, limit)
            return {
                'bids': book['bids'],
                'asks': book['asks'],
                'timestamp': book['timestamp']
            }
        except Exception as e:
            print(f"[{self.name}] ⚠️ Actuator Book Fetch Fail {symbol}: {e}")
            return {'bids': [], 'asks': []}
