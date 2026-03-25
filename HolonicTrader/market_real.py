import time
import ccxt
import logging
from typing import Dict, Any, List, Optional, Literal
from HolonicTrader.holon_core import Disposition
from HolonicTrader.market_interface import MarketHolon
import config

logger = logging.getLogger("HolonicTrader.RealMarket")

class RealMarketHolon(MarketHolon):
    """
    Real-world market interaction holon using CCXT and Kraken.
    """
    def __init__(self, name: str = "RealMarket"):
        super().__init__(name=name, disposition=Disposition(autonomy=0.2, integration=1.0))
        
        # Initialize exchange
        if config.TRADING_MODE == 'FUTURES':
            self.exchange_id = 'krakenfutures'
            api_key = config.KRAKEN_FUTURES_API_KEY or config.API_KEY
            api_secret = config.KRAKEN_FUTURES_PRIVATE_KEY or config.API_SECRET
        else:
            self.exchange_id = 'kraken'
            api_key = config.API_KEY
            api_secret = config.API_SECRET

        self.exchange = getattr(ccxt, self.exchange_id)({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future' if config.TRADING_MODE == 'FUTURES' else 'spot'
            }
        })
        
        self.symbol_map = config.KRAKEN_SYMBOL_MAP

    def get_equity(self) -> float:
        try:
            balance = self.exchange.fetch_balance()
            if config.TRADING_MODE == 'FUTURES':
                return float(balance.get('info', {}).get('accounts', {}).get('flex', {}).get('marginEquity', 0.0))
            else:
                return float(balance.get('info', {}).get('eb', 0.0))
        except Exception as e:
            logger.error(f"Failed to fetch equity: {e}")
            return 0.0

    def get_balance(self) -> float:
        try:
            balance = self.exchange.fetch_balance()
            if config.TRADING_MODE == 'FUTURES':
                return float(balance.get('info', {}).get('accounts', {}).get('flex', {}).get('availableMargin', 0.0))
            else:
                return balance['free'].get('USDT', balance['free'].get('USD', 0.0))
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return 0.0

    def get_buying_power(self, leverage: float = 5.0) -> float:
        return self.get_balance() * leverage

    def get_wallet_balance(self) -> float:
        """
        Fetch wallet balance from Kraken Futures.
        FIX 2026-03-02: Use marginEquity as primary source (includes unrealized PnL).
        Fall back to availableMargin, then walletBalance.
        """
        try:
            balance = self.exchange.fetch_balance()
            if config.TRADING_MODE == 'FUTURES':
                info = balance.get('info', {})
                accounts = info.get('accounts', {})
                flex = accounts.get('flex', {})

                # Priority 1: marginEquity (total equity including unrealized PnL)
                margin_equity = flex.get('marginEquity', 0.0)
                # FIX 2026-03-08: Convert string to float (API returns strings)
                if isinstance(margin_equity, str):
                    margin_equity = float(margin_equity) if margin_equity else 0.0

                # Priority 2: availableMargin (free collateral)
                available = flex.get('availableMargin', 0.0)
                if isinstance(available, str):
                    available = float(available) if available else 0.0

                # Priority 3: walletBalance (pure collateral, no PnL)
                wallet_bal = flex.get('walletBalance', 0.0)
                if isinstance(wallet_bal, str):
                    wallet_bal = float(wallet_bal) if wallet_bal else 0.0

                # Use marginEquity if available and reasonable (> $10)
                if margin_equity and margin_equity > 10.0:
                    result = float(margin_equity)
                elif available and available > 10.0:
                    result = float(available)
                else:
                    result = float(wallet_bal) if wallet_bal else 0.0

                # Log if balance seems too low (debugging aid)
                if result < 10.0:
                    logger.warning(f"[MarketHolon] ⚠️ Low balance detected: ${result:.2f} (marginEquity=${margin_equity:.2f}, available=${available:.2f}, wallet=${wallet_bal:.2f})")

                return result
            else:
                return balance.get('total', {}).get('USDT', balance.get('total', {}).get('USD', 0.0))
        except Exception as e:
            logger.error(f"Failed to fetch wallet balance: {e}")
            return 0.0

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
        
        exec_symbol = self.symbol_map.get(symbol, symbol)
        side = 'buy' if direction == 'BUY' else 'sell'
        
        params = kwargs.copy()
        if leverage > 1.0:
            try:
                self.exchange.set_leverage(leverage, exec_symbol)
            except Exception as e:
                logger.warning(f"Failed to set leverage for {exec_symbol}: {e}")

        if order_type == 'limit':
            if not urgent:
                params['postOnly'] = True

        # FIX 2026-03-19: Take-profit orders must always be reduce-only
        take_profit = kwargs.get('take_profit', False)
        if reduce_only or take_profit:
            params['reduceOnly'] = True

        try:
            # Apply exchange precision helpers to avoid unexpected rounding
            try:
                qty_str = self.exchange.amount_to_precision(exec_symbol, quantity)
                prc_str = None if order_type != 'limit' else self.exchange.price_to_precision(exec_symbol, price)
                final_qty = float(qty_str)
                final_price = float(prc_str) if prc_str is not None else None
            except Exception:
                final_qty = quantity
                final_price = price if order_type == 'limit' else None

            order = self.exchange.create_order(
                symbol=exec_symbol,
                type=order_type,
                side=side,
                amount=final_qty,
                price=final_price if order_type == 'limit' else None,
                params=params
            )

            # Normalize order result to a simple dict expected by Executor
            normalized = {
                'id': order.get('id'),
                'order_id': order.get('id'),
                'status': order.get('status', 'open'),
                'symbol': symbol,
                'filled_qty': float(order.get('filled', order.get('amount', 0.0) or 0.0)),
                'avg_fill_price': order.get('average') or order.get('price') or (final_price or price),
                'fee': order.get('fee', {}),
                'raw': order
            }
            return normalized
        except Exception as e:
            logger.error(f"Order failed for {symbol}: {e}")
            return None

    def cancel_all_orders(self, symbol: str):
        exec_symbol = self.symbol_map.get(symbol, symbol)
        try:
            self.exchange.cancel_all_orders(exec_symbol)
        except Exception as e:
            logger.error(f"Cancel all orders failed for {symbol}: {e}")

    def fetch_positions(self) -> List[Dict[str, Any]]:
        try:
            positions_raw = self.exchange.fetch_positions()
            positions = []
            for p in positions_raw:
                size = float(p.get('contracts') or p.get('size') or 0.0)
                if size == 0: continue
                positions.append(p)
            return positions
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        exec_symbol = self.symbol_map.get(symbol, symbol)
        return self.exchange.fetch_ticker(exec_symbol)

    def fetch_order_book(self, symbol: str, limit: int = 50) -> Dict[str, Any]:
        exec_symbol = self.symbol_map.get(symbol, symbol)
        return self.exchange.fetch_order_book(exec_symbol, limit=limit)

    def receive_message(self, sender: Any, content: Any) -> None:
        pass
