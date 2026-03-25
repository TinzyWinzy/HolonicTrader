from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Literal
from HolonicTrader.holon_core import Holon, Disposition

class MarketHolon(Holon, ABC):
    """
    Unified interface for market interaction, covering both real and simulated environments.
    """
    def __init__(self, name: str, disposition: Disposition):
        super().__init__(name=name, disposition=disposition)
        self.pending_orders: List[Dict[str, Any]] = []

    @abstractmethod
    def get_equity(self) -> float:
        """Fetch total account equity."""
        pass

    @abstractmethod
    def get_balance(self) -> float:
        """Fetch available margin/balance."""
        pass

    @abstractmethod
    def get_buying_power(self, leverage: float = 5.0) -> float:
        """Fetch total buying power (balance * leverage)."""
        pass

    @abstractmethod
    def get_wallet_balance(self) -> float:
        """Fetch wallet balance (cash + realized PnL)."""
        pass

    @abstractmethod
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
        """Place an order (Market or Limit)."""
        pass

    @abstractmethod
    def cancel_all_orders(self, symbol: str):
        """Cancel all open orders for a specific symbol."""
        pass

    @abstractmethod
    def fetch_positions(self) -> List[Dict[str, Any]]:
        """Fetch current open positions."""
        pass

    @abstractmethod
    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch current market ticker for a symbol."""
        pass

    @abstractmethod
    def fetch_order_book(self, symbol: str, limit: int = 50) -> Dict[str, Any]:
        """Fetch order book for liquidity analysis."""
        pass
