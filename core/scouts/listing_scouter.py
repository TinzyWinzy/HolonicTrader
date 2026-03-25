
import json
import os
import logging
from typing import List, Dict, Any, Set

logger = logging.getLogger('ListingScouter')

class ListingScouter:
    """
    Scouts for newly listed assets on the exchange by comparing 
    current market symbols against a known history.
    """
    def __init__(self, state_file: str = "known_assets.json"):
        self.state_file = state_file
        self.known_assets: Set[str] = self._load_state()

    def _load_state(self) -> Set[str]:
        """Load known assets from disk."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    # Handle list or dict format
                    if isinstance(data, list):
                        return set(data)
                    elif isinstance(data, dict):
                        return set(data.keys())
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        return set()

    def _save_state(self):
        """Persist known assets to disk."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(list(self.known_assets), f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def check_for_new_listings(self, exchange_markets: Dict[str, Any]) -> List[str]:
        """
        Compare current markets against known history.
        
        Args:
            exchange_markets: Result of ccxt.load_markets() (dict of symbol -> info)
            
        Returns:
            List of newly detected symbols.
        """
        if not exchange_markets:
            return []

        current_symbols = set(exchange_markets.keys())
        
        # If this is the VERY FIRST run, just populate state and return empty
        # (Otherwise we'd alert on 500 "new" coins)
        if not self.known_assets:
            logger.info(f"First run: Initializing state with {len(current_symbols)} assets.")
            self.known_assets = current_symbols
            self._save_state()
            return []
            
        # Check for diff
        new_listings = list(current_symbols - self.known_assets)
        
        if new_listings:
            logger.info(f"🚨 NEW LISTINGS DETECTED: {new_listings}")
            # Update state
            self.known_assets.update(new_listings)
            self._save_state()
            
        return new_listings
