"""
MonteCarloPositionManager - Advanced Position Health Analyzer

Uses Monte Carlo simulations to evaluate existing positions and determine
if losing positions should be closed based on probability of recovery.

"I see the future paths, and I cut the bad ones."
"""

import time
from typing import Dict, Any, Optional, Tuple
from HolonicTrader.holon_core import Holon, Disposition
from HolonicTrader.sde_engine import SDEEngine
import config

class MonteCarloPositionManager(Holon):
    def __init__(self, name: str = "MonteCarloPositionManager"):
        super().__init__(name=name, disposition=Disposition(autonomy=0.7, integration=0.6))
        self.last_evaluation_time = {}
        self.position_eval_cache = {}
        self.evaluation_interval = 300  # Evaluate every 5 minutes
        
    def evaluate_position_for_closure(self, 
                                    symbol: str, 
                                    current_price: float, 
                                    entry_price: float, 
                                    direction: str, 
                                    position_age_hours: float,
                                    sde_params: Dict[str, Any],
                                    pnl_pct: float = 0.0,
                                    take_profit_price: Optional[float] = None,
                                    stop_loss_price: Optional[float] = None) -> Tuple[bool, float, str]:
        """
        Evaluate if a position should be closed using Monte Carlo simulation.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            entry_price: Position entry price
            direction: 'BUY' or 'SELL'
            position_age_hours: How long the position has been open
            sde_params: SDE physics parameters (mu, sigma, lambda)
            pnl_pct: Current PnL percentage
            take_profit_price: Current take profit price
            stop_loss_price: Current stop loss price
            
        Returns:
            Tuple of (should_close, confidence_score, reason)
        """
        # Only evaluate losing positions or positions with toxic characteristics
        if pnl_pct > -0.005:  # If not losing (or barely losing), check for other issues
            # Check if this is a toxic funding position (paying high funding rates)
            # Get toxic funding info from position data if available
            toxic_funding = sde_params.get('toxic_funding', False)
            if not toxic_funding and hasattr(self, 'governor_positions'):
                pos_data = self.governor_positions.get(symbol)
                if pos_data is not None:
                    toxic_funding = self._pos_attr(pos_data, 'toxic_funding', False)

            if toxic_funding:
                print(f"[{self.name}] 🚨 TOXIC FUNDING DETECTED: {symbol} - forcing Monte Carlo evaluation")
                # For toxic funding positions, we should close regardless of PnL
                return True, 0.95, "TOXIC_FUNDING_IMMEDIATE_CLOSURE"  # High confidence to close
            else:
                return False, 0.0, "Position is profitable or nearly breakeven"
            
        # Don't evaluate too recently
        current_time = time.time()
        last_eval = self.last_evaluation_time.get(symbol, 0)
        if current_time - last_eval < self.evaluation_interval:
            return False, 0.0, "Too soon for re-evaluation"

        # Set up realistic exit targets if not provided
        if not take_profit_price:
            # Use conservative target based on current PnL
            if direction == 'BUY':
                take_profit_price = entry_price * 1.02  # 2% target
            else:
                take_profit_price = entry_price * 0.98  # 2% target

        if not stop_loss_price:
            # Use current price as stop if we're losing
            if pnl_pct < -0.03:  # If losing more than 3%
                stop_loss_price = current_price
            else:
                if direction == 'BUY':
                    stop_loss_price = entry_price * 0.97  # 3% stop
                else:
                    stop_loss_price = entry_price * 1.03  # 3% stop
        
        try:
            # Calculate probability of hitting TP vs SL using Monte Carlo
            params = {
                'mu': sde_params.get('mu', 0.0),
                'sigma': sde_params.get('sigma', 0.1),
                'lambda': sde_params.get('lambda', 0.1)
            }
            
            # Calculate probability of hitting stop loss before take profit
            # Use conservative parameters to avoid memory issues
            horizon_steps = min(int(position_age_hours * 60 * 60), 10000)  # Cap at 10k steps
            paths = 500  # Reduced from 1000 to avoid memory issues

            # Calculate probability of hitting stop loss before take profit
            # Using the Governor's method for calculating ruin probability
            from HolonicTrader.sde_engine import SDEEngine
            prob_sl_before_tp = SDEEngine.calculate_ruin_probability(
                model='GBM',
                params=params,
                start_price=current_price,
                sl_price=stop_loss_price,
                tp_price=take_profit_price,
                horizon=horizon_steps,
                paths=paths
            )

            # Calculate probability of recovery (hitting breakeven before SL)
            breakeven_price = entry_price
            prob_recovery = 1.0 - SDEEngine.calculate_ruin_probability(
                model='GBM',
                params=params,
                start_price=current_price,
                sl_price=stop_loss_price,
                tp_price=breakeven_price,  # Changed from 'breakeven_price' to 'tp_price' to match function signature
                horizon=horizon_steps,
                paths=paths
            )
            
            # Decision logic
            should_close = False
            reason = ""

            # If probability of hitting stop loss is very high, close
            if prob_sl_before_tp > 0.70:  # Lowered threshold for more sensitivity
                should_close = True
                reason = f"High SL probability: {prob_sl_before_tp:.2%}"

            # If probability of recovery is very low and position is old
            elif prob_recovery < 0.30 and position_age_hours > 1:  # Lowered thresholds
                should_close = True
                reason = f"Low recovery chance: {prob_recovery:.2%}, age: {position_age_hours:.1f}h"

            # If position is losing significantly with poor prospects
            elif pnl_pct < -0.03 and prob_recovery < 0.40:  # Lowered thresholds
                should_close = True
                reason = f"Deep loss with poor recovery: {pnl_pct:.2%}, rec: {prob_recovery:.2%}"

            # If position has been losing for a long time with no improvement
            elif position_age_hours > 4 and pnl_pct < -0.015 and prob_recovery < 0.35:  # Lowered thresholds
                should_close = True
                reason = f"Extended losing period: {position_age_hours:.1f}h, rec: {prob_recovery:.2%}"

            # Additional check: If position is losing and Monte Carlo shows very poor prospects
            elif pnl_pct < -0.02 and prob_recovery < 0.25:
                should_close = True
                reason = f"Poor recovery prospects: {pnl_pct:.2%}, rec: {prob_recovery:.2%}"
            
            # Update evaluation time
            self.last_evaluation_time[symbol] = current_time
            
            confidence = prob_sl_before_tp if should_close else (1.0 - prob_recovery)
            
            if should_close:
                print(f"[{self.name}] Monte Carlo Closure Signal: {symbol} - {reason} (Conf: {confidence:.2%})")
                
            return should_close, confidence, reason
            
        except Exception as e:
            print(f"[{self.name}] Monte Carlo evaluation error for {symbol}: {e}")
            return False, 0.0, f"Evaluation error: {str(e)}"
    
    def run_position_health_check(self,
                                 positions: Dict[str, Any],
                                 current_prices: Dict[str, float],
                                 sde_data: Dict[str, Any]) -> list:
        """
        Run health check on all positions and return closure recommendations.

        Args:
            positions: Dictionary of all current positions
            current_prices: Current market prices for all symbols
            sde_data: SDE physics parameters for each symbol

        Returns:
            List of symbols to close with reasons
        """
        # Store positions for access in evaluation methods
        self.governor_positions = positions

        # Ensure we always return a list
        closure_recommendations = []

        # Check inputs
        if not positions or not current_prices:
            return closure_recommendations

        for symbol, pos_data in list(positions.items()):
            # Skip if no current price available
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            # Support both Position objects (attribute access) and legacy dicts (.get)
            entry_price = self._pos_attr(pos_data, 'entry_price', 0.0)
            direction = self._pos_attr(pos_data, 'direction', 'BUY')
            quantity = self._pos_attr(pos_data, 'quantity', 0.0)

            if quantity <= 0 or entry_price <= 0:
                continue

            # Calculate current PnL
            if direction == 'BUY':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            # Get position age (Position objects don't have first_entry_time; use 0 as fallback)
            first_entry_time = self._pos_attr(pos_data, 'first_entry_time', None)
            if first_entry_time is None:
                first_entry_time = self._pos_attr(pos_data, 'entry_timestamp', None)
                if first_entry_time is not None:
                    try:
                        import datetime
                        ts = datetime.datetime.fromisoformat(str(first_entry_time))
                        first_entry_time = ts.timestamp()
                    except Exception:
                        first_entry_time = time.time()
                else:
                    first_entry_time = time.time()
            position_age_hours = (time.time() - first_entry_time) / 3600.0

            # Get SDE parameters
            sde_params = sde_data.get(symbol, {
                'mu': 0.0,
                'sigma': 0.1,
                'lambda': 0.1
            })

            # Add toxic funding info to sde_params if available
            if self._pos_attr(pos_data, 'toxic_funding', False):
                sde_params['toxic_funding'] = True
                sde_params['funding_apy'] = self._pos_attr(pos_data, 'funding_apy', 0.0)

            # Evaluate position
            eval_result = self.evaluate_position_for_closure(
                symbol=symbol,
                current_price=current_price,
                entry_price=entry_price,
                direction=direction,
                position_age_hours=position_age_hours,
                sde_params=sde_params,
                pnl_pct=pnl_pct
            )

            # Check if result is valid
            if eval_result is None or not isinstance(eval_result, tuple) or len(eval_result) < 3:
                continue  # Skip if evaluation failed

            should_close, confidence, reason = eval_result[0], eval_result[1], eval_result[2]

            if should_close:
                closure_recommendations.append({
                    'symbol': symbol,
                    'reason': reason,
                    'confidence': confidence,
                    'pnl_pct': pnl_pct,
                    'age_hours': position_age_hours,
                    'current_price': current_price,
                    'entry_price': entry_price
                })

        return closure_recommendations

    @staticmethod
    def _pos_attr(pos, key: str, default=None):
        """Read a field from either a Position object or a legacy dict."""
        if isinstance(pos, dict):
            return pos.get(key, default)
        return getattr(pos, key, default)
    
    def receive_message(self, sender: Any, content: Any) -> Any:
        """Handle incoming messages."""
        msg_type = content.get('type', '')
        
        if msg_type == 'MONTE_CARLO_EVALUATE_POSITION':
            return self.evaluate_position_for_closure(**content)
        elif msg_type == 'MONTE_CARLO_HEALTH_CHECK':
            return self.run_position_health_check(**content)
        elif msg_type == 'GET_STATUS':
            return {
                'active_evaluations': len(self.last_evaluation_time),
                'last_evaluation_times': dict(list(self.last_evaluation_time.items())[:5])
            }
        
        return None