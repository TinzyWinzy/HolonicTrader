"""
ML Exit Optimization Module

Predicts optimal exit timing for active positions based on:
- Current PnL
- Hold time
- Market conditions
- Historical patterns

Usage:
    from HolonicTrader.ml_exit_optimizer import MLExitOptimizer
    
    optimizer = MLExitOptimizer()
    recommendation = optimizer.predict_exit(symbol, position)
"""

import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, Any, Optional
from datetime import datetime

class MLExitOptimizer:
    """
    ML-based exit timing optimization
    """
    
    def __init__(self, model_path: str = 'models/lgbm_exit_timing.pkl'):
        """
        Initialize exit optimizer
        
        Args:
            model_path: Path to trained exit timing model
        """
        self.model = None
        self.model_path = model_path
        self.exit_history = []
        
        # Feature list
        self.FEATURES = [
            'current_pnl_pct',
            'hold_time_hours',
            'direction_encoded',
            'entry_price',
            'current_price',
            'price_change_pct',
            'hour',
            'day_of_week'
        ]
        
        # Try to load existing model
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                print(f"[MLExitOptimizer] Loaded model from {model_path}")
            except Exception as e:
                print(f"[MLExitOptimizer] Failed to load model: {e}")
        else:
            print(f"[MLExitOptimizer] No model found at {model_path}, will use rule-based logic")
    
    def predict_exit(self, symbol: str, position: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict optimal exit timing for a position
        
        Args:
            symbol: Trading pair symbol
            position: Position data dict with keys:
                - direction: 'BUY' or 'SELL'
                - entry_price: Entry price
                - current_price: Current price
                - pnl_percent: Current PnL %
                - entry_time: Entry timestamp
                - hold_time_hours: How long position has been open (optional)
        
        Returns:
            Dictionary with exit recommendation:
            {
                'recommendation': 'HOLD' | 'TAKE_PROFIT' | 'CUT_LOSS' | 'TRAILING',
                'confidence': float (0-1),
                'reason': str,
                'suggested_action': str,
                'urgency': str ('LOW' | 'MEDIUM' | 'HIGH'),
                'expected_improvement': float (expected PnL improvement %)
            }
        """
        # Extract position data
        direction = position.get('direction', 'BUY')
        entry_price = position.get('entry_price', 0)
        current_price = position.get('current_price', 0)
        pnl_pct = position.get('pnl_percent', 0)
        entry_time = position.get('entry_time', datetime.now())
        
        # Calculate hold time
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time)
        hold_time_hours = (datetime.now() - entry_time).total_seconds() / 3600
        
        # Calculate price change
        if direction == 'BUY':
            price_change_pct = (current_price - entry_price) / entry_price * 100
        else:
            price_change_pct = (entry_price - current_price) / entry_price * 100
        
        # Prepare features
        now = datetime.now()
        features = {
            'current_pnl_pct': pnl_pct,
            'hold_time_hours': hold_time_hours,
            'direction_encoded': 1 if direction == 'BUY' else 0,
            'entry_price': entry_price,
            'current_price': current_price,
            'price_change_pct': price_change_pct,
            'hour': now.hour,
            'day_of_week': now.weekday()
        }
        
        # Try ML prediction if model available
        if self.model is not None:
            try:
                X = pd.DataFrame([features])[self.FEATURES]
                pred = self.model.predict(X)[0]
                
                # Convert prediction to recommendation
                if pred > 0.7:  # High probability of improvement
                    recommendation = 'HOLD'
                    confidence = pred
                    reason = f"ML predicts {pred:.1%} chance of improvement"
                    urgency = 'LOW'
                elif pred > 0.5:  # Moderate
                    recommendation = 'TRAILING'
                    confidence = pred
                    reason = f"ML predicts {pred:.1%} chance - consider trailing stop"
                    urgency = 'MEDIUM'
                else:  # Low probability
                    if pnl_pct > 0.02:
                        recommendation = 'TAKE_PROFIT'
                        reason = f"ML predicts {(1-pred):.1%} chance of reversal - take profit"
                    elif pnl_pct < -0.02:
                        recommendation = 'CUT_LOSS'
                        reason = f"ML predicts {(1-pred):.1%} chance of further loss"
                    else:
                        recommendation = 'HOLD'
                        reason = "No clear signal"
                    confidence = 1 - pred
                    urgency = 'HIGH' if pnl_pct < -0.03 else 'MEDIUM'
                
                return {
                    'recommendation': recommendation,
                    'confidence': confidence,
                    'reason': reason,
                    'suggested_action': self._get_action_text(recommendation),
                    'urgency': urgency,
                    'expected_improvement': 0.0,  # Would need regression model
                    'ml_prediction': pred
                }
                
            except Exception as e:
                print(f"[MLExitOptimizer] Prediction failed: {e}")
                # Fall through to rule-based
        
        # Rule-based fallback
        return self._rule_based_exit(pnl_pct, hold_time_hours, direction, price_change_pct)
    
    def _rule_based_exit(self, pnl_pct: float, hold_time_hours: float, 
                         direction: str, price_change_pct: float) -> Dict[str, Any]:
        """
        Rule-based exit logic (fallback when ML model unavailable)
        
        UPDATED 2026-03-22: More aggressive loss cutting based on audit findings
        """
        # Strong profit - take it
        if pnl_pct > 0.03:  # >3% profit
            return {
                'recommendation': 'TAKE_PROFIT',
                'confidence': 0.8,
                'reason': f"Strong profit at {pnl_pct:.1%}",
                'suggested_action': 'Close position for profit',
                'urgency': 'MEDIUM',
                'expected_improvement': 0.0
            }
        
        # Moderate profit - consider trailing
        elif pnl_pct > 0.015:  # >1.5% profit
            return {
                'recommendation': 'TRAILING',
                'confidence': 0.6,
                'reason': f"Moderate profit at {pnl_pct:.1%} - trail to protect",
                'suggested_action': 'Move stop to breakeven or trail',
                'urgency': 'LOW',
                'expected_improvement': 0.01
            }
        
        # Small profit - hold for more
        elif pnl_pct > 0.005:  # >0.5% profit
            return {
                'recommendation': 'HOLD',
                'confidence': 0.5,
                'reason': f"Small profit at {pnl_pct:.1%} - let it run",
                'suggested_action': 'Continue holding',
                'urgency': 'LOW',
                'expected_improvement': 0.02
            }
        
        # Small loss - cut it quickly (UPDATED: was -2%, now -0.5%)
        elif pnl_pct > -0.005:  # <0.5% loss
            if hold_time_hours < 1:
                return {
                    'recommendation': 'HOLD',
                    'confidence': 0.4,
                    'reason': f"Small loss {pnl_pct:.1%}, position is very recent ({hold_time_hours:.1f}h)",
                    'suggested_action': 'Continue holding (recent entry)',
                    'urgency': 'LOW',
                    'expected_improvement': 0.01
                }
            else:
                return {
                    'recommendation': 'CUT_LOSS',
                    'confidence': 0.6,
                    'reason': f"Small loss {pnl_pct:.1%} persisting ({hold_time_hours:.1f}h) - cut early",
                    'suggested_action': 'Cut loss early before it grows',
                    'urgency': 'MEDIUM',
                    'expected_improvement': -0.005
                }
        
        # Moderate loss - urgent exit (UPDATED: was -5%, now -1%)
        elif pnl_pct > -0.01:  # <1% loss
            return {
                'recommendation': 'CUT_LOSS',
                'confidence': 0.7,
                'reason': f"Moderate loss {pnl_pct:.1%} - exit before it grows",
                'suggested_action': 'Close position to limit loss',
                'urgency': 'HIGH',
                'expected_improvement': -0.01
            }
        
        # Large loss - VERY urgent exit
        else:  # >1% loss
            return {
                'recommendation': 'CUT_LOSS',
                'confidence': 0.9,
                'reason': f"Large loss {pnl_pct:.1%} - exit IMMEDIATELY",
                'suggested_action': 'URGENT: Close position now',
                'urgency': 'VERY_HIGH',
                'expected_improvement': -0.02
            }
    
    def _get_action_text(self, recommendation: str) -> str:
        """Get human-readable action text"""
        actions = {
            'HOLD': 'Continue holding position',
            'TAKE_PROFIT': 'Close position for profit',
            'CUT_LOSS': 'Close position to limit loss',
            'TRAILING': 'Move stop loss to protect profit'
        }
        return actions.get(recommendation, 'No action')
    
    def record_exit(self, symbol: str, exit_pnl: float, hold_time: float, 
                   recommendation: str, was_correct: bool):
        """
        Record exit outcome for model improvement
        
        Args:
            symbol: Trading pair
            exit_pnl: Actual PnL at exit
            hold_time: Hold time in hours
            recommendation: What ML recommended
            was_correct: Whether recommendation was correct
        """
        self.exit_history.append({
            'symbol': symbol,
            'exit_pnl': exit_pnl,
            'hold_time': hold_time,
            'recommendation': recommendation,
            'was_correct': was_correct,
            'timestamp': datetime.now()
        })
        
        # Log accuracy
        if len(self.exit_history) >= 10:
            recent = self.exit_history[-20:]
            correct = sum(1 for e in recent if e['was_correct'])
            accuracy = correct / len(recent)
            print(f"[MLExitOptimizer] Exit accuracy (last 20): {accuracy:.1%} ({correct}/{len(recent)})")


# Singleton instance
_optimizer_instance = None

def get_exit_optimizer() -> MLExitOptimizer:
    """Get or create exit optimizer singleton"""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = MLExitOptimizer()
    return _optimizer_instance


# Convenience function
def predict_exit(symbol: str, position: Dict[str, Any]) -> Dict[str, Any]:
    """Get exit prediction for a position"""
    optimizer = get_exit_optimizer()
    return optimizer.predict_exit(symbol, position)


print("[MLExitOptimizer] Module loaded")
