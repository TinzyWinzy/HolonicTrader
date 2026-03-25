"""
ML Model Integration Module for HolonicTrader

Provides model predictions for:
1. Win/Loss classification
2. PnL regression
3. Position sizing recommendations

Integrates with Governor and Executor for real-time trading decisions.
"""
import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

class MLTradingAdvisor:
    """
    ML-based trading advisor that provides predictions for entry/exit decisions.
    
    Usage:
        advisor = MLTradingAdvisor()
        prediction = advisor.predict_trade(symbol, direction, price, quantity)
        
        if prediction['win_probability'] > 0.6:
            # High confidence trade
            execute_trade()
    """
    
    def __init__(self, 
                 clf_model_path: str = 'models/lgbm_win_classifier_augmented.pkl',
                 reg_model_path: str = 'models/lgbm_pnl_regression_augmented.pkl',
                 db_summary_path: str = 'datasets/db_trades_summary.json'):
        """
        Initialize ML advisor with trained models.
        
        Args:
            clf_model_path: Path to win/loss classifier model (default: augmented)
            reg_model_path: Path to PnL regression model (default: augmented)
            db_summary_path: Path to database summary statistics
        """
        self.clf_model = None
        self.reg_model = None
        self.db_stats = None
        self.feature_importance = None
        
        # Feature list from training (augmented)
        self.FEATURES = ['quantity', 'price', 'cost_usd', 'hour', 'day_of_week', 
                         'mfe', 'mae', 'direction_encoded']
        
        # Load models
        self._load_models(clf_model_path, reg_model_path)
        self._load_stats(db_summary_path)
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_ttl = 60  # seconds
        
        print(f"[MLTradingAdvisor] Initialized with {len(self.FEATURES)} features")
        if self.db_stats:
            print(f"[MLTradingAdvisor] Database stats: {self.db_stats.get('total_trades', 0)} historical trades")
    
    def _load_models(self, clf_path: str, reg_path: str):
        """Load trained models from disk."""
        try:
            if os.path.exists(clf_path):
                self.clf_model = joblib.load(clf_path)
                print(f"[MLTradingAdvisor] Loaded classifier: {clf_path}")
            else:
                print(f"[MLTradingAdvisor] WARNING: Classifier not found: {clf_path}")
            
            if os.path.exists(reg_path):
                self.reg_model = joblib.load(reg_path)
                print(f"[MLTradingAdvisor] Loaded regression: {reg_path}")
            else:
                print(f"[MLTradingAdvisor] WARNING: Regression model not found: {reg_path}")
        except Exception as e:
            print(f"[MLTradingAdvisor] ERROR loading models: {e}")
    
    def _load_stats(self, stats_path: str):
        """Load database summary statistics."""
        try:
            if os.path.exists(stats_path):
                import json
                with open(stats_path, 'r') as f:
                    self.db_stats = json.load(f)
        except Exception as e:
            print(f"[MLTradingAdvisor] WARNING: Could not load stats: {e}")
    
    def _prepare_features(self, symbol: str, direction: str, price: float, 
                         quantity: float, cost_usd: float = None,
                         entropy: float = None, regime: str = None,
                         conviction: float = None) -> pd.DataFrame:
        """
        Prepare feature vector for model prediction.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            direction: 'BUY' or 'SELL'
            price: Current price
            quantity: Position quantity
            cost_usd: Total cost in USD
            entropy: Market entropy score (optional)
            regime: Market regime (optional)
            conviction: Oracle conviction (optional)
            
        Returns:
            DataFrame with features in correct format
        """
        now = datetime.now()
        
        # Calculate cost if not provided
        if cost_usd is None:
            cost_usd = price * quantity
        
        # Base features
        features = {
            'quantity': quantity,
            'price': price,
            'cost_usd': cost_usd,
            'hour': now.hour,
            'day_of_week': now.weekday(),
            'mfe': 0.0,  # Will be updated during trade
            'mae': 0.0,  # Will be updated during trade
            'direction_encoded': 1 if direction == 'BUY' else 0,
        }
        
        # Add optional features if available
        if entropy is not None:
            features['entropy_score'] = entropy
        if regime is not None:
            # One-hot encode regime
            for r in ['ORDERED', 'TRANSITION', 'CHAOTIC']:
                features[f'regime_{r}'] = 1 if regime == r else 0
        if conviction is not None:
            features['conviction'] = conviction
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Ensure all required columns exist
        for col in self.FEATURES:
            if col not in df.columns:
                df[col] = 0.0
        
        # Select only required features in correct order
        return df[self.FEATURES]
    
    def predict_trade(self, symbol: str, direction: str, price: float,
                     quantity: float, cost_usd: float = None,
                     entropy: float = None, regime: str = None,
                     conviction: float = None,
                     min_confidence: float = 0.5) -> Dict[str, Any]:
        """
        Predict trade outcome.
        
        Args:
            symbol: Trading pair
            direction: 'BUY' or 'SELL'
            price: Current price
            quantity: Position quantity
            cost_usd: Total cost
            entropy: Market entropy
            regime: Market regime
            conviction: Oracle conviction
            min_confidence: Minimum confidence to recommend trade
            
        Returns:
            Dictionary with predictions and recommendation
        """
        # Check cache
        cache_key = f"{symbol}_{direction}_{price}_{quantity}_{datetime.now().strftime('%Y-%m-%d %H')}"
        if cache_key in self.prediction_cache:
            cached = self.prediction_cache[cache_key]
            if (datetime.now().timestamp() - cached['timestamp']) < self.cache_ttl:
                return cached['prediction']
        
        # Prepare features
        X = self._prepare_features(
            symbol, direction, price, quantity, cost_usd,
            entropy, regime, conviction
        )
        
        # Predict
        result = {
            'symbol': symbol,
            'direction': direction,
            'price': price,
            'quantity': quantity,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Win/Loss classification
        if self.clf_model is not None:
            win_prob = float(self.clf_model.predict(X)[0])
            result['win_probability'] = win_prob
            result['predicted_class'] = 'WIN' if win_prob > 0.5 else 'LOSS'
        else:
            result['win_probability'] = 0.5
            result['predicted_class'] = 'UNKNOWN'
        
        # PnL regression
        if self.reg_model is not None:
            pnl_pred = float(self.reg_model.predict(X)[0])
            result['predicted_pnl_percent'] = pnl_pred
            result['predicted_pnl_usd'] = pnl_pred * cost_usd / 100 if cost_usd else None
        else:
            result['predicted_pnl_percent'] = 0.0
            result['predicted_pnl_usd'] = 0.0
        
        # Recommendation
        confidence = result['win_probability']
        if confidence >= min_confidence + 0.1:  # High confidence
            result['recommendation'] = 'STRONG_BUY' if direction == 'BUY' else 'STRONG_SELL'
            result['confidence_level'] = 'HIGH'
        elif confidence >= min_confidence:  # Moderate confidence
            result['recommendation'] = 'BUY' if direction == 'BUY' else 'SELL'
            result['confidence_level'] = 'MEDIUM'
        else:  # Low confidence
            result['recommendation'] = 'SKIP'
            result['confidence_level'] = 'LOW'
        
        # Position sizing recommendation
        if self.db_stats:
            base_size = self.db_stats.get('pnl_stats', {}).get('win_rate', 0.5)
            # Scale position by confidence
            size_multiplier = confidence * 2  # 1.0 at 50% confidence
            result['recommended_size_pct'] = min(base_size * size_multiplier, 0.10)  # Max 10%
        else:
            result['recommended_size_pct'] = 0.05 * confidence
        
        # Cache prediction
        self.prediction_cache[cache_key] = {
            'timestamp': datetime.now().timestamp(),
            'prediction': result
        }
        
        return result
    
    def update_trade_tracking(self, symbol: str, entry_price: float, 
                             current_price: float, direction: str):
        """
        Update MFE/MAE tracking for active trades.
        
        Args:
            symbol: Trading pair
            entry_price: Trade entry price
            current_price: Current market price
            direction: 'BUY' or 'SELL'
        """
        # Calculate MFE (Maximum Favorable Excursion)
        if direction == 'BUY':
            mfe = (max(entry_price, current_price) - entry_price) / entry_price
            mae = (entry_price - min(entry_price, current_price)) / entry_price
        else:
            mfe = (entry_price - min(entry_price, current_price)) / entry_price
            mae = (max(entry_price, current_price) - entry_price) / entry_price
        
        # Update cache with MFE/MAE
        cache_key = f"{symbol}_{direction}_{entry_price}"
        if cache_key in self.prediction_cache:
            self.prediction_cache[cache_key]['mfe'] = mfe
            self.prediction_cache[cache_key]['mae'] = mae
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get model status and statistics."""
        return {
            'classifier_loaded': self.clf_model is not None,
            'regression_loaded': self.reg_model is not None,
            'features_used': len(self.FEATURES),
            'feature_list': self.FEATURES,
            'database_trades': self.db_stats.get('total_trades', 0) if self.db_stats else 0,
            'database_win_rate': self.db_stats.get('pnl_stats', {}).get('win_rate', 0) if self.db_stats else 0,
            'cache_size': len(self.prediction_cache),
        }


# Singleton instance
_advisor_instance = None

def get_ml_advisor() -> MLTradingAdvisor:
    """Get or create ML advisor singleton."""
    global _advisor_instance
    if _advisor_instance is None:
        _advisor_instance = MLTradingAdvisor()
    return _advisor_instance


# Convenience functions for direct integration
def predict_trade(symbol: str, direction: str, price: float, quantity: float,
                 **kwargs) -> Dict[str, Any]:
    """Predict trade outcome using ML models."""
    advisor = get_ml_advisor()
    return advisor.predict_trade(symbol, direction, price, quantity, **kwargs)


def get_model_status() -> Dict[str, Any]:
    """Get model status."""
    advisor = get_ml_advisor()
    return advisor.get_model_status()
