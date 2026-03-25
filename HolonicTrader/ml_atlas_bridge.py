"""
ML-Atlas Bridge - Resolves conflicts between ML predictions and Atlas filters

Problem: ML says 79.8% win probability, but Atlas vetoes due to low volatility
Solution: Combine ML confidence with Atlas edge scoring for unified decision
"""

from typing import Dict, Any, Tuple

class MLAtlasBridge:
    """
    Bridges ML Advisor with Atlas Profit Filter
    
    Use Cases:
    1. ML high confidence + Atlas veto → Override with reduced size
    2. ML low confidence + Atlas approve → Reduce size or skip
    3. Both agree → Full confidence
    """
    
    def __init__(self, ml_advisor=None, atlas_filter=None):
        self.ml_advisor = ml_advisor
        # Atlas filter might be nested (atlas.profit_filter)
        self.atlas_filter = None
        if atlas_filter:
            if hasattr(atlas_filter, 'evaluate_trade'):
                self.atlas_filter = atlas_filter
            elif hasattr(atlas_filter, 'profit_filter') and hasattr(atlas_filter.profit_filter, 'evaluate_trade'):
                self.atlas_filter = atlas_filter.profit_filter
        
        # Configuration
        self.ml_override_threshold = 0.75  # ML win prob to consider override
        self.min_atlas_edge = 0.008  # Minimum Atlas edge
        self.volatility_flex = 0.003  # Allow volatility flex down to 0.3%
        
    def evaluate_trade(self, symbol: str, direction: str, price: float,
                      quantity: float, market_data: Dict[str, Any],
                      signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combined ML + Atlas evaluation
        
        Returns: {
            'approved': bool,
            'reason': str,
            'size_adjustment': float,
            'confidence': str,
            'ml_win_prob': float,
            'atlas_edge': float,
            'combined_score': float
        }
        """
        result = {
            'symbol': symbol,
            'approved': False,
            'reason': '',
            'size_adjustment': 1.0,
            'confidence': 'LOW',
        }
        
        # === ML PREDICTION WITH CACHING (2026-03-22) ===
        # Prevent duplicate ML calls - use cached result if available
        try:
            from .signal_quality_gate import get_signal_quality_gate
            
            gate = get_signal_quality_gate()
            
            # Prepare signal for caching
            cache_signal = {
                'symbol': symbol,
                'direction': direction,
                'price': price,
                'quantity': quantity,
            }
            
            # Try to get cached ML result
            ml_pred = gate._get_ml_confidence_cached(cache_signal)
            
            if not ml_pred:
                # Not cached - fetch fresh and cache
                ml_pred = self.ml_advisor.predict_trade(
                    symbol=symbol,
                    direction=direction,
                    price=price,
                    quantity=quantity,
                    cost_usd=price * quantity,
                )
                # Cache for later use (Governor will use same cached value)
                gate.cache_ml_result(cache_signal, ml_pred)
            
            result['ml_win_prob'] = ml_pred['win_probability']
            result['ml_confidence'] = ml_pred['confidence_level']
            result['ml_predicted_pnl'] = ml_pred['predicted_pnl_percent']
            
        except Exception as e:
            # Fallback if gate not available
            if self.ml_advisor:
                ml_pred = self.ml_advisor.predict_trade(
                    symbol=symbol,
                    direction=direction,
                    price=price,
                    quantity=quantity,
                    cost_usd=price * quantity,
                )
                result['ml_win_prob'] = ml_pred['win_probability']
                result['ml_confidence'] = ml_pred['confidence_level']
            else:
                result['ml_win_prob'] = 0.5
                result['ml_confidence'] = 'UNKNOWN'
        # ============================================
        
        # 2. Get Atlas evaluation
        if self.atlas_filter:
            atlas_approved, atlas_reason, atlas_meta = self.atlas_filter.evaluate_trade(
                signal_data=signal_data,
                market_data=market_data,
                portfolio_state={}
            )
            result['atlas_approved'] = atlas_approved
            result['atlas_reason'] = atlas_reason
            result['atlas_edge'] = atlas_meta.get('expected_profit_pct', 0.0)
            result['atlas_position_size'] = atlas_meta.get('position_size_usd', 0.0)
        else:
            result['atlas_approved'] = True
            result['atlas_reason'] = 'NO_ATLAS'
            result['atlas_edge'] = 0.02  # Default edge
            result['atlas_position_size'] = price * quantity
        
        # 3. Resolve conflicts
        ml_high_conf = result['ml_win_prob'] >= self.ml_override_threshold
        atlas_veto = not result['atlas_approved']
        atlas_volatility_veto = 'INSUFFICIENT_VOLATILITY' in result['atlas_reason']
        
        # Scenario 1: ML HIGH confidence + Atlas volatility veto → OVERRIDE with reduced size
        if ml_high_conf and atlas_veto and atlas_volatility_veto:
            result['approved'] = True
            result['reason'] = f'ML_OVERRIDE_VOL_VETO (ML {result["ml_win_prob"]:.1%})'
            result['size_adjustment'] = 0.5  # 50% size due to low vol
            result['confidence'] = 'MEDIUM'  # Downgraded from HIGH
            
            print(f"🤖🗺️ ML-ATLAS BRIDGE: ML override - {symbol} high confidence ({result['ml_win_prob']:.1%}) but low volatility")
            print(f"   → Approving with 50% size reduction")
            
        # Scenario 2: Both approve → Full confidence
        elif result['ml_win_prob'] > 0.5 and result['atlas_approved']:
            result['approved'] = True
            result['reason'] = 'ML_ATLAS_AGREE'
            result['size_adjustment'] = 1.0
            
            if result['ml_win_prob'] > 0.65 and result['atlas_edge'] > 0.01:
                result['confidence'] = 'HIGH'
                print(f"🤖🗺️ ML-ATLAS BRIDGE: Strong agreement on {symbol}")
            else:
                result['confidence'] = 'MEDIUM'
                print(f"🤖🗺️ ML-ATLAS BRIDGE: Moderate agreement on {symbol}")
        
        # Scenario 3: ML low confidence + Atlas approve → Reduce size
        elif result['ml_win_prob'] < 0.45 and result['atlas_approved']:
            result['approved'] = True
            result['reason'] = 'ATLAS_OVERRIDE_ML_LOW_CONF'
            result['size_adjustment'] = 0.3  # 30% size
            result['confidence'] = 'LOW'
            print(f"🤖🗺️ ML-ATLAS BRIDGE: Atlas approve but ML low confidence ({result['ml_win_prob']:.1%})")
            print(f"   → Approving with 30% size")
        
        # Scenario 4: Both reject → Hard veto
        elif result['ml_win_prob'] < 0.45 and atlas_veto:
            result['approved'] = False
            result['reason'] = f'ML_ATLAS_REJECT (ML {result["ml_win_prob"]:.1%}, Atlas {result["atlas_reason"]})'
            result['size_adjustment'] = 0.0
            result['confidence'] = 'VERY_LOW'
            print(f"🤖🗺️ ML-ATLAS BRIDGE: Both reject {symbol}")
        
        # Scenario 5: ML approve + Atlas other veto → Case by case
        elif ml_high_conf and atlas_veto and not atlas_volatility_veto:
            # Check if veto reason is serious
            serious_vetoes = ['BLACKLIST', 'LIQUIDITY', 'SPREAD']
            is_serious = any(veto in result['atlas_reason'] for veto in serious_vetoes)
            
            if is_serious:
                result['approved'] = False
                result['reason'] = f'SERIOUS_ATLAS_VETO: {result["atlas_reason"]}'
                result['size_adjustment'] = 0.0
                print(f"🤖🗺️ ML-ATLAS BRIDGE: Serious Atlas veto overrides ML: {result['atlas_reason']}")
            else:
                result['approved'] = True
                result['reason'] = f'ML_OVERRIDE_ATLAS ({result["ml_win_prob"]:.1%})'
                result['size_adjustment'] = 0.4  # 40% size
                result['confidence'] = 'LOW'
                print(f"🤖🗺️ ML-ATLAS BRIDGE: ML override non-serious Atlas veto")
        
        # Default: Follow Atlas
        else:
            result['approved'] = result['atlas_approved']
            result['reason'] = result['atlas_reason'] if atlas_veto else 'ATLAS_APPROVED'
            result['size_adjustment'] = 1.0 if result['atlas_approved'] else 0.0
        
        # Calculate combined score
        result['combined_score'] = (
            result['ml_win_prob'] * 0.5 +  # 50% from ML
            (result['atlas_edge'] / 0.02) * 0.3 +  # 30% from Atlas edge (normalized to 2%)
            (1.0 if result['approved'] else 0.0) * 0.2  # 20% from approval
        )
        
        return result
    
    def get_volatility_flex(self, current_volatility: float) -> float:
        """
        Calculate flexible volatility threshold based on ML confidence
        
        Args:
            current_volatility: Market volatility (e.g., 0.0005 = 0.05%)
            
        Returns:
            Adjusted volatility threshold
        """
        # Base threshold
        base_threshold = 0.005  # 0.5%
        
        # If ML very confident, allow lower volatility
        if hasattr(self, '_last_ml_pred') and self._last_ml_pred:
            ml_conf = self._last_ml_pred.get('win_probability', 0.5)
            
            if ml_conf > 0.75:
                # High confidence → allow down to 0.3% vol
                return max(0.003, base_threshold * (1.0 - (ml_conf - 0.5)))
        
        return base_threshold


# Convenience function for Governor integration
def check_ml_atlas_consensus(ml_advisor, atlas_filter, symbol, direction, price, quantity, market_data, signal_data):
    """
    Quick consensus check for Governor integration
    
    Returns: (approved, size_multiplier, reason)
    """
    bridge = MLAtlasBridge(ml_advisor, atlas_filter)
    result = bridge.evaluate_trade(symbol, direction, price, quantity, market_data, signal_data)
    
    return (
        result['approved'],
        result['size_adjustment'],
        result['reason']
    )


print("ML-Atlas Bridge loaded - Ready to resolve conflicts")
