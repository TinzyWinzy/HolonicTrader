"""
ML Advisor Integration Patch for Governor

Integrates ML predictions into Governor's position sizing and veto logic.

Usage:
    Apply this patch to agent_governor.py or import MLTradingAdvisor directly.
"""

# Add to top of agent_governor.py:
"""
# === ML ADVISOR INTEGRATION (Add after other imports) ===
try:
    from HolonicTrader.ml_advisor import get_ml_advisor, MLTradingAdvisor
    ML_ENABLED = True
except ImportError:
    ML_ENABLED = False
    print("[Governor] ML Advisor not available - trading without ML guidance")
# =========================================================
"""

# Add to Governor's __init__ method:
"""
# ML Advisor initialization
if ML_ENABLED:
    try:
        self.ml_advisor = get_ml_advisor()
        print(f"[{self.name}] ML Advisor initialized")
    except Exception as e:
        print(f"[{self.name}] ML Advisor init failed: {e}")
        self.ml_advisor = None
else:
    self.ml_advisor = None
"""

# Add to calc_position_size method (before final size calculation):
"""
# === ML-BASED POSITION SIZING ADJUSTMENT ===
if self.ml_advisor is not None:
    try:
        # Get ML prediction for this trade
        ml_prediction = self.ml_advisor.predict_trade(
            symbol=symbol,
            direction=direction,
            price=asset_price,
            quantity=final_notional / asset_price if asset_price > 0 else 0,
            cost_usd=final_notional,
            entropy=metadata.get('entropy') if metadata else None,
            regime=metadata.get('regime') if metadata else None,
            conviction=conviction,
        )
        
        # Adjust position based on ML confidence
        win_prob = ml_prediction.get('win_probability', 0.5)
        confidence_level = ml_prediction.get('confidence_level', 'MEDIUM')
        
        # High confidence win - allow full size
        if win_prob > 0.6 and confidence_level == 'HIGH':
            ml_adjustment = 1.0  # Full size
            print(f"[{self.name}] 🤖 ML HIGH CONFIDENCE: {win_prob:.1%} win prob - allowing full size")
        
        # Moderate confidence - reduce size
        elif win_prob > 0.5:
            ml_adjustment = 0.7  # 70% size
            print(f"[{self.name}] 🤖 ML MODERATE: {win_prob:.1%} win prob - reducing to 70%")
        
        # Low confidence - significant reduction or skip
        elif win_prob > 0.4:
            ml_adjustment = 0.3  # 30% size
            print(f"[{self.name}] 🤖 ML LOW CONFIDENCE: {win_prob:.1%} - reducing to 30%")
        else:
            # Very low confidence - recommend skip
            print(f"[{self.name}] 🤖 ML VERY LOW: {win_prob:.1%} - recommending SKIP")
            
            # Only proceed if risk-reducing or override
            if not is_risk_reducing and not is_override:
                # Apply heavy penalty to discourage trade
                ml_adjustment = 0.1  # 10% size as strong signal
            else:
                ml_adjustment = 0.5  # 50% for risk-reducing
        
        # Apply ML adjustment
        final_notional *= ml_adjustment
        
        # Log ML guidance
        if hasattr(self, 'ml_trade_log'):
            self.ml_trade_log.append({
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'win_prob': win_prob,
                'confidence': confidence_level,
                'adjustment': ml_adjustment,
            })
        
    except Exception as e:
        print(f"[{self.name}] ML position sizing failed: {e}")
        # Continue without ML adjustment on error
# ============================================
"""

# Add to veto logic (before final veto decision):
"""
# === ML-BASED VETO CHECK ===
if self.ml_advisor is not None and not is_override and not is_risk_reducing:
    try:
        ml_prediction = self.ml_advisor.predict_trade(
            symbol=symbol,
            direction=direction,
            price=asset_price,
            quantity=1.0,  # Dummy quantity for prediction
            cost_usd=100.0,  # Dummy cost
            conviction=conviction,
        )
        
        win_prob = ml_prediction.get('win_probability', 0.5)
        
        # Veto if ML very low confidence
        if win_prob < 0.3:
            print(f"[{self.name}] 🤖 ML VETO: {win_prob:.1%} win probability - too low")
            return self._record_veto(symbol, "ML_LOW_CONFIDENCE", metadata)
    
    except Exception as e:
        print(f"[{self.name}] ML veto check failed: {e}")
        # Continue without ML veto on error
# ===========================
"""

# Add to open_position method (for trade tracking):
"""
# === TRACK ML PREDICTION VS ACTUAL ===
if self.ml_advisor is not None:
    try:
        # Store prediction for later validation
        if not hasattr(self, 'ml_predictions'):
            self.ml_predictions = {}
        
        self.ml_predictions[symbol] = {
            'entry_time': datetime.now(),
            'entry_price': entry_price,
            'direction': direction,
        }
    except Exception as e:
        print(f"[{self.name}] ML tracking setup failed: {e}")
# =====================================
"""

# Add to close_position or register_trade_outcome:
"""
# === RECORD ML PREDICTION ACCURACY ===
if self.ml_advisor is not None and hasattr(self, 'ml_predictions'):
    if symbol in self.ml_predictions:
        try:
            pred = self.ml_predictions[symbol]
            actual_pnl = pnl_pct  # Already in register_trade_outcome
            
            # Compare prediction vs actual
            print(f"[{self.name}] 🤖 ML PREDICTION vs ACTUAL for {symbol}:")
            print(f"    Entry: ${pred['entry_price']:.4f} | PnL: {actual_pnl:.2f}%")
            
            # Store for batch analysis
            if not hasattr(self, 'ml_performance'):
                self.ml_performance = []
            
            self.ml_performance.append({
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'predicted_win_prob': pred.get('win_prob', 0.5),
                'actual_pnl': actual_pnl,
                'actual_win': actual_pnl > 0,
            })
            
            # Clean up
            del self.ml_predictions[symbol]
        except Exception as e:
            print(f"[{self.name}] ML performance tracking failed: {e}")
# =====================================
"""

# Add to get_status method:
"""
# ML Advisor status
if ML_ENABLED and self.ml_advisor is not None:
    status['ml_advisor'] = {
        'enabled': True,
        'models_loaded': self.ml_advisor.get_model_status(),
        'predictions_made': len(getattr(self, 'ml_predictions', {})),
        'performance_tracked': len(getattr(self, 'ml_performance', [])),
    }
    
    # Calculate ML accuracy if we have performance data
    if hasattr(self, 'ml_performance') and len(self.ml_performance) > 0:
        perf = self.ml_performance[-20:]  # Last 20 trades
        correct = sum(1 for p in perf if (p['predicted_win_prob'] > 0.5) == p['actual_win'])
        status['ml_advisor']['recent_accuracy'] = correct / len(perf) if perf else 0
else:
    status['ml_advisor'] = {'enabled': False}
"""

print("[ML Governor Patch] Ready to apply - see comments for integration points")
