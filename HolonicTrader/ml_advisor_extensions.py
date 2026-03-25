"""
ML Advisor Extended Integration Points

Identifies all locations where ML predictions can enhance decision-making
beyond just position sizing.
"""

# ============================================================================
# 1. ENTRY SIGNAL FILTERING (Pre-Governor)
# ============================================================================
# Location: agent_trader.py or signal generation layer
# Purpose: Filter out low-confidence signals BEFORE they reach Governor
# Benefit: Reduces computational load, prevents bad trades early

"""
# In signal generation loop:
from HolonicTrader.ml_advisor import predict_trade

for signal in signals:
    # ML Pre-filter
    ml_pred = predict_trade(
        symbol=signal.symbol,
        direction=signal.direction,
        price=signal.price,
        quantity=1.0  # Dummy for prediction
    )
    
    # Skip very low confidence signals
    if ml_pred['win_probability'] < 0.35:
        print(f"🤖 ML FILTER: Skipping {signal.symbol} - only {ml_pred['win_probability']:.1%} win chance")
        continue
    
    # Add ML score to signal metadata
    signal.metadata['ml_win_prob'] = ml_pred['win_probability']
    signal.metadata['ml_confidence'] = ml_pred['confidence_level']
    
    # Pass to Governor
    process_signal(signal)
"""

# ============================================================================
# 2. EXIT TIMING OPTIMIZATION
# ============================================================================
# Location: agent_executor.py or exit handler
# Purpose: Predict optimal exit timing based on current PnL and market state
# Benefit: Improve win rate by exiting at better times

"""
# In position monitoring loop:
for symbol, position in positions.items():
    current_pnl = position.pnl_percent
    hold_time = (time.time() - position.entry_time) / 3600  # hours
    
    # ML Exit Advisor
    exit_pred = ml_advisor.predict_exit(
        symbol=symbol,
        current_pnl=current_pnl,
        hold_time_hours=hold_time,
        direction=position.direction,
        entry_price=position.entry_price,
        current_price=position.current_price
    )
    
    # Exit recommendations:
    # - "HOLD" - Continue holding
    # - "TAKE_PROFIT" - Close for gain
    # - "CUT_LOSS" - Close to prevent further loss
    # - "TRAILING" - Move stop to breakeven
    
    if exit_pred['recommendation'] == 'TAKE_PROFIT' and current_pnl > 0.02:
        close_position(symbol)
    elif exit_pred['recommendation'] == 'CUT_LOSS' and current_pnl < -0.02:
        close_position(symbol)
"""

# ============================================================================
# 3. SYMBOL SELECTION / PRIORITIZATION
# ============================================================================
# Location: agent_atlas_strategist.py or portfolio allocation
# Purpose: Rank symbols by ML-predicted opportunity quality
# Benefit: Allocate capital to best opportunities

"""
# In capital allocation logic:
symbol_scores = []

for symbol in watchlist:
    pred = predict_trade(symbol, 'BUY', prices[symbol], 1.0)
    
    symbol_scores.append({
        'symbol': symbol,
        'win_probability': pred['win_probability'],
        'expected_pnl': pred['predicted_pnl_percent'],
        'confidence': pred['confidence_level'],
        'score': pred['win_probability'] * pred['predicted_pnl_percent']  # Combined score
    })

# Sort by ML score
symbol_scores.sort(key=lambda x: x['score'], reverse=True)

# Allocate capital to top N opportunities
top_symbols = symbol_scores[:config.MAX_POSITIONS]
for allocation in top_symbols:
    allocate_capital(allocation['symbol'])
"""

# ============================================================================
# 4. RISK MANAGEMENT ADJUSTMENTS
# ============================================================================
# Location: governor_risk.py or risk management layer
# Purpose: Adjust risk limits based on ML confidence
# Benefit: Dynamic risk management based on market conditions

"""
# In risk calculation:
def calculate_risk_limits(portfolio_state):
    base_risk = config.BASE_RISK_PER_TRADE  # e.g., 2%
    
    # Get ML confidence for recent trades
    recent_ml = ml_performance[-10:] if ml_performance else []
    if recent_ml:
        accuracy = sum(1 for p in recent_ml if p['prediction_correct']) / len(recent_ml)
        
        # Adjust risk based on ML accuracy
        if accuracy > 0.70:
            risk_multiplier = 1.2  # Increase risk when ML is hot
        elif accuracy > 0.55:
            risk_multiplier = 1.0  # Normal risk
        else:
            risk_multiplier = 0.5  # Reduce risk when ML struggling
        
        return base_risk * risk_multiplier
    
    return base_risk
"""

# ============================================================================
# 5. PORTFOLIO ALLOCATION
# ============================================================================
# Location: Atlas or portfolio management layer
# Purpose: Optimize portfolio weights based on ML predictions
# Benefit: Better capital efficiency

"""
# In portfolio optimization:
def optimize_portfolio(symbols, ml_advisor):
    """
    Mean-variance optimization with ML-adjusted expected returns
    """
    expected_returns = {}
    
    for symbol in symbols:
        pred = ml_advisor.predict_trade(symbol, 'BUY', prices[symbol], 1.0)
        
        # ML-adjusted expected return
        # Base expectation adjusted by win probability
        expected_returns[symbol] = (
            pred['win_probability'] * pred['predicted_pnl_percent'] +
            (1 - pred['win_probability']) * (-5.0)  # Assume -5% on loss
        )
    
    # Run optimization with ML-adjusted returns
    weights = mean_variance_optimization(expected_returns, covariance_matrix)
    
    return weights
"""

# ============================================================================
# 6. TRADE FREQUENCY CONTROL
# ============================================================================
# Location: Governor cycle logic
# Purpose: Adjust trade frequency based on ML confidence distribution
# Benefit: Trade more when conditions favorable, less when uncertain

"""
# In trading cycle:
def should_enter_trade(cycle_state):
    # Check recent ML confidence distribution
    recent_predictions = ml_performance[-20:] if ml_performance else []
    
    if not recent_predictions:
        return True  # Default allow
    
    high_conf_ratio = sum(1 for p in recent_predictions if p['confidence'] == 'HIGH') / len(recent_predictions)
    
    # Adjust frequency based on high-confidence opportunities
    if high_conf_ratio > 0.3:  # >30% high confidence
        return True  # Normal trading
    elif high_conf_ratio > 0.15:  # >15% high confidence
        return len(active_positions) < config.MAX_POSITIONS * 0.7  # Reduce to 70% capacity
    else:  # <15% high confidence
        return len(active_positions) < config.MAX_POSITIONS * 0.3  # Only 30% capacity
"""

# ============================================================================
# 7. REGIME-SPECIFIC TUNING
# ============================================================================
# Location: SMCE regime engine or regime detection
# Purpose: Adjust ML thresholds based on market regime
# Benefit: Better performance across different market conditions

"""
# In regime-aware trading:
def get_regime_thresholds(regime):
    """
    Adjust ML thresholds based on market regime
    """
    if regime == 'ORDERED':  # Trending, low volatility
        return {
            'min_win_prob': 0.45,  # Lower threshold (trends reliable)
            'size_multiplier': 1.2,  # Larger positions
            'enable_veto': False   # Don't veto in clear trends
        }
    elif regime == 'TRANSITION':  # Mixed signals
        return {
            'min_win_prob': 0.55,  # Standard threshold
            'size_multiplier': 1.0,  # Normal positions
            'enable_veto': True    # Enable veto
        }
    elif regime == 'CHAOTIC':  # High volatility, unpredictable
        return {
            'min_win_prob': 0.65,  # Higher threshold (only best trades)
            'size_multiplier': 0.5,  # Half size
            'enable_veto': True    # Strict veto
        }
    else:
        return {'min_win_prob': 0.50, 'size_multiplier': 1.0, 'enable_veto': True}
"""

# ============================================================================
# 8. MODEL CONFIDENCE FOR VETO DECISIONS
# ============================================================================
# Location: Governor veto logic
# Purpose: Use ML as additional veto layer
# Benefit: Catch bad trades that pass other filters

"""
# In veto logic:
def ml_veto_check(symbol, direction, price, metadata):
    """
    ML-based veto - can block trades that pass other filters
    """
    pred = predict_trade(symbol, direction, price, 1.0)
    
    # Veto if ML very low confidence
    if pred['win_probability'] < 0.30:
        return {
            'vetoed': True,
            'reason': f"ML_LOW_CONFIDENCE ({pred['win_probability']:.1%} win prob)",
            'confidence': pred['confidence_level']
        }
    
    # Veto if predicted loss > threshold
    if pred['predicted_pnl_percent'] < -10.0:  # Predicted loss > 10%
        return {
            'vetoed': True,
            'reason': f"ML_PREDICTED_LOSS ({pred['predicted_pnl_percent']:.1f}%)",
            'confidence': pred['confidence_level']
        }
    
    return {'vetoed': False}
"""

# ============================================================================
# 9. POST-TRADE ANALYSIS
# ============================================================================
# Location: Performance tracking / analytics
# Purpose: Analyze ML prediction accuracy by symbol, regime, time
# Benefit: Continuous improvement of ML system

"""
# In performance analysis:
def analyze_ml_performance(ml_performance_data):
    """
    Detailed ML performance breakdown
    """
    analysis = {
        'overall_accuracy': 0,
        'by_symbol': {},
        'by_confidence': {},
        'by_regime': {},
        'by_hour': {},
        'calibration': {},
    }
    
    # Overall accuracy
    correct = sum(1 for p in ml_performance_data if p['prediction_correct'])
    analysis['overall_accuracy'] = correct / len(ml_performance_data)
    
    # By symbol
    for symbol in set(p['symbol'] for p in ml_performance_data):
        symbol_preds = [p for p in ml_performance_data if p['symbol'] == symbol]
        symbol_acc = sum(1 for p in symbol_preds if p['prediction_correct']) / len(symbol_preds)
        analysis['by_symbol'][symbol] = {
            'accuracy': symbol_acc,
            'trades': len(symbol_preds)
        }
    
    # By confidence level
    for conf in ['HIGH', 'MEDIUM', 'LOW']:
        conf_preds = [p for p in ml_performance_data if p['predicted_confidence'] == conf]
        if conf_preds:
            conf_acc = sum(1 for p in conf_preds if p['prediction_correct']) / len(conf_preds)
            analysis['by_confidence'][conf] = {
                'accuracy': conf_acc,
                'trades': len(conf_preds)
            }
    
    # Calibration (does 70% predicted win rate actually win 70%?)
    for prob_range in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
        range_preds = [p for p in ml_performance_data if prob_range[0] < p['predicted_win_prob'] <= prob_range[1]]
        if range_preds:
            actual_win_rate = sum(1 for p in range_preds if p['actual_win']) / len(range_preds)
            analysis['calibration'][f"{prob_range[0]:.0f}-{prob_range[1]:.0f}"] = {
                'predicted': (prob_range[0] + prob_range[1]) / 2,
                'actual': actual_win_rate,
                'trades': len(range_preds)
            }
    
    return analysis
"""

# ============================================================================
# 10. ATLAS EDGE AMPLIFICATION
# ============================================================================
# Location: Atlas strategist integration
# Purpose: Combine ML predictions with Atlas edge signals
# Benefit: Enhanced signal quality

"""
# In Atlas integration:
def amplify_atlas_edge(atlas_signal, ml_advisor):
    """
    Combine Atlas edge with ML prediction
    """
    # Get ML prediction
    ml_pred = ml_advisor.predict_trade(
        symbol=atlas_signal.symbol,
        direction=atlas_signal.direction,
        price=atlas_signal.price,
        quantity=1.0
    )
    
    # Combined score
    atlas_edge = atlas_signal.edge_score  # 0-100
    ml_confidence = ml_pred['win_probability']  # 0-1
    
    # Weighted combination
    combined_score = (atlas_edge / 100) * 0.6 + ml_confidence * 0.4
    
    # Amplify if both agree
    if atlas_edge > 70 and ml_confidence > 0.6:
        amplification = 1.5  # 50% size boost
    elif atlas_edge > 50 and ml_confidence > 0.5:
        amplification = 1.0  # Normal size
    else:
        amplification = 0.5  # Reduce size
    
    return {
        'combined_score': combined_score,
        'amplification': amplification,
        'action': 'AMPLIFY' if amplification > 1.0 else 'REDUCE'
    }
"""

print("ML Advisor Extended Integration Points documented")
print("10 integration points identified across the trading system")
