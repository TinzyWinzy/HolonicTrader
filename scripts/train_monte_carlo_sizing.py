"""
Monte Carlo Position Sizing Training & Optimization

Analyzes historical trades to optimize:
1. Position sizing parameters (Kelly fraction, volatility scalar)
2. Risk limits (max drawdown, CVaR thresholds)
3. Monte Carlo simulation parameters (paths, horizons)

Uses actual trade outcomes to find optimal position sizing strategy.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json

BASE = os.path.join(os.path.dirname(__file__), '..')
COMPLETE_TRADES = os.path.join(BASE, 'datasets', 'complete_trades_v2.parquet')
ML_FEATURES = os.path.join(BASE, 'datasets', 'ml_training_features_v2.parquet')
REPORTS_DIR = os.path.join(BASE, 'reports')

os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 70)
print("MONTE CARLO POSITION SIZING OPTIMIZATION")
print("=" * 70)

# Load trade data
if not os.path.exists(COMPLETE_TRADES):
    print(f'Missing: {COMPLETE_TRADES}')
    raise SystemExit(1)

trades_df = pd.read_parquet(COMPLETE_TRADES)
print(f"Loaded {len(trades_df)} complete trades")

# Load features if available
if os.path.exists(ML_FEATURES):
    ml_df = pd.read_parquet(ML_FEATURES)
    print(f"Loaded {len(ml_df)} samples with features")
else:
    ml_df = trades_df.copy()

print("\n" + "=" * 70)
print("HISTORICAL TRADE ANALYSIS")
print("=" * 70)

# Basic statistics
pnl_series = trades_df['exit_pnl_percent'] / 100  # Convert to decimal

print(f"\nPnL Statistics:")
print(f"  Mean: {pnl_series.mean():.4f} ({pnl_series.mean()*100:.2f}%)")
print(f"  Std:  {pnl_series.std():.4f} ({pnl_series.std()*100:.2f}%)")
print(f"  Min:  {pnl_series.min():.4f} ({pnl_series.min()*100:.2f}%)")
print(f"  Max:  {pnl_series.max():.4f} ({pnl_series.max()*100:.2f}%)")

win_mask = pnl_series > 0
win_rate = win_mask.mean()
avg_win = pnl_series[win_mask].mean() if win_mask.any() else 0
avg_loss = pnl_series[~win_mask].mean() if (~win_mask).any() else 0

print(f"\nWin Rate: {win_rate:.2%}")
print(f"Avg Win:  {avg_win:.4f} ({avg_win*100:.2f}%)")
print(f"Avg Loss: {avg_loss:.4f} ({avg_loss*100:.2f}%)")

# Kelly Criterion
if avg_loss != 0:
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    kelly_fraction = win_rate - (1 - win_rate) / win_loss_ratio if win_loss_ratio > 0 else 0
    kelly_fraction = max(0, min(kelly_fraction, 1))  # Clamp to [0, 1]
    
    print(f"\nKelly Criterion:")
    print(f"  Win/Loss Ratio: {win_loss_ratio:.3f}")
    print(f"  Kelly Fraction: {kelly_fraction:.4f} ({kelly_fraction*100:.2f}% of equity)")
    print(f"  Half-Kelly:     {kelly_fraction/2:.4f} ({kelly_fraction/2*100:.2f}% of equity)")
else:
    kelly_fraction = 0
    print("\nCannot calculate Kelly (no winning trades)")

print("\n" + "=" * 70)
print("MONTE CARLO SIMULATION OPTIMIZATION")
print("=" * 70)

# Simulate different position sizing strategies
initial_equity = 100.0
n_simulations = 1000

def simulate_strategy(pnl_series: pd.Series, sizing_method: str, 
                     fraction: float, initial_equity: float = 100.0,
                     n_paths: int = 100) -> Dict[str, float]:
    """
    Simulate trading strategy with different position sizing.
    
    Args:
        pnl_series: Historical PnL percentages (decimal)
        sizing_method: 'fixed', 'kelly', 'volatility_scaled'
        fraction: Position size fraction (for fixed) or Kelly fraction
        initial_equity: Starting equity
        n_paths: Number of simulation paths
    
    Returns:
        Dictionary with simulation statistics
    """
    np.random.seed(42)
    n_trades = len(pnl_series)
    
    final_equities = []
    max_drawdowns = []
    sharpe_ratios = []
    
    for _ in range(n_paths):
        # Sample trades with replacement
        sampled_pnl = pnl_series.sample(n=n_trades, replace=True).values
        
        equity = initial_equity
        equity_curve = [equity]
        peak = equity
        
        for pnl in sampled_pnl:
            if sizing_method == 'fixed':
                position_size = equity * fraction
            elif sizing_method == 'kelly':
                position_size = equity * fraction  # Kelly fraction already computed
            elif sizing_method == 'volatility_scaled':
                # Scale by recent volatility (simplified)
                vol = np.std(sampled_pnl[:max(1, len(sampled_pnl)//4)])
                target_vol = 0.02  # 2% target volatility
                vol_scalar = target_vol / max(vol, 0.001)
                position_size = equity * fraction * vol_scalar
            else:
                position_size = equity * 0.1  # Default 10%
            
            # Apply PnL
            equity_change = position_size * pnl
            equity += equity_change
            equity_curve.append(equity)
            
            # Track drawdown
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_drawdowns.append(drawdown)
        
        final_equities.append(equity)
        
        # Calculate Sharpe ratio
        returns = np.diff(equity_curve) / equity_curve[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        else:
            sharpe = 0
        sharpe_ratios.append(sharpe)
    
    return {
        'final_equity_mean': np.mean(final_equities),
        'final_equity_std': np.std(final_equities),
        'final_equity_median': np.median(final_equities),
        'max_drawdown_mean': np.mean(max_drawdowns),
        'max_drawdown_95pct': np.percentile(max_drawdowns, 95),
        'sharpe_mean': np.mean(sharpe_ratios),
        'sharpe_median': np.median(sharpe_ratios),
        'profitable_paths': np.mean([e > initial_equity for e in final_equities]),
    }

# Test different sizing fractions
print("\nTesting Position Sizing Strategies:")
print("-" * 70)

fractions = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
results = []

for frac in fractions:
    result = simulate_strategy(pnl_series, 'fixed', frac, initial_equity, n_paths=500)
    result['fraction'] = frac
    results.append(result)
    
    print(f"Fraction {frac:.0%}: "
          f"Final Equity ${result['final_equity_median']:.1f}, "
          f"Max DD {result['max_drawdown_95pct']:.1%}, "
          f"Sharpe {result['sharpe_median']:.2f}, "
          f"Win% {result['profitable_paths']:.1%}")

# Find optimal fraction (maximize Sharpe while limiting drawdown)
max_dd_limit = 0.20  # 20% max drawdown limit
valid_results = [r for r in results if r['max_drawdown_95pct'] <= max_dd_limit]

if valid_results:
    optimal = max(valid_results, key=lambda x: x['sharpe_median'])
    print(f"\n✓ Optimal Fraction: {optimal['fraction']:.0%} "
          f"(Sharpe={optimal['sharpe_median']:.2f}, "
          f"DD={optimal['max_drawdown_95pct']:.1%})")
else:
    # If all exceed DD limit, pick lowest
    optimal = min(results, key=lambda x: x['max_drawdown_95pct'])
    print(f"⚠ All fractions exceed {max_dd_limit:.0%} DD limit. "
          f"Most conservative: {optimal['fraction']:.0%}")

print("\n" + "=" * 70)
print("CVAR ANALYSIS (Conditional Value at Risk)")
print("=" * 70)

# Calculate CVaR at different confidence levels
def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall)."""
    var = np.percentile(returns, (1 - confidence) * 100)
    cvar = returns[returns <= var].mean() if (returns <= var).any() else var
    return abs(cvar)

returns = pnl_series.values
cvar_95 = calculate_cvar(returns, 0.95)
cvar_99 = calculate_cvar(returns, 0.99)

print(f"\nCVaR (95%): {cvar_95:.4f} ({cvar_95*100:.2f}%)")
print(f"CVaR (99%): {cvar_99:.4f} ({cvar_99*100:.2f}%)")

# Recommended CVaR limits for Monte Carlo veto
recommended_cvar_limit = min(cvar_95 * 2, 0.08)  # 2x observed or 8%, whichever is lower
print(f"\nRecommended CVaR Limit for Monte Carlo Veto: {recommended_cvar_limit:.4f} ({recommended_cvar_limit*100:.2f}%)")

print("\n" + "=" * 70)
print("RECOMMENDED MONTE CARLO PARAMETERS")
print("=" * 70)

# Optimal parameters based on analysis
optimal_params = {
    'position_size_fraction': float(optimal['fraction']),
    'kelly_fraction': float(kelly_fraction),
    'half_kelly': float(kelly_fraction / 2),
    'max_drawdown_limit': float(optimal['max_drawdown_95pct'] * 1.2),  # 20% buffer
    'cvar_95_limit': float(recommended_cvar_limit),
    'cvar_99_limit': float(cvar_99 * 1.5),
    'sharpe_target': float(optimal['sharpe_median']),
    'volatility_target': 0.02,  # 2% daily volatility
    'monte_carlo_paths': 1000,
    'monte_carlo_horizon_hours': 24,
    'stress_test_paths': 5000,
    'stress_test_adverse_move': 0.03,  # 3% adverse move
}

print(f"""
Position Sizing:
  • Fixed Fraction:     {optimal_params['position_size_fraction']:.1%} of equity
  • Kelly Fraction:     {optimal_params['kelly_fraction']:.1%} of equity
  • Half-Kelly:         {optimal_params['half_kelly']:.1%} of equity (recommended)

Risk Limits:
  • Max Drawdown:       {optimal_params['max_drawdown_limit']:.1%}
  • CVaR (95%):         {optimal_params['cvar_95_limit']:.1%}
  • CVaR (99%):         {optimal_params['cvar_99_limit']:.1%}

Monte Carlo Settings:
  • Simulation Paths:   {optimal_params['monte_carlo_paths']}
  • Horizon:            {optimal_params['monte_carlo_horizon_hours']} hours
  • Stress Test Paths:  {optimal_params['stress_test_paths']}
  • Stress Move:        {optimal_params['stress_test_adverse_move']:.1%}
""")

# Save optimization report
report = {
    'analysis_timestamp': pd.Timestamp.now().isoformat(),
    'n_trades': len(trades_df),
    'pnl_statistics': {
        'mean': float(pnl_series.mean()),
        'std': float(pnl_series.std()),
        'min': float(pnl_series.min()),
        'max': float(pnl_series.max()),
        'win_rate': float(win_rate),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
    },
    'kelly_criterion': {
        'win_loss_ratio': float(win_loss_ratio) if avg_loss != 0 else 0,
        'full_kelly': float(kelly_fraction),
        'half_kelly': float(kelly_fraction / 2),
    },
    'optimal_sizing': optimal_params,
    'sizing_test_results': results,
    'cvar_analysis': {
        'cvar_95': float(cvar_95),
        'cvar_99': float(cvar_99),
        'recommended_limit': float(recommended_cvar_limit),
    },
}

report_path = os.path.join(REPORTS_DIR, 'monte_carlo_sizing_optimization.json')
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"Saved optimization report to {report_path}")

print("\n" + "=" * 70)
print("MONTE CARLO OPTIMIZATION COMPLETE")
print("=" * 70)

# Integration instructions
print("""
INTEGRATION WITH SMCE MONTE CARLO COURT:

1. Update config.py with optimized parameters:
   ```python
   MONTE_CARLO_POSITION_SIZE = {optimal_params['position_size_fraction']:.3f}
   MONTE_CARLO_CVAR_LIMIT = {optimal_params['cvar_95_limit']:.4f}
   MONTE_CARLO_DRAWDOWN_LIMIT = {optimal_params['max_drawdown_limit']:.3f}
   ```

2. Update smce_monte_carlo_court.py thresholds:
   ```python
   VETO_CVAR_LIMIT = {optimal_params['cvar_95_limit']:.4f}
   VETO_DRAWDOWN_LIMIT = {optimal_params['max_drawdown_limit']:.3f}
   ```

3. Run optimization after collecting 50+ trades for better statistics
""")
