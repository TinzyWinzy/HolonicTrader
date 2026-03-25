import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

try:
    import holonic_speed
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print(">> [SDE] Warning: holonic_speed (Rust) not found. Falling back to Python defaults.")

class SDEEngine:
    """
    SDE Engine: The Physics Layer Utility.
    Provides Maximum Likelihood Estimation (MLE) for stochastic processes 
    and path simulation for risk modeling.
    Accelerated by Rust (holonic_speed).
    """
    
    @staticmethod
    def estimate_ou_parameters(prices: np.ndarray, dt: float = 1/35040) -> Dict[str, float]:
        """
        Estimate parameters for the Ornstein-Uhlenbeck (OU) process.
        Uses Rust (holonic_speed) if available.
        """
        if len(prices) < 2:
            return {'lambda': 0.0, 'mu': 0.0, 'sigma': 0.0, 'half_life': 0.0, 'drift': 0.0}
            
        if RUST_AVAILABLE and hasattr(holonic_speed, 'sde_estimate_ou'):
            # Convert to list for PyO3
            prices_list = prices.tolist() if isinstance(prices, np.ndarray) else list(prices)
            params = holonic_speed.sde_estimate_ou(prices_list, dt)
            
            # Add half-life (not in Rust yet for simplicity)
            lambda_val = params.get('lambda', 0.1)
            params['half_life'] = np.log(2) / lambda_val if lambda_val > 0 else 0.0
            return params

        # Python Fallback
        x_prev = prices[:-1]
        x_curr = prices[1:]
        mu = np.mean(prices)
        x_prev_demeaned = x_prev - mu
        x_curr_demeaned = x_curr - mu
        a = np.sum(x_prev_demeaned * x_curr_demeaned) / (np.sum(x_prev_demeaned**2) + 1e-12)
        a = max(0.0001, min(0.99999, a))
        lambda_val = -np.log(a) / dt
        res = x_curr_demeaned - a * x_prev_demeaned
        sigma_sq = np.var(res) * 2 * lambda_val / (1 - a**2 + 1e-12)
        sigma = np.sqrt(sigma_sq)
        half_life = np.log(2) / lambda_val if lambda_val > 0 else 0.0
        
        return {
            'lambda': float(lambda_val),
            'mu': float(mu),
            'sigma': float(sigma),
            'half_life': float(half_life),
            'drift': float(lambda_val * (mu - prices.iloc[-1] if hasattr(prices, 'iloc') else prices[-1]))
        }

    @staticmethod
    def estimate_gbm_parameters(prices: np.ndarray, dt: float = 1/35040) -> Dict[str, float]:
        """
        Estimate parameters for Geometric Brownian Motion (GBM).
        Uses Rust (holonic_speed) if available.
        """
        if len(prices) < 2:
            return {'drift': 0.0, 'diffusion': 0.0}
            
        if RUST_AVAILABLE and hasattr(holonic_speed, 'sde_estimate_gbm'):
            prices_list = prices.tolist() if isinstance(prices, np.ndarray) else list(prices)
            return holonic_speed.sde_estimate_gbm(prices_list, dt)

        # Python Fallback
        log_returns = np.diff(np.log(prices + 1e-12))
        sigma_sq = np.var(log_returns) / dt
        sigma = np.sqrt(sigma_sq)
        mu = (np.mean(log_returns) / dt) + (0.5 * sigma_sq)
        
        return {
            'drift': float(mu),
            'diffusion': float(sigma)
        }

    @staticmethod
    def simulate_paths(process_type: str, params: Dict[str, float], start_price: float, 
                       horizon: int = 100, paths: int = 1000, dt: float = 1/35040) -> np.ndarray:
        """
        Simulate future paths for Monte Carlo risk analysis.
        Uses Rust (holonic_speed) if available.
        """
        if RUST_AVAILABLE and hasattr(holonic_speed, 'sde_simulate_paths'):
            # Rust ignores dt internally currently (treated as unit step)
            # but we can pass params as they are.
            rust_paths = holonic_speed.sde_simulate_paths(
                process_type, 
                params, 
                float(start_price), 
                int(horizon), 
                int(paths),
                float(dt)
            )
            return np.array(rust_paths)

        # Python Fallback
        results = np.zeros((paths, horizon))
        results[:, 0] = start_price
        for p in range(paths):
            curr_price = start_price
            for t in range(1, horizon):
                dw = np.random.normal(0, np.sqrt(dt))
                if process_type == 'GBM':
                    mu = params.get('mu', params.get('drift', 0.1))
                    sigma = params.get('sigma', params.get('diffusion', 0.2))
                    curr_price *= np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * dw)
                elif process_type == 'OU':
                    lam = params.get('lambda', 0.1)
                    mu = params.get('mu', 0.0)
                    sig = params.get('sigma', 0.1)
                    # For Python, use discrete exact AR(1) or continuous if a/b not provided
                    a = params.get('a')
                    if a is not None:
                        curr_price = a * curr_price + params.get('b', 0.0) + params.get('residual_sigma', 0.01) * dw
                    else:
                        curr_price += lam * (mu - curr_price) * dt + sig * np.sqrt(dt) * dw
                results[p, t] = curr_price
        return results

    @staticmethod
    def calculate_ruin_probability(model: str, params: Dict[str, float], start_price: float,
                                    sl_price: float, tp_price: float, horizon: int = 100, 
                                    paths: int = 1000) -> float:
        """
        Calculates the probability of hitting Stop Loss before Take Profit.
        Directly calls high-performance Rust implementation for maximal speed.
        """
        if RUST_AVAILABLE and hasattr(holonic_speed, 'sde_calculate_ruin_probability'):
            return holonic_speed.sde_calculate_ruin_probability(
                model, params, float(start_price), float(sl_price), float(tp_price), 
                int(horizon), int(paths), float(1/35040)  # default dt if not passed
            )
            
        # Python Fallback
        sim_paths = SDEEngine.simulate_paths(model, params, start_price, horizon, paths)
        is_buy = start_price < tp_price
        
        failures = 0
        for p in range(len(sim_paths)):
            path = sim_paths[p]
            for step in path:
                if is_buy:
                    if step <= sl_price:
                        failures += 1
                        break
                    if step >= tp_price:
                        break
                else:
                    if step >= sl_price:
                        failures += 1
                        break
                    if step <= tp_price:
                        break
                    
        return failures / paths

    @staticmethod
    def calculate_hit_probability(model: str, params: Dict[str, float], start_price: float,
                                    target_price: float, horizon: int = 100, 
                                    paths: int = 1000, dt: float = 1/35040) -> float:
        """
        Calculates the probability of hitting Target Price within the horizon.
        """
        # Python Fallback (No Rust equivalent yet)
        sim_paths = SDEEngine.simulate_paths(model, params, start_price, horizon, paths, dt)
        is_upward = target_price > start_price
        
        hits = 0
        for p in range(len(sim_paths)):
            path = sim_paths[p]
            if is_upward:
                if np.any(path >= target_price):
                    hits += 1
            else:
                if np.any(path <= target_price):
                    hits += 1
                    
        return hits / paths

    @staticmethod
    def estimate_survival_horizon(model: str, params: Dict[str, float], start_price: float,
                                    stop_price: float, max_horizon: int = 24*30, 
                                    paths: int = 1000, dt: float = 1/24) -> float:
        """
        Calculates the median time (in steps/hours) until price hits the Stop Price.
        Returns 'max_horizon' if median survival exceeds the simulation window.
        """
        sim_paths = SDEEngine.simulate_paths(model, params, start_price, max_horizon, paths, dt)
        is_long = start_price > stop_price
        
        survival_times = []
        
        for p in range(len(sim_paths)):
            path = sim_paths[p]
            if is_long:
                # Find first index where price <= stop_price
                hit_indices = np.where(path <= stop_price)[0]
            else:
                # Short: Find first index where price >= stop_price
                hit_indices = np.where(path >= stop_price)[0]
                
            if len(hit_indices) > 0:
                survival_times.append(hit_indices[0])
            else:
                survival_times.append(max_horizon)
                
        # Return Median Survival Time
        return float(np.median(survival_times))
