"""
Test SDE Engine parameter estimation.
"""
import sys
import os
import numpy as np
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from HolonicTrader.sde_engine import SDEEngine

class TestSDEEngine(unittest.TestCase):
    """Tests for SDE Engine parameter estimation."""
    
    def test_ou_estimation(self):
        """Test OU parameter estimation."""
        # Generate synthetic OU path
        true_lambda = 50.0
        true_mu = 1.2
        true_sigma = 0.2
        dt = 1/35040  # 15 min
        steps = 50000

        np.random.seed(42)  # For reproducibility
        x = np.zeros(steps)
        x[0] = true_mu

        for t in range(1, steps):
            dw = np.random.normal(0, np.sqrt(dt))
            x[t] = x[t-1] + true_lambda * (true_mu - x[t-1]) * dt + true_sigma * dw

        # Estimate
        params = SDEEngine.estimate_ou_parameters(x, dt=dt)

        # Verify with loose tolerance
        self.assertAlmostEqual(params['mu'], true_mu, delta=0.2)
        self.assertAlmostEqual(params['lambda'], true_lambda, delta=20.0)
        self.assertAlmostEqual(params['sigma'], true_sigma, delta=0.1)

    def test_gbm_estimation(self):
        """Test GBM parameter estimation."""
        # Generate synthetic GBM path with more data for better estimation
        true_drift = 0.15  # 15% annual drift
        true_vol = 0.8  # 80% annual vol
        dt = 1/35040
        steps = 10000  # More data points

        np.random.seed(42)  # For reproducibility
        s = np.zeros(steps)
        s[0] = 100.0

        for t in range(1, steps):
            dw = np.random.normal(0, np.sqrt(dt))
            s[t] = s[t-1] * np.exp((true_drift - 0.5 * true_vol**2) * dt + true_vol * dw)

        # Estimate
        params = SDEEngine.estimate_gbm_parameters(s, dt=dt)

        # Verify - diffusion (vol) is easier to estimate than drift
        self.assertAlmostEqual(params['diffusion'], true_vol, delta=0.2)
        # Drift estimation is very noisy - use very loose tolerance
        self.assertAlmostEqual(params['drift'], true_drift, delta=5.0)


if __name__ == "__main__":
    unittest.main()
