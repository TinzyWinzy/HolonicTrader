"""
EntropyHolon - The Entropy Engine (Phase 3)

This agent acts as the 'risk manager' of the AEHML core.
It calculates Shannon Entropy on market returns to judge
market order vs. chaos.

High entropy means the market is too random to trade safely.
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy
from typing import Any, Literal
import pandas as pd

from HolonicTrader.holon_core import Holon, Disposition


class EntropyHolon(Holon):
    """
    EntropyHolon is the 'Brain' that judges market order vs. chaos.
    It calculates Shannon Entropy on a returns series and classifies
    the market regime as ORDERED, CHAOTIC, or TRANSITION.
    """

    def __init__(self, name: str = "EntropyAgent"):
        # Initialize with default disposition
        default_disposition = Disposition(autonomy=0.7, integration=0.6)
        super().__init__(name=name, disposition=default_disposition)

    def calculate_shannon_entropy(self, returns_series: pd.Series) -> float:
        """
        Calculate Shannon Entropy for a given returns series.

        Steps:
            A. Discretize the input returns_series into 10 fixed bins using np.histogram.
            B. Calculate probabilities p(x) for each bin.
            C. Return the Shannon Entropy value (S = -sum(p * log(p))).

        Args:
            returns_series: A pandas Series of log returns.

        Returns:
            The Shannon Entropy value (in nats, using natural log).
        """
        # Step A: Discretize into 10 bins
        counts, bin_edges = np.histogram(returns_series, bins=10)

        # Step B: Calculate probabilities (normalize counts)
        # Sum of probabilities must equal 1
        total_count = counts.sum()
        if total_count == 0:
            # Edge case: no data, return 0 entropy
            return 0.0

        probabilities = counts / total_count

        # Verification: probabilities should sum to 1
        # assert np.isclose(probabilities.sum(), 1.0), "Probabilities do not sum to 1!"

        # Step C: Calculate Shannon Entropy using scipy.stats.entropy
        # scipy.stats.entropy calculates S = -sum(pk * log(pk)) for pk > 0
        # It handles zero probabilities gracefully (0 * log(0) = 0)
        shannon_entropy = scipy_entropy(probabilities)

        return float(shannon_entropy)

    def calculate_renyi_entropy(self, returns_series: pd.Series, alpha: float = 2.0) -> float:
        """
        Calculate Rényi Entropy for a given returns series.
        
        Formula: H_alpha(X) = (1 / (1 - alpha)) * ln(sum(p_i ^ alpha))
        
        Args:
            returns_series: A pandas Series of log returns.
            alpha: The order of Rényi entropy. 
                   alpha=0 -> Max Entropy (Hartley)
                   alpha=1 -> Shannon Entropy (limit as alpha->1)
                   alpha=2 -> Collision Entropy (focus on peaks)
                   alpha=inf -> Min Entropy
                   
        Returns:
            The Rényi Entropy value (in nats).
        """
        # Step A: Discretize (Same as Shannon)
        counts, _ = np.histogram(returns_series, bins=10)
        
        # Step B: Probabilities
        total_count = counts.sum()
        if total_count == 0:
            return 0.0
            
        probabilities = counts / total_count
        
        # Step C: Rényi Calculation
        # Avoid log(0) issues not needed here as we sum first, but p^alpha is safe for p=0
        
        # alpha = 1.0 is a special case (Shannon), but typically handled by separate func or limit.
        if np.isclose(alpha, 1.0):
            return self.calculate_shannon_entropy(returns_series)
            
        sum_p_alpha = np.sum(probabilities ** alpha)
        
        if sum_p_alpha == 0:
            return 0.0 # Should not happen if total_count > 0
            
        renyi_entropy = (1.0 / (1.0 - alpha)) * np.log(sum_p_alpha)
        
        return float(renyi_entropy)

    def determine_regime(self, entropy_value: float) -> Literal['ORDERED', 'CHAOTIC', 'TRANSITION']:
        """
        Determine market regime based on entropy value.
        
        THRESHOLDS (Calibrated Phase 34 - TUNED RELAXATION):
            ORDERED:    < 1.00 (Allowed for complex structure)
            CHAOTIC:    > 1.35 (Just below Gaussian Noise ~1.4)
            TRANSITION: 1.00 - 1.35
            
        Args:
            entropy_value: The calculated Shannon Entropy.

        Returns:
            'ORDERED' if entropy < 1.00
            'CHAOTIC' if entropy > 1.35
            'TRANSITION' otherwise
        """
        if entropy_value < 1.00:
            return 'ORDERED'
        elif entropy_value > 1.35:
            return 'CHAOTIC'
        else:
            return 'TRANSITION'

    def get_health(self) -> dict:
        """Report agent health status."""
        return {
            'status': 'OK',
            'last_entropy': 'N/A' # Could track last value
        }

    def receive_message(self, sender: Any, content: Any) -> None:
        """Handle incoming messages."""
        if isinstance(content, Message):
            pass
        else:
            pass
