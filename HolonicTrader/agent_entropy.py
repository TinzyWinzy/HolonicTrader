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
        self.last_entropy = None  # Track last computed Shannon entropy for SMCE/Governor


    def calculate_shannon_entropy(self, returns_series: pd.Series) -> float:
        """
        Calculate Shannon Entropy using Rust Engine (Holonic Speed) if available.
        Fallback to Python/Scipy if not.
        """
        result = 0.0
        # Try Rust Path (100x Faster)
        try:
            import holonic_speed
            # Rust expects a flat list of floats
            # We must ensure data is clean (no NaNs/Infs) or Rust might panic/return NaN
            data = returns_series.dropna().values.tolist()
            if not data:
                self.last_entropy = 0.0
                return 0.0
            result = float(holonic_speed.calculate_shannon_entropy(data))
            self.last_entropy = result
            return result
            
        except ImportError:
            # Fallback to Legacy Python
            counts, bin_edges = np.histogram(returns_series, bins=10)
            total_count = counts.sum()
            if total_count == 0:
                self.last_entropy = 0.0
                return 0.0
            probabilities = counts / total_count
            result = float(scipy_entropy(probabilities))
            self.last_entropy = result
            return result
        except Exception as e:
            # Safety Net
            # print(f"Rust Entropy Error: {e}") 
            counts, bin_edges = np.histogram(returns_series, bins=10)
            total_count = counts.sum()
            if total_count == 0:
                self.last_entropy = 0.0
                return 0.0
            probabilities = counts / total_count
            result = float(scipy_entropy(probabilities))
            self.last_entropy = result
            return result

    def calculate_renyi_entropy(self, returns_series: pd.Series, alpha: float = 2.0) -> float:
        """
        Calculate Rényi Entropy using Rust Engine.
        """
        try:
            import holonic_speed
            data = returns_series.dropna().values.tolist()
            if not data: return 0.0
            return float(holonic_speed.calculate_renyi_entropy(data, alpha))
            
        except ImportError:
            # Fallback
            counts, _ = np.histogram(returns_series, bins=10)
            total_count = counts.sum()
            if total_count == 0: return 0.0
            probabilities = counts / total_count
            if np.isclose(alpha, 1.0):
                return self.calculate_shannon_entropy(returns_series)
            sum_p_alpha = np.sum(probabilities ** alpha)
            if sum_p_alpha == 0: return 0.0
            return float((1.0 / (1.0 - alpha)) * np.log(sum_p_alpha))

    def calculate_multiscale_entropy(self, returns_series: pd.Series, max_scale: int = 10, m: int = 2) -> list:
        """
        AEHML 2.0: Calculate Multiscale Entropy (MSE/RCMWPE).
        Returns a list of entropy values for scales 1 to max_scale.
        """
        try:
            import holonic_speed
            data = returns_series.dropna().values.tolist()
            if not data: return [0.0] * max_scale
            return holonic_speed.calculate_multiscale_entropy(data, max_scale, m)
        except Exception as e:
            print(f"[{self.name}] RCMWPE Error: {e}")
            # Fallback: Just return naive Shannon repeated or zeros
            return [self.calculate_shannon_entropy(returns_series)] * max_scale

    def calculate_permutation_entropy(self, returns_series: pd.Series, m: int = 3, delay: int = 1) -> float:
        """
        AEHML 2.0: Calculate Permutation Entropy (Structural Complexity).
        """
        try:
            import holonic_speed
            data = returns_series.dropna().values.tolist()
            if not data: return 0.0
            return float(holonic_speed.calculate_permutation_entropy(data, m, delay))
        except Exception as e:
            # Fallback? PE is hard to replicate simply in numpy without loop.
            return 0.0

    def determine_regime(self, entropy_value: float) -> Literal['ORDERED', 'CHAOTIC', 'TRANSITION']:
        """
        Determine market regime based on entropy value.
        
        THRESHOLDS (Calibrated 2026-03-19 — Correct 10-bin ln scale):
            Shannon H over 10-bin histogram (nats) has max ln(10) ≈ 2.303.
            Gaussian returns ≈ 1.9-2.1 nats.  Structured trends ≈ 1.0-1.5.
            
            ORDERED:    < 1.20  (clearly trending / structured)
            TRANSITION: 1.20 - 1.85  (normal tradeable conditions)
            CHAOTIC:    > 1.85  (near-uniform / truly random)
            
        Args:
            entropy_value: The calculated Shannon Entropy.

        Returns:
            'ORDERED' if entropy < 1.20
            'CHAOTIC' if entropy > 1.85
            'TRANSITION' otherwise
        """
        if entropy_value < 1.20:
            return 'ORDERED'
        elif entropy_value > 1.85:
            return 'CHAOTIC'
        else:
            return 'TRANSITION'

    def predict_trend(self, returns_series: pd.Series, model_path: str = "rust_engine/src/onnx_models/trend.onnx") -> float:
        """
        Phase 2: Use ONNX models via Rust to predict the trend probability.
        Returns a float between 0.0 and 1.0 representing the upward trend probability.
        """
        from config import USE_ONNX
        if not USE_ONNX:
            return 0.5  # Neutral fallback if disabled in config

        try:
            import holonic_speed
            # Ensure model receives clean f32 float values
            data = returns_series.dropna().astype(np.float32).values.tolist()
            if not data: return 0.5

            # The ONNX model is hardcoded to expect exactly 50 sequence length.
            target_len = 50
            if len(data) > target_len:
                data = data[-target_len:]
            elif len(data) < target_len:
                data = [0.0] * (target_len - len(data)) + data

            prediction = holonic_speed.onnx_predict_trend(model_path, data)
            return float(prediction)
        except Exception as e:
            print(f"[{self.name}] ONNX Prediction Error: {e}")
            return 0.5

    def get_health(self) -> dict:
        """Report agent health status."""
        return {
            'status': 'OK',
            'last_entropy': self.last_entropy if self.last_entropy is not None else 'N/A'
        }

    def receive_message(self, sender: Any, content: Any) -> None:
        """Handle incoming messages."""
        if isinstance(content, Message):
            pass
        else:
            pass
