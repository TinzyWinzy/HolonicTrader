import numpy as np
from typing import Tuple

class KalmanFilter1D:
    """
    A simple 1D Kalman Filter implementation.
    
    Model:
    x_k = x_{k-1} + w_k  (Process noise w_k ~ N(0, Q))
    z_k = x_k + v_k      (Measurement noise v_k ~ N(0, R))
    
    Used for estimating the "true" price from noisy observations.
    """
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1, estimated_error: float = 1.0):
        # Q: Process Noise Covariance (System volatility/uncertainty)
        self.Q = process_noise 
        
        # R: Measurement Noise Covariance (Market noise/spread error)
        self.R = measurement_noise
        
        # P: Estimation Error Covariance (Initial uncertainty)
        self.P = estimated_error
        
        # x: State Estimate (Initial value)
        self.x = 0.0 
        self.initialized = False

    def update(self, measurement: float) -> float:
        """
        Perform one update step (Predict + Update).
        
        Args:
            measurement: The latest noisy observation (price).
            
        Returns:
            The filtered state estimate.
        """
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return self.x
            
        # 1. Prediction Step
        # x_pred = x_prev (Naive constant velocity model for price, assuming no drift)
        x_pred = self.x
        
        # P_pred = P_prev + Q
        P_pred = self.P + self.Q
        
        # 2. Update Step
        # Kalman Gain: K = P_pred / (P_pred + R)
        K = P_pred / (P_pred + self.R)
        
        # Update State: x = x_pred + K * (measurement - x_pred)
        self.x = x_pred + K * (measurement - x_pred)
        
        # Update Covariance: P = (1 - K) * P_pred
        self.P = (1.0 - K) * P_pred
        
        return float(self.x)
