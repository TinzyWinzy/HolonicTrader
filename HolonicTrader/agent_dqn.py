"""
DeepQLearningHolon - The Advanced RL Agent (Phase B - TensorFlow Upgrade)

This agent uses a TensorFlow/Keras Neural Network (DQN) to approximate
the Q-value function, allowing for continuous state inputs and robust training.

Dependencies: tensorflow, numpy
"""

import numpy as np
import random
import os
from collections import deque
from typing import List, Any

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import Sequential, layers, models, optimizers

from HolonicTrader.holon_core import Holon, Disposition

# ==============================================================================
# Deep Q-Learning Agent (TensorFlow Version)
# ==============================================================================

class DeepQLearningHolon(Holon):
    """
    DeepQLearningHolon uses a Keras Neural Network to predict Q-values from continuous state.
    """
    
    def __init__(
        self, 
        name: str = "DQN_Agent", 
        storage_path: str = "dqn_model.keras",
        epsilon: float = 1.0,         # Start with high exploration
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        gamma: float = 0.95,           # Optimized
        learning_rate: float = 0.0001, # Optimized
        hidden_size: int = 24          # Optimized
    ):
        super().__init__(name=name, disposition=Disposition(autonomy=0.9, integration=0.1))
        
        self.storage_path = storage_path
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        
        # State Size: 4 [Entropy, RSI, Returns, Bias/Constant]
        # Action Size: 3 [EXECUTE, HALT, REDUCE]
        self.state_size = 4
        self.action_size = 3
        self.actions = ['EXECUTE', 'HALT', 'REDUCE']
        
        # Replay Buffer
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        
        # Initialize Neural Network
        self.model = self._build_model()
        
        self.load_knowledge()

    def _build_model(self):
        """Builds the Keras Sequential model."""
        model = Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(self.hidden_size, activation='relu'),
            layers.Dense(self.hidden_size, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            loss='mse',
            optimizer=optimizers.Adam(learning_rate=self.learning_rate)
        )
        return model

    def load_knowledge(self) -> None:
        if os.path.exists(self.storage_path):
            try:
                self.model = models.load_model(self.storage_path)
                # If loaded, maybe lower epsilon considerably but keep some exploration
                self.epsilon = max(self.epsilon, 0.2) 
                print(f"[{self.name}] Loaded DQN model from {self.storage_path}")
            except Exception as e:
                print(f"[{self.name}] Failed to load model: {e}. Starting fresh.")
        else:
            print(f"[{self.name}] Initialized new DQN model.")

    def save_knowledge(self) -> None:
        try:
            self.model.save(self.storage_path)
            print(f"[{self.name}] Saved DQN model to {self.storage_path}")
        except Exception as e:
            print(f"[{self.name}] Error saving model: {e}")

    def get_action(self, state: List[float]) -> str:
        """
        Epsilon-Greedy Action Selection.
        State should be list [Entropy, RSI, Returns, 1.0]
        """
        if np.random.rand() <= self.epsilon:
            return random.choice(self.actions)
        
        # Predict Q-values
        # Keras expects (batch_size, input_dim), so we expand dims
        state_tensor = np.array([state], dtype=np.float32) 
        
        # Run prediction (verbose=0 to avoid log spam)
        q_values = self.model.predict(state_tensor, verbose=0)
        action_idx = np.argmax(q_values[0])
        
        return self.actions[action_idx]

    def remember(self, state, action_idx, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action_idx, reward, next_state, done))

    def replay(self) -> float:
        """
        Train the network on a batch of experiences.
        Returns the loss.
        """
        if len(self.memory) < self.batch_size:
            return 0.0
            
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch arrays
        states = np.array([m[0] for m in minibatch], dtype=np.float32)
        next_states = np.array([m[3] for m in minibatch], dtype=np.float32)
        
        # Predict Q-values for current and next states
        # Doing this in batch is much faster than one by one
        current_qs = self.model.predict(states, verbose=0)
        next_qs = self.model.predict(next_states, verbose=0)
        
        X = states
        y = current_qs.copy() # We only update the Q-value for the taken action
        
        for i, (state, action_idx, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(next_qs[i])
            
            y[i][action_idx] = target
        
        # Train on the batch
        # train_on_batch returns the loss (scalar)
        loss = self.model.train_on_batch(X, y)
        
        # Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return float(loss)

    def get_action_index(self, action_str: str) -> int:
        return self.actions.index(action_str)
    
    def get_health(self) -> dict:
        """Report agent health status."""
        return {
            'status': 'ACTIVE',
            'epsilon': f"{self.epsilon:.4f}",
            'memory': len(self.memory)
        }

    def receive_message(self, sender: Any, content: Any) -> None:
        pass
