"""
PPOHolon - The Sovereign Strategic Brain (Phase 22)

Implements Proximal Policy Optimization (PPO) for continuous risk orchestration.
This agent distills market regime and portfolio health into a single conviction factor [0.0, 1.0].

Architecture: Actor-Critic with Clipping.
State Vector (6): [Regime_ID, Entropy, WinRate, ATR_Ratio, Drawdown, Margin]
Action (1): Conviction Scale [0, 1]
"""

import numpy as np
from typing import Any
import os
import tensorflow
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from HolonicTrader.holon_core import Holon, Disposition
import config
try:
    import openvino as ov
except ImportError:
    ov = None

class PPOHolon(Holon):
    def __init__(
        self,
        name: str = "PPO_Sovereign",
        storage_path: str = "ppo_brain",
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        batch_size: int = 64
    ):
        super().__init__(name=name, disposition=Disposition(autonomy=1.0, integration=0.2))
        
        self.storage_path = storage_path
        self.lr = learning_rate
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.batch_size = batch_size
        
        self.state_size = 6
        self.action_size = 1 # Continuous [0, 1]
        
        # Networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.ov_actor = None
        self.ov_critic = None
        
        # Buffer for PPO updates
        self.states = []
        self.actions = []
        self.rewards = []
        self.old_probs = []
        self.values = []
        self.dones = []
        
        self.load_knowledge()

    def _build_actor(self):
        """Actor network: state -> mean of action distribution."""
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        # We output a mean in [0, 1] for conviction
        output = layers.Dense(1, activation='sigmoid')(x)
        model = models.Model(inputs, output)
        model.compile(optimizer=optimizers.Adam(learning_rate=self.lr))
        return model

    def _build_critic(self):
        """Critic network: state -> value."""
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        output = layers.Dense(1)(x)
        model = models.Model(inputs, output)
        model.compile(optimizer=optimizers.Adam(learning_rate=self.lr), loss='mse')
        return model

    def get_conviction(self, state: np.ndarray, training: bool = False) -> float:
        """
        Predict conviction factor using the Actor.
        State: [Regime_ID, Entropy, WinRate, ATR_Ratio, Drawdown, Margin]
        Returns: float in [0.0, 1.0]
        """
        state_tensor = np.expand_dims(state, axis=0)
        
        # High-Performance OpenVINO Inference
        if self.ov_actor:
             # state_tensor is already (1, 6)
             mu = float(self.ov_actor(state_tensor)[0][0][0])
        else:
            s_tensor = tf.convert_to_tensor(state_tensor, dtype=tf.float32)
            mu_tensor = self.actor(s_tensor, training=False)
            mu = float(mu_tensor.numpy()[0][0])
        
        if training:
            # Add Gaussian noise for exploration during training
            sigma = 0.1 # Exploration noise
            action = np.random.normal(mu, sigma)
            return float(np.clip(action, 0.01, 0.99))
        
        return mu

    def learn(self):
        """
        PPO clipped policy update.
        """
        if len(self.states) < self.batch_size:
            return

        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        old_probs = np.array(self.old_probs)
        values = np.array(self.values)
        dones = np.array(self.dones)

        # 1. Calculate Returns and Advantages
        returns = []
        discounted_sum = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                discounted_sum = 0
            discounted_sum = r + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        
        returns = np.array(returns)
        advantages = returns - values.flatten()
        # Standardize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)

        # 2. PPO Clipped Update
        # We use a custom training loop to apply the PPO loss
        @tf.function
        def train_step(states, actions, old_probs, advantages, returns):
            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
                # Actor Loss (Clipped)
                # Note: We assume Gaussian policy with fixed sigma=0.1 for simplicity
                # Log Prob = -0.5 * ((action-mu)/sigma)^2 - log(sigma * sqrt(2*PI))
                mu = self.actor(states, training=True)
                sigma = 0.1
                
                # Simplified log_prob for our [0,1] conviction
                log_probs = -0.5 * (tf.square((actions - mu) / sigma)) - tf.math.log(sigma * tf.math.sqrt(2 * np.pi))
                
                ratios = tf.exp(log_probs - old_probs)
                surr1 = ratios * advantages
                surr2 = tf.clip_by_value(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
                actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

                # Critic Loss (MSE)
                v_preds = self.critic(states, training=True)
                critic_loss = tf.reduce_mean(tf.square(returns - v_preds))

            actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
            
            return actor_loss, critic_loss

        # Perform updates
        a_loss, c_loss = train_step(
            tf.convert_to_tensor(states, dtype=tf.float32),
            tf.convert_to_tensor(actions, dtype=tf.float32),
            tf.convert_to_tensor(old_probs, dtype=tf.float32),
            tf.convert_to_tensor(advantages, dtype=tf.float32),
            tf.convert_to_tensor(returns, dtype=tf.float32)
        )

        # Clear buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.old_probs = []
        self.values = []
        self.dones = []
        
        return float(a_loss), float(c_loss)

    def remember(self, state, action, reward, prob, val, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.old_probs.append(prob)
        self.values.append(val)
        self.dones.append(done)

    def get_value(self, state: np.ndarray) -> float:
        state_tensor = np.expand_dims(state, axis=0)
        if self.ov_critic:
            val = float(self.ov_critic(state_tensor)[0][0][0])
        else:
            val = self.critic.predict(state_tensor, verbose=0)[0][0]
        return float(val)
    
    def get_log_prob(self, state: np.ndarray, action: float) -> float:
        state_tensor = np.expand_dims(state, axis=0)
        if self.ov_actor:
            mu = float(self.ov_actor(state_tensor)[0][0][0])
        else:
            mu = self.actor.predict(state_tensor, verbose=0)[0][0]
        sigma = 0.1
        log_prob = -0.5 * (((action - mu) / sigma)**2) - np.log(sigma * np.sqrt(2 * np.pi))
        return float(log_prob)

    def load_knowledge(self):
        # Prioritize native .keras format, fallback to legacy .h5
        actor_keras = os.path.join(self.storage_path, "actor.keras")
        critic_keras = os.path.join(self.storage_path, "critic.keras")
        actor_h5 = os.path.join(self.storage_path, "actor.h5")
        critic_h5 = os.path.join(self.storage_path, "critic.h5")
        
        path_actor = actor_keras if os.path.exists(actor_keras) else actor_h5
        path_critic = critic_keras if os.path.exists(critic_keras) else critic_h5

        if os.path.exists(path_actor) and os.path.exists(path_critic):
            try:
                self.actor = models.load_model(path_actor, compile=False)
                self.critic = models.load_model(path_critic, compile=False)
                
                # Re-compile manually to fix serialization issues (e.g. mse)
                self.actor.compile(optimizer=optimizers.Adam(learning_rate=self.lr))
                self.critic.compile(optimizer=optimizers.Adam(learning_rate=self.lr), loss='mse')
                
                print(f"[{self.name}] Monolith-V5 PPO Brain loaded successfully from {path_actor}/{path_critic}.")
            except Exception as e:
                print(f"[{self.name}] Load failed: {e}. Starting fresh.")
        else:
            print(f"[{self.name}] Dynamic PPO Brain initialized.")

        # OpenVINO Optimization
        if ov is not None and config.USE_OPENVINO:
             try:
                 core = ov.Core()
                 device = "GPU" if config.USE_INTEL_GPU else "CPU"
                 # Compile Actor
                 self.ov_actor = core.compile_model(ov.convert_model(self.actor), device)
                 # Compile Critic
                 self.ov_critic = core.compile_model(ov.convert_model(self.critic), device)
                 print(f"[{self.name}] OpenVINO PPO Backend initialized on {device}.")
             except Exception as e:
                 print(f"[{self.name}] OpenVINO PPO Setup failed: {e}. Falling back to native.")

    def save_knowledge(self):
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
        
        # Save in native Keras format to resolve legacy warnings
        self.actor.save(os.path.join(self.storage_path, "actor.keras"))
        self.critic.save(os.path.join(self.storage_path, "critic.keras"))

        # Re-initialize OpenVINO inference with updated weights
        if ov is not None and config.USE_OPENVINO:
            try:
                core = ov.Core()
                device = "GPU" if config.USE_INTEL_GPU else "CPU"
                self.ov_actor = core.compile_model(ov.convert_model(self.actor), device)
                self.ov_critic = core.compile_model(ov.convert_model(self.critic), device)
                print(f"[{self.name}] OpenVINO PPO Backend REFRESHED on {device}.")
            except: pass

    def get_health(self) -> dict:
        return {
            'status': 'ACTIVE',
            'model': 'PPO (Actor-Critic)',
            'state_dim': self.state_size,
            'actor_lr': self.lr
        }

    def receive_message(self, sender: Any, content: Any) -> None:
        pass
