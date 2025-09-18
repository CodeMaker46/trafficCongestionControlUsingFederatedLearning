import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from typing import Tuple, List

class DQNAgent:
    """
    Deep Q-Network agent for traffic light control
    """
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        
        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        
        # Neural network
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self) -> Sequential:
        """Build the neural network model"""
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['mae']
        )
        
        return model
    
    def update_target_model(self):
        """Copy weights from main model to target model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.model.predict(state[np.newaxis], verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self) -> float:
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([e[0] for e in minibatch])
        actions = np.array([e[1] for e in minibatch])
        rewards = np.array([e[2] for e in minibatch])
        next_states = np.array([e[3] for e in minibatch])
        dones = np.array([e[4] for e in minibatch])
        
        # Current Q values
        current_q_values = self.model.predict(states, verbose=0)
        
        # Next Q values from target model
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Calculate target Q values
        target_q_values = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the model
        history = self.model.fit(
            states, target_q_values,
            epochs=1,
            verbose=0,
            batch_size=self.batch_size
        )
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return history.history['loss'][0]
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a given state"""
        return self.model.predict(state[np.newaxis], verbose=0)[0]
    
    def save_model(self, filepath: str):
        """Save model weights"""
        self.model.save_weights(filepath)
    
    def load_model(self, filepath: str):
        """Load model weights"""
        self.model.load_weights(filepath)
        self.update_target_model()
    
    def get_weights(self) -> List[np.ndarray]:
        """Get model weights for federated learning"""
        return self.model.get_weights()
    
    def set_weights(self, weights: List[np.ndarray]):
        """Set model weights from federated learning"""
        self.model.set_weights(weights)
        self.update_target_model()
    
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.model.summary()
        sys.stdout = old_stdout
        
        return buffer.getvalue()
