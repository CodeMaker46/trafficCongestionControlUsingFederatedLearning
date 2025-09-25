import numpy as np
import random
from collections import deque
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim

class DQNNetwork(nn.Module):
    """Three-hidden-layer MLP used for Q-value approximation."""
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """
    Deep Q-Network agent for traffic light control
    """
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 1e-3, device: str | None = None):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        
        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration start
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.target_update_freq = 100
        self.train_step = 0
        
        # Neural networks
        self.policy_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
    def update_target_model(self):
        """Copy weights from policy network to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(s)
            return int(torch.argmax(q_values, dim=1).item())
    
    def replay(self) -> float:
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.as_tensor(np.array([e[0] for e in minibatch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.array([e[1] for e in minibatch]), dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(np.array([e[2] for e in minibatch]), dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.array([e[3] for e in minibatch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(np.array([e[4] for e in minibatch]), dtype=torch.float32, device=self.device)

        # Compute current Q estimates
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            targets = rewards + (1.0 - dones) * self.gamma * next_q_values

        loss_fn = nn.MSELoss()
        loss = loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_model()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return float(loss.item())
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a given state"""
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy_net(s)[0].detach().cpu().numpy()
            return q
    
    def save_model(self, filepath: str):
        """Save model weights"""
        torch.save(self.policy_net.state_dict(), filepath)
    
    def load_model(self, filepath: str):
        """Load model weights"""
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.update_target_model()
    
    def get_weights(self) -> List[np.ndarray]:
        """Get model weights for federated learning"""
        # Serialize parameters as list of numpy arrays for federated learning
        return [p.detach().cpu().numpy() for p in self.policy_net.parameters()]
    
    def set_weights(self, weights: List[np.ndarray]):
        """Set model weights from federated learning"""
        with torch.no_grad():
            for param, w in zip(self.policy_net.parameters(), weights):
                param.copy_(torch.as_tensor(w, dtype=param.dtype, device=self.device))
        self.update_target_model()
    
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        num_params = sum(p.numel() for p in self.policy_net.parameters())
        return f"DQNNetwork(state={self.state_size}, actions={self.action_size}, params={num_params})"
