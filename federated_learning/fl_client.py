import numpy as np
from typing import Dict, List, Tuple, Optional
import flwr as fl
from agents.dqn_agent import DQNAgent
from agents.traffic_environment import SUMOTrafficEnvironment
import os
import json
import time

class TrafficFLClient(fl.client.NumPyClient):
    """
    Federated Learning client for traffic control
    Each client represents a different intersection or traffic scenario
    """
    
    def __init__(self, client_id: str, sumo_config_path: str, 
                 state_size: int = 12, action_size: int = 4,
                 gui: bool = False, show_phase_console: bool = False, show_gst_gui: bool = False):
        self.client_id = client_id
        self.state_size = state_size
        self.action_size = action_size
        
        # Initialize DQN agent
        self.agent = DQNAgent(state_size, action_size)
        
        # Initialize traffic environment
        self.env = SUMOTrafficEnvironment(sumo_config_path, gui=gui, show_phase_console=show_phase_console, show_gst_gui=show_gst_gui)
        
        # Training parameters
        self.episodes_per_round = 10
        self.max_steps_per_episode = 1000
        
        # Performance tracking
        self.training_history = []
        self.performance_metrics = []
        
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return current model parameters"""
        return self.agent.get_weights()
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """Set model parameters"""
        self.agent.set_weights(parameters)
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the model on local data
        """
        # Set global parameters
        self.set_parameters(parameters)
        
        # Training configuration
        episodes = config.get("episodes", self.episodes_per_round)
        learning_rate = config.get("learning_rate", 0.001)
        
        # Update agent learning rate
        self.agent.learning_rate = learning_rate
        
        # Train the agent
        training_metrics = self._train_agent(episodes)
        
        # Store training history
        self.training_history.append({
            'round': config.get('round', 0),
            'episodes': episodes,
            'metrics': training_metrics
        })
        
        # Return updated parameters and metrics
        return (
            self.get_parameters(config),
            episodes * self.max_steps_per_episode,
            training_metrics
        )
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate the model on local data
        """
        # Set global parameters
        self.set_parameters(parameters)
        
        # Evaluate the agent
        evaluation_metrics = self._evaluate_agent()
        
        # Store performance metrics
        self.performance_metrics.append({
            'round': config.get('round', 0),
            'metrics': evaluation_metrics
        })
        
        # Return loss, number of samples, and metrics
        loss = evaluation_metrics.get('average_reward', 0.0)
        num_samples = evaluation_metrics.get('total_steps', 1)
        
        return loss, num_samples, evaluation_metrics
    
    def _train_agent(self, episodes: int) -> Dict:
        """Train the DQN agent for specified number of episodes"""
        total_reward = 0
        total_steps = 0
        losses = []
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            
            for step in range(self.max_steps_per_episode):
                # Choose action
                action = self.agent.act(state, training=True)
                
                # Execute action
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, done)
                
                # Train the agent
                if len(self.agent.memory) > self.agent.batch_size:
                    loss = self.agent.replay()
                    if loss is not None:
                        losses.append(loss)
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                if done:
                    break
            
            total_reward += episode_reward
            total_steps += episode_steps
            
            # Update target network every 10 episodes
            if episode % 10 == 0:
                self.agent.update_target_model()
        
        # Close environment
        self.env.close()
        
        return {
            'average_reward': total_reward / episodes,
            'total_steps': total_steps,
            'average_loss': np.mean(losses) if losses else 0.0,
            'episodes': episodes
        }
    
    def _evaluate_agent(self) -> Dict:
        """Evaluate the DQN agent"""
        state = self.env.reset()
        total_reward = 0
        total_steps = 0
        
        for step in range(self.max_steps_per_episode):
            # Choose action (no exploration during evaluation)
            action = self.agent.act(state, training=False)
            
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            
            state = next_state
            total_reward += reward
            total_steps += 1
            
            if done:
                break
        
        # Get performance metrics
        performance = self.env.get_performance_metrics()
        
        # Close environment
        self.env.close()
        
        return {
            'total_reward': total_reward,
            'average_reward': total_reward / max(total_steps, 1),
            'total_steps': total_steps,
            'waiting_time': performance['total_waiting_time'],
            'queue_length': performance['average_queue_length'],
            'max_queue_length': performance['max_queue_length'],
            'green_signal_time': performance.get('green_signal_time', {}),
            'per_lane_metrics': performance.get('per_lane_metrics', {}),
            'lane_summary': performance.get('lane_summary', {})
        }
    
    def save_training_history(self, filepath: str):
        """Save training history to file"""
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def save_performance_metrics(self, filepath: str):
        """Save performance metrics to file"""
        with open(filepath, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
    
    def get_client_info(self) -> Dict:
        """Get client information"""
        return {
            'client_id': self.client_id,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'episodes_per_round': self.episodes_per_round,
            'max_steps_per_episode': self.max_steps_per_episode
        }
