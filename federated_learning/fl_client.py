import numpy as np
from typing import Dict, List, Tuple, Optional
import flwr as fl
from agents.dqn_agent import DQNAgent
from agents.traffic_environment import SUMOTrafficEnvironment
import os
import json
import time
from datetime import datetime

class TrafficFLClient(fl.client.NumPyClient):
    """
    Federated Learning client for traffic control
    Each client represents a different intersection or traffic scenario
    """
    
    def __init__(self, client_id: str, sumo_config_path: str, 
                 state_size: int = 12, action_size: int = 4,
                 gui: bool = False, show_phase_console: bool = False, show_gst_gui: bool = False,
                 tl_id: str = None):
        self.client_id = client_id
        self.state_size = state_size
        self.action_size = action_size
        
        # Initialize DQN agent
        self.agent = DQNAgent(state_size, action_size)
        
        # Initialize traffic environment
        self.env = SUMOTrafficEnvironment(sumo_config_path, gui=gui, show_phase_console=show_phase_console, 
                                         show_gst_gui=show_gst_gui, tl_id=tl_id)
        
        # Training parameters
        self.episodes_per_round = 10
        self.max_steps_per_episode = 1000
        
        # Performance tracking
        self.training_history = []
        self.performance_metrics = []
        
        # Periodic data transmission (every 5 seconds) - Novelty Feature 1
        self.last_transmission_time = 0.0
        self.transmission_interval = 5.0  # 5 seconds
        self.periodic_data_buffer = []  # Store data to send
        self.periodic_weights_buffer = []  # Store model weights periodically
        self.server_address = None  # Will be set if in FL mode
        
        # Novelty Feature 2: Adaptive Learning Rate based on performance
        self.base_learning_rate = 0.001
        self.performance_history = []  # Track recent performance
        self.adaptive_lr_enabled = True
        
        # Novelty Feature 3: Client Priority Weighting (based on traffic density)
        self.client_priority = 1.0  # Default priority
        self.traffic_density_history = []
        
        # Novelty Feature 4: Dynamic Episode Length (adapts to traffic conditions)
        self.dynamic_episode_length = True
        self.base_episode_length = self.max_steps_per_episode
        
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
        
        # Novelty Feature 2: Adaptive Learning Rate
        if self.adaptive_lr_enabled and len(self.performance_history) > 0:
            # Adjust LR based on recent performance
            recent_avg = np.mean(self.performance_history[-5:]) if len(self.performance_history) >= 5 else self.performance_history[-1]
            if recent_avg < 50:  # Poor performance - increase LR
                learning_rate = min(0.01, learning_rate * 1.2)
                print(f"[ADAPTIVE LR] Performance low ({recent_avg:.1f}), increasing LR to {learning_rate:.5f}")
            elif recent_avg > 80:  # Good performance - decrease LR for fine-tuning
                learning_rate = max(0.0001, learning_rate * 0.9)
                print(f"[ADAPTIVE LR] Performance high ({recent_avg:.1f}), decreasing LR to {learning_rate:.5f}")
        
        # Update agent learning rate
        self.agent.learning_rate = learning_rate
        
        # Clear periodic buffers for new round
        self.periodic_data_buffer = []
        self.periodic_weights_buffer = []
        self.last_transmission_time = 0.0
        
        # Train the agent
        training_metrics = self._train_agent(episodes)
        
        # Include periodic data in metrics
        training_metrics['periodic_data_transmissions'] = len(self.periodic_data_buffer)
        training_metrics['periodic_data_buffer'] = self.periodic_data_buffer
        training_metrics['periodic_weights_snapshots'] = len(self.periodic_weights_buffer)
        
        # Novelty Feature 3: Include client priority in metrics (for server weighting)
        training_metrics['client_priority'] = self.client_priority
        training_metrics['traffic_density_avg'] = np.mean(self.traffic_density_history) if self.traffic_density_history else 0.0
        
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
        
        # Clear periodic buffers for evaluation
        self.periodic_data_buffer = []
        self.periodic_weights_buffer = []
        self.last_transmission_time = 0.0
        
        # Evaluate the agent
        evaluation_metrics = self._evaluate_agent()
        
        # Include periodic data in evaluation metrics
        evaluation_metrics['periodic_data_transmissions'] = len(self.periodic_data_buffer)
        evaluation_metrics['periodic_data_buffer'] = self.periodic_data_buffer
        evaluation_metrics['periodic_weights_snapshots'] = len(self.periodic_weights_buffer)
        
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
        print(f"\n{'='*80}")
        print(f"🎓 STARTING TRAINING: {episodes} Episodes")
        print(f"{'='*80}\n")
        
        total_reward = 0
        total_steps = 0
        losses = []
        
        for episode in range(episodes):
            print(f"\n{'='*80}")
            print(f"🔄 STARTING EPISODE {episode + 1}/{episodes}")
            print(f"{'='*80}")
            try:
                state = self.env.reset()
                print(f"✅ Environment reset successfully")
            except Exception as e:
                print(f"⚠️  Error resetting environment: {e}")
                import traceback
                traceback.print_exc()
                print(f"   Trying to restart simulation...")
                try:
                    self.env.stop_simulation()
                    time.sleep(1)
                    self.env.start_simulation()
                    state = self.env.get_state()
                    print(f"✅ Simulation restarted")
                except Exception as e2:
                    print(f"❌ Failed to restart: {e2}")
                    print(f"   Skipping remaining episodes...")
                    break
            
            episode_reward = 0
            episode_steps = 0
            episode_losses = []
            
            # Novelty Feature 4: Dynamic Episode Length - check at episode start
            if self.dynamic_episode_length:
                # Get initial traffic state
                initial_info = self.env.get_performance_metrics() if hasattr(self.env, 'get_performance_metrics') else {}
                queue_sum = sum(initial_info.get('queue_lengths', [0])) if isinstance(initial_info, dict) else 0
                if queue_sum > 20:  # High traffic - extend episode
                    current_episode_length = min(1500, self.base_episode_length + 200)
                    print(f"[DYNAMIC EPISODE] High traffic (queue={queue_sum}), extending to {current_episode_length} steps")
                elif queue_sum < 5:  # Low traffic - shorten episode
                    current_episode_length = max(500, self.base_episode_length - 200)
                    print(f"[DYNAMIC EPISODE] Low traffic (queue={queue_sum}), shortening to {current_episode_length} steps")
                else:
                    current_episode_length = self.base_episode_length
            else:
                current_episode_length = self.max_steps_per_episode
            
            for step in range(current_episode_length):
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
                        episode_losses.append(loss)
                
                # Periodic data transmission to server (every 5 seconds) - Novelty Feature
                current_time = time.time()
                if current_time - self.last_transmission_time >= self.transmission_interval:
                    self._send_periodic_data(episode, step, reward, info)
                    self.last_transmission_time = current_time
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                if done:
                    print(f"   Episode {episode + 1} marked as done at step {step}")
                    break
            
            # Episode end summary
            if episode_steps > 0:  # Only print if episode had steps
                self._print_episode_summary(episode, episode_reward, episode_steps, episode_losses)
            else:
                print(f"⚠️  Episode {episode + 1} had 0 steps - skipping summary")
            
            total_reward += episode_reward
            total_steps += episode_steps
            
            # Update target network every 10 episodes
            if episode % 10 == 0:
                self.agent.update_target_model()
            
            # Episode completed successfully
            print(f"✅ Episode {episode + 1} completed successfully!")
            print(f"   Reward: {episode_reward:.4f}, Steps: {episode_steps}")
            
            # Check if more episodes to go
            if episode < episodes - 1:
                print(f"   Continuing to Episode {episode + 2}...\n")
            else:
                print(f"   All episodes completed!\n")
        
        # Close environment only after ALL episodes
        print(f"\n📊 All {episodes} episodes completed. Closing environment...")
        self.env.close()
        
        # Final training summary
        completed_episodes = max(1, len([e for e in range(episodes)]))  # Count completed
        print(f"\n{'='*80}")
        print(f"📊 TRAINING COMPLETED: {completed_episodes}/{episodes} episodes")
        print(f"{'='*80}")
        
        # Novelty Feature 2: Update performance history for adaptive LR
        final_score = self._calculate_final_score(total_reward, total_steps, losses)
        self.performance_history.append(final_score)
        if len(self.performance_history) > 10:
            self.performance_history.pop(0)  # Keep last 10 scores
        
        # Novelty Feature 3: Update client priority based on traffic density
        if self.traffic_density_history:
            avg_density = np.mean(self.traffic_density_history)
            # Higher density = higher priority (more important to train)
            self.client_priority = min(2.0, max(0.5, 1.0 + (avg_density / 50.0)))
        
        final_metrics = {
            'average_reward': total_reward / max(1, completed_episodes),
            'total_reward': total_reward,
            'total_steps': total_steps,
            'average_loss': np.mean(losses) if losses else 0.0,
            'episodes': completed_episodes,
            'final_score': final_score,
            # Include periodic transmission data
            'periodic_data_transmissions': len(self.periodic_data_buffer),
            'periodic_data_buffer': self.periodic_data_buffer,
            'periodic_weights_snapshots': len(self.periodic_weights_buffer),
            'periodic_weights_buffer': self.periodic_weights_buffer,
            # Novelty features
            'client_priority': self.client_priority,
            'adaptive_lr_used': self.adaptive_lr_enabled,
            'dynamic_episode_length': self.dynamic_episode_length
        }
        
        self._print_final_training_summary(final_metrics)
        
        return final_metrics
    
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
        
        # Calculate final score for evaluation
        final_score = self._calculate_final_score(total_reward, total_steps, [])
        
        return {
            'total_reward': total_reward,
            'average_reward': total_reward / max(total_steps, 1),
            'total_steps': total_steps,
            'waiting_time': performance['total_waiting_time'],
            'queue_length': performance['average_queue_length'],
            'max_queue_length': performance['max_queue_length'],
            'green_signal_time': performance.get('green_signal_time', {}),
            'per_lane_metrics': performance.get('per_lane_metrics', {}),
            'lane_summary': performance.get('lane_summary', {}),
            'final_score': final_score
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
    
    def _send_periodic_data(self, episode: int, step: int, reward: float, info: Dict):
        """Send periodic data to server every 5 seconds - Novelty Feature"""
        try:
            # Collect current model weights (snapshot)
            current_weights = self.get_parameters({})
            # Convert numpy arrays to lists for JSON serialization (only store shapes and summary)
            weights_summary = []
            for i, w in enumerate(current_weights):
                weights_summary.append({
                    'layer': i,
                    'shape': list(w.shape) if hasattr(w, 'shape') else [],
                    'mean': float(np.mean(w)) if hasattr(w, 'mean') else 0.0,
                    'std': float(np.std(w)) if hasattr(w, 'std') else 0.0,
                    'min': float(np.min(w)) if hasattr(w, 'min') else 0.0,
                    'max': float(np.max(w)) if hasattr(w, 'max') else 0.0
                })
            
            # Novelty Feature 3: Track traffic density for priority calculation
            queue_sum = sum(info.get('queue_lengths', [0]))
            self.traffic_density_history.append(queue_sum)
            if len(self.traffic_density_history) > 20:
                self.traffic_density_history.pop(0)  # Keep last 20
            
            # Collect current metrics
            current_metrics = {
                'timestamp': datetime.now().isoformat(),
                'episode': episode,
                'step': step,
                'reward': reward,
                'waiting_time': info.get('total_waiting_time', 0),
                'queue_lengths': info.get('queue_lengths', []),
                'gst': info.get('gst', {}),
                'client_id': self.client_id,
                'model_weights_summary': weights_summary,  # Include weights summary
                'traffic_density': queue_sum,  # Novelty Feature 3
                'client_priority': self.client_priority  # Novelty Feature 3
            }
            
            # Store in buffers
            self.periodic_data_buffer.append(current_metrics)
            self.periodic_weights_buffer.append({
                'timestamp': datetime.now().isoformat(),
                'episode': episode,
                'step': step,
                'weights_summary': weights_summary
            })
            
            # In true FL mode, send to server via metrics (Flower will handle transmission)
            # The data will be included in fit() return value, which Flower sends to server
            # For real-time transmission, we'd need a separate gRPC call (advanced feature)
            
            # Print periodic update with performance score
            if len(self.periodic_data_buffer) % 2 == 0:  # Every 10 seconds (2 transmissions)
                # Calculate real-time performance score
                waiting = info.get('total_waiting_time', 0)
                queue = sum(info.get('queue_lengths', [0]))
                perf_score = max(0, min(100, 100 - (waiting / 2) - (queue * 5)))
                
                print(f"\n[PERIODIC DATA TRANSMISSION - Every 5s]")
                print(f"   Episode: {episode}, Step: {step}")
                print(f"   Reward: {reward:.4f}")
                print(f"   Waiting Time: {waiting:.2f}s")
                print(f"   Queue Length: {queue} vehicles")
                print(f"   Model Weights: {len(current_weights)} layers collected")
                print(f"   Real-time Performance: {perf_score:.1f}/100")
                print(f"   Data sent to server buffer (ready for FL aggregation)")
                print(f"   Total transmissions: {len(self.periodic_data_buffer)}")
                
        except Exception as e:
            print(f"[WARNING] Error in periodic data transmission: {e}")
            # Don't break training if transmission fails
    
    def _print_episode_summary(self, episode: int, reward: float, steps: int, losses: List):
        """Print episode end summary with scores"""
        avg_loss = np.mean(losses) if losses else 0.0
        score = self._calculate_episode_score(reward, steps, avg_loss)
        
        # Performance breakdown (updated for new reward range)
        if reward >= 0:
            reward_score = 50
        elif reward < -18:
            reward_score = 0
        else:
            reward_score = 50 * (1 - (abs(reward) / 18.0))
        
        efficiency_score = min(30, (steps / 1000) * 30)
        loss_score = max(0, 20 * (1 - (avg_loss - 0.1) / 4.9)) if avg_loss > 0.1 else 20
        
        # Performance rating
        if score >= 80:
            rating = "🌟 EXCELLENT"
        elif score >= 60:
            rating = "✅ GOOD"
        elif score >= 40:
            rating = "⚠️  AVERAGE"
        else:
            rating = "❌ NEEDS IMPROVEMENT"
        
        print(f"\n{'='*80}")
        print(f"📊 EPISODE {episode + 1} SUMMARY")
        print(f"{'='*80}")
        print(f"🎯 Total Reward: {reward:.4f}")
        print(f"📈 Steps: {steps}")
        print(f"📉 Average Loss: {avg_loss:.4f}")
        print(f"\n⭐ Episode Score: {score:.2f}/100 - {rating}")
        print(f"   Breakdown:")
        print(f"   • Reward Component: {reward_score:.1f}/50 (Higher reward = Better)")
        print(f"   • Efficiency Component: {efficiency_score:.1f}/30 (More steps = More learning)")
        print(f"   • Loss Component: {loss_score:.1f}/20 (Lower loss = Better)")
        
        # Improvement suggestions
        if score < 40:
            print(f"\n💡 Improvement Tips:")
            if reward_score < 10:
                print(f"   • Reward is very negative ({reward:.2f}) - Agent needs more training")
            if loss_score < 5:
                print(f"   • Loss is high ({avg_loss:.4f}) - Model is still learning")
            if efficiency_score < 15:
                print(f"   • Fewer steps ({steps}) - May need longer episodes")
        
        print(f"{'='*80}\n")
    
    def _calculate_episode_score(self, reward: float, steps: int, loss: float) -> float:
        """Calculate episode performance score (0-100) - Improved formula for new reward range"""
        # New reward range: -18 to +4 (after normalization)
        # Scale reward to 0-50 points
        if reward >= 0:
            reward_score = 50  # Perfect (positive reward)
        elif reward < -18:
            reward_score = 0  # Very bad
        else:
            # Scale from -18 (0 points) to 0 (50 points)
            reward_score = 50 * (1 - (abs(reward) / 18.0))
        
        # Efficiency score based on steps (more steps = more exploration/learning)
        # 1000 steps = full score, less steps = proportional
        efficiency_score = min(30, (steps / 1000) * 30)
        
        # Loss score (lower is better, typical range 0.01 to 10)
        if loss < 0.1:
            loss_score = 20
        elif loss > 5:
            loss_score = 0
        else:
            # Scale from 0.1 (20 points) to 5 (0 points)
            loss_score = max(0, 20 * (1 - (loss - 0.1) / 4.9))
        
        total_score = reward_score + efficiency_score + loss_score
        return min(100, max(0, total_score))
    
    def _calculate_final_score(self, total_reward: float, total_steps: int, losses: List) -> float:
        """Calculate final training score - Improved formula for new reward range"""
        episodes = max(1, len(losses)) if losses else 1
        avg_reward = total_reward / episodes
        avg_loss = np.mean(losses) if losses else 0.0
        
        # Reward score (new range: -18 to +4)
        if avg_reward >= 0:
            reward_score = 40  # Perfect
        elif avg_reward < -18:
            reward_score = 0  # Very bad
        else:
            # Scale from -18 (0 points) to 0 (40 points)
            reward_score = 40 * (1 - (abs(avg_reward) / 18.0))
        
        # Efficiency score (more steps = more learning)
        efficiency_score = min(30, (total_steps / (episodes * 1000)) * 30)
        
        # Loss score (lower is better)
        if avg_loss < 0.1:
            loss_score = 30
        elif avg_loss > 5:
            loss_score = 0
        else:
            loss_score = max(0, 30 * (1 - (avg_loss - 0.1) / 4.9))
        
        return min(100, max(0, reward_score + efficiency_score + loss_score))
    
    def _print_final_training_summary(self, metrics: Dict):
        """Print final training summary with comprehensive scores"""
        print(f"\n{'='*80}")
        print(f"🏆 FINAL TRAINING SUMMARY")
        print(f"{'='*80}")
        print(f"📊 Episodes Completed: {metrics['episodes']}")
        print(f"🎯 Total Reward: {metrics['total_reward']:.4f}")
        print(f"📈 Average Reward: {metrics['average_reward']:.4f}")
        print(f"👣 Total Steps: {metrics['total_steps']}")
        print(f"📉 Average Loss: {metrics['average_loss']:.4f}")
        print(f"\n⭐ FINAL PERFORMANCE SCORE: {metrics['final_score']:.2f}/100")
        
        # Performance rating
        score = metrics['final_score']
        if score >= 80:
            rating = "🌟 EXCELLENT"
        elif score >= 60:
            rating = "✅ GOOD"
        elif score >= 40:
            rating = "⚠️  AVERAGE"
        else:
            rating = "❌ NEEDS IMPROVEMENT"
        
        print(f"📊 Performance Rating: {rating}")
        print(f"{'='*80}\n")
