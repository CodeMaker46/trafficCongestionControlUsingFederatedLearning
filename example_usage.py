#!/usr/bin/env python3
"""
Example usage of the Federated Learning Traffic Control System
This script demonstrates how to use the system for traffic congestion control
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.dqn_agent import DQNAgent
from agents.traffic_environment import SUMOTrafficEnvironment
from federated_learning.fl_client import TrafficFLClient
from federated_learning.fl_server import TrafficFLServer
from utils.visualization import TrafficVisualizer

def example_single_agent_training():
    """Example: Train a single DQN agent"""
    print("=" * 60)
    print("EXAMPLE 1: Single Agent Training")
    print("=" * 60)
    
    # Create environment
    env = SUMOTrafficEnvironment("sumo_configs2/osm.sumocfg", gui=False)
    
    # Create agent
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
    
    # Training parameters
    episodes = 20
    max_steps = 100
    
    print(f"Training agent for {episodes} episodes...")
    
    # Training loop
    rewards = []
    losses = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_losses = []
        
        for step in range(max_steps):
            # Choose action
            action = agent.act(state, training=True)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                if loss is not None:
                    episode_losses.append(loss)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        rewards.append(episode_reward)
        losses.append(np.mean(episode_losses) if episode_losses else 0)
        
        if episode % 5 == 0:
            print(f"Episode {episode}: Reward = {episode_reward:.2f}, Loss = {np.mean(episode_losses):.4f}")
    
    # Close environment
    env.close()
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title("Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/single_agent_training.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Single agent training completed!")
    print("Results saved to results/single_agent_training.png")

def example_federated_learning():
    """Example: Federated learning with multiple clients"""
    print("=" * 60)
    print("EXAMPLE 2: Federated Learning Simulation")
    print("=" * 60)
    
    # Create multiple clients
    clients = []
    client_configs = [
        {"id": "intersection_1", "config": "sumo_configs2/osm.sumocfg"},
        {"id": "intersection_2", "config": "sumo_configs2/osm.sumocfg"},
        {"id": "intersection_3", "config": "sumo_configs2/osm.sumocfg"}
    ]
    
    for config in client_configs:
        client = TrafficFLClient(
            client_id=config["id"],
            sumo_config_path=config["config"],
            gui=False
        )
        clients.append(client)
    
    # Simulate federated learning rounds
    num_rounds = 5
    global_params = None
    round_metrics = []
    
    print(f"Running federated learning for {num_rounds} rounds...")
    
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1} ---")
        
        # Train all clients
        client_metrics = []
        for i, client in enumerate(clients):
            print(f"Training client {i + 1}...")
            
            # Get current parameters
            current_params = client.get_parameters({})
            
            # Train client
            config = {
                "round": round_num,
                "episodes": 3,
                "learning_rate": 0.001
            }
            
            updated_params, num_samples, metrics = client.fit(current_params, config)
            client_metrics.append(metrics)
            
            # Store parameters for aggregation
            if global_params is None:
                global_params = updated_params
            else:
                # Simple average aggregation
                for j in range(len(global_params)):
                    global_params[j] = (global_params[j] + updated_params[j]) / 2
        
        # Update all clients with aggregated parameters
        for client in clients:
            client.set_parameters(global_params)
        
        # Evaluate all clients
        eval_metrics = []
        for i, client in enumerate(clients):
            eval_result = client.evaluate(global_params, config)
            eval_metrics.append(eval_result[2])
        
        # Calculate round metrics
        avg_reward = np.mean([m.get('average_reward', 0) for m in client_metrics])
        avg_waiting = np.mean([m.get('waiting_time', 0) for m in eval_metrics])
        
        round_metrics.append({
            'round': round_num,
            'avg_reward': avg_reward,
            'avg_waiting_time': avg_waiting,
            'num_clients': len(clients)
        })
        
        print(f"Round {round_num + 1} Summary:")
        print(f"  Average Reward: {avg_reward:.4f}")
        print(f"  Average Waiting Time: {avg_waiting:.2f}")
    
    # Plot federated learning results
    plt.figure(figsize=(12, 4))
    
    rounds = [m['round'] for m in round_metrics]
    rewards = [m['avg_reward'] for m in round_metrics]
    waiting_times = [m['avg_waiting_time'] for m in round_metrics]
    
    plt.subplot(1, 2, 1)
    plt.plot(rounds, rewards, 'b-o', linewidth=2, markersize=6)
    plt.title("Federated Learning - Average Reward")
    plt.xlabel("Round")
    plt.ylabel("Average Reward")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(rounds, waiting_times, 'r-o', linewidth=2, markersize=6)
    plt.title("Federated Learning - Waiting Time")
    plt.xlabel("Round")
    plt.ylabel("Average Waiting Time (s)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/federated_learning_example.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Federated learning simulation completed!")
    print("Results saved to results/federated_learning_example.png")

def example_performance_comparison():
    """Example: Compare different training strategies"""
    print("=" * 60)
    print("EXAMPLE 3: Performance Comparison")
    print("=" * 60)
    
    # Test different learning rates
    learning_rates = [0.001, 0.01, 0.1]
    results = {}
    
    for lr in learning_rates:
        print(f"Testing learning rate: {lr}")
        
        # Create environment and agent
        env = SUMOTrafficEnvironment("sumo_configs2/osm.sumocfg", gui=False)
        agent = DQNAgent(state_size=env.state_size, action_size=env.action_size, learning_rate=lr)
        
        # Training
        episodes = 10
        rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            
            for step in range(50):
                action = agent.act(state, training=True)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                
                if len(agent.memory) > agent.batch_size:
                    agent.replay()
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            rewards.append(episode_reward)
        
        results[lr] = rewards
        env.close()
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    for lr, rewards in results.items():
        plt.plot(rewards, label=f"LR = {lr}", linewidth=2)
    
    plt.title("Performance Comparison - Different Learning Rates")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Performance comparison completed!")
    print("Results saved to results/performance_comparison.png")

def main():
    """Run all examples"""
    print("Federated Learning Traffic Control System - Examples")
    print("=" * 60)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Run examples
    try:
        example_single_agent_training()
        print("\n")
        
        example_federated_learning()
        print("\n")
        
        example_performance_comparison()
        print("\n")
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("Check the results/ directory for generated plots and data.")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure SUMO is installed and accessible.")
        print("Run 'python setup.py' to check your installation.")

if __name__ == "__main__":
    main()
