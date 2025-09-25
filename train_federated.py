#!/usr/bin/env python3
"""
Federated Learning Traffic Control System
Main training script for traffic congestion control using federated learning
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from federated_learning.fl_server import TrafficFLServer
from federated_learning.fl_client import TrafficFLClient
from utils.visualization import TrafficVisualizer
import flwr as fl

def create_client_script():
    """Create a client script that can be run independently"""
    client_script = '''#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from federated_learning.fl_client import TrafficFLClient
import flwr as fl

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=str, required=True, help="Client ID")
    parser.add_argument("--sumo-config", type=str, required=True, help="Path to SUMO config")
    parser.add_argument("--server-address", type=str, default="localhost:8080", help="Server address")
    parser.add_argument("--gui", action="store_true", help="Enable SUMO GUI")
    args = parser.parse_args()
    
    # Create client
    client = TrafficFLClient(
        client_id=args.client_id,
        sumo_config_path=args.sumo_config,
        gui=args.gui
    )
    
    # Start client
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client
    )
'''
    
    with open("client.py", "w") as f:
        f.write(client_script)
    
    os.chmod("client.py", 0o755)
    print("Client script created: client.py")

def run_server(num_rounds=10, min_clients=2, server_address="localhost:8080"):
    """Run the federated learning server"""
    print("Starting Federated Learning Server...")
    print(f"Server Address: {server_address}")
    print(f"Rounds: {num_rounds}, Min Clients: {min_clients}")
    
    # Create server
    server = TrafficFLServer(
        num_rounds=num_rounds,
        min_clients=min_clients,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients
    )
    
    # Run server
    server.run_federated_learning(server_address)

def run_single_client_training(args=None):
    """Run training with a single client for testing"""
    print("Running single client training for testing...")
    
    # Create client
    client = TrafficFLClient(
        client_id="test_client",
        sumo_config_path="osm_sumo_configs/osm.sumocfg",
        gui=(getattr(args, 'gui', False) if args is not None else False)
    )
    
    # Simulate federated learning rounds
    num_rounds = 5
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1} ---")
        
        # Get current parameters
        current_params = client.get_parameters({})
        
        # Train client
        config = {
            "round": round_num,
            "episodes": 5,
            "learning_rate": 0.001
        }
        
        updated_params, num_samples, metrics = client.fit(current_params, config)
        
        # Evaluate client
        eval_metrics = client.evaluate(updated_params, config)
        
        print(f"Training Metrics: {metrics}")
        print(f"Evaluation Metrics: {eval_metrics[2]}")
        
        # Update client parameters
        client.set_parameters(updated_params)
    
    # Save results
    client.save_training_history("results/single_client_training.json")
    client.save_performance_metrics("results/single_client_performance.json")
    
    print("\nSingle client training completed!")
    print("Results saved to results/ directory")

def run_multi_client_simulation(args=None):
    """Run simulation with multiple clients"""
    print("Running multi-client simulation...")
    
    # Create multiple clients with different configurations
    clients = []
    want_gui = getattr(args, 'gui', False) if args is not None else False
    client_configs = [
        {"id": "client_1", "config": "osm_sumo_configs/osm.sumocfg", "gui": want_gui},
        {"id": "client_2", "config": "osm_sumo_configs/osm.sumocfg", "gui": want_gui},
        {"id": "client_3", "config": "osm_sumo_configs/osm.sumocfg", "gui": want_gui}
    ]
    
    for config in client_configs:
        client = TrafficFLClient(
            client_id=config["id"],
            sumo_config_path=config["config"],
            gui=config["gui"]
        )
        clients.append(client)
    
    # Simulate federated learning
    num_rounds = 10
    global_params = None
    
    for round_num in range(num_rounds):
        print(f"\n--- Federated Learning Round {round_num + 1} ---")
        
        # Train all clients
        client_metrics = []
        for i, client in enumerate(clients):
            print(f"Training client {i + 1}...")
            
            # Get current parameters
            current_params = client.get_parameters({})
            
            # Train client
            config = {
                "round": round_num,
                "episodes": 5,
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
            print(f"Client {i + 1} evaluation: {eval_result[2]}")
        
        # Print round summary
        avg_reward = np.mean([m.get('average_reward', 0) for m in client_metrics])
        avg_waiting = np.mean([m.get('waiting_time', 0) for m in eval_metrics])
        print(f"Round {round_num + 1} Summary:")
        print(f"  Average Reward: {avg_reward:.4f}")
        print(f"  Average Waiting Time: {avg_waiting:.2f}")
    
    # Save results
    for i, client in enumerate(clients):
        client.save_training_history(f"results/client_{i+1}_training.json")
        client.save_performance_metrics(f"results/client_{i+1}_performance.json")
    
    print("\nMulti-client simulation completed!")
    print("Results saved to results/ directory")

def main():
    parser = argparse.ArgumentParser(description="Federated Learning Traffic Control")
    parser.add_argument("--mode", choices=["server", "client", "single", "multi"], 
                       default="single", help="Run mode")
    parser.add_argument("--client-id", type=str, help="Client ID (for client mode)")
    parser.add_argument("--sumo-config", type=str, default="osm_sumo_configs/osm.sumocfg",
                       help="Path to SUMO config")
    parser.add_argument("--server-address", type=str, default="localhost:8080",
                       help="Server address")
    parser.add_argument("--num-rounds", type=int, default=10, help="Number of rounds")
    parser.add_argument("--min-clients", type=int, default=2, help="Minimum clients")
    parser.add_argument("--gui", action="store_true", help="Enable SUMO GUI")
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    if args.mode == "server":
        run_server(args.num_rounds, args.min_clients, args.server_address)
    elif args.mode == "client":
        if not args.client_id:
            print("Error: --client-id is required for client mode")
            return
        
        client = TrafficFLClient(
            client_id=args.client_id,
            sumo_config_path=args.sumo_config,
            gui=args.gui
        )
        
        fl.client.start_numpy_client(
            server_address=args.server_address,
            client=client
        )
    elif args.mode == "single":
        run_single_client_training(args)
    elif args.mode == "multi":
        run_multi_client_simulation(args)
    
    print("Training completed!")

if __name__ == "__main__":
    # Create client script
    create_client_script()
    
    # Run main function
    main()
