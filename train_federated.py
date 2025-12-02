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
    parser.add_argument("--show-phase-console", action="store_true", help="Print TLS phase/time each step")
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
        min_eval_clients=min_clients
    )
    
    # Run server
    server.run_federated_learning(server_address)

def run_single_client_training(gui: bool = False, show_phase_console: bool = False, 
                               sumo_config: str = "sumo_configs2/osm.sumocfg", 
                               tl_id: str = None):
    """Run training with a single client for testing"""
    print("Running single client training for testing...")
    if tl_id:
        print(f"🎯 Using specified Traffic Light ID: {tl_id}")
    
    # Create client
    client = TrafficFLClient(
        client_id="test_client",
        sumo_config_path=sumo_config,
        gui=gui,
        show_phase_console=show_phase_console,
        show_gst_gui=gui,
        tl_id=tl_id
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
        
        print(f"📊 Training configuration:")
        print(f"   Episodes per round: {config['episodes']}")
        print(f"   Learning rate: {config['learning_rate']}")
        print(f"   Starting training...\n")
        
        try:
            updated_params, num_samples, metrics = client.fit(current_params, config)
            print(f"✅ Training completed for Round {round_num + 1}")
        except Exception as e:
            print(f"❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            print(f"   Skipping to next round...")
            continue
        
        # Evaluate client
        eval_metrics = client.evaluate(updated_params, config)
        
        print(f"Training Metrics: {metrics}")
        
        # Print comprehensive evaluation metrics
        print("\n" + "="*60)
        print("DETAILED EVALUATION METRICS")
        print("="*60)
        
        # Overall metrics
        print(f"Overall Performance:")
        print(f"  Average Reward: {eval_metrics[2].get('average_reward', 0):.4f}")
        print(f"  Total Waiting Time: {eval_metrics[2].get('waiting_time', 0):.2f}s")
        print(f"  Average Queue Length: {eval_metrics[2].get('queue_length', 0):.2f}")
        print(f"  Max Queue Length: {eval_metrics[2].get('max_queue_length', 0)}")
        
        # Green Signal Time metrics
        gst = eval_metrics[2].get('green_signal_time', {})
        print(f"\nGreen Signal Time (GST):")
        try:
            avg_gst = gst.get('avg_gst', 0.0)
            num_lanes = gst.get('num_lanes', 0)
            print(f"  Average GST: {avg_gst:.2f}s over {num_lanes} lanes")
            per_edge = gst.get('per_edge', {})
            if per_edge:
                for edge_id, val in per_edge.items():
                    print(f"    Lane {edge_id}: {float(val):.2f}s")
            else:
                print("    Per-lane GST: Not available")
        except Exception:
            print("  GST: Not available")
        
        # Detailed per-road metrics - Each road separately
        per_lane_metrics = eval_metrics[2].get('per_lane_metrics', {})
        lane_summary = eval_metrics[2].get('lane_summary', {})
        
        if per_lane_metrics:
            print(f"\n" + "="*80)
            print("TARGET INTERSECTION - EACH ROAD METRICS")
            print("="*80)
            
            road_count = 1
            for edge_id, lane_data in per_lane_metrics.items():
                print(f"\nROAD #{road_count}: {edge_id}")
                print("-" * 50)
                
                # Core metrics you requested
                vehicles = lane_data.get('vehicle_count', 0)
                queue_length = lane_data.get('queue_length', 0)
                waiting_time = lane_data.get('waiting_time', 0)
                green_signal_time = lane_data.get('green_signal_time', 0)
                avg_speed = lane_data.get('average_speed', 0)
                
                print(f"Vehicles on Road: {vehicles}")
                print(f"Queue Length: {queue_length} vehicles")
                print(f"Waiting Time: {waiting_time:.2f} seconds")
                print(f"Green Signal Time: {green_signal_time:.2f} seconds")
                print(f"Average Speed: {avg_speed:.2f} m/s")
                
                # Add traffic light status (this will be added to lane_data in next update)
                print(f"Traffic Light Status: Will be shown in real-time console")
                
                # Additional useful info
                occupancy = lane_data.get('occupancy_percent', 0)
                lane_length = lane_data.get('lane_length', 0)
                print(f"Road Occupancy: {occupancy:.1f}%")
                print(f"Road Length: {lane_length:.1f} meters")
                
                # Vehicle type breakdown
                vehicle_types = lane_data.get('vehicle_types', {})
                if vehicle_types:
                    type_str = ", ".join([f"{vtype}: {count}" for vtype, count in vehicle_types.items()])
                    print(f"Vehicle Types: {type_str}")
                else:
                    print(f"Vehicle Types: No vehicles")
                
                road_count += 1
        
        if lane_summary:
            print(f"Lane Summary Statistics:")
            print(f"  Total Vehicles: {lane_summary.get('total_vehicles', 0)}")
            print(f"  Total Queue Length: {lane_summary.get('total_queue_length', 0)}")
            print(f"  Total Waiting Time: {lane_summary.get('total_waiting_time', 0):.2f}s")
            print(f"  Average GST: {lane_summary.get('average_gst', 0):.2f}s")
            print(f"  Min GST: {lane_summary.get('min_gst', 0):.2f}s")
            print(f"  Max GST: {lane_summary.get('max_gst', 0):.2f}s")
            print(f"  Max Queue Length: {lane_summary.get('max_queue_length', 0)}")
            print(f"  Max Waiting Time: {lane_summary.get('max_waiting_time', 0):.2f}s")
            print(f"  Average Speed: {lane_summary.get('average_speed', 0):.2f} m/s")
            print(f"  Active Lanes: {lane_summary.get('num_active_lanes', 0)}")
            print(f"  Congested Lanes: {lane_summary.get('num_congested_lanes', 0)}")
            print(f"  Total Occupancy: {lane_summary.get('total_occupancy', 0):.1f}%")
        
        print("="*60)
        
        # Print final round score
        final_score = eval_metrics[2].get('final_score', 0)
        avg_reward = eval_metrics[2].get('average_reward', 0)
        total_waiting = eval_metrics[2].get('waiting_time', 0)
        
        print(f"\n{'='*80}")
        print(f"🏆 ROUND {round_num + 1} FINAL SCORES")
        print(f"{'='*80}")
        print(f"⭐ Performance Score: {final_score:.2f}/100")
        print(f"🎯 Average Reward: {avg_reward:.4f}")
        print(f"⏱️  Total Waiting Time: {total_waiting:.2f}s")
        print(f"📊 Queue Length: {eval_metrics[2].get('queue_length', 0):.2f}")
        
        # Calculate efficiency metrics
        efficiency = max(0, 100 - (total_waiting / 10))  # Inverse of waiting time
        print(f"⚡ Traffic Efficiency: {efficiency:.2f}%")
        print(f"{'='*80}\n")
        
        # Update client parameters
        client.set_parameters(updated_params)
    
    # Save results
    client.save_training_history("results/single_client_training.json")
    client.save_performance_metrics("results/single_client_performance.json")
    
    # Final overall summary
    print(f"\n{'='*80}")
    print(f"🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"📁 Results saved to: results/ directory")
    print(f"📊 Training History: results/single_client_training.json")
    print(f"📈 Performance Metrics: results/single_client_performance.json")
    print(f"\n💡 Next Steps:")
    print(f"   • Review performance metrics in JSON files")
    print(f"   • Run multi-client training: python train_federated.py --mode multi")
    print(f"   • Generate comparison charts: python report_charts.py")
    print(f"{'='*80}\n")

def run_multi_client_simulation():
    """Run simulation with multiple clients using different intersections"""
    print("="*80)
    print("MULTI-CLIENT FEDERATED LEARNING SIMULATION")
    print("="*80)
    print("\n⚠️  IMPORTANT: Each client should use a DIFFERENT intersection!")
    print("   Use extract_intersection.py to create separate configs first.\n")
    
    # Use full network with different TL IDs (MORE RELIABLE than extracted networks)
    multi_config_dir = "sumo_configs/multi_intersections"
    summary_file = os.path.join(multi_config_dir, "intersections_summary.txt")
    client_configs = None
    
    # Try to read TL IDs from summary file
    if os.path.exists(summary_file):
        print(f"[INFO] Found intersection summary: {summary_file}")
        try:
            import re
            with open(summary_file, 'r') as f:
                content = f.read()
                # Extract junction IDs from summary
                matches = re.findall(r'Junction ID: ([^\n]+)', content)
                tl_ids = matches[:3]  # Get first 3
            
            if len(tl_ids) >= 2:
                print(f"[OK] Found {len(tl_ids)} TL IDs, using full network with different intersections")
                client_configs = [
                    {"id": "client_1", "config": "sumo_configs2/osm.sumocfg", "gui": False, "tl_id": tl_ids[0]},
                    {"id": "client_2", "config": "sumo_configs2/osm.sumocfg", "gui": False, "tl_id": tl_ids[1]},
                    {"id": "client_3", "config": "sumo_configs2/osm.sumocfg", "gui": False, "tl_id": tl_ids[2] if len(tl_ids) > 2 else None}
                ]
                print(f"   Client 1 TL ID: {tl_ids[0]}")
                print(f"   Client 2 TL ID: {tl_ids[1]}")
                if len(tl_ids) > 2:
                    print(f"   Client 3 TL ID: {tl_ids[2]}")
        except Exception as e:
            print(f"[WARNING] Could not read summary file: {e}")
            client_configs = None
    
    # Fallback: Use same config but with different TL IDs (BETTER APPROACH)
    if client_configs is None or len(client_configs) < 2:
        print("\n[INFO] Using full network with different TL IDs for each client")
        print("   This is MORE RELIABLE than extracted networks.\n")
        
        # Try to read TL IDs from summary file
        summary_file = os.path.join(multi_config_dir, "intersections_summary.txt")
        tl_ids = []
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r') as f:
                    content = f.read()
                    # Extract junction IDs from summary
                    import re
                    matches = re.findall(r'Junction ID: ([^\n]+)', content)
                    tl_ids = matches[:3]  # Get first 3
            except:
                pass
        
        # If we found TL IDs, use them; otherwise use None (auto-discover)
        if len(tl_ids) >= 2:
            print(f"   Found {len(tl_ids)} TL IDs from summary file")
            client_configs = [
                {"id": "client_1", "config": "sumo_configs2/osm.sumocfg", "gui": False, "tl_id": tl_ids[0] if len(tl_ids) > 0 else None},
                {"id": "client_2", "config": "sumo_configs2/osm.sumocfg", "gui": False, "tl_id": tl_ids[1] if len(tl_ids) > 1 else None},
                {"id": "client_3", "config": "sumo_configs2/osm.sumocfg", "gui": False, "tl_id": tl_ids[2] if len(tl_ids) > 2 else None}
            ]
        else:
            print("   Warning: Could not find TL IDs. Using auto-discovery (may use same intersection)")
            client_configs = [
                {"id": "client_1", "config": "sumo_configs2/osm.sumocfg", "gui": False, "tl_id": None},
                {"id": "client_2", "config": "sumo_configs2/osm.sumocfg", "gui": False, "tl_id": None},
                {"id": "client_3", "config": "sumo_configs2/osm.sumocfg", "gui": False, "tl_id": None}
            ]
    
    # Create multiple clients with different configurations
    clients = []
    print(f"\n📋 Creating {len(client_configs)} clients...")
    for i, config in enumerate(client_configs, 1):
        print(f"   Client {i}: {config['id']}")
        print(f"      Config: {config['config']}")
        if config.get('tl_id'):
            print(f"      Junction ID: {config['tl_id']}")
        
        client = TrafficFLClient(
            client_id=config["id"],
            sumo_config_path=config["config"],
            gui=config["gui"],
            tl_id=config.get("tl_id")
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
    parser.add_argument("--sumo-config", type=str, default="sumo_configs2/osm.sumocfg",
                       help="Path to SUMO config")
    parser.add_argument("--server-address", type=str, default="localhost:8080",
                       help="Server address")
    parser.add_argument("--num-rounds", type=int, default=10, help="Number of rounds")
    parser.add_argument("--min-clients", type=int, default=2, help="Minimum clients")
    parser.add_argument("--gui", action="store_true", help="Enable SUMO GUI")
    parser.add_argument("--show-phase-console", action="store_true", help="Print TLS phase/time each step")
    parser.add_argument("--tl-id", type=str, default=None, 
                       help="Specify Traffic Light ID directly (skips auto-discovery)")
    
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
            gui=args.gui,
            show_phase_console=args.show_phase_console,
            tl_id=args.tl_id
        )
        
        fl.client.start_numpy_client(
            server_address=args.server_address,
            client=client
        )
    elif args.mode == "single":
        run_single_client_training(gui=args.gui, show_phase_console=args.show_phase_console,
                                  sumo_config=args.sumo_config, tl_id=args.tl_id)
    elif args.mode == "multi":
        run_multi_client_simulation()
    
    print("Training completed!")

if __name__ == "__main__":
    # Create client script
    create_client_script()
    
    # Run main function
    main()
