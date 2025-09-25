#!/usr/bin/env python3
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
