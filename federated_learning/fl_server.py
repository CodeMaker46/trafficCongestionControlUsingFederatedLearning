import flwr as fl
from typing import Dict, List, Tuple, Optional
import numpy as np
import json
import time
from datetime import datetime
import os

class TrafficFLServer:
    """
    Federated Learning server for traffic control
    Coordinates training across multiple traffic intersections
    """
    
    def __init__(self, num_rounds: int = 10, min_clients: int = 2, 
                 min_fit_clients: int = 2, min_eval_clients: int = 2):
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients
        
        # Server state
        self.current_round = 0
        self.server_metrics = []
        self.client_metrics = {}
        
        # Results directory
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def get_fit_config(self, server_round: int) -> Dict:
        """Return configuration for fit round"""
        return {
            "round": server_round,
            "episodes": 10,
            "learning_rate": 0.001,
            "batch_size": 32
        }
    
    def get_eval_config(self, server_round: int) -> Dict:
        """Return configuration for eval round"""
        return {
            "round": server_round,
            "episodes": 5
        }
    
    def aggregate_fit(self, server_round: int, results: List[Tuple], 
                     failures: List) -> Tuple[Optional[Dict], Dict]:
        """Aggregate fit results from clients"""
        if not results:
            return None, {}
        
        # Extract parameters and metrics
        parameters = [result[0] for result in results]
        metrics = [result[2] for result in results]
        
        # Weighted average of parameters
        aggregated_parameters = self._weighted_average_parameters(parameters, metrics)
        
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(metrics)
        
        # Store metrics
        self.server_metrics.append({
            'round': server_round,
            'type': 'fit',
            'metrics': aggregated_metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, server_round: int, results: List[Tuple], 
                          failures: List) -> Tuple[Optional[float], Dict]:
        """Aggregate evaluation results from clients"""
        if not results:
            return None, {}
        
        # Extract metrics
        metrics = [result[2] for result in results]
        
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(metrics)
        
        # Calculate average loss
        losses = [result[0] for result in results]
        avg_loss = np.mean(losses)
        
        # Store metrics
        self.server_metrics.append({
            'round': server_round,
            'type': 'evaluate',
            'metrics': aggregated_metrics,
            'average_loss': avg_loss,
            'timestamp': datetime.now().isoformat()
        })
        
        return avg_loss, aggregated_metrics
    
    def _weighted_average_parameters(self, parameters_list: List, 
                                   metrics_list: List) -> List[np.ndarray]:
        """Calculate weighted average of model parameters"""
        if not parameters_list:
            return []
        
        # Use number of training steps as weights
        weights = [metrics.get('total_steps', 1) for metrics in metrics_list]
        total_weight = sum(weights)
        
        if total_weight == 0:
            # Fallback to simple average
            return [np.mean([params[i] for params in parameters_list], axis=0) 
                   for i in range(len(parameters_list[0]))]
        
        # Weighted average
        weighted_params = []
        for i in range(len(parameters_list[0])):
            weighted_sum = np.zeros_like(parameters_list[0][i])
            for j, params in enumerate(parameters_list):
                weighted_sum += params[i] * weights[j]
            weighted_params.append(weighted_sum / total_weight)
        
        return weighted_params
    
    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Aggregate metrics from multiple clients"""
        if not metrics_list:
            return {}
        
        aggregated = {}
        
        # Aggregate numerical metrics
        numerical_keys = ['average_reward', 'total_steps', 'average_loss', 
                         'waiting_time', 'queue_length', 'max_queue_length']
        
        for key in numerical_keys:
            values = [m.get(key, 0) for m in metrics_list if key in m]
            if values:
                aggregated[f'avg_{key}'] = np.mean(values)
                aggregated[f'std_{key}'] = np.std(values)
                aggregated[f'min_{key}'] = np.min(values)
                aggregated[f'max_{key}'] = np.max(values)
        
        # Count clients
        aggregated['num_clients'] = len(metrics_list)
        
        return aggregated
    
    def run_federated_learning(self, server_address: str = "localhost:8080"):
        """Run federated learning server"""
        print(f"Starting Federated Learning Server on {server_address}")
        print(f"Rounds: {self.num_rounds}, Min clients: {self.min_clients}")
        
        # Configure server
        strategy = fl.server.strategy.FedAvg(
            min_available_clients=self.min_clients,
            min_fit_clients=self.min_fit_clients,
            min_eval_clients=self.min_eval_clients,
            on_fit_config_fn=self.get_fit_config,
            on_evaluate_config_fn=self.get_eval_config,
        )
        
        # Start server
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=strategy,
        )
    
    def save_results(self):
        """Save server results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save server metrics
        server_file = os.path.join(self.results_dir, f"server_metrics_{timestamp}.json")
        with open(server_file, 'w') as f:
            json.dump(self.server_metrics, f, indent=2)
        
        print(f"Results saved to {server_file}")
    
    def print_round_summary(self, round_num: int, metrics: Dict):
        """Print summary for a round"""
        print(f"\n--- Round {round_num} Summary ---")
        print(f"Average Reward: {metrics.get('avg_average_reward', 0):.4f}")
        print(f"Average Loss: {metrics.get('avg_average_loss', 0):.4f}")
        print(f"Average Waiting Time: {metrics.get('avg_waiting_time', 0):.2f}")
        print(f"Average Queue Length: {metrics.get('avg_queue_length', 0):.2f}")
        print(f"Number of Clients: {metrics.get('num_clients', 0)}")
        print("-" * 30)
