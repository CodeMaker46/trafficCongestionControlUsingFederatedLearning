import flwr as fl
from typing import Dict, List, Tuple, Optional
import numpy as np
import json
import time
from datetime import datetime
import os

class TrafficFLServer:
    
    def __init__(self, num_rounds: int = 10, min_clients: int = 2, 
                 min_fit_clients: int = 2, min_eval_clients: int = 2):
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients
        
        self.current_round = 0
        self.server_metrics = []
        self.client_metrics = {}
        
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def get_fit_config(self, server_round: int) -> Dict:
        return {
            "round": server_round,
            "episodes": 10,
            "learning_rate": 0.001,
            "batch_size": 32
        }
    
    def get_eval_config(self, server_round: int) -> Dict:
        return {
            "round": server_round,
            "episodes": 5
        }
    
    def aggregate_fit(self, server_round: int, results: List[Tuple], 
                     failures: List) -> Tuple[Optional[Dict], Dict]:
        if not results:
            return None, {}
        
        parameters = [result[0] for result in results]
        metrics = [result[2] for result in results]
        
        aggregated_parameters = self._weighted_average_parameters(parameters, metrics)
        
        aggregated_metrics = self._aggregate_metrics(metrics)
        
        self.server_metrics.append({
            'round': server_round,
            'type': 'fit',
            'metrics': aggregated_metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, server_round: int, results: List[Tuple], 
                          failures: List) -> Tuple[Optional[float], Dict]:
        if not results:
            return None, {}
        
        metrics = [result[2] for result in results]
        
        aggregated_metrics = self._aggregate_metrics(metrics)
        
        losses = [result[0] for result in results]
        avg_loss = np.mean(losses)
        
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
        if not parameters_list:
            return []
        
        weights = [metrics.get('total_steps', 1) for metrics in metrics_list]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return [np.mean([params[i] for params in parameters_list], axis=0) 
                   for i in range(len(parameters_list[0]))]
        
        weighted_params = []
        for i in range(len(parameters_list[0])):
            weighted_sum = np.zeros_like(parameters_list[0][i])
            for j, params in enumerate(parameters_list):
                weighted_sum += params[i] * weights[j]
            weighted_params.append(weighted_sum / total_weight)
        
        return weighted_params
    
    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict:
        if not metrics_list:
            return {}
        
        aggregated = {}
        
        numerical_keys = ['average_reward', 'total_steps', 'average_loss', 
                         'waiting_time', 'queue_length', 'max_queue_length']
        
        for key in numerical_keys:
            values = [m.get(key, 0) for m in metrics_list if key in m]
            if values:
                aggregated[f'avg_{key}'] = np.mean(values)
                aggregated[f'std_{key}'] = np.std(values)
                aggregated[f'min_{key}'] = np.min(values)
                aggregated[f'max_{key}'] = np.max(values)
        
        aggregated['num_clients'] = len(metrics_list)
        
        return aggregated
    
    def run_federated_learning(self, server_address: str = "localhost:8080"):
        print(f"Starting Federated Learning Server on {server_address}")
        print(f"Rounds: {self.num_rounds}, Min clients: {self.min_clients}")
        
        strategy = fl.server.strategy.FedAvg(
            min_available_clients=self.min_clients,
            min_fit_clients=self.min_fit_clients,
            min_evaluate_clients=self.min_eval_clients,
            on_fit_config_fn=self.get_fit_config,
            on_evaluate_config_fn=self.get_eval_config,
        )
        
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=strategy,
        )
    
    def save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        server_file = os.path.join(self.results_dir, f"server_metrics_{timestamp}.json")
        with open(server_file, 'w') as f:
            json.dump(self.server_metrics, f, indent=2)
        
        print(f"Results saved to {server_file}")
    
    def print_round_summary(self, round_num: int, metrics: Dict):
        print(f"\n--- Round {round_num} Summary ---")
        print(f"Average Reward: {metrics.get('avg_average_reward', 0):.4f}")
        print(f"Average Loss: {metrics.get('avg_average_loss', 0):.4f}")
        print(f"Average Waiting Time: {metrics.get('avg_waiting_time', 0):.2f}")
        print(f"Average Queue Length: {metrics.get('avg_queue_length', 0):.2f}")
        print(f"Number of Clients: {metrics.get('num_clients', 0)}")
        print("-" * 30)
