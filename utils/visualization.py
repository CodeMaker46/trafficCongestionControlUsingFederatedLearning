import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Dict, Optional
import os

class TrafficVisualizer:
    """
    Visualization utilities for traffic control results
    """
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_training_convergence(self, metrics_data: List[Dict], 
                                save_path: Optional[str] = None):
        """Plot training convergence metrics"""
        if not metrics_data:
            print("No metrics data to plot")
            return
        
        # Extract data
        rounds = [m['round'] for m in metrics_data if m['type'] == 'fit']
        rewards = [m['metrics'].get('avg_average_reward', 0) for m in metrics_data if m['type'] == 'fit']
        losses = [m['metrics'].get('avg_average_loss', 0) for m in metrics_data if m['type'] == 'fit']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot rewards
        ax1.plot(rounds, rewards, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Federated Learning Round')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Training Reward Convergence')
        ax1.grid(True, alpha=0.3)
        
        # Plot losses
        ax2.plot(rounds, losses, 'r-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Federated Learning Round')
        ax2.set_ylabel('Average Loss')
        ax2.set_title('Training Loss Convergence')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_metrics(self, metrics_data: List[Dict], 
                               save_path: Optional[str] = None):
        """Plot performance metrics over rounds"""
        if not metrics_data:
            print("No metrics data to plot")
            return
        
        # Extract evaluation data
        eval_data = [m for m in metrics_data if m['type'] == 'evaluate']
        if not eval_data:
            print("No evaluation data to plot")
            return
        
        rounds = [m['round'] for m in eval_data]
        waiting_times = [m['metrics'].get('avg_waiting_time', 0) for m in eval_data]
        queue_lengths = [m['metrics'].get('avg_queue_length', 0) for m in eval_data]
        max_queue_lengths = [m['metrics'].get('avg_max_queue_length', 0) for m in eval_data]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot waiting times
        ax1.plot(rounds, waiting_times, 'g-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Federated Learning Round')
        ax1.set_ylabel('Average Waiting Time (s)')
        ax1.set_title('Traffic Waiting Time')
        ax1.grid(True, alpha=0.3)
        
        # Plot queue lengths
        ax2.plot(rounds, queue_lengths, 'orange', marker='o', linewidth=2, markersize=6)
        ax2.set_xlabel('Federated Learning Round')
        ax2.set_ylabel('Average Queue Length')
        ax2.set_title('Traffic Queue Length')
        ax2.grid(True, alpha=0.3)
        
        # Plot max queue lengths
        ax3.plot(rounds, max_queue_lengths, 'purple', marker='o', linewidth=2, markersize=6)
        ax3.set_xlabel('Federated Learning Round')
        ax3.set_ylabel('Max Queue Length')
        ax3.set_title('Maximum Queue Length')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_client_comparison(self, client_metrics: Dict, 
                             save_path: Optional[str] = None):
        """Plot comparison between different clients"""
        if not client_metrics:
            print("No client metrics to plot")
            return
        
        # Prepare data
        clients = list(client_metrics.keys())
        rewards = []
        waiting_times = []
        
        for client_id, metrics in client_metrics.items():
            if metrics:
                avg_reward = np.mean([m.get('average_reward', 0) for m in metrics])
                avg_waiting = np.mean([m.get('waiting_time', 0) for m in metrics])
                rewards.append(avg_reward)
                waiting_times.append(avg_waiting)
            else:
                rewards.append(0)
                waiting_times.append(0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot rewards by client
        bars1 = ax1.bar(clients, rewards, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Client ID')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Client Performance Comparison - Rewards')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, rewards):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Plot waiting times by client
        bars2 = ax2.bar(clients, waiting_times, color='lightcoral', alpha=0.7)
        ax2.set_xlabel('Client ID')
        ax2.set_ylabel('Average Waiting Time (s)')
        ax2.set_title('Client Performance Comparison - Waiting Times')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars2, waiting_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curves(self, training_data: List[Dict], 
                           save_path: Optional[str] = None):
        """Plot detailed learning curves"""
        if not training_data:
            print("No training data to plot")
            return
        
        # Extract data
        rounds = [d['round'] for d in training_data]
        episodes = [d['episodes'] for d in training_data]
        rewards = [d['metrics'].get('average_reward', 0) for d in training_data]
        losses = [d['metrics'].get('average_loss', 0) for d in training_data]
        steps = [d['metrics'].get('total_steps', 0) for d in training_data]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot rewards
        ax1.plot(rounds, rewards, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Reward Progression')
        ax1.grid(True, alpha=0.3)
        
        # Plot losses
        ax2.plot(rounds, losses, 'r-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Average Loss')
        ax2.set_title('Loss Progression')
        ax2.grid(True, alpha=0.3)
        
        # Plot episodes
        ax3.plot(rounds, episodes, 'g-o', linewidth=2, markersize=6)
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Episodes')
        ax3.set_title('Training Episodes per Round')
        ax3.grid(True, alpha=0.3)
        
        # Plot steps
        ax4.plot(rounds, steps, 'purple', marker='o', linewidth=2, markersize=6)
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Total Steps')
        ax4.set_title('Training Steps per Round')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_summary_report(self, metrics_data: List[Dict], 
                            client_metrics: Dict, 
                            save_path: Optional[str] = None):
        """Create a comprehensive summary report"""
        if not metrics_data:
            print("No data to create report")
            return
        
        # Create summary statistics
        eval_data = [m for m in metrics_data if m['type'] == 'evaluate']
        fit_data = [m for m in metrics_data if m['type'] == 'fit']
        
        report = {
            'summary': {
                'total_rounds': len(fit_data),
                'total_evaluations': len(eval_data),
                'num_clients': len(client_metrics)
            },
            'final_performance': {},
            'improvement': {}
        }
        
        if eval_data:
            final_metrics = eval_data[-1]['metrics']
            report['final_performance'] = {
                'average_reward': final_metrics.get('avg_average_reward', 0),
                'waiting_time': final_metrics.get('avg_waiting_time', 0),
                'queue_length': final_metrics.get('avg_queue_length', 0),
                'max_queue_length': final_metrics.get('avg_max_queue_length', 0)
            }
        
        if len(eval_data) > 1:
            initial_metrics = eval_data[0]['metrics']
            final_metrics = eval_data[-1]['metrics']
            
            report['improvement'] = {
                'reward_improvement': final_metrics.get('avg_average_reward', 0) - initial_metrics.get('avg_average_reward', 0),
                'waiting_time_reduction': initial_metrics.get('avg_waiting_time', 0) - final_metrics.get('avg_waiting_time', 0),
                'queue_length_reduction': initial_metrics.get('avg_queue_length', 0) - final_metrics.get('avg_queue_length', 0)
            }
        
        # Save report
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        # Print summary
        print("=== FEDERATED LEARNING SUMMARY REPORT ===")
        print(f"Total Rounds: {report['summary']['total_rounds']}")
        print(f"Total Evaluations: {report['summary']['total_evaluations']}")
        print(f"Number of Clients: {report['summary']['num_clients']}")
        print("\nFinal Performance:")
        for key, value in report['final_performance'].items():
            print(f"  {key}: {value:.4f}")
        print("\nImprovements:")
        for key, value in report['improvement'].items():
            print(f"  {key}: {value:.4f}")
        print("=" * 40)
        
        return report
