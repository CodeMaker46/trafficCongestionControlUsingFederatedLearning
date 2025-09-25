# Federated Learning Traffic Congestion Control with SUMO

This project implements a federated learning system for traffic congestion control using SUMO (Simulation of Urban Mobility) and Deep Q-Networks (DQN). The system allows multiple traffic intersections to collaboratively learn optimal traffic light control strategies without sharing raw data.

## Features

- **Federated Learning**: Multiple traffic intersections learn collaboratively
- **SUMO Integration**: Realistic traffic simulation using SUMO
- **Deep Q-Networks**: Reinforcement learning for traffic light control
- **Distributed Training**: Clients can run on different machines
- **Performance Visualization**: Comprehensive metrics and plots
- **Scalable Architecture**: Easy to add more intersections

## Project Structure

```
TCC/
├── agents/
│   ├── dqn_agent.py          # DQN agent implementation
│   └── traffic_environment.py # SUMO environment wrapper
├── federated_learning/
│   ├── fl_client.py          # Federated learning client
│   └── fl_server.py          # Federated learning server
├── sumo_configs/
│   ├── intersection.net.xml  # SUMO network file
│   ├── intersection.rou.xml  # SUMO routes file
│   └── intersection.sumocfg  # SUMO configuration
├── utils/
│   └── visualization.py      # Visualization utilities
├── results/                  # Training results and plots
├── train_federated.py        # Main training script
├── client.py                 # Client script (auto-generated)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

### Prerequisites

1. **Python 3.8+**
2. **SUMO (Simulation of Urban Mobility)**
   - Download from: https://sumo.dlr.de/docs/Downloads.php
   - Add SUMO to your PATH environment variable

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd TCC
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify SUMO installation**:
   ```bash
   sumo --version
   ```

## Usage

### Quick Start (Single Client Testing)

Run a single client for testing and development:

```bash
python train_federated.py --mode single
```

### Multi-Client Simulation

Run multiple clients on the same machine:

```bash
python train_federated.py --mode multi
```

### Distributed Federated Learning

#### 1. Start the Server

On the server machine:

```bash
python train_federated.py --mode server --num-rounds 10 --min-clients 2
```

#### 2. Start Clients

On client machines (or same machine in different terminals):

```bash
# Client 1
   python client.py --client-id client_1 --sumo-config sumo_configs2/osm.sumocfg

# Client 2
   python client.py --client-id client_2 --sumo-config sumo_configs2/osm.sumocfg

# Client 3
   python client.py --client-id client_3 --sumo-config sumo_configs2/osm.sumocfg
```

### With SUMO GUI

To visualize the traffic simulation:

```bash
python train_federated.py --mode single --gui
```

## Configuration

### SUMO Configuration

The traffic simulation can use configs from `sumo_configs/` or OSM-based files in `sumo_configs2/`. By default, examples reference `sumo_configs2/osm.sumocfg`.

- `intersection.net.xml`: Network topology (intersections, roads, lanes)
- `intersection.rou.xml`: Vehicle routes and traffic flows
- `intersection.sumocfg`: Main SUMO configuration

### Training Parameters

Modify training parameters in the respective files:

- **DQN Agent**: `agents/dqn_agent.py`
- **Environment**: `agents/traffic_environment.py`
- **Federated Learning**: `federated_learning/fl_server.py`

## Results and Visualization

Training results are saved in the `results/` directory:

- `server_metrics_*.json`: Server-side metrics
- `client_*_training.json`: Client training history
- `client_*_performance.json`: Client performance metrics

### Visualization

The system includes comprehensive visualization tools:

```python
from utils.visualization import TrafficVisualizer

# Create visualizer
viz = TrafficVisualizer("results")

# Load and plot results
metrics_data = [...]  # Load from JSON files
viz.plot_training_convergence(metrics_data)
viz.plot_performance_metrics(metrics_data)
```

## Architecture

### Federated Learning Flow

1. **Server Initialization**: Global model parameters are initialized
2. **Client Training**: Each client trains on local traffic data
3. **Parameter Aggregation**: Server aggregates client parameters
4. **Model Update**: Updated model is distributed to all clients
5. **Evaluation**: Performance is evaluated across all clients

### DQN Agent

The DQN agent uses:
- **State Space**: Queue lengths and waiting times for each direction
- **Action Space**: 4 traffic light phases (EW Green, EW Yellow, NS Green, NS Yellow)
- **Reward Function**: Minimizes waiting time and queue length

### SUMO Integration

The system interfaces with SUMO through:
- **TraCI**: Traffic Control Interface for real-time control
- **Custom Environment**: Wrapper for reinforcement learning
- **Performance Metrics**: Waiting time, queue length, throughput

## Customization

### Adding New Intersections

1. Create new SUMO configuration files
2. Modify the traffic environment for new topology
3. Update state/action spaces if needed

### Modifying Reward Function

Edit the `_calculate_reward` method in `agents/traffic_environment.py`:

```python
def _calculate_reward(self) -> float:
    # Custom reward function
    waiting_penalty = -(ew_waiting + ns_waiting) / 100.0
    queue_penalty = -(ew_queue + ns_queue) / 10.0
    flow_reward = 1.0 if (ew_queue + ns_queue) < 5 else 0.0
    
    return waiting_penalty + queue_penalty + flow_reward
```

### Adding New Clients

1. Create client with unique ID
2. Point to appropriate SUMO configuration
3. Start client with `client.py`

## Performance Metrics

The system tracks several performance metrics:

- **Average Reward**: Overall performance indicator
- **Waiting Time**: Total time vehicles spend waiting
- **Queue Length**: Number of vehicles in queue
- **Throughput**: Number of vehicles processed
- **Loss**: Training loss for DQN

## Troubleshooting

### Common Issues

1. **SUMO not found**: Ensure SUMO is installed and in PATH
2. **Port already in use**: Change server address in configuration
3. **Memory issues**: Reduce batch size or number of clients
4. **Training not converging**: Adjust learning rate or reward function

### Debug Mode

Enable debug output by setting environment variable:

```bash
export SUMO_DEBUG=1
python train_federated.py --mode single
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- SUMO (Simulation of Urban Mobility) team
- Flower (Federated Learning) framework
- TensorFlow/Keras for deep learning
- OpenAI Gym for RL environment interface

## Citation

If you use this code in your research, please cite:

```bibtex
@software{federated_traffic_control,
  title={Federated Learning Traffic Congestion Control with SUMO},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/federated-traffic-control}
}
```
