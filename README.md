# Federated RL Traffic Control with SUMO and Flower

This project trains adaptive traffic light controllers using federated reinforcement learning over SUMO simulations. Each intersection (client) learns locally with a DQN and shares only model weights/metrics with a central server (Flower FedAvg). No raw traffic data leaves clients.

## Key features

- **Federated Learning (Flower)**: FedAvg across multiple clients with configurable round settings
- **SUMO integration (TraCI)**: Real-time control of traffic lights; supports GUI and headless modes
- **DQN (PyTorch)**: 3 hidden layers (128, 128, 64), replay buffer (10k), epsilon-greedy, target updates every 100 steps
- **General network support**: Auto-detects first traffic light and incoming edges from any `.sumocfg` (OSM-ready)
- **Robust simulation**: Minimum green time, collision teleport, optional GUI view settings for signal colors
- **Metrics and persistence**: Training/performance metrics saved in `results/`

## Project structure

```
traffic/
├── agents/
│   ├── dqn_agent.py              # PyTorch DQN agent
│   └── traffic_environment.py    # SUMO wrapper (state/action/reward, TLS control)
├── federated_learning/
│   ├── fl_client.py              # Flower NumPyClient wrapper
│   └── fl_server.py              # Flower FedAvg server utilities
├── sumo_configs/                 # Example simple intersection config
├── sumo_configs2/                # OSM-based city network (default)
├── utils/visualization.py        # Result plotting helpers
├── train_federated.py            # Main entry (server, client, single, multi)
├── client.py                     # Standalone client launcher
├── requirements.txt              # Pinned deps (TF 2.15 + Torch + Flower 1.4)
└── README.md
```

## Requirements and installation

- Python 3.8+
- SUMO 1.19+ (CLI `sumo` and GUI `sumo-gui` must be in PATH)

Install Python dependencies (Windows PowerShell):
```bash
python -m pip install -r requirements.txt
```

Notes on versions:
- TensorFlow 2.15.0 and Flower 1.4.0 require `protobuf<4`; this is pinned.
- PyTorch (CPU) is installed from PyPI automatically.

Verify SUMO:
```bash
sumo --version
sumo-gui --version
```

## How it works (flow)

1) Server starts FedAvg and waits for clients.
2) Each client runs a local SUMO simulation and a DQN agent:
   - State (12 dims): for up to 4 incoming edges × [vehicle_count, queue_length, waiting_time], normalized
   - Action: integer mapped to a traffic light phase index (wraps modulo available phases)
   - Reward: penalize waiting and queues, small bonus for flow
   - Safety: minimum green time (default 5s) and collision teleport to avoid pile-ups
3) After local episodes, clients send model weights and metrics to server.
4) Server aggregates weights (weighted by total steps) and broadcasts the global model.

## Running the system

Default SUMO config: `sumo_configs2/osm.sumocfg` (OSM network). GUI loads `osm.view.xml` when present.

### Quick single-client smoke test
```bash
python train_federated.py --mode single --sumo-config sumo_configs2/osm.sumocfg
```

### Federated training (server + clients)

- Server (1 terminal):
```bash
python train_federated.py --mode server --num-rounds 10 --min-clients 2 --server-address localhost:8080
```

- Clients (N terminals):
```bash
python client.py --client-id client_1 --sumo-config sumo_configs2/osm.sumocfg --server-address localhost:8080 --gui
python client.py --client-id client_2 --sumo-config sumo_configs2/osm.sumocfg --server-address localhost:8080 --gui
```

Remove `--gui` to run headless. When GUI is enabled, you’ll see signal colors and live phase switching.

## DQN agent details (`agents/dqn_agent.py`)

- Architecture: Linear(12→128) → ReLU → 128 → ReLU → 64 → ReLU → Linear(→4)
- Replay buffer: 10,000 transitions, batch 32, Adam lr=1e-3
- Exploration: start 1.0, min 0.01, decay 0.995
- Target network: synchronize every 100 gradient steps
- Save/Load: `torch.save`/`torch.load`; weights exposed as NumPy arrays for Flower

## SUMO environment (`agents/traffic_environment.py`)

- Auto-discovers first TLS id and its incoming edges via TraCI
- State: 4 edges × [veh_count, queue_len, waiting_time] (padded if fewer edges)
- Action application: set phase by index; enforce minimum green time (default 5s)
- Reward: `-waiting/200 - queue/10 + flow_bonus`
- Stability: `--collision.action teleport`, `--time-to-teleport 60`, `--step-length 1.0`
- GUI: if `--gui` is used, attempts to load `osm.view.xml` to show TLS indicators

## Server (`federated_learning/fl_server.py`)

- Flower 1.4 FedAvg with:
  - `min_available_clients`, `min_fit_clients`, `min_evaluate_clients`
  - weighted aggregation by client `total_steps`
  - configurable fit/eval configs per round

## Results and visualization

Artifacts saved to `results/` by training scripts (client histories, performance metrics, optional server metrics). Plot helpers are in `utils/visualization.py`.

## Troubleshooting

- SUMO GUI not opening: ensure `sumo-gui` is on PATH (`sumo-gui --version`). Use `--gui` flag on client command.
- Flower server keyword error: Flower 1.4 uses `min_evaluate_clients` (not `min_eval_clients`). Already fixed in this repo.
- Windows PowerShell tips: use separate commands (no `&&`). Inline Python via `python -c "..."`.
- Protobuf conflicts: this repo pins TF 2.15 + `protobuf<4` + Flower 1.4. Do not upgrade TF beyond 2.15 unless you also upgrade Flower and adjust protobuf.


## Acknowledgments

- SUMO team
- Flower (flwr)
- PyTorch