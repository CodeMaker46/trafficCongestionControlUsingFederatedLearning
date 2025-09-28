# Smart Traffic Control with AI

A simple yet powerful system that uses artificial intelligence to control traffic lights and reduce congestion. Think of it as teaching multiple intersections to work together to keep traffic flowing smoothly.

## What This Does

Instead of having traffic lights that just follow a fixed timer, this system uses **federated learning** to make them smarter. Each intersection learns from its own traffic patterns and shares knowledge with other intersections - but without sharing any personal data about drivers.

**Key Benefits:**
- 🚦 **Smarter Traffic Lights**: AI learns the best timing for each intersection
- 🔄 **Collaborative Learning**: Intersections share knowledge to improve city-wide traffic
- 🛡️ **Privacy Safe**: No personal data is shared between intersections
- 📊 **Real Results**: See actual improvements in waiting times and traffic flow

## How It Works (Simple Version)

1. **Each intersection** runs its own traffic simulation and AI agent
2. **The AI learns** the best timing by watching traffic patterns
3. **Intersections share** their learning (but not personal data) with a central server
4. **Everyone gets smarter** as the system learns from all intersections together

## What You Need

### Software Requirements
- **Python 3.8 or newer** (download from python.org)
- **SUMO Traffic Simulator** (download from sumo.dlr.de)

### Quick Setup
1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify SUMO is working:**
   ```bash
   sumo --version
   ```

3. **Run the setup script:**
   ```bash
   python setup.py
   ```

## Getting Started

### Quick single-client smoke test
```bash
python train_federated.py --mode single --sumo-config sumo_configs2/osm.sumocfg --gui --show-phase-console
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

Remove `--gui` to run headless. When GUI is enabled, you'll see signal colors and live phase switching.

## Understanding the Results

The system automatically saves results to the `results/` folder:
- **Training progress**: How well the AI is learning
- **Performance metrics**: Waiting times, queue lengths, traffic flow
- **Visual charts**: Easy-to-read graphs showing improvements

## Project Structure

```
traffic/
├── agents/                    # The AI brains
│   ├── dqn_agent.py          # Deep learning agent
│   └── traffic_environment.py # Traffic simulation interface
├── federated_learning/       # Collaboration system
│   ├── fl_client.py          # Individual intersection
│   └── fl_server.py          # Central coordinator
├── sumo_configs/             # Simple test intersection
├── sumo_configs2/            # Real city network
├── utils/visualization.py    # Results and charts
├── train_federated.py        # Main program
└── client.py                 # Individual intersection runner
```

## Technical Details (For Developers)

### AI Agent (DQN)
- **Architecture**: 3-layer neural network (128 → 128 → 64 neurons)
- **Learning**: Uses experience replay and target networks
- **Exploration**: Starts random, becomes smarter over time
- **Memory**: Remembers 10,000 traffic situations

### Traffic Simulation (SUMO)
- **State**: Monitors vehicle count, queue length, and waiting time
- **Actions**: Controls traffic light phases
- **Safety**: Minimum green times and collision prevention
- **Realism**: Based on real traffic patterns

### Federated Learning (Flower)
- **Privacy**: Only model weights are shared, never raw data
- **Aggregation**: Server combines learning from all intersections
- **Scalability**: Works with any number of intersections

## Troubleshooting

### Common Issues

**"SUMO not found" error:**
- Make sure SUMO is installed and added to your system PATH
- Test with: `sumo --version`

**"GUI not opening":**
- Try running without `--gui` first
- Make sure `sumo-gui` is installed

**"Python import errors":**
- Run: `pip install -r requirements.txt`
- Make sure you're using Python 3.8+

**"Server connection failed":**
- Check that the server is running first
- Make sure all clients use the same server address

### Getting Help

If you run into issues:
1. Check the error messages carefully
2. Make sure all requirements are installed
3. Try running the simple test first: `python train_federated.py --mode single`

## Examples and Use Cases

### For Researchers
- Study federated learning in real-world applications
- Experiment with different AI architectures
- Analyze traffic optimization algorithms

### For Students
- Learn about reinforcement learning
- Understand federated learning concepts
- Practice with traffic simulation

### For Cities
- Test traffic optimization strategies
- Evaluate AI-based traffic management
- Plan smart city infrastructure

## Contributing

We welcome contributions! Areas where help is especially appreciated:
- Adding new traffic scenarios
- Improving the AI algorithms
- Creating better visualizations
- Writing documentation

## License

This project is open source. Feel free to use it for research, education, or commercial applications.

## Acknowledgments

Thanks to:
- **SUMO team** for the excellent traffic simulator
- **Flower team** for federated learning framework
- **PyTorch team** for the deep learning tools

---

**Ready to make traffic smarter? Start with the quick test and watch AI learn to control traffic lights!**