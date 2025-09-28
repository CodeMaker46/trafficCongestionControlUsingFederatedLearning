
import os
import sys
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    print("Testing imports...")
    try:
        from agents.dqn_agent import DQNAgent
        from agents.traffic_environment import SUMOTrafficEnvironment
        from federated_learning.fl_client import TrafficFLClient
        from federated_learning.fl_server import TrafficFLServer
        from utils.visualization import TrafficVisualizer
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_dqn_agent():
    print("Testing DQN agent...")
    try:
        from agents.dqn_agent import DQNAgent
        
        agent = DQNAgent(state_size=4, action_size=4)
        
        state = np.random.random(4)
        action = agent.act(state)
        assert 0 <= action < 4, "Invalid action"
        
        next_state = np.random.random(4)
        agent.remember(state, action, 1.0, next_state, False)
        
        q_values = agent.get_q_values(state)
        assert len(q_values) == 4, "Invalid Q-values length"
        
        print("✓ DQN agent test passed")
        return True
    except Exception as e:
        print(f"✗ DQN agent test failed: {e}")
        return False

def test_traffic_environment():
    print("Testing traffic environment...")
    try:
        from agents.traffic_environment import SUMOTrafficEnvironment
        
        # Create environment
        env = SUMOTrafficEnvironment("sumo_configs2/osm.sumocfg", gui=False)
        
        assert env.state_size == 4, "Invalid state size"
        assert env.action_size == 4, "Invalid action size"
        
        print("✓ Traffic environment test passed")
        return True
    except Exception as e:
        print(f"✗ Traffic environment test failed: {e}")
        return False

def test_federated_learning():
    print("Testing federated learning components...")
    try:
        from federated_learning.fl_client import TrafficFLClient
        from federated_learning.fl_server import TrafficFLServer
        
        server = TrafficFLServer(num_rounds=1, min_clients=1)
        assert server.num_rounds == 1, "Invalid server configuration"
        
        client = TrafficFLClient(
            client_id="test_client",
            sumo_config_path="sumo_configs2/osm.sumocfg",
            gui=False
        )
        assert client.client_id == "test_client", "Invalid client ID"
        
        print("✓ Federated learning test passed")
        return True
    except Exception as e:
        print(f"✗ Federated learning test failed: {e}")
        return False

def test_visualization():
    print("Testing visualization utilities...")
    try:
        from utils.visualization import TrafficVisualizer
        
        # Create visualizer
        viz = TrafficVisualizer("results")
        
        dummy_metrics = [
            {'round': 0, 'type': 'fit', 'metrics': {'avg_average_reward': 0.5}},
            {'round': 1, 'type': 'fit', 'metrics': {'avg_average_reward': 0.6}},
            {'round': 0, 'type': 'evaluate', 'metrics': {'avg_waiting_time': 10.0}},
            {'round': 1, 'type': 'evaluate', 'metrics': {'avg_waiting_time': 8.0}}
        ]
        
        report = viz.create_summary_report(dummy_metrics, {})
        assert 'summary' in report, "Invalid report structure"
        
        print("✓ Visualization test passed")
        return True
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False

def test_sumo_configs():
    print("Testing SUMO configuration files...")
    try:
        config_files = [
            "sumo_configs2/osm.sumocfg"
        ]
        
        for config_file in config_files:
            assert os.path.exists(config_file), f"Missing config file: {config_file}"
        
        print("✓ SUMO configuration files exist")
        return True
    except Exception as e:
        print(f"✗ SUMO configuration test failed: {e}")
        return False

def main():
    print("Federated Learning Traffic Control System - Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_dqn_agent,
        test_traffic_environment,
        test_federated_learning,
        test_visualization,
        test_sumo_configs
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! System is ready to use.")
        return True
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
