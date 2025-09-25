#!/usr/bin/env python3
"""
Setup script for Federated Learning Traffic Control System
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✓ Python version: {sys.version}")
    return True

def check_sumo_installation():
    """Check if SUMO is installed and accessible"""
    try:
        result = subprocess.run(['sumo', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ SUMO is installed and accessible")
            return True
        else:
            print("✗ SUMO is not accessible")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("✗ SUMO is not installed or not in PATH")
        return False

def install_requirements():
    """Install Python requirements"""
    print("Installing Python requirements...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True)
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['results', 'logs', 'models']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def make_scripts_executable():
    """Make Python scripts executable"""
    scripts = ['train_federated.py', 'client.py']
    for script in scripts:
        if os.path.exists(script):
            os.chmod(script, 0o755)
            print(f"✓ Made {script} executable")

def verify_installation():
    """Verify the installation by running a simple test"""
    print("\nVerifying installation...")
    try:
        # Test imports
        from agents.dqn_agent import DQNAgent
        from agents.traffic_environment import SUMOTrafficEnvironment
        from federated_learning.fl_client import TrafficFLClient
        from federated_learning.fl_server import TrafficFLServer
        print("✓ All modules can be imported successfully")
        
        # Test DQN agent creation
        agent = DQNAgent(state_size=4, action_size=4)
        print("✓ DQN agent can be created")
        
        return True
    except Exception as e:
        print(f"✗ Installation verification failed: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nTo get started, run one of these commands:")
    print("\n1. Single client testing:")
    print("   python train_federated.py --mode single")
    print("\n2. Multi-client simulation:")
    print("   python train_federated.py --mode multi")
    print("\n3. Distributed federated learning:")
    print("   # Terminal 1 (Server):")
    print("   python train_federated.py --mode server")
    print("   # Terminal 2 (Client 1):")
    print("   python client.py --client-id client_1 --sumo-config sumo_configs2/osm.sumocfg")
    print("   # Terminal 3 (Client 2):")
    print("   python client.py --client-id client_2 --sumo-config sumo_configs2/osm.sumocfg")
    print("\n4. With SUMO GUI visualization:")
    print("   python train_federated.py --mode single --gui")
    print("\nFor more information, see README.md")
    print("="*60)

def main():
    """Main setup function"""
    print("Federated Learning Traffic Control System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check SUMO installation
    if not check_sumo_installation():
        print("\nSUMO installation required:")
        print("1. Download SUMO from: https://sumo.dlr.de/docs/Downloads.php")
        print("2. Add SUMO to your PATH environment variable")
        print("3. Run this setup script again")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Make scripts executable
    make_scripts_executable()
    
    # Verify installation
    if not verify_installation():
        print("\nInstallation verification failed. Please check the error messages above.")
        sys.exit(1)
    
    # Print usage instructions
    print_usage_instructions()

if __name__ == "__main__":
    main()
