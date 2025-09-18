import traci
import numpy as np
import random
import os
import sys
from typing import Tuple, List, Dict
import time

class SUMOTrafficEnvironment:
    """
    Traffic environment that interfaces with SUMO for realistic traffic simulation
    """
    
    def __init__(self, sumo_config_path: str, gui: bool = False):
        self.sumo_config_path = sumo_config_path
        self.gui = gui
        self.episode_count = 0
        self.step_count = 0
        self.max_steps = 1000
        
        # Traffic light phases: 0=EW_Green, 1=EW_Yellow, 2=NS_Green, 3=NS_Yellow
        self.phases = {
            0: "GGrrGGrr",  # East-West Green
            1: "yyrryyrr",  # East-West Yellow
            2: "rrGGrrGG",  # North-South Green
            3: "rryyrryy"   # North-South Yellow
        }
        
        # State space: [queue_length_ew, queue_length_ns, waiting_time_ew, waiting_time_ns]
        self.state_size = 4
        self.action_size = 4  # 4 different phase actions
        
        # Traffic light ID (assuming single intersection)
        self.tl_id = "junction2"
        
        # Performance metrics
        self.total_waiting_time = 0
        self.total_vehicles = 0
        self.queue_lengths = []
        
    def start_simulation(self):
        """Start SUMO simulation"""
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        sumo_cmd = [sumo_binary, "-c", self.sumo_config_path, "--no-step-log", "true"]
        
        traci.start(sumo_cmd)
        self.episode_count += 1
        self.step_count = 0
        self.total_waiting_time = 0
        self.total_vehicles = 0
        self.queue_lengths = []
        
    def stop_simulation(self):
        """Stop SUMO simulation"""
        traci.close()
        
    def reset(self) -> np.ndarray:
        """Reset environment and return initial state"""
        if self.episode_count > 0:
            self.stop_simulation()
            time.sleep(1)  # Brief pause between episodes
            
        self.start_simulation()
        return self.get_state()
        
    def get_state(self) -> np.ndarray:
        """Get current state of the traffic environment"""
        try:
            # Get queue lengths for each direction
            ew_queue = self._get_queue_length("edge1") + self._get_queue_length("edge2")
            ns_queue = self._get_queue_length("edge3") + self._get_queue_length("edge4")
            
            # Get average waiting time for each direction
            ew_waiting = self._get_waiting_time("edge1") + self._get_waiting_time("edge2")
            ns_waiting = self._get_waiting_time("edge3") + self._get_waiting_time("edge4")
            
            # Normalize state values
            state = np.array([
                min(ew_queue / 20.0, 1.0),      # Normalize queue length
                min(ns_queue / 20.0, 1.0),
                min(ew_waiting / 100.0, 1.0),   # Normalize waiting time
                min(ns_waiting / 100.0, 1.0)
            ])
            
            return state
            
        except Exception as e:
            print(f"Error getting state: {e}")
            return np.zeros(self.state_size)
    
    def _get_queue_length(self, edge_id: str) -> int:
        """Get queue length for a specific edge"""
        try:
            vehicles = traci.edge.getLastStepVehicleIDs(edge_id)
            queue_length = 0
            for veh_id in vehicles:
                if traci.vehicle.getSpeed(veh_id) < 0.1:  # Vehicle is stopped
                    queue_length += 1
            return queue_length
        except:
            return 0
    
    def _get_waiting_time(self, edge_id: str) -> float:
        """Get total waiting time for vehicles on an edge"""
        try:
            return traci.edge.getWaitingTime(edge_id)
        except:
            return 0.0
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state, reward, done, info"""
        try:
            # Apply traffic light phase
            self._set_traffic_light_phase(action)
            
            # Advance simulation by one step
            traci.simulationStep()
            self.step_count += 1
            
            # Get new state
            next_state = self.get_state()
            
            # Calculate reward
            reward = self._calculate_reward()
            
            # Check if episode is done
            done = self.step_count >= self.max_steps or traci.simulation.getMinExpectedNumber() == 0
            
            # Collect info
            info = {
                'step': self.step_count,
                'total_waiting_time': self.total_waiting_time,
                'total_vehicles': self.total_vehicles,
                'queue_lengths': self.queue_lengths.copy()
            }
            
            return next_state, reward, done, info
            
        except Exception as e:
            print(f"Error in step: {e}")
            return self.get_state(), -10.0, True, {}
    
    def _set_traffic_light_phase(self, phase: int):
        """Set traffic light to specified phase"""
        try:
            if phase in self.phases:
                traci.trafficlight.setRedYellowGreenState(self.tl_id, self.phases[phase])
        except Exception as e:
            print(f"Error setting traffic light phase: {e}")
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on traffic performance"""
        try:
            # Get current waiting times
            ew_waiting = self._get_waiting_time("edge1") + self._get_waiting_time("edge2")
            ns_waiting = self._get_waiting_time("edge3") + self._get_waiting_time("edge4")
            
            # Get current queue lengths
            ew_queue = self._get_queue_length("edge1") + self._get_queue_length("edge2")
            ns_queue = self._get_queue_length("edge3") + self._get_queue_length("edge4")
            
            # Update metrics
            self.total_waiting_time += ew_waiting + ns_waiting
            self.total_vehicles += ew_queue + ns_queue
            self.queue_lengths.append(ew_queue + ns_queue)
            
            # Reward function: minimize waiting time and queue length
            # Negative reward for high waiting times and queue lengths
            waiting_penalty = -(ew_waiting + ns_waiting) / 100.0
            queue_penalty = -(ew_queue + ns_queue) / 10.0
            
            # Small positive reward for keeping traffic flowing
            flow_reward = 1.0 if (ew_queue + ns_queue) < 5 else 0.0
            
            total_reward = waiting_penalty + queue_penalty + flow_reward
            
            return total_reward
            
        except Exception as e:
            print(f"Error calculating reward: {e}")
            return -1.0
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for the current episode"""
        return {
            'total_waiting_time': self.total_waiting_time,
            'total_vehicles': self.total_vehicles,
            'average_queue_length': np.mean(self.queue_lengths) if self.queue_lengths else 0,
            'max_queue_length': max(self.queue_lengths) if self.queue_lengths else 0,
            'steps': self.step_count
        }
    
    def close(self):
        """Close the environment"""
        try:
            self.stop_simulation()
        except:
            pass
