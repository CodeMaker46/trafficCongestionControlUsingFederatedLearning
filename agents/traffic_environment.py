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
        
        # Traffic light ID (detected dynamically after SUMO starts)
        self.tl_id = None
        # Incoming edges controlled by the selected traffic light (derived dynamically)
        self.incoming_edges: List[str] = []
        
        # Performance metrics
        self.total_waiting_time = 0
        self.total_vehicles = 0
        self.queue_lengths = []
        
    def start_simulation(self):
        """Start SUMO simulation"""
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        sumo_cmd = [sumo_binary, "-c", self.sumo_config_path, "--no-step-log", "true"]
        
        traci.start(sumo_cmd)
        # Initialize dynamic network entities (traffic light and edges)
        self._init_network_entities()
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
            # Dynamically split incoming edges into two groups to approximate EW/NS
            if not self.incoming_edges:
                all_edges = list(traci.edge.getIDList())
                self.incoming_edges = [e for e in all_edges if not e.startswith(":")][:4]
            group_a = self.incoming_edges[0::2]
            group_b = self.incoming_edges[1::2]

            # Get queue lengths for each group
            ew_queue = sum(self._get_queue_length(e) for e in group_a)
            ns_queue = sum(self._get_queue_length(e) for e in group_b)
            
            # Get waiting time for each group
            ew_waiting = sum(self._get_waiting_time(e) for e in group_a)
            ns_waiting = sum(self._get_waiting_time(e) for e in group_b)
            
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
            if self.tl_id is None:
                return
            # Use numeric phase indices compatible with the current program
            num_phases = traci.trafficlight.getPhaseNumber(self.tl_id)
            if num_phases and num_phases > 0:
                traci.trafficlight.setPhase(self.tl_id, phase % num_phases)
        except Exception as e:
            print(f"Error setting traffic light phase: {e}")
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on traffic performance"""
        try:
            # Use the same dynamic grouping as in get_state
            if not self.incoming_edges:
                all_edges = list(traci.edge.getIDList())
                self.incoming_edges = [e for e in all_edges if not e.startswith(":")][:4]
            group_a = self.incoming_edges[0::2]
            group_b = self.incoming_edges[1::2]

            # Current waiting times
            ew_waiting = sum(self._get_waiting_time(e) for e in group_a)
            ns_waiting = sum(self._get_waiting_time(e) for e in group_b)
            
            # Current queue lengths
            ew_queue = sum(self._get_queue_length(e) for e in group_a)
            ns_queue = sum(self._get_queue_length(e) for e in group_b)
            
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

    def _init_network_entities(self) -> None:
        """Detect traffic light and incoming edges dynamically from the loaded network."""
        try:
            # Detect a traffic light to control (pick the first if available)
            tls = list(traci.trafficlight.getIDList())
            self.tl_id = tls[0] if tls else None
            self.incoming_edges = []
            if self.tl_id is not None:
                # Derive incoming edges from controlled lanes
                lanes = traci.trafficlight.getControlledLanes(self.tl_id)
                edge_ids: List[str] = []
                for lane in lanes:
                    # lane format: edgeId_laneIndex (e.g., "12345_0")
                    edge_id = lane.split('_')[0]
                    if not edge_id.startswith(':') and edge_id not in edge_ids:
                        edge_ids.append(edge_id)
                # Keep a small subset for state calculation stability
                self.incoming_edges = edge_ids[:6] if edge_ids else []
        except Exception as e:
            print(f"Warning: failed to initialize network entities: {e}")
    
    def close(self):
        """Close the environment"""
        try:
            self.stop_simulation()
        except:
            pass
