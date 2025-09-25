import traci
import numpy as np
import random
import os
import sys
from typing import Tuple, List, Dict, Optional
import time

class SUMOTrafficEnvironment:
    """
    Traffic environment that interfaces with SUMO for realistic traffic simulation
    """
    
    def __init__(self, sumo_config_path: str, gui: bool = False, tl_id: Optional[str] = None, show_phase_console: bool = False):
        self.sumo_config_path = sumo_config_path
        self.gui = gui
        self.show_phase_console = show_phase_console
        self.episode_count = 0
        self.step_count = 0
        self.max_steps = 1000
        self.min_phase_duration_s: float = 5.0
        self.last_phase_switch_time: float = 0.0
        
        # Traffic light phases: 0=EW_Green, 1=EW_Yellow, 2=NS_Green, 3=NS_Yellow
        self.phases = {
            0: "GGrrGGrr",  # East-West Green
            1: "yyrryyrr",  # East-West Yellow
            2: "rrGGrrGG",  # North-South Green
            3: "rryyrryy"   # North-South Yellow
        }
        
        # State space: 4 incoming edges × (vehicle_count, queue_length, waiting_time)
        self.state_size = 12
        self.action_size = 4  # actions mapped to phase indices modulo phase count
        
        # Traffic light ID and discovered topology
        self.tl_id: Optional[str] = tl_id
        self.incoming_edges: List[str] = []
        self.num_phases: int = 1
        
        # Performance metrics
        self.total_waiting_time = 0
        self.total_vehicles = 0
        self.queue_lengths = []
        
    def start_simulation(self):
        """Start SUMO simulation"""
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", self.sumo_config_path,
            "--no-step-log", "true",
            "--collision.action", "teleport",
            "--time-to-teleport", "60",
            "--step-length", "1.0"
        ]
        # Attach GUI view settings if available (helps show TLS indicators/colors)
        try:
            cfg_dir = os.path.dirname(self.sumo_config_path)
            view_file = os.path.join(cfg_dir, "osm.view.xml")
            if self.gui and os.path.isfile(view_file):
                sumo_cmd += ["--gui-settings-file", view_file]
        except Exception:
            pass
        
        traci.start(sumo_cmd)

        # Discover traffic light and incoming edges if not provided
        try:
            if self.tl_id is None:
                tls_ids = traci.trafficlight.getIDList()
                self.tl_id = tls_ids[0] if tls_ids else None
            if self.tl_id is not None:
                # Determine incoming edges from controlled links (unique upstream lane edges)
                links = traci.trafficlight.getControlledLinks(self.tl_id)
                lane_ids = []
                for conn_group in links:
                    for conn in conn_group:
                        if conn and len(conn) >= 1:
                            lane_ids.append(conn[0])
                edge_ids = []
                for lane_id in lane_ids:
                    try:
                        edge_ids.append(traci.lane.getEdgeID(lane_id))
                    except Exception:
                        pass
                # Keep unique incoming edges in stable order
                seen = set()
                ordered_unique = []
                for e in edge_ids:
                    if e not in seen and e != '':
                        seen.add(e)
                        ordered_unique.append(e)
                self.incoming_edges = ordered_unique[:4]

                # Discover number of phases from program logic
                try:
                    logics = traci.trafficlight.getAllProgramLogics(self.tl_id)
                    if logics:
                        logic = logics[0]
                        phases = getattr(logic, 'phases', None)
                        if phases is None:
                            # Fallback for older API name
                            phases = getattr(logic, 'getPhases', lambda: [])()
                        self.num_phases = max(1, len(phases))
                    else:
                        self.num_phases = 1
                except Exception:
                    self.num_phases = 1
        except Exception as e:
            print(f"Topology discovery failed: {e}")
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
            metrics: List[float] = []
            edges = self.incoming_edges[:4]
            # pad to 4 edges if fewer discovered
            while len(edges) < 4:
                edges.append("")
            for edge_id in edges:
                veh_count = self._get_vehicle_count(edge_id) if edge_id else 0
                queue_len = self._get_queue_length(edge_id) if edge_id else 0
                wait_time = self._get_waiting_time(edge_id) if edge_id else 0.0
                metrics.extend([
                    min(veh_count / 20.0, 1.0),
                    min(queue_len / 20.0, 1.0),
                    min(wait_time / 200.0, 1.0)
                ])
            return np.array(metrics, dtype=float)
        except Exception as e:
            print(f"Error getting state: {e}")
            return np.zeros(self.state_size)
    
    def _get_vehicle_count(self, edge_id: str) -> int:
        """Get total vehicles currently on an edge"""
        try:
            return int(traci.edge.getLastStepVehicleNumber(edge_id))
        except:
            return 0

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
            phase_info = self.get_current_phase_info()
            if self.show_phase_console and phase_info:
                print(f"[TLS {self.tl_id}] phase={phase_info.get('phase')} remaining={phase_info.get('remaining_s'):.1f}s/{phase_info.get('duration_s'):.1f}s")

            info = {
                'step': self.step_count,
                'total_waiting_time': self.total_waiting_time,
                'total_vehicles': self.total_vehicles,
                'queue_lengths': self.queue_lengths.copy()
            }
            if phase_info:
                info['phase'] = phase_info
            
            return next_state, reward, done, info
            
        except Exception as e:
            print(f"Error in step: {e}")
            return self.get_state(), -10.0, True, {}
    
    def _set_traffic_light_phase(self, phase: int):
        """Set traffic light to specified phase"""
        try:
            if self.tl_id is None:
                return
            # Use phase indices modulo available phases for generality
            desired = int(phase) % max(1, self.num_phases)
            sim_time = traci.simulation.getTime()
            try:
                current = traci.trafficlight.getPhase(self.tl_id)
            except Exception:
                current = -1
            # Enforce a minimum green time to reduce sudden stops/collisions
            if desired != current:
                if sim_time - self.last_phase_switch_time >= self.min_phase_duration_s:
                    traci.trafficlight.setPhase(self.tl_id, desired)
                    try:
                        traci.trafficlight.setPhaseDuration(self.tl_id, self.min_phase_duration_s)
                    except Exception:
                        pass
                    self.last_phase_switch_time = sim_time
        except Exception as e:
            print(f"Error setting traffic light phase: {e}")
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on traffic performance"""
        try:
            # Aggregate metrics across discovered incoming edges
            waits = 0.0
            queues = 0
            vehs = 0
            for e in self.incoming_edges:
                waits += self._get_waiting_time(e)
                q = self._get_queue_length(e)
                queues += q
                vehs += self._get_vehicle_count(e)

            # Update metrics
            self.total_waiting_time += waits
            self.total_vehicles += vehs
            self.queue_lengths.append(queues)
            
            # Reward function: minimize waiting time and queue length
            # Negative reward for high waiting times and queue lengths
            waiting_penalty = -waits / 200.0
            queue_penalty = -queues / 10.0
            
            # Small positive reward for keeping traffic flowing
            flow_reward = 1.0 if queues < 5 else 0.0
            
            total_reward = waiting_penalty + queue_penalty + flow_reward
            
            return total_reward
            
        except Exception as e:
            print(f"Error calculating reward: {e}")
            return -1.0

    def get_current_phase_info(self) -> Dict:
        """Return current phase index and remaining/total duration in seconds if available."""
        try:
            if self.tl_id is None:
                return {}
            phase_index = traci.trafficlight.getPhase(self.tl_id)
            # Estimate remaining time until next switch
            next_switch = traci.trafficlight.getNextSwitch(self.tl_id)
            sim_time = traci.simulation.getTime()
            remaining = max(0.0, next_switch - sim_time)
            # Try to fetch total duration from program logic
            duration = 0.0
            try:
                logics = traci.trafficlight.getAllProgramLogics(self.tl_id)
                if logics:
                    phases = getattr(logics[0], 'phases', None)
                    if phases is None:
                        phases = getattr(logics[0], 'getPhases', lambda: [])()
                    if phases and 0 <= phase_index < len(phases):
                        duration = float(getattr(phases[phase_index], 'duration', 0.0))
            except Exception:
                pass
            return {
                'phase': int(phase_index),
                'remaining_s': float(remaining),
                'duration_s': float(duration),
                'num_phases': int(self.num_phases)
            }
        except Exception:
            return {}
    
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
