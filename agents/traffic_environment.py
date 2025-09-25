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
    
    def __init__(self, sumo_config_path: str, gui: bool = False, tl_id: Optional[str] = None, show_phase_console: bool = False, show_gst_gui: bool = False):
        self.sumo_config_path = sumo_config_path
        self.gui = gui
        self.show_phase_console = show_phase_console
        self.show_gst_gui = show_gst_gui
        self.episode_count = 0
        self.step_count = 0
        self.max_steps = 1000
        self.min_phase_duration_s: float = 5.0
        self.yellow_duration_s: float = 3.0
        self.last_phase_switch_time: float = 0.0
        self._pending_target_phase: Optional[int] = None
        
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
        # Keep GST snapshots for analysis and saving
        self.gst_history: List[Dict] = []
        self._gst_poi_ids: Dict[str, str] = {}
        # Parameters for Green Signal Time (GST) computation
        # avg crossing time per vehicle class (seconds) and startup time per vehicle
        # The mapping keys follow SUMO classes grouped to paper categories
        self.class_avg_time_s: Dict[str, float] = {
            'car': 2.0,
            'truck': 3.5,
            'bus': 3.5,
            'motorcycle': 1.5,
            'bicycle': 2.0
        }
        self.class_startup_time_s: Dict[str, float] = {
            'car': 0.7,
            'truck': 1.2,
            'bus': 1.2,
            'motorcycle': 0.5,
            'bicycle': 0.6
        }
        
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
                # Map one representative lane for each incoming edge for GUI placement
                self._edge_to_lane: Dict[str, str] = {}
                for conn_group in links:
                    for conn in conn_group:
                        if conn and len(conn) >= 1:
                            lane_id = conn[0]
                            try:
                                edge_id = traci.lane.getEdgeID(lane_id)
                                if edge_id in self.incoming_edges and edge_id not in self._edge_to_lane:
                                    self._edge_to_lane[edge_id] = lane_id
                            except Exception:
                                pass

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
        self._setup_gst_gui_pois()
        
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

    def _map_vehicle_to_class(self, veh_id: str) -> str:
        """Map SUMO vehicle to simplified class for GST calculation."""
        try:
            vclass = traci.vehicle.getVehicleClass(veh_id)
        except Exception:
            try:
                type_id = traci.vehicle.getTypeID(veh_id)
                vclass = traci.vehicletype.getVehicleClass(type_id)
            except Exception:
                vclass = 'passenger'
        # Normalize to our buckets
        if vclass in ('passenger', 'emergency', 'taxi'):  # cars
            return 'car'
        if vclass in ('truck', 'delivery'):  # heavy
            return 'truck'
        if vclass in ('bus', 'coach'):  # buses counted as heavy
            return 'bus'
        if vclass in ('motorcycle',):
            return 'motorcycle'
        if vclass in ('bicycle',):
            return 'bicycle'
        return 'car'

    def _compute_green_signal_times(self) -> Dict:
        """Compute Green Signal Time (GST) per incoming edge based on the paper formula.

        For each incoming edge e, let count_c be number of vehicles of class c on e.
        Adjusted time per class c: adj_c = avg_time_c + startup_time_c.
        AllClassAverageTime for edge e is sum(count_c * adj_c) over classes.
        GST_e = AllClassAverageTime / (num_lanes + 1), where num_lanes is the
        number of incoming edges considered.
        """
        try:
            edges = self.incoming_edges[:4]
            num_lanes = max(1, len(edges))
            gst_per_edge: Dict[str, float] = {}
            for edge_id in edges:
                if not edge_id:
                    continue
                vehicle_ids = []
                try:
                    vehicle_ids = traci.edge.getLastStepVehicleIDs(edge_id)
                except Exception:
                    vehicle_ids = []
                # If edge-level query yields none (common with long edges), fall back to the
                # controlled upstream lane closest to the junction.
                if not vehicle_ids:
                    lane_id = getattr(self, '_edge_to_lane', {}).get(edge_id, None)
                    if lane_id:
                        try:
                            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                        except Exception:
                            vehicle_ids = []
                class_counts: Dict[str, int] = {}
                for vid in vehicle_ids:
                    cls = self._map_vehicle_to_class(vid)
                    class_counts[cls] = class_counts.get(cls, 0) + 1

                all_class_time = 0.0
                for cls, cnt in class_counts.items():
                    avg_t = self.class_avg_time_s.get(cls, 2.0)
                    start_t = self.class_startup_time_s.get(cls, 0.7)
                    all_class_time += float(cnt) * (avg_t + start_t)

                gst = all_class_time / (num_lanes + 1.0)
                gst_per_edge[edge_id] = float(gst)

            # Aggregate statistics
            avg_gst = float(np.mean(list(gst_per_edge.values()))) if gst_per_edge else 0.0
            snapshot = {
                'per_edge': gst_per_edge,
                'avg_gst': avg_gst,
                'num_lanes': num_lanes
            }
            # Record for history with current sim time
            try:
                snapshot_with_time = dict(snapshot)
                snapshot_with_time['time'] = float(traci.simulation.getTime())
                self.gst_history.append(snapshot_with_time)
                # Keep history bounded
                if len(self.gst_history) > 2000:
                    self.gst_history = self.gst_history[-2000:]
            except Exception:
                pass
            return snapshot
        except Exception as e:
            print(f"Error computing GST: {e}")
            return {'per_edge': {}, 'avg_gst': 0.0, 'num_lanes': 0}

    def _setup_gst_gui_pois(self):
        """Create POIs in the GUI to display GST values near incoming lanes."""
        if not (self.gui and self.show_gst_gui and self.tl_id and self.incoming_edges):
            return
        try:
            for edge_id in self.incoming_edges:
                lane_id = getattr(self, '_edge_to_lane', {}).get(edge_id, None)
                if not lane_id:
                    continue
                try:
                    shape = traci.lane.getShape(lane_id)
                    # Use the first point on the lane as placement
                    x, y = shape[0]
                except Exception:
                    # Fallback: junction center
                    pos = traci.junction.getPosition(self.tl_id)
                    x, y = pos[0], pos[1]
                poi_id = f"gst_{edge_id}"
                # Remove if exists
                try:
                    traci.poi.remove(poi_id)
                except Exception:
                    pass
                try:
                    traci.poi.add(poi_id, x, y, (0, 255, 0, 255), name=f"GST {edge_id}: --s")
                except Exception:
                    # Some SUMO versions require fewer args
                    traci.poi.add(poi_id, x, y)
                self._gst_poi_ids[edge_id] = poi_id
        except Exception:
            pass

    def _update_gst_gui_labels(self, gst_snapshot: Dict):
        if not (self.gui and self.show_gst_gui):
            return
        try:
            per_edge = gst_snapshot.get('per_edge', {})
            for edge_id, val in per_edge.items():
                poi_id = self._gst_poi_ids.get(edge_id)
                if not poi_id:
                    continue
                label = f"GST {edge_id[-6:]}: {float(val):.1f}s"
                try:
                    traci.poi.setParameter(poi_id, "name", label)
                except Exception:
                    # If parameter not supported, try color intensity as cue
                    v = float(val)
                    r = int(max(0, min(255, 40 + 10 * v)))
                    g = int(max(0, 255 - 5 * v))
                    traci.poi.setColor(poi_id, (r, g, 0, 255))
        except Exception:
            pass
    
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
            # Compute GST and optionally print/overlay each step
            gst_snapshot = self._compute_green_signal_times()
            if self.show_phase_console:
                try:
                    per_edge = gst_snapshot.get('per_edge', {})
                    items = [{ 'lane': e, 'gst': round(float(v), 2) } for e, v in per_edge.items()]
                    if items:
                        print(f"[GST] {items} avg={gst_snapshot.get('avg_gst', 0.0):.2f}s")
                except Exception:
                    pass
            self._update_gst_gui_labels(gst_snapshot)
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
            # Attach GST snapshot for this step
            info['gst'] = gst_snapshot

            # Adaptive green extension to reduce long queues/waits
            try:
                if phase_info:
                    phase_idx = int(phase_info.get('phase', -1))
                    remaining = float(phase_info.get('remaining_s', 0.0))
                    # Only extend during main green phases (0 and 2)
                    if phase_idx in (0, 2) and remaining <= 0.5:
                        # Use queue length across edges and GST average to compute extension
                        queue_sum = 0
                        for e in self.incoming_edges:
                            queue_sum += self._get_queue_length(e)
                        # Base on average GST plus small factor of queue size
                        base = float(gst_snapshot.get('avg_gst', 0.0))
                        # More aggressive extension for large queues
                        extension = base + 0.5 * queue_sum
                        extension = max(0.0, min(20.0, extension))
                        if extension > 0.5:
                            try:
                                traci.trafficlight.setPhaseDuration(self.tl_id, remaining + extension)
                            except Exception:
                                pass
            except Exception:
                pass
            
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
            # Safe switching with yellow insertion when crossing directions
            def is_green(p: int) -> bool:
                return p in (0, 2)

            def yellow_for_transition(src: int, dst: int) -> Optional[int]:
                if src == 0 and dst == 2:
                    return 1  # EW -> yellow
                if src == 2 and dst == 0:
                    return 3  # NS -> yellow
                # If transitioning between same corridor or to yellow itself, none
                return None

            # If a pending target exists, try to complete it once yellow finished
            if self._pending_target_phase is not None:
                remaining = 0.0
                try:
                    next_switch = traci.trafficlight.getNextSwitch(self.tl_id)
                    remaining = max(0.0, next_switch - sim_time)
                except Exception:
                    pass
                if remaining <= 0.1:
                    traci.trafficlight.setPhase(self.tl_id, self._pending_target_phase)
                    try:
                        traci.trafficlight.setPhaseDuration(self.tl_id, self.min_phase_duration_s)
                    except Exception:
                        pass
                    self.last_phase_switch_time = sim_time
                    self._pending_target_phase = None
                return

            # Enforce a minimum green time before initiating a new switch
            if desired != current and (sim_time - self.last_phase_switch_time) >= self.min_phase_duration_s:
                if is_green(current) and is_green(desired) and current != desired:
                    y = yellow_for_transition(current, desired)
                    if y is not None:
                        traci.trafficlight.setPhase(self.tl_id, y)
                        try:
                            traci.trafficlight.setPhaseDuration(self.tl_id, self.yellow_duration_s)
                        except Exception:
                            pass
                        self.last_phase_switch_time = sim_time
                        self._pending_target_phase = desired
                        return
                # Direct set when not crossing conflicting greens
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
            
            # Reward function: minimize waiting time, queue length, and GST
            # Stronger penalties to push the policy to reduce queues quickly
            waiting_penalty = -waits / 100.0
            queue_penalty = -queues / 6.0
            try:
                gst_avg = float(self.gst_history[-1]['avg_gst']) if self.gst_history else 0.0
            except Exception:
                gst_avg = 0.0
            gst_penalty = -gst_avg / 6.0
            
            # Small positive reward for keeping traffic flowing
            flow_reward = 1.0 if queues < 5 else 0.0
            
            total_reward = waiting_penalty + queue_penalty + gst_penalty + flow_reward
            
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
        metrics = {
            'total_waiting_time': self.total_waiting_time,
            'total_vehicles': self.total_vehicles,
            'average_queue_length': np.mean(self.queue_lengths) if self.queue_lengths else 0,
            'max_queue_length': max(self.queue_lengths) if self.queue_lengths else 0,
            'steps': self.step_count
        }
        # Attach GST aggregates
        last_gst = self._compute_green_signal_times()
        metrics['green_signal_time'] = last_gst
        try:
            # Compute per-edge average over history
            per_edge_sums: Dict[str, float] = {}
            per_edge_counts: Dict[str, int] = {}
            for snap in self.gst_history:
                for e, val in snap.get('per_edge', {}).items():
                    per_edge_sums[e] = per_edge_sums.get(e, 0.0) + float(val)
                    per_edge_counts[e] = per_edge_counts.get(e, 0) + 1
            avg_per_edge = {e: (per_edge_sums[e] / max(1, per_edge_counts[e])) for e in per_edge_sums}
            metrics['green_signal_time_avg_per_edge'] = avg_per_edge
            metrics['green_signal_time_history'] = self.gst_history[-200:]
        except Exception:
            pass
        return metrics
    
    def close(self):
        """Close the environment"""
        try:
            self.stop_simulation()
        except:
            pass
