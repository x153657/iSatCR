from SatelliteNetworkSimulator_Computing import SatelliteNetworkSimulator_OnbardComputing
from SatelliteNetworkSimulation import SatelliteSimulation
from Make_Satellite_Graph import SatelliteTracker,SatelliteGraph
from skyfield.api import load
from Read_Ground_Imformation import extract_landmarks, get_connections_h3
from SatelliteNetworkSimulator_Beta import Logger
from Draw_Graph_Quiker import SatelliteVisualizer_geo
import hashlib
import json
import random
import copy
import os
import csv
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point


def load_computing_abilities(tle_filepath, satellite_names, fallback_ability=None):
    """Load per-satellite computing abilities from the TLE's companion CSV."""
    ability_filepath = f"{os.path.splitext(tle_filepath)[0]}_computing_ability.csv"
    satellite_names = set(satellite_names)

    if not os.path.exists(ability_filepath):
        if fallback_ability is None:
            raise FileNotFoundError(
                f"Computing ability file not found for {tle_filepath}: {ability_filepath}"
            )
        return {name: float(fallback_ability) for name in satellite_names}, None

    abilities = {}
    with open(ability_filepath, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required_columns = {"satellite_name", "computing_ability_gflops"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{ability_filepath} must contain columns: {sorted(required_columns)}"
            )

        for row_number, row in enumerate(reader, start=2):
            name = (row.get("satellite_name") or "").strip()
            if not name:
                raise ValueError(f"Missing satellite_name at {ability_filepath}:{row_number}")
            if name in abilities:
                raise ValueError(f"Duplicate satellite_name {name!r} in {ability_filepath}")
            try:
                ability_gflops = float(row["computing_ability_gflops"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid computing ability for {name!r} at {ability_filepath}:{row_number}"
                ) from exc
            if ability_gflops <= 0:
                raise ValueError(
                    f"Computing ability must be positive for {name!r} at "
                    f"{ability_filepath}:{row_number}"
                )
            abilities[name] = ability_gflops * 1e9

    missing = sorted(satellite_names - set(abilities))
    extra = sorted(set(abilities) - satellite_names)
    if missing or extra:
        raise ValueError(
            f"Computing ability entries do not match {tle_filepath}; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    return abilities, ability_filepath


class SatelliteEnv(SatelliteSimulation):
    def __init__(self, mode, select_mode, q_net, gnn, epsilon, reward_factors, device, mission_possibility, poisson_rate_s, poisson_rate_g,
                 packet_frequency, computing_demand_factor, computing_demand_factor_2, size_after_computing_factor, size_after_computing_1,
                 begin_time, end_time, time_stride, tle_filepath, SOD_file_path, world_path, mean_interval_time, memory,
                 computing_ability, transmission_rate, downlink_rate, downstream_delays, packet_size_range, state_update_period, print_cycle,del_cycle, visualize=False,
                 print_info=False, save_log=False, show_detail=False, random_edges_del=0, random_nodes_del=0, update_cycle=1, save_training_data=None,
                 elevation_angle=45, pole=False, markov_mode=False, node_p=0, node_q=0, edge_p=0, edge_q=0,
                  ppo_shuffle_func=None, scenario_seed=None, fault_compute_double_mean=True,
                  fault_fill_multiplier=2.0,
                  historical_self_feature_elimination=False, disable_self_feature_elimination=False, gnn_information_hops=4,
                  gnn_include_self=False, gnn_node_feature_dim=3, experience_callback=None,
                  tradition_replan_interval=None, require_computed_before_downlink=False,
                  tradition_use_genetic_search=False, tradition_ga_population_size=20,
                  tradition_ga_generations=10, tradition_ga_path_candidates=3,
                  tradition_ga_mutation_rate=0.1, workload_trace_path=None,
                  workload_trace_sha256=None, workload_trace_task_count=None,
                  fault_trace_path=None, fault_trace_sha256=None,
                  fault_trace_event_count=None, uncomputed_reach_penalty=0.0,
                    heartbeat_timeout=0.15):
        self.tracker = SatelliteTracker(tle_filepath)
        self.coordinates = extract_landmarks(SOD_file_path)
        self.world = gpd.read_file(world_path)
        self.graph_builder = SatelliteGraph()
        self.begin_time = begin_time
        self.end_time = end_time
        self.time_stride = time_stride
        self.mean_interval_time = mean_interval_time
        self.visualizer = SatelliteVisualizer_geo(self.world, edge_color=False) if visualize else None
        self.logger = Logger(detail=show_detail, save_log=save_log, verbose=print_info)
        self.ts = load.timescale()
        self.transmission_rate = transmission_rate
        self.downstream_delays=downstream_delays
        self.downlink_rate=downlink_rate
        self.packet_size_range = packet_size_range
        self.state_update_period = state_update_period
        self.heartbeat_timeout = heartbeat_timeout
        self.random_edges_del = random_edges_del
        self.random_nodes_del = random_nodes_del
        self.markov_mode = markov_mode
        self.node_p = node_p
        self.node_q = node_q
        self.edge_p = edge_p
        self.edge_q = edge_q
        self.elevation_angle = elevation_angle
        self.pole = pole
        self.statics = []
        self.time_acc = 0.0
        self.mode=mode
        self.select_mode=select_mode
        self.q_net=q_net
        self.gnn=gnn
        self.ppo_shuffle_func=ppo_shuffle_func
        self.epsilon=epsilon
        self.reward_factors = reward_factors
        self.uncomputed_reach_penalty = uncomputed_reach_penalty
        self.device=device
        self.poisson_rate_s = poisson_rate_s
        self.poisson_rate_g = poisson_rate_g
        self.packet_frequency=packet_frequency
        self.computing_demand_factor=computing_demand_factor
        self.computing_demand_factor_2=computing_demand_factor_2
        self.size_after_computing_1=size_after_computing_1
        self.size_after_computing_factor=size_after_computing_factor
        self.memory=memory
        self.computing_ability=computing_ability
        self.computing_abilities, self.computing_ability_filepath = load_computing_abilities(
            tle_filepath,
            (satellite.name for satellite in self.tracker.satellites),
            fallback_ability=computing_ability,
        )
        self.mission_possibility=mission_possibility

        self.update_cycle = update_cycle
        self.current_cycle = 0.0
        self.del_cycle = del_cycle
        self.del_update=True
        self.current_del_cycle = 0.0
        self.last_removed_nodes = set()
        self.last_removed_edges = set()
        self.scenario_seed = scenario_seed
        self.fault_compute_double_mean = fault_compute_double_mean
        self.fault_fill_multiplier = fault_fill_multiplier
        self.historical_self_feature_elimination = historical_self_feature_elimination
        self.disable_self_feature_elimination = disable_self_feature_elimination
        self.require_computed_before_downlink = require_computed_before_downlink
        self.gnn_information_hops = gnn_information_hops
        self.gnn_include_self = gnn_include_self
        self.gnn_node_feature_dim = gnn_node_feature_dim
        self.tradition_replan_interval = tradition_replan_interval
        self.tradition_use_genetic_search = tradition_use_genetic_search
        self.tradition_ga_population_size = tradition_ga_population_size
        self.tradition_ga_generations = tradition_ga_generations
        self.tradition_ga_path_candidates = tradition_ga_path_candidates
        self.tradition_ga_mutation_rate = tradition_ga_mutation_rate
        self.experience_callback = experience_callback
        self.workload_trace_path = workload_trace_path
        self.workload_trace_sha256 = workload_trace_sha256
        self.workload_trace_task_count = workload_trace_task_count
        self.fault_trace_path = Path(fault_trace_path).resolve() if fault_trace_path else None
        self.fault_trace_expected_sha256 = fault_trace_sha256
        self.fault_trace_expected_count = (
            int(fault_trace_event_count)
            if fault_trace_event_count is not None else None
        )
        self.fixed_fault_events = None
        self.fixed_fault_trace_content = None
        if self.fault_trace_path is not None:
            self._load_fixed_fault_trace()
        self.scenario_rng = random.Random(f'fault:{scenario_seed}') if scenario_seed is not None else random
        self.fault_trace_hasher = hashlib.sha256()
        self.fault_trace_count = 0

        self.print_cycle = print_cycle
        self.current_print_cycle = 0.0
        self.iteration_counter = 0
        self.print_cycle_iterations = int(print_cycle / time_stride)
        self.save_training_data=save_training_data

        self.step_num=0
        self.reporting_step_offset = 0
        self.warmup_active = False
        self.rewards=[]
        self.last_average_ending_reward = None
        self.last_interval_metrics = None

        self.current_graph = None
        self.topology_graph = None
        self.connections = None
        self.ground_cover_sats = None

        self.link_failure_time_total = 0

        self.reset(self.begin_time)
        self.current_time = self.begin_time

    def _load_fixed_fault_trace(self):
        content = self.fault_trace_path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if (
            self.fault_trace_expected_sha256 is not None
            and digest != self.fault_trace_expected_sha256
        ):
            raise ValueError(f'Fault trace SHA256 mismatch: {self.fault_trace_path}')

        events = {}
        failed = set()
        event_count = 0
        last_time = -1.0
        for line_number, raw_line in enumerate(content.splitlines(), 1):
            if not raw_line.strip():
                continue
            event = json.loads(raw_line)
            event_time = round(float(event['time']), 9)
            if event_time < last_time or event_time < 0:
                raise ValueError(
                    f'Invalid fault event time at line {line_number}'
                )
            if abs(event_time % self.update_cycle) > 1e-9:
                raise ValueError(
                    f'Fault event time is not aligned with topology updates at '
                    f'line {line_number}'
                )
            edge = tuple(sorted(str(node) for node in event['edge']))
            if len(edge) != 2:
                raise ValueError(f'Invalid fault edge at line {line_number}')
            action = event['action']
            if action == 'fail':
                if edge in failed:
                    raise ValueError(f'Duplicate fault at line {line_number}')
                failed.add(edge)
            elif action == 'recover':
                if edge not in failed:
                    raise ValueError(f'Unexpected recovery at line {line_number}')
                failed.remove(edge)
            else:
                raise ValueError(f'Invalid fault action at line {line_number}')
            events.setdefault(event_time, []).append((action, edge))
            event_count += 1
            last_time = event_time

        if (
            self.fault_trace_expected_count is not None
            and event_count != self.fault_trace_expected_count
        ):
            raise ValueError(
                f'Fault trace event count mismatch: expected '
                f'{self.fault_trace_expected_count}, got {event_count}'
            )
        self.fault_trace_expected_count = event_count
        self.fixed_fault_events = events
        self.fixed_fault_trace_content = content

    def _apply_fixed_fault_events(self, event_time):
        events = self.fixed_fault_events.get(round(float(event_time), 9), ())
        for action, edge in events:
            if action == 'fail':
                self.last_removed_edges.add(edge)
            else:
                self.last_removed_edges.discard(edge)
        self.fault_trace_count += len(events)


    def remove_random_edges(self,G, n,update=False):
        if n > G.number_of_edges():
            raise ValueError("Cannot remove more edges than exist in the graph")
        if update:
            self.last_removed_edges = self.scenario_rng.sample(sorted(G.edges()), n)
        G.remove_edges_from(self.last_removed_edges)
        return G

    def remove_random_nodes(self,G, n, update=False):
        if n > G.number_of_nodes():
            raise ValueError("Cannot remove more nodes than exist in the graph")
        if update:
            self.last_removed_nodes = self.scenario_rng.sample(sorted(G.nodes()), n)
        G.remove_nodes_from(self.last_removed_nodes)
        return G

    def remove_random_edges_markov(self, G, p, q, update=False):
        if update:
            alive_edges = sorted(set(G.edges()) - self.last_removed_edges)
            new_failure_edges = {e for e in alive_edges if self.scenario_rng.random() < p}
            failed_edges = sorted(self.last_removed_edges)
            new_alive_edges = {e for e in failed_edges if self.scenario_rng.random() < q}
            self.last_removed_edges = (self.last_removed_edges - new_alive_edges) | new_failure_edges
        self.link_failure_time_total += len(self.last_removed_edges) * self.del_cycle * 2
        G.remove_edges_from(self.last_removed_edges)
        return G

    def remove_random_nodes_markov(self, G, p, q, update=False):
        if update:
            alive_nodes = sorted(set(G.nodes()) - self.last_removed_nodes)
            new_failure_nodes = {e for e in alive_nodes if self.scenario_rng.random() < p}
            failed_nodes = sorted(self.last_removed_nodes)
            new_alive_nodes = {e for e in failed_nodes if self.scenario_rng.random() < q}
            self.last_removed_nodes = (self.last_removed_nodes - new_alive_nodes) | new_failure_nodes
        G.remove_nodes_from(self.last_removed_nodes)
        return G

    def record_fault_trace(self):
        edges = sorted(
            tuple(sorted((str(left), str(right))))
            for left, right in self.last_removed_edges
        )
        nodes = sorted(str(node) for node in self.last_removed_nodes)
        record = (
            f'{self.current_time}|nodes={";".join(nodes)}|edges='
            + ';'.join(f'{left}>{right}' for left, right in edges)
            + '\n'
        )
        self.fault_trace_hasher.update(record.encode('utf-8'))
        self.fault_trace_count += 1

    def get_satellites_over_region(self, sat_LLA_dict):
        satellites_over_region = set()
        geometries = self.world.geometry
        spatial_index = self.world.sindex

        for sat_name, coords in sat_LLA_dict.items():
            lat = coords["latitude"]
            lon = coords["longitude"]
            point = Point(lon, lat)
            for index in spatial_index.query(point):
                geom = geometries.iloc[index]
                if isinstance(geom, (Polygon, MultiPolygon)) and geom.contains(point):
                    satellites_over_region.add(sat_name)
                    break
        return satellites_over_region

    def step(self,epsilon):
        self.step_num+=1

        if self.fixed_fault_events is not None:
            simulation_time = round(float(self.simulator.env.now), 9)
            update_due = (
                simulation_time > 0
                and abs(simulation_time % self.update_cycle) <= 1e-9
            )
            fault_due = (
                simulation_time > 0
                and simulation_time in self.fixed_fault_events
            )
            if update_due or fault_due:
                coordinates_s = self.tracker.generate_satellite_LLA_dict(
                    self.time_from_str(self.current_time)
                )
                ground_cover_sats = self.get_satellites_over_region(coordinates_s)
                connections = get_connections_h3(
                    self.coordinates, coordinates_s, self.elevation_angle
                )
                topology_graph = self.graph_builder.build_graph_with_fixed_edges(
                    self.tracker,
                    self.time_from_str(self.current_time),
                    pole=False,
                )
                current_graph = topology_graph.copy()
                self._apply_fixed_fault_events(simulation_time)
                current_graph.remove_edges_from(self.last_removed_edges)
                self.current_graph = current_graph
                self.topology_graph = topology_graph
                self.connections = connections
                self.ground_cover_sats = ground_cover_sats
            self.link_failure_time_total += (
                len(self.last_removed_edges) * self.time_stride * 2
            )
        else:
            if self.current_del_cycle >= self.del_cycle:
                self.current_del_cycle -= self.del_cycle
                self.del_update = True
            if self.current_cycle >= self.update_cycle:
                self.current_cycle -= self.update_cycle
                coordinates_s = self.tracker.generate_satellite_LLA_dict(self.time_from_str(self.current_time))
                ground_cover_sats = self.get_satellites_over_region(coordinates_s)
                connections = get_connections_h3(self.coordinates, coordinates_s, self.elevation_angle)
                old_nodes = set(self.simulator.graph.nodes())
                topology_graph = self.graph_builder.build_graph_with_fixed_edges(self.tracker, self.time_from_str(self.current_time),pole=False)
                current_graph = topology_graph.copy()

                fault_updated = self.del_update
                if not self.markov_mode:
                    self.remove_random_nodes(current_graph, self.random_nodes_del, self.del_update)
                    self.remove_random_edges(current_graph, self.random_edges_del, self.del_update)
                else:
                    self.remove_random_nodes_markov(current_graph, self.node_p, self.node_q, self.del_update)
                    self.remove_random_edges_markov(current_graph, self.edge_p, self.edge_q, self.del_update)

                if fault_updated:
                    self.record_fault_trace()

                self.del_update = False
                new_nodes = set(current_graph.nodes())
                lost_nodes = old_nodes - new_nodes
                for landmark, satellites in connections.items():
                    for lost_node in sorted(lost_nodes):
                        if lost_node in satellites:
                            connections[landmark].remove(lost_node)
                self.current_graph=current_graph
                self.topology_graph = topology_graph
                self.connections=connections
                self.ground_cover_sats = ground_cover_sats

            self.current_del_cycle += self.time_stride
            self.current_cycle += self.time_stride

        self.current_graph.graph['simulation_step'] = self.step_num
        self.simulator.upgrade_all(
            self.current_graph,
            self.connections,
            self.ground_cover_sats,
            topology_graph=self.topology_graph,
        )
        for satellite in self.simulator.satellites:
            self.simulator.satellites[satellite].epsilon=epsilon

        self.simulator.run(self.time_stride)
        experiences=self.simulator.propagator.experiences
        self.simulator.propagator.experiences = []
        self.rewards.extend(self.simulator.propagator.final_rewards)
        self.iteration_counter += 1
        if self.iteration_counter >= self.print_cycle_iterations:
            self.iteration_counter = 0
            if not self.warmup_active:
                self.print_and_save_accumulated_data()
            self.current_print_cycle = 0.0
            self.rewards=[]

        self.time_acc += self.time_stride
        if self.time_acc >= 1.0:
            self.current_time = self.add_time_to_str(self.current_time, (0, int(self.time_acc)))
            self.time_acc -= int(self.time_acc)

        return experiences

    def apply_scenario(self, environment):
        for name in (
            'poisson_rate_s', 'poisson_rate_g', 'packet_frequency',
            'mean_interval_time', 'random_edges_del', 'random_nodes_del',
            'markov_mode', 'node_p', 'node_q', 'edge_p', 'edge_q',
            'fault_compute_double_mean', 'historical_self_feature_elimination',
            'disable_self_feature_elimination',
            'gnn_information_hops', 'gnn_include_self', 'gnn_node_feature_dim',
        ):
            if name in environment:
                setattr(self, name, environment[name])
        if 'scenario_seed' in environment:
            self.scenario_seed = environment['scenario_seed']
            self.scenario_rng = (
                random.Random(f'fault:{self.scenario_seed}')
                if self.scenario_seed is not None else random
            )

    def reset(self,begin_time):
        self.reporting_step_offset = 0
        self.warmup_active = False
        self.statics= []
        self.rewards=[]
        self.last_average_ending_reward = None
        self.last_interval_metrics = None
        self.begin_time=begin_time
        init_time = self.time_from_str(begin_time)
        topology_graph = self.graph_builder.build_graph_with_fixed_edges(
            self.tracker, init_time, pole=self.pole
        )
        current_graph = topology_graph.copy()
        self.num_edges = len(topology_graph.edges())
        self.num_nodes = len(topology_graph.nodes())
        coordinates_s = self.tracker.generate_satellite_LLA_dict(init_time)
        ground_cover_sats = self.get_satellites_over_region(coordinates_s)
        self.ground_cover_sats = ground_cover_sats
        connections = get_connections_h3(self.coordinates, coordinates_s, self.elevation_angle)
        self.simulator = SatelliteNetworkSimulator_OnbardComputing(
            mode=self.mode,
            select_mode=self.select_mode,
            q_net=self.q_net,
            gnn=self.gnn,
            ppo_shuffle_func=self.ppo_shuffle_func,
            reward_factors = self.reward_factors,
            uncomputed_reach_penalty=self.uncomputed_reach_penalty,
            epsilon= self.epsilon,
            device= self.device,
            mission_possibility=self.mission_possibility,
            poisson_rate_s= self.poisson_rate_s,
            poisson_rate_g= self.poisson_rate_g,
            packet_frequency= self.packet_frequency,
            computing_demand_factor= self.computing_demand_factor,
            computing_demand_factor_2=self.computing_demand_factor_2,
            size_after_computing_factor= self.size_after_computing_factor,
            size_after_computing_1=self.size_after_computing_1,
            graph=current_graph,
            topology_graph=topology_graph,
            landmarks=connections,
            ground_cover_sats=ground_cover_sats,
            mean_interval_time=self.mean_interval_time,
            memory=self.memory,
            computing_abilities=self.computing_abilities,
            transmission_rate=self.transmission_rate,
            downstream_delays=self.downstream_delays,
            downlink_rate=self.downlink_rate,
            packet_size_range=self.packet_size_range,
            state_update_period=self.state_update_period,
            heartbeat_timeout=self.heartbeat_timeout,
            logger=self.logger,
            scenario_seed=self.scenario_seed,
            fault_compute_double_mean=self.fault_compute_double_mean,
            fault_fill_multiplier=self.fault_fill_multiplier,
            historical_self_feature_elimination=self.historical_self_feature_elimination,
            disable_self_feature_elimination=self.disable_self_feature_elimination,
            require_computed_before_downlink=self.require_computed_before_downlink,
            gnn_information_hops=self.gnn_information_hops,
            gnn_include_self=self.gnn_include_self,
            gnn_node_feature_dim=self.gnn_node_feature_dim,
            tradition_replan_interval=self.tradition_replan_interval,
            tradition_use_genetic_search=self.tradition_use_genetic_search,
            tradition_ga_population_size=self.tradition_ga_population_size,
            tradition_ga_generations=self.tradition_ga_generations,
            tradition_ga_path_candidates=self.tradition_ga_path_candidates,
            tradition_ga_mutation_rate=self.tradition_ga_mutation_rate,
            experience_callback=self.experience_callback,
            workload_trace_path=self.workload_trace_path,
            workload_trace_sha256=self.workload_trace_sha256,
            workload_trace_task_count=self.workload_trace_task_count)

        self.current_time = self.begin_time
        self.time_acc = 0.0
        self.current_cycle =0.0
        self.current_print_cycle=0.0
        self.current_del_cycle = 0.0
        self.iteration_counter = 0
        self.del_update = True
        self.last_removed_nodes = set()
        self.last_removed_edges = set()
        self.link_failure_time_total = 0
        self.fault_trace_hasher = hashlib.sha256(
            self.fixed_fault_trace_content or b''
        )
        self.fault_trace_count = 0
        old_nodes = set(self.simulator.graph.nodes())
        topology_graph = self.graph_builder.build_graph_with_fixed_edges(
            self.tracker, self.time_from_str(self.current_time), pole=False
        )
        current_graph = topology_graph.copy()

        if self.fixed_fault_events is not None:
            self._apply_fixed_fault_events(0.0)
            current_graph.remove_edges_from(self.last_removed_edges)
        elif not self.markov_mode:
            self.remove_random_nodes(current_graph, self.random_nodes_del, self.del_update)
            self.remove_random_edges(current_graph, self.random_edges_del, self.del_update)
        else:
            self.remove_random_nodes_markov(current_graph, self.node_p, self.node_q, self.del_update)
            self.remove_random_edges_markov(current_graph, self.edge_p, self.edge_q, self.del_update)
        if self.fixed_fault_events is None:
            self.record_fault_trace()
        self.del_update = False
        new_nodes = set(current_graph.nodes())
        lost_nodes = old_nodes - new_nodes
        for landmark, satellites in connections.items():
            for lost_node in sorted(lost_nodes):
                if lost_node in satellites:
                    connections[landmark].remove(lost_node)
        current_graph.graph['simulation_step'] = 0
        self.current_graph = current_graph
        self.topology_graph = topology_graph
        self.connections = connections

    def render(self):
        self.visualize(self.simulator.graph, self.current_time, self.simulator)

    def visualize(self,current_graph,current_time,simulator):
        system_state=simulator.get_system_state()

        G_draw=current_graph.copy()
        self.update_node_colors(G_draw, system_state)

        for landmark, pos_value in self.coordinates.items():
            G_draw.add_node(landmark)
            G_draw.nodes[landmark]['pos_0'] = [pos_value["latitude"],pos_value ["longitude"], pos_value["altitude"]]
            G_draw.nodes[landmark]['color'] = 'purple'
        self.visualizer.draw_graph(G_draw)

    def show_satellite_computing_time(self):
        satellite_computing_times={}
        for satellite in self.simulator.satellites:
            satellite_computing_times[satellite]=self.simulator.satellites[satellite].computing_time
        self.print_and_save(str(satellite_computing_times))

    def show_network_transmission_occupation(self, duration):
        total_occupation_time = 0
        if not self.markov_mode:
            self.link_failure_time_total = self.random_edges_del * duration * 2
        for satellite in self.simulator.satellites:
            total_occupation_time += self.simulator.satellites[satellite].transmission_time
        total_usable_time = duration * self.num_edges * 2 - self.link_failure_time_total
        average_network_capacity = total_usable_time / duration * self.transmission_rate
        average_network_occupation = total_occupation_time/duration * self.transmission_rate
        average_network_occupation_rate = average_network_occupation/average_network_capacity
        self.print_and_save(f"Average network capacity: {'{:.3f} Gbps'.format(average_network_capacity/1e9)}")
        self.print_and_save(f"Average network flow: {'{:.3f} Gbps'.format(average_network_occupation/1e9)}")
        self.print_and_save(f"Average network occupation rate: {'{:.2%}'.format(average_network_occupation_rate)}")

    def show_equivalent_transmission_data(self, duration):
        equivalent_transmission_data_size = 0
        for satellite in self.simulator.satellites:
            equivalent_transmission_data_size += self.simulator.satellites[satellite].equivalent_transmission_data_size
        self.print_and_save(f"Average network equivalent transmission data size: {'{:.3f} GB/s'.format(equivalent_transmission_data_size / duration / 8 / 1e9)}")

    def validate_fixed_trace_replay(self):
        if (
            self.simulator.fixed_workload_sessions is not None
            and self.simulator.workload_trace_count
            != self.simulator.workload_trace_expected_count
        ):
            raise RuntimeError(
                f'Incomplete workload trace replay: expected '
                f'{self.simulator.workload_trace_expected_count}, got '
                f'{self.simulator.workload_trace_count}'
            )
        if (
            self.fixed_fault_events is not None
            and self.fault_trace_count != self.fault_trace_expected_count
        ):
            raise RuntimeError(
                f'Incomplete fault trace replay: expected '
                f'{self.fault_trace_expected_count}, got '
                f'{self.fault_trace_count}'
            )

    def show_test_trace_fingerprints(self):
        self.print_and_save(
            f'Workload trace count: {self.simulator.workload_trace_count}'
        )
        self.print_and_save(
            f'Workload trace SHA256: {self.simulator.workload_trace_hasher.hexdigest()}'
        )
        self.print_and_save(f'Fault trace count: {self.fault_trace_count}')
        self.print_and_save(
            f'Fault trace SHA256: {self.fault_trace_hasher.hexdigest()}'
        )
        self.validate_fixed_trace_replay()
        if self.fixed_fault_events is not None:
            self.print_and_save('Fixed test trace replay: verified')

    def print_and_save(self, message):
        print(message)
        if self.save_training_data:
            file_path = os.path.join('./training_process_data', self.save_training_data)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'a') as file:
                file.write(message + '\n')

    def print_and_save_accumulated_data(self):
        self.print_and_save(f"====== step {self.step_num - self.reporting_step_offset} ======")
        self.print_and_save(f"====== {self.current_time} ======")
        current_statics_snapshot = copy.deepcopy(self.simulator.statics_datas)
        if self.statics:
            current_statics = {k: current_statics_snapshot[k] - self.statics[-1][k] for k in current_statics_snapshot}
        else:
            current_statics = current_statics_snapshot
        self.statics.append(current_statics_snapshot)

        d = current_statics
        loss_breakdown = {
            key: d[key]
            for key in sorted(d)
            if key.startswith('Lost_')
            and key not in {'Lost_upload', 'Lost_relay_0', 'Lost_relay_1'}
        }
        packet_loss_rates = (d['Lost_relay_0'] + d['Lost_relay_1'] + d['Lost_upload']) / (d['Lost_relay_0'] + d['Lost_relay_1'] + d['Lost_upload'] + d['Reached_0'] + d['Reached_1']) if d['Lost_relay_0'] + d['Lost_relay_1'] + d['Lost_upload'] + d['Reached_0'] + d['Reached_1'] > 0 else None
        average_delays = (d['Total_delay_0'] + d['Total_delay_1']) / (d['Reached_0'] + d['Reached_1']) if d['Reached_0'] + d['Reached_1'] > 0 else None
        average_hops = (d['Total_hops_0'] + d['Total_hops_1']) / (d['Reached_0'] + d['Reached_1']) if d['Reached_0'] + d['Reached_1'] > 0 else None
        average_computing_ratio = d['Is_computing'] / self.num_nodes / (self.print_cycle_iterations)
        average_computing_waiting_time = (d['Computing_waiting_time']) / (d['Reached_0'] + d['Reached_1']) if d['Reached_0'] + d['Reached_1'] > 0 else None
        reached = d['Reached_0'] + d['Reached_1']
        average_delay_components = {
            component: (
                (d[f'Total_{component}_delay_0'] + d[f'Total_{component}_delay_1']) / reached
                if reached > 0 else None
            )
            for component in ('computation', 'queueing', 'propagation', 'transmission')
        }
        average_component_sum = (
            sum(average_delay_components.values()) if reached > 0 else None
        )
        average_delay_accounting_gap = (
            average_delays - average_component_sum if reached > 0 else None
        )

        self.print_and_save(f"current_statics: {current_statics}")
        self.print_and_save(f"Packet loss breakdown: {json.dumps(loss_breakdown, sort_keys=True)}")
        self.print_and_save(f"Packet loss rate: {'{:.2%}'.format(packet_loss_rates) if packet_loss_rates is not None else 'None'}")
        self.print_and_save(f"Average delay for successful transmissions: {'{:.3f} seconds'.format(average_delays) if average_delays is not None else 'None'}")
        self.print_and_save(f"Average hop count for successful transmissions: {'{:.3f} hops'.format(average_hops) if average_hops is not None else 'None'}")
        self.print_and_save(f"Proportion of satellites in computation: {'{:.2%}'.format(average_computing_ratio) if average_computing_ratio is not None else 'None'}")
        self.print_and_save(f"Average waiting time for computing: {'{:.3f} seconds'.format(average_computing_waiting_time) if average_computing_waiting_time is not None else 'None'}")
        self.print_and_save(f"Average computation delay: {'{:.6f} seconds'.format(average_delay_components['computation']) if reached > 0 else 'None'}")
        self.print_and_save(f"Average queueing delay: {'{:.6f} seconds'.format(average_delay_components['queueing']) if reached > 0 else 'None'}")
        self.print_and_save(f"Average propagation delay: {'{:.6f} seconds'.format(average_delay_components['propagation']) if reached > 0 else 'None'}")
        self.print_and_save(f"Average transmission delay: {'{:.6f} seconds'.format(average_delay_components['transmission']) if reached > 0 else 'None'}")
        self.print_and_save(f"Average delay component sum: {'{:.6f} seconds'.format(average_component_sum) if reached > 0 else 'None'}")
        self.print_and_save(f"Average delay accounting gap: {'{:.9f} seconds'.format(average_delay_accounting_gap) if reached > 0 else 'None'}")

        rewards = sum(self.rewards) / len(self.rewards) if len(self.rewards) > 0 else None
        self.last_average_ending_reward = rewards
        self.last_interval_metrics = {
            'average_reward': rewards,
            'average_delay': average_delays,
            'packet_loss_rate': packet_loss_rates,
            'average_hop_count': average_hops,
            'average_computation_delay': average_delay_components['computation'],
            'average_queueing_delay': average_delay_components['queueing'],
            'average_propagation_delay': average_delay_components['propagation'],
            'average_transmission_delay': average_delay_components['transmission'],
            'average_delay_component_sum': average_component_sum,
            'average_delay_accounting_gap': average_delay_accounting_gap,
        }
        self.print_and_save(f"Average ending reward: {rewards if rewards is not None else 'None'}")
