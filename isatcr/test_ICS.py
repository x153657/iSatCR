"""Evaluate periodic centralized batch genetic-search ICS (GA-ICS)."""

import ast
import copy
import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice, product
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

import RL_environment_for_computing as environment_module
import SatelliteNetworkSimulator_Computing as simulator_module
from train_4_5seed import TRAINING_FAULT_FILL_MULTIPLIER
from training_runner import run_training, training_input_hashes, runtime_acceleration_contract, archive_test_outputs


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / 'training_process_data' / 'v1' / 'test' / 'test' / 'ICS'
CONFIG_ROOT = PROJECT_ROOT / 'training_process_data' / 'v1' / 'test' / 'test' / 'config'
SATELLITE_TLE_FILE = './Satellite_Data/wallker_tles.txt'
GROUND_STATION_FILE = './Ground_Data/11_ground_stations.txt'
COMPUTING_ABILITY_FILE = './Satellite_Data/wallker_tles_computing_ability.csv'


def read_fixed_config(name):
    return json.loads((CONFIG_ROOT / name).read_text(encoding='utf-8'))


COMMON_TEST = read_fixed_config('common.json')
COMMON_TEST['environment']['fault_fill_multiplier'] = TRAINING_FAULT_FILL_MULTIPLIER
LOAD_TESTS = read_fixed_config('load_experiments.json')
FAULT_TESTS = read_fixed_config('fault_experiments.json')
# ICS-specific fixed conditions; keep the shared test contracts unchanged.
LOAD_TESTS.update(edge_p=0.18, edge_q=0.82)
FAULT_TESTS['workload'] = {
    field: next(case for case in LOAD_TESTS['cases'] if case['id'] == 'load_160')[field]
    for field in ('target_load_tasks_per_second', 'poisson_rate_g', 'poisson_rate_s')
}
SCENARIO_MANIFEST = read_fixed_config(
    f"scenario_seed_{COMMON_TEST['general']['scenario_seed']}/manifest.json"
)


def resolve_trace(entry):
    path = (CONFIG_ROOT / entry['path']).resolve()
    if CONFIG_ROOT.resolve() not in path.parents:
        raise ValueError(f'Trace path is outside the test config directory: {path}')
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry['sha256']:
        raise ValueError(f'Trace SHA256 mismatch: {path}')
    return path


LOAD_FAULT_CASE = next(
    case for case in FAULT_TESTS['cases']
    if all(
        abs(float(case[field]) - float(LOAD_TESTS[field])) <= 1e-12
        for field in ('edge_p', 'edge_q')
    )
)


def case_trace_entries(test_type, setting):
    if test_type == 'load':
        workload = SCENARIO_MANIFEST['workloads'][setting['id']]
        fault = SCENARIO_MANIFEST['faults'][LOAD_FAULT_CASE['id']]
    else:
        workload_case = next(
            case for case in LOAD_TESTS['cases']
            if all(
                float(case[field]) == float(FAULT_TESTS['workload'][field])
                for field in (
                    'target_load_tasks_per_second',
                    'poisson_rate_g',
                    'poisson_rate_s',
                )
            )
        )
        workload = SCENARIO_MANIFEST['workloads'][workload_case['id']]
        fault = SCENARIO_MANIFEST['faults'][setting['id']]
    return workload, fault


class PICSReservationBook:
    """Track memory and compute commitments for plans not fully executed yet."""

    def __init__(self, graph, memory):
        self.graph = graph
        self.memory = float(memory)
        self.next_id = 0
        self.entries = {}
        self.reserved_memory = {}
        self.reserved_computing = {}
        self.active_computing = {}
        for node in graph.nodes:
            graph.nodes[node]['reserved_memory'] = 0.0
            graph.nodes[node]['reserved_computing'] = 0.0
            graph.nodes[node]['active_computing'] = 0.0

    def _adjust(self, values, node, amount):
        updated = values.get(node, 0.0) + amount
        if abs(updated) < 1e-9:
            values.pop(node, None)
            updated = 0.0
        else:
            values[node] = updated
        if node in self.graph.nodes:
            field = (
                'reserved_memory'
                if values is self.reserved_memory
                else (
                    'reserved_computing'
                    if values is self.reserved_computing
                    else 'active_computing'
                )
            )
            self.graph.nodes[node][field] = updated

    def memory_for(self, node):
        return self.reserved_memory.get(node, 0.0)

    def compute_for(self, node):
        return self.reserved_computing.get(node, 0.0)

    def _entry_for(self, packet):
        reservation_id = getattr(packet, 'pics_reservation_id', None)
        return self.entries.get(reservation_id)

    def memory_for_packet(self, packet, node):
        entry = self._entry_for(packet)
        if entry is None:
            return 0.0
        return entry['memory'].get(node, 0.0)

    def can_accept(self, node, payload, packet=None):
        own_reservation = self.memory_for_packet(packet, node) if packet else 0.0
        reserved_for_other_packets = max(
            self.memory_for(node) - own_reservation,
            0.0,
        )
        current = self.graph.nodes[node].get('current_memory_occupy', 0.0)
        return current + reserved_for_other_packets + payload <= self.memory

    def _store_entry(self, packet, memory_by_node, compute_by_node):
        reservation_id = self.next_id
        self.next_id += 1
        entry = {
            'memory': dict(memory_by_node),
            'computing': dict(compute_by_node),
            'active_computing': {},
        }
        self.entries[reservation_id] = entry
        for node, amount in entry['memory'].items():
            self._adjust(self.reserved_memory, node, amount)
        for node, amount in entry['computing'].items():
            self._adjust(self.reserved_computing, node, amount)
        packet.pics_reservation_id = reservation_id
        return True

    def release(self, packet):
        reservation_id = getattr(packet, 'pics_reservation_id', None)
        entry = self.entries.pop(reservation_id, None)
        if entry is None:
            packet.pics_reservation_id = None
            return None
        for node, amount in entry['memory'].items():
            self._adjust(self.reserved_memory, node, -amount)
        for node, amount in entry['computing'].items():
            self._adjust(self.reserved_computing, node, -amount)
        for node, amount in entry['active_computing'].items():
            self._adjust(self.active_computing, node, -amount)
        packet.pics_reservation_id = None
        return entry

    def restore(self, packet, entry):
        if entry is None:
            return False
        memory_by_node = entry['memory']
        if any(
            not self.can_accept(node, amount, packet)
            for node, amount in memory_by_node.items()
        ):
            return False
        return self._store_entry(
            packet,
            memory_by_node,
            entry['computing'],
        )

    def commit(self, packet, route, compute_node, size, reduced_size, is_computed):
        memory_by_node = {}
        if is_computed:
            for node in route:
                memory_by_node[node] = memory_by_node.get(node, 0.0) + size
        else:
            # Routes exclude the current satellite. When computation is local,
            # every remaining hop carries the reduced payload.
            compute_index = route.index(compute_node) if compute_node in route else -1
            for index, node in enumerate(route):
                memory_by_node[node] = memory_by_node.get(node, 0.0) + (
                    size if index <= compute_index
                    else reduced_size
                )

        if any(
            not self.can_accept(node, amount, packet)
            for node, amount in memory_by_node.items()
        ):
            return False
        compute_by_node = (
            {} if is_computed or compute_node is None
            else {compute_node: int(packet.information[2])}
        )
        return self._store_entry(packet, memory_by_node, compute_by_node)

    def arrive(self, packet, node):
        entry = self._entry_for(packet)
        if entry is None:
            return
        amount = min(entry['memory'].get(node, 0.0), packet.size)
        remaining = entry['memory'].get(node, 0.0) - amount
        if remaining:
            entry['memory'][node] = remaining
        else:
            entry['memory'].pop(node, None)
        if amount:
            self._adjust(self.reserved_memory, node, -amount)

    def start_computing(self, packet, node):
        entry = self._entry_for(packet)
        if entry is None:
            return
        amount = entry['computing'].pop(node, 0.0)
        if amount:
            self._adjust(self.reserved_computing, node, -amount)
            entry['active_computing'][node] = amount
            self._adjust(self.active_computing, node, amount)

    def finish_computing(self, packet, node):
        entry = self._entry_for(packet)
        if entry is None:
            return
        amount = entry['active_computing'].pop(node, 0.0)
        if amount:
            self._adjust(self.active_computing, node, -amount)


class PICSSatellite(simulator_module.Satellite_with_Computing):
    """Tradition satellite used by the centralized GA-ICS planner."""

    def tradition_routing(self, current_state, computing=True):
        if self.mode == 'Tradition':
            return self.exhaustive_tradition_routing(current_state, computing)
        return super().tradition_routing(current_state, computing)

    def exhaustive_tradition_routing(self, current_state, computing=True):
        """Enumerate compute nodes and minimize complete end-to-end delay."""
        planning_graph = self.propagator.graph
        _, destination, size, computing_demand, size_after_computing, is_computed = current_state
        # A periodic replan runs at the satellite currently holding the packet.
        # packet.source remains the original source for accounting only; using it
        # here would produce a route whose first hop is not adjacent to self.name.
        source = self.name
        if source not in planning_graph or destination not in planning_graph:
            return [None], None

        def active(node):
            satellite = self.propagator.satellites.get(node)
            return satellite is not None and satellite.active

        available = nx.subgraph_view(
            planning_graph,
            filter_node=active,
            filter_edge=lambda left, right: not planning_graph[left][right].get(
                'missing', 1
            ),
        )
        if source not in available or destination not in available:
            return [None], None

        reservation_book = getattr(self.propagator, 'reservation_book', None)

        def node_can_hold(node, payload, exceptions=()):
            reserved = (
                reservation_book.memory_for(node)
                if reservation_book is not None else 0.0
            )
            return (
                node in exceptions
                or planning_graph.nodes[node].get('current_memory_occupy', 0)
                + reserved
                + payload < self.memory
            )

        def edge_delay(left, right, edge, payload):
            source_satellite = self.propagator.satellites.get(left)
            transmission_size = getattr(
                source_satellite, 'transmission_size', {}
            )
            return (
                transmission_size.get(right, 0) / self.transmission_rate
                + payload / self.transmission_rate
                + edge.get('propagation_weight', 0)
            )

        reward_function = getattr(self.propagator, 'reward_function', None)
        delay_factor = float(getattr(reward_function, 'delay_factor', 1.0))
        memory_factor = float(getattr(reward_function, 'memory_factor', 0.0))
        memory_threshold = getattr(reward_function, 'memory_threshold', None)
        # The config stores the reserved free-space fraction, not occupancy.
        reserved_occupancy = (
            self.memory * (1.0 - float(memory_threshold))
            if memory_threshold is not None else None
        )

        def storage_penalty(route, payload):
            if reserved_occupancy is None or memory_factor == 0.0:
                return 0.0
            return sum(
                memory_factor
                for node in route
                if planning_graph.nodes[node].get('current_memory_occupy', 0)
                + (
                    reservation_book.memory_for(node)
                    if reservation_book is not None else 0.0
                )
                + payload >= reserved_occupancy
            )

        def dijkstra_all(graph, start, payload, reverse=False):
            try:
                return nx.single_source_dijkstra(
                    graph,
                    start,
                    weight=lambda left, right, edge: edge_delay(
                        right if reverse else left,
                        left if reverse else right,
                        edge,
                        payload,
                    ),
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return {}, {}

        if is_computed:
            route_graph = nx.subgraph_view(
                available,
                filter_node=lambda node: node_can_hold(
                    node, size, exceptions=(source,)
                ),
            )
            _, paths = dijkstra_all(route_graph, source, size)
            if destination not in paths:
                return [None], None
            return paths[destination][1:], None

        candidates = sorted(
            node for node in available.nodes
            if planning_graph.nodes[node].get('computing_ability', 0) > 0
            and node_can_hold(node, size, exceptions=(source,))
        )
        first_graph = nx.subgraph_view(
            available,
            filter_node=lambda node: (
                node != destination
                and node_can_hold(node, size, exceptions=(source,))
            ),
        )
        first_distances, first_paths = dijkstra_all(first_graph, source, size)
        second_graph = nx.subgraph_view(
            available,
            filter_node=lambda node: node_can_hold(node, size_after_computing),
        )
        second_distances, second_paths = dijkstra_all(
            second_graph, destination, size_after_computing, reverse=True
        )

        def first_path_to(node):
            if node != destination:
                if node not in first_paths:
                    return None, None
                return first_distances[node], first_paths[node]
            if source == destination:
                return 0.0, [source]
            best_path = None
            for predecessor in planning_graph.neighbors(destination):
                if predecessor not in first_paths:
                    continue
                edge = planning_graph[predecessor][destination]
                if edge.get('missing', 1):
                    continue
                candidate = (
                    first_distances[predecessor]
                    + edge_delay(predecessor, destination, edge, size),
                    first_paths[predecessor] + [destination],
                )
                if best_path is None or candidate[0] < best_path[0]:
                    best_path = candidate
            return best_path if best_path is not None else (None, None)

        def second_path_from(node):
            if node == destination:
                return 0.0, [destination]
            if node in second_paths:
                return (
                    second_distances[node],
                    list(reversed(second_paths[node])),
                )
            best_path = None
            for neighbor in planning_graph.neighbors(node):
                if neighbor not in second_paths:
                    continue
                edge = planning_graph[node][neighbor]
                if edge.get('missing', 1):
                    continue
                candidate = (
                    edge_delay(node, neighbor, edge, size_after_computing)
                    + second_distances[neighbor],
                    [node] + list(reversed(second_paths[neighbor])),
                )
                if best_path is None or candidate[0] < best_path[0]:
                    best_path = candidate
            return best_path if best_path is not None else (None, None)

        best = None
        for computing_node in candidates:
            node_state = planning_graph.nodes[computing_node]
            first_cost, first_path = first_path_to(computing_node)
            if first_path is None:
                continue
            second_cost, second_path = second_path_from(computing_node)
            if second_path is None:
                continue

            first_route = first_path[1:]
            second_route = second_path[1:]
            computing_ability = node_state['computing_ability']
            computing_delay = (
                node_state.get('computing_remain', 0) / computing_ability
                + computing_demand / computing_ability
            )
            predicted_delay = (
                computing_delay
                + float(first_cost)
                + float(second_cost)
                + self.propagator.downstream_delays
            )
            route = first_route + second_route
            score = (
                delay_factor * predicted_delay
                + storage_penalty(first_route, size)
                + storage_penalty(second_route, size_after_computing),
                predicted_delay,
                len(route),
                computing_node,
            )
            if best is None or score < best[0]:
                best = (score, route, computing_node)

        if best is None:
            return [None], None
        _, routing, computing_node = best
        if simulator_module.PRE and computing:
            planning_graph.nodes[computing_node]['computing_remain'] = (
                planning_graph.nodes[computing_node].get('computing_remain', 0)
                + computing_demand
            )
        return routing, computing_node if computing else None


class PICSBatchGAScheduler:
    """Batch unstarted P-ICS tasks with a resource-constrained genetic search."""

    def __init__(self, simulator):
        self.simulator = simulator
        self.interval = float(simulator.tradition_replan_interval or 0.1)
        self.population_size = max(int(simulator.tradition_ga_population_size), 2)
        self.generations = max(int(simulator.tradition_ga_generations), 1)
        self.path_candidates = max(int(simulator.tradition_ga_path_candidates), 1)
        self.mutation_rate = min(
            max(float(simulator.tradition_ga_mutation_rate), 0.0),
            1.0,
        )
        self.max_batch_tasks = 80
        self.priority_age_window = 30.0
        self._ga_report_count = 0
        self._commit_report_count = 0
        self._pending_report_count = 0
        self._candidate_failure_report_count = 0

    @property
    def graph(self):
        return self.simulator.graph

    @property
    def book(self):
        return self.simulator.propagator.reservation_book

    def defer(self, satellite, packet):
        self._release_old_plan(packet)
        packet.routing = None
        packet.computing_node = None
        packet.tradition_last_plan_epoch = None
        simulator_module.mark_queue_entry(packet, 'forward', self.simulator.env.now)
        satellite.forward_queue.put(packet)

    def retry_delay(self, satellite):
        # Waiting for a plan must not block packets already ready to execute.
        ready = any(
            packet.destination == satellite.name
            or packet.computing_node == satellite.name
            or (packet.routing
                and packet.routing[0] in satellite.neighbors
                and getattr(packet, 'tradition_topology_version', None)
                == getattr(self.simulator.propagator, 'topology_version', 0))
            for packet in satellite.forward_queue.items
        )
        return 0.0 if ready else self.interval

    def _active_graph(self, source, payload, blocked=()):
        blocked = set(blocked)

        def active(node):
            satellite = self.simulator.satellites.get(node)
            return satellite is not None and satellite.active

        def can_hold(node):
            if node == source:
                return True
            if node in blocked:
                return False
            return (
                self.graph.nodes[node].get('current_memory_occupy', 0)
                + self.book.memory_for(node)
                + payload
                <= self.simulator.memory
            )

        allowed_nodes = [
            node for node in self.graph
            if active(node) and can_hold(node)
        ]
        active_graph = self.graph.subgraph(allowed_nodes).copy()
        # The live topology already removes failed links. During a topology
        # refresh, remaining edges may not have ``missing`` yet; those edges
        # are usable. Materializing this graph avoids repeating the filters
        # during every NetworkX shortest-path expansion.
        active_graph.remove_edges_from([
            (left, right)
            for left, right, edge in active_graph.edges(data=True)
            if edge.get('missing', 0)
        ])
        return active_graph

    def _edge_delay(self, left, right, edge, payload):
        satellite = self.simulator.satellites.get(left)
        transmission_size = getattr(satellite, 'transmission_size', {})
        return (
            transmission_size.get(right, 0) / satellite.transmission_rate
            + satellite._remaining_transmission_time(right)
            + payload / satellite.transmission_rate
            + edge.get('propagation_weight', 0)
        )

    def _compute_wait(self, node):
        satellite = self.simulator.satellites[node]
        ability = self.graph.nodes[node]['computing_ability']
        elapsed = (self.simulator.env.now - satellite.last_computing_time
                   if satellite.is_computing else 0.0)
        # The counter retains the full active job until completion. Keep
        # queued/reserved work, but discount processing already performed.
        return max(self.graph.nodes[node].get('computing_remain', 0)
                   / ability - elapsed, 0.0)

    def _path_options(self, graph, source, target, payload):
        if source == target:
            return [([source], 0.0)]
        if source not in graph or target not in graph:
            return []
        try:
            # Port queues belong to senders: opposite directions on the same
            # physical link can have different waiting/serialization costs.
            weighted_graph = nx.DiGraph(graph)
            for left, right, edge in weighted_graph.edges(data=True):
                edge['_pics_weight'] = self._edge_delay(
                    left,
                    right,
                    self.graph[left][right],
                    payload,
                )
            paths = nx.shortest_simple_paths(
                weighted_graph,
                source,
                target,
                weight='_pics_weight',
            )
            options = []
            path_limit = max(
                int(getattr(self, '_path_limit', self.path_candidates)),
                1,
            )
            for path in islice(paths, path_limit):
                cost = sum(
                    weighted_graph[left][right]['_pics_weight']
                    for left, right in zip(path, path[1:])
                )
                options.append((path, cost))
            return options
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def _make_plan(
        self,
        first_option,
        second_option,
        compute_node,
        is_computed,
        size,
        reduced_size,
        demand,
    ):
        first_path, first_cost = first_option
        second_path, second_cost = second_option
        if first_path[-1] != compute_node and not is_computed:
            return None
        route = first_path[1:] + second_path[1:]
        if (len(first_path) != len(set(first_path))
                or len(second_path) != len(set(second_path))):
            return None

        memory = {}
        links = {}

        def add_path(path, payload):
            for node in path[1:]:
                memory[node] = memory.get(node, 0.0) + payload
            for left, right in zip(path, path[1:]):
                links[(left, right)] = links.get((left, right), 0) + payload
            return True

        if not add_path(first_path, size):
            return None
        if not is_computed and not add_path(second_path, reduced_size):
            return None

        compute_time = 0.0
        compute_ready = 0.0
        compute_service = 0.0
        if not is_computed:
            ability = self.graph.nodes[compute_node]['computing_ability']
            compute_ready = self._compute_wait(compute_node)
            compute_service = demand / ability
            compute_time = max(compute_ready - float(first_cost), 0.0) + compute_service
        return {
            'route': route,
            'compute_node': None if is_computed else compute_node,
            'compute_arrival': float(first_cost),
            'compute_ready': compute_ready,
            'compute_service': compute_service,
            'memory': memory,
            'computing': {} if is_computed else {compute_node: demand},
            'links': links,
            'delay': (
                compute_time
                + float(first_cost)
                + float(second_cost)
                + self.simulator.downstream_delays
                + (size if is_computed else reduced_size)
                / self.simulator.downlink_rate
            ),
            'hops': len(route),
            'forward_payloads': list(zip(first_path[1:], [size] * (len(first_path) - 1)))
            + ([] if is_computed else list(zip(second_path[1:], [reduced_size] * (len(second_path) - 1)))),
        }

    def _candidate_plans(self, satellite, packet):
        information = packet.information
        is_computed, _, demand, reduced_size = information[:4]
        source = satellite.name
        destination = packet.destination
        size = packet.size
        if source not in self.graph or destination not in self.graph:
            return []

        if is_computed:
            graph = self._active_graph(source, size)
            path_options = self._path_options(
                graph, source, destination, size
            )
            plans = []
            for path, cost in path_options:
                plan = self._make_plan(
                    (path, cost),
                    ([destination], 0.0),
                    None,
                    True,
                    size,
                    reduced_size,
                    demand,
                )
                if plan is not None:
                    plans.append(plan)
            if not plans:
                self._report_candidate_failure(
                    packet,
                    is_computed=True,
                    compute_nodes=0,
                    first_options=len(path_options),
                    second_options=0,
                )
            return plans

        # Two states per satellite enforce exactly one computation while
        # allowing a physical satellite to be revisited after compression.
        first_graph = self._active_graph(source, size)
        second_graph = self._active_graph(source, reduced_size)
        layered = nx.DiGraph()
        for stage, graph, payload in ((0, first_graph, size),
                                      (1, second_graph, reduced_size)):
            layered.add_nodes_from((node, stage) for node in graph)
            for left, right, edge in graph.edges(data=True):
                for sender, receiver in ((left, right), (right, left)):
                    if stage == 0 and sender == destination:
                        continue
                    layered.add_edge((sender, stage), (receiver, stage),
                                     weight=self._edge_delay(sender, receiver, edge, payload))
        for node in first_graph:
            if node == destination and not self.simulator.require_computed_before_downlink:
                continue
            ability = self.graph.nodes[node].get('computing_ability', 0)
            if node in second_graph and ability > 0:
                layered.add_edge((node, 0), (node, 1),
                                 weight=self._compute_wait(node) + demand / ability)

        plans = []
        if (source, 0) in layered and (destination, 1) in layered:
            try:
                paths = nx.shortest_simple_paths(
                    layered, (source, 0), (destination, 1), weight='weight')
                for path in islice(paths, int(getattr(self, '_path_limit', self.path_candidates))):
                    compute_index = next(i for i in range(len(path) - 1)
                                         if path[i][1] == 0 and path[i + 1][1] == 1)
                    first_path = [node for node, stage in path[:compute_index + 1]]
                    second_path = [node for node, stage in path[compute_index + 1:]]
                    first_cost = sum(layered[a][b]['weight'] for a, b in
                                     zip(path[:compute_index], path[1:compute_index + 1]))
                    second_cost = sum(layered[a][b]['weight'] for a, b in
                                      zip(path[compute_index + 1:], path[compute_index + 2:]))
                    plan = self._make_plan((first_path, first_cost),
                                           (second_path, second_cost), first_path[-1],
                                           False, size, reduced_size, demand)
                    if plan is not None:
                        plans.append(plan)
            except nx.NetworkXNoPath:
                pass
        if plans:
            return plans

        # If the downstream route is blocked, reserve just the next compute
        # stage. The smaller result receives a fresh route after computation.
        if destination in first_graph:
            first_graph.remove_node(destination)
        distances = nx.single_source_dijkstra_path_length(
            first_graph, source,
            weight=lambda left, right, edge: self._edge_delay(left, right, edge, size),
        ) if source in first_graph else {}
        compute_nodes = [node for node in distances
                         if self.graph.nodes[node].get('computing_ability', 0) > 0]
        compute_nodes.sort(key=lambda node: (
            max(distances[node], self._compute_wait(node)) + demand
            / self.graph.nodes[node]['computing_ability'], node))
        for node in compute_nodes[:max(8, self.path_candidates * 4)]:
            for first_option in self._path_options(first_graph, source, node, size):
                plan = self._make_plan(first_option, ([node], 0.0), node,
                                       False, size, reduced_size, demand)
                if plan is not None:
                    plans.append(plan)
        return plans

    def _report_candidate_failure(
        self,
        packet,
        *,
        is_computed,
        compute_nodes,
        first_options,
        second_options,
    ):
        self._candidate_failure_report_count += 1
        if (
            self._candidate_failure_report_count <= 20
            or self._candidate_failure_report_count % 100 == 0
        ):
            print(
                '[P-ICS CANDIDATE]',
                f'source={packet.source}',
                f'destination={packet.destination}',
                f'is_computed={is_computed}',
                f'compute_nodes={compute_nodes}',
                f'first_options={first_options}',
                f'second_options={second_options}',
                f'graph_nodes={len(self.graph.nodes)}',
                f'graph_edges={len(self.graph.edges)}',
                flush=True,
            )

    def _evaluate(self, chromosome, candidates):
        memory = {}
        compute_jobs = {}
        links = {}
        total_delay = 0.0
        total_hops = 0
        memory_penalties = 0
        violations = 0.0
        reward = self.simulator.propagator.reward_function
        for index, candidate_index in enumerate(chromosome):
            plan = candidates[index][candidate_index]
            total_delay += plan['delay']
            total_hops += plan['hops']
            # CPU order depends on arrival at the node, not batch priority.
            for node in plan['computing']:
                compute_jobs.setdefault(node, []).append((
                    plan['compute_arrival'], index, plan['compute_ready'],
                    plan['compute_service'],
                ))
            for (left, right), amount in plan['links'].items():
                total_delay += links.get((left, right), 0) / self.simulator.satellites[left].transmission_rate
            for node, amount in plan['forward_payloads']:
                occupied = (self.graph.nodes[node].get('current_memory_occupy', 0)
                            + self.book.memory_for(node) + memory.get(node, 0) + amount)
                memory_penalties += (1 - occupied / self.simulator.memory < reward.memory_threshold)
            for node, amount in plan['memory'].items():
                memory[node] = memory.get(node, 0) + amount
            for edge, amount in plan['links'].items():
                links[edge] = links.get(edge, 0) + amount

        for jobs in compute_jobs.values():
            available = max(job[2] for job in jobs)
            for arrival, _, ready, service in sorted(jobs):
                finish = max(arrival, available) + service
                # Individual plan delay already includes existing CPU work.
                total_delay += finish - (max(arrival, ready) + service)
                available = finish

        for node, amount in memory.items():
            occupied = (
                self.graph.nodes[node].get('current_memory_occupy', 0)
                + self.book.memory_for(node)
                + amount
            )
            if occupied > self.simulator.memory:
                violations += (occupied - self.simulator.memory) / self.simulator.memory + 1.0
        estimated_reward = (len(chromosome) * reward.reach_factor
                            - reward.delay_factor * total_delay
                            - reward.memory_factor * memory_penalties
                            - reward.hop_factor * total_hops)
        # Capacity feasibility comes first; fewer hops break equal reward/delay ties.
        score = (violations, -estimated_reward, total_delay, total_hops)
        return score, violations == 0.0

    def _greedy_initial_chromosome(self, candidates):
        """Build one low-cost chromosome while preserving feasible prefixes."""
        chromosome = []
        for task_candidates in candidates:
            ranked = []
            for candidate_index in range(len(task_candidates)):
                score, feasible = self._evaluate(
                    chromosome + [candidate_index],
                    candidates,
                )
                ranked.append((not feasible, score, candidate_index))
            chromosome.append(min(ranked)[2])
        return chromosome

    def _run_ga(self, candidates, seed):
        rng = random.Random(seed)
        task_count = len(candidates)
        score_cache = {}

        def evaluate(chromosome):
            key = tuple(chromosome)
            if key not in score_cache:
                score_cache[key] = self._evaluate(chromosome, candidates)
            return score_cache[key]

        # The normal load produces one or two pending tasks per planning
        # cycle.  Enumerating those small candidate spaces is both exact and
        # much cheaper than restarting the configured GA for every cycle.
        combination_count = 1
        for options in candidates:
            combination_count *= len(options)
            if combination_count > 4096:
                break
        if combination_count <= 4096:
            best_feasible = None
            best_any = None
            for chromosome in product(*(range(len(options)) for options in candidates)):
                score, feasible = evaluate(chromosome)
                item = (chromosome, score)
                if best_any is None or score < best_any[1]:
                    best_any = item
                if feasible and (
                    best_feasible is None or score < best_feasible[1]
                ):
                    best_feasible = item
            selected = best_feasible or best_any
            return list(selected[0]), best_feasible is not None

        population = [self._greedy_initial_chromosome(candidates)]
        population.extend(
            [rng.randrange(len(options)) for options in candidates]
            for _ in range(self.population_size - 1)
        )
        best_score = None
        stalled_generations = 0
        stall_limit = min(12, max(3, self.generations // 10))
        for _ in range(self.generations):
            population.sort(key=lambda chromosome: evaluate(chromosome)[0])
            current_score, current_feasible = evaluate(population[0])
            if best_score is None or current_score < best_score:
                best_score = current_score
                stalled_generations = 0
            else:
                stalled_generations += 1
            if current_feasible and stalled_generations >= stall_limit:
                break
            next_population = population[:2]
            while len(next_population) < self.population_size:
                parent_pool = rng.sample(population, min(3, len(population)))
                parent1 = min(parent_pool, key=lambda chromosome: evaluate(chromosome)[0])
                parent_pool = rng.sample(population, min(3, len(population)))
                parent2 = min(parent_pool, key=lambda chromosome: evaluate(chromosome)[0])
                if task_count > 1:
                    cut = rng.randrange(1, task_count)
                    child = parent1[:cut] + parent2[cut:]
                else:
                    child = list(parent1)
                for index, options in enumerate(candidates):
                    if rng.random() < self.mutation_rate:
                        child[index] = rng.randrange(len(options))
                next_population.append(child)
            population = next_population
        evaluated = [
            (chromosome, *evaluate(chromosome))
            for chromosome in population
        ]
        feasible = [item for item in evaluated if item[2]]
        if feasible:
            best, _, _ = min(feasible, key=lambda item: item[1])
            return best, True
        best, _, _ = min(evaluated, key=lambda item: item[1])
        return best, False

    def _release_old_plan(self, packet):
        old_route = list(packet.routing) if packet.routing else packet.routing
        old_compute = packet.computing_node
        old_epoch = getattr(packet, 'tradition_last_plan_epoch', None)
        old_entry = self.book.release(packet)
        if (
            old_compute is not None
            and not packet.information[0]
            and old_compute in self.graph.nodes
        ):
            self.graph.nodes[old_compute]['computing_remain'] = max(
                self.graph.nodes[old_compute].get('computing_remain', 0)
                - int(packet.information[2]),
                0,
            )
        return old_route, old_compute, old_epoch, old_entry

    def _restore_old_plan(self, packet, old_state):
        old_route, old_compute, old_epoch, old_entry = old_state
        if old_entry is not None and self.book.restore(packet, old_entry):
            if (
                old_compute is not None
                and not packet.information[0]
                and old_compute in self.graph.nodes
            ):
                self.graph.nodes[old_compute]['computing_remain'] = (
                    self.graph.nodes[old_compute].get('computing_remain', 0)
                    + int(packet.information[2])
                )
            packet.routing = old_route
            packet.computing_node = old_compute
            packet.tradition_last_plan_epoch = old_epoch
            return
        packet.routing = None
        packet.computing_node = None
        packet.tradition_last_plan_epoch = None

    def replan(self):
        pending = []
        for satellite in sorted(self.simulator.satellites.values(), key=lambda item: item.name):
            for packet in list(satellite.forward_queue.items):
                if packet.destination != satellite.name:
                    # Local computation is already executable. Replanning it
                    # can offload it again before the forward consumer runs.
                    if not packet.information[0] and packet.computing_node == satellite.name:
                        continue
                    if (getattr(packet, 'tradition_topology_version', None)
                            != getattr(self.simulator.propagator, 'topology_version', 0)):
                        self._release_old_plan(packet)
                        packet.routing = None
                        packet.computing_node = None
                        packet.tradition_last_plan_epoch = None
                    elif (packet.routing and packet.routing[0] in satellite.neighbors
                          and self.book._entry_for(packet) is not None):
                        # Committed, executable routes need no new batch slot.
                        # Fault/stale routes are invalidated above or by defer().
                        continue
                    pending.append((satellite, packet))
        def priority(item):
            satellite, packet = item
            age = max(self.simulator.env.now - packet.creation_time, 0.0)
            pressure = min(self.graph.nodes[satellite.name].get(
                'current_memory_occupy', 0.0) / self.simulator.memory, 1.0)
            released_fraction = max(1.0 - packet.information[3] / packet.size, 0.0)
            benefit = (1.0 if packet.information[0]
                       else pressure * released_fraction)
            return -(age + self.priority_age_window * benefit), packet.creation_time

        pending.sort(key=priority)
        live_graph = nx.subgraph_view(
            self.graph,
            filter_node=lambda name: name in self.simulator.satellites
            and self.simulator.satellites[name].active,
            filter_edge=lambda left, right: not self.graph[left][right].get('missing', 0),
        )
        components = {
            node: index
            for index, group in enumerate(nx.connected_components(live_graph))
            for node in group
        }
        pending = [
            (satellite, packet) for satellite, packet in pending
            if not packet.information[0] or (
                satellite.name in components
                and packet.destination in components
                and components[satellite.name] == components[packet.destination]
            )
        ][: self.max_batch_tasks]
        self._pending_report_count += 1
        report_pending = (
            self._pending_report_count <= 20
            or self._pending_report_count % 100 == 0
        )
        if report_pending:
            print(
                f'[P-ICS GA] pending={len(pending)}',
                flush=True,
            )
        if not pending:
            return

        old_states = {
            id(packet): self._release_old_plan(packet)
            for _, packet in pending
        }
        tasks = []
        self._path_limit = self.path_candidates
        try:
            for satellite, packet in pending:
                candidates = self._candidate_plans(satellite, packet)
                candidates = [plan for plan in candidates
                              if packet.hops + len(plan['route']) <= 2 * satellite.max_hop]
                if candidates:
                    tasks.append((satellite, packet, candidates))
                else:
                    self._restore_old_plan(packet, old_states[id(packet)])
        finally:
            self._path_limit = self.path_candidates
        if not tasks:
            if report_pending:
                print('[P-ICS GA] candidate_tasks=0', flush=True)
            return

        candidates = [item[2] for item in tasks]
        seed = 'pics-batch-ga:' + ':'.join(
            f'{item[1].source}:{item[1].creation_time}' for item in tasks
        )
        chromosome, feasible = self._run_ga(candidates, seed)
        self._ga_report_count += 1
        if self._ga_report_count <= 20 or self._ga_report_count % 100 == 0:
            print(
                f'[P-ICS GA] tasks={len(tasks)} feasible={feasible}',
                flush=True,
            )
        epoch = int(getattr(self.simulator.propagator, 'pics_state_version', 0))
        for index, (satellite, packet, task_candidates) in enumerate(tasks):
            if feasible:
                selected = chromosome[index]
            else:
                # Commit a feasible subset instead of stalling the entire
                # batch behind one congested task. Each accepted plan updates
                # the reservation book before the next task is evaluated.
                choice, task_feasible = self._run_ga([task_candidates], seed)
                if not task_feasible:
                    self._restore_old_plan(packet, old_states[id(packet)])
                    continue
                selected = choice[0]
            plan = task_candidates[selected]
            compute_node = plan['compute_node']
            if compute_node is not None:
                self.graph.nodes[compute_node]['computing_remain'] = (
                    self.graph.nodes[compute_node].get('computing_remain', 0)
                    + int(packet.information[2])
                )
            if not self.book.commit(
                packet,
                plan['route'],
                compute_node,
                packet.size,
                int(packet.information[3]),
                bool(packet.information[0]),
            ):
                if compute_node is not None:
                    self.graph.nodes[compute_node]['computing_remain'] = max(
                        self.graph.nodes[compute_node].get('computing_remain', 0)
                        - int(packet.information[2]),
                        0,
                    )
                self._restore_old_plan(packet, old_states[id(packet)])
                continue
            packet.routing = list(plan['route'])
            packet.computing_node = compute_node
            packet.tradition_last_plan_epoch = epoch
            packet.tradition_topology_version = int(
                getattr(self.simulator.propagator, 'topology_version', 0)
            )
            simulator_module.mark_queue_entry(
                packet,
                'forward',
                self.simulator.env.now,
            )
            if packet not in satellite.forward_queue.items:
                satellite.forward_queue.put(packet)
            self._commit_report_count += 1
            if (
                self._commit_report_count <= 20
                or self._commit_report_count % 100 == 0
            ):
                print(
                    'GA COMMIT',
                    packet.source,
                    packet.destination,
                    packet.routing,
                    packet.computing_node,
                    flush=True,
                )


class PICSOnboardSimulator(simulator_module.SatelliteNetworkSimulator_OnbardComputing):
    """Add a lightweight central replanning clock to the ICS simulator."""

    def generate_fixed_traffic(self, satellite):
        """Admit fixed-destination packets by capacity, even without a route.

        The ICS planner can wait for a route after admission, just like RL.
        Keep this override local to ICS while sharing packet and loss accounting.
        """
        for session in self.fixed_workload_sessions.get(satellite, ()):
            start = float(session['session_start_time'])
            if start > self.env.now:
                yield self.env.timeout(start - self.env.now)
            destination = str(session['destination'])
            if satellite in self.satellites:
                self.satellites[satellite].is_producing = 1
            for item in session['packets']:
                arrival = start + int(item['packet_index']) / self.packet_frequency
                if arrival > self.env.now:
                    yield self.env.timeout(arrival - self.env.now)
                self.workload_trace_count += 1
                self.statics_datas['Total'] += 1
                packet = simulator_module.Packet(
                    satellite, destination, self.env.now, int(item['packet_size']))
                packet.extra_information([
                    False, int(session['task_type']), int(item['computing_demand']),
                    int(item['size_after_computing']), self.env.now, None, None,
                ])
                reason = None
                if satellite not in self.satellites:
                    reason = 'source_or_destination_missing'
                elif destination not in self.satellite_names:
                    reason = 'no_route'
                elif not self.satellites[satellite].push_forward(packet):
                    reason = 'memory_overflow'
                if reason is not None:
                    self.statics_datas['Lost_upload'] += 1
                    simulator_module.record_packet_loss(self.statics_datas, 'upload', reason)
                    self.propagator.finalize_packet_reward(
                        packet, self.propagator.reward_function.loss_reward(
                            self.env.now - packet.creation_time))
            end = float(session['session_end_time'])
            if end > self.env.now:
                yield self.env.timeout(end - self.env.now)
            if satellite in self.satellites:
                self.satellites[satellite].is_producing = 0

    def upgrade_all(self, graph, landmarks, ground_cover_sats, topology_graph=None):
        topology_changed = (
            set(graph.nodes()) != set(self.graph.nodes())
            or set(graph.edges()) != set(self.graph.edges())
        )
        super().upgrade_all(
            graph,
            landmarks,
            ground_cover_sats,
            topology_graph=topology_graph,
        )

        if not hasattr(self.propagator, 'topology_version'):
            self.propagator.topology_version = 0
        if topology_changed:
            self.propagator.topology_version += 1

        # The base simulator replaces its graph object on topology changes,
        # while P-ICS keeps resource reservations across those changes.
        # Rebind both views and restore the live per-satellite counters before
        # the next GA cycle or computing event reads them.
        self.propagator.graph = self.graph
        self.propagator.update(self.graph)
        reservation_book = getattr(self.propagator, 'reservation_book', None)
        if reservation_book is not None:
            reservation_book.graph = self.graph

        for node, satellite in self.satellites.items():
            if node not in self.graph.nodes:
                continue
            node_data = self.graph.nodes[node]
            node_data['current_memory_occupy'] = satellite.current_memory_occupy
            node_data['computing_remain'] = satellite.computing_remain
            if reservation_book is not None:
                # Local queued/active work is already in computing_remain.
                # Keep reservations for tasks still travelling to this node,
                # excluding reservations already in its computing queue.
                queued_reserved = 0.0
                for packet in satellite.computing_queue.items:
                    entry = reservation_book._entry_for(packet)
                    if entry is not None:
                        queued_reserved += entry['computing'].get(node, 0.0)
                node_data['computing_remain'] += (
                    reservation_book.compute_for(node) - queued_reserved
                )
                node_data['reserved_memory'] = reservation_book.memory_for(node)
                node_data['reserved_computing'] = reservation_book.compute_for(node)
                node_data['active_computing'] = reservation_book.active_computing.get(
                    node,
                    0.0,
                )

        if topology_changed and getattr(self.propagator, 'batch_scheduler', None) is not None:
            self.env.process(self._replan_after_topology_change())

    def _replan_after_topology_change(self):
        yield self.env.timeout(0)
        self.propagator.batch_scheduler.replan()

    def run(self, duration):
        if not getattr(self, '_pics_replanner_started', False):
            self._pics_replanner_started = True
            self.propagator.pics_state_version = 0
            self.propagator.topology_version = getattr(
                self.propagator, 'topology_version', 0
            )
            self.propagator.reservation_book = PICSReservationBook(
                self.graph,
                self.memory,
            )
            self.propagator.batch_scheduler = PICSBatchGAScheduler(self)
            self.propagator.before_computation = self._before_computation
            self.env.process(self._periodic_central_replanning())
        return super().run(duration)

    def _periodic_central_replanning(self):
        interval = float(self.tradition_replan_interval or 0.1)
        if interval <= 0:
            raise ValueError('tradition_replan_interval must be positive')
        while True:
            yield self.env.timeout(interval)
            self.propagator.pics_state_version += 1
            self.propagator.batch_scheduler.replan()

    @staticmethod
    def _valid_route(route):
        return route and route != [None]

    def _release_compute_reservation(self, node, demand):
        if node in self.graph.nodes:
            current = self.graph.nodes[node].get('computing_remain', 0)
            self.graph.nodes[node]['computing_remain'] = max(
                current - demand, 0
            )

    def _before_computation(self, satellite, packet):
        information = packet.information
        if len(information) < 7 or information[0]:
            return True
        if getattr(self.propagator, 'batch_scheduler', None) is not None:
            return True
        demand, reduced_size = int(information[2]), int(information[3])
        interval = float(self.tradition_replan_interval or 0.1)
        time_version = int(self.env.now // interval)
        state_version = max(
            time_version,
            int(getattr(self.propagator, 'pics_state_version', 0)),
        )
        last_version = getattr(packet, 'tradition_last_plan_epoch', None)
        if last_version is not None and state_version <= last_version:
            return True

        reservation_book = getattr(self.propagator, 'reservation_book', None)
        old_reservation = (
            reservation_book.release(packet)
            if reservation_book is not None else None
        )
        self._release_compute_reservation(satellite.name, demand)
        current_state = [
            satellite.name, packet.destination, packet.size,
            demand, reduced_size, False,
        ]
        route, compute_node = satellite.tradition_routing(
            current_state, computing=True
        )
        if not self._valid_route(route):
            self.graph.nodes[satellite.name]['computing_remain'] = (
                self.graph.nodes[satellite.name].get('computing_remain', 0)
                + demand
            )
            if reservation_book is not None:
                reservation_book.restore(packet, old_reservation)
            return True
        if reservation_book is not None and not reservation_book.commit(
            packet,
            route,
            compute_node,
            packet.size,
            reduced_size,
            False,
        ):
            if compute_node in self.graph.nodes:
                self._release_compute_reservation(compute_node, demand)
            self.graph.nodes[satellite.name]['computing_remain'] = (
                self.graph.nodes[satellite.name].get('computing_remain', 0)
                + demand
            )
            reservation_book.restore(packet, old_reservation)
            return True
        packet.routing = route
        packet.tradition_last_plan_epoch = state_version
        packet.tradition_topology_version = int(
            getattr(self.propagator, 'topology_version', 0)
        )
        if compute_node == satellite.name:
            packet.computing_node = None
            return True
        satellite.current_computing_queue_size -= packet.size
        satellite.computing_remain -= demand
        packet.computing_node = compute_node
        simulator_module.mark_queue_entry(packet, 'forward', self.env.now)
        satellite.forward_queue.put(packet)
        return False


def install_pics_mode():
    """Install P-ICS classes only inside the worker running this test."""
    simulator_module.Satellite_with_Computing = PICSSatellite
    environment_module.SatelliteNetworkSimulator_OnbardComputing = (
        PICSOnboardSimulator
    )

# Base configuration comes from the fixed test contract. Only the P-ICS
# routing mode and its periodic replanning policy are defined here.
FIXED_CONFIG = {
    'general': copy.deepcopy(COMMON_TEST['general']),
    'agent': {
        'mode': 'Tradition',
        'neighbors_dim': 60,
        'edges_dim': 4,
        'distance_dim': 4,
        'mission_dim': 6,
        'current_dim': 4,
        'action_dim': 5,
        'hidden_dim': 256,
        'buffer_length': 30000,
        'batch_size': 1024,
        'gamma': 0.99,
        'q_mask': -1,
        'activation': 'LeakyRelu',
        'hidden_layers': 2,
        'dueling': True,
        'shuffle': True,
        'learning_rate': 0.0002,
        'repeat': 1,
        'update_cycle': 240,
        'model_path': '',
    },
    'environment': copy.deepcopy(COMMON_TEST['environment']),
}
FIXED_CONFIG['general']['algorithm_name'] = 'ICS'
FIXED_CONFIG['environment'].update({
    'computing_ability': None,
    'pole': False,
    'tradition_replan_interval': 0.1,
    'tradition_use_genetic_search': False,
    'tradition_search_method': 'batch_ga',
    'tradition_ga_population_size': 50,
    'tradition_ga_generations': 100,
    'tradition_ga_path_candidates': 3,
    'tradition_ga_mutation_rate': 0.1,
})

LOAD_SETTINGS = tuple(LOAD_TESTS['cases'])
FAULT_SETTINGS = tuple(FAULT_TESTS['cases'])
FAULT_LOAD_SETTING = dict(FAULT_TESTS['workload'])
RESULT_FIELDS = (
    'algorithm',
    'test_type',
    'test_case_id',
    'target_load_tasks_per_second',
    'poisson_rate_s',
    'poisson_rate_g',
    'packet_frequency',
    'scenario_seed',
    'fault_percent',
    'edge_p',
    'edge_q',
    'fault_fill_multiplier',
    'replan_interval_seconds',
    'search_method',
    'ga_population_size',
    'ga_generations',
    'ga_path_candidates',
    'ga_mutation_rate',
    'generated_tasks',
    'actual_tasks_per_second',
    'reached_tasks',
    'lost_tasks',
    'unfinished_tasks',
    'finished_tasks_percent',
    'lost_generated_percent',
    'packet_loss_rate_percent',
    'loss_breakdown_json',
    'loss_breakdown_total',
    *(key.lower() for key in simulator_module.LOSS_BREAKDOWN_KEYS),
    'average_delay_seconds',
    'average_computation_delay_seconds',
    'average_queueing_delay_seconds',
    'average_propagation_delay_seconds',
    'average_transmission_delay_seconds',
    'average_delay_component_sum_seconds',
    'average_delay_accounting_gap_seconds',
    'average_hop_count',
    'average_computing_percent',
    'average_waiting_seconds',
    'average_ending_reward',
    'workload_trace_count',
    'workload_trace_sha256',
    'fault_trace_count',
    'fault_trace_sha256',
    'raw_result_file',
)


def load_base_config():
    tle_path = (PROJECT_ROOT / SATELLITE_TLE_FILE).resolve()
    ground_path = (PROJECT_ROOT / GROUND_STATION_FILE).resolve()
    ability_path = (PROJECT_ROOT / COMPUTING_ABILITY_FILE).resolve()
    for path in (tle_path, ground_path, ability_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    expected_ability_path = tle_path.with_name(f'{tle_path.stem}_computing_ability.csv')
    if ability_path != expected_ability_path:
        raise ValueError('Computing-ability file does not match the configured TLE file')
    if FIXED_CONFIG['environment']['tle_filepath'] != SATELLITE_TLE_FILE:
        raise ValueError('Test config TLE does not match test_ICS.py')
    if FIXED_CONFIG['environment']['SOD_file_path'] != GROUND_STATION_FILE:
        raise ValueError('Test config ground-station file does not match test_ICS.py')
    if SCENARIO_MANIFEST['tle_filepath'] != SATELLITE_TLE_FILE:
        raise ValueError('Scenario manifest TLE does not match test_ICS.py')
    if int(SCENARIO_MANIFEST['satellite_count']) != satellite_count(FIXED_CONFIG):
        raise ValueError('Scenario manifest satellite count does not match the TLE')
    for path, key in ((tle_path, 'tle_sha256'), (ability_path, 'computing_ability_sha256')):
        if hashlib.sha256(path.read_bytes()).hexdigest() != SCENARIO_MANIFEST[key]:
            raise ValueError(f'Scenario resource SHA256 mismatch: {path}')
    if int(SCENARIO_MANIFEST['duration']) != int(FIXED_CONFIG['general']['duration']):
        raise ValueError('Scenario duration does not match the common test config')
    from train_ISACR_self_elimination import CONFIG as training_config
    for key in ('reward_factors', 'select_mode', 'begin_time', 'time_stride'):
        if json.loads(json.dumps(training_config['general'][key])) != FIXED_CONFIG['general'][key]:
            raise ValueError(f'ICS/common training setting mismatch: general.{key}')
    for key, value in COMMON_TEST['environment'].items():
        if key in {'computing_ability', 'visualize', 'print_info', 'save_log', 'show_detail'}:
            continue
        if json.loads(json.dumps(training_config['environment'].get(key))) != value:
            raise ValueError(f'ICS/common training setting mismatch: environment.{key}')
    return copy.deepcopy(FIXED_CONFIG)


def build_config(base_config, test_type, setting):
    config = copy.deepcopy(base_config)
    workload_entry, fault_entry = case_trace_entries(test_type, setting)
    config['environment'].update({
        'workload_trace_path': str(resolve_trace(workload_entry)),
        'workload_trace_sha256': workload_entry['sha256'],
        'workload_trace_task_count': int(workload_entry['task_count']),
        'fault_trace_path': str(resolve_trace(fault_entry)),
        'fault_trace_sha256': fault_entry['sha256'],
        'fault_trace_event_count': int(fault_entry['event_count']),
    })
    if test_type == 'load':
        config['environment'].update(setting)
        config['environment'].update({
            'poisson_rate_g': setting['poisson_rate_g'],
            'poisson_rate_s': setting['poisson_rate_s'],
            'edge_p': LOAD_TESTS['edge_p'],
            'edge_q': LOAD_TESTS['edge_q'],
            'save_training_data': (
                f'v1/test/test/ICS/raw/ICS_load_'
                f'{setting["target_load_tasks_per_second"]}.txt'
            ),
        })
    else:
        config['environment'].update(FAULT_LOAD_SETTING)
        config['environment'].update({
            'id': f'load_{int(FAULT_LOAD_SETTING["target_load_tasks_per_second"])}',
            'poisson_rate_g': FAULT_LOAD_SETTING['poisson_rate_g'],
            'poisson_rate_s': FAULT_LOAD_SETTING['poisson_rate_s'],
            'edge_p': setting['edge_p'],
            'edge_q': setting['edge_q'],
            'save_training_data': (
                f'v1/test/test/ICS/raw/ICS_fault_'
                f'{setting["fault_percent"]}.txt'
            ),
        })
    config['test_contract'] = {
        'common_config': str((CONFIG_ROOT / 'common.json').resolve()),
        'load_config': str((CONFIG_ROOT / 'load_experiments.json').resolve()),
        'fault_config': str((CONFIG_ROOT / 'fault_experiments.json').resolve()),
        'scenario_manifest': str(
            (CONFIG_ROOT / f"scenario_seed_{COMMON_TEST['general']['scenario_seed']}" / 'manifest.json').resolve()
        ),
        'workload_trace': workload_entry,
        'fault_trace': fault_entry,
        'replan_interval_seconds': config['environment']['tradition_replan_interval'],
        'objective': 'capacity_feasible_then_max_reward_then_delay_then_hops',
        'reward_factors': config['general']['reward_factors'],
        'python_hash_seed': str(COMMON_TEST['python_hash_seed']),
        'runtime_acceleration': runtime_acceleration_contract(),
        'source_and_resource_sha256': {
            **training_input_hashes(config),
            'test_ICS.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    return config


def parse_optional_number(text, label):
    number = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    match = re.search(rf'{re.escape(label)}: ({number}|None)', text)
    if not match or match.group(1) == 'None':
        return None
    return float(match.group(1))


def parse_optional_text(text, label):
    match = re.search(rf'{re.escape(label)}: (.+)', text)
    return match.group(1).strip() if match else None


def parse_result(raw_path, config, test_type, setting, satellite_count):
    content = raw_path.read_text(encoding='utf-8')
    snapshots = []
    rewards = []
    for block in content.split('====== step'):
        statics_match = re.search(r'current_statics: ({.*})', block)
        if not statics_match:
            continue
        snapshot = ast.literal_eval(statics_match.group(1))
        snapshots.append(snapshot)
        reward = parse_optional_number(block, 'Average ending reward')
        if reward is not None:
            rewards.append(reward)
    if not snapshots:
        raise RuntimeError(f'No periodic statistics found in {raw_path}')

    totals = {
        key: sum(snapshot.get(key, 0) for snapshot in snapshots)
        for key in snapshots[0]
    }
    reached = totals['Reached_0'] + totals['Reached_1']
    lost = totals['Lost_upload'] + totals['Lost_relay_0'] + totals['Lost_relay_1']
    finished = reached + lost
    duration = config['general']['duration']
    loss_breakdown = {
        key: totals[key]
        for key in sorted(totals)
        if key.startswith('Lost_')
        and key not in {'Lost_upload', 'Lost_relay_0', 'Lost_relay_1'}
    }

    def divide(numerator, denominator):
        return numerator / denominator if denominator else None

    components = {
        name: divide(
            totals.get(f'Total_{name}_delay_0', 0)
            + totals.get(f'Total_{name}_delay_1', 0),
            reached,
        )
        for name in ('computation', 'queueing', 'propagation', 'transmission')
    }
    average_delay = divide(
        totals['Total_delay_0'] + totals['Total_delay_1'], reached
    )
    component_sum = (
        sum(components.values())
        if reached and all(value is not None for value in components.values())
        else None
    )
    load_setting = setting if test_type == 'load' else FAULT_LOAD_SETTING
    return {
        'algorithm': 'ICS',
        'test_type': test_type,
        'test_case_id': setting['id'],
        'target_load_tasks_per_second': load_setting['target_load_tasks_per_second'],
        'poisson_rate_s': load_setting['poisson_rate_s'],
        'poisson_rate_g': load_setting['poisson_rate_g'],
        'packet_frequency': config['environment']['packet_frequency'],
        'scenario_seed': COMMON_TEST['scenario_seeds'][0],
        'fault_percent': (
            int(round(float(LOAD_TESTS['edge_p']) * 100))
            if test_type == 'load' else setting['fault_percent']
        ),
        'edge_p': config['environment']['edge_p'],
        'edge_q': config['environment']['edge_q'],
        'fault_fill_multiplier': config['environment']['fault_fill_multiplier'],
        'replan_interval_seconds': config['environment']['tradition_replan_interval'],
        'search_method': 'batch_ga',
        'ga_population_size': config['environment']['tradition_ga_population_size'],
        'ga_generations': config['environment']['tradition_ga_generations'],
        'ga_path_candidates': config['environment']['tradition_ga_path_candidates'],
        'ga_mutation_rate': config['environment']['tradition_ga_mutation_rate'],
        'generated_tasks': totals['Total'],
        'actual_tasks_per_second': divide(totals['Total'], duration),
        'reached_tasks': reached,
        'lost_tasks': lost,
        'unfinished_tasks': totals['Total'] - reached - lost,
        'finished_tasks_percent': divide(finished * 100, totals['Total']),
        'lost_generated_percent': divide(lost * 100, totals['Total']),
        'packet_loss_rate_percent': divide(lost * 100, finished),
        'loss_breakdown_json': json.dumps(loss_breakdown, sort_keys=True),
        'loss_breakdown_total': sum(loss_breakdown.values()),
        **{
            key.lower(): totals.get(key, 0)
            for key in simulator_module.LOSS_BREAKDOWN_KEYS
        },
        'average_delay_seconds': average_delay,
        'average_computation_delay_seconds': components['computation'],
        'average_queueing_delay_seconds': components['queueing'],
        'average_propagation_delay_seconds': components['propagation'],
        'average_transmission_delay_seconds': components['transmission'],
        'average_delay_component_sum_seconds': component_sum,
        'average_delay_accounting_gap_seconds': (
            average_delay - component_sum
            if average_delay is not None and component_sum is not None else None
        ),
        'average_hop_count': divide(
            totals['Total_hops_0'] + totals['Total_hops_1'], reached
        ),
        'average_computing_percent': divide(
            totals['Is_computing'] * 100,
            satellite_count * duration,
        ),
        'average_waiting_seconds': divide(totals['Computing_waiting_time'], reached),
        'average_ending_reward': divide(sum(rewards), len(rewards)),
        'workload_trace_count': parse_optional_number(content, 'Workload trace count'),
        'workload_trace_sha256': parse_optional_text(content, 'Workload trace SHA256'),
        'fault_trace_count': parse_optional_number(content, 'Fault trace count'),
        'fault_trace_sha256': parse_optional_text(content, 'Fault trace SHA256'),
        'raw_result_file': str(raw_path),
    }


def satellite_count(config):
    tle_path = (PROJECT_ROOT / config['environment']['tle_filepath']).resolve()
    return sum(
        line.startswith('1 ')
        for line in tle_path.read_text(encoding='utf-8').splitlines()
    )


def plot_results(rows):
    paths = []
    figures = (
        ('average_delay_seconds', 'Average delay (s)', 'delay'),
        ('packet_loss_rate_percent', 'Packet loss rate (%)', 'packet_loss'),
        ('average_hop_count', 'Average hop count', 'hops'),
    )
    for test_type, x_field, x_label, prefix in (
        ('load', 'target_load_tasks_per_second', 'Target whole-network load (tasks/s)', 'load'),
        ('fault', 'fault_percent', 'Edge fault rate (%)', 'fault'),
    ):
        subset = sorted(
            (row for row in rows if row['test_type'] == test_type),
            key=lambda row: row[x_field],
        )
        x_values = [row[x_field] for row in subset]
        for metric, ylabel, metric_name in figures:
            fig, axis = plt.subplots(figsize=(5, 3.5), dpi=200)
            axis.plot(
                x_values,
                [row[metric] for row in subset],
                marker='o',
                color='#1f77b4',
            )
            axis.set_xlabel(x_label)
            axis.set_ylabel(ylabel)
            axis.set_xticks(x_values)
            axis.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            fig.tight_layout()
            path = OUTPUT_DIR / f'ICS_{prefix}_{metric_name}.png'
            fig.savefig(path, bbox_inches='tight')
            plt.close(fig)
            paths.append(path)
    return paths


def write_results(rows):
    ordered_rows = sorted(rows, key=lambda row: (
        0 if row['test_type'] == 'load' else 1,
        row['target_load_tasks_per_second']
        if row['test_type'] == 'load' else row['fault_percent'],
    ))
    result_path = OUTPUT_DIR / 'PICS_test_results.csv'
    with result_path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(ordered_rows)
    return result_path


def run_experiment(test_type, setting):
    status_dir = OUTPUT_DIR / 'statuses'
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / f'{test_type}_{setting["id"]}.json'
    status = {'test_type': test_type, 'test_case_id': setting['id'], 'status': 'running'}
    status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
    raw_path = None
    try:
        os.chdir(PROJECT_ROOT)
        base_config = load_base_config()
        count = satellite_count(base_config)
        config = build_config(base_config, test_type, setting)
        install_pics_mode()
        case_label = (
            f'load={setting["target_load_tasks_per_second"]}'
            if test_type == 'load'
            else f'fault={setting["fault_percent"]}%'
        )
        raw_path = PROJECT_ROOT / 'training_process_data' / config['environment']['save_training_data']
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'configs').mkdir(parents=True, exist_ok=True)
        archive_test_outputs((raw_path,))
        archive_test_outputs((OUTPUT_DIR / 'configs' / (raw_path.stem + '.json'),))
        (OUTPUT_DIR / 'configs' / (raw_path.stem + '.json')).write_text(
            json.dumps(config, indent=2), encoding='utf-8',
        )
        print(
            f'\n===== P-ICS batch GA search, '
            f'replan={config["environment"]["tradition_replan_interval"]}s, {case_label}, '
            f'poisson_rate_g={config["environment"]["poisson_rate_g"]}, '
            f'poisson_rate_s={config["environment"]["poisson_rate_s"]}, '
            f'edge_p={config["environment"]["edge_p"]} =====',
            flush=True,
        )
        run_training(config)
        content = raw_path.read_text(encoding='utf-8')
        steps = re.findall(r'====== step (\d+)', content)
        final_step = int(steps[-1]) if steps else 0
        expected_steps = int(config['general']['duration'] / config['general']['time_stride'])
        if final_step != expected_steps or 'Fixed test trace replay: verified' not in content:
            raise RuntimeError(f'Incomplete ICS case: step {final_step}/{expected_steps}')
        row = parse_result(raw_path, config, test_type, setting, count)
    except Exception:
        detail = traceback.format_exc()
        steps = (
            re.findall(r'====== step (\d+)', raw_path.read_text(encoding='utf-8'))
            if raw_path is not None and raw_path.exists() else []
        )
        status.update(status='failed', final_step=int(steps[-1]) if steps else 0,
                      traceback=detail)
        status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
        print(f'FAILED P-ICS {test_type} {setting["id"]}:\n{detail}', flush=True)
        raise
    status.update(status='complete', final_step=final_step, raw_result_file=str(raw_path))
    status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')
    return row


def publish_intersection(load_row):
    """Publish the shared load/fault point without another simulation."""
    setting = LOAD_FAULT_CASE
    config = build_config(load_base_config(), 'fault', setting)
    source = Path(load_row['raw_result_file'])
    target = PROJECT_ROOT / 'training_process_data' / config['environment']['save_training_data']
    workload, fault = case_trace_entries('fault', setting)
    if (load_row['workload_trace_sha256'] != workload['sha256']
            or load_row['fault_trace_sha256'] != fault['sha256']):
        raise ValueError('ICS intersection trace mismatch')
    archive_test_outputs((target,))
    archive_test_outputs((OUTPUT_DIR / 'configs' / (target.stem + '.json'),))
    shutil.copy2(source, target)
    (OUTPUT_DIR / 'configs' / (target.stem + '.json')).write_text(
        json.dumps(config, indent=2), encoding='utf-8',
    )
    row = parse_result(target, config, 'fault', setting, satellite_count(config))
    (OUTPUT_DIR / 'statuses' / f'fault_{setting["id"]}.json').write_text(
        json.dumps(dict(test_type='fault', test_case_id=setting['id'],
                        status='complete', final_step=int(config['general']['duration']
                                                         / config['general']['time_stride']),
                        raw_result_file=str(target), reused_from=str(source)), indent=2),
        encoding='utf-8',
    )
    return row


def main():
    os.chdir(PROJECT_ROOT)
    (OUTPUT_DIR / 'raw').mkdir(parents=True, exist_ok=True)
    experiments = [
        ('load', setting) for setting in sorted(
            LOAD_SETTINGS, key=lambda item: item['target_load_tasks_per_second'], reverse=True,
        )
    ] + [
        ('fault', setting) for setting in sorted(
            FAULT_SETTINGS, key=lambda item: item['edge_p'], reverse=True,
        ) if setting['id'] != LOAD_FAULT_CASE['id']
    ]
    rows = []
    failures = {}
    batch_status = {'status': 'running', 'expected_cases': len(LOAD_SETTINGS) + len(FAULT_SETTINGS),
                    'expected_simulations': len(experiments),
                    'max_workers': 7,
                    'completed_cases': 0, 'failed_cases': 0, 'errors': failures}
    status_path = OUTPUT_DIR / 'TEST_STATUS.json'
    status_path.write_text(json.dumps(batch_status, indent=2), encoding='utf-8')
    # Submit descending cases; finish order depends on simulation runtime.
    with ProcessPoolExecutor(max_workers=min(7, len(experiments))) as executor:
        futures = {
            executor.submit(run_experiment, test_type, setting): (test_type, setting)
            for test_type, setting in experiments
        }
        for future in as_completed(futures):
            test_type, setting = futures[future]
            try:
                row = future.result()
            except Exception as error:
                failures[f'{test_type}/{setting["id"]}'] = str(error)
                print(f'FAILED P-ICS {test_type} {setting["id"]}: {error}', flush=True)
            else:
                rows.append(row)
                if (test_type == 'load'
                        and setting['target_load_tasks_per_second']
                        == FAULT_LOAD_SETTING['target_load_tasks_per_second']):
                    try:
                        rows.append(publish_intersection(row))
                    except Exception as error:
                        failures[f'fault/{LOAD_FAULT_CASE["id"]}'] = str(error)
                result_path = write_results(rows)
                print(
                    f'Completed P-ICS {row["test_type"]} case {row["test_case_id"]}; '
                    f'partial CSV: {result_path}',
                    flush=True,
                )
            batch_status.update(completed_cases=len(rows), failed_cases=len(failures))
            status_path.write_text(json.dumps(batch_status, indent=2), encoding='utf-8')

    batch_status['status'] = 'failed' if failures else 'complete'
    status_path.write_text(json.dumps(batch_status, indent=2), encoding='utf-8')
    if failures:
        raise RuntimeError(f'{len(failures)} ICS cases failed; see {status_path}')
    result_path = write_results(rows)
    figure_paths = plot_results(rows)
    print(f'\nCSV: {result_path}')
    for figure_path in figure_paths:
        print(f'Figure: {figure_path}')


def run_fixed_config_entrypoint():
    hash_seed = str(COMMON_TEST['python_hash_seed'])
    if os.environ.get('PYTHONHASHSEED') != hash_seed:
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        return subprocess.call(
            [sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=PROJECT_ROOT, env=env,
        )
    main()
    return 0


if __name__ == '__main__':
    raise SystemExit(run_fixed_config_entrypoint())
