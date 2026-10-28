"""GSJF load/fault evaluation.

GSJF (Greedy Satellite Job First) compares local and legal one-hop computation
completion estimates, computes jobs in arrival order without preemption,
and returns results by legal minimum-hop forwarding. Transmission remains FIFO.

Examples:
    python test_GCSR.py --dry-run
    python test_GCSR.py --experiment load --load-point 160
    python test_GCSR.py --experiment fault --fault-percent 18
    python test_GCSR.py --strategy one-hop
    python test_GCSR.py --strategy multi-hop
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent

# Match the fixed scenario library used by the four-algorithm, ablation, and
# ICS test entry points before importing the shared baseline module.
CONFIG_ROOT = PROJECT_ROOT / 'training_process_data' / 'v1' / 'test' / 'test' / 'config'
os.environ.setdefault('SATELLITE_TEST_CONFIG_ROOT', str(CONFIG_ROOT))
import test_4_5seed as baseline
from training_runner import run_training, runtime_acceleration_contract


OUTPUT_ROOT = PROJECT_ROOT / 'training_process_data' / 'v1' / 'test' / 'test' / 'GSJF'
LOCK_PATH = OUTPUT_ROOT / 'GSJF_TEST_RUNNING.lock'
SEEDS = baseline.DEFAULT_SEEDS
LOAD_SETTINGS = baseline.LOAD_SETTINGS
FAULT_PROBABILITIES = baseline.FAULT_PROBABILITIES
LOAD_FAULT_PROBABILITY = baseline.LOAD_FAULT_PROBABILITY
FAULT_LOAD_SETTING = baseline.FAULT_LOAD_SETTING
FAULT_CASE_BY_PROBABILITY = baseline.FAULT_CASE_BY_PROBABILITY
FAULT_WORKLOAD_CASE = baseline.FAULT_WORKLOAD_CASE
DURATION = baseline.DURATION
EXPECTED_STEP = baseline.EXPECTED_TEST_STEP
MAX_WORKERS = min(baseline.MAX_WORKERS, 10)
DEFAULT_WORKERS = min(MAX_WORKERS, 2)


@dataclass(frozen=True)
class ExecutionEstimate:
    queueing: float
    computation: float
    send_waiting: float = 0.0
    transmission: float = 0.0
    propagation: float = 0.0

    @property
    def total(self):
        return (self.queueing + self.computation + self.send_waiting
                + self.transmission + self.propagation)


def _positive(value):
    return max(float(value or 0.0), 0.0)


def estimate_computation(node, demand, now=0.0):
    """Estimate queueing and task computation time at one satellite."""
    ability = _positive(node.get('computing_ability', 0.0))
    if ability <= 0:
        return ExecutionEstimate(math.inf, math.inf)
    remain = _positive(node.get('computing_remain', 0.0))
    active = node.get('active_computation_started_at')
    # A live task's elapsed portion is excluded from the remaining queue.
    if active is not None:
        remain = max(0.0, remain - _positive(now - active) * ability)
    return ExecutionEstimate(remain / ability, _positive(demand) / ability)


def estimate_neighbor(node, demand, packet_size, transmission_rate,
                      send_waiting=0.0, propagation=0.0, now=0.0):
    """Estimate one-hop send, propagation, queueing and compute delay."""
    compute = estimate_computation(node, demand, now)
    rate = _positive(transmission_rate)
    transmission = math.inf if rate <= 0 else _positive(packet_size) / rate
    return ExecutionEstimate(
        compute.queueing, compute.computation, _positive(send_waiting),
        transmission, _positive(propagation),
    )


def _edge_cost(graph, left, right, packet_size, transmission_rate):
    # Missing physical state must not become an oracle-based exclusion or
    # infinite penalty before heartbeat detection.  Use the known packet
    # transmission time when the graph no longer has the link attributes.
    edge = graph.get_edge_data(left, right, default={})
    rate = _positive(transmission_rate)
    queued = _positive(edge.get('transmission_weight', 0.0))
    tx = (queued + _positive(packet_size)) / rate if rate else math.inf
    return tx + _positive(edge.get('propagation_weight', 0.0)) + _positive(edge.get('gosr_waiting', 0.0))


def greedy_min_hop_route(graph, source, target, packet_size, transmission_rate,
                   memory=None, active_nodes=None, max_hops=None,
                   forbidden_nodes=()):
    """Return a route by repeatedly selecting a minimum-hop next neighbor."""
    if source == target:
        return []
    nodes = graph.graph.get('_gosr_nodes')
    if nodes is None:
        nodes = frozenset(graph.nodes)
    feasible_nodes = (nodes if active_nodes is None or active_nodes is nodes
                      else nodes.intersection(active_nodes))
    blocked_nodes = set() if feasible_nodes is nodes else set(nodes - feasible_nodes)
    if memory is not None:
        occupied = graph.graph.get('_gosr_memory')
        if occupied is None:
            occupied = {node: _positive(graph.nodes[node].get('current_memory_occupy', 0.0))
                        for node in feasible_nodes}
        size, capacity = _positive(packet_size), _positive(memory)
        excluded = (feasible_nodes - {source} if size > capacity else {
            node for node, amount in occupied.items()
            if node != source and amount + size > capacity})
        if excluded:
            feasible_nodes = feasible_nodes.difference(excluded)
            blocked_nodes.update(excluded)
    if source not in feasible_nodes or target not in feasible_nodes:
        return None
    forbidden_nodes = set(forbidden_nodes) - {source, target}
    if forbidden_nodes:
        feasible_nodes = feasible_nodes.difference(forbidden_nodes)
        blocked_nodes.update(nodes.intersection(forbidden_nodes))
    if source not in feasible_nodes or target not in feasible_nodes:
        return None
    cache = graph.graph.setdefault('_gosr_hop_cache', {})
    # On this topology, the small excluded set identifies the same feasible
    # graph without retaining a full constellation-sized set per destination.
    cache_key = (target, frozenset(blocked_nodes))
    hop_distance = cache.get(cache_key)
    if hop_distance is None:
        hop_distance = {target: 0}
        pending = deque([target])
        while pending:
            node = pending.popleft()
            distance = hop_distance[node] + 1
            for neighbor, edge in graph.adj[node].items():
                if (neighbor in feasible_nodes and neighbor not in hop_distance
                        and not edge.get('missing', 0)):
                    hop_distance[neighbor] = distance
                    pending.append(neighbor)
        # Bound each satellite's target/feasibility cache during long runs.
        if len(cache) >= 32:
            cache.clear()
        cache[cache_key] = hop_distance
    if source not in hop_distance:
        return None
    if max_hops is not None and hop_distance[source] > int(max_hops):
        return None
    path = []
    current = source
    while current != target:
        distance = hop_distance[current]
        choices = [neighbor for neighbor in graph.neighbors(current)
                   if neighbor in feasible_nodes
                   and not graph[current][neighbor].get('missing', 0)
                   and hop_distance.get(neighbor) == distance - 1]
        if not choices:
            return None
        current = min(choices, key=lambda neighbor: (
            _edge_cost(graph, current, neighbor, packet_size, transmission_rate),
            neighbor,
        ))
        path.append(current)
    return path


# Compatibility name for existing diagnostics; GSJF calls the greedy method.
dijkstra_route = greedy_min_hop_route


class GSJFSatellite:
    """Mixin installed on the simulator satellite for a GSJF test worker.

    ``one_hop`` commits to a directly reachable compute neighbor. In
    ``multi_hop`` mode a forwarding decision returns one hop with no compute
    reservation; the receiving satellite re-enters this method and compares
    its own local estimate with its new neighbors.
    """

    gosr_scope = 'one_hop'
    gosr_use_action_slots = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, 'neighbors'):
            return
        initial = [0, 1, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0]
        self.current_state = list(initial)
        self.neighbor_states = {neighbor: list(initial) for neighbor in self.neighbors}
        self.neighbor_elimination_features = {
            neighbor: [0, 1, 0, 0, 0, 0] for neighbor in self.neighbors}

    def prepare_gosr_routing(self, packet):
        # Use precisely the same admission mask as the execution stage.
        self._gosr_packet = packet
        if packet.information[0] or packet.computing_node is None:
            packet.routing = None

    def _gosr_legal_actions(self, destination, size, is_computed):
        packet = getattr(self, '_gosr_packet', None)
        if packet is None:
            return None, True
        mask = self.valid_action_mask(
            destination, size, is_computed, packet.hops,
            getattr(packet, 'wait_count', 0),
            previous_hop=packet.previous_hop,
            postcompute_previous_hop=getattr(packet, 'postcompute_previous_hop', None),
        )
        slots = getattr(self, 'action_neighbors', self.neighbors)
        legal = {neighbor for index, neighbor in enumerate(slots[:4])
                 if mask[index] and self._gosr_forward_capacity(neighbor)}
        return legal, bool(mask[4])

    def _gosr_ensure_graph_reservations(self):
        graph = getattr(getattr(self, 'propagator', None), 'graph', None)
        if graph is None or self.name not in graph:
            return
        attrs = graph.nodes[self.name]
        attrs.setdefault('current_memory_occupy',
                         _positive(getattr(self, 'current_memory_occupy', 0.0)))
        attrs.setdefault('computing_remain',
                         _positive(getattr(self, 'computing_remain', 0.0)))

    def _release_transmission_memory(self, packet):
        self._gosr_ensure_graph_reservations()
        return super()._release_transmission_memory(packet)

    def pop_transmission(self, neighbor, retain_memory=False):
        packet = yield self.transmission_queue[neighbor].get()
        import SatelliteNetworkSimulator_Computing as simulation
        simulation.record_queue_wait(packet, f'transmission:{neighbor}', self.env.now)
        self.current_queue_size -= packet.size
        self.transmission_size[neighbor] -= packet.size
        self.transmission_length[neighbor] -= 1
        if not retain_memory:
            self.current_memory_occupy -= packet.size
        if self.mode in ['Tradition', 'Ground']:
            graph = self.propagator.graph
            if graph.has_edge(self.name, neighbor):
                edge = graph[self.name][neighbor]
                edge['transmission_weight'] = max(
                    0.0, edge.get('transmission_weight', 0.0) - packet.size,
                )
                self._gosr_ensure_graph_reservations()
                if not retain_memory:
                    graph.nodes[self.name]['current_memory_occupy'] -= packet.size
        return packet

    def computing_packet(self):
        """Keep simulator graph reservations present across topology refreshes."""
        self._gosr_ensure_graph_reservations()
        yield from super().computing_packet()

    def push_computing(self, packet, computing_demand):
        # PRE reservations keep the compute demand on the topology graph, so
        # initialize the memory field before the simulator's completion hook
        # subtracts the pre-compute packet size.
        self._gosr_ensure_graph_reservations()
        return super().push_computing(packet, computing_demand)

    def _gosr_edge_wait(self, neighbor):
        queue = getattr(self, 'transmission_queue', {}).get(neighbor)
        if queue is None:
            return 0.0
        rate = _positive(self.transmission_rate)
        queued = _positive(getattr(self, 'transmission_size', {}).get(neighbor, 0.0))
        active = (_positive(self._remaining_transmission_time(neighbor))
                  if hasattr(self, '_remaining_transmission_time') else 0.0)
        return queued / rate + active if rate else math.inf

    def _gosr_graph(self):
        """Plan from received topology and resource messages, never the live graph."""
        from SatelliteNetworkSimulator_Computing import MAX_COMPUTING_ABILITY, CT_FAC
        topology = (getattr(self, '_gosr_adjacency_version', 0),
                    id(self.adjacency_table), tuple(self.neighbors),
                    tuple(self.neighbor_states))
        if topology != getattr(self, '_gosr_topology_key', None):
            graph = nx.Graph()
            for node, (neighbors, _) in self.adjacency_table.items():
                graph.add_node(node)
                graph.add_edges_from((node, neighbor) for neighbor in neighbors)
            graph.add_nodes_from(self.neighbor_states)
            graph.add_node(self.name)
            graph.remove_edges_from(list(graph.edges(self.name)))
            graph.add_edges_from((self.name, neighbor) for neighbor in self.neighbors)
            graph.graph['_gosr_nodes'] = frozenset(graph.nodes)
            self._gosr_topology_key = topology
            self._gosr_planning_graph = graph
        graph = self._gosr_planning_graph
        remaining = max(0.0, self.computing_remain
                        - (self.env.now - self.last_computing_time)
                        * self.computing_ability * bool(self.is_computing))
        graph.nodes[self.name].update(
            computing_ability=self.computing_ability,
            computing_remain=remaining,
            current_memory_occupy=self.current_memory_occupy)
        for neighbor, state in self.neighbor_states.items():
            graph.nodes[neighbor].clear()
            if (not self.neighbor_has_resource_state(neighbor)
                    or len(state) < 4):
                graph.nodes[neighbor]['gosr_resource_missing'] = True
                continue
            ability = math.sqrt(_positive(state[2])) * MAX_COMPUTING_ABILITY
            graph.nodes[neighbor].update(
                computing_ability=ability,
                computing_remain=_positive(state[3]) * CT_FAC * ability,
                current_memory_occupy=(1.0 - float(state[1])) * self.memory)
        for neighbor in self.neighbors:
            graph[self.name][neighbor].update(
                propagation_weight=getattr(self, '_gosr_received_delays', {}).get(neighbor, 0.0),
                transmission_weight=_positive(self.transmission_size.get(neighbor, 0.0)))
        graph.graph['_gosr_memory'] = {
            node: _positive(graph.nodes[node].get('current_memory_occupy', 0.0))
            for node in (self.name, *self.neighbor_states)
            if _positive(graph.nodes[node].get('current_memory_occupy', 0.0)) > 0}
        return graph

    def update_adjacency_dict(self, table):
        topology_updated = super().update_adjacency_dict(table)
        if topology_updated:
            self._gosr_adjacency_version = getattr(self, '_gosr_adjacency_version', 0) + 1
        return topology_updated

    def on_neighbor_state_received(self, neighbor, sent_at=None):
        super().on_neighbor_state_received(neighbor, sent_at=sent_at)
        message = self.neighbor_states[neighbor]
        if hasattr(message, 'propagation_delay'):
            if not hasattr(self, '_gosr_received_delays'):
                self._gosr_received_delays = {}
            self._gosr_received_delays[neighbor] = message.propagation_delay

    def _gosr_filled_node(self, graph, node):
        """Fill unavailable resources from received healthy neighbor features."""
        attrs = dict(graph.nodes[node])
        required = ('computing_ability', 'computing_remain', 'current_memory_occupy')
        if (not attrs.get('gosr_resource_missing', False)
                and all(key in attrs for key in required)
                and _positive(attrs.get('computing_ability')) > 0):
            return attrs, False
        peers = [graph.nodes[neighbor] for neighbor in self.neighbors
                 if neighbor != node and neighbor in graph
                 and not graph.nodes[neighbor].get('gosr_resource_missing', False)
                 and _positive(graph.nodes[neighbor].get('computing_ability')) > 0]
        if not peers:
            return attrs, False
        multiplier = _positive(self.fault_fill_multiplier) or 1.0
        from SatelliteNetworkSimulator_Computing import MAX_COMPUTING_ABILITY, CT_FAC
        # Match ISACR's normalized fault block: zero free memory, reduced
        # ability and increased CPU wait, computed only from received peers.
        ability_feature = sum((peer['computing_ability'] / MAX_COMPUTING_ABILITY) ** 2
                              for peer in peers) / len(peers)
        queue_feature = sum(peer.get('computing_remain', 0.0)
                            / peer['computing_ability'] / CT_FAC
                            for peer in peers) / len(peers)
        enabled = getattr(self, 'fault_compute_double_mean', True)
        ability = math.sqrt(min(1.0, ability_feature / multiplier)) * MAX_COMPUTING_ABILITY if enabled else 0.0
        wait = min(1.0, queue_feature * multiplier) * CT_FAC if enabled else CT_FAC
        attrs.update(computing_ability=ability, computing_remain=wait * ability,
                     current_memory_occupy=self.memory)
        return attrs, True

    def _gosr_active_nodes(self, graph):
        # Reachability is learned through adjacency messages and heartbeats.
        nodes = graph.graph.get('_gosr_nodes')
        return nodes if nodes is not None else frozenset(graph.nodes)

    def _gosr_memory(self):
        memory = getattr(self, 'memory', None)
        return None if memory is None else _positive(memory)

    def _gosr_can_hold(self, node, packet_size, graph, attrs=None):
        attrs = graph.nodes[node] if attrs is None else attrs
        return (_positive(attrs.get('current_memory_occupy', 0.0))
                + _positive(packet_size) <= self._gosr_memory())

    def _gosr_forward_capacity(self, neighbor):
        """Return false when this link's forwarding queue is unavailable."""
        queue = getattr(self, 'transmission_queue', {}).get(neighbor)
        if queue is None:
            return False
        limit = getattr(self, 'queue_length', None)
        if limit is None:
            return True
        return len(queue.items) < int(limit)

    def _gosr_neighbors(self, graph, destination):
        return [neighbor for neighbor in self.neighbors
                if neighbor in graph and neighbor != destination
                and graph.has_edge(self.name, neighbor)]

    def _gosr_candidate_nodes(self, graph, destination):
        if self.gosr_scope == 'one_hop':
            return self._gosr_neighbors(graph, destination)
        return sorted(node for node in graph.nodes
                      if node not in (self.name, destination))

    def _gosr_path_capacity(self, path):
        # Only this node's own outgoing queue is locally observable.
        return not path or self._gosr_forward_capacity(path[0])

    def _gosr_path_delay(self, graph, path, packet_size):
        rate = _positive(self.transmission_rate)
        if rate <= 0:
            return math.inf
        return sum(
            (graph[left][right].get('transmission_weight', 0.0)
             + _positive(packet_size)) / rate
            + _positive(graph[left][right].get('propagation_weight', 0.0))
            + _positive(graph[left][right].get('gosr_waiting', 0.0))
            for left, right in zip([self.name] + list(path), path)
        )

    def _gosr_path_parts(self, graph, path, packet_size):
        rate = _positive(self.transmission_rate)
        if rate <= 0:
            return math.inf, math.inf, math.inf
        waiting = sum(
            graph[left][right].get('transmission_weight', 0.0) / rate
            + _positive(graph[left][right].get('gosr_waiting', 0.0))
            for left, right in zip([self.name] + list(path), path)
        )
        transmission = len(path) * _positive(packet_size) / rate
        propagation = sum(
            _positive(graph[left][right].get('propagation_weight', 0.0))
            for left, right in zip([self.name] + list(path), path)
        )
        return waiting, transmission, propagation

    def _gosr_return_delay(self, graph, source, path, packet_size):
        return sum(_edge_cost(graph, left, right, packet_size, self.transmission_rate)
                   for left, right in zip([source] + list(path), path))

    def _gosr_route(self, graph, source, target, packet_size):
        return greedy_min_hop_route(
            graph, source, target, packet_size, self.transmission_rate,
            memory=self._gosr_memory(),
            active_nodes=self._gosr_active_nodes(graph),
            max_hops=max(int(getattr(self, 'max_hop', 1)), 1),
        )

    def _gosr_return_key(self, graph, neighbor, route, size):
        return (len(route), _edge_cost(
            graph, self.name, neighbor, size, self.transmission_rate), neighbor)

    def _gosr_choose_compute(self, candidates, size, reduced_size):
        return min(candidates, key=lambda item: (item[0], item[1] != self.name, item[1]))

    def tradition_routing(self, current_state, computing=True):
        source, destination, size, demand, reduced_size, is_computed = current_state
        graph = self._gosr_graph()
        legal, can_compute = self._gosr_legal_actions(destination, size, is_computed)
        if (not getattr(self, 'active', True)
                or source not in graph or destination not in graph):
            return [None], None
        if is_computed or not computing:
            candidates = []
            for neighbor in (self.neighbors if legal is None else legal):
                route = self._gosr_route(graph, neighbor, destination, size)
                if route is not None:
                    candidates.append((self._gosr_return_key(graph, neighbor, route, size), neighbor))
            if not candidates:
                return [None], None
            return [min(candidates)[1]], None

        now = float(getattr(self.env, 'now', 0.0))
        local_node, local_filled = self._gosr_filled_node(graph, self.name)
        local = estimate_computation(local_node, demand, now)
        candidates = []
        if (can_compute and local.computation < math.inf
                and _positive(local_node.get('current_memory_occupy', 0.0)) <= self._gosr_memory()):
            local_route = self._gosr_route(graph, self.name, destination, reduced_size)
            if local_route is not None:
                total = local.total
                candidates.append((total, self.name, local, [], local_route, local_filled))
        candidate_nodes = self._gosr_neighbors(graph, destination)
        for neighbor in candidate_nodes:
            if legal is not None and neighbor not in legal:
                continue
            node, filled = self._gosr_filled_node(graph, neighbor)
            if not self._gosr_can_hold(neighbor, size, graph, node):
                continue
            if self.gosr_scope == 'multi_hop':
                first = ([neighbor] if graph.has_edge(self.name, neighbor)
                         and not graph[self.name][neighbor].get('missing', 0)
                         else None)
            else:
                # A compute candidate is a direct neighbor: no BFS needed.
                first = ([neighbor] if not graph[self.name][neighbor].get('missing', 0)
                         else None)
            if first is None or not self._gosr_path_capacity(first):
                continue
            second = [] if self.gosr_scope == 'multi_hop' else self._gosr_route(
                graph, neighbor, destination, reduced_size,
            )
            if second is None or not self._gosr_path_capacity([*first, *second]):
                continue
            if self.gosr_scope == 'one_hop':
                send_waiting = self._gosr_edge_wait(neighbor)
                edge = graph[self.name][neighbor]
                propagation = edge.get('propagation_weight', 0.0)
                transmission = size / _positive(self.transmission_rate)
            else:
                send_waiting, transmission, propagation = self._gosr_path_parts(
                    graph, first, size,
                )
            estimate = estimate_neighbor(
                node, demand, size, self.transmission_rate,
                send_waiting=send_waiting,
                propagation=propagation, now=now,
            )
            estimate = ExecutionEstimate(
                # The neighbor can drain its existing compute queue while
                # this task waits for transmission and travels there.
                max(0.0, estimate.queueing - estimate.send_waiting
                    - transmission - estimate.propagation), estimate.computation,
                estimate.send_waiting, transmission,
                estimate.propagation,
            )
            total = estimate.total
            candidates.append((total, neighbor, estimate, first, second, filled))
        if not candidates:
            return [None], None
        _, compute_node, estimate, first, second, selected_filled = self._gosr_choose_compute(
            candidates, size, reduced_size)
        if self.gosr_scope == 'multi_hop' and compute_node != self.name:
            # Sequential greedy mode: forward only one legal hop. The next
            # satellite calls this method again and may compute locally or
            # forward to another neighbor.
            self.last_gosr_estimate = {
                'local': asdict(local),
                'selected': asdict(estimate),
                'compute_node': compute_node,
                'destination': destination,
                'scope': self.gosr_scope,
                'local_fault_filled': local_filled,
                'selected_fault_filled': selected_filled,
            }
            return [compute_node], None
        # Match the simulator's reservation contract. The node attribute must
        # exist even when the selected compute node is the current satellite;
        # computing_packet() decrements it after the task finishes.
        reservation_graph = self.propagator.graph
        compute_attrs = (reservation_graph.nodes[compute_node]
                         if compute_node in reservation_graph else graph.nodes[compute_node])
        compute_attrs.setdefault('computing_remain', 0.0)
        compute_attrs.setdefault('current_memory_occupy', 0.0)
        compute_attrs['computing_remain'] = (
            _positive(compute_attrs.get('computing_remain', 0.0))
            + _positive(demand)
        )
        self.last_gosr_estimate = {
            'local': asdict(local),
            'selected': asdict(estimate),
            'compute_node': compute_node,
            'destination': destination,
            'local_fault_filled': local_filled,
            'selected_fault_filled': selected_filled,
        }
        # ``self.name`` explicitly means compute here; ``None`` is reserved
        # for multi-hop forwarding where the next node must re-decide.
        return first + second, compute_node


# Keep the old import name for diagnostic scripts that subclass this mixin.
GOSRSatellite = GSJFSatellite


@contextmanager
def gosr_environment(scope='one_hop'):
    import SatelliteNetworkSimulator_Computing as simulation
    original = simulation.Satellite_with_Computing
    original_propagator = simulation.Propagator_Computing

    class StateMessage(list):
        pass

    class GOSRPropagator(original_propagator):
        def send_state(self, node, neighbor, value, eliminated_self_feature=None):
            message = StateMessage(value)
            message.propagation_delay = self.propagation_delays.get((node, neighbor), 0.0)
            yield from super().send_state(node, neighbor, message, eliminated_self_feature)

        def update(self, graph):
            super().update(graph)
            graph.graph.pop('_gosr_hop_cache', None)
            for name, satellite in self.satellites.items():
                if name not in self.graph:
                    continue
                attrs = self.graph.nodes[name]
                attrs.setdefault('computing_ability', satellite.computing_ability)
                attrs.setdefault('current_memory_occupy',
                                 _positive(getattr(satellite, 'current_memory_occupy', 0.0)))
                attrs.setdefault('computing_remain',
                                 _positive(getattr(satellite, 'computing_remain', 0.0)))

    simulation.Propagator_Computing = GOSRPropagator
    simulation.Satellite_with_Computing = type(
        'GSJFSatellite', (GOSRSatellite, original), {}
    )
    simulation.Satellite_with_Computing.gosr_scope = scope
    try:
        yield
    finally:
        simulation.Satellite_with_Computing = original
        simulation.Propagator_Computing = original_propagator


def _trace(entry):
    return baseline.resolve_trace(entry)


def build_config(kind, value, raw_path, scope='one_hop'):
    load = value if kind == 'load' else FAULT_WORKLOAD_CASE
    probability = LOAD_FAULT_PROBABILITY if kind == 'load' else float(value)
    workload_id = load['id']
    fault_id = baseline.FAULT_CASE_BY_PROBABILITY[probability]['id']
    workload = baseline.SCENARIO_MANIFEST['workloads'][workload_id]
    fault = baseline.SCENARIO_MANIFEST['faults'][fault_id]
    config = copy.deepcopy(baseline.COMMON_TEST)
    config['general'].update(
        phase='test', algorithm_name=f'GSJF_{scope}', random_seed=baseline.RANDOM_SEED,
        scenario_seed=baseline.TEST_SEED, duration=DURATION, rounds=1,
        epsilon=0.0, model_saving_enabled=False, print_cycle=1,
        show_test_summary=True,
    )
    config['agent'] = {
        'mode': 'Tradition', 'neighbors_dim': 60, 'edges_dim': 4,
        'distance_dim': 4, 'mission_dim': 6, 'current_dim': 4,
        'action_dim': 5, 'hidden_dim': 1, 'buffer_length': 1,
        'batch_size': 1, 'gamma': 0.99, 'q_mask': -1,
        'activation': 'LeakyRelu', 'hidden_layers': 1, 'dueling': False,
        'shuffle': False, 'learning_rate': 0.001, 'repeat': 1,
        'update_cycle': 1, 'model_path': '',
    }
    config['environment'].update(load)
    config['environment'].update(
        id=load['id'], edge_p=probability, edge_q=1.0 - probability,
        node_p=0.0, node_q=0.0, markov_mode=True,
        require_computed_before_downlink=False, visualize=False,
        save_log=False, print_info=False, show_detail=False,
        save_training_data=str(raw_path),
        workload_trace_path=str(_trace(workload)),
        workload_trace_sha256=workload['sha256'],
        workload_trace_task_count=int(workload['task_count']),
        fault_trace_path=str(_trace(fault)),
        fault_trace_sha256=fault['sha256'],
        fault_trace_event_count=int(fault['event_count']),
    )
    config['gosr_contract'] = {
        'algorithm': f'GSJF_{scope}',
        'policy': f'local_or_{scope}_estimated_delay',
        'scope': scope,
        'route_after_compute': 'greedy_min_hop',
        'compute_selection': 'local_or_neighbor_compute_completion_excluding_return',
        'compute_queue': 'fifo',
        'compute_preemption': False,
        'transmission_queue': 'fifo',
        'resource_observation': 'received_isacr_state_messages',
        'topology_observation': 'received_adjacency_and_heartbeat',
        'estimate_fields': ['queueing', 'computation', 'send_waiting',
                            'transmission', 'propagation'],
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'runtime_acceleration': runtime_acceleration_contract(),
    }
    return config


def complete(raw_path):
    return raw_path.is_file() and baseline.last_step(raw_path) == EXPECTED_STEP


def run_case(job):
    scope, kind, value, force = job
    output_root = OUTPUT_ROOT / scope
    case_id = value['id'] if kind == 'load' else f'fault_{int(round(float(value) * 100)):02d}'
    directory = output_root / kind / case_id
    directory.mkdir(parents=True, exist_ok=True)
    raw = directory / 'result.txt'
    running = directory / 'result.running.txt'
    config_path = directory / 'config.json'
    console = directory / 'console.txt'
    if complete(raw) and not force:
        return {'kind': kind, 'case_id': case_id, 'status': 'complete', 'raw_result_file': str(raw)}
    for path in (raw, running, config_path, console):
        path.unlink(missing_ok=True)
    config = build_config(kind, value, running, scope)
    config_path.write_text(json.dumps(config, indent=2, default=str), encoding='utf-8')
    with console.open('w', encoding='utf-8', buffering=1) as stream:
        with redirect_stdout(stream), redirect_stderr(stream), gosr_environment(scope):
            run_training(config)
    if not complete(running):
        raise RuntimeError(f'Incomplete GSJF result: {running}')
    running.replace(raw)
    config['environment']['save_training_data'] = str(raw)
    config_path.write_text(json.dumps(config, indent=2, default=str), encoding='utf-8')
    return {'kind': kind, 'case_id': case_id, 'status': 'complete', 'raw_result_file': str(raw)}


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row}) or ['status']
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help=f'parallel cases (default: {DEFAULT_WORKERS}; maximum: {MAX_WORKERS})')
    parser.add_argument('--experiment', choices=('both', 'load', 'fault'), default='both')
    parser.add_argument('--strategy', choices=('both', 'one-hop', 'multi-hop'), default='one-hop')
    parser.add_argument('--load-point', type=int)
    parser.add_argument('--fault-percent', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--rerun', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f'--workers must be between 1 and {MAX_WORKERS}')
    if args.experiment == 'load' and args.fault_percent is not None:
        parser.error('--fault-percent cannot be used with --experiment load')
    if args.experiment == 'fault' and args.load_point is not None:
        parser.error('--load-point cannot be used with --experiment fault')
    loads = tuple(item for item in LOAD_SETTINGS
                  if args.experiment != 'fault'
                  and (args.load_point is None or int(item['target_load_tasks_per_second']) == args.load_point))
    faults = tuple(prob for prob in FAULT_PROBABILITIES
                   if args.experiment != 'load'
                   and (args.fault_percent is None or int(round(prob * 100)) == args.fault_percent))
    scopes = ({'one-hop': ('one_hop',), 'multi-hop': ('multi_hop',)}
              .get(args.strategy, ('one_hop', 'multi_hop')))
    jobs = [
        (scope, kind, value, args.rerun)
        for scope in scopes
        for kind, values in (('load', loads), ('fault', faults))
        for value in values
    ]
    if args.dry_run:
        plan = [{'strategy': scope, 'kind': kind,
                 'case_id': value['id'] if kind == 'load' else f'fault_{int(round(float(value) * 100)):02d}'}
                for scope, kind, value, _ in jobs]
        write_rows(OUTPUT_ROOT / 'GSJF_TEST_PLAN.csv', plan)
        print(f'DRY_RUN: {len(jobs)} cases; {OUTPUT_ROOT / "GSJF_TEST_PLAN.csv"}')
        return
    if LOCK_PATH.exists():
        print(f'STATUS=busy: {LOCK_PATH}')
        return
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(str(os.getpid()), encoding='ascii')
    rows, errors = [], []
    try:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs) or 1)) as executor:
            futures = {executor.submit(run_case, job): job for job in jobs}
            for future in as_completed(futures):
                try:
                    row = future.result()
                    rows.append(row)
                    print(f"Completed {row['kind']}/{row['case_id']}", flush=True)
                except Exception as error:
                    job = futures[future]
                    errors.append({'kind': job[0], 'value': job[1], 'error': str(error)})
                    print(f'ERROR {job[0]}/{job[1]}: {error}', flush=True)
        write_rows(OUTPUT_ROOT / 'GSJF_RESULTS.csv', rows)
        (OUTPUT_ROOT / 'GSJF_STATUS.json').write_text(json.dumps({
            'algorithms': [f'GSJF_{scope}' for scope in scopes],
            'strategies': list(scopes), 'load_cases_per_strategy': len(loads),
            'fault_cases_per_strategy': len(faults),
            'completed_cases': len(rows), 'errors': errors,
            'updated_at': datetime.now().astimezone().isoformat(),
        }, indent=2), encoding='utf-8')
    finally:
        LOCK_PATH.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
