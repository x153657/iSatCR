from SatelliteNetworkSimulator_Beta import SatelliteNetworkSimulator,Packet,Satellite,Propagator
import hashlib
import json
import math
import random
from pathlib import Path
import numpy as np
import simpy
import networkx as nx
import torch
from collections import deque
from itertools import islice

from torch_geometric.data import Data
from torch_geometric.data import Batch

PRE=True
CT_FAC = 50.0  # CPU wait normalization scale, in seconds
COMPUTE_SCORE_BETA = 50.0
TRANSMISSION_WAIT_TIME_SCALE = 3.0  # seconds
PASSIVE_WAIT_MAX_RETRIES = float('inf')  # Keep blocked packets until a legal action returns.
LOCAL_COMPUTE_WAIT_THRESHOLD = 0.5  # seconds; only gates no-forward uncomputed RL packets
MAX_COMPUTING_ABILITY = 100e9  # fixed highest resource tier, in FLOPS
MAX_COMPUTING_DEMAND = 300e9
NODE_FEATURE_DIM = 5
MULTIHOP_FEATURE_DIM = NODE_FEATURE_DIM * 3
ISACR_NODE_FEATURE_DIM = 6
NEIGHBOR_SLOT_DIM = ISACR_NODE_FEATURE_DIM * 3 + 2

LOSS_BREAKDOWN_KEYS = (
    'Lost_upload_source_or_destination_missing',
    'Lost_upload_no_route',
    'Lost_upload_memory_overflow',
    'Lost_relay_link_break',
    'Lost_relay_memory_overflow',
    'Lost_relay_node_missing',
    'Lost_relay_node_failure',
    'Lost_relay_no_route',
    'Lost_relay_invalid_action',
    'Lost_relay_timeout',
    'Lost_relay_hop_limit',
    'Lost_relay_wait_limit',
    'Lost_compute_node_failure',
    'Lost_offload_node_failure',
)


def record_packet_loss(statics, stage, reason):
    key = f'Lost_{stage}_{reason}'
    statics[key] = statics.get(key, 0) + 1


def _clip_normalized(value):
    """Keep normalized state/fault-fill values in the closed unit interval."""
    return min(max(float(value), 0.0), 1.0)


def _normalized_computing_demand(computing_demand):
    """Represent the packet workload independently of the current satellite."""
    return _clip_normalized(computing_demand / MAX_COMPUTING_DEMAND)


def add_packet_delay(packet, component, duration):
    packet.delay_components[component] += max(float(duration), 0.0)


def mark_queue_entry(packet, queue_name, now):
    packet.queue_enter_times[queue_name] = float(now)


def record_queue_wait(packet, queue_name, now):
    entered_at = packet.queue_enter_times.pop(queue_name, None)
    if entered_at is not None:
        add_packet_delay(packet, 'queueing', float(now) - entered_at)


class Reward_Function:
    def __init__(self,reach_factor,delay_factor,loss_factor,memory_threshold,memory_factor,
                 hop_factor=0.0, uncomputed_reach_penalty=0.0):
        self.reach_factor=reach_factor
        self.delay_factor=delay_factor
        self.loss_factor=loss_factor
        self.memory_threshold=memory_threshold
        self.memory_factor=memory_factor
        self.hop_factor=hop_factor
        self.uncomputed_reach_penalty=uncomputed_reach_penalty

    def reach_reward(self,delay):
        return self.reach_factor

    def reach_reward_abnormal(self,delay):
        return -self.uncomputed_reach_penalty

    def normal_reward(self,delay,memory_remain,hop_count=0):
        # memory_remain is the selected neighbor's projected free ratio at
        # the forwarding decision; below 10% free incurs one memory penalty.
        return (
            -self.delay_factor * delay
            - self.memory_factor * (
                memory_remain is not None
                and memory_remain < self.memory_threshold
            )
            - self.hop_factor * hop_count
        )

    def loss_reward(self,delay):
        return -self.loss_factor


class Propagator_Computing(Propagator):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experiences = []
        self.final_rewards = []
        self.ppo_trajectory_counter = 0
        self.experience_callback = None
        self.link_failure_events = {}

    def update(self, graph):
        old_edges = set(self.graph.edges())
        super().update(graph)
        removed_edges = old_edges - set(graph.edges())
        for node, neighbor in sorted(removed_edges):
            for key in ((node, neighbor), (neighbor, node)):
                event = self.link_failure_events.pop(key, None)
                if event is not None and not event.triggered:
                    event.succeed()

    def watch_link_failure(self, node, neighbor):
        if not self.graph.has_edge(node, neighbor):
            return None
        key = (node, neighbor)
        event = self.link_failure_events.get(key)
        if event is None or event.triggered:
            event = self.env.event()
            self.link_failure_events[key] = event
        return event

    def clear_link_failure_watch(self, node, neighbor, event):
        key = (node, neighbor)
        if self.link_failure_events.get(key) is event:
            self.link_failure_events.pop(key, None)

    def trans_parameters(self,max_hop,downstream_delays,reward_function):
        self.max_hop=max_hop
        self.downstream_delays=downstream_delays
        self.reward_function = reward_function

    def reset_parameters(self):
        self.experiences = []
        self.final_rewards = []

    def record_experience(
        self, experience, valid_action_mask=None, packet=None,
        current_action_mask=None,
    ):
        if packet is not None and not bool(experience[5]):
            self.add_packet_reward(packet, experience[3])
        if current_action_mask is not None:
            experience = list(experience) + [
                np.asarray(current_action_mask, dtype=bool).copy(),
                (np.asarray(valid_action_mask, dtype=bool).copy()
                 if valid_action_mask is not None else None),
            ]
        elif valid_action_mask is not None:
            experience = list(experience) + [
                np.asarray(valid_action_mask, dtype=bool).copy()
            ]
        self.experiences.append(experience)
        if self.experience_callback is not None:
            self.experience_callback(experience)

    @staticmethod
    def add_packet_reward(packet, reward):
        if reward is None or packet.reward_finalized:
            return
        packet.cumulative_reward += float(reward)
        packet.reward_step_count += 1

    def finalize_packet_reward(self, packet, terminal_reward):
        """Publish one complete packet return exactly once."""
        if packet.reward_finalized:
            return
        packet.terminal_reward = float(terminal_reward)
        packet.cumulative_reward += packet.terminal_reward
        packet.reward_finalized = True
        self.final_rewards.append(packet.cumulative_reward)

    def record_terminal_loss(self, node, packet, reward):
        """Record topology and link failures as terminal PPO transitions."""
        reservation_book = getattr(self, 'reservation_book', None)
        if reservation_book is not None:
            reservation_book.release(packet)
        information = packet.information
        last_state, last_action = information[-2], information[-1]
        # Charge the unfinished decision interval for every failure path,
        # including cases where the node or policy state is unavailable.
        reward -= self.reward_function.delay_factor * (self.env.now - information[4])
        if last_state is None or node not in self.satellites:
            self.finalize_packet_reward(packet, reward)
            return

        is_computed, packet_type, computing_demand, size_after_computing = information[:4]
        satellite = self.satellites[node]
        current_state = satellite.get_current_state(
            packet.destination,
            packet.hops,
            is_computed,
            [
                packet_type,
                packet.size / satellite.max_size,
                _normalized_computing_demand(computing_demand),
                size_after_computing / satellite.max_size,
            ],
        )
        mark = current_state[-1] if 'GNN' not in satellite.mode else current_state[-1][-1]
        transition_state = satellite.ppo_transition_state(current_state)
        self.record_experience(
            [last_state, mark, last_action, reward, transition_state, 1],
            packet=packet,
            current_action_mask=packet.forward_action_mask,
        )
        self.finalize_packet_reward(packet, reward)

    def propagate(self,node,next_hop,packet):
        is_computed, type, computing_demand, size_after_computing, last_time, last_state, last_action = packet.information
        if (node, next_hop) in self.propagation_delays:
            propagation_delay = self.propagation_delays[(node, next_hop)]
            yield self.env.timeout(propagation_delay)
            add_packet_delay(packet, 'propagation', propagation_delay)
            # The sender may have started this hop before the physical link
            # failed.  Heartbeat state is used for decisions, but delivery
            # must still obey the current physical graph.
            if not self.graph.has_edge(node, next_hop):
                sender = self.satellites.get(node)
                on_resolved = getattr(sender, 'on_lcsr_packet_resolved', None)
                if on_resolved is not None:
                    on_resolved(next_hop, packet.size)
                reward = self.reward_function.loss_reward(self.env.now - packet.creation_time)
                self.record_terminal_loss(node, packet, reward)
                if packet.computing_node and PRE and packet.computing_node in self.graph.nodes:
                    self.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                if type == 0:
                    self.statics_data['Lost_relay_0'] += 1
                else:
                    self.statics_data['Lost_relay_1'] += 1
                record_packet_loss(self.statics_data, 'relay', 'link_break')
                self.logger.log(
                    f"Time {self.env.now:.3f}: in-flight link {(node, next_hop)} failed, "
                    f"dropping packet {(packet.source, packet.destination, packet.creation_time)}."
                )
                return
            if next_hop in self.node_names and next_hop in self.satellites:
                receiver = self.satellites[next_hop]
                packet.previous_hop = node
                packet.postcompute_previous_hop = None
                accepted = receiver.push_forward(packet, from_transmission=True)
                if not accepted and getattr(receiver, 'defer_memory_overflow', False):
                    accepted = yield self.env.process(
                        receiver.wait_for_packet_capacity(packet, node)
                    )
                sender = self.satellites.get(node)
                on_resolved = getattr(sender, 'on_lcsr_packet_resolved', None)
                if on_resolved is not None:
                    on_resolved(next_hop, packet.size)
                if accepted:
                    self.logger.log(f"Time {self.env.now:.3f}: {next_hop}: Packet {(packet.source,packet.destination,packet.creation_time)} received by router. Memory remain: {self.satellites[next_hop].current_memory_occupy}.",detail=True)
                else:
                    reward = self.reward_function.loss_reward(self.env.now - packet.creation_time)
                    self.record_terminal_loss(node, packet, reward)
                    if packet.computing_node and PRE and packet.computing_node in self.graph.nodes:
                        self.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                    self.statics_data[f'Lost_relay_{type}'] += 1
                    record_packet_loss(self.statics_data, 'relay', 'memory_overflow')
                    self.logger.log(f"Time {self.env.now:.3f}: {next_hop} memory overflow, dropped packet {(packet.source, packet.destination, packet.creation_time)}")
            else:
                sender = self.satellites.get(node)
                on_resolved = getattr(sender, 'on_lcsr_packet_resolved', None)
                if on_resolved is not None:
                    on_resolved(next_hop, packet.size)
                reward = self.reward_function.loss_reward(self.env.now - packet.creation_time)
                self.record_terminal_loss(node, packet, reward)
                if packet.computing_node and PRE and packet.computing_node in self.graph.nodes:
                    self.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                if type==0:
                    self.statics_data['Lost_relay_0'] += 1
                else:
                    self.statics_data['Lost_relay_1'] += 1
                record_packet_loss(self.statics_data, 'relay', 'node_missing')
                self.logger.log(f"Time {self.env.now:.3f}: {next_hop} is missed, dropped 1 packet.")
        else:
            sender = self.satellites.get(node)
            on_resolved = getattr(sender, 'on_lcsr_packet_resolved', None)
            if on_resolved is not None:
                on_resolved(next_hop, packet.size)
            reward = self.reward_function.loss_reward(self.env.now - packet.creation_time)
            self.record_terminal_loss(node, packet, reward)
            if packet.computing_node and PRE and packet.computing_node in self.graph.nodes:
                self.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
            if type == 0:
                self.statics_data['Lost_relay_0'] += 1
            else:
                self.statics_data['Lost_relay_1'] += 1
            record_packet_loss(self.statics_data, 'relay', 'link_break')
            self.logger.log(f"Time {self.env.now:.3f}: connection {(node, next_hop)} is missed, dropped 1 packet")

    def downstream(self, node, packet):
        source, destination, hops, creation_time,size = packet.source, packet.destination, packet.hops, packet.creation_time,packet.size
        is_computed,type, computing_demand, size_after_computing, last_time, last_state, last_action = packet.information
        satellite = self.satellites.get(node)
        current_state = (
            satellite.get_current_state(
                destination, hops, is_computed,
                [type, size / satellite.max_size,
                 _normalized_computing_demand(computing_demand),
                 size_after_computing / satellite.max_size],
            )
            if satellite is not None else None
        )
        done = 1
        if satellite is not None:
            yield self.env.timeout(self.downstream_delays)
            add_packet_delay(packet, 'propagation', self.downstream_delays)
            if is_computed:
                reward = self.reward_function.reach_reward(self.env.now-creation_time)
                if type==0:
                    self.statics_data['Reached_after_computed_0'] += 1
                else:
                    self.statics_data['Reached_after_computed_1'] += 1
            else:
                reward = self.reward_function.reach_reward_abnormal(self.env.now-creation_time)
            # Settle only the unfinished last decision interval, including
            # relay/downlink queues, transmission, and propagation.
            reward -= self.reward_function.delay_factor * (self.env.now - last_time)
            if type == 0:
                self.statics_data['Reached_0'] += 1
            else:
                self.statics_data['Reached_1'] += 1
            if type == 0:
                self.statics_data['Total_hops_0'] += packet.hops
            else:
                self.statics_data['Total_hops_1'] += packet.hops
            if type == 0:
                self.statics_data['Total_delay_0'] += self.env.now - packet.creation_time - packet.cancelled_transmission_time
            else:
                self.statics_data['Total_delay_1'] += self.env.now - packet.creation_time - packet.cancelled_transmission_time
            suffix = str(type)
            for component, value in packet.delay_components.items():
                self.statics_data[f'Total_{component}_delay_{suffix}'] += value
            self.statics_data['Computing_waiting_time'] += packet.computing_waiting_time
            self.logger.log( f"Time {self.env.now:.3f}: Packet {(source, destination, packet.creation_time)} reached its destination {node}.")
        else:
            reward = self.reward_function.loss_reward(self.env.now-creation_time)
            reward -= self.reward_function.delay_factor * (self.env.now - last_time)
            reservation_book = getattr(self, 'reservation_book', None)
            if reservation_book is not None:
                reservation_book.release(packet)
            if packet.computing_node and PRE and packet.computing_node in self.graph.nodes:
                self.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
            if type == 0:
                self.statics_data['Lost_relay_0'] += 1
            else:
                self.statics_data['Lost_relay_1'] += 1
            record_packet_loss(self.statics_data, 'relay', 'node_missing')
            self.logger.log(f"Time {self.env.now:.3f}: downlink of {node} is missed, dropped 1 packet")
        if last_state is not None and current_state is not None:
            if 'GNN' not in satellite.mode:
                mark = current_state[-1]
            else:
                mark = current_state[-1][-1]
            transition_state = satellite.ppo_transition_state(current_state)
            self.record_experience(
                [last_state, mark, last_action, reward, transition_state, done],
                packet=packet,
                current_action_mask=packet.forward_action_mask,
            )
        self.finalize_packet_reward(packet, reward)

    def send_two_hop_info(self, node, neighbor, info):
        if (node, neighbor) in self.propagation_delays:
            yield self.env.timeout(self.propagation_delays[(node, neighbor)])
            if not self.graph.has_edge(node, neighbor):
                return
            if neighbor in self.node_names and neighbor in self.satellites:
                if node in self.satellites[neighbor].neighbors:
                    self.satellites[neighbor].update_three_hop_info_db(info)
                else:
                    self.satellites[neighbor].update_three_hop_info_db(info)
            else:
                self.logger.log(f"Time {self.env.now:.3f}: {neighbor} is missed, info update failed.")
        else:
            self.logger.log(
                f"Time {self.env.now:.3f}: connection {(node, neighbor)} is missed, info update failed.")

    def send_gnn_embedding_message(self, node, neighbor, stage, version, value):
        if (node, neighbor) not in self.propagation_delays:
            self.logger.log(
                f"Time {self.env.now:.3f}: connection {(node, neighbor)} is missed, GraphSAGE {stage} update failed.")
            return
        link_key = frozenset((node, neighbor))
        generation = self.link_generations.get(link_key, 0)
        yield self.env.timeout(self.propagation_delays[(node, neighbor)])
        if (not self.graph.has_edge(node, neighbor)
                or generation != self.link_generations.get(link_key, 0)):
            return
        if neighbor not in self.node_names or neighbor not in self.satellites:
            self.logger.log(
                f"Time {self.env.now:.3f}: {neighbor} is missed, GraphSAGE {stage} update failed.")
            return
        receiver = self.satellites[neighbor]
        validator = getattr(receiver, 'valid_resource_message', None)
        if validator is not None and not validator(stage, value):
            return
        if getattr(receiver, 'aligned_graphsage_policy', False):
            records = getattr(receiver, 'gnn_pipeline_messages', {}).get(node, ())
            if records and version <= records[-1][0]:
                return
        if node not in receiver.neighbors:
            if not getattr(receiver, 'aligned_graphsage_policy', False) or node not in receiver.action_neighbors:
                return
            receiver.add_neighbor(node)
        receiver.receive_gnn_embedding_message(node, stage, version, value)

class Satellite_with_Computing(Satellite):
    def __init__(self,mode,select_mode,epsilon,max_hop,max_size,device,env,name,neighbors,memory,computing_ability,transmission_rate,downlink_rate,state_update_period,is_downlink,logger,statics_data={},num=None,processing_time=1e-9,heartbeat_timeout=0.15,fault_compute_double_mean=True,fault_fill_multiplier=2.0,historical_self_feature_elimination=False,disable_self_feature_elimination=False,gnn_information_hops=4,gnn_include_self=False,gnn_node_feature_dim=3,tradition_replan_interval=None,require_computed_before_downlink=False,tradition_use_genetic_search=False,tradition_ga_population_size=20,tradition_ga_generations=10,tradition_ga_path_candidates=3,tradition_ga_mutation_rate=0.1,topology_neighbors=None):
        self.num=num
        self.mode=mode
        self.fault_compute_double_mean=fault_compute_double_mean
        self.fault_fill_multiplier=fault_fill_multiplier
        self.historical_self_feature_elimination=historical_self_feature_elimination
        self.disable_self_feature_elimination=disable_self_feature_elimination
        self.require_computed_before_downlink=require_computed_before_downlink
        self.gnn_information_hops=gnn_information_hops
        self.gnn_include_self=gnn_include_self
        self.gnn_node_feature_dim=gnn_node_feature_dim
        self.tradition_replan_interval = tradition_replan_interval
        self.tradition_use_genetic_search = tradition_use_genetic_search
        self.tradition_ga_population_size = tradition_ga_population_size
        self.tradition_ga_generations = tradition_ga_generations
        self.tradition_ga_path_candidates = tradition_ga_path_candidates
        self.tradition_ga_mutation_rate = tradition_ga_mutation_rate
        self.select_mode=select_mode
        self.q_net=None
        self.gnn=None
        self.epsilon=epsilon
        self.max_hop=max_hop
        self.max_size=max_size
        self.device=device
        self.name=name
        initial_neighbors = sorted(neighbors)
        wait_for_state = (self.mode in ('New_DQN', 'Pure_DQN', 'Pure_PPO')
                          or getattr(self, 'aligned_graphsage_policy', False))
        self.neighbors = [] if wait_for_state else initial_neighbors
        self.neighbor_requires_state = True
        self.pending_state_neighbors = set(initial_neighbors) if wait_for_state else set()
        self.pending_link_since = dict.fromkeys(self.pending_state_neighbors, env.now)
        self.pending_adjacency_records = {}
        self._state_probe_event = env.event()
        if self.mode in ('New_DQN', 'Pure_DQN', 'Pure_PPO') or getattr(self, 'aligned_graphsage_policy', False):
            expected_neighbors = sorted(
                initial_neighbors if topology_neighbors is None else topology_neighbors
            )
            if len(expected_neighbors) > 4:
                raise ValueError('RL policy supports at most four forwarding actions')
            self.topology_neighbors = tuple(expected_neighbors)
            self.action_neighbors = expected_neighbors + [None] * (4 - len(expected_neighbors))
        self.env=env
        self.gat_node_dim=MULTIHOP_FEATURE_DIM
        self.orbit_altitude, self.orbit_number, self.sat_number = map(int, self.name.split('_')[1:])
        self.memory=memory
        self.computing_ability=computing_ability
        self.transmission_rate=transmission_rate
        self.downlink_rate=downlink_rate
        self.state_update_period=state_update_period
        self.logger=logger
        self.computing_queue = simpy.Store(self.env)
        self.offload_queue=simpy.Store(self.env)
        self.offload_size=0
        self.offload_length=0
        self.transmission_queue = {neighbor: simpy.Store(self.env) for neighbor in self.neighbors}
        self.active_transmissions = {}
        self.neighbor_failure_events = {}
        self.neighbor_state_versions = {neighbor: 0 for neighbor in self.neighbors}
        self.neighbor_state_events = {neighbor: self.env.event() for neighbor in self.neighbors}
        self.transmission_size ={neighbor: 0 for neighbor in self.neighbors}
        self.transmission_length ={neighbor: 0 for neighbor in self.neighbors}
        self.neighbor_hops = {neighbor: {} for neighbor in self.neighbors}
        self._neighbor_route_since = {}
        self.current_queue_size=0
        self.current_computing_queue_size=0
        self.forward_queue = simpy.Store(self.env)
        self.waiting_forward = deque()
        self.current_memory_occupy=0
        self.active=True
        self.routing_tables={}
        if 'New' in self.mode:
            initial_state = [0,1,0,0,0,0,0,4,0,0,0,0,0,12,0,0,0,0]
            self.current_state = list(initial_state)
            self.neighbor_states = {
                neighbor: list(initial_state) for neighbor in self.neighbors
            }
            self.neighbor_elimination_features = {
                neighbor: [0,1,0,0,0,0] for neighbor in self.neighbors
            }
        elif 'GNN' in self.mode:
            self.current_state = [0 for _ in range(self.gat_node_dim)]
            self.neighbor_states = {
                neighbor: [0,1,0,0,0] + ([0] if self.gnn_node_feature_dim == ISACR_NODE_FEATURE_DIM else [])
                for neighbor in self.neighbors
            }
        else:
            self.current_state = [0,1,0,0,0] + ([0] if self.mode in ('Pure_DQN', 'Pure_PPO') else [])
            self.neighbor_states={
                neighbor: list(self.current_state) for neighbor in self.neighbors
            }
        self.propagator=None
        self.statics_data=statics_data
        self.processing_time=processing_time
        self.heartbeat_timeout = heartbeat_timeout
        self.last_heartbeat = {neighbor: env.now for neighbor in self.neighbors}
        self.last_heartbeat_sent_times = {}
        self.is_downlink=is_downlink
        self.is_producing=0

        self.hops={}
        self.adjacency_table = {self.name: (self.neighbors, self.env.now)}
        self.computing_remain=0
        self.neighbor_graph=None
        self.is_computing=False
        self.computing_time=0
        self.transmission_time=0
        self.equivalent_transmission_data_size=0

        self.last_computing_time=0

        self.three_hop_info_db = {}

        self.encode_data=None
        self.encode_data_update_time=self.env.now
        self.gnn_cycle_version = 0
        self.gnn_raw_snapshots = {}
        self.gnn_raw_messages = {}
        self.gnn_layer1_messages = {}
        self.gnn_layer2_messages = {}
        self.direct_concat_received_at = {}
        self.gnn_active_neighbor_embeddings = {}
        self.gnn_active_version = None

    def _normalized_compute_ability(self):
        """Squared normalized capacity; larger values mean higher capacity."""
        return _clip_normalized(self.computing_ability / MAX_COMPUTING_ABILITY) ** 2

    def _normalized_compute_time(self, queue_seconds):
        """Encode CPU backlog linearly, saturating at fifty seconds."""
        queue_seconds = max(float(queue_seconds), 0.0)
        return min(queue_seconds / CT_FAC, 1.0)

    def _normalized_transmission_wait(self, wait_seconds):
        """Encode an ISL wait linearly, saturating at three seconds."""
        wait_seconds = max(float(wait_seconds), 0.0)
        return min(wait_seconds / TRANSMISSION_WAIT_TIME_SCALE, 1.0)

    def _normalized_compute_completion_time(self):
        """Capacity discounted by capped CPU wait; retain the legacy name."""
        queue_seconds = max(0.0, self.computing_remain / self.computing_ability
                            - self.is_computing * (self.env.now - self.last_computing_time))
        ability = _clip_normalized(self.computing_ability / MAX_COMPUTING_ABILITY)
        return ability / (1.0 + COMPUTE_SCORE_BETA * self._normalized_compute_time(queue_seconds))

    def _local_fault_compute_score(self):
        """An unavailable resource block has zero ability and unit CPU wait."""
        return 0.0

    def _normalized_link_transmission_wait(self, neighbor):
        """Encode queued work and the current transmission on one ISL."""
        return self._normalized_transmission_wait(
            self.transmission_size.get(neighbor, 0) / self.transmission_rate
            + self._remaining_transmission_time(neighbor)
        )

    def _normalized_mean_transmission_wait(self):
        """Normalize the mean four-slot ISL wait with the shared encoding."""
        mean_wait_seconds = (
            (sum(self.transmission_size.values()) / self.transmission_rate
             + sum(self._remaining_transmission_time(neighbor) for neighbor in self.neighbors))
            / 4.0
        )
        return self._normalized_transmission_wait(mean_wait_seconds)

    def _remaining_transmission_time(self, neighbor):
        transmission = self.active_transmissions.get(neighbor)
        return max(0.0, transmission[0] - self.env.now) if transmission is not None else 0.0

    def _heartbeat_deadline(self, neighbor):
        sent_at = self.last_heartbeat_sent_times.get(
            neighbor, self.last_heartbeat.get(neighbor, -float('inf')),
        )
        if not math.isfinite(sent_at):
            return -float('inf')
        cycle = math.floor(sent_at / self.state_update_period + 1e-9)
        return cycle * self.state_update_period + self.heartbeat_timeout

    def neighbor_is_alive(self, neighbor):
        return self.env.now <= self._heartbeat_deadline(neighbor)

    def neighbor_has_resource_state(self, neighbor):
        return self.neighbor_is_alive(neighbor) and self.neighbor_state_versions.get(neighbor, 0) > 0

    def monitor_single_neighbor(self, neighbor):
        if (self.mode not in ('New_DQN', 'Pure_DQN', 'Pure_PPO')
                and not getattr(self, 'aligned_graphsage_policy', False)
                and not getattr(self, 'gosr_use_action_slots', False)):
            yield from super().monitor_single_neighbor(neighbor)
            return
        while self.active and neighbor in self.neighbors:
            if not self.neighbor_is_alive(neighbor):
                yield self.env.process(self.del_neighbor(neighbor))
                return
            deadline = self._heartbeat_deadline(neighbor)
            # Equality remains alive; wake just after the deadline.
            yield self.env.timeout(math.nextafter(deadline, math.inf) - self.env.now)

    def on_neighbor_state_received(self, neighbor, sent_at=None):
        if self.mode == 'New_DQN' and self.historical_self_feature_elimination:
            if not hasattr(self, 'aggregation_resource_history'):
                self.aggregation_resource_history = {}
            records = self.aggregation_resource_history.setdefault(neighbor, [])
            records.append((self.env.now, sent_at, list(self.neighbor_states[neighbor])))
            del records[:-2]
        if sent_at is not None:
            self.last_heartbeat_sent_times[neighbor] = max(
                sent_at, self.last_heartbeat_sent_times.get(neighbor, -float('inf')),
            )
        self.neighbor_state_versions[neighbor] = self.neighbor_state_versions.get(neighbor, 0) + 1
        event = self.neighbor_state_events.get(neighbor)
        if event is not None and not event.triggered:
            event.succeed()
        self.neighbor_state_events[neighbor] = self.env.event()
        self._schedule_waiting_forward_retry()

    def _schedule_waiting_forward_retry(self):
        if not (self.mode in ('Pure_DQN', 'New_DQN', 'Pure_PPO') or getattr(self, 'aligned_graphsage_policy', False)):
            self._retry_waiting_forward()
            return
        cycle = int(np.floor(self.env.now / self.state_update_period + 1e-9))
        if not hasattr(self, '_waiting_retry_cycles'):
            self._waiting_retry_cycles = set()
        if cycle in self._waiting_retry_cycles:
            return
        self._waiting_retry_cycles.add(cycle)
        self.env.process(self._retry_waiting_forward_at(cycle))

    def _retry_waiting_forward_at(self, update_cycle):
        # Updates in [29.1, 29.2) are consumed together at 29.25.
        retry_time = (update_cycle + 1.5) * self.state_update_period
        yield self.env.timeout(max(0., retry_time - self.env.now))
        self._waiting_retry_cycles.discard(update_cycle)
        if self.active:
            self._retry_waiting_forward(retry_cycle=update_cycle + 1)
            if self.waiting_forward:
                self._schedule_waiting_forward_retry()

    def _retry_waiting_forward(self, retry_cycle=None):
        ready_packets = []
        for _ in range(len(self.waiting_forward)):
            packet = self.waiting_forward.popleft()
            if retry_cycle is not None and getattr(packet, 'wait_retry_cycle', 0) > retry_cycle:
                self.waiting_forward.append(packet)
                continue
            if (self.mode in ('Pure_DQN', 'New_DQN', 'Pure_PPO') or getattr(self, 'aligned_graphsage_policy', False)):
                packet.wait_count = getattr(packet, 'wait_count', 0) + 1
                can_act = packet.destination == self.name or self.valid_action_mask(
                    packet.destination, packet.size, packet.information[0], packet.hops,
                    previous_hop=packet.previous_hop,
                    postcompute_previous_hop=getattr(packet, 'postcompute_previous_hop', None),
                ).any()
                if not can_act:
                    if packet.wait_count >= PASSIVE_WAIT_MAX_RETRIES:
                        record_queue_wait(packet, 'waiting_forward', self.env.now)
                        self._drop_no_route(packet, packet.size, packet.information[1],
                                            packet.information[2], packet.creation_time,
                                            reason='wait_limit')
                    else:
                        packet.wait_retry_cycle = (retry_cycle + 1 if retry_cycle is not None
                                                   else int(np.floor(self.env.now / self.state_update_period)) + 1)
                        self.waiting_forward.append(packet)
                    continue
            if not (self.mode in ('Pure_DQN', 'New_DQN', 'Pure_PPO') or getattr(self, 'aligned_graphsage_policy', False)) and packet.destination != self.name and (
                    packet.destination not in self.routing_tables
                    or not self.valid_action_mask(
                        packet.destination, packet.size,
                        packet.information[0], packet.hops,
                        previous_hop=packet.previous_hop,
                        postcompute_previous_hop=getattr(packet, 'postcompute_previous_hop', None),
                    )[:4].any()):
                self.waiting_forward.append(packet)
                continue
            record_queue_wait(packet, 'waiting_forward', self.env.now)
            mark_queue_entry(packet, 'forward', self.env.now)
            ready_packets.append(packet)
        queued_count = len(self.forward_queue.items)
        for packet in ready_packets:
            self.forward_queue.put(packet)
        # Promote only ready waiting packets, retaining FIFO order in both groups.
        # Keep Store.put() so a pending consumer is also woken normally.
        self.forward_queue.items[:] = (
            self.forward_queue.items[queued_count:]
            + self.forward_queue.items[:queued_count]
        )

    def update_topology_slots(self, neighbors):
        """Keep fixed action slots; unavailable slots use fault-filled state."""
        if not hasattr(self, 'action_neighbors'):
            return
        expected = sorted(
            neighbor for neighbor in neighbors
            if neighbor in self.neighbors or neighbor in self.action_neighbors
        )
        if len(expected) > 4:
            raise ValueError('RL policy supports at most four forwarding actions')

        current = list(self.action_neighbors)
        updated = [neighbor if neighbor in expected else None for neighbor in current]
        missing = [neighbor for neighbor in expected if neighbor not in updated]
        for index, neighbor in enumerate(updated):
            if neighbor is None and missing:
                updated[index] = missing.pop(0)
        self.topology_neighbors = tuple(expected)
        self.action_neighbors = updated

    def _empty_neighbor_state_chunk(self):
        if 'New' in self.mode:
            return [1, 0, 0, 1, 1, self._local_fault_compute_score()] * 3 + [1, 2]
        if 'GNN' in self.mode:
            return [1, 2]
        if self.mode in ('Pure_DQN', 'Pure_PPO'):
            return [1, 0, 0, 1, 1, self._local_fault_compute_score(), 1, 2]
        return [1, 0, 0, 1, 1, 1, 2]

    def _fault_neighbor_state(self):
        """Return the local neighbor-state form used for an unavailable slot."""
        if 'New' in self.mode or getattr(self, 'gosr_use_action_slots', False):
            return [value for scale in (1, 4, 12)
                    for value in [1, 0, 0, 1, 1, scale * self._local_fault_compute_score()]]
        if (self.mode in ('Pure_DQN', 'Pure_PPO')
                or ('GNN' in self.mode and self.gnn_node_feature_dim == ISACR_NODE_FEATURE_DIM)):
            return [1, 0, 0, 1, 1, self._local_fault_compute_score()]
        return [1, 0, 0, 1, 1]

    def _effective_memory_remain(self):
        """Return the node's actual free-memory ratio."""
        return 1.0 - self.current_memory_occupy / self.memory

    def _refresh_transmission_features(self):
        """Refresh per-neighbor queue features from the real SimPy queues."""
        for neighbor, queue in self.transmission_queue.items():
            items = list(queue.items)
            self.transmission_length[neighbor] = len(items)
            self.transmission_size[neighbor] = sum(packet.size for packet in items)

    def _projected_neighbor_memory_remain(self, neighbor, packet_size):
        """Return the slot state's free ratio after admitting the current packet."""
        if (getattr(getattr(self, 'gnn', None), 'direct_concat', False)
                and self.fresh_direct_concat_view(neighbor) is None):
            return -float('inf')
        state = self.neighbor_states.get(neighbor, self._fault_neighbor_state())
        return float(state[1]) - float(packet_size) / self.memory

    def _memory_allowed_action_indices(self, packet_size):
        """Return live neighbor slots that can admit the packet."""
        action_neighbors = (
            self.action_neighbors if hasattr(self, 'action_neighbors') else self.neighbors
        )
        return {
            index for index, neighbor in enumerate(action_neighbors)
            if neighbor in self.neighbors
            and (self.mode not in ('New_DQN', 'Pure_DQN', 'Pure_PPO')
                 and not getattr(self, 'aligned_graphsage_policy', False)
                 and not getattr(self, 'gosr_use_action_slots', False) or self.neighbor_is_alive(neighbor))
            and self._projected_neighbor_memory_remain(neighbor, packet_size) >= 0.0
        }

    def _update_neighbor_memory_after_forward(self, neighbor, packet_size):
        """Debit receiving memory; forwarding does not assign CPU work."""
        projected = self._projected_neighbor_memory_remain(neighbor, packet_size)
        self.neighbor_states[neighbor][1] = projected
        return projected

    def _live_action_indices(self, packet_size=None):
        """Return live action slots, optionally checking projected memory."""
        action_neighbors = (
            self.action_neighbors if hasattr(self, 'action_neighbors') else self.neighbors
        )
        live = [
            (index, neighbor) for index, neighbor in enumerate(action_neighbors)
            if neighbor in self.neighbors
        ]
        if packet_size is not None:
            memory_allowed = self._memory_allowed_action_indices(packet_size)
            live = [item for item in live if item[0] in memory_allowed]

        return {index for index, _ in live}

    def valid_action_mask(self, destination, packet_size, is_computed, hops, wait_count=0,
                          previous_hop=None, postcompute_previous_hop=None):
        """Five policy actions; route waiting is handled by the environment."""
        mask = np.zeros(5, dtype=bool)
        if destination != self.name and hops >= 2 * self.max_hop:
            return mask
        if not is_computed:
            mask[4] = True
        if destination == self.name and not is_computed:
            return mask

        # A slot needs both a live neighbor and enough projected memory.
        memory_allowed = self._memory_allowed_action_indices(packet_size)
        action_neighbors = getattr(self, 'action_neighbors', self.neighbors)
        remaining_hops = 2 * self.max_hop - hops - 1
        memory_allowed = {
            index for index in memory_allowed
            if self.neighbor_hops.get(action_neighbors[index], {}).get(
                destination, float('inf')
            ) <= remaining_hops
        }
        if hasattr(self, 'action_neighbors'):
            for index in range(min(4, len(self.action_neighbors))):
                if index in memory_allowed:
                    mask[index] = True
        else:
            for index in memory_allowed:
                if 0 <= index < 4:
                    mask[index] = True
        # Return to a previous node only when it is strictly closer to the
        # destination. Physical admission and hop-budget checks still apply.
        current_route = self.routing_tables.get(destination)
        for index, neighbor in enumerate(action_neighbors[:4]):
            if previous_hop is not None and neighbor == previous_hop:
                neighbor_distance = self.neighbor_hops.get(neighbor, {}).get(
                    destination, float('inf')
                )
                if current_route is None or not neighbor_distance < current_route[1]:
                    mask[index] = False
        if is_computed and postcompute_previous_hop is not None:
            for index, neighbor in enumerate(action_neighbors[:4]):
                if neighbor == postcompute_previous_hop:
                    neighbor_distance = self.neighbor_hops.get(neighbor, {}).get(
                        destination, float('inf')
                    )
                    if current_route is None or not neighbor_distance < current_route[1]:
                        mask[index] = False
        if (not is_computed and not mask[:4].any()
                and (self.mode in ('New_DQN', 'Pure_DQN', 'Pure_PPO')
                     or getattr(self, 'aligned_graphsage_policy', False))):
            cpu_wait = max(0.0, self.computing_remain / self.computing_ability
                           - self.is_computing * (self.env.now - self.last_computing_time))
            if cpu_wait > LOCAL_COMPUTE_WAIT_THRESHOLD:
                mask[4] = False
        return mask

    def replay_action_mask(self, physical_mask, action_mapping=None):
        if 'DQN' not in self.mode or 'PPO' in self.mode:
            return None
        if action_mapping is None:
            return physical_mask
        mapped = np.zeros_like(physical_mask, dtype=bool)
        for policy_index, physical_index in enumerate(np.asarray(action_mapping, dtype=int)):
            mapped[policy_index] = physical_mask[physical_index]
        mapped[4:] = physical_mask[4:]
        return mapped

    def trans_parameters(self,q_net, reward_function, direct_satellites, gnn=None, ppo_shuffle_func=None):
        self.q_net=q_net
        self.gnn=gnn
        self.ppo_shuffle_func=ppo_shuffle_func
        self.reward_function = reward_function
        self.direct_satellites = direct_satellites

    def build_routing_table(self):
        routing_adjacency = self.update_adjacency_dict_for_bfs()
        adjacency = {node: sorted(value[0]) for node, value in routing_adjacency.items()}
        for neighbor in self._neighbor_route_since:
            adjacency[neighbor] = []
        for neighbor in self.neighbors:
            adjacency.setdefault(neighbor, [])
        cache_key = (self.name, tuple(self.neighbors), adjacency)
        cached = getattr(self, '_routing_cache', None)
        if cached is not None and cached[0] == cache_key:
            self.routing_tables = {node: (list(next_hops), hops)
                                   for node, (next_hops, hops) in cached[1].items()}
            self.neighbor_hops = {node: dict(hops) for node, hops in cached[2].items()}
            return
        result_dict = {}
        self.neighbor_hops = {neighbor: {} for neighbor in self.neighbors}
        for start in [self.name] + self.neighbors:
            if start == self.name:
                queue = deque(
                    (neighbor, neighbor, 1)
                    for neighbor in adjacency[start]
                )
                best_hops = {}
                while queue:
                    node, first_hop, hops = queue.popleft()
                    if node not in best_hops:
                        best_hops[node] = hops
                        result_dict[node] = ([first_hop], hops)
                        queue.extend(
                            (neighbor, first_hop, hops + 1)
                            for neighbor in adjacency[node]
                            if neighbor not in best_hops
                        )
                    elif best_hops[node] == hops:
                        result_dict[node][0].append(first_hop)
            else:
                neighbor_hops = self.neighbor_hops[start]
                neighbor_hops[start] = 0
                queue = deque(
                    (neighbor, 1)
                    for neighbor in adjacency[start]
                )
                while queue:
                    node, hops = queue.popleft()
                    if node not in neighbor_hops:
                        neighbor_hops[node] = hops
                        queue.extend(
                            (neighbor, hops + 1)
                            for neighbor in adjacency[node]
                            if neighbor not in neighbor_hops
                        )
        result_dict[self.name] = ([self.name], 0)
        self.routing_tables = result_dict
        self._routing_cache = (
            cache_key,
            {node: (list(next_hops), hops) for node, (next_hops, hops) in result_dict.items()},
            {node: dict(hops) for node, hops in self.neighbor_hops.items()},
        )

    def update_adjacency_table(self, table):
        for neighbor in self.pending_state_neighbors:
            record = table.get(neighbor)
            if (record is not None
                    and record[1] >= self.pending_link_since.get(neighbor, self.env.now)
                    and self.name in record[0]):
                previous = self.pending_adjacency_records.get(neighbor)
                if previous is None or record[1] >= previous[1]:
                    self.pending_adjacency_records[neighbor] = (list(record[0]), record[1])
        super().update_adjacency_table(table)
        refreshed = [
            neighbor for neighbor, since in self._neighbor_route_since.items()
            if neighbor in table
            and neighbor in self.adjacency_table
            and self.adjacency_table[neighbor][1] >= since
            and self.name in self.adjacency_table[neighbor][0]
        ]
        if refreshed:
            for neighbor in refreshed:
                del self._neighbor_route_since[neighbor]
            self.build_routing_table()
            self._schedule_waiting_forward_retry()

    def push_forward(self,packet,from_transmission=False):
        """Reject an incoming packet if memory is full; preserve waiting packets."""
        reservation_book = getattr(self.propagator, 'reservation_book', None)
        can_accept = self.current_memory_occupy + packet.size <= self.memory and (from_transmission or (
            reservation_book.can_accept(self.name, packet.size, packet)
            if reservation_book is not None
            else True
        ))
        if can_accept:
            if reservation_book is not None:
                reservation_book.arrive(packet, self.name)
            self.current_memory_occupy += packet.size
            mark_queue_entry(packet, 'forward', self.env.now)
            self.forward_queue.put(packet)
            if self.mode in ["Tradition","Ground"]:
                if 'current_memory_occupy' in self.propagator.graph.nodes[self.name]:
                    self.propagator.graph.nodes[self.name]['current_memory_occupy'] += packet.size
                else:
                    self.propagator.graph.nodes[self.name]['current_memory_occupy'] = packet.size
            return True
        else:
            return False

    def push_computing(self,packet,computing_demand):
        self.current_computing_queue_size += packet.size
        self.computing_remain += computing_demand
        mark_queue_entry(packet, 'computing', self.env.now)
        self.computing_queue.put(packet)
        if not PRE:
            if 'computing_remain' in self.propagator.graph.nodes[self.name]:
                self.propagator.graph.nodes[self.name]['computing_remain'] += computing_demand
            else:
                self.propagator.graph.nodes[self.name]['computing_remain'] = computing_demand

    def push_transmission(self,neighbor,packet):
        self.current_queue_size += packet.size
        self.transmission_length[neighbor]+=1
        self.transmission_size[neighbor]+=packet.size
        mark_queue_entry(packet, f'transmission:{neighbor}', self.env.now)
        self.transmission_queue[neighbor].put(packet)
        if self.mode in ["Tradition","Ground"]:
            if 'transmission_weight' in self.propagator.graph[self.name][neighbor]:
                self.propagator.graph[self.name][neighbor]['transmission_weight']+=packet.size
            else:
                self.propagator.graph[self.name][neighbor]['transmission_weight'] = packet.size
        return True

    def pop_transmission(self,neighbor,retain_memory=False):
        packet = yield self.transmission_queue[neighbor].get()
        record_queue_wait(packet, f'transmission:{neighbor}', self.env.now)
        self.current_queue_size-=packet.size
        self.transmission_size[neighbor]-=packet.size
        self.transmission_length[neighbor]-=1
        if not retain_memory:
            self.current_memory_occupy -= packet.size
        if self.mode in ["Tradition","Ground"]:
            if self.propagator.graph.has_edge(self.name, neighbor):
                self.propagator.graph[self.name][neighbor]['transmission_weight'] -= packet.size
            if not retain_memory:
                self.propagator.graph.nodes[self.name]['current_memory_occupy'] -= packet.size
        return packet

    def push_offload(self,packet):
        self.current_queue_size += packet.size
        self.offload_size+=packet.size
        self.offload_length+=1
        mark_queue_entry(packet, 'offload', self.env.now)
        self.offload_queue.put(packet)

    def pop_offload(self,retain_memory=False):
        packet = yield self.offload_queue.get()
        record_queue_wait(packet, 'offload', self.env.now)
        self.current_queue_size-=packet.size
        self.offload_size-=packet.size
        self.offload_length-=1
        if not retain_memory:
            self.current_memory_occupy -= packet.size
        if not retain_memory and self.mode in ["Tradition","Ground"]:
            self.propagator.graph.nodes[self.name]['current_memory_occupy'] -= packet.size
        return packet

    def get_current_state(self, destination, hops, is_computed,mission_state):
        self._refresh_transmission_features()
        current_state = []
        neighbors_state = []
        state_neighbors = self.neighbors
        if self.mode in ('New_DQN', 'Pure_DQN', 'Pure_PPO'):
            state_neighbors = [neighbor for neighbor in state_neighbors
                               if self.neighbor_has_resource_state(neighbor)]
        feature_dim = (ISACR_NODE_FEATURE_DIM
                       if 'New' in self.mode or self.mode in ('Pure_DQN', 'Pure_PPO')
                       else NODE_FEATURE_DIM)
        remaining_queue_seconds = (
            self.computing_remain / self.computing_ability
            - self.is_computing * (self.env.now - self.last_computing_time)
        )
        current_node_state=[
            self.is_producing,
            1 - self.current_memory_occupy / self.memory,
            self._normalized_compute_ability(),
            self._normalized_compute_time(remaining_queue_seconds),
        ]
        if (self.mode in ('New_DQN', 'Pure_DQN', 'Pure_PPO')
                or ('GNN' in self.mode and self.gnn_node_feature_dim == ISACR_NODE_FEATURE_DIM)):
            current_node_state.append(self._normalized_compute_completion_time())
        for neighbor in state_neighbors:
            neighbor_values = list(self.neighbor_states[neighbor])
            if 'New' in self.mode:
                neighbors_state.extend(
                    neighbor_values[0:feature_dim]
                    + [x / 4 for x in neighbor_values[feature_dim:2 * feature_dim]]
                    + [x / 12 for x in neighbor_values[2 * feature_dim:3 * feature_dim]]
                )
            elif 'GNN' in self.mode:
                # The graph encoder already carries the neighbor node features.
                # Keep only the action-slot-local queue and distance values here.
                pass
            else:
                neighbors_state.extend(neighbor_values)
            neighbors_state.append(self._normalized_link_transmission_wait(neighbor))
            if destination in self.neighbor_hops[neighbor]:
                neighbors_state.append(self.neighbor_hops[neighbor][destination] / self.max_hop)
            else:
                neighbors_state.append(2)
        if len(state_neighbors) < 4:
            if 'New' in self.mode:
                if neighbors_state and self.fault_compute_double_mean:
                    ability_multiplier = 1 / self.fault_fill_multiplier
                    ca1, ca2, ca3 = (
                        _clip_normalized(
                            ability_multiplier
                            * sum(neighbors_state[2::NEIGHBOR_SLOT_DIM])
                            / len(state_neighbors)
                        ),
                        _clip_normalized(
                            ability_multiplier
                            * sum(neighbors_state[8::NEIGHBOR_SLOT_DIM])
                            / len(state_neighbors)
                        ),
                        _clip_normalized(
                            ability_multiplier
                            * sum(neighbors_state[14::NEIGHBOR_SLOT_DIM])
                            / len(state_neighbors)
                        ),
                    )
                    cv1,cv2,cv3=(_clip_normalized(self.fault_fill_multiplier*sum(neighbors_state[3::NEIGHBOR_SLOT_DIM])/len(state_neighbors)),_clip_normalized(self.fault_fill_multiplier*sum(neighbors_state[9::NEIGHBOR_SLOT_DIM])/len(state_neighbors)),_clip_normalized(self.fault_fill_multiplier*sum(neighbors_state[15::NEIGHBOR_SLOT_DIM])/len(state_neighbors)))
                else:
                    ca1, ca2, ca3 = 0, 0, 0
                    cv1,cv2,cv3=1,1,1
                if neighbors_state:
                    fill_multiplier = self.fault_fill_multiplier if self.fault_compute_double_mean else 2
                    av1,av2,av3=(_clip_normalized(fill_multiplier*sum(neighbors_state[4::NEIGHBOR_SLOT_DIM])/len(state_neighbors)),_clip_normalized(fill_multiplier*sum(neighbors_state[10::NEIGHBOR_SLOT_DIM])/len(state_neighbors)),_clip_normalized(fill_multiplier*sum(neighbors_state[16::NEIGHBOR_SLOT_DIM])/len(state_neighbors)))
                else:
                    av1,av2,av3=1,1,1
                if neighbors_state and self.fault_compute_double_mean:
                    g1, g2, g3 = (
                        _clip_normalized(sum(neighbors_state[offset + 5::NEIGHBOR_SLOT_DIM])
                                         / len(state_neighbors) / self.fault_fill_multiplier)
                        for offset in (0, 6, 12)
                    )
                else:
                    g1, g2, g3 = 0, 0, 0
            elif 'GNN' in self.mode:
                cv, av = 1, 1
            elif neighbors_state and (self.fault_compute_double_mean or self.mode in ('Pure_DQN', 'Pure_PPO')):
                ability_multiplier = 1 / self.fault_fill_multiplier
                ca = _clip_normalized(
                    ability_multiplier
                    * sum(neighbors_state[2::(feature_dim + 2)])
                    / len(state_neighbors)
                ) if self.fault_compute_double_mean else 0
                cv = (_clip_normalized(self.fault_fill_multiplier * sum(neighbors_state[3::(feature_dim + 2)]) / len(state_neighbors))
                      if self.fault_compute_double_mean else 1)
                fill_multiplier = self.fault_fill_multiplier if self.fault_compute_double_mean else 2
                av = _clip_normalized(fill_multiplier * sum(neighbors_state[4::(feature_dim + 2)]) / len(state_neighbors))
            else:
                ca = 0
                cv,av=1,1
            if self.mode in ('Pure_DQN', 'Pure_PPO'):
                g = (_clip_normalized(sum(neighbors_state[5::(feature_dim + 2)])
                                      / len(state_neighbors) / self.fault_fill_multiplier)
                     if neighbors_state and self.fault_compute_double_mean else 0)
            for _ in range(4 - len(state_neighbors)):
                if 'New' in self.mode:
                    neighbors_state.extend([1,0,ca1,cv1,av1,g1,1,0,ca2,cv2,av2,g2,1,0,ca3,cv3,av3,g3,1,2])
                elif 'GNN' in self.mode:
                    # Missing actions receive only a fixed invalid-slot token.
                    # Do not reuse iSatCR's learned fault-fill rule in GNN baselines.
                    neighbors_state.extend([1, 2])
                else:
                    fault_state = [1,0,ca,cv,av]
                    if self.mode in ('Pure_DQN', 'Pure_PPO'):
                        fault_state.append(g)
                    neighbors_state.extend(fault_state + [1,2])
        if hasattr(self, 'action_neighbors'):
            slot_dim = (2 if getattr(self, 'aligned_graphsage_policy', False)
                        else NEIGHBOR_SLOT_DIM if self.mode == 'New_DQN' else feature_dim + 2)
            chunks = [neighbors_state[i:i + slot_dim]
                      for i in range(0, len(neighbors_state), slot_dim)]
            live_chunks = dict(zip(state_neighbors, chunks))
            fault_chunk = chunks[len(state_neighbors)] if len(state_neighbors) < 4 else None
            missing_chunk = fault_chunk or self._empty_neighbor_state_chunk()
            neighbors_state = [
                value
                for neighbor in self.action_neighbors
                for value in (
                    (missing_chunk if self.mode in ('New_DQN', 'Pure_DQN', 'Pure_PPO') else self._empty_neighbor_state_chunk())
                    if neighbor is None
                    else live_chunks.get(neighbor, missing_chunk)
                    if not getattr(self, 'aligned_graphsage_policy', False) or self.neighbor_is_alive(neighbor)
                    else missing_chunk
                )
            ]
        if is_computed and self.mode in ('New_DQN', 'Pure_DQN', 'Pure_PPO'):
            # Mask only this packet's policy input; keep shared resource messages intact.
            slot_dim = NEIGHBOR_SLOT_DIM if self.mode == 'New_DQN' else feature_dim + 2
            resource_layers = 3 if self.mode == 'New_DQN' else 1
            for slot_start in range(0, len(neighbors_state), slot_dim):
                for layer in range(resource_layers):
                    compute_start = slot_start + layer * feature_dim + 2
                    neighbors_state[compute_start:compute_start + 2] = [0.0, 0.0]
                    neighbors_state[slot_start + layer * feature_dim + 5] = 0.0
            current_node_state[2:4] = [0.0, 0.0]
            current_node_state[4] = 0.0
        if is_computed and 'GNN' in self.mode and self.gnn_node_feature_dim == ISACR_NODE_FEATURE_DIM:
            current_node_state[2:5] = [0.0, 0.0, 0.0]
        current_state.extend(neighbors_state)
        current_state.extend(current_node_state)
        current_state.extend(mission_state)
        current_state.extend([hops / self.max_hop,is_computed])
        if 'GNN' in self.mode:
            if self.gnn_node_feature_dim in (NODE_FEATURE_DIM, ISACR_NODE_FEATURE_DIM):
                embeddings = []
                embedding_dim = getattr(self.gnn, 'output_dim', self.gat_node_dim)
                missing_embedding = getattr(self, 'gnn_direct_concat_fault_view',
                                            getattr(self, 'gnn_fault_embedding', [0.0] * embedding_dim))
                if getattr(self.gnn, 'direct_concat', False) and not hasattr(self, 'gnn_direct_concat_fault_view'):
                    missing_embedding = [1, 0, 0, 1, 1, self._local_fault_compute_score()] * 21
                if getattr(self.gnn, 'direct_concat', False):
                    views = {neighbor: self.fresh_direct_concat_view(neighbor) for neighbor in self.neighbors}
                    valid_views = [view for view in views.values() if view is not None]
                    missing_embedding = [value for start in range(0, embedding_dim, 6)
                                         for value in self.direct_concat_fault_feature(
                                             [view[start:start + 6] for view in valid_views])]
                slots = getattr(self, 'action_neighbors', self.neighbors)
                for neighbor in slots:
                    if getattr(self.gnn, 'direct_concat', False):
                        embeddings.extend(views.get(neighbor) or missing_embedding)
                    else:
                        embeddings.extend(self.gnn_active_neighbor_embeddings.get(
                            neighbor, missing_embedding
                        ) if neighbor in self.neighbors else missing_embedding)
                for _ in range(4 - len(slots)):
                    embeddings.extend(missing_embedding)
                if getattr(self.gnn, 'direct_concat', False) and is_computed:
                    for start in range(0, len(embeddings), 6):
                        embeddings[start + 2:start + 4] = [0.0, 0.0]
                        embeddings[start + 5] = 0.0
                    current_state[len(neighbors_state) + 2:len(neighbors_state) + 4] = [0.0, 0.0]
            if (not self.encode_data) or (self.env.now - self.encode_data_update_time > 0.05):
                self.encode_data_update_time = self.env.now
                nodes, edges = self.construct_gnn_graph()
                self.encode_data = self.encode_for_GNN(nodes, edges)
            data = self.encode_data.clone()
            if self.gnn_node_feature_dim in (NODE_FEATURE_DIM, ISACR_NODE_FEATURE_DIM):
                data.decision_embeddings = torch.tensor([embeddings], dtype=torch.float)
                if getattr(self.gnn, 'direct_concat', False):
                    direct_dim = self.gnn.output_dim
                    direct_slots = torch.tensor(embeddings, dtype=torch.float).view(4, direct_dim)
                    data.direct_concat_embeddings = torch.zeros(
                        (data.x.size(0), direct_dim), dtype=torch.float
                    )
                    data.direct_concat_embeddings[:4] = direct_slots
            if (is_computed and self.gnn_node_feature_dim == ISACR_NODE_FEATURE_DIM
                    and not getattr(self.gnn, 'direct_concat', False)
                    and not getattr(self, 'aligned_graphsage_policy', False)):
                data = data.clone()
                data.x[:, 2:4] = 0
                data.x[:, 5] = 0
                with torch.no_grad():
                    data.decision_embeddings = self.gnn(data)[:4].reshape(1, -1)
            return [data, np.array(current_state)]
        else:
            return np.array(current_state)

    def tradition_routing(self,current_state,computing=True):
        if self.mode == "Tradition" and self.tradition_use_genetic_search:
            return self.genetic_tradition_routing(current_state, computing)
        return self.legacy_tradition_routing(current_state, computing)

    def genetic_tradition_routing(self, current_state, computing=True):
        planning_graph = self.propagator.graph
        _, destination, size, computing_demand, size_after_computing, is_computed = current_state
        source = self.name
        for node1, node2 in planning_graph.edges():
            edge = planning_graph[node1][node2]
            edge['weight'] = (
                edge.get('transmission_weight', 0) / self.transmission_rate
                + edge['propagation_weight']
            )
            if max(
                planning_graph.nodes[node1].get('current_memory_occupy', 0),
                planning_graph.nodes[node2].get('current_memory_occupy', 0),
            ) / self.memory > 0.9:
                edge['weight'] += 10

        def node_is_available(node):
            satellite = self.propagator.satellites.get(node)
            return satellite is not None and satellite.active

        available_graph = nx.subgraph_view(
            planning_graph,
            filter_node=node_is_available,
            filter_edge=lambda node1, node2: not planning_graph[node1][node2].get(
                'missing', 1
            ),
        )
        if source not in available_graph or destination not in available_graph:
            return [None], None
        computing_nodes = (
            [source]
            if is_computed else sorted(
                node for node in available_graph.nodes
                if planning_graph.nodes[node].get('computing_ability', 0) > 0
            )
        )
        if not computing_nodes:
            return [None], None

        path_count = max(int(self.tradition_ga_path_candidates), 1)
        path_cache = {}

        def path_options(path_source, path_target, forbid_destination=False):
            key = (path_source, path_target, forbid_destination)
            if key in path_cache:
                return path_cache[key]
            if path_source == path_target:
                path_cache[key] = [([], 0.0)]
                return path_cache[key]
            forbidden = {destination} if forbid_destination else set()
            route_graph = nx.subgraph_view(
                available_graph,
                filter_node=lambda node: (
                    node not in forbidden or node in {path_source, path_target}
                ),
            )
            try:
                full_paths = islice(
                    nx.shortest_simple_paths(
                        route_graph, path_source, path_target, weight='weight'
                    ),
                    path_count,
                )
                options = []
                for full_path in full_paths:
                    cost = sum(
                        route_graph[node1][node2]['weight']
                        for node1, node2 in zip(full_path[:-1], full_path[1:])
                    )
                    options.append((full_path[1:], cost))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                options = []
            path_cache[key] = options
            return options

        evaluation_cache = {}

        def evaluate(chromosome):
            if chromosome in evaluation_cache:
                return evaluation_cache[chromosome]
            computing_node, first_path_index, second_path_index = chromosome
            node_state = planning_graph.nodes[computing_node]
            if (not is_computed
                    and node_state.get('current_memory_occupy', 0) + size > self.memory):
                result = (float('inf'), None)
                evaluation_cache[chromosome] = result
                return result
            first_options = path_options(
                source, computing_node,
                forbid_destination=computing_node != destination,
            )
            second_options = path_options(computing_node, destination)
            if not first_options or not second_options:
                result = (float('inf'), None)
                evaluation_cache[chromosome] = result
                return result
            first_route, first_cost = first_options[first_path_index % len(first_options)]
            second_route, second_cost = second_options[second_path_index % len(second_options)]
            first_packet_size = size
            second_packet_size = size if is_computed else size_after_computing
            if any(
                planning_graph.nodes[node].get('current_memory_occupy', 0)
                + first_packet_size > self.memory
                for node in first_route
            ) or any(
                planning_graph.nodes[node].get('current_memory_occupy', 0)
                + second_packet_size > self.memory
                for node in second_route
            ):
                result = (float('inf'), None)
                evaluation_cache[chromosome] = result
                return result
            if is_computed:
                computing_cost = 0.0
            else:
                computing_ability = node_state['computing_ability']
                computing_cost = (
                    node_state.get('computing_remain', 0) / computing_ability
                    + computing_demand / computing_ability
                )
            transmission_cost = (
                len(first_route) * first_packet_size
                + len(second_route) * second_packet_size
            ) / self.transmission_rate
            predicted_delay = (
                computing_cost + first_cost + second_cost + transmission_cost
                + self.propagator.downstream_delays
            )
            # Keep delay in the heuristic route-planning score.  This is not
            # the terminal delivery reward used by the learning algorithms.
            estimated_reward = (
                self.propagator.reward_function.reach_factor
                - self.propagator.reward_function.delay_factor * predicted_delay
            )
            result = (
                -estimated_reward,
                first_route + second_route,
            )
            evaluation_cache[chromosome] = result
            return result

        population_size = max(int(self.tradition_ga_population_size), 2)
        generations = max(int(self.tradition_ga_generations), 1)
        mutation_rate = min(max(float(self.tradition_ga_mutation_rate), 0.0), 1.0)
        rng = random.Random(
            f'ics-ga:{self.name}:{destination}:{self.env.now:.9f}:'
            f'{size}:{computing_demand}'
        )
        ranked_nodes = sorted(
            computing_nodes,
            key=lambda node: (
                planning_graph.nodes[node].get('computing_remain', 0)
                / planning_graph.nodes[node]['computing_ability']
            ),
        )
        seeded_nodes = list(dict.fromkeys(
            [self.name, destination] + ranked_nodes[:population_size]
        ))
        population = [
            (node, 0, 0) for node in seeded_nodes if node in computing_nodes
        ][:population_size]
        while len(population) < population_size:
            population.append((
                rng.choice(computing_nodes),
                rng.randrange(path_count),
                rng.randrange(path_count),
            ))

        def select_parent():
            tournament = rng.sample(population, min(3, len(population)))
            return min(tournament, key=lambda chromosome: evaluate(chromosome)[0])

        for _ in range(generations):
            population.sort(key=lambda chromosome: evaluate(chromosome)[0])
            next_population = population[:2]
            while len(next_population) < population_size:
                parent1, parent2 = select_parent(), select_parent()
                child = (
                    parent1[0] if rng.random() < 0.5 else parent2[0],
                    parent1[1] if rng.random() < 0.5 else parent2[1],
                    parent1[2] if rng.random() < 0.5 else parent2[2],
                )
                if rng.random() < mutation_rate:
                    child = (rng.choice(computing_nodes), child[1], child[2])
                if rng.random() < mutation_rate:
                    child = (child[0], rng.randrange(path_count), child[2])
                if rng.random() < mutation_rate:
                    child = (child[0], child[1], rng.randrange(path_count))
                next_population.append(child)
            population = next_population

        best = min(population, key=lambda chromosome: evaluate(chromosome)[0])
        best_cost, routing = evaluate(best)
        if routing is None or not np.isfinite(best_cost):
            return [None], None
        computing_decision = None if is_computed else best[0]
        if PRE and computing and computing_decision is not None:
            planning_graph.nodes[computing_decision]['computing_remain'] = (
                planning_graph.nodes[computing_decision].get('computing_remain', 0)
                + computing_demand
            )
        return routing, computing_decision if computing else None

    def legacy_tradition_routing(self,current_state,computing=True):
        planning_graph = (
            self.propagator.graph if self.mode == "Tradition" else self.neighbor_graph
        )
        def shortest_path_and_cost(source, target, forbidden_nodes=()):
            if source == target:
                return [], 0
            else:
                try:
                    route_graph = nx.subgraph_view(
                        planning_graph,
                        filter_node=lambda node: node not in forbidden_nodes,
                    )
                    full_path = nx.shortest_path(
                        route_graph, source=source, target=target, weight='weight'
                    )
                    cost = sum(
                        route_graph[u][v]['weight']
                        for u, v in zip(full_path[:-1], full_path[1:])
                    )
                    return full_path[1:], cost
                except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                    return None, float('inf')
        for edge in planning_graph.edges():
            node1,node2=edge
            planning_graph[node1][node2]['weight']=planning_graph[node1][node2]['missing']*10+planning_graph[node1][node2].get('transmission_weight', 0)/self.transmission_rate+planning_graph[node1][node2]['propagation_weight']
            if max(
                self.propagator.graph.nodes[node1].get('current_memory_occupy', 0),
                self.propagator.graph.nodes[node2].get('current_memory_occupy', 0),
            ) / self.memory > 0.9:
                planning_graph[node1][node2]['weight'] += 10
        source, destination, size, computing_demand, size_after_computing,is_computed = current_state
        if is_computed:
            routing,_=shortest_path_and_cost(self.name, destination)
            return routing, None
        else:
            times={}
            routings={}
            n = max(int(len(planning_graph.nodes) * 1 / 2), 1)
            computing_nodes = [node for node in planning_graph.nodes if planning_graph.nodes[node].get('computing_remain', 0) == 0 and node!=destination]
            non_computing_nodes = sorted((node for node in planning_graph.nodes if planning_graph.nodes[node].get('computing_remain', 0) != 0 and node!=destination),key=lambda k: planning_graph.nodes[k].get('computing_remain', 0))
            computing_nodes += non_computing_nodes[:max(n - len(computing_nodes), 0)]
            for c_node in computing_nodes:
                computing_ability = self.propagator.graph.nodes[c_node]['computing_ability']
                time=planning_graph.nodes[c_node].get('computing_remain', 0)/computing_ability+ computing_demand/computing_ability + (self.propagator.graph.nodes[c_node].get('current_memory_occupy',0)/self.memory > 0.9)*10
                routing_1,time_1=shortest_path_and_cost(
                    self.name, c_node, forbidden_nodes={destination}
                )
                routing_2,time_2=shortest_path_and_cost(c_node, destination)
                if routing_1 is None or routing_2 is None:
                    continue
                times[c_node]=time+time_1+time_2+(len(routing_1)*size+len(routing_2)*size_after_computing)/self.transmission_rate
                routings[c_node]=routing_1+routing_2
            if times:
                computing_decision=sorted(times, key=times.get)[0]
            else:
                return [None],None
            if PRE and computing:
                if 'computing_remain' in self.propagator.graph.nodes[computing_decision]:
                    self.propagator.graph.nodes[computing_decision]['computing_remain'] += computing_demand
                else:
                    self.propagator.graph.nodes[computing_decision]['computing_remain'] = computing_demand
            if not computing:
                return routings[computing_decision], None
            else:
                return routings[computing_decision],computing_decision

    @torch.no_grad()
    def ppo_transition_state(self, current_state):
        """Return the policy-frame state used as a transition next_state.

        The callback is shared by PPO and the non-GNN DQN modes.  DQN now
        stores states after the same action-time neighbor permutation used for
        action selection, so replay no longer needs to reshuffle them.
        """
        if self.ppo_shuffle_func is None or 'GNN' in self.mode:
            return current_state
        policy_state = current_state[1] if 'GNN' in self.mode else current_state
        policy_state, _ = self.ppo_shuffle_func(policy_state)
        return policy_state

    @torch.no_grad()
    def get_next_hop(
        self,
        current_state,
        destination,
        packet_size=None,
        forced_action=None,
        prepared_policy_state=None,
        prepared_action_mapping=None,
        *,
        hops,
        wait_count=0,
        previous_hop=None,
        postcompute_previous_hop=None,
    ):
        def gnn_forward(data, other_state):
            if self.gnn_node_feature_dim in (NODE_FEATURE_DIM, ISACR_NODE_FEATURE_DIM) and hasattr(data, 'decision_embeddings'):
                communicated_state = data.decision_embeddings.to(self.device)
                return torch.cat([communicated_state, other_state], dim=-1)
            gnn_data = Batch.from_data_list([data]).to(self.device)
            with torch.no_grad():
                gnn_state = self.gnn(gnn_data)
            unique_batches = torch.unique(gnn_data.batch).tolist()

            combined_vectors = []
            for batch_idx in unique_batches:
                batch_output = gnn_state[gnn_data.batch == batch_idx]
                combined_vector = batch_output[:4].reshape(-1)
                combined_vectors.append(combined_vector)
            combined_vector = torch.stack(combined_vectors, dim=0)
            return torch.cat([combined_vector, other_state], dim=-1)
        if "GNN" in self.mode:
            data, current_state= current_state[0], current_state[1]
        is_computed = current_state[-1]
        physical_mask = self.valid_action_mask(
            destination, packet_size, is_computed, hops, wait_count, previous_hop,
            postcompute_previous_hop,
        )
        candidate_neighbors = np.flatnonzero(physical_mask[:4]).tolist()
        if not physical_mask.any():
            return None
        if 'DQN' in self.mode:
            policy_shuffle_enabled = (
                self.ppo_shuffle_func is not None
                and ('GNN' not in self.mode or getattr(self, 'aligned_graphsage_policy', False))
            )
            if policy_shuffle_enabled:
                if prepared_policy_state is None:
                    shuffle_state = ([data, current_state]
                                     if getattr(self, 'aligned_graphsage_policy', False) else current_state)
                    policy_state, action_mapping = self.ppo_shuffle_func(shuffle_state)
                else:
                    policy_state = prepared_policy_state
                    action_mapping = (
                        np.arange(4)
                        if prepared_action_mapping is None else prepared_action_mapping
                    )
                action_mapping = np.asarray(action_mapping, dtype=int)
            else:
                policy_state = current_state
                action_mapping = np.arange(4)

            policy_transition_state = policy_state
            if policy_shuffle_enabled and getattr(self, 'aligned_graphsage_policy', False):
                data, policy_state = policy_state

            inverse_action_mapping = np.empty(4, dtype=int)
            inverse_action_mapping[action_mapping] = np.arange(4)

            def policy_action_for_environment(environment_action):
                if environment_action < 4:
                    return int(inverse_action_mapping[environment_action])
                return int(environment_action)

            def environment_action_for_policy(policy_action):
                if policy_action < 4:
                    return int(action_mapping[policy_action])
                return int(policy_action)

            def min_hop_candidate_neighbors():
                action_neighbors = (
                    self.action_neighbors
                    if hasattr(self, 'action_neighbors')
                    else self.neighbors
                )
                distances = {}
                for index in candidate_neighbors:
                    if not 0 <= index < min(4, len(action_neighbors)):
                        continue
                    neighbor = action_neighbors[index]
                    neighbor_hops = self.neighbor_hops.get(neighbor, {})
                    if destination in neighbor_hops and self.max_hop:
                        distances[index] = neighbor_hops[destination] / self.max_hop
                    else:
                        distances[index] = 2
                if not distances:
                    return []
                minimum = min(distances.values())
                return [
                    index for index, distance in distances.items()
                    if distance == minimum
                ]

            if forced_action is not None:
                environment_action = int(forced_action)
                if (
                    environment_action < 4
                    and candidate_neighbors is not None
                    and environment_action not in candidate_neighbors
                ):
                    return None
                policy_action = policy_action_for_environment(environment_action)
            elif not candidate_neighbors:
                legal_actions = np.flatnonzero(physical_mask)
                if np.random.rand() <= self.epsilon:
                    environment_action = int(np.random.choice(legal_actions))
                else:
                    policy_tensor = torch.tensor(policy_state, dtype=torch.float).unsqueeze(0).to(self.device)
                    if 'GNN' in self.mode:
                        policy_tensor = gnn_forward(data, policy_tensor)
                    output = self.q_net(policy_tensor, physical_mask)
                    environment_action = int(legal_actions[torch.argmax(output[0, legal_actions]).item()])
                policy_action = policy_action_for_environment(environment_action)
            elif np.random.rand() <= self.epsilon:
                if np.random.rand() <= 0.5:
                    if not is_computed and np.random.rand() <= 0.2:
                        environment_action = 4
                    elif candidate_neighbors:
                        environment_action = np.random.choice(candidate_neighbors)
                    elif not is_computed:
                        environment_action = 4
                    else:
                        return None
                else:
                    if not is_computed and np.random.rand() <= 0.2:
                        environment_action = 4
                    else:
                        minimum_neighbors = min_hop_candidate_neighbors()
                        if minimum_neighbors:
                            environment_action = np.random.choice(minimum_neighbors)
                        elif not is_computed:
                            environment_action = 4
                        else:
                            return None
                policy_action = policy_action_for_environment(int(environment_action))
            else:
                policy_tensor = torch.tensor(
                    policy_state, dtype=torch.float
                ).unsqueeze(0).to(self.device)
                if "GNN" in self.mode:
                    policy_tensor = gnn_forward(data, policy_tensor)
                action_count = 4 if is_computed else 5
                if candidate_neighbors is not None:
                    candidate_environment_actions = list(candidate_neighbors)
                    if not is_computed:
                        candidate_environment_actions.append(4)
                    if not candidate_environment_actions:
                        return None
                    candidate_indices = [
                        int(inverse_action_mapping[index]) if index < 4 else index
                        for index in candidate_environment_actions
                    ]
                    policy_mask = np.zeros_like(physical_mask)
                    policy_mask[candidate_indices] = True
                    main_output = self.q_net(policy_tensor, policy_mask)
                    candidate_q = main_output[0, candidate_indices].clone()
                    policy_action = candidate_indices[torch.argmax(candidate_q).item()]
                else:
                    policy_mask = np.arange(5) < action_count
                    main_output = self.q_net(policy_tensor, policy_mask)
                    policy_action = torch.argmax(main_output[0, :action_count]).item()
                policy_action = int(policy_action)
                environment_action = environment_action_for_policy(policy_action)
            if policy_shuffle_enabled:
                return [environment_action, policy_action, policy_transition_state]
            return environment_action
        else:
            if prepared_policy_state is None:
                policy_state = current_state
                action_mapping = np.arange(4)
                if self.ppo_shuffle_func and 'GNN' not in self.mode:
                    policy_state, action_mapping = self.ppo_shuffle_func(current_state)
            else:
                policy_state = prepared_policy_state
                action_mapping = (
                    np.arange(4)
                    if prepared_action_mapping is None else prepared_action_mapping
                )
            current_state = torch.tensor(policy_state, dtype=torch.float).unsqueeze(0).to(self.device)
            if "GNN" in self.mode:
                current_state = gnn_forward(data, current_state)
            with torch.no_grad():
                main_output = self.q_net(current_state)
            policy_valid_mask = np.zeros_like(physical_mask)
            if candidate_neighbors is not None:
                candidate_environment_actions = np.flatnonzero(physical_mask).tolist()
                if not candidate_environment_actions:
                    return None
                action_mapping_array = np.asarray(action_mapping)
                candidate_policy_actions = [
                    int(np.where(action_mapping_array == action)[0][0])
                    if action < 4 else action
                    for action in candidate_environment_actions
                ]
                policy_valid_mask[:] = False
                policy_valid_mask[candidate_policy_actions] = True
                masked_output = torch.full_like(main_output, float('-inf'))
                masked_output[:, candidate_policy_actions] = main_output[:, candidate_policy_actions]
                main_output = masked_output
            if forced_action is not None:
                if (
                    int(forced_action) < 4
                    and candidate_neighbors is not None
                    and int(forced_action) not in candidate_neighbors
                ):
                    return None
                policy_action = int(forced_action)
                log_probability = 0.0
                forced_action_flag = True
            elif getattr(self.q_net, 'ppo_deterministic_eval', False):
                policy_action = torch.argmax(main_output, dim=1).item()
                log_probability = 0.0
                forced_action_flag = False
            else:
                dist = torch.distributions.Categorical(logits=main_output)
                action = dist.sample()
                policy_action = action.item()
                log_probability = dist.log_prob(action).item()
                forced_action_flag = False
            environment_action = int(action_mapping[policy_action]) if policy_action < 4 else policy_action
            return [environment_action, [policy_action, log_probability,
                                         getattr(self.q_net, 'ppo_policy_version', 0),
                                         forced_action_flag,
                                         policy_valid_mask.tolist()], policy_state]

    def find_highest_score(self, destinations, mission_state, hops, is_computed):
        highest_score= -2
        mission_state=[mission_state[0], mission_state[1]/ self.max_size, _normalized_computing_demand(mission_state[2]), mission_state[3] / self.max_size]
        best_destinations=[]
        for destination in destinations:
            if destination in self.routing_tables:
                current_state = self.get_current_state(destination, hops, is_computed, mission_state)
                score = self.cal_score(current_state,is_computed)
                if score > highest_score:
                    highest_score = score
                    best_destinations = [destination]
                elif score == highest_score:
                    best_destinations.append(destination)
        return best_destinations

    @torch.no_grad()
    def cal_score(self, current_state,is_computed):
        current_state = torch.tensor(current_state, dtype=torch.float).unsqueeze(0).to(self.device)
        main_output = self.q_net(current_state)
        if not is_computed:
            score = torch.max(main_output).item()
        else:
            score = torch.max(main_output[0][0:4]).item()
        return score

    def forward_packet(self):
        while self.active:
            packet = yield self.forward_queue.get()
            record_queue_wait(packet, 'forward', self.env.now)
            source, destination,hops,creation_time,size = packet.source, packet.destination,packet.hops,packet.creation_time,packet.size
            is_computed,type,computing_demand, size_after_computing,last_time, last_state, last_action = packet.information
            previous_action_mask = packet.forward_action_mask
            yield self.env.timeout(self.processing_time)
            add_packet_delay(packet, 'computation', self.processing_time)
            if self.mode in ["Tradition","Ground"]:
                prepare_routing = getattr(self, 'prepare_gosr_routing', None)
                if prepare_routing is not None:
                    prepare_routing(packet)
                current_state = [source, destination, size, computing_demand, size_after_computing,is_computed]
                planning_step = float(self.env.now)
                time_epoch = (
                    int(planning_step // self.tradition_replan_interval)
                    if self.tradition_replan_interval else 0
                )
                planning_epoch = max(
                    time_epoch,
                    int(getattr(self.propagator, 'pics_state_version', 0)),
                )
                last_plan_epoch = getattr(packet, 'tradition_last_plan_epoch', None)
                batch_scheduler = getattr(
                    self.propagator, 'batch_scheduler', None
                )
                topology_version = int(
                    getattr(self.propagator, 'topology_version', 0)
                )
                route_topology_version = getattr(
                    packet, 'tradition_topology_version', None
                )
                route_stale = (
                    batch_scheduler is not None
                    and route_topology_version is not None
                    and route_topology_version != topology_version
                )
                next_hop_unavailable = bool(packet.routing) and (
                    packet.routing[0] not in self.neighbors
                )
                periodic_replan_due = (
                    self.mode == "Tradition"
                    and self.tradition_replan_interval is not None
                    and last_plan_epoch is not None
                    and planning_epoch > last_plan_epoch
                    and packet.computing_node != self.name
                    and batch_scheduler is None
                )
                routing_graph_ready = (
                    self.propagator.graph is not None
                    if self.mode == "Tradition" else self.neighbor_graph is not None
                )
                if (
                    batch_scheduler is not None
                    and self.mode == "Tradition"
                    and (route_stale or next_hop_unavailable)
                    and destination != self.name
                    and packet.computing_node != self.name
                    and routing_graph_ready
                ):
                    packet.routing = None
                    packet.tradition_last_plan_epoch = None
                    batch_scheduler.defer(self, packet)
                    yield self.env.timeout(batch_scheduler.retry_delay(self))
                    continue
                if (
                    batch_scheduler is not None
                    and self.mode == "Tradition"
                    and not packet.routing
                    and destination != self.name
                    and packet.computing_node != self.name
                    and routing_graph_ready
                ):
                    batch_scheduler.defer(self, packet)
                    yield self.env.timeout(batch_scheduler.retry_delay(self))
                    continue
                if ((not packet.routing or periodic_replan_due)
                        and destination != self.name and routing_graph_ready
                        and packet.computing_node != self.name):
                    reservation_book = getattr(
                        self.propagator, 'reservation_book', None
                    )
                    old_route = list(packet.routing or [])
                    old_computing_node = packet.computing_node
                    old_reservation = (
                        reservation_book.release(packet)
                        if reservation_book is not None and periodic_replan_due
                        else None
                    )
                    if periodic_replan_due and not is_computed and packet.computing_node:
                        old_computing_node = packet.computing_node
                        if old_computing_node in self.propagator.graph.nodes:
                            old_reserved = self.propagator.graph.nodes[old_computing_node].get(
                                'computing_remain', 0
                            )
                            self.propagator.graph.nodes[old_computing_node]['computing_remain'] = max(
                                old_reserved - computing_demand,
                                0,
                            )
                    computing = False if self.mode == "Ground" else True
                    new_route, new_computing_node = self.tradition_routing(
                        current_state,
                        computing,
                    )
                    route_valid = new_route and new_route != [None]
                    committed = route_valid or reservation_book is None
                    if route_valid and reservation_book is not None:
                        committed = reservation_book.commit(
                            packet,
                            new_route,
                            new_computing_node,
                            size,
                            size_after_computing,
                            is_computed,
                        )
                    if not committed and old_reservation is not None and old_route:
                        if route_valid and not is_computed and new_computing_node in self.propagator.graph.nodes:
                            self.propagator.graph.nodes[new_computing_node]['computing_remain'] = max(
                                self.propagator.graph.nodes[new_computing_node].get(
                                    'computing_remain', 0
                                ) - computing_demand,
                                0,
                            )
                        if not is_computed and old_computing_node in self.propagator.graph.nodes:
                            self.propagator.graph.nodes[old_computing_node]['computing_remain'] = (
                                self.propagator.graph.nodes[old_computing_node].get(
                                    'computing_remain', 0
                                ) + computing_demand
                            )
                        reservation_book.restore(packet, old_reservation)
                        new_route = old_route
                        new_computing_node = old_computing_node
                        committed = True
                    if not committed:
                        if route_valid and not is_computed and new_computing_node in self.propagator.graph.nodes:
                            self.propagator.graph.nodes[new_computing_node]['computing_remain'] = max(
                                self.propagator.graph.nodes[new_computing_node].get(
                                    'computing_remain', 0
                                ) - computing_demand,
                                0,
                            )
                        new_route = [None]
                        new_computing_node = None
                    packet.routing = new_route
                    packet.computing_node = new_computing_node
                    packet.tradition_last_plan_epoch = planning_epoch
            else:
                if self.select_mode == 3 and destination != self.name:
                    min_hops_destinations=self.find_min_hops_destinations(5)
                    hightest_score_destinations=self.find_highest_score(min_hops_destinations,[type,size,computing_demand,size_after_computing],hops, is_computed)
                    if hightest_score_destinations:
                        destination = np.random.choice(hightest_score_destinations)
                    else:
                        destination = 'False'
                    packet.destination = destination
                current_state = self.get_current_state(destination, hops,is_computed,[type,size/self.max_size,_normalized_computing_demand(computing_demand),size_after_computing/self.max_size])
            physical_valid_action_mask = self.valid_action_mask(
                destination, size, is_computed, hops, getattr(packet, 'wait_count', 0),
                previous_hop=packet.previous_hop,
                postcompute_previous_hop=getattr(packet, 'postcompute_previous_hop', None),
            )
            if (getattr(self, 'gosr_use_action_slots', False)
                    and destination != self.name
                    and packet.computing_node != self.name
                    and (not packet.routing or packet.routing == [None])):
                self._drop_no_route(
                    packet, size, type, computing_demand, creation_time,
                    reason='hop_limit' if hops >= 2 * self.max_hop else 'no_route',
                )
                continue
            if not self.active:
                reservation_book = getattr(self.propagator, 'reservation_book', None)
                if reservation_book is not None:
                    reservation_book.release(packet)
                if packet.computing_node and PRE and packet.computing_node in self.propagator.graph.nodes:
                    self.propagator.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                if type==0:
                    self.statics_data['Lost_relay_0'] += 1
                else:
                    self.statics_data['Lost_relay_1'] += 1
                record_packet_loss(self.statics_data, 'relay', 'node_failure')
                self.propagator.record_terminal_loss(
                    self.name,
                    packet,
                    self.reward_function.loss_reward(self.env.now - creation_time),
                )
                self.logger.log(f"Time {self.env.now:.3f}: {self.name} is missed, dropped 1 packet")
                break
            requires_computing_decision = (
                self.require_computed_before_downlink
                and destination == self.name
                and not is_computed
            )
            previous_transition_recorded = False
            # When target computation is required, the destination satellite
            # is the fallback compute node before the normal downlink queue.
            prepared_policy_state = None
            prepared_action_mapping = None
            if destination != self.name or requires_computing_decision:
                # Stop another forwarding decision once the packet has used
                # twice the initial network diameter in actual hops.
                within_hop_limit = hops < 2 * self.max_hop
                if (within_hop_limit and self.mode not in ['Tradition', 'Ground']
                        and destination != self.name
                        and ((is_computed and (destination not in self.routing_tables
                                               or not physical_valid_action_mask[:4].any()))
                             or (not is_computed and not physical_valid_action_mask.any()))):
                    mark_queue_entry(packet, 'waiting_forward', self.env.now)
                    if (self.mode in ('Pure_DQN', 'New_DQN', 'Pure_PPO') or getattr(self, 'aligned_graphsage_policy', False)):
                        if getattr(packet, 'passive_wait_started', None) is None:
                            packet.passive_wait_started = self.env.now
                            packet.wait_count = 0
                        if packet.wait_count >= PASSIVE_WAIT_MAX_RETRIES:
                            record_queue_wait(packet, 'waiting_forward', self.env.now)
                            self._drop_no_route(packet, size, type, computing_demand,
                                                creation_time, reason='wait_limit')
                            continue
                        packet.wait_retry_cycle = int(np.floor(self.env.now / self.state_update_period + 1e-9)) + 1
                    self.waiting_forward.append(packet)
                    if (self.mode in ('Pure_DQN', 'New_DQN', 'Pure_PPO') or getattr(self, 'aligned_graphsage_policy', False)):
                        self._schedule_waiting_forward_retry()
                    continue
                if within_hop_limit or requires_computing_decision:
                    if self.ppo_shuffle_func and ('GNN' not in self.mode or getattr(self, 'aligned_graphsage_policy', False)):
                        policy_input_state = (
                            current_state[1] if 'GNN' in self.mode and not getattr(self, 'aligned_graphsage_policy', False) else current_state
                        )
                        prepared_policy_state, prepared_action_mapping = (
                            self.ppo_shuffle_func(policy_input_state)
                        )
                    decision_action_mask = self.replay_action_mask(
                        physical_valid_action_mask, prepared_action_mapping
                    )
                    if (
                        last_state is not None
                        and self.mode not in ["Tradition", "Ground"]
                        and physical_valid_action_mask.any()
                    ):
                        # Complete the incoming hop before selecting this node's next action.
                        # The memory term belongs to the previous sender decision.
                        remaining_hops = (
                            self.routing_tables[destination][1]
                            if destination in self.routing_tables else 0
                        )
                        previous_memory_remain = packet.forward_memory_remain
                        previous_reward = self.reward_function.normal_reward(
                            self.env.now - last_time,
                            (
                                previous_memory_remain
                                if previous_memory_remain is not None
                                else self._effective_memory_remain()
                            ),
                            hop_count=max(packet.hops - remaining_hops, 0),
                        )
                        packet.forward_memory_remain = None
                        mark = current_state[-1] if 'GNN' not in self.mode else current_state[-1][-1]
                        transition_state = (
                            prepared_policy_state
                            if prepared_policy_state is not None else current_state
                        )
                        self.propagator.record_experience(
                            [last_state, mark, last_action, previous_reward, transition_state, 0],
                            decision_action_mask,
                            packet=packet,
                            current_action_mask=previous_action_mask,
                        )
                        previous_transition_recorded = True
                        packet.passive_wait_started = None
                    if self.mode in ["Tradition","Ground"]:
                        if packet.computing_node==self.name or requires_computing_decision:
                            if (requires_computing_decision
                                    and packet.computing_node != self.name and PRE):
                                self.propagator.graph.nodes[self.name]['computing_remain'] += computing_demand
                            action=4
                            packet.computing_node=None
                        else:
                            if packet.routing:
                                next_hop = packet.routing[0]
                                if (
                                    self.mode == "Tradition"
                                    and (
                                        next_hop not in self.neighbors
                                        or not self.propagator.graph.has_edge(
                                            self.name, next_hop
                                        )
                                    )
                                ):
                                    packet.routing = None
                                    packet.tradition_last_plan_epoch = None
                                    if getattr(self, 'defer_unavailable_route', False):
                                        mark_queue_entry(packet, 'waiting_forward', self.env.now)
                                        self.waiting_forward.append(packet)
                                        yield self.env.timeout(
                                            max(float(self.state_update_period), 0.01)
                                        )
                                        continue
                                    if batch_scheduler is not None:
                                        batch_scheduler.defer(self, packet)
                                        yield self.env.timeout(batch_scheduler.retry_delay(self))
                                        continue
                                    action = 5
                                else:
                                    packet.routing.pop(0)
                                    action_neighbors = (
                                        self.action_neighbors
                                        if getattr(self, 'gosr_use_action_slots', False)
                                        and hasattr(self, 'action_neighbors') else self.neighbors
                                    )
                                    action=action_neighbors.index(next_hop)
                            else:
                                if getattr(self, 'defer_unavailable_route', False):
                                    packet.routing = None
                                    packet.tradition_last_plan_epoch = None
                                    mark_queue_entry(packet, 'waiting_forward', self.env.now)
                                    self.waiting_forward.append(packet)
                                    yield self.env.timeout(
                                        max(float(self.state_update_period), 0.01)
                                    )
                                    continue
                                action=5
                    else:
                        action = self.get_next_hop(
                            current_state,
                            destination,
                            packet_size=size,
                            forced_action=4 if requires_computing_decision else None,
                            prepared_policy_state=prepared_policy_state,
                            prepared_action_mapping=prepared_action_mapping,
                            hops=hops,
                            wait_count=getattr(packet, 'wait_count', 0),
                            previous_hop=packet.previous_hop,
                            postcompute_previous_hop=getattr(packet, 'postcompute_previous_hop', None),
                        )
                        if action is None:
                            self._drop_no_route(
                                packet, size, type, computing_demand, creation_time,
                                reason='wait_limit' if getattr(packet, 'wait_count', 0) >= PASSIVE_WAIT_MAX_RETRIES else 'no_route',
                            )
                            continue
                    packet.passive_wait_started = None
                    packet.wait_count = 0
                    stored_state = current_state
                    stored_action = action
                    if 'PPO' in self.mode:
                        next_index = action[0]
                        stored_state = action[2]
                        policy_metadata = list(action[1][:3])
                        forced_action_flag = bool(
                            action[1][3] if len(action[1]) > 3 else False
                        )
                        if not hasattr(packet, 'ppo_trajectory_id'):
                            packet.ppo_trajectory_id = self.propagator.ppo_trajectory_counter
                            self.propagator.ppo_trajectory_counter += 1
                            packet.ppo_step = 0
                        policy_metadata.extend([
                            packet.ppo_trajectory_id,
                            packet.ppo_step,
                            forced_action_flag,
                            action[1][4],
                        ])
                        packet.ppo_step += 1
                        stored_action = policy_metadata
                    elif self.ppo_shuffle_func and ('GNN' not in self.mode or getattr(self, 'aligned_graphsage_policy', False)):
                        next_index, stored_action, stored_state = action
                    else:
                        next_index = action
                    packet.forward_action_mask = (
                        decision_action_mask.copy()
                        if decision_action_mask is not None else None
                    )
                    if (destination in self.routing_tables or destination == self.name
                            or ((self.mode in ('Pure_DQN', 'New_DQN', 'Pure_PPO') or getattr(self, 'aligned_graphsage_policy', False)) and next_index == 4)
                            or (self.mode == "Tradition" and batch_scheduler is not None
                                and 0 <= next_index <= 4)):
                        action_neighbors = (
                            self.action_neighbors if hasattr(self, 'action_neighbors') else self.neighbors
                        )
                        next_hop = (
                            action_neighbors[next_index]
                            if 0 <= next_index < min(4, len(action_neighbors)) else None
                        )
                        if next_hop in self.neighbors:
                            done = 0
                            remaining_hops = (
                                self.routing_tables[destination][1]
                                if destination in self.routing_tables else 0
                            )
                            projected_memory_remain = (
                                self._update_neighbor_memory_after_forward(
                                    next_hop, size,
                                )
                            )
                            reward = self.reward_function.normal_reward(
                                self.env.now-last_time,
                                projected_memory_remain,
                                hop_count=max(
                                    packet.hops - remaining_hops,
                                    0,
                                ),
                            )
                            self.push_transmission(next_hop, packet)
                            packet.forward_memory_remain = projected_memory_remain
                            self._refresh_transmission_features()
                            packet.extra_information([is_computed,type, computing_demand, size_after_computing, self.env.now, stored_state,stored_action])
                        elif next_index==4:
                            done = 0
                            packet.forward_memory_remain = None
                            reward = self.reward_function.normal_reward(
                                self.env.now-last_time,
                                self._effective_memory_remain(),
                            )
                            packet.extra_information([is_computed,type,computing_demand, size_after_computing,self.env.now, stored_state, stored_action])
                            self.push_computing(packet,computing_demand)
                        else:
                            done = 1
                            reward = self.reward_function.loss_reward(self.env.now-creation_time)
                            reservation_book = getattr(self.propagator, 'reservation_book', None)
                            if reservation_book is not None:
                                reservation_book.release(packet)
                            self.current_memory_occupy-=size
                            if packet.computing_node and PRE and packet.computing_node in self.propagator.graph.nodes:
                                self.propagator.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                            if type == 0:
                                self.statics_data['Lost_relay_0'] += 1
                            else:
                                self.statics_data['Lost_relay_1'] += 1
                            record_packet_loss(self.statics_data, 'relay', 'invalid_action')
                            self.logger.log(f"Time {self.env.now:.3f}: wrong forward decision, dropped 1 packet")
                            if self.mode not in ["Tradition", "Ground"]:
                                mark = current_state[-1] if 'GNN' not in self.mode else current_state[-1][-1]
                                if last_state is not None and not previous_transition_recorded:
                                    previous_reward = self.reward_function.normal_reward(
                                        self.env.now - last_time,
                                        None,
                                    )
                                    self.propagator.record_experience(
                                        [last_state, mark, last_action, previous_reward, stored_state, 0],
                                        decision_action_mask,
                                        packet=packet,
                                        current_action_mask=previous_action_mask,
                                    )
                                self.propagator.record_experience(
                                    [stored_state, mark, stored_action, reward, stored_state, 1],
                                    packet=packet,
                                    current_action_mask=packet.forward_action_mask,
                                )
                            self.propagator.finalize_packet_reward(packet, reward)
                            reward = None  # Both transitions were recorded in decision order.
                    else:
                        self._drop_no_route(
                            packet, size, type, computing_demand, creation_time
                        )
                        reward = None
                else:
                    reservation_book = getattr(self.propagator, 'reservation_book', None)
                    if reservation_book is not None:
                        reservation_book.release(packet)
                    self.current_memory_occupy -= size
                    done = 1
                    reward = self.reward_function.loss_reward(self.env.now-creation_time)
                    reward -= self.reward_function.delay_factor * (self.env.now - last_time)
                    if packet.computing_node and PRE and packet.computing_node in self.propagator.graph.nodes:
                        self.propagator.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                    if type == 0:
                        self.statics_data['Lost_relay_0'] += 1
                    else:
                        self.statics_data['Lost_relay_1'] += 1
                    record_packet_loss(self.statics_data, 'relay', 'hop_limit')
                    self.logger.log(f"Time {self.env.now:.3f}: cumulative hop limit {2 * self.max_hop} reached, dropped 1 packet")
            else:
                reward = None
                self.push_offload(packet)
            if last_state is not None and reward is not None and not previous_transition_recorded:
                if 'GNN' not in self.mode:
                    mark = current_state[-1]
                else:
                    mark = current_state[-1][-1]
                transition_state = (
                    prepared_policy_state
                    if prepared_policy_state is not None
                    else self.ppo_transition_state(current_state)
                )
                self.propagator.record_experience(
                    [last_state, mark, last_action, reward, transition_state, done],
                    self.replay_action_mask(
                        physical_valid_action_mask, prepared_action_mapping
                    ),
                    packet=packet,
                    current_action_mask=previous_action_mask,
                )
            if reward is not None and done:
                self.propagator.finalize_packet_reward(packet, reward)

    def _record_relay_loss(self, packet, reason):
        is_computed, packet_type, computing_demand = packet.information[:3]
        reward = self.propagator.reward_function.loss_reward(
            self.env.now - packet.creation_time
        )
        self.propagator.record_terminal_loss(self.name, packet, reward)
        if packet.computing_node and PRE and packet.computing_node in self.propagator.graph.nodes:
            self.propagator.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
        if packet_type == 0:
            self.statics_data['Lost_relay_0'] += 1
        else:
            self.statics_data['Lost_relay_1'] += 1
        record_packet_loss(
            self.statics_data,
            'relay',
            'link_break' if 'link' in reason.lower() else 'node_failure',
        )
        self.logger.log(
            f"Time {self.env.now:.3f}: {reason}, "
            f"dropping packet {(packet.source, packet.destination, packet.creation_time)}"
        )

    def _drop_no_route(self, packet, size, packet_type, computing_demand,
                       creation_time, reason='no_route'):
        reservation_book = getattr(self.propagator, 'reservation_book', None)
        if reservation_book is not None:
            reservation_book.release(packet)
        self.current_memory_occupy -= size
        if packet.computing_node and PRE and packet.computing_node in self.propagator.graph.nodes:
            self.propagator.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
        if packet_type == 0:
            self.statics_data['Lost_relay_0'] += 1
        else:
            self.statics_data['Lost_relay_1'] += 1
        record_packet_loss(self.statics_data, 'relay', reason)
        self.propagator.record_terminal_loss(
            self.name,
            packet,
            self.reward_function.loss_reward(self.env.now - creation_time),
        )
        self.logger.log(f"Time {self.env.now:.3f}: {packet.destination} {reason}, dropped 1 packet")

    def _return_transmission_for_redecision(self, packet, cancelled_time=0.0):
        """An already admitted packet keeps its memory while being rerouted."""
        reservation_book = getattr(self.propagator, 'reservation_book', None)
        if reservation_book is not None:
            reservation_book.release(packet)
        if packet.computing_node:
            if PRE and packet.computing_node in self.propagator.graph.nodes:
                node = self.propagator.graph.nodes[packet.computing_node]
                node['computing_remain'] = max(
                    node.get('computing_remain', 0) - packet.information[2], 0,
                )
            packet.computing_node = None
        packet.routing = None
        packet.tradition_topology_version = None
        packet.tradition_last_plan_epoch = None
        # Keep earlier queue waits, but do not charge the cancelled send interval.
        packet.cancelled_transmission_time += cancelled_time
        packet.information[4] += cancelled_time
        mark_queue_entry(packet, 'forward', self.env.now)
        self.forward_queue.put(packet)

    def _release_transmission_memory(self, packet):
        self.current_memory_occupy -= packet.size
        if self.mode in ['Tradition', 'Ground']:
            self.propagator.graph.nodes[self.name]['current_memory_occupy'] -= packet.size

    def transmit_packet(self,neighbor):
        while self.active and neighbor in self.neighbors:
            packet = yield self.env.process(self.pop_transmission(neighbor, retain_memory=True))
            # Preserve the forwarding decision time so the next transition
            # reward includes the wait in this transmission queue.
            transmit_time_consume=packet.size / self.transmission_rate
            transmission_start = self.env.now
            failure_event = self.propagator.watch_link_failure(self.name, neighbor)
            detected_event = self.neighbor_failure_events.setdefault(neighbor, self.env.event())
            transmission_done = self.env.timeout(transmit_time_consume)
            transmission = (transmission_start + transmit_time_consume, transmission_done)
            self.active_transmissions[neighbor] = transmission
            if failure_event is None:
                transmission_events = yield (transmission_done | detected_event)
            else:
                transmission_events = yield (transmission_done | failure_event | detected_event)
                # Physical failure is not yet a locally detected failure.
                if failure_event in transmission_events and not (
                    transmission_done in transmission_events or detected_event in transmission_events
                ):
                    transmission_events = yield (transmission_done | detected_event)
            # A cancelled sender must not clear a replacement transmission.
            if self.active_transmissions.get(neighbor) is transmission:
                self.active_transmissions.pop(neighbor)
            link_failed = (
                failure_event is None or failure_event.triggered
            ) or not self.propagator.graph.has_edge(self.name, neighbor)
            self.propagator.clear_link_failure_watch(self.name, neighbor, failure_event)
            on_resolved = getattr(self, 'on_lcsr_packet_resolved', None)
            if self.active and detected_event in transmission_events and self.env.now < transmission[0]:
                if on_resolved is not None:
                    on_resolved(neighbor, packet.size)
                self._return_transmission_for_redecision(packet, self.env.now - transmission_start)
                continue
            if self.active:
                self._release_transmission_memory(packet)
            if link_failed or neighbor not in self.neighbors or not self.active:
                add_packet_delay(packet, 'transmission', self.env.now - transmission_start)
                if getattr(self, 'defer_link_failure', False):
                    on_resolved = getattr(self, 'on_lcsr_packet_resolved', None)
                    if on_resolved is not None:
                        on_resolved(neighbor, packet.size)
                    packet.routing = None
                    packet.tradition_last_plan_epoch = None
                    if self.active and self.push_forward(packet):
                        continue
                else:
                    on_resolved = getattr(self, 'on_lcsr_packet_resolved', None)
                    if on_resolved is not None:
                        on_resolved(neighbor, packet.size)
                self._record_relay_loss(packet, 'transmission link failed')
            else:
                # Every successfully completed relay hop contributes one
                # serialization/transmission interval to the packet's
                # end-to-end delay.  The downlink path records its own
                # interval in offload_to_ground(); relay hops must do the
                # same here before propagation to the next node starts.
                add_packet_delay(packet, 'transmission', transmit_time_consume)
                packet.hops += 1
                self.transmission_time += transmit_time_consume
                self.logger.log(f"Time {self.env.now:.3f}: {self.name}: Packet {(packet.source,packet.destination,packet.creation_time)} departed. Memory remain: {(self.memory-self.current_memory_occupy)}",detail=True)
                self.env.process(self.propagator.propagate(self.name,neighbor, packet))

    def offload_to_ground(self):
        while self.active:
            packet = yield self.env.process(self.pop_offload(retain_memory=True))
            is_computed,type, computing_demand, size_after_computing, last_time, current_state, next_index = packet.information
            # Keep the last decision time until terminal reward settlement.
            downlink_transmission_delay = packet.size / self.downlink_rate
            yield self.env.timeout(downlink_transmission_delay)
            add_packet_delay(packet, 'transmission', downlink_transmission_delay)
            if self.active:
                self._release_transmission_memory(packet)
            if not self.active:
                reservation_book = getattr(self.propagator, 'reservation_book', None)
                if reservation_book is not None:
                    reservation_book.release(packet)
                if packet.computing_node and PRE and packet.computing_node in self.propagator.graph.nodes:
                    self.propagator.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                if type==0:
                    self.statics_data['Lost_relay_0'] += 1
                else:
                    self.statics_data['Lost_relay_1'] += 1
                record_packet_loss(self.statics_data, 'offload', 'node_failure')
                self.propagator.record_terminal_loss(
                    self.name,
                    packet,
                    self.reward_function.loss_reward(self.env.now - packet.creation_time),
                )
                self.logger.log(f"Time {self.env.now:.3f}: offload stopped, dropped 1 packet")
                break
            self.equivalent_transmission_data_size+=packet.original_size
            self.logger.log(f"Time {self.env.now:.3f}: {self.name}: Packet {(packet.source,packet.destination,packet.creation_time)} is offloading. Memory remain: {(self.memory-self.current_memory_occupy)}",detail=True)
            self.env.process(self.propagator.downstream(self.name, packet))

    def computing_packet(self):
        while self.active:
            packet = yield self.computing_queue.get()
            record_queue_wait(packet, 'computing', self.env.now)
            before_computation = getattr(
                self.propagator, 'before_computation', None
            )
            if (
                before_computation is not None
                and not before_computation(self, packet)
            ):
                continue
            reservation_book = getattr(self.propagator, 'reservation_book', None)
            if reservation_book is not None:
                reservation_book.start_computing(packet, self.name)
            self.is_computing=True
            self.last_computing_time=self.env.now
            is_computed,type,computing_demand, size_after_computing,last_time, last_state, last_action=packet.information
            computing_time_consume=computing_demand / self.computing_ability
            yield self.env.timeout(computing_time_consume)
            add_packet_delay(packet, 'computation', computing_time_consume)
            packet.computing_waiting_time+=(self.env.now-last_time)
            if self.mode in ["Tradition","Ground"]:
                self.propagator.graph.nodes[self.name]['current_memory_occupy'] -= packet.size-size_after_computing
                self.propagator.graph.nodes[self.name]['computing_remain'] -= computing_demand
            self.current_computing_queue_size -= packet.size
            self.computing_remain -= computing_demand
            self.current_memory_occupy -= packet.size - size_after_computing
            packet.size=size_after_computing
            self.is_computing = False
            self.last_computing_time = 0
            if reservation_book is not None:
                reservation_book.finish_computing(packet, self.name)
            if not self.active:
                if reservation_book is not None:
                    reservation_book.release(packet)
                self.current_memory_occupy -= packet.size
                if self.mode in ["Tradition","Ground"]:
                    self.propagator.graph.nodes[self.name]['current_memory_occupy'] -= packet.size
                if packet.computing_node and PRE and packet.computing_node in self.propagator.graph.nodes:
                    self.propagator.graph.nodes[packet.computing_node]['computing_remain'] -= computing_demand
                if type==0:
                    self.statics_data['Lost_relay_0'] += 1
                else:
                    self.statics_data['Lost_relay_1'] += 1
                record_packet_loss(self.statics_data, 'compute', 'node_failure')
                self.propagator.record_terminal_loss(
                    self.name,
                    packet,
                    self.reward_function.loss_reward(self.env.now - packet.creation_time),
                )
                self.logger.log(f"Time {self.env.now:.3f}: computing stopped, dropped 1 packet")
                break
            self.computing_time += computing_time_consume
            packet.postcompute_previous_hop = packet.previous_hop
            packet.previous_hop = None
            packet.extra_information([True,type,0,size_after_computing,last_time, last_state, last_action])
            self.logger.log(f"Time {self.env.now:.3f}: {self.name}: Packet {(packet.source,packet.destination,packet.creation_time)} finished computing. Memory remain: {(self.memory-self.current_memory_occupy)}",detail=True)
            mark_queue_entry(packet, 'forward', self.env.now)
            self.forward_queue.put(packet)

    def add_neighbor(self,neighbor):
        if neighbor not in self.neighbors:
            self.pending_state_neighbors.discard(neighbor)
            link_since = self.pending_link_since.pop(neighbor, self.env.now)
            if hasattr(self, 'action_neighbors') and neighbor not in self.action_neighbors:
                vacancy = next(
                    (index for index, value in enumerate(self.action_neighbors) if value is None),
                    None,
                )
                if vacancy is None:
                    vacancy = next(
                        (index for index, value in enumerate(self.action_neighbors)
                         if value not in self.neighbors),
                        None,
                    )
                if vacancy is None:
                    raise ValueError('RL policy has no free forwarding action slot')
                self.action_neighbors[vacancy] = neighbor
                self.topology_neighbors = tuple(
                    value for value in self.action_neighbors if value is not None
                )
            self.neighbors.append(neighbor)
            self.neighbors=sorted(self.neighbors)
            if neighbor not in self.transmission_queue:
                self.transmission_queue[neighbor] = simpy.Store(self.env)
                self.env.process(self.transmit_packet(neighbor))
            self.transmission_size[neighbor] = 0
            self.transmission_length[neighbor] = 0
            if "New" in self.mode or getattr(self, "gosr_use_action_slots", False):
                self.neighbor_states[neighbor] = [0,1,0,0,0,0,0,4,0,0,0,0,0,12,0,0,0,0]
                self.neighbor_elimination_features[neighbor] = [0,1,0,0,0,0]
            else:
                self.neighbor_states[neighbor] = [0,1,0,0,0] + ([0] if self.mode in ('Pure_DQN', 'Pure_PPO') or ('GNN' in self.mode and self.gnn_node_feature_dim == ISACR_NODE_FEATURE_DIM) else [])
            self.last_heartbeat[neighbor] = self.env.now
            self.last_heartbeat_sent_times.pop(neighbor, None)
            self.neighbor_state_versions[neighbor] = 0
            self.neighbor_state_events[neighbor] = self.env.event()
            self.adjacency_table [self.name]=(self.neighbors, self.env.now)
            record = self.pending_adjacency_records.pop(neighbor, None)
            if record is not None:
                self.adjacency_table[neighbor] = record
            else:
                record = self.adjacency_table.get(neighbor)
            if (record is None or record[1] < link_since
                    or self.name not in record[0]):
                self._neighbor_route_since[neighbor] = link_since
            self.build_routing_table()
            if 'GNN' in self.mode:
                self.three_hop_info_db.clear()
                self.encode_data = None
            self.adjacency_table_exchanger()
            self.env.process(self.monitor_single_neighbor(neighbor))

    def mark_neighbor_fault(self, neighbor):
        """Keep an unavailable neighbor represented by a fault-filled state."""
        self.neighbor_states[neighbor] = self._fault_neighbor_state()

    def del_neighbor(self,neighbor):
        if self.active:
            if neighbor in self.neighbors:
                self.neighbors.remove(neighbor)
                detected_event = self.neighbor_failure_events.pop(neighbor, None)
                if detected_event is not None and not detected_event.triggered:
                    detected_event.succeed()
                while self.transmission_queue[neighbor].items:
                    packet = yield self.env.process(self.pop_transmission(neighbor, retain_memory=True))
                    on_resolved = getattr(self, 'on_lcsr_packet_resolved', None)
                    if on_resolved is not None:
                        on_resolved(neighbor, packet.size)
                    self._return_transmission_for_redecision(packet)
                # Recovery must create a fresh queue and restart its sender.
                del self.transmission_queue[neighbor]
                self.active_transmissions.pop(neighbor, None)
                del self.transmission_size[neighbor]
                self.mark_neighbor_fault(neighbor)
                if 'New' in self.mode and neighbor in self.neighbor_elimination_features:
                    del self.neighbor_elimination_features[neighbor]
                # Another removal may rebuild routes while queued packets return.
                self.neighbor_hops.pop(neighbor, None)
                self._neighbor_route_since.pop(neighbor, None)
                del self.transmission_length[neighbor]
                self.adjacency_table[self.name]=(self.neighbors, self.env.now)
                self.build_routing_table()
                if 'GNN' in self.mode:
                    # Cached remote features may have arrived through the failed
                    # link.  Drop them immediately; surviving paths repopulate
                    # the cache through normal feature messages.
                    self.three_hop_info_db.clear()
                    self.encode_data = None
                self.adjacency_table_exchanger()
                if self.propagator.graph.has_edge(self.name, neighbor):
                    self.pending_state_neighbors.add(neighbor)
                    self.pending_link_since[neighbor] = self.env.now
                    self.pending_adjacency_records.pop(neighbor, None)
                    if not self._state_probe_event.triggered:
                        self._state_probe_event.succeed()
                return True
            else:
                return False
        else:
            return False

    def find_min_hops_destinations(self, n):
        hop_counts = []
        for destination in sorted(self.direct_satellites):
            if destination in self.routing_tables:
                hops = self.routing_tables[destination][1]
            else:
                hops = np.inf
            hop_counts.append((destination, hops))
        if len(hop_counts)>n:
            hop_counts.sort(key=lambda x: x[1])
            return [a for a,b in hop_counts[:n]]
        else:
            return [a for a,b in hop_counts]

    def _wait_for_state_exchange(self):
        next_cycle = math.floor(self.env.now / self.state_update_period + 1e-9) + 1
        timeout = self.env.timeout(next_cycle * self.state_update_period - self.env.now)
        wake = self._state_probe_event
        yield timeout
        if wake.triggered:
            self._state_probe_event = self.env.event()

    def _aggregation_resource_states(self):
        """Use delivered snapshots strictly before this aggregation boundary."""
        history = getattr(self, 'aggregation_resource_history', {})
        result = {}
        for neighbor in self.neighbors:
            if not self.neighbor_has_resource_state(neighbor):
                continue
            if neighbor not in history:
                result[neighbor] = list(self.neighbor_states[neighbor])
                continue
            for received_at, sent_at, value in reversed(history[neighbor]):
                deadline = (math.floor(sent_at / self.state_update_period + 1e-9)
                            * self.state_update_period + self.heartbeat_timeout
                            if sent_at is not None else received_at + self.heartbeat_timeout)
                if (received_at < self.env.now - 1e-9 and self.env.now <= deadline
                        and self.env.now <= received_at + self.heartbeat_timeout):
                    result[neighbor] = value
                    break
        return result

    def state_exchanger(self):
        while self.active:
            yield from self._wait_for_state_exchange()
            if not self.active:
                break
            targets = sorted(set(self.neighbors) | self.pending_state_neighbors)
            received_states = {
                neighbor: self.neighbor_states[neighbor] for neighbor in self.neighbors
                if self.neighbor_has_resource_state(neighbor)
            } if ('New' in self.mode or getattr(self, 'gosr_use_action_slots', False)) else {}
            if self.mode == 'New_DQN' and self.historical_self_feature_elimination:
                received_states = self._aggregation_resource_states()
            if 'GNN' in self.mode and self.gnn_node_feature_dim in (NODE_FEATURE_DIM, ISACR_NODE_FEATURE_DIM):
                if getattr(self, 'aligned_graphsage_policy', False):
                    continue
                for neighbor in sorted(self.pending_state_neighbors):
                    self.env.process(self.propagator.send_state(
                        self.name, neighbor, self.current_gnn_raw_state()
                    ))
                continue
            if self.mode == 'New_DQN' and self.historical_self_feature_elimination and not received_states:
                # Keep the same 1/4/12 storage scales as the normal shift path.
                # No valid neighbor contributes to the fault-fill mean.
                local = self.current_gnn_raw_state()[:5]
                local.append(self._normalized_compute_completion_time())
                fault = [1, 0, 0, 1, 1, self._local_fault_compute_score()]
                self.current_state = local + [value * scale for scale in (4, 12) for value in fault]
                for neighbor in targets:
                    self.env.process(self.propagator.send_state(
                        self.name, neighbor, list(self.current_state)
                    ))
                continue
            if ('New' in self.mode or getattr(self, 'gosr_use_action_slots', False)) and self.historical_self_feature_elimination and received_states:
                # Recipient-specific three-hop aggregation with fault slots.
                # Aggregation is performed on normalized features.  Raw second-
                # and third-hop blocks are scaled by 4 and 12 only when stored,
                # because get_current_state() applies the inverse normalization.
                aggregation_degree = 4
                remaining_degree = aggregation_degree - 1
                n = len(received_states)
                neighbor_states_snapshot = {
                    neighbor: list(state) for neighbor, state in received_states.items()
                }
                normalized_neighbor_states = {
                    neighbor: state[0:ISACR_NODE_FEATURE_DIM]
                    + [value / aggregation_degree for value in state[ISACR_NODE_FEATURE_DIM:2 * ISACR_NODE_FEATURE_DIM]]
                    + [value / (aggregation_degree * remaining_degree) for value in state[2 * ISACR_NODE_FEATURE_DIM:3 * ISACR_NODE_FEATURE_DIM]]
                    for neighbor, state in neighbor_states_snapshot.items()
                }
                position_sums = [sum(items) for items in zip(*normalized_neighbor_states.values())]
                if self.fault_compute_double_mean:
                    ability_multiplier = 1 / self.fault_fill_multiplier
                    ca1, ca2, ca3 = (_clip_normalized(ability_multiplier * position_sums[2] / n), _clip_normalized(ability_multiplier * position_sums[8] / n), _clip_normalized(ability_multiplier * position_sums[14] / n))
                    cv1, cv2, cv3 = (_clip_normalized(self.fault_fill_multiplier * position_sums[3] / n), _clip_normalized(self.fault_fill_multiplier * position_sums[9] / n), _clip_normalized(self.fault_fill_multiplier * position_sums[15] / n))
                else:
                    ca1, ca2, ca3 = 0, 0, 0
                    cv1, cv2, cv3 = 1, 1, 1
                fill_multiplier = self.fault_fill_multiplier if self.fault_compute_double_mean else 2
                av1, av2, av3 = (_clip_normalized(fill_multiplier * position_sums[4] / n), _clip_normalized(fill_multiplier * position_sums[10] / n), _clip_normalized(fill_multiplier * position_sums[16] / n))
                g1, g2, g3 = (
                    _clip_normalized(position_sums[offset + 5] / n / self.fault_fill_multiplier)
                    if self.fault_compute_double_mean else 0
                    for offset in (0, 6, 12)
                )
                specified_values_list = [1,0,ca1,cv1,av1,g1,1,0,ca2,cv2,av2,g2,1,0,ca3,cv3,av3,g3]
                self_value=[
                    self.is_producing,
                    1 - self.current_memory_occupy / self.memory,
                    self._normalized_compute_ability(),
                    self._normalized_compute_time(
                        self.computing_remain / self.computing_ability
                        - self.is_computing * (self.env.now - self.last_computing_time)
                    ),
                    self._normalized_mean_transmission_wait(),
                ]
                self_value.append(self._normalized_compute_completion_time())
                missing_count = aggregation_degree - n
                second_hop_state = [
                    (position_sums[index] + missing_count * specified_values_list[index])
                    / aggregation_degree
                    for index in range(ISACR_NODE_FEATURE_DIM)
                ]
                third_hop_state = [
                    (position_sums[index + ISACR_NODE_FEATURE_DIM] + missing_count * specified_values_list[index + ISACR_NODE_FEATURE_DIM])
                    / aggregation_degree
                    for index in range(ISACR_NODE_FEATURE_DIM)
                ]
                raw_second_hop_state = [aggregation_degree * value for value in second_hop_state]
                raw_third_hop_state = [
                    aggregation_degree * remaining_degree * value
                    for value in third_hop_state
                ]
                self.current_state = self_value + raw_second_hop_state + raw_third_hop_state
                for neighbor in targets:
                    if self.disable_self_feature_elimination:
                        # Keep the ISACR aggregate and fault filling unchanged;
                        # omit only its recipient-specific self elimination.
                        self.env.process(self.propagator.send_state(
                            self.name, neighbor, list(self.current_state)
                        ))
                        continue
                    if neighbor in normalized_neighbor_states and missing_count:
                        # Fault-fill means must exclude the message recipient too.
                        other_states = [state for name, state in normalized_neighbor_states.items() if name != neighbor]
                        recipient_fill = list(specified_values_list[:2 * ISACR_NODE_FEATURE_DIM])
                        for layer in range(2):
                            offset = layer * ISACR_NODE_FEATURE_DIM
                            for index in (2, 3, 4):
                                if not other_states:
                                    recipient_fill[offset + index] = 0 if index == 2 else 1
                                elif self.fault_compute_double_mean or index == 4:
                                    multiplier = self.fault_fill_multiplier if self.fault_compute_double_mean else 2
                                    if index == 2:
                                        multiplier = 1 / self.fault_fill_multiplier
                                    fill_value = multiplier * sum(state[offset + index] for state in other_states) / len(other_states)
                                    recipient_fill[offset + index] = _clip_normalized(fill_value)
                            recipient_fill[offset + 5] = (
                                _clip_normalized(sum(state[offset + 5] for state in other_states)
                                                 / len(other_states) / self.fault_fill_multiplier)
                                if other_states and self.fault_compute_double_mean else 0
                            )
                        recipient_second_hop = [
                            (sum(state[index] for state in other_states) + missing_count * recipient_fill[index]) / remaining_degree
                            for index in range(ISACR_NODE_FEATURE_DIM)
                        ]
                        recipient_third_hop = [
                            (sum(state[index + ISACR_NODE_FEATURE_DIM] for state in other_states) + missing_count * recipient_fill[index + ISACR_NODE_FEATURE_DIM]) / remaining_degree
                            for index in range(ISACR_NODE_FEATURE_DIM)
                        ]
                    else:
                        recipient_feature = normalized_neighbor_states.get(neighbor, specified_values_list)[0:ISACR_NODE_FEATURE_DIM]
                        recipient_second_feature = normalized_neighbor_states.get(neighbor, specified_values_list)[ISACR_NODE_FEATURE_DIM:2 * ISACR_NODE_FEATURE_DIM]
                        recipient_second_hop = [
                            (aggregation_degree * value - recipient_value) / remaining_degree
                            for value, recipient_value in zip(second_hop_state, recipient_feature)
                        ]
                        recipient_third_hop = [
                            (aggregation_degree * value - recipient_value) / remaining_degree
                            for value, recipient_value in zip(
                                third_hop_state,
                                recipient_second_feature,
                            )
                        ]
                    recipient_state = (
                        self_value
                        + [aggregation_degree * value for value in recipient_second_hop]
                        + [
                            aggregation_degree * remaining_degree * value
                            for value in recipient_third_hop
                        ]
                    )
                    self.env.process(self.propagator.send_state(
                        self.name,
                        neighbor,
                        recipient_state,
                    ))
                continue
            for neighbor in targets:
                if ('New' in self.mode or getattr(self, 'gosr_use_action_slots', False)):
                    n = len(received_states)
                    if n:
                        position_sums = [sum(items) for items in zip(*received_states.values())]
                        if self.fault_compute_double_mean:
                            ability_multiplier = 1 / self.fault_fill_multiplier
                            ca1, ca2, ca3 = (_clip_normalized(ability_multiplier * position_sums[2] / n), 4 * _clip_normalized(ability_multiplier * position_sums[8] / (4 * n)), 12 * _clip_normalized(ability_multiplier * position_sums[14] / (12 * n)))
                            cv1, cv2, cv3 = (_clip_normalized(self.fault_fill_multiplier * position_sums[3] / n), _clip_normalized(self.fault_fill_multiplier * position_sums[9] / n), _clip_normalized(self.fault_fill_multiplier * position_sums[15] / n))
                        else:
                            ca1, ca2, ca3 = 0, 0, 0
                            cv1, cv2, cv3 = 1, 1, 1
                        fill_multiplier = self.fault_fill_multiplier if self.fault_compute_double_mean else 2
                        av1, av2, av3 = (_clip_normalized(fill_multiplier * position_sums[4] / n), _clip_normalized(fill_multiplier * position_sums[10] / n), _clip_normalized(fill_multiplier * position_sums[16] / n))
                        g1, g2, g3 = (
                            scale * _clip_normalized(position_sums[offset + 5]
                                                     / (scale * n) / self.fault_fill_multiplier)
                            if self.fault_compute_double_mean else 0
                            for offset, scale in ((0, 1), (6, 4), (12, 12))
                        )
                    else:
                        g1, g2, g3 = (scale * self._local_fault_compute_score() for scale in (1, 4, 12))
                        ca1, ca2, ca3 = 0, 0, 0
                        cv1, cv2, cv3 = 1, 1, 1
                        av1, av2, av3 = 1, 1, 1
                    specified_values_list = [1,0,ca1,cv1,av1,g1,1,0,ca2,cv2,av2,g2,1,0,ca3,cv3,av3,g3]
                    self_value=[
                        self.is_producing,
                        1 - self.current_memory_occupy / self.memory,
                        self._normalized_compute_ability(),
                        self._normalized_compute_time(
                            self.computing_remain / self.computing_ability
                            - self.is_computing * (self.env.now - self.last_computing_time)
                        ),
                        self._normalized_mean_transmission_wait(),
                    ]
                    self_value.append(self._normalized_compute_completion_time())
                    additional_values = [x * (4 - n) for x in specified_values_list]
                    if n == 0:
                        temp = additional_values
                    elif self.disable_self_feature_elimination:
                        temp = [
                            sum(tup) + add
                            for tup, add in zip(zip(*received_states.values()), additional_values)
                        ]
                    else:
                        self_values= [0] * ISACR_NODE_FEATURE_DIM + [x * n for x in self.current_state] + [0] * ISACR_NODE_FEATURE_DIM
                        temp = [sum(tup) + add - sub for tup, add,sub in zip(zip(*received_states.values()), additional_values,self_values)]
                    self.current_state=self_value+temp[0:ISACR_NODE_FEATURE_DIM * 2]
                    self.env.process(self.propagator.send_state(self.name, neighbor, self.current_state))
                else:
                    transmission_feature = (
                        self._normalized_mean_transmission_wait()
                        if (
                            self.mode in ('Pure_DQN', 'Pure_PPO')
                            or self.gnn_node_feature_dim in (NODE_FEATURE_DIM, ISACR_NODE_FEATURE_DIM)
                        )
                        else self.transmission_size.get(neighbor, 0) / self.memory
                    )
                    message = [
                        self.is_producing,
                        1-self.current_memory_occupy/self.memory,
                        self._normalized_compute_ability(),
                        self._normalized_compute_time(
                            self.computing_remain / self.computing_ability
                            - self.is_computing * (self.env.now - self.last_computing_time)
                        ),
                        transmission_feature,
                    ]
                    if self.mode in ('Pure_DQN', 'Pure_PPO'):
                        message.append(self._normalized_compute_completion_time())
                    self.env.process(self.propagator.send_state(self.name, neighbor, message))

    def all_start(self):
        super().all_start()
        self.env.process(self.computing_packet())
        self.env.process(self.offload_to_ground())
        if "GNN" in self.mode:
            if self.gnn_node_feature_dim in (NODE_FEATURE_DIM, ISACR_NODE_FEATURE_DIM):
                if getattr(self.gnn, 'direct_concat', False):
                    self.env.process(self.direct_concat_exchanger())
                else:
                    self.env.process(self.gnn_embedding_exchanger())
            else:
                self.env.process(self.two_hop_info_exchanger())

    def self_missing(self):
        self.active = False
        self.active_transmissions.clear()
        while self.waiting_forward:
            packet = self.waiting_forward.popleft()
            record_queue_wait(packet, 'waiting_forward', self.env.now)
            self._drop_no_route(
                packet, packet.size, packet.information[1],
                packet.information[2], packet.creation_time, reason='node_failure',
            )
        while self.computing_queue.items:
            packet=yield self.computing_queue.get()
            reservation_book = getattr(self.propagator, 'reservation_book', None)
            if reservation_book is not None:
                reservation_book.release(packet)
            self.logger.log(f"Time {self.env.now:.3f}: {packet} is dropped because of satellite missing.")
        self.current_computing_queue_size = 0
        while self.offload_queue.items:
            packet = yield self.env.process(self.pop_offload())
            reservation_book = getattr(self.propagator, 'reservation_book', None)
            if reservation_book is not None:
                reservation_book.release(packet)
            self.logger.log(f"Time {self.env.now:.3f}: {packet} is dropped because of satellite missing.")
        for neighbor in self.neighbors:
            while self.transmission_queue[neighbor].items:
                packet = yield self.env.process(self.pop_transmission(neighbor))
                reservation_book = getattr(self.propagator, 'reservation_book', None)
                if reservation_book is not None:
                    reservation_book.release(packet)
                self.logger.log(f"Time {self.env.now:.3f}: {packet} is dropped because of satellite missing.")
            while self.forward_queue.items:
                packet = yield self.env.process(self.forward_queue.get())
                reservation_book = getattr(self.propagator, 'reservation_book', None)
                if reservation_book is not None:
                    reservation_book.release(packet)
                self.logger.log(f"Time {self.env.now:.3f}: {packet} is dropped because of satellite missing.")
        self.current_memory_occupy=0
        if self.mode in ["Tradition","Ground"]:
            self.propagator.graph.nodes[self.name]['current_memory_occupy'] = 0
        self.is_producing = 0
        self.computing_remain = 0

    def gather_two_hop_info(self):
        two_hop_info = {}
        exchange_hop_limit = max(self.gnn_information_hops - 2, 0)
        for node, (_, hops) in self.routing_tables.items():
            if hops <= exchange_hop_limit:
                if hops != 0:
                    default_node_info = tuple([0,1,0,0,0] + [0] * (self.gnn_node_feature_dim - NODE_FEATURE_DIM)) if self.gnn_node_feature_dim in (NODE_FEATURE_DIM, ISACR_NODE_FEATURE_DIM) else (0,1,0)
                    node_info = self.three_hop_info_db.get(node, {}).get('node_info', default_node_info)
                    link_info = {}
                    for neighbor in self.adjacency_table[node][0]:
                        link_info[neighbor] = self.three_hop_info_db.get(node, {}).get('link_info', {}).get(neighbor,0)
                else:
                    node_info=(
                        self.is_producing,
                        (self.memory-self.current_memory_occupy)/self.memory,
                        self._normalized_compute_ability(),
                        self._normalized_compute_time(
                            self.computing_remain / self.computing_ability
                            - self.is_computing * (self.env.now - self.last_computing_time)
                        ),
                    )
                    if self.gnn_node_feature_dim in (NODE_FEATURE_DIM, ISACR_NODE_FEATURE_DIM):
                        node_info += (self._normalized_mean_transmission_wait(),)
                        if self.gnn_node_feature_dim == ISACR_NODE_FEATURE_DIM:
                            node_info += (self._normalized_compute_completion_time(),)
                    link_info = {}
                    for neighbor in self.neighbors:
                        link_info[neighbor] = (
                            self._normalized_link_transmission_wait(neighbor)
                            if self.gnn_node_feature_dim in (NODE_FEATURE_DIM, ISACR_NODE_FEATURE_DIM)
                            else self.transmission_size[neighbor]/self.memory
                        )
                two_hop_info[node] = {
                    'timestamp': self.env.now,
                    'node_info': node_info,
                    'link_info': link_info
                }
        return two_hop_info

    def current_gnn_raw_state(self):
        values = [
            self.is_producing,
            1 - self.current_memory_occupy / self.memory,
            self._normalized_compute_ability(),
            self._normalized_compute_time(
                self.computing_remain / self.computing_ability
                - self.is_computing * (self.env.now - self.last_computing_time)
            ),
            self._normalized_mean_transmission_wait(),
        ]
        if (getattr(getattr(self, 'gnn', None), 'direct_concat', False)
                or self.gnn_node_feature_dim == ISACR_NODE_FEATURE_DIM):
            values.append(self._normalized_compute_completion_time())
        return values

    def current_fault_node_state(self):
        valid_states = [
            list(self.neighbor_states[neighbor][:self.gnn_node_feature_dim])
            for neighbor in self.neighbors
            if neighbor in self.neighbor_states
            and len(self.neighbor_states[neighbor]) >= self.gnn_node_feature_dim
            and self.env.now - self.last_heartbeat.get(neighbor, -float('inf')) <= self.heartbeat_timeout
        ]
        if not valid_states:
            return [1, 0, 0, 1, 1] + ([0] if self.gnn_node_feature_dim == ISACR_NODE_FEATURE_DIM else [])
        means = [sum(state[index] for state in valid_states) / len(valid_states)
                 for index in (2, 3, 4)]
        factor = self.fault_fill_multiplier
        ability = _clip_normalized(means[0] / factor) if self.fault_compute_double_mean else 0
        wait = _clip_normalized(means[1] * factor) if self.fault_compute_double_mean else 1
        transmission = _clip_normalized(means[2] * (factor if self.fault_compute_double_mean else 2))
        result = [1, 0, ability, wait, transmission]
        if self.gnn_node_feature_dim == ISACR_NODE_FEATURE_DIM:
            result.append(_clip_normalized(sum(state[5] for state in valid_states)
                                          / len(valid_states) / factor)
                          if self.fault_compute_double_mean else 0)
        return result

    def _gnn_fault_features(self):
        raw = self.current_fault_node_state()
        device = next(self.gnn.parameters()).device
        edges = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 2), dtype=torch.float, device=device)
        x = torch.tensor([raw], dtype=torch.float, device=device)
        with torch.no_grad():
            if hasattr(self.gnn, 'aggregate_stage'):
                first = self.gnn.aggregate_stage(x, edges, 1, edge_attr)
                second = self.gnn.aggregate_stage(first, edges, 2, edge_attr)
            else:
                first = torch.nn.functional.leaky_relu(self.gnn.conv1(x, edges))
                second = self.gnn.conv2(first, edges)
            if hasattr(self.gnn, 'combine_hops'):
                final = self.gnn.combine_hops(x, first, second)
            elif hasattr(self.gnn, 'combine_embeddings'):
                final = self.gnn.combine_embeddings(first, second)
            else:
                final = second
        return raw, first[0].cpu().tolist(), final[0].cpu().tolist()

    def receive_gnn_embedding_message(self, neighbor, stage, version, value):
        if stage == 'pipeline':
            if version <= self.gnn_layer2_messages.get(neighbor, (-1, []))[0]:
                return
            raw_dim = self.gnn_node_feature_dim
            hidden_dim = getattr(self.gnn, 'message_dim', self.gnn.output_dim)
            self.gnn_raw_messages[neighbor] = (version, list(value[:raw_dim]))
            self.gnn_layer1_messages[neighbor] = (version, list(value[raw_dim:raw_dim + hidden_dim]))
            self.gnn_layer2_messages[neighbor] = (version, list(value[raw_dim + hidden_dim:]))
            self.neighbor_states[neighbor] = list(value[:raw_dim])
            self.last_heartbeat[neighbor] = self.env.now
            return
        caches = {
            'raw': self.gnn_raw_messages,
            'layer1': self.gnn_layer1_messages,
            'layer2': self.gnn_layer2_messages,
        }
        if getattr(self.gnn, 'direct_concat', False):
            if version <= caches[stage].get(neighbor, (-1, []))[0]:
                return
            if stage == 'layer2':
                self.direct_concat_received_at[neighbor] = self.env.now
                if not hasattr(self, 'direct_concat_origins'):
                    self.direct_concat_origins = {}
                self.direct_concat_origins[neighbor] = tuple(getattr(value, 'node_names', ()))
                from ablation.message_aggregation import DirectConcatView
                if not hasattr(self, 'direct_concat_history'):
                    self.direct_concat_history = {}
                records = self.direct_concat_history.setdefault(neighbor, [])
                records.append((version, self.env.now, DirectConcatView(
                    value, self.direct_concat_origins[neighbor])))
                del records[:-2]
                # The sender's own resource block drives admission masking.
                # Copy it so local forwarding debits never alter the message.
                self.neighbor_states[neighbor] = list(value[:self.gnn_node_feature_dim])
                self.on_neighbor_state_received(neighbor, sent_at=version * self.state_update_period)
        caches[stage][neighbor] = (version, list(value))
        self.last_heartbeat[neighbor] = self.env.now

    def _gnn_link_features(self, neighbor):
        record = self.three_hop_info_db.get(neighbor, {})
        remote = (record.get('link_info', {}).get(self.name, 0.0)
                  if self.env.now - record.get('timestamp', -float('inf')) <= self.heartbeat_timeout else 0.0)
        return [remote, self._normalized_link_transmission_wait(neighbor)]

    def aggregate_gnn_layer(self, self_value, neighbor_cache, version, layer):
        fault = self._gnn_fault_features()
        default_value = fault[0 if layer == 1 else 1]
        values = [self_value]
        for neighbor in self.neighbors:
            cached_version, cached_value = neighbor_cache.get(
                neighbor, (None, default_value)
            )
            fresh = (cached_version is not None and cached_version < version
                     and self.env.now - self.last_heartbeat.get(neighbor, -float('inf'))
                     <= self.heartbeat_timeout)
            values.append(cached_value if fresh else default_value)
        device = next(self.gnn.parameters()).device
        x = torch.tensor(values, dtype=torch.float, device=device)
        if len(values) > 1:
            edge_index = torch.tensor(
                [list(range(1, len(values))), [0] * (len(values) - 1)],
                dtype=torch.long,
                device=device,
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        with torch.no_grad():
            if hasattr(self.gnn, 'aggregate_stage'):
                # GAT uses the same two directional link-wait features as
                # encode_for_GNN, including the distributed inference path.
                edge_values = []
                for neighbor in self.neighbors:
                    record = self.three_hop_info_db.get(neighbor, {})
                    remote_load = (
                        record.get('link_info', {}).get(self.name, 0.0)
                        if self.env.now - record.get('timestamp', -float('inf'))
                        <= self.heartbeat_timeout else 0.0
                    )
                    edge_values.append([
                        remote_load, self._normalized_link_transmission_wait(neighbor),
                    ])
                edge_attr = torch.tensor(edge_values, dtype=torch.float, device=device).reshape(-1, 2)
                output = self.gnn.aggregate_stage(x, edge_index, layer, edge_attr)[0]
            elif layer == 1:
                output = torch.nn.functional.leaky_relu(self.gnn.conv1(x, edge_index))[0]
            else:
                output = self.gnn.conv2(x, edge_index)[0]
        return output.detach().cpu().tolist()

    def direct_concat_slots(self):
        if hasattr(self, 'action_neighbors'):
            return list(self.action_neighbors)
        slots = list(getattr(self, 'direct_concat_neighbor_slots', [None] * 4))
        for neighbor in self.neighbors:
            if neighbor not in slots:
                index = next(i for i, node in enumerate(slots) if node not in self.neighbors)
                slots[index] = neighbor
        self.direct_concat_neighbor_slots = slots
        return slots

    def fresh_direct_concat_view(self, neighbor, before_tick=None):
        if before_tick is not None:
            version = round(before_tick / self.state_update_period)
            for cached_version, received_at, value in reversed(
                    getattr(self, 'direct_concat_history', {}).get(neighbor, ())):
                if (neighbor in self.neighbors and cached_version < version
                        and received_at < before_tick - 1e-9
                        and self.env.now <= received_at + self.heartbeat_timeout):
                    return value
            return None
        received_at = self.direct_concat_received_at.get(neighbor, -float('inf'))
        if neighbor not in self.neighbors or self.env.now - received_at > self.heartbeat_timeout:
            return None
        return self.gnn_layer2_messages.get(neighbor, (None, None))[1]

    def direct_concat_raw_records(self, before_tick=None):
        records = []
        for node in self.direct_concat_slots():
            view = self.fresh_direct_concat_view(node, before_tick)
            records.append((node, view[:6] if view is not None else [], view is not None))
        return records

    def direct_concat_fault_feature(self, values):
        if not values:
            return [1, 0, 0, 1, 1, self._local_fault_compute_score()]
        mean = [sum(row[i] for row in values) / len(values) for i in range(6)]
        factor = self.fault_fill_multiplier
        ability = _clip_normalized(mean[2] / factor)
        wait = _clip_normalized(mean[3] * factor)
        return [1, 0, ability, wait, _clip_normalized(mean[4] * factor),
                _clip_normalized(mean[5] / factor)]

    def build_direct_concat_view(self, raw_state, recipient=None, before_tick=None):
        from ablation.message_aggregation import DirectConcatView
        direct = [(name, value, valid and name != recipient)
                  for name, value, valid in self.direct_concat_raw_records(before_tick)]
        second = []
        for neighbor, _, valid in direct:
            view = self.fresh_direct_concat_view(neighbor, before_tick) if valid else None
            origins = getattr(view, 'node_names',
                              getattr(self, 'direct_concat_origins', {}).get(neighbor, ()))
            for index in range(1, 5):
                origin = origins[index] if len(origins) > index else None
                usable = view is not None and (recipient is None or origin != recipient)
                second.append((origin, view[index * 6:(index + 1) * 6] if usable else [], usable))
        direct_fill = self.direct_concat_fault_feature([value for _, value, valid in direct if valid])
        second_fill = self.direct_concat_fault_feature([value for _, value, valid in second if valid])
        values = list(raw_state) + [x for _, value, valid in direct for x in (value if valid else direct_fill)] + [
            x for _, value, valid in second for x in (value if valid else second_fill)]
        names = [self.name] + [name if valid else None for name, _, valid in direct + second]
        return DirectConcatView(values, names)

    def direct_concat_exchanger(self):
        next_tick = math.floor(self.env.now / self.state_update_period + 1e-9) + 1
        wait = next_tick * self.state_update_period - self.env.now
        if wait > 1e-9:
            yield self.env.timeout(wait)
        self.gnn_cycle_version = next_tick
        while self.active:
            version = self.gnn_cycle_version
            raw_state = self.current_gnn_raw_state()[:self.gnn.raw_feature_dim]
            self.gnn_raw_snapshots[version] = raw_state
            local_view = self.build_direct_concat_view(raw_state, before_tick=self.env.now)
            self.direct_concat_local_view = local_view
            for neighbor in sorted(set(self.neighbors) | self.pending_state_neighbors):
                self.env.process(self.propagator.send_gnn_embedding_message(
                    self.name, neighbor, 'layer2', version, self.build_direct_concat_view(
                        raw_state, recipient=neighbor, before_tick=self.env.now)
                ))

            neighbor_views = {neighbor: self.fresh_direct_concat_view(neighbor)
                              for neighbor in self.neighbors}
            valid_views = [view for view in neighbor_views.values() if view is not None]
            fills = [self.direct_concat_fault_feature([view[i:i + 6] for view in valid_views])
                     for i in range(0, self.gnn.output_dim, 6)]
            fault_view = [x for feature in fills for x in feature]
            self.gnn_active_neighbor_embeddings = {
                neighbor: view if view is not None else list(fault_view)
                for neighbor, view in neighbor_views.items()
            }
            self.gnn_direct_concat_fault_view = fault_view
            self.gnn_active_version = version
            self.gnn_cycle_version += 1
            self.encode_data = None
            for old_version in list(self.gnn_raw_snapshots):
                if (old_version < version - 1
                        and old_version != self.gnn_active_version):
                    del self.gnn_raw_snapshots[old_version]
            yield self.env.timeout(self.gnn_cycle_version * self.state_update_period - self.env.now)

    def gnn_embedding_exchanger(self):
        next_tick = math.floor(self.env.now / self.state_update_period + 1e-9) + 1
        wait = next_tick * self.state_update_period - self.env.now
        if wait > 1e-9:
            yield self.env.timeout(wait)
        self.gnn_cycle_version = next_tick
        while self.active:
            version = self.gnn_cycle_version
            raw_state = self.current_gnn_raw_state()
            self.gnn_raw_snapshots[version] = raw_state
            layer1 = self.aggregate_gnn_layer(
                raw_state, self.gnn_raw_messages, version, layer=1
            )
            layer2 = self.aggregate_gnn_layer(
                layer1, self.gnn_layer1_messages, version, layer=2
            )
            device = next(self.gnn.parameters()).device
            if hasattr(self.gnn, 'combine_hops'):
                with torch.no_grad():
                    final_embedding = self.gnn.combine_hops(
                        torch.tensor(raw_state, dtype=torch.float, device=device),
                        torch.tensor(layer1, dtype=torch.float, device=device),
                        torch.tensor(layer2, dtype=torch.float, device=device),
                    ).cpu().tolist()
            elif hasattr(self.gnn, 'combine_embeddings'):
                with torch.no_grad():
                    final_embedding = self.gnn.combine_embeddings(
                        torch.tensor(layer1, dtype=torch.float, device=device),
                        torch.tensor(layer2, dtype=torch.float, device=device),
                    ).cpu().tolist()
            else:
                final_embedding = layer2
            message = list(raw_state) + layer1 + final_embedding
            for neighbor in self.neighbors:
                self.env.process(self.propagator.send_gnn_embedding_message(
                    self.name, neighbor, 'pipeline', version, message
                ))
            self.gnn_fault_embedding = self._gnn_fault_features()[2]
            self.gnn_active_neighbor_embeddings = {
                neighbor: value for neighbor, (cached_version, value) in self.gnn_layer2_messages.items()
                if neighbor in self.neighbors and cached_version < version
                and self.env.now - self.last_heartbeat.get(neighbor, -float('inf')) <= self.heartbeat_timeout
            }
            self.gnn_active_version = version
            self.gnn_cycle_version += 1
            self.encode_data = None
            for old_version in list(self.gnn_raw_snapshots):
                if old_version < version - 1:
                    del self.gnn_raw_snapshots[old_version]
            yield from self._wait_for_state_exchange()
            if not self.active:
                break

    def deliver_two_hop_info(self):
        two_hop_info = self.gather_two_hop_info()
        for neighbor in self.neighbors:
            self.env.process(self.propagator.send_two_hop_info(self.name, neighbor, two_hop_info))

    def two_hop_info_exchanger(self):
        while self.active:
            yield self.env.timeout(self.state_update_period)
            if not self.active:
                break
            self.deliver_two_hop_info()

    def update_three_hop_info_db(self, received_info):
        updated = False
        for node, data in received_info.items():
            if node not in self.three_hop_info_db:
                self.three_hop_info_db[node] = data
                updated = True
            else:
                old_time = self.three_hop_info_db[node].get('timestamp', 0)
                if data['timestamp'] > old_time:
                    self.three_hop_info_db[node] = data
                    updated = True
        return updated

    def construct_gnn_graph(self):
        if self.gnn_node_feature_dim in (NODE_FEATURE_DIM, ISACR_NODE_FEATURE_DIM):
            def known_neighbors(node):
                if node == self.name:
                    return set(self.neighbors)
                record = self.adjacency_table.get(node)
                if record is None:
                    return set()
                neighbors, timestamp = record
                if self.env.now - timestamp > self.heartbeat_timeout:
                    return set()
                return set(neighbors)

            def known_node_state(node):
                if node == self.name:
                    # DirectConcat's sixth feature travels in its dedicated
                    # messages; the legacy topology cache remains five-dimensional.
                    return self.current_gnn_raw_state()[:self.gnn_node_feature_dim]
                if node in self.neighbor_states:
                    return list(self.neighbor_states[node][:self.gnn_node_feature_dim])
                record = self.three_hop_info_db.get(node, {})
                if self.env.now - record.get('timestamp', -float('inf')) <= self.heartbeat_timeout:
                    value = record.get('node_info')
                    if value is not None:
                        return list(value[:self.gnn_node_feature_dim])
                return self.current_fault_node_state()

            def known_link_load(source, target):
                if source == self.name:
                    return self._normalized_link_transmission_wait(target)
                record = self.three_hop_info_db.get(source, {})
                if self.env.now - record.get('timestamp', -float('inf')) <= self.heartbeat_timeout:
                    return float(record.get('link_info', {}).get(target, 0.0))
                return 0.0

            nodes = {}
            selected_nodes = {
                node for node, (_, hops) in self.routing_tables.items()
                if hops <= self.gnn_information_hops
            }
            selected_nodes.add(self.name)
            for node in selected_nodes:
                nodes[node] = {
                    'attributes': known_node_state(node),
                    'position_encoding': [0, 0, 0, 0, 0],
                    'is_neighbor': node in self.neighbors,
                }
            edges = []
            for source in selected_nodes:
                for target in known_neighbors(source):
                    if target not in selected_nodes:
                        continue
                    edges.append({
                        'source': source,
                        'target': target,
                        'attributes': known_link_load(source, target),
                    })
            for index in range(4 - len(self.neighbors)):
                nodes[f"Z_missing_{index}"] = {
                    'attributes': self.current_fault_node_state(),
                    'position_encoding': [0, 0, 0, 0, 0],
                    'is_neighbor': True,
                }
            return nodes, edges

        nodes = {}
        edges = []
        routing_hop_limit = max(self.gnn_information_hops - 1, 0)
        for node, (_, hops) in self.routing_tables.items():
            if hops <= routing_hop_limit:
                if hops != 0:
                    remote_record = self.three_hop_info_db.get(node, {})
                    record_age = self.env.now - remote_record.get('timestamp', -float('inf'))
                    record_is_fresh = record_age <= self.heartbeat_timeout
                    default_node_info = (0, 1, 0, 0) if self.gnn_node_feature_dim == 4 else (0, 1, 0)
                    node_info = (
                        remote_record.get('node_info', default_node_info)
                        if record_is_fresh else (0, 1, 0)
                    )
                    position_encoding = []
                    for neighbor in self.neighbors:
                        if node in self.neighbor_hops[neighbor]:
                            position_encoding.append(self.neighbor_hops[neighbor][node])
                        else:
                            position_encoding.append(self.max_hop*2)
                    if len(self.neighbors)<4:
                        for _ in range(4-len(self.neighbors)):
                            position_encoding.append(self.max_hop*2)
                    position_encoding.append(max(min(position_encoding)-1,0))
                    nodes[node] = {
                        'attributes': node_info,
                        'position_encoding': position_encoding,
                        'is_neighbor': True if node in self.neighbors else False
                    }

                    link_info = {}
                    if record_is_fresh:
                        for neighbor in self.adjacency_table[node][0]:
                            link_info[neighbor] = remote_record.get('link_info', {}).get(neighbor, 0)

                    for neighbor, edge_data in link_info.items():
                        if neighbor in self.routing_tables and self.routing_tables[neighbor][1] <= routing_hop_limit:
                            edges.append({
                                'source': node,
                                'target': neighbor,
                                'attributes': edge_data
                            })
                elif self.gnn_include_self:
                    nodes[node] = {
                        'attributes': (
                            self.is_producing,
                            (self.memory-self.current_memory_occupy)/self.memory,
                            self._normalized_compute_time(
                                self.computing_remain / self.computing_ability
                                - self.is_computing * (self.env.now - self.last_computing_time)
                            ),
                        ) + ((sum(self.transmission_size.values()) / self.memory,) if self.gnn_node_feature_dim == 4 else ()),
                        'position_encoding': [0,0,0,0,0],
                        'is_neighbor': False,
                    }
                    for neighbor in self.neighbors:
                        if neighbor in self.routing_tables and self.routing_tables[neighbor][1] <= routing_hop_limit:
                            edges.append({
                                'source': node,
                                'target': neighbor,
                                'attributes': self.transmission_size[neighbor]/self.memory,
                            })
        if len(self.neighbors)<4:
            for i in range(4-len(self.neighbors)):
                position_encoding=[self.max_hop*2, self.max_hop*2, self.max_hop*2, self.max_hop*2]
                position_encoding[i]=0
                position_encoding.append(max(min(position_encoding) - 1, 0))
                nodes[f"Z_missing_{i}"]={
                    'attributes': (1,0,1,0) if self.gnn_node_feature_dim == 4 else (1,0,1),
                    'position_encoding': position_encoding,
                    'is_neighbor':True

                }

        return nodes, edges

    def encode_for_GNN(self, nodes, edges):
        neighbors_of_self = [node for node in nodes if nodes[node]['is_neighbor']]
        other_nodes = [node for node in nodes if node not in neighbors_of_self ]

        sorted_neighbors = sorted(neighbors_of_self)
        sorted_nodes  = sorted_neighbors + other_nodes

        node_name_to_id = {name: idx for idx, name in enumerate(sorted_nodes)}

        concatenated_features = []
        for node in sorted_nodes:
            node_attr = torch.tensor(nodes[node]['attributes'], dtype=torch.float)
            if node_attr.numel() != self.gnn_node_feature_dim:
                raise ValueError(f"GNN node {node} has {node_attr.numel()} features; expected {self.gnn_node_feature_dim}")
        #     node_pos_enc = torch.tensor(nodes[node]['position_encoding'], dtype=torch.float)
        #     concatenated_features.append(torch.cat((node_attr, node_pos_enc)))
            concatenated_features.append(node_attr)

        # x = torch.stack(concatenated_features)

        x = torch.stack(concatenated_features)

        valid_node_set = set(node_name_to_id.keys())
        valid_edges = [(node_name_to_id[edge['target']], node_name_to_id[edge['source']]) for edge in edges if
                       edge['source'] in valid_node_set and edge['target'] in valid_node_set]
        if valid_edges:
            edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        edge_attributes_map = {(edge['source'], edge['target']): edge['attributes'] for edge in edges}

        edge_attr_list = []
        for edge in valid_edges:
            forward_attr = edge_attributes_map.get((sorted_nodes[edge[0]], sorted_nodes[edge[1]]),1)
            reverse_attr = edge_attributes_map.get((sorted_nodes[edge[1]], sorted_nodes[edge[0]]),1)
            combined_attr = torch.tensor([forward_attr, reverse_attr],dtype=torch.float)
            edge_attr_list.append(combined_attr)

        if edge_attr_list:
            edge_attr = torch.stack(edge_attr_list)
        else:
            edge_attr = torch.empty((0, 2), dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

class SatelliteNetworkSimulator_OnbardComputing(SatelliteNetworkSimulator):
    def __init__(self,mode,select_mode,q_net,gnn,epsilon,reward_factors,device,mission_possibility,poisson_rate_g,poisson_rate_s,packet_frequency,computing_demand_factor,computing_demand_factor_2,size_after_computing_factor,size_after_computing_1,graph,landmarks,ground_cover_sats,mean_interval_time,memory,computing_abilities,transmission_rate,downlink_rate,downstream_delays,packet_size_range,state_update_period,logger,ppo_shuffle_func=None,scenario_seed=None,fault_compute_double_mean=True,fault_fill_multiplier=2.0,historical_self_feature_elimination=False,disable_self_feature_elimination=False,gnn_information_hops=4,gnn_include_self=False,gnn_node_feature_dim=3,experience_callback=None,tradition_replan_interval=None,require_computed_before_downlink=False,tradition_use_genetic_search=False,tradition_ga_population_size=20,tradition_ga_generations=10,tradition_ga_path_candidates=3,tradition_ga_mutation_rate=0.1,workload_trace_path=None,workload_trace_sha256=None,workload_trace_task_count=None,uncomputed_reach_penalty=0.0,heartbeat_timeout=0.15,topology_graph=None):
        self.q_net = q_net
        self.gnn = gnn
        self.aligned_graphsage_policy = getattr(gnn, 'aligned_graphsage_policy', False)
        self.satellite_class = getattr(gnn, 'satellite_class', Satellite_with_Computing)
        self.ppo_shuffle_func = ppo_shuffle_func
        self.mode=mode
        self.fault_compute_double_mean=fault_compute_double_mean
        self.fault_fill_multiplier=fault_fill_multiplier
        self.historical_self_feature_elimination=historical_self_feature_elimination
        self.disable_self_feature_elimination=disable_self_feature_elimination
        self.require_computed_before_downlink=require_computed_before_downlink
        self.gnn_information_hops=gnn_information_hops
        self.gnn_include_self=gnn_include_self
        self.gnn_node_feature_dim=gnn_node_feature_dim
        self.tradition_replan_interval = tradition_replan_interval
        self.tradition_use_genetic_search = tradition_use_genetic_search
        self.tradition_ga_population_size = tradition_ga_population_size
        self.tradition_ga_generations = tradition_ga_generations
        self.tradition_ga_path_candidates = tradition_ga_path_candidates
        self.tradition_ga_mutation_rate = tradition_ga_mutation_rate
        self.epsilon=epsilon
        self.reward_function=Reward_Function(*reward_factors, uncomputed_reach_penalty=uncomputed_reach_penalty)
        self.device=device
        self.mission_possibility=mission_possibility
        self.poisson_rate_g = poisson_rate_g
        self.poisson_rate_s = poisson_rate_s
        self.packet_frequency=packet_frequency
        self.computing_demand_factor=computing_demand_factor
        self.computing_demand_factor_2=computing_demand_factor_2
        self.size_after_computing_factor=size_after_computing_factor
        self.size_after_computing_1=size_after_computing_1
        self.graph = graph
        self.topology_graph = topology_graph if topology_graph is not None else graph
        self.max_hop=nx.diameter(self.graph)
        self.max_size=packet_size_range[1]
        self.env = simpy.Environment()
        self.memory = memory
        self.computing_abilities = dict(computing_abilities)
        missing_abilities = sorted(set(self.graph.nodes) - set(self.computing_abilities))
        if missing_abilities:
            raise ValueError(f"Missing computing abilities for satellites: {missing_abilities[:5]}")
        self.logger=logger
        self.transmission_rate=transmission_rate
        self.downstream_delays=downstream_delays
        self.downlink_rate=downlink_rate
        self.state_update_period=state_update_period
        self.heartbeat_timeout = heartbeat_timeout
        self.landmarks=landmarks
        self.ground_cover_sats=ground_cover_sats
        self.scenario_seed = scenario_seed
        self.traffic_rngs = {}
        self.workload_trace_path = Path(workload_trace_path).resolve() if workload_trace_path else None
        self.workload_trace_expected_sha256 = workload_trace_sha256
        self.workload_trace_expected_count = (
            int(workload_trace_task_count)
            if workload_trace_task_count is not None else None
        )
        self.fixed_workload_sessions = None
        self.fixed_workload_max_time = None
        self.workload_trace_hasher = hashlib.sha256()
        self.workload_trace_count = 0
        self.direct_satellites = set(sum(self.landmarks.values(), []))
        self.statics_datas = {
            'Total': 0,
            'Reached_0': 0,
            'Reached_1': 0,
            'Reached_after_computed_0': 0,
            'Reached_after_computed_1': 0,
            'Lost_upload': 0,
            'Lost_relay_0': 0,
            'Lost_relay_1': 0,
            **{key: 0 for key in LOSS_BREAKDOWN_KEYS},
            'Total_delay_0': 0,
            'Total_delay_1': 0,
            'Total_computation_delay_0': 0,
            'Total_computation_delay_1': 0,
            'Total_queueing_delay_0': 0,
            'Total_queueing_delay_1': 0,
            'Total_propagation_delay_0': 0,
            'Total_propagation_delay_1': 0,
            'Total_transmission_delay_0': 0,
            'Total_transmission_delay_1': 0,
            'Total_hops_0': 0,
            'Total_hops_1': 0,
            'Is_computing': 0,
            'Computing_waiting_time': 0,
        }
        self.satellite_names=[node for node in self.graph.nodes]
        if self.workload_trace_path is not None:
            self._load_fixed_workload_trace()
        self.satellites={}
        self.select_mode=select_mode
        for node in self.graph.nodes:
            self.graph.nodes[node]['computing_ability'] = self.computing_abilities[node]
            satellite_name_with_suffix = node
            self.satellites[satellite_name_with_suffix] = self.satellite_class(
                self.mode,
                self.select_mode,
                self.epsilon,
                self.max_hop,
                self.max_size,
                self.device,
                self.env,
                satellite_name_with_suffix,
                [neighbor for neighbor in self.graph.neighbors(node)],
                self.memory,
                self.computing_abilities[node],
                self.transmission_rate,
                self.downlink_rate,
                self.state_update_period,
                True if node in self.direct_satellites else False,
                self.logger,
                self.statics_datas,
                heartbeat_timeout=self.heartbeat_timeout,
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
                 topology_neighbors=[
                     neighbor for neighbor in self.topology_graph.neighbors(node)
                 ] if node in self.topology_graph else [],
             )
        self.propagator = Propagator_Computing(self.env, graph, logger, self.satellites,self.statics_datas, True if self.mode in ["Tradition","Ground"] else False)
        self.propagator.experience_callback = experience_callback
        self.mean_interval_time=mean_interval_time
        self.size_range = packet_size_range
        self.propagator.trans_parameters(self.max_hop,self.downstream_delays,self.reward_function)
        for satellite in self.satellites:
            self.satellites[satellite].adjacency_table=self.extract_adjacency_dict()
            self.satellites[satellite].trans_parameters(self.q_net,self.reward_function,self.direct_satellites,self.gnn,self.ppo_shuffle_func)
            self.satellites[satellite].set_propagator(self.propagator)
            self.satellites[satellite].build_routing_table()

    def _load_fixed_workload_trace(self):
        content = self.workload_trace_path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if (
            self.workload_trace_expected_sha256 is not None
            and digest != self.workload_trace_expected_sha256
        ):
            raise ValueError(
                f'Workload trace SHA256 mismatch: {self.workload_trace_path}'
            )

        sessions_by_satellite = {name: [] for name in self.satellite_names}
        packet_count = 0
        max_session_end_time = 0.0
        for line_number, raw_line in enumerate(content.splitlines(), 1):
            if not raw_line.strip():
                continue
            session = json.loads(raw_line)
            satellite = str(session['satellite'])
            if satellite not in sessions_by_satellite:
                raise ValueError(
                    f'Unknown satellite in workload trace line {line_number}: '
                    f'{satellite}'
                )
            destination = session.get('destination')
            if destination is not None and str(destination) not in sessions_by_satellite:
                raise ValueError(
                    f'Unknown destination satellite in workload trace line '
                    f'{line_number}: {destination}'
                )
            start = float(session['session_start_time'])
            end = float(session['session_end_time'])
            if start < 0 or end < start:
                raise ValueError(
                    f'Invalid session time in workload trace line {line_number}'
                )
            packets = session['packets']
            for expected_index, packet in enumerate(packets, 1):
                if int(packet['packet_index']) != expected_index:
                    raise ValueError(
                        f'Invalid packet index in workload trace line {line_number}'
                    )
                for field in (
                    'packet_size', 'computing_demand', 'size_after_computing'
                ):
                    if int(packet[field]) < 0:
                        raise ValueError(
                            f'Negative {field} in workload trace line {line_number}'
                        )
            packet_count += len(packets)
            max_session_end_time = max(max_session_end_time, end)
            sessions_by_satellite[satellite].append(session)

        for satellite, sessions in sessions_by_satellite.items():
            sessions.sort(key=lambda item: int(item['session_index']))
            previous_end = 0.0
            for expected_index, session in enumerate(sessions, 1):
                if int(session['session_index']) != expected_index:
                    raise ValueError(
                        f'Non-contiguous session index for {satellite}'
                    )
                start = float(session['session_start_time'])
                if start < previous_end:
                    raise ValueError(f'Overlapping sessions for {satellite}')
                previous_end = float(session['session_end_time'])

        if (
            self.workload_trace_expected_count is not None
            and packet_count != self.workload_trace_expected_count
        ):
            raise ValueError(
                f'Workload trace task count mismatch: expected '
                f'{self.workload_trace_expected_count}, got {packet_count}'
            )
        self.workload_trace_expected_count = packet_count
        self.fixed_workload_sessions = sessions_by_satellite
        self.fixed_workload_max_time = max_session_end_time
        self.workload_trace_hasher = hashlib.sha256(content)

    def record_workload_trace(self, satellite, destination, task_type, size,
                              computing_demand, size_after_computing):
        record = (
            f'{satellite}|{destination}|{float(self.env.now):.9f}|'
            f'{int(task_type)}|{int(size)}|{int(computing_demand)}|'
            f'{int(size_after_computing)}\n'
        )
        self.workload_trace_hasher.update(record.encode('utf-8'))
        self.workload_trace_count += 1

    def extract_adjacency_dict(self):
        adjacency_dict = {}
        for node in self.satellite_names:
            neighbors = sorted(f"{neighbor}" for neighbor in self.graph.neighbors(node))
            adjacency_dict[f"{node}"] = (neighbors, self.env.now)
        return adjacency_dict

    def generate_traffic(self, satellite):
        traffic_rng = self.traffic_rngs.setdefault(
            satellite,
            random.Random(f'traffic:{self.scenario_seed}:{satellite}')
            if self.scenario_seed is not None else random,
        )
        while satellite in self.satellite_names:
            self.satellites[satellite].is_producing=0
            if self.satellites[satellite].name in self.ground_cover_sats:
                session_start_time = min(traffic_rng.expovariate(1.0 / self.poisson_rate_g), self.poisson_rate_g * 3)
            else:
                session_start_time = min(traffic_rng.expovariate(1.0 / self.poisson_rate_s), self.poisson_rate_s * 3)
            yield self.env.timeout(session_start_time)
            if not satellite in self.satellite_names:
                self.logger.log(f"Time {self.env.now:.3f}: {satellite} is missed, packets failed to generate.")
                break
            type = traffic_rng.choices([0, 1], weights=self.mission_possibility, k=1)[0]
            if satellite in self.direct_satellites:
                continue
            else:
                if self.select_mode == 1 or self.mode in ["Tradition","Ground"]:
                    min_hops = np.inf
                    min_hops_destinations=[]
                    for destination in sorted(self.direct_satellites):
                        if destination in self.satellites[satellite].routing_tables:
                            hops = self.satellites[satellite].routing_tables[destination][1]
                        else:
                            hops = np.inf
                        if hops < min_hops:
                            min_hops = hops
                            min_hops_destinations = [destination]
                        elif hops == min_hops:
                            min_hops_destinations.append(destination)
                    if min_hops_destinations:
                        destination = traffic_rng.choice(min_hops_destinations)
                    else:
                        self.logger.log(f"Time {self.env.now:.3f}: connections for {satellite} is not availiable, packet failed to generate.")
            session_duration = min(traffic_rng.expovariate(1.0 / self.mean_interval_time),self.mean_interval_time*3)

            end_time = self.env.now + session_duration
            self.satellites[satellite].is_producing = 1
            if self.mode in ["Tradition","Ground"]:
                neighbors=[]
                for _satellite in self.satellites:
                    if satellite in self.satellites[_satellite].routing_tables and destination in self.satellites[_satellite].routing_tables:
                        if (self.satellites[_satellite].routing_tables[satellite][1]+self.satellites[_satellite].routing_tables[destination][1])<=min_hops+4:
                            neighbors.append(self.satellites[_satellite].name)
                self.satellites[satellite].neighbor_graph=self.propagator.graph.subgraph(neighbors)
            while self.env.now < end_time:
                yield self.env.timeout(1.0 / self.packet_frequency)
                size = traffic_rng.randint(self.size_range[0], self.size_range[1])
                if type == 0:
                    size_after_computing = int(traffic_rng.uniform(self.size_after_computing_factor[0], self.size_after_computing_factor[1]) * size)
                    computing_demand = int(traffic_rng.uniform(self.computing_demand_factor[0], self.computing_demand_factor[1]) * size)
                else:
                    size_after_computing = self.size_after_computing_1
                    computing_demand = int(traffic_rng.uniform(self.computing_demand_factor_2[0], self.computing_demand_factor_2[1]) * size)
                if self.select_mode != 1:
                    min_hops_destinations=self.satellites[satellite].find_min_hops_destinations(5)
                    hightest_score_destinations=self.satellites[satellite].find_highest_score(min_hops_destinations,[type,size,computing_demand,size_after_computing],0,0)
                    if hightest_score_destinations:
                        destination = traffic_rng.choice(hightest_score_destinations)
                    else:
                        destination = 'False'
                if not (destination in self.satellite_names and satellite in self.satellite_names):
                    self.logger.log(f"Time {self.env.now:.3f}: {satellite} or {destination} is missed, packet failed to generate.")
                    break
                if destination in self.satellites[satellite].routing_tables:
                    packet = Packet(satellite, destination, self.env.now, size)
                    packet.extra_information([False, type, computing_demand, size_after_computing, self.env.now, None, None])
                    self.record_workload_trace(
                        satellite, destination, type, size,
                        computing_demand, size_after_computing,
                    )
                    self.statics_datas['Total'] += 1
                    self.logger.log(f"Time {self.env.now:.3f}: {satellite}: Packet generated: {(satellite, destination,packet.creation_time)}.")
                    success = self.satellites[satellite].push_forward(packet)
                    if success:
                        self.logger.log(f"Time {self.env.now:.3f}: {satellite}: Packet {(packet.source, packet.destination,packet.creation_time)} received by router. Memory remain: {self.satellites[satellite].current_memory_occupy}.",detail=True)
                    else:
                        self.statics_datas['Lost_upload'] += 1
                        record_packet_loss(self.statics_datas, 'upload', 'memory_overflow')
                        self.propagator.finalize_packet_reward(
                            packet,
                            self.propagator.reward_function.loss_reward(
                                self.env.now - packet.creation_time
                            ),
                        )
                        self.logger.log(f"Time {self.env.now:.3f}: {satellite}: Routing queue is full, discarding packet ({satellite}, {destination},packet.creation_time).")
                    if self.mode in ["Tradition","Ground"]:
                        neighbors = []
                        for _satellite in self.satellites:
                            if _satellite in self.satellites[satellite].routing_tables and destination in self.satellites[_satellite].routing_tables:
                                if (self.satellites[satellite].routing_tables[_satellite][1] + self.satellites[_satellite].routing_tables[destination][1]) <= min_hops + 4:
                                    neighbors.append(self.satellites[_satellite].name)
                        self.satellites[satellite].neighbor_graph = self.propagator.graph.subgraph(neighbors)
                else:
                    self.logger.log(f"Time {self.env.now:.3f}: {destination} is missed, packet failed to generate.")
                    break
            if satellite in self.satellites:
                self.satellites[satellite].is_producing = 0

    def _select_fixed_trace_destination(self, satellite, rng):
        min_hops = np.inf
        destinations = []
        for destination in sorted(self.direct_satellites):
            if destination in self.satellites[satellite].routing_tables:
                hops = self.satellites[satellite].routing_tables[destination][1]
            else:
                hops = np.inf
            if hops < min_hops:
                min_hops = hops
                destinations = [destination]
            elif hops == min_hops:
                destinations.append(destination)
        return rng.choice(destinations) if destinations else None

    def generate_fixed_traffic(self, satellite):
        sessions = self.fixed_workload_sessions.get(satellite, ())
        packet_interval = 1.0 / self.packet_frequency
        for session in sessions:
            session_start = float(session['session_start_time'])
            if session_start > self.env.now:
                yield self.env.timeout(session_start - self.env.now)

            task_type = int(session['task_type'])
            destination_rng = random.Random(
                f'destination:{self.scenario_seed}:{satellite}:'
                f'{int(session["session_index"])}'
            )
            fixed_destination = session.get('destination')
            destination = (
                str(fixed_destination)
                if fixed_destination is not None else None
            )
            if satellite in self.satellites:
                self.satellites[satellite].is_producing = 1
                if destination is None and self.select_mode == 1:
                    destination = self._select_fixed_trace_destination(
                        satellite, destination_rng
                    )

            for packet_data in session['packets']:
                packet_time = (
                    session_start
                    + int(packet_data['packet_index']) * packet_interval
                )
                if packet_time > self.env.now:
                    yield self.env.timeout(packet_time - self.env.now)

                self.workload_trace_count += 1
                self.statics_datas['Total'] += 1
                size = int(packet_data['packet_size'])
                computing_demand = int(packet_data['computing_demand'])
                size_after_computing = int(packet_data['size_after_computing'])
                if satellite not in self.satellites:
                    self.statics_datas['Lost_upload'] += 1
                    record_packet_loss(
                        self.statics_datas, 'upload',
                        'source_or_destination_missing',
                    )
                    packet = Packet(
                        satellite,
                        destination if destination is not None else 'False',
                        self.env.now,
                        size,
                    )
                    self.propagator.finalize_packet_reward(
                        packet,
                        self.propagator.reward_function.loss_reward(0.0),
                    )
                    continue

                if fixed_destination is None and self.select_mode != 1:
                    min_hops_destinations = self.satellites[
                        satellite
                    ].find_min_hops_destinations(5)
                    highest_score_destinations = self.satellites[
                        satellite
                    ].find_highest_score(
                        min_hops_destinations,
                        [task_type, size, computing_demand, size_after_computing],
                        0,
                        0,
                    )
                    destination = (
                        destination_rng.choice(highest_score_destinations)
                        if highest_score_destinations else None
                    )

                if (
                    destination not in self.satellite_names
                    or (not (self.mode in ('Pure_DQN', 'New_DQN', 'Pure_PPO') or getattr(self, 'aligned_graphsage_policy', False))
                        and destination not in self.satellites[satellite].routing_tables)
                ):
                    self.statics_datas['Lost_upload'] += 1
                    record_packet_loss(self.statics_datas, 'upload', 'no_route')
                    packet = Packet(
                        satellite,
                        destination if destination is not None else 'False',
                        self.env.now,
                        size,
                    )
                    self.propagator.finalize_packet_reward(
                        packet,
                        self.propagator.reward_function.loss_reward(0.0),
                    )
                    self.logger.log(
                        f"Time {self.env.now:.3f}: no destination route for "
                        f"fixed-trace packet from {satellite}."
                    )
                    continue

                packet = Packet(satellite, destination, self.env.now, size)
                packet.extra_information([
                    False,
                    task_type,
                    computing_demand,
                    size_after_computing,
                    self.env.now,
                    None,
                    None,
                ])
                success = self.satellites[satellite].push_forward(packet)
                if not success:
                    self.statics_datas['Lost_upload'] += 1
                    record_packet_loss(self.statics_datas, 'upload', 'memory_overflow')
                    self.propagator.finalize_packet_reward(
                        packet,
                        self.propagator.reward_function.loss_reward(
                            self.env.now - packet.creation_time
                        ),
                    )

            session_end = float(session['session_end_time'])
            if session_end > self.env.now:
                yield self.env.timeout(session_end - self.env.now)
            if satellite in self.satellites:
                self.satellites[satellite].is_producing = 0

    def get_system_state(self):
        total_queue_usage = {}
        total_computing_memory={}
        for node in self.satellite_names:
            average_usage = min(self.satellites[node].current_memory_occupy / self.memory,1)
            computing_memory = self.satellites[node].current_computing_queue_size / self.memory
            total_queue_usage[node] = average_usage
            total_computing_memory[node] = computing_memory
        return total_queue_usage

    def run(self, duration):
        if self.env.now==0:
            traffic_generator = (
                self.generate_fixed_traffic
                if self.fixed_workload_sessions is not None
                else self.generate_traffic
            )
            for satellite in self.satellite_names:
                self.env.process(traffic_generator(satellite))
            for satellite in self.satellites:
                self.satellites[satellite].all_start()
        end_time = self.env.now + duration
        self.env.run(until=end_time)
        # SimPy treats ``until`` as an exclusive boundary.  Only extend the
        # final run that reaches the trace endpoint; extending every one-
        # second control step would accumulate epsilon drift (e.g. 30.00000003)
        # and prevent fixed fault events at t=30, 60, ... from matching.
        if (
            self.fixed_workload_sessions is not None
            and self.fixed_workload_max_time is not None
            and end_time + 1e-9 >= self.fixed_workload_max_time
        ):
            self.env.run(until=end_time + 1e-9)
        if 'Is_computing' in self.statics_datas:
            for satellite in self.satellites:
                self.statics_datas['Is_computing']+= self.satellites[satellite].is_computing

    def upgrade_all(self,graph,landmarks,ground_cover_sats,topology_graph=None):
        topology_graph = topology_graph if topology_graph is not None else graph
        self.topology_graph = topology_graph

        same_topology = (
            set(graph.nodes()) == set(self.graph.nodes())
            and set(graph.edges()) == set(self.graph.edges())
        )
        if same_topology:
            self.landmarks = landmarks
            self.ground_cover_sats = ground_cover_sats
            self.propagator.reset_parameters()
            self.satellite_names = [node for node in graph]
            self.propagator.update(graph)
            self.direct_satellites = set(sum(landmarks.values(), []))
            for node, satellite in self.satellites.items():
                satellite.is_downlink = node in self.direct_satellites
                if node in topology_graph:
                    satellite.update_topology_slots(topology_graph.neighbors(node))
                satellite.trans_parameters(
                    self.q_net, self.reward_function, self.direct_satellites,
                    self.gnn, self.ppo_shuffle_func,
                )
            return

        self.landmarks = landmarks
        self.ground_cover_sats = ground_cover_sats
        new_nodes = set(graph.nodes())
        old_nodes = set(self.graph.nodes())
        new_edges = set(graph.edges())
        old_edges = set(self.graph.edges())
        old_direct_satellites=self.direct_satellites

        self.satellite_names = [node for node in graph]
        self.graph = graph
        for node in self.graph:
            self.graph.nodes[node]['computing_ability'] = self.computing_abilities[node]
        for left, right in self.graph.edges:
            left_satellite = self.satellites.get(left)
            right_satellite = self.satellites.get(right)
            left_load = (
                left_satellite.transmission_size.get(right, 0)
                if left_satellite is not None else 0
            )
            right_load = (
                right_satellite.transmission_size.get(left, 0)
                if right_satellite is not None else 0
            )
            self.graph[left][right]['transmission_weight'] = left_load + right_load
        self.propagator.update(graph)
        self.propagator.reset_parameters()
        self.propagator.trans_parameters(self.max_hop,self.downstream_delays,self.reward_function)

        flattened_list = set(sum(self.landmarks.values(), []))
        new_direct_satellites = flattened_list
        self.direct_satellites = flattened_list
        for node in old_direct_satellites-new_direct_satellites:
            if node in self.satellites:
                self.satellites[node].is_downlink=False
        for node in new_direct_satellites-old_direct_satellites:
            if node in self.satellites:
                self.satellites[node].is_downlink = True

        for node in sorted(new_nodes - old_nodes):
            self.satellites[node] = self.satellite_class(self.mode,self.select_mode,self.epsilon,self.max_hop,self.max_size,self.device,self.env, node, [],self.memory,self.computing_abilities[node],self.transmission_rate,self.downlink_rate, self.state_update_period,True if node in self.direct_satellites else False, self.logger, self.statics_datas, heartbeat_timeout=self.heartbeat_timeout, fault_compute_double_mean=self.fault_compute_double_mean, fault_fill_multiplier=self.fault_fill_multiplier, historical_self_feature_elimination=self.historical_self_feature_elimination, disable_self_feature_elimination=self.disable_self_feature_elimination, gnn_information_hops=self.gnn_information_hops, gnn_include_self=self.gnn_include_self, gnn_node_feature_dim=self.gnn_node_feature_dim, tradition_replan_interval=self.tradition_replan_interval, require_computed_before_downlink=self.require_computed_before_downlink, tradition_use_genetic_search=self.tradition_use_genetic_search, tradition_ga_population_size=self.tradition_ga_population_size, tradition_ga_generations=self.tradition_ga_generations, tradition_ga_path_candidates=self.tradition_ga_path_candidates, tradition_ga_mutation_rate=self.tradition_ga_mutation_rate, topology_neighbors=[])
            self.satellites[node].set_propagator(self.propagator)
            self.satellites[node].all_start()
            self.satellites[node].adjacency_table_exchanger()
            if self.fixed_workload_sessions is None:
                self.env.process(self.generate_traffic(node))

        for node in sorted(old_nodes - new_nodes):
            self.env.process(self.satellites[node].self_missing())
            del self.satellites[node]
        for edge in sorted(new_edges - old_edges):
            node, neighbor = edge
            for source, target in ((node, neighbor), (neighbor, node)):
                satellite = self.satellites[source]
                if target not in satellite.neighbors:
                    satellite.pending_state_neighbors.add(target)
                    satellite.pending_link_since[target] = self.env.now
                    satellite.pending_adjacency_records.pop(target, None)
                    if not satellite._state_probe_event.triggered:
                        satellite._state_probe_event.succeed()
        for edge in sorted(old_edges - new_edges):
            node, neighbor = edge
            for source, target in ((node, neighbor), (neighbor, node)):
                if source in self.satellites:
                    self.satellites[source].pending_state_neighbors.discard(target)
                    self.satellites[source].pending_link_since.pop(target, None)
                    self.satellites[source].pending_adjacency_records.pop(target, None)
        for node, satellite in self.satellites.items():
            if node in topology_graph:
                satellite.update_topology_slots(topology_graph.neighbors(node))
        for satellite in self.satellites:
            self.satellites[satellite].trans_parameters(self.q_net,self.reward_function,self.direct_satellites,self.gnn,self.ppo_shuffle_func)

