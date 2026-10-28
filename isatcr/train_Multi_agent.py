"""Five-seed GraphPR training in the current computing/routing simulator.

Source: grl_energy_route_revise_extend, algorithm=GraphPR (not energy gpr).
The policy learns four routing actions. Packets retain their original size and
reach the destination without any onboard computation.
Battery reward is omitted: this simulator has no battery/illumination model.
No training starts on import. Default seeds 0..4 train in parallel processes.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

import SatelliteNetworkSimulator_Computing as computing
from train_D3QN import CONFIG as COMMON_CONFIG
from training_runner import (PROJECT_ROOT, build_environment, training_input_hashes,
                             configure_reproducible_training,
                             metric_checkpoint_path,
                             save_replacing_checkpoint)
from train_4_5seed import TRAINING_SCENARIO_SEED

TRAINING_SEEDS = (0, 1, 2, 3, 4)
TRAINING_EPISODES = 1000
EPISODE_SECONDS = 1
SOURCE_ROOT = Path(r'C:\Users\Administrator\Desktop\grl_energy_route_revise_extend')


def acceleration_enabled():
    value = os.environ.get('GRAPHPR_ACCELERATION', '1')
    if value not in ('0', '1'):
        raise ValueError('GRAPHPR_ACCELERATION must be 0 or 1')
    return value == '1'


def layout(tle):
    names = Path(tle).read_text(encoding='utf-8-sig').splitlines()[::3]
    indexes = {name: tuple(map(int, name.split('_')[1:])) for name in names}
    planes = sorted({values[1] for values in indexes.values()})
    slots = sorted({values[2] for values in indexes.values()})
    identities = {}
    for name, (_, plane, slot) in indexes.items():
        identity = [0.0] * (len(planes) + len(slots))
        identity[planes.index(plane)] = 1.0
        identity[len(planes) + slots.index(slot)] = 1.0
        identities[name] = identity
    return identities, {name: 6371.0 + values[0] for name, values in indexes.items()}


def seed_config(seed):
    config = copy.deepcopy(COMMON_CONFIG)
    config['graphpr_execution'] = dict(accelerated=acceleration_enabled(),
                                      cpu_inference=acceleration_enabled())
    identities, radii = layout(PROJECT_ROOT / config['environment']['tle_filepath'])
    identity_dim = len(next(iter(identities.values())))
    config['general'].update(algorithm_name='GraphPR', random_seed=seed,
                             scenario_seed=TRAINING_SCENARIO_SEED,
                             duration=TRAINING_EPISODES * EPISODE_SECONDS,
                             training_episodes=TRAINING_EPISODES,
                             warmup_steps=0,
                             episode_interval_seconds=EPISODE_SECONDS,
                             print_cycle=EPISODE_SECONDS,
                             training_reward_save_start_step=900 * EPISODE_SECONDS,
                             python_hash_seed='0',
                             epsilon=0.9, min_epsilon=0.001,
                             epsilon_decay=math.exp(-0.002),
                             epsilon_decay_interval_seconds=0.3,
                             reward_factors=(1, 0, 1, 0, 0))
    config['environment']['require_computed_before_downlink'] = False
    config['environment']['uncomputed_reach_penalty'] = 0.0
    config['environment'].pop('target_load_tasks_per_second', None)
    config['environment'].update(
        id='graphpr_original_flows', edge_p=0.0, edge_q=1.0,
        node_p=0.0, node_q=1.0, random_edges_del=0, random_nodes_del=0,
        markov_mode=False, packet_size_range=[512000, 512000],
        mission_possibility=[1.0, 0.0], computing_demand_factor=[0, 0],
        computing_demand_factor_2=[0, 0], size_after_computing_factor=[1.0, 1.0],
        size_after_computing_1=512000, poisson_rate_s=0.0, poisson_rate_g=0.0,
        packet_frequency=6000000 / 512000, mean_interval_time=60,
        downstream_delays=0.0,
    )
    for key in ('workload_trace_path', 'workload_trace_sha256', 'workload_trace_task_count',
                'fault_trace_path', 'fault_trace_sha256', 'fault_trace_event_count'):
        config['environment'].pop(key, None)
    config['graphpr_training_traffic'] = dict(
        mode='original_satellite_flows', num_flows=50, flow_rate_bits_per_second=6000000,
        packet_size_bits=64000 * 8, refresh_seconds=60,
        destination_selection='random_distinct_satellites',
    )
    config['agent'] = dict(mode='Pure_DQN', action_dim=4, hidden_dim=256,
                           hidden_layers=3, learning_rate=0.001, gamma=0.9,
                           buffer_length=50000, batch_size=4096, shuffle=False,
                           dueling=False, activation='ReLU', priority_alpha=0.6,
                           priority_epsilon=0.01, learn_interval_seconds=0.05,
                           target_interval_seconds=60.0,
                           model_path=f'./training_process_data/v1/model/Multi_agent/GraphPR_seed{seed}.pth')
    config['environment']['save_training_data'] = f'v1/model/Multi_agent/train_GraphPR_seed{seed}.txt'
    config['graphpr'] = dict(identity_dim=identity_dim, identities=identities,
                             radii=radii, node_feature_dim=identity_dim + 11,
                             packet_feature_dim=identity_dim + 7,
                             policy_action_dim=4, environment_action_dim=4,
                             local_compute_before_routing=False,
                             parameter_sharing=True, heads=3, dropout=0.01,
                             aggregation='local_raw_plus_previous_neighbor_hidden',
                             queue_capacity_packets=15, wp=0.4, wq=0.4,
                             battery_reward='absent_in_current_environment',
                             terminal_reward='success=1,loss=-1',
                             rsph_weights={'1': 0.01, '0': 0.3, '-1': 0.5, '-2': 0.7},
                             hidden_reset_probability=0.33,
                             route_cache_seconds=0.3,
                             adaptations=['routing_only_original_packet_size', 'current_legality',
                                          'no_battery_model', 'instant_queue_samples',
                                          'current_link_rate_and_propagation_normalization'])
    return config


class GraphPRPolicy(nn.Module):
    """Source single GAT, three heads averaged, ReLU MLP, ordinary Q output."""
    def __init__(self, identity_dim, hidden_dim=256, hidden_layers=3):
        super().__init__()
        self.model_config = dict(identity_dim=identity_dim, hidden_dim=hidden_dim,
                                 hidden_layers=hidden_layers)
        self.feature_dim = identity_dim + 11
        self.packet_dim = identity_dim + 7
        # Twelve reward scalars and one environment-only computed flag.
        self.state_dim = 5 * self.feature_dim + self.packet_dim + 13
        self.projection = nn.Linear(self.feature_dim, 3 * self.feature_dim, bias=False)
        self.attention_source = nn.Parameter(torch.empty(3, self.feature_dim))
        self.attention_target = nn.Parameter(torch.empty(3, self.feature_dim))
        self.gat_bias = nn.Parameter(torch.zeros(3, self.feature_dim))
        nn.init.xavier_normal_(self.projection.weight, gain=math.sqrt(2))
        nn.init.xavier_normal_(self.attention_source, gain=math.sqrt(2))
        nn.init.xavier_normal_(self.attention_target, gain=math.sqrt(2))
        self.dropout = nn.Dropout(0.01)
        self.fc_input = nn.Linear(self.feature_dim + self.packet_dim, hidden_dim)
        self.fc_hiddens = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim)
                                      for _ in range(hidden_layers))
        self.fc_output = nn.Linear(hidden_dim, 4)

    def aggregate(self, graph):
        values = self.projection(self.dropout(graph)).reshape(-1, 5, 3, self.feature_dim)
        scores = torch.nn.functional.leaky_relu(
            (values * self.attention_source).sum(-1)
            + (values[:, :1] * self.attention_target).sum(-1), 0.2)
        weights = self.dropout(scores.softmax(dim=1))
        return ((weights.unsqueeze(-1) * values).sum(1) + self.gat_bias).mean(1)

    def forward(self, state, valid_mask=None):
        if state.shape[-1] != self.state_dim:
            raise ValueError(f'GraphPR expects {self.state_dim} features, got {state.shape[-1]}')
        graph_end = 5 * self.feature_dim
        hidden = self.aggregate(state[:, :graph_end].reshape(-1, 5, self.feature_dim))
        # The computed flag stays in the physical interface, outside the policy.
        packet = state[:, graph_end:graph_end + self.packet_dim]
        x = torch.relu(self.fc_input(torch.cat((hidden, packet), dim=1)))
        for layer in self.fc_hiddens:
            x = torch.relu(layer(x))
        return self.fc_output(x)

    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = cls(**checkpoint['model_config']).to(device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.eval()
        return model


class PriorityReplay:
    """Source proportional PER, stratified sampling, no importance weights."""
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.accelerated = acceleration_enabled()
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write = self.size = 0
        self.priority_max = 0.1

    def set_priority(self, leaf, error):
        priority = (abs(float(error)) + 0.01) ** 0.6
        change = priority - self.tree[leaf]
        self.tree[leaf] = priority
        while leaf:
            leaf = (leaf - 1) // 2
            self.tree[leaf] += change

    def push(self, experience):
        self.data[self.write] = experience
        self.set_priority(self.write + self.capacity - 1, self.priority_max)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, count):
        if self.accelerated:
            segment = self.tree[0] / count
            # Preserve Python RNG calls, stratification and <= boundary rules.
            values = np.array([random.uniform(i * segment, (i + 1) * segment)
                               for i in range(count)])
            leaves = np.zeros(count, dtype=np.int64)
            active = np.flatnonzero(2 * leaves + 1 < len(self.tree))
            while active.size:
                left = 2 * leaves[active] + 1
                go_right = values[active] > self.tree[left]
                values[active] -= np.where(go_right, self.tree[left], 0.0)
                leaves[active] = left + go_right
                active = active[2 * leaves[active] + 1 < len(self.tree)]
            return leaves.tolist(), [self.data[i] for i in leaves - self.capacity + 1]
        leaves, samples = [], []
        segment = self.tree[0] / count
        for index in range(count):
            value = random.uniform(index * segment, (index + 1) * segment)
            leaf = 0
            while 2 * leaf + 1 < len(self.tree):
                left = 2 * leaf + 1
                if value <= self.tree[left]:
                    leaf = left
                else:
                    value -= self.tree[left]
                    leaf = left + 1
            leaves.append(leaf)
            samples.append(self.data[leaf - self.capacity + 1])
        return leaves, samples


    def update_priorities(self, leaves, errors):
        if not self.accelerated:
            for leaf, error in zip(leaves, errors):
                self.set_priority(leaf, error)
            return
        # Repeated leaves must see the immediately preceding update. Parents
        # receive additions in sample order, preserving floating-point sums.
        priorities = np.array([(abs(float(error)) + 0.01) ** 0.6 for error in errors])
        changes = np.empty(len(leaves), dtype=np.float64)
        previous = {}
        for i, (leaf, priority) in enumerate(zip(leaves, priorities)):
            changes[i] = priority - previous.get(leaf, self.tree[leaf])
            previous[leaf] = priority
        self.tree[leaves] = priorities
        parents = (np.asarray(leaves, dtype=np.int64) - 1) // 2
        paths = []
        while np.any(parents >= 0):
            paths.append(parents)
            parents = np.where(parents > 0, (parents - 1) // 2, -1)
        if not paths:
            return
        paths = np.stack(paths, axis=1)
        valid = paths >= 0
        np.add.at(self.tree, paths[valid], np.broadcast_to(changes[:, None], paths.shape)[valid])


class GraphPRAgent:
    def __init__(self, config, device):
        self.config, self.device = config, device
        self.epsilon = config['general']['epsilon']
        self.online_net = GraphPRPolicy(config['graphpr']['identity_dim']).to(device)
        self.target_net = copy.deepcopy(self.online_net).eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=0.001)
        self.memory = PriorityReplay()
        capacity, width = self.memory.capacity, self.online_net.state_dim
        self._replay_columns = (
            np.empty((capacity, width), np.float32), np.empty(capacity, np.int64),
            np.empty(capacity, np.float32), np.empty((capacity, width), np.float32),
            np.empty(capacity, bool), np.empty((capacity, 4), bool),
        ) if self.memory.accelerated else None
        self.updates = self.target_updates = 0
        self.online_net.eval()
        # SimPy decisions use small tensors; synchronize immediately after
        # every optimizer step, including updates inside env.step().
        self.inference_net = (copy.deepcopy(self.online_net).cpu().eval()
                              if acceleration_enabled() and device.type == 'cuda'
                              else self.online_net)

    def observe(self, experience):
        # Copy arrays; replay must not alias mutable packet/neighbor state.
        state, _, action, reward, next_state, done = experience[:6]
        if not 0 <= int(action) < 4:
            raise ValueError('GraphPR replay accepts only four routing actions')
        next_mask = experience[7] if len(experience) >= 8 else (
            experience[6] if len(experience) == 7 else None)
        stored = (np.asarray(state, dtype=np.float32).copy(), int(action),
                  float(reward), (np.zeros_like(state, dtype=np.float32) if next_state is None
                                  else np.asarray(next_state, dtype=np.float32).copy()),
                  bool(done), None if next_mask is None else np.array(next_mask[:4], copy=True))
        if self._replay_columns is not None:
            for column, value in zip(self._replay_columns, stored):
                column[self.memory.write] = np.ones(4, bool) if value is None else value
        self.memory.push(stored)

    def learn(self):
        # Source checks 2048 although it samples 4096: require a full batch.
        if self.memory.size < 4096:
            return
        leaves, batch = self.memory.sample(4096)
        if self._replay_columns is not None:
            indices = np.asarray(leaves, dtype=np.int64) - self.memory.capacity + 1
            states, actions, rewards, next_states, done, masks = [
                torch.as_tensor(column[indices], device=self.device)
                for column in self._replay_columns]
            actions = actions.unsqueeze(1)
        else:
            states, actions, rewards, next_states, done, masks = zip(*batch)
            states = torch.as_tensor(np.stack(states), device=self.device)
            next_states = torch.as_tensor(np.stack(next_states), device=self.device)
            actions = torch.tensor(actions, device=self.device).unsqueeze(1)
            rewards = torch.tensor(rewards, device=self.device)
            done = torch.tensor(done, dtype=torch.bool, device=self.device)
            masks = torch.tensor(np.stack([np.ones(4, dtype=bool) if mask is None else mask
                                           for mask in masks]), device=self.device)
        self.online_net.train()
        try:
            prediction = self.online_net(states).gather(1, actions).squeeze(1)
            with torch.no_grad():
                # Source DQN target max (not Double-DQN); enforce local legality.
                values = self.target_net(next_states).masked_fill(~masks, -torch.inf).max(1).values
                values = torch.where(done | ~masks.any(1), 0.0, values)
                target = rewards + 0.9 * values
            loss = torch.nn.functional.smooth_l1_loss(prediction, target)
            errors = (prediction - target).detach().cpu().tolist()
            self.memory.priority_max = max(self.memory.priority_max, max(map(abs, errors)))
            self.memory.update_priorities(leaves, errors)
            self.optimizer.zero_grad()
            loss.backward()
            for parameter in self.online_net.parameters():
                if parameter.grad is not None:
                    parameter.grad.clamp_(-1, 1)
            self.optimizer.step()
            if self.inference_net is not self.online_net:
                self.inference_net.load_state_dict(self.online_net.state_dict())
            self.updates += 1
        finally:
            self.online_net.eval()

    def exploration_scheduler(self, simulator):
        general = self.config['general']
        while True:
            yield simulator.env.timeout(general['epsilon_decay_interval_seconds'])
            self.epsilon = max(general['min_epsilon'], self.epsilon * general['epsilon_decay'])
            simulator.epsilon = self.epsilon
            for satellite in simulator.satellites.values():
                satellite.epsilon = self.epsilon

    def scheduler(self, simulator):
        elapsed, next_target = 0, 1200
        while True:
            yield simulator.env.timeout(0.05)
            self.learn()
            elapsed += 1
            if elapsed == next_target:
                self.target_net.load_state_dict(self.online_net.state_dict())
                self.target_updates += 1
                next_target += 1200
                if random.random() < 0.33:
                    for satellite in simulator.satellites.values():
                        satellite.graphpr_hidden = satellite.raw_feature()
                        satellite.graphpr_reset_round = satellite.graphpr_round

    def save_model(self, path):
        torch.save(dict(state_dict=self.online_net.state_dict(),
                        model_config=self.online_net.model_config), path)


@contextmanager
def graphpr_environment(config):
    """Only this run uses GraphPR features/rewards; common physical code is intact."""
    options = config['graphpr']
    offloading = config.get('test_contract', {}).get('compute_rule') in ('local', 'least_finish')
    if offloading and config['general']['phase'] != 'test':
        raise ValueError('Heuristic offloading is evaluation-only')
    identities, radii = options['identities'], options['radii']
    feature_dim = options['node_feature_dim']
    identity_dim = options['identity_dim']
    original_satellite = computing.Satellite_with_Computing
    original_propagator = computing.Propagator_Computing
    original_reward = computing.Reward_Function
    import RL_environment_for_computing as environment_module
    original_simulator = environment_module.SatelliteNetworkSimulator_OnbardComputing
    traffic = (config.get('graphpr_training_traffic')
               if config['general']['phase'] == 'train' else None)

    class GraphPRFlowSimulator(original_simulator):
        def generate_traffic(self, satellite):
            # The shared simulator starts one generator per node; only one
            # coordinator creates the original fixed-rate source/destination flows.
            if satellite != min(self.satellite_names):
                return
            while True:
                eligible = sorted(name for name, node in self.satellites.items()
                                  if node.active and self.graph.degree(name) >= 3)
                extra = sorted(name for name in eligible
                               if self.graph.degree(name) == 4)
                count = traffic['num_flows']
                pairs = []
                for pool, amount in ((eligible, min(count, len(eligible))),
                                     (extra, max(0, count - len(eligible)))):
                    if amount > len(pool):
                        raise ValueError('Not enough eligible satellites for original GraphPR flows')
                    if not amount:
                        continue
                    while True:
                        sources = np.random.choice(pool, amount, replace=False).tolist()
                        targets = np.random.choice(pool, amount, replace=False).tolist()
                        if all(left != right for left, right in zip(sources, targets)):
                            break
                    pairs.extend(zip(sources, targets))
                if not pairs:
                    raise ValueError('No eligible satellites for original GraphPR flows')
                end = self.env.now + traffic['refresh_seconds']
                for source, target in pairs:
                    self.env.process(self._graphpr_flow(source, target, end))
                yield self.env.timeout(traffic['refresh_seconds'])

        def _graphpr_flow(self, source, target, end):
            size = traffic['packet_size_bits']
            interval = size / traffic['flow_rate_bits_per_second']
            while self.env.now < end:
                yield self.env.timeout(interval)
                packet = computing.Packet(source, target, self.env.now, size)
                packet.extra_information([False, 0, 0, size, self.env.now, None, None])
                self.statics_datas['Total'] += 1
                self.record_workload_trace(source, target, 0, size, 0, size)
                if not self.satellites[source].push_forward(packet):
                    self.statics_datas['Lost_upload'] += 1
                    computing.record_packet_loss(self.statics_datas, 'upload', 'memory_overflow')
                    self.propagator.finalize_packet_reward(
                        packet, self.propagator.reward_function.loss_reward(0.0))

    class GraphPRReward(original_reward):
        def reach_reward_abnormal(self, delay):
            # Pure routing succeeds without a computed flag.
            return self.reach_reward(delay)

    class GraphPRSatellite(original_satellite):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.require_computed_before_downlink = offloading
            self.graphpr_routes = {}

        def push_offload(self, packet):
            if traffic is None:
                return super().push_offload(packet)
            # A routing-only flow succeeds at its named destination satellite;
            # no ground destination selection, downlink queue or computation.
            assert packet.destination == self.name
            self._release_transmission_memory(packet)
            self.env.process(self.propagator.downstream(self.name, packet))

        def valid_action_mask(self, destination, packet_size, is_computed, hops,
                              wait_count=0, previous_hop=None,
                              postcompute_previous_hop=None):
            # Reuse routing legality without consulting CPU availability.
            mask = super().valid_action_mask(destination, packet_size,
                                              is_computed if offloading else True,
                                              hops, wait_count, previous_hop,
                                              postcompute_previous_hop)
            if not offloading:
                mask[4] = False
            return mask

        def push_computing(self, packet, computing_demand):
            if offloading:
                return super().push_computing(packet, computing_demand)
            raise RuntimeError('GraphPR routing-only packets must never enter computation')

        @torch.no_grad()
        def get_next_hop(self, current_state, destination, packet_size=None,
                         forced_action=None, prepared_policy_state=None,
                         prepared_action_mapping=None, *, hops, wait_count=0,
                         previous_hop=None, postcompute_previous_hop=None):
            mask = self.valid_action_mask(destination, packet_size, bool(current_state[-1]),
                                          hops, wait_count, previous_hop,
                                          postcompute_previous_hop)
            if forced_action is not None:
                return int(forced_action) if 0 <= int(forced_action) < len(mask) and mask[int(forced_action)] else None
            candidates = np.flatnonzero(mask[:4])
            if not len(candidates):
                return None
            cached = self.graphpr_routes.get(destination)
            if cached is not None:
                action, neighbor, expires = cached
                if (self.env.now < expires and action in candidates
                        and self.slots()[action] == neighbor):
                    return action
                del self.graphpr_routes[destination]
            # Source GraphPR uses uniform legal-action epsilon exploration.
            if np.random.rand() < self.epsilon:
                action = int(np.random.choice(candidates))
            else:
                state = torch.as_tensor(current_state, dtype=torch.float32,
                                        device=self.device).unsqueeze(0)
                values = self.q_net(state, mask)[0, candidates]
                action = int(candidates[values.argmax().item()])
            self.graphpr_routes[destination] = (
                action, self.slots()[action], self.env.now + options['route_cache_seconds'])
            return action

        def slots(self):
            return list(getattr(self, 'action_neighbors', self.neighbors))[:4]

        def queue_idle(self):
            return [max(0.0, 1 - self.transmission_length.get(neighbor, 0) / 15)
                    if neighbor in self.neighbors else 0.0
                    for neighbor in self.slots()] + [0.0] * (4 - len(self.slots()))

        def raw_feature(self):
            position = np.asarray(self.propagator.node_positions[self.name]) / radii[self.name]
            rates = [1.0 if neighbor in self.neighbors else 0.0 for neighbor in self.slots()]
            return np.asarray(identities[self.name] + position.tolist()
                              + self.queue_idle() + rates + [0.0] * (4 - len(rates)), dtype=np.float32)

        def trans_parameters(self, *args, **kwargs):
            super().trans_parameters(*args, **kwargs)
            # Topology refreshes rebind parameters every second. Keep one
            # exchange process and its hidden-state history per satellite.
            if hasattr(self, 'graphpr_exchange_process'):
                return
            self.graphpr_hidden = np.zeros(feature_dim, dtype=np.float32)
            self.graphpr_graph = None
            self.graphpr_round = 0
            self.graphpr_reset_round = 0
            self.graphpr_exchange_process = self.env.process(self.exchange())

        def graph_input(self):
            local = self.raw_feature()
            neighbors = []
            for neighbor in self.slots():
                if neighbor not in self.neighbors:
                    neighbors.append(np.zeros(feature_dim))
                    continue
                version, hidden = self.gnn_layer1_messages.get(neighbor, (None, None))
                if self.graphpr_round == self.graphpr_reset_round:
                    # Simultaneous raw initialization, as in the source.
                    neighbors.append(self.propagator.satellites[neighbor].raw_feature())
                elif version == self.graphpr_round - 1:
                    neighbors.append(hidden)
                else:
                    neighbors.append(np.zeros(feature_dim))
            neighbors += [np.zeros(feature_dim)] * (4 - len(neighbors))
            return np.asarray([local] + neighbors, dtype=np.float32)

        def exchange(self):
            while self.active:
                with torch.no_grad():
                    self.graphpr_graph = self.graph_input()
                    graph = torch.as_tensor(self.graphpr_graph[None], device=self.device)
                    self.graphpr_hidden = self.q_net.aggregate(graph)[0].cpu().numpy()
                for neighbor in self.neighbors:
                    self.env.process(self.propagator.send_gnn_embedding_message(
                        self.name, neighbor, 'layer1', self.graphpr_round,
                        self.graphpr_hidden.tolist()))
                self.graphpr_round += 1
                yield self.env.timeout(self.state_update_period)

        def get_current_state(self, destination, hops, is_computed, mission_state):
            destination_position = self.propagator.node_positions.get(destination)
            destination_identity = identities.get(destination, [0.0] * identity_dim)
            position = (destination_position / radii[destination]).tolist() if destination_position is not None else [0.0] * 3
            packet = destination_identity + position + self.queue_idle()
            # Reward-only metadata; excluded from all network inputs.
            current_hops = (0 if destination == self.name else
                            self.routing_tables.get(destination, (None, float('inf')))[1])
            deltas, propagation, rates = [], [], []
            max_delay = max(self.propagator.propagation_delays.values(), default=1e-9)
            for neighbor in self.slots():
                next_hops = self.neighbor_hops.get(neighbor, {}).get(destination, float('inf'))
                delta = current_hops - next_hops
                deltas.append(delta if math.isfinite(delta) else -3.0)
                propagation.append(self.propagator.propagation_delays.get((self.name, neighbor), 0.0) / max_delay)
                rates.append(1.0 if neighbor in self.neighbors else 0.0)
            for values in (deltas, propagation, rates):
                values.extend([0.0] * (4 - len(values)))
            # Replay and decisions use exactly the graph that produced h_t;
            # aggregating freshly received h_t again would add an extra round.
            graph = self.graphpr_graph if self.graphpr_graph is not None else self.graph_input()
            return np.concatenate((graph.ravel(), packet, deltas,
                                   propagation, rates, [float(is_computed)])).astype(np.float32)

    class GraphPRPropagator(original_propagator):
        def record_experience(self, experience, *args, **kwargs):
            if offloading:
                return super().record_experience(experience, *args, **kwargs)
            experience = list(experience)
            state, _, action, _, _, done = experience[:6]
            if not 0 <= int(action) < 4:
                raise ValueError('GraphPR experience contains a non-routing action')
            if not done:
                state = np.asarray(state)
                graph_end = 5 * feature_dim
                packet_end = graph_end + identity_dim + 7
                queue = max(0.0, 1 - state[graph_end + identity_dim + 3 + int(action)])
                delta = state[packet_end + int(action)]
                weight = {1: 0.01, 0: 0.3, -1: 0.5, -2: 0.7}.get(delta, 1.0)
                delay = state[packet_end + 4 + int(action)]
                rate = state[packet_end + 8 + int(action)]
                reward = -0.4 * (queue ** rate if queue else 0.0)
                reward -= 0.4 * weight * (delay if weight <= 0.5 else 1.0)
                experience[3] = reward
            return super().record_experience(experience, *args, **kwargs)

    computing.Satellite_with_Computing = GraphPRSatellite
    computing.Propagator_Computing = GraphPRPropagator
    computing.Reward_Function = GraphPRReward
    if traffic is not None:
        environment_module.SatelliteNetworkSimulator_OnbardComputing = GraphPRFlowSimulator
    try:
        yield
    finally:
        computing.Satellite_with_Computing = original_satellite
        computing.Propagator_Computing = original_propagator
        computing.Reward_Function = original_reward
        environment_module.SatelliteNetworkSimulator_OnbardComputing = original_simulator


def run_seed(seed):
    configure_reproducible_training()
    os.chdir(PROJECT_ROOT)
    config = seed_config(seed)
    output = PROJECT_ROOT / 'training_process_data/v1/model/Multi_agent'
    output.mkdir(parents=True, exist_ok=True)
    result = output / f'process_result_seed{seed}.json'
    lock = output / f'TRAINING_seed{seed}.lock'
    if result.exists() or list(output.glob(f'GraphPR_seed{seed}*.pth')):
        raise FileExistsError(f'Seed {seed} already has training outputs; archive them explicitly before restarting')
    with lock.open('x', encoding='utf-8') as file:
        json.dump(dict(pid=os.getpid(), seed=seed, started=datetime.now(timezone.utc).isoformat()), file)
    process_result = dict(seed=seed, pid=os.getpid(), exit_code=None, completed_steps=0)
    try:
        config['training_input_sha256'] = training_input_hashes(config)
        config['method_sha256'] = {Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        config['graphpr']['source_sha256'] = {
            name: hashlib.sha256((SOURCE_ROOT / name).read_bytes()).hexdigest()
            for name in ('main.py', 'algorithm/agents.py', 'MyEnv/satellite_env.py')}
        (output / f'training_config_seed{seed}.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        agent = GraphPRAgent(config, device)
        inference_device = next(agent.inference_net.parameters()).device
        with graphpr_environment(config):
            env = build_environment(config, SimpleNamespace(online_net=agent.inference_net),
                                    inference_device, None, experience_callback=agent.observe)
            env.reset(config['general']['begin_time'])
            env.simulator.env.process(agent.scheduler(env.simulator))
            env.simulator.env.process(agent.exploration_scheduler(env.simulator))
            best_reward, best_path, metrics = -math.inf, None, None
            for step in range(1, config['general']['duration'] + 1):
                env.step(agent.epsilon)
                metrics = env.last_interval_metrics
                if step % EPISODE_SECONDS == 0:
                    env.print_and_save(f'GraphPR episode {step // EPISODE_SECONDS}/{TRAINING_EPISODES}; optimizer_updates={agent.updates}; target_syncs={agent.target_updates}')
                process_result['completed_steps'] = step
                if (metrics is not None and step % EPISODE_SECONDS == 0
                        and step >= config['general']['training_reward_save_start_step']):
                    reward = metrics.get('average_reward')
                    if reward is not None and math.isfinite(reward) and reward > best_reward:
                        path = metric_checkpoint_path(config, 'train_reward_best', metrics)
                        best_path = save_replacing_checkpoint(agent, path, best_path, training_config=config)
                        best_reward = reward
            if best_path is None or metrics is None:
                raise RuntimeError('No valid reward-best checkpoint during the final training window')
            latest = metric_checkpoint_path(config, 'latest', metrics)
            save_replacing_checkpoint(agent, latest, training_config=config)
            process_result.update(exit_code=0, reward_best=best_path, latest=latest,
                                  completed_episodes=TRAINING_EPISODES,
                                  optimizer_updates=agent.updates)
    except BaseException as error:
        process_result.update(exit_code=1, error=repr(error))
        raise
    finally:
        result.write_text(json.dumps(process_result, indent=2), encoding='utf-8')
        lock.unlink()


def run_parallel_seeds():
    environment = os.environ.copy()
    environment['PYTHONHASHSEED'] = '0'
    processes = [
        (seed, subprocess.Popen(
            [sys.executable, '-u', str(Path(__file__).resolve()), '--seed', str(seed)],
            cwd=PROJECT_ROOT, env=environment,
        ))
        for seed in TRAINING_SEEDS
    ]
    failures = []
    for seed, process in processes:
        exit_code = process.wait()
        print(f'GraphPR seed {seed}: exit_code={exit_code}', flush=True)
        if exit_code:
            failures.append((seed, exit_code))
    if failures:
        raise RuntimeError(f'GraphPR training workers failed: {failures}')


def run_pipeline():
    MODEL_ROOT = PROJECT_ROOT / 'training_process_data/v1/model/Multi_agent'
    TEST_ROOT = PROJECT_ROOT / 'training_process_data/v1/test/test/Multi_agent'
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    status_path = MODEL_ROOT / 'PIPELINE_STATUS.json'
    lock = MODEL_ROOT / 'PIPELINE_RUNNING.lock'
    with lock.open('x', encoding='utf-8') as file:
        json.dump({'pid': os.getpid()}, file)
    status = {'pid': os.getpid(), 'state': 'training', 'training_workers': 5,
              'test_workers': 5, 'seeds': list(range(5)), 'expected_test_cases': 70,
              'started': datetime.now(timezone.utc).isoformat()}

    def save():
        status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')

    def run_stage(script, arguments, prefix):
        environment = os.environ.copy()
        environment['PYTHONHASHSEED'] = '2025' if script == 'test_Multi_agent.py' else '0'
        with (MODEL_ROOT / f'{prefix}.stdout.log').open('w', encoding='utf-8') as out, \
             (MODEL_ROOT / f'{prefix}.stderr.log').open('w', encoding='utf-8') as err:
            process = subprocess.Popen([sys.executable, '-B', '-u', str(PROJECT_ROOT / script), *arguments],
                                       cwd=PROJECT_ROOT, env=environment, stdout=out, stderr=err)
            status['stage_pid'] = process.pid
            save()
            code = process.wait()
        status[f'{prefix}_exit_code'] = code
        if code:
            raise RuntimeError(f'{prefix} exited with code {code}; see {prefix}.stderr.log')

    try:
        save()
        run_stage('train_Multi_agent.py', [], 'training')
        from training_runner import load_checkpoint_training_config
        for seed in range(5):
            result = json.loads((MODEL_ROOT / f'process_result_seed{seed}.json').read_text())
            if (result['exit_code'] != 0
                    or result['completed_steps'] != TRAINING_EPISODES * EPISODE_SECONDS
                    or result.get('completed_episodes') != TRAINING_EPISODES):
                raise RuntimeError(f'Seed {seed} did not complete training')
            if (MODEL_ROOT / f'TRAINING_seed{seed}.lock').exists():
                raise RuntimeError(f'Seed {seed} still has a training lock')
            for key in ('reward_best', 'latest'):
                load_checkpoint_training_config(result[key])
        status.update(state='testing', completed_training_seeds=5)
        save()
        run_stage('test_Multi_agent.py', ['--workers', '5'], 'testing')
        test_status = json.loads((TEST_ROOT / 'TEST_STATUS.json').read_text())
        if test_status['state'] != 'completed' or test_status['completed_cases'] != 70:
            raise RuntimeError('Test completion check failed')
        status.update(state='completed', completed_test_cases=70)
    except Exception as error:
        status.update(state='failed', error=str(error))
        raise
    finally:
        status['updated'] = datetime.now(timezone.utc).isoformat()
        save()
        lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--seed', type=int, choices=TRAINING_SEEDS)
    selection.add_argument('--pipeline', action='store_true',
                           help='Train five seeds, validate models, then run five-worker tests')
    parser.add_argument('--dry-run', action='store_true', help='Print contracts without creating outputs or running simulation')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    seeds = (args.seed,) if args.seed is not None else TRAINING_SEEDS
    if args.dry_run:
        for seed in seeds:
            config = seed_config(seed)
            print(json.dumps(config, indent=2))
    elif args.pipeline:
        run_pipeline()
    elif args.seed is not None:
        if os.environ.get('PYTHONHASHSEED') != '0':
            environment = os.environ.copy()
            environment['PYTHONHASHSEED'] = '0'
            raise SystemExit(subprocess.call(
                [sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:]],
                cwd=PROJECT_ROOT, env=environment,
            ))
        run_seed(args.seed)
    else:
        run_parallel_seeds()
