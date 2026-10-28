"""Check normalized resources separately from distance tokens and GNN embeddings."""
import unittest
import os
import ast
from contextlib import contextmanager
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import numpy as np
import networkx as nx
import simpy

import SatelliteNetworkSimulator_Computing as simulation
from ablation.aggregation_models import make_encoder_factory
from ablation.test_graphsage import simulator
from ablation.train_graphsage import GraphSAGEEncoder
from train_isacr_2_4_5seed import isacr_hop_environment


def resource(rng):
    ability, compute, transmit = rng.random(3)
    return [1, rng.random(), ability, compute, transmit,
            np.sqrt(ability) / (1 + 15 * compute)]


def fault_product(node, layer=0, exclude=()):
    states = [state for name, state in node.neighbor_states.items()
              if name not in exclude and node.neighbor_has_resource_state(name)]
    if not states or not node.fault_compute_double_mean:
        return 0.0
    scales = node.resource_scales if hasattr(node, 'resource_scales') else (1, 4, 12)
    scale = scales[layer] if node.mode == 'New_DQN' else 1
    offset = layer * 6
    score = sum(state[offset + 5] / scale for state in states) / len(states)
    return min(max(score / node.fault_fill_multiplier, 0), 1)


def node_fixture(mode, count, factor, rng):
    sim, names = simulator(make_encoder_factory('gat')(5, 15, 2, 15))
    node = sim.satellites[names[0]]
    node.env = simpy.Environment()
    node.mode = mode
    node.neighbors = [f'n{i}' for i in range(count)]
    node.action_neighbors = [f'n{i}' for i in range(4)]
    node.transmission_queue = {name: simpy.Store(node.env) for name in node.neighbors}
    for queue in node.transmission_queue.values():
        queue.put(SimpleNamespace(size=float(rng.uniform(0, 8e9))))
    node.transmission_size = dict.fromkeys(node.neighbors, 0)
    node.transmission_length = dict.fromkeys(node.neighbors, 0)
    node.active_transmissions = {name: (float(rng.uniform(0, 0.5)),)
                                 for name in node.neighbors}
    node.last_heartbeat = dict.fromkeys(node.neighbors, 0)
    node.neighbor_state_versions = dict.fromkeys(node.neighbors, 1)
    node.neighbor_hops = {name: {'target': 1} for name in node.neighbors}
    node.max_hop = 10
    node.fault_fill_multiplier = factor
    node.fault_compute_double_mean = True
    node.historical_self_feature_elimination = True
    node.disable_self_feature_elimination = False
    node.pending_state_neighbors = set()
    node._state_probe_event = node.env.event()
    node.computing_ability = 100e9
    node.computing_remain = float(rng.uniform(0, 100)) * node.computing_ability
    node.current_memory_occupy = node.memory * rng.random()
    node.current_state = [0, 1, 0, 0, 0, 0] * 3
    node.neighbor_states = {}
    for name in node.neighbors:
        raw = [value * scale for scale in (1, 4, 12) for value in resource(rng)]
        node.neighbor_states[name] = raw if mode == 'New_DQN' else raw[:6] if mode in ('Pure_DQN', 'Pure_PPO') else raw[:5]
    return node


class FeatureRangeTests(unittest.TestCase):
    def test_squared_capacity_is_clipped_before_squaring(self):
        node = simulation.Satellite_with_Computing.__new__(simulation.Satellite_with_Computing)
        for ability, expected in ((-1e9, 0), (0, 0), (50e9, .25),
                                  (75e9, .5625), (100e9, 1), (150e9, 1)):
            node.computing_ability = ability
            self.assertEqual(node._normalized_compute_ability(), expected)

    def test_passive_wait_five_retries_and_recovery_for_four_algorithms(self):
        from collections import deque
        for mode in ('New_DQN', 'Pure_DQN', 'Pure_PPO'):
            for recover in (False, True):
                with self.subTest(mode=mode, recover=recover):
                    node = simulation.Satellite_with_Computing.__new__(simulation.Satellite_with_Computing)
                    node.env = simpy.Environment()
                    node.mode = mode
                    node.state_update_period = .1
                    node.waiting_forward = deque()
                    node.forward_queue = simpy.Store(node.env)
                    node.name = 'source'
                    packet = simulation.Packet('source', 'target', 0, 1)
                    packet.extra_information([True, 0, 0, 1, 0, None, None])
                    packet.wait_count = 0
                    packet.wait_retry_cycle = 1
                    simulation.mark_queue_entry(packet, 'waiting_forward', 0)
                    node.waiting_forward.append(packet)
                    dropped = []
                    node._drop_no_route = lambda *args, **kw: dropped.append(kw['reason'])
                    node.valid_action_mask = lambda *args, **kw: np.array([
                        recover and packet.wait_count == 5, False, False, False, False])
                    for retry in range(1, 5):
                        node._retry_waiting_forward(retry_cycle=retry)
                        self.assertEqual(packet.wait_count, retry)
                        self.assertEqual(list(node.waiting_forward), [packet])
                        self.assertEqual(dropped, [])
                    node._retry_waiting_forward(retry_cycle=5)
                    self.assertEqual(packet.wait_count, 5)
                    self.assertEqual(len(node.waiting_forward), 0)
                    self.assertEqual(dropped, [] if recover else ['wait_limit'])
                    self.assertEqual(node.forward_queue.items, [packet] if recover else [])

    def test_compute_score_retains_capacity_wait_pairing_after_aggregation(self):
        messages = []
        expected_scores = []
        for pairs in (((1., .5), (.5, 0.)), ((1., 0.), (.5, .5))):
            node = node_fixture('New_DQN', 3, 1.125, np.random.default_rng(12))
            for name, (ability, wait) in zip(('n1', 'n2'), pairs):
                raw = [1, .8, ability, wait, .3, np.sqrt(ability) / (1 + 15 * wait)]
                node.neighbor_states[name] = [value * scale for scale in (1, 4, 12) for value in raw]
            captured = {}

            def send(source, target, state):
                captured[target] = list(state)
                yield node.env.timeout(0)

            node.propagator = SimpleNamespace(send_state=send)
            node.env.process(node.state_exchanger())
            node.env.run(until=.101)
            messages.append(captured['n0'])
            score_sum = sum(np.sqrt(ability) / (1 + 15 * wait) for ability, wait in pairs)
            expected_scores.append((score_sum + score_sum / 2 / 1.125) / 3)
        self.assertAlmostEqual(messages[0][8], messages[1][8])
        self.assertAlmostEqual(messages[0][9], messages[1][9])
        self.assertAlmostEqual(messages[0][11] / 4, expected_scores[0])
        self.assertAlmostEqual(messages[1][11] / 4, expected_scores[1])
        self.assertNotAlmostEqual(messages[0][11], messages[1][11])

    def test_disabled_compute_fault_fill_has_zero_score(self):
        for mode in ('New_DQN', 'Pure_DQN', 'Pure_PPO'):
            node = node_fixture(mode, 2, 1.125, np.random.default_rng(13))
            node.fault_compute_double_mean = False
            state = node.get_current_state('target', 0, False, [0] * 4)
            slot_dim = 20 if mode == 'New_DQN' else 8
            for layer in range(3 if mode == 'New_DQN' else 1):
                offset = 2 * slot_dim + layer * 6
                np.testing.assert_array_equal(state[offset + 2:offset + 4], [0, 1])
                self.assertEqual(state[offset + 5], 0)
        for hops in (2, 4):
            with isacr_hop_environment(hops):
                node = node_fixture('New_DQN', 0, 1.125, np.random.default_rng(14))
                node.fault_compute_double_mean = False
                fill = node.fault_fill([[1, .8, .5, .2, .3, .9] * hops])
                for layer in range(hops):
                    self.assertEqual(fill[layer * 6 + 5], 0)

    def test_legacy_isacr_exchange_fault_score_keeps_raw_layer_scales(self):
        node = node_fixture('New_DQN', 1, 1.125, np.random.default_rng(15))
        node.historical_self_feature_elimination = False
        node.neighbor_states['n0'] = [value * scale for scale in (1, 4, 12)
                                      for value in [1, .8, .5, .2, .3, .1]]
        messages = []

        def send(source, target, state):
            messages.append(list(state))
            yield node.env.timeout(0)

        node.propagator = SimpleNamespace(send_state=send)
        node.env.process(node.state_exchanger())
        node.env.run(until=.101)
        self.assertAlmostEqual(messages[0][11], .1 + 3 * .1 / 1.125)
        self.assertAlmostEqual(messages[0][17], .4 + 3 * .4 / 1.125)

    def test_capacity_queue_score(self):
        node = node_fixture('Pure_DQN', 0, 1.5, np.random.default_rng(1))
        for ability in (50e9, 75e9, 100e9):
            node.computing_ability = ability
            node.is_computing = False
            for wait in (-1, 0, 6, 10, 12, 49, 80):
                node.computing_remain = wait * ability
                self.assertAlmostEqual(node._normalized_compute_completion_time(),
                                       (ability / 100e9) / (1 + 15 * min(max(wait, 0) / 12, 1)))
        node.is_computing = True
        node.last_computing_time = node.env.now - 5
        node.computing_remain = 15 * node.computing_ability
        self.assertAlmostEqual(node._normalized_compute_completion_time(), 1 / (1 + 15 * 10 / 12))

    def test_beta15_unsquared_score_local_and_broadcast(self):
        for mode in ('New_DQN', 'Pure_DQN', 'Pure_PPO'):
            node = node_fixture(mode, 1, 1.125, np.random.default_rng(16))
            node.computing_ability = 50e9
            node.is_computing = False
            node.computing_remain = 6 * node.computing_ability
            expected = .5 / 8.5
            state = node.get_current_state('target', 0, False, [0] * 4)
            local_start = 80 if mode == 'New_DQN' else 32
            self.assertEqual(state[local_start + 2], .25)
            self.assertAlmostEqual(state[local_start + 4], expected)
            messages = []

            def send(source, target, state):
                messages.append(list(state))
                yield node.env.timeout(0)

            node.propagator = SimpleNamespace(send_state=send)
            node.env.process(node.state_exchanger())
            node.env.run(until=.101)
            self.assertAlmostEqual(messages[0][5], expected)
            node.computing_remain = 12 * node.computing_ability
            at_twelve = node._normalized_compute_completion_time()
            node.computing_remain = 24 * node.computing_ability
            self.assertEqual(node._normalized_compute_completion_time(), at_twelve)
            self.assertEqual(at_twelve, .5 / 16)
            self.assertEqual(node._normalized_compute_time(24), 1)

    def test_shared_local_and_one_hop_scores_before_and_after_compute(self):
        for count in range(5):
            for computed in (False, True):
                states = {}
                for mode in ('New_DQN', 'Pure_DQN', 'Pure_PPO'):
                    node = node_fixture(mode, count, 1.125, np.random.default_rng(6))
                    for values in node.neighbor_states.values():
                        values[:6] = [1, .8, .5, .2, .3, .1]
                    states[mode] = node.get_current_state('target', 0, computed, [0] * 4)
                for mode in ('Pure_DQN', 'Pure_PPO'):
                    for slot in range(4):
                        np.testing.assert_array_equal(states[mode][slot * 8:slot * 8 + 6],
                                                      states['New_DQN'][slot * 20:slot * 20 + 6])
                    np.testing.assert_array_equal(states[mode][32:], states['New_DQN'][80:])

    def test_initial_neighbors_wait_for_actual_state_message(self):
        for mode in ('New_DQN', 'Pure_DQN', 'Pure_PPO'):
            sim, names = simulator(make_encoder_factory('gat')(5, 15, 2, 15))
            reference = sim.satellites[names[0]]
            node = simulation.Satellite_with_Computing(
                mode, reference.select_mode, reference.epsilon, reference.max_hop,
                reference.max_size, reference.device, sim.env, names[0], [names[1]],
                reference.memory, reference.computing_ability, reference.transmission_rate,
                reference.downlink_rate, reference.state_update_period, False, reference.logger)
            self.assertEqual(node.neighbors, [])
            self.assertEqual(node.pending_state_neighbors, {names[1]})
            self.assertEqual(node.action_neighbors[0], names[1])
            self.assertNotIn(names[1], node.neighbor_states)
            node.set_propagator(sim.propagator)
            sim.propagator.satellites[names[0]] = node
            initial_state = node.get_current_state('target', 0, False, [0] * 4)
            self.assertEqual(initial_state[5], 0)
            raw = [1, .8, .5, .2, .3, .4]
            message = [v * scale for scale in (1, 4, 12) for v in raw] if mode == 'New_DQN' else raw
            sim.env.process(sim.propagator.send_state(names[1], names[0], message))
            self.assertEqual(node.neighbors, [])
            sim.env.run(until=.001)
            self.assertIn(names[1], node.neighbors)
            self.assertNotIn(names[1], node.pending_state_neighbors)
            self.assertEqual(node.neighbor_state_versions[names[1]], 1)
            self.assertTrue(node.neighbor_has_resource_state(names[1]))
            self.assertEqual(node.neighbor_states[names[1]], message)
            self.assertAlmostEqual(node.get_current_state('target', 0, False, [0] * 4)[5], .4)

    def test_unreceived_information_uses_only_received_mean_and_clips(self):
        for mode in ('New_DQN', 'Pure_DQN', 'Pure_PPO'):
            node = node_fixture(mode, 3, 1.125, np.random.default_rng(7))
            node.neighbor_state_versions['n1'] = 0
            node.action_neighbors[-1] = None
            for name, score in (('n0', .6), ('n1', .99), ('n2', .3)):
                for layer, scale in enumerate((1, 4, 12) if mode == 'New_DQN' else (1,)):
                    node.neighbor_states[name][layer * 6 + 5] = score * scale
            state = node.get_current_state('target', 0, False, [0] * 4)
            slot_dim = 20 if mode == 'New_DQN' else 8
            for slot in (1, 3):
                for layer in range(3 if mode == 'New_DQN' else 1):
                    self.assertAlmostEqual(state[slot * slot_dim + layer * 6 + 5], fault_product(node, layer))
            # Fault scores depend on received sixth fields, not filled ability/wait.
            for name in ('n0', 'n2'):
                for layer, scale in enumerate((1, 4, 12) if mode == 'New_DQN' else (1,)):
                    node.neighbor_states[name][layer * 6 + 2:layer * 6 + 4] = [2 * scale] * 2
            state = node.get_current_state('target', 0, False, [0] * 4)
            self.assertAlmostEqual(state[slot_dim + 5], .45 / 1.125)
            node.neighbor_state_versions = dict.fromkeys(node.neighbors, 0)
            self.assertEqual(node.get_current_state('target', 0, False, [0] * 4)[5], 0)

    def test_unreceived_information_in_recipient_specific_messages(self):
        node = node_fixture('New_DQN', 3, 1.125, np.random.default_rng(8))
        node.neighbor_state_versions['n1'] = 0
        for name, score in (('n0', .6), ('n1', .99), ('n2', .3)):
            for layer, scale in enumerate((1, 4, 12)):
                node.neighbor_states[name][layer * 6 + 5] = score * scale
        messages = {}

        def send(source, target, state):
            messages[target] = list(state)
            yield node.env.timeout(0)

        node.propagator = SimpleNamespace(send_state=send)
        node.env.process(node.state_exchanger())
        node.env.run(until=.101)
        self.assertAlmostEqual(node.current_state[11] / 4, (.9 + 2 * fault_product(node)) / 4)
        for layer, (index, scale) in enumerate(((11, 4), (17, 12))):
            self.assertAlmostEqual(messages['n1'][index] / scale, (.9 + fault_product(node, layer)) / 3)
            self.assertAlmostEqual(messages['n0'][index] / scale, (.3 + 2 * fault_product(node, layer, ('n0',))) / 3)
            self.assertAlmostEqual(messages['n2'][index] / scale, (.6 + 2 * fault_product(node, layer, ('n2',))) / 3)

    def test_variable_hop_unreceived_information_is_fault_filled(self):
        for hops in (2, 4):
            with isacr_hop_environment(hops):
                node = node_fixture('New_DQN', 2, 1.125, np.random.default_rng(9))
                node.neighbor_state_versions['n1'] = 0
                node.action_neighbors[-1] = None
                node.neighbor_states = {
                    name: [v * scale for scale in node.resource_scales
                           for v in [1, .8, .5, .2, .3, score]]
                    for name, score in (('n0', .6), ('n1', .99))}
                state = node.get_current_state('target', 0, False, [0] * 4)
                for slot in (1, 3):
                    for layer in range(hops):
                        self.assertAlmostEqual(state[slot * (hops * 6 + 2) + layer * 6 + 5], fault_product(node, layer))

    def test_fault_score_uses_shared_fill_factor_and_clips(self):
        rng = np.random.default_rng(2)
        for mode in ('New_DQN', 'Pure_DQN', 'Pure_PPO'):
            for count in (0, 2):
                for factor in (0.5, 1.125, 1.5, 2):
                    node = node_fixture(mode, count, factor, rng)
                    for resource_state in node.neighbor_states.values():
                        for layer, scale in enumerate((1, 4, 12) if mode == 'New_DQN' else (1,)):
                            resource_state[layer * 6 + 5] = 0.7 * scale
                    state = node.get_current_state('target', 0, False, [0] * 4)
                    slot_dim = 20 if mode == 'New_DQN' else 8
                    self.assertEqual(len(state), 91 if mode == 'New_DQN' else 43)
                    for layer in range(3 if mode == 'New_DQN' else 1):
                        self.assertAlmostEqual(state[count * slot_dim + layer * 6 + 5], fault_product(node, layer))
                    self.assertEqual(node._empty_neighbor_state_chunk()[5], 0)
                    self.assertEqual(node._fault_neighbor_state()[5], 0)
                    self.assertAlmostEqual(node.direct_concat_fault_feature([[0, 1, .5, .2, .3, .7]])[5],
                                           min(.7 / factor, 1))

    def test_baseline_messages_carry_capacity_queue_score(self):
        for mode in ('Pure_DQN', 'Pure_PPO'):
            node = node_fixture(mode, 1, 1.5, np.random.default_rng(3))
            node.computing_ability = 50e9
            node.computing_remain = 10 * node.computing_ability
            node.is_computing = False
            messages = []

            def send(source, target, state):
                messages.append(list(state))
                yield node.env.timeout(0)

            node.propagator = SimpleNamespace(send_state=send)
            node.env.process(node.state_exchanger())
            node.env.run(until=0.101)
            self.assertEqual(len(messages), 1)
            self.assertEqual(len(messages[0]), 6)
            self.assertAlmostEqual(messages[0][2], .25)
            self.assertAlmostEqual(messages[0][5], .5 / (1 + 15 * 10 / 12))
            node.neighbor_states[node.neighbors[0]] = messages[0]
            state = node.get_current_state('target', 0, False, [0] * 4)
            self.assertAlmostEqual(state[5], .5 / (1 + 15 * 10 / 12))

    def test_recipient_excluded_score_fault_mean(self):
        node = node_fixture('New_DQN', 2, 1.5, np.random.default_rng(4))
        for index, values in enumerate(node.neighbor_states.values()):
            for layer, scale in enumerate((1, 4, 12)):
                values[layer * 6 + 5] = (0.2 + 0.2 * index) * scale
        messages = {}

        def send(source, target, state):
            messages[target] = list(state)
            yield node.env.timeout(0)

        node.propagator = SimpleNamespace(send_state=send)
        node.env.process(node.state_exchanger())
        node.env.run(until=0.101)
        self.assertAlmostEqual(messages['n0'][11] / 4, (0.4 + 2 * fault_product(node, exclude=('n0',))) / 3)
        self.assertAlmostEqual(messages['n1'][11] / 4, (0.2 + 2 * fault_product(node, exclude=('n1',))) / 3)
        for hops in (2, 4):
            with isacr_hop_environment(hops):
                node = node_fixture('New_DQN', 0, 1.5, np.random.default_rng(5))
                self.assertEqual(node.fault_fill([])[5], 0)
                self.assertAlmostEqual(node.fault_fill([[0, 1, .5, .2, .3, .4] * hops])[5], .4 / 1.5)

    def test_no_valid_neighbors_fill_from_current_local_score(self):
        for mode in ('New_DQN', 'Pure_DQN', 'Pure_PPO'):
            node = node_fixture(mode, 0, 1.125, np.random.default_rng(10))
            for ability, wait in ((100e9, 0), (50e9, 10), (100e9, 80)):
                node.computing_ability = ability
                node.computing_remain = ability * wait
                node.is_computing = False
                expected = 0
                state = node.get_current_state('target', 0, False, [0] * 4)
                slot_dim = 20 if mode == 'New_DQN' else 8
                for slot in range(4):
                    for layer in range(3 if mode == 'New_DQN' else 1):
                        self.assertAlmostEqual(state[slot * slot_dim + layer * 6 + 5], expected)
                self.assertAlmostEqual(node.direct_concat_fault_feature([])[5], expected)

    def test_no_other_received_neighbor_message_fill_uses_local_score(self):
        node = node_fixture('New_DQN', 1, 1.125, np.random.default_rng(11))
        node.computing_ability = 100e9
        node.computing_remain = 0
        node.is_computing = False
        messages = []

        def send(source, target, state):
            messages.append(list(state))
            yield node.env.timeout(0)

        node.propagator = SimpleNamespace(send_state=send)
        node.env.process(node.state_exchanger())
        node.env.run(until=.101)
        self.assertEqual(messages[0][11], 0)
        self.assertEqual(messages[0][17], 0)

    def assert_unit(self, values):
        values = np.asarray(values)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values >= -1e-12).all(), values)
        self.assertTrue((values <= 1 + 1e-12).all(), values)

    def test_four_algorithm_policy_resources_and_fault_slots(self):
        rng = np.random.default_rng(2025)
        for mode in ('New_DQN', 'Pure_DQN', 'Pure_PPO'):
            for count in range(5):
                for factor in (0.5, 1.125, 1.5, 2):
                    for computed in (False, True):
                        for _ in range(3):
                            node = node_fixture(mode, count, factor, rng)
                            state = node.get_current_state('target', 0, computed, [0] * 6)
                            slot_dim = 20 if mode == 'New_DQN' else 8
                            for start in range(0, 4 * slot_dim, slot_dim):
                                # Distance=2 is an intentional unknown-route token.
                                self.assert_unit(state[start:start + slot_dim - 1])
                            self.assert_unit(state[4 * slot_dim:4 * slot_dim + 5])
                            if not computed:
                                self.assertAlmostEqual(state[4 * slot_dim + 4],
                                                       node._normalized_compute_completion_time())

    def test_isacr_and_no_self_feature_exchange_ranges(self):
        rng = np.random.default_rng(2026)
        for count in range(1, 5):
            for factor in (0.5, 1.125, 1.5, 2):
                for disable in (False, True):
                    node = node_fixture('New_DQN', count, factor, rng)
                    node.disable_self_feature_elimination = disable
                    messages = []

                    def send(source, target, state):
                        messages.append(list(state))
                        yield node.env.timeout(0)

                    node.propagator = SimpleNamespace(send_state=send)
                    node.env.process(node.state_exchanger())
                    node.env.run(until=0.101)
                    self.assertEqual(len(messages), count)
                    for state in messages + [node.current_state]:
                        self.assert_unit([value / scale for layer, scale in enumerate((1, 4, 12))
                                          for value in state[layer * 6:(layer + 1) * 6]])

    def test_gnn_and_direct_concat_raw_fault_ranges(self):
        rng = np.random.default_rng(2027)
        for count in range(5):
            for factor in (0.5, 1.125, 1.5, 2):
                node = node_fixture('GNN_DQN', count, factor, rng)
                self.assert_unit(node.current_gnn_raw_state())
                self.assert_unit(node.current_fault_node_state())
                values = [resource(rng) for _ in range(count)]
                self.assert_unit(node.direct_concat_fault_feature(values))
                self.assert_unit(node.direct_concat_fault_feature([]))
                node.gnn = make_encoder_factory('direct_concat')(6, 15, 2, 15)
                self.assert_unit(node.current_gnn_raw_state())
                node.direct_concat_received_at = dict.fromkeys(node.neighbors, 0)
                node.gnn_layer2_messages = {
                    name: (0, [v for _ in range(21) for v in resource(rng)])
                    for name in node.neighbors}
                self.assert_unit(node.build_direct_concat_view(node.current_gnn_raw_state()))
                node.gnn = GraphSAGEEncoder(5, 5, 2, 5)
                node.__class__ = GraphSAGEEncoder.satellite_class
                node.gnn_pipeline_messages = {}
                self.assert_unit(node.current_fault_node_state())
                node.gnn_cycle_version = 1
                node.gnn_pipeline_messages = {
                    name: [(0, 0, resource(rng)[:5] + [-2.] * 10)]
                    for name in node.neighbors}
                self.assert_unit(node.current_fault_node_state())
                node.env.run(until=0.2)
                self.assertEqual(node.current_fault_node_state(), [1, 0, 0, 1, 1])

    def test_two_and_four_hop_isacr_fault_ranges(self):
        rng = np.random.default_rng(2028)
        for hops in (2, 4):
            with isacr_hop_environment(hops):
                for count in range(5):
                    node = node_fixture('New_DQN', count, 1.5, rng)
                    values = [[v for _ in range(hops) for v in resource(rng)]
                              for _ in range(count)]
                    self.assert_unit(node.fault_fill(values))
                    message = node.aggregate_message(resource(rng), dict(enumerate(values)),
                                                     recipient=0)
                    self.assert_unit(node.normalized_resources(message))
                    node.neighbor_states = {
                        name: [value * scale for layer, scale in enumerate(node.resource_scales)
                               for value in values[index][layer * 6:(layer + 1) * 6]]
                        for index, name in enumerate(node.neighbors)}
                    for computed in (False, True):
                        state = node.get_current_state('target', 0, computed, [0] * 6)
                        slot_dim = hops * 6 + 2
                        for start in range(0, 4 * slot_dim, slot_dim):
                            self.assert_unit(state[start:start + slot_dim - 1])
                        self.assert_unit(state[4 * slot_dim:4 * slot_dim + 5])


class ExtraD3QNInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Test policy adapters independently of the evaluation manifest/config.
        path = Path(__file__).resolve().parents[1] / 'test_4_5seed.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('destination_selection', 'random_offloading')]
        namespace = {'contextmanager': contextmanager}
        exec(compile(tree, str(path), 'exec'), namespace)
        cls.adapters = namespace

    def decision_node(self):
        import torch
        node = node_fixture('Pure_DQN', 4, 1.125, np.random.default_rng(9))
        node.__class__ = simulation.Satellite_with_Computing
        node.device = 'cpu'
        node.epsilon = 0
        node.ppo_shuffle_func = lambda state: (state.copy(), np.arange(4))
        node.q_net = lambda state, mask: torch.zeros((len(state), 5))
        return node

    def test_destination_selection_preserves_43_fields_and_action_mapping(self):
        import torch
        with self.adapters['destination_selection']():
            node = self.decision_node()
            state = node.get_current_state('target', 0, False, [0] * 4)
            self.assertEqual(len(state), 43)
            mapping = np.array([3, 1, 0, 2])
            node._destination_shuffle = lambda values: (values, mapping)
            node.routing_tables = {'target': (None, 1)}
            node.valid_action_mask = lambda *args: np.ones(5, dtype=bool)
            node.q_net = lambda values, mask: torch.tensor([[9., 0., 0., 0., 0.]])
            self.assertEqual(node.find_highest_score(['target'], [0] * 4, 0, False),
                             ['target'])
            decision = node._destination_decision
            np.testing.assert_array_equal(decision['policy_state'][:32],
                                          state[:32].reshape(4, 8)[mapping].reshape(-1))
            np.testing.assert_array_equal(decision['policy_state'][32:], state[32:])
            self.assertEqual(decision['action'], 3)
            result = node.get_next_hop(state, 'target', hops=0)
            self.assertEqual(result[:2], [3, 0])

    def test_random_compute_and_destination_return_transition_tuple(self):
        base = simulation.Satellite_with_Computing
        with self.adapters['random_offloading'](2025):
            node = self.decision_node()
            state = np.zeros(43)
            for random_compute, destination in ((True, 'target'), (False, node.name)):
                node._ro_compute = random_compute
                with mock.patch.object(base, 'valid_action_mask',
                                       return_value=np.ones(5, dtype=bool)):
                    result = node.get_next_hop(state, destination, hops=0)
                    action, policy_action, policy_state = result
                    self.assertEqual((action, policy_action), (4, 4))
                    np.testing.assert_array_equal(policy_state, state)
                with mock.patch.object(base, 'valid_action_mask',
                                       return_value=np.zeros(5, dtype=bool)):
                    self.assertIsNone(node.get_next_hop(state, destination, hops=0))

    def test_random_offloading_masks_resources_before_and_after_compute(self):
        with self.adapters['random_offloading'](2025):
            node = self.decision_node()
            for computed in (False, True):
                state = node.get_current_state('target', 0, computed, [0] * 4)
                self.assertEqual(len(state), 43)
                for offset in range(0, 32, 8):
                    np.testing.assert_array_equal(state[offset + np.array([2, 3, 5])], 0)
                np.testing.assert_array_equal(state[34:37], 0)

    def test_random_offloading_routes_and_restores_q_network(self):
        import torch
        base = simulation.Satellite_with_Computing
        with self.adapters['random_offloading'](2025):
            node = self.decision_node()
            node._ro_compute = False
            # Even a high compute Q-value must not override the random decision.
            node.q_net = lambda state, mask: torch.tensor([[1., 0., 0., 0., 100.]])
            original_q = node.q_net
            with mock.patch.object(base, 'valid_action_mask',
                                   return_value=np.ones(5, dtype=bool)):
                result = node.get_next_hop(np.zeros(43), 'target', hops=0)
            self.assertEqual(result[:2], [0, 0])
            self.assertIs(node.q_net, original_q)


class ReproducibilityTests(unittest.TestCase):
    def test_failure_callbacks_follow_sorted_edges(self):
        node = simulation.Propagator_Computing.__new__(simulation.Propagator_Computing)
        node.graph = nx.Graph()
        edges = [('Z', 'Y'), ('B', 'C'), ('A', 'D')]
        node.graph.add_edges_from(edges)
        env = simpy.Environment()
        node.link_failure_events = {}
        order = []
        for edge in node.graph.edges():
            event = env.event()
            event.callbacks.append(lambda _, edge=edge: order.append(edge))
            node.link_failure_events[edge] = event
        expected = sorted(node.graph.edges())
        with mock.patch.object(simulation.Propagator, 'update',
                               lambda self, graph: setattr(self, 'graph', graph)):
            node.update(nx.Graph())
        env.run()
        self.assertEqual(order, expected)

    def test_training_rejects_unfixed_process_hash(self):
        from training_runner import configure_reproducible_training
        with mock.patch.dict(os.environ, {'PYTHONHASHSEED': 'random'}):
            with self.assertRaisesRegex(RuntimeError, 'Python startup'):
                configure_reproducible_training()

    def test_training_enables_strict_determinism(self):
        import torch
        from training_runner import configure_reproducible_training
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            with mock.patch.dict(os.environ, {'PYTHONHASHSEED': '0'}):
                configure_reproducible_training()
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
            self.assertEqual(torch.get_num_threads(), 1)
        finally:
            torch.use_deterministic_algorithms(previous)

    def test_startup_propagates_hash_and_child_exit_status(self):
        from training_runner import prepare_reproducible_process
        with mock.patch.dict(os.environ, {'PYTHONHASHSEED': 'random'}):
            with mock.patch('subprocess.call', return_value=7) as child:
                with self.assertRaises(SystemExit) as result:
                    prepare_reproducible_process()
                self.assertEqual(result.exception.code, 7)
                self.assertEqual(child.call_args.kwargs['env']['PYTHONHASHSEED'], '0')


if __name__ == '__main__':
    unittest.main()
