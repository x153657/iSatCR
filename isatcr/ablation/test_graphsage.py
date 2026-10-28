import unittest
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import numpy as np
import simpy
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch

from ablation import training_entry
from ablation.aggregation_models import make_encoder_factory
from ablation.train_graphsage import GraphSAGEEncoder, GraphSAGESatellite, seed_config
from Base_Agents import GNN_DDQN_Agent, ShuffleEx_gnn
import SatelliteNetworkSimulator_Computing as simulation


def satellite(encoder, neighbors=('a', 'b')):
    node = GraphSAGESatellite.__new__(GraphSAGESatellite)
    node.env = SimpleNamespace(now=0.1)
    node.neighbors = list(neighbors)
    node.gnn = encoder
    node.gnn_node_feature_dim = 5
    node.gat_node_dim = encoder.output_dim
    node.gnn_cycle_version = 1
    node.state_update_period = 0.1
    node.heartbeat_timeout = 0.15
    node.fault_fill_multiplier = 2.0
    node.last_heartbeat = {}
    node.neighbor_states = {}
    node.on_neighbor_state_received = lambda neighbor, **kwargs: None
    node.pending_state_neighbors = set()
    node._gnn_link_features = lambda neighbor: [0., 0.]
    node.gnn_pipeline_messages = {}
    node.gnn_active_neighbor_embeddings = {}
    node.name = 'local'
    return node


def message(encoder, raw=(1, 0.6, 0.3, 0.2, 0.1), version=0):
    node = satellite(encoder, ())
    node.env.now = version * 0.1
    node.gnn_cycle_version = version
    return node.aggregate_gnn_message(list(raw), version)


def distributed_messages(encoder):
    node = satellite(encoder)
    remote = message(encoder, (0, 0.8, 0.4, 0.3, 0.2))
    node.env.now = 0.01
    node.receive_gnn_embedding_message('a', 'pipeline', 0, remote)
    node.env.now = 0.1
    local = node.aggregate_gnn_message([1, 0.6, 0.3, 0.2, 0.1], 1)
    fault = node._fault_message()
    return [local, fault, local, fault]


def simulator(encoder, reverse=False):
    names = ['sat_550_0_0', 'sat_550_0_1', 'sat_550_0_2']
    graph = nx.Graph()
    # Zero propagation delay exercises simultaneous delivery ordering.
    for name in reversed(names[:2]) if reverse else names[:2]:
        graph.add_node(name, pos=(0, 0, 0))
    graph.add_edge(*names[:2])
    sim = simulation.SatelliteNetworkSimulator_OnbardComputing(
        mode='GNN_DQN', select_mode=1, q_net=None, gnn=encoder,
        epsilon=0, reward_factors=(1, 0.05, 1, 0.1, 0.25), device='cpu',
        mission_possibility=[0.5, 0.5], poisson_rate_g=1, poisson_rate_s=1,
        packet_frequency=1, computing_demand_factor=[150, 250],
        computing_demand_factor_2=[300, 500], size_after_computing_factor=[0.08, 0.125],
        size_after_computing_1=40000, graph=graph, landmarks={}, ground_cover_sats={},
        mean_interval_time=30, memory=12e9, computing_abilities=dict.fromkeys(names, 50e9),
        transmission_rate=1.2e9, downlink_rate=3e9, downstream_delays=0.001,
        packet_size_range=[200e6, 600e6], state_update_period=0.1,
        logger=SimpleNamespace(log=lambda *args, **kwargs: None), gnn_node_feature_dim=5,
    )
    for node in sim.satellites.values():
        sim.env.process(node.gnn_embedding_exchanger())
    return sim, names


class GraphSAGERegressionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.encoder = GraphSAGEEncoder(5, 5, 2, 5)

    def test_compute_backlog_alignment(self):
        sim, names = simulator(self.encoder)
        node = sim.satellites[names[0]]
        for wait, expected in ((-1, 0), (0, 0), (6, .12), (10, .2),
                               (12, .24), (25, .5), (80, 1)):
            self.assertAlmostEqual(node._normalized_compute_time(wait), expected)
        node.computing_ability = 100e9
        node.computing_remain = 30 * node.computing_ability
        node.is_computing = True
        node.last_computing_time = sim.env.now - 5
        self.assertAlmostEqual(node.current_gnn_raw_state()[3], .5)
        node.gnn = make_encoder_factory('direct_concat')(6, 15, 2, 15)
        self.assertAlmostEqual(node.current_gnn_raw_state()[5], 1 / 26)
        node.computing_remain = 100 * node.computing_ability
        self.assertEqual(node.current_gnn_raw_state()[3], 1)
        self.assertAlmostEqual(node.current_gnn_raw_state()[5], 1 / 51)

    def test_transmission_wait_alignment(self):
        sim, names = simulator(self.encoder)
        local, remote = (sim.satellites[name] for name in names[:2])
        local.neighbors = [names[1]]
        remote.neighbors = [names[0]]
        local.transmission_size[names[1]] = 2.4e9
        local.active_transmissions[names[1]] = (0.5,)
        remote.transmission_size[names[0]] = 6e9
        for wait, expected in ((-1, 0), (0, 0), (1.5, 0.5), (3, 1), (6, 1)):
            self.assertAlmostEqual(local._normalized_transmission_wait(wait), expected)
        self.assertAlmostEqual(local._normalized_link_transmission_wait(names[1]), 5 / 6)
        self.assertEqual(remote._normalized_link_transmission_wait(names[0]), 1)
        self.assertAlmostEqual(local.current_gnn_raw_state()[4], 5 / 24)
        # Average physical waits before clipping, rather than clipped features.
        self.assertAlmostEqual(remote.current_gnn_raw_state()[4], 5 / 12)
        local.routing_tables = {names[0]: (names[0], 0), names[1]: (names[1], 1)}
        remote.routing_tables = {names[1]: (names[1], 0)}
        local.adjacency_table[names[1]] = ([names[0]], sim.env.now)
        local.three_hop_info_db.update(remote.gather_two_hop_info())
        local.neighbor_states[names[1]] = remote.current_gnn_raw_state()
        _, graph_edges = simulation.Satellite_with_Computing.construct_gnn_graph(local)
        edges = {(edge['source'], edge['target']): edge['attributes']
                 for edge in graph_edges}
        self.assertAlmostEqual(edges[names[0], names[1]], 5 / 6)
        self.assertEqual(edges[names[1], names[0]], 1)
        local.gnn = make_encoder_factory('gat')(5, 15, 2, 15)
        with patch.object(local.gnn, 'aggregate_stage',
                          side_effect=lambda x, *args: x[0:1]) as aggregate:
            local.aggregate_gnn_layer(local.current_gnn_raw_state(), {}, 1, 1)
        self.assertTrue(torch.allclose(aggregate.call_args.args[3],
                                      torch.tensor([[1., 5 / 6]])))

    def test_configuration_and_scoped_factory(self):
        config = training_entry.build_config('graphsage', 2)
        self.assertEqual([config['agent'][key] for key in
                          ('in_channels', 'hidden_channels', 'gnn_dim')], [6, 5, 5])
        encoder = config['agent']['gnn_factory'](6, 5, 2, 5)
        self.assertEqual(encoder.output_dim, 16)
        self.assertIs(encoder.satellite_class, GraphSAGESatellite)
        self.assertIsNot(simulation.Satellite_with_Computing, GraphSAGESatellite)
        self.assertIs(make_encoder_factory('gat')(5, 15, 2, 15).satellite_class, GraphSAGESatellite)
        self.assertEqual(training_entry.build_config('gat')['agent']['hidden_channels'], 15)
        actual = seed_config(2)
        for section, key in [('general', 'scenario_seed'), ('general', 'random_seed'),
                             ('agent', 'model_path'), ('environment', 'save_training_data')]:
            self.assertEqual(actual[section][key], config[section][key])

    def test_four_neighbor_formula_ignores_neighbor_second_layer(self):
        node = satellite(self.encoder, ('a', 'b', 'c', 'd'))
        remotes = [message(self.encoder, (i % 2, 0.2 * i, 0.3, 0.1 * i, 0.4))
                   for i in range(1, 5)]
        node.env.now = 0.01
        for name, remote in zip(node.neighbors, remotes):
            node.receive_gnn_embedding_message(name, 'pipeline', 0, remote)
        node.env.now = 0.1
        own = [1, 0.6, 0.3, 0.2, 0.1]
        result = node.aggregate_gnn_message(own, 1)
        # Independent explicit mean formula, including SAGEConv's bias.
        conv1, conv2 = self.encoder.conv1, self.encoder.conv2
        raw_mean = torch.tensor([m[:5] for m in remotes]).mean(0)
        h1 = F.leaky_relu(conv1.lin_r(torch.tensor(own)) + conv1.lin_l(raw_mean))
        hidden_mean = torch.tensor([m[5:10] for m in remotes]).mean(0)
        h2 = F.leaky_relu(conv2.lin_r(h1) + conv2.lin_l(hidden_mean))
        torch.testing.assert_close(torch.tensor(result), torch.cat([torch.tensor(own), h1, h2]))
        for records in node.gnn_pipeline_messages.values():
            records[-1][2][10:15] = [999.0] * 5
        self.assertEqual(node.aggregate_gnn_message(own, 1), result)

    def test_previous_tick_survives_new_current_tick_arrival(self):
        node = satellite(self.encoder, ('a',))
        old, current = message(self.encoder, version=0), message(self.encoder, version=1)
        node.env.now = 0.01
        node.receive_gnn_embedding_message('a', 'pipeline', 0, old)
        node.env.now = 0.1
        node.receive_gnn_embedding_message('a', 'pipeline', 1, current)
        self.assertEqual(node._fresh_message(node.gnn_pipeline_messages, 'a', 1), old)
        self.assertEqual(node._fresh_message(node.gnn_pipeline_messages, 'a', 1, False), current)
        # A previous packet delivered exactly at the boundary is excluded.
        late = satellite(self.encoder, ('a',))
        late.receive_gnn_embedding_message('a', 'pipeline', 0, old)
        self.assertIsNone(late._fresh_message(late.gnn_pipeline_messages, 'a', 1))

    def test_old_duplicate_and_expired_messages_are_rejected(self):
        node = satellite(self.encoder)
        newer = message(self.encoder, version=1)
        node.env.now = 0.11
        node.receive_gnn_embedding_message('a', 'pipeline', 1, newer)
        node.env.now = 0.2
        node.receive_gnn_embedding_message('a', 'pipeline', 0, message(self.encoder))
        node.receive_gnn_embedding_message('a', 'pipeline', 1, newer)
        self.assertEqual(node.last_heartbeat['a'], 0.11)
        self.assertEqual(len(node.gnn_pipeline_messages['a']), 1)
        node.env.now = 0.3
        self.assertIsNone(node._fresh_message(node.gnn_pipeline_messages, 'a', 2))

    def test_fault_fill_and_no_neighbors(self):
        node = satellite(self.encoder)
        node.env.now = 0.01
        node.receive_gnn_embedding_message('a', 'pipeline', 0,
                                          message(self.encoder, (0, 1, 1, 0.4, 0.8)))
        node.env.now = 0.1
        np.testing.assert_allclose(node.current_fault_node_state(), [1, 0, 0.5, 0.8, 1])
        self.assertEqual(len(node._fault_message()), 15)
        node.env.now = 1
        self.assertEqual(node.current_fault_node_state(), [1, 0, 0, 1, 1])
        self.assertEqual(satellite(self.encoder, ()).current_fault_node_state(), [1, 0, 0, 1, 1])

    def test_replay_matches_unchanged_parameters_and_batch_offsets(self):
        messages = distributed_messages(self.encoder)
        data = self.encoder.replay_data(messages)
        expected = torch.tensor(messages)
        self.assertEqual(tuple(data.decision_embeddings.shape), (1, 60))
        torch.testing.assert_close(self.encoder(data)[:4], expected)
        fault = satellite(self.encoder)._fault_message()
        small = self.encoder.replay_data([fault] * 4)
        batch = Batch.from_data_list([data, small])
        result = self.encoder(batch)
        torch.testing.assert_close(result[batch.batch == 0][:4], expected)
        torch.testing.assert_close(result[batch.batch == 1][:4], torch.tensor([fault] * 4))
        before = data.x.clone()
        messages[0][0] += 10
        torch.testing.assert_close(data.x, before)

    def test_replay_reencodes_using_current_parameters(self):
        data = self.encoder.replay_data(distributed_messages(self.encoder))
        actual_decision = data.decision_embeddings.clone()
        with torch.no_grad():
            self.encoder.conv2.lin_r.weight.add_(0.5)
        self.assertFalse(torch.allclose(self.encoder(data)[:4].reshape(1, -1), actual_decision))
        torch.testing.assert_close(data.decision_embeddings, actual_decision)

    def test_simulator_sends_one_15d_message_every_point_one_seconds(self):
        sim, names = simulator(self.encoder)
        sent = []
        original = sim.propagator.send_gnn_embedding_message
        def record(source, neighbor, stage, version, payload):
            self.assertIs(type(payload), list)
            self.assertEqual(stage, 'pipeline')
            self.assertEqual(len(payload), 15)
            sent.append((source, sim.env.now, version))
            yield from original(source, neighbor, stage, version, payload)
        sim.propagator.send_gnn_embedding_message = record
        sim.env.run(until=0.31)
        ticks = [(time, version) for name, time, version in sent if name == names[0]]
        np.testing.assert_allclose([time for time, _ in ticks], [0.1, 0.2, 0.3])
        self.assertEqual([version for _, version in ticks], [1, 2, 3])
        self.assertEqual(sim.propagator.gnn_training_metadata, {})
        node = sim.satellites[names[0]]
        state = node.get_current_state(names[1], 0, False, [0, 0.5, 0.5, 0.5])
        self.assertEqual(tuple(state[0].decision_embeddings.shape), (1, 60))
        self.assertEqual(len(state[1]), 18)
        torch.testing.assert_close(self.encoder(state[0])[:4].reshape(1, -1),
                                   state[0].decision_embeddings)
        self.assertTrue(torch.all(state[0].message_model_version == 3))
        self.assertIn(3, state[0].message_layer1_model_version.tolist())
        remote = node.gnn_pipeline_messages[names[1]][-1][2]
        torch.testing.assert_close(state[0].decision_embeddings[0, :15], torch.tensor(remote))

    def test_simultaneous_process_order_does_not_change_messages(self):
        first, names = simulator(self.encoder)
        reverse, _ = simulator(self.encoder, reverse=True)
        first.env.run(until=0.31)
        reverse.env.run(until=0.31)
        for name in names[:2]:
            torch.testing.assert_close(torch.tensor(first.satellites[name].gnn_local_message),
                                       torch.tensor(reverse.satellites[name].gnn_local_message))

    def test_each_tick_freezes_both_layers_and_joining_aligns(self):
        sim, names = simulator(self.encoder)
        sim.env.run(until=0.15)
        first = sim.satellites[names[0]].gnn_cycle_encoder
        before = first.conv2.lin_r.weight.clone()
        with torch.no_grad():
            self.encoder.conv2.lin_r.weight.add_(0.5)
        self.assertIs(self.encoder.communication_snapshot(sim.env, 1), first)
        torch.testing.assert_close(first.conv2.lin_r.weight, before)
        sim.env.run(until=0.21)
        node = sim.satellites[names[0]]
        self.assertEqual(node.gnn_cycle_encoder.model_version, 2)
        self.assertIs(node.gnn_cycle_encoder, sim.satellites[names[1]].gnn_cycle_encoder)
        # Previous-tick hidden inputs are intentionally valid after a weight update.
        self.assertTrue(node.gnn_local_message.trace[2][2])
        joining = satellite(self.encoder, ())
        joining.env = sim.env
        joining.active = True
        joining.current_gnn_raw_state = lambda: [1, 0.6, 0.3, 0.2, 0.1]
        sim.env.process(joining.gnn_embedding_exchanger())
        sim.env.run(until=0.31)
        self.assertEqual(joining.gnn_cycle_version, 3)
        self.assertIs(joining.gnn_cycle_encoder, node.gnn_cycle_encoder)
        upgraded = sim.graph.copy()
        upgraded.add_node(names[2], pos=(0, 0, 0))
        upgraded.add_edge(names[1], names[2])
        sim.fixed_workload_sessions = {}
        sim.upgrade_all(upgraded, {}, {})
        added = sim.satellites[names[2]]
        self.assertIsInstance(added, GraphSAGESatellite)
        state = added.get_current_state(names[1], 0, False, [0, 0.5, 0.5, 0.5])
        self.assertEqual(tuple(state[0].decision_embeddings.shape), (1, 60))

    def test_alignment_fixed_slots_and_memory_admission(self):
        sim, names = simulator(self.encoder)
        a, b = (sim.satellites[n] for n in names[:2])
        self.assertEqual(a.action_neighbors, [names[1], None, None, None])
        b.current_memory_occupy = 0.8 * b.memory
        sim.env.run(until=0.11)
        self.assertAlmostEqual(a.neighbor_states[names[1]][1], 0.2)
        self.assertFalse(a._memory_allowed_action_indices(0.3*a.memory))

    def test_alignment_computed_state_masks_inputs_without_mutating_messages(self):
        sim, names = simulator(self.encoder)
        sim.env.run(until=0.21)
        node = sim.satellites[names[0]]
        before = list(node.gnn_pipeline_messages[names[1]][-1][2])
        plain = node.get_current_state(names[1], 0, False, [0, 0.5, 0.5, 0.5])
        masked = node.get_current_state(names[1], 0, True, [0, 0.5, 0.5, 0.5])
        self.assertTrue(torch.all(masked[0].x[:, 2:4] == 0))
        np.testing.assert_array_equal(masked[1][10:12], [0, 0])
        self.assertEqual(node.gnn_pipeline_messages[names[1]][-1][2], before)
        self.assertTrue(torch.any(plain[0].x[:, 2] != 0))
        torch.testing.assert_close(node.gnn_cycle_encoder(masked[0])[:4].reshape(1,-1),
                                   masked[0].decision_embeddings)

    def test_alignment_heartbeat_boundary_and_old_multihop_trace(self):
        sim, names = simulator(self.encoder)
        sim.env.run(until=0.11)
        node = sim.satellites[names[0]]
        self.assertTrue(node.neighbor_is_alive(names[1]))
        node.env = SimpleNamespace(now=0.25)
        self.assertTrue(node.neighbor_is_alive(names[1]))
        node.env.now = 0.250001
        self.assertFalse(node.neighbor_is_alive(names[1]))
        # Direct receipt refreshes liveness even though the trace is old.
        node.env.now = 1.0
        node.on_neighbor_state_received = lambda neighbor, **kwargs: None
        old = message(self.encoder)
        node.receive_gnn_embedding_message(names[1], 'pipeline', 9, old)
        node.env.now = 1.11
        self.assertIsNotNone(node._fresh_message(node.gnn_pipeline_messages, names[1], 11))

    def test_alignment_policy_permutation_remaps_replay_roots(self):
        from unittest.mock import patch
        sim, names = simulator(self.encoder)
        sim.env.run(until=0.21)
        state = sim.satellites[names[0]].get_current_state(names[1],0,False,[0,0.5,0.5,0.5])
        indices = np.array([2,0,3,1])
        with patch('numpy.random.permutation', return_value=indices):
            shuffled, mapping = self.encoder.shuffle_for_policy(state)
        np.testing.assert_array_equal(mapping, indices)
        torch.testing.assert_close(shuffled[0].decision_embeddings.view(4,15),
                                   state[0].decision_embeddings.view(4,15)[indices])
        torch.testing.assert_close(self.encoder(shuffled[0])[:4].reshape(1,-1),
                                   shuffled[0].decision_embeddings)
        np.testing.assert_array_equal(shuffled[1][:8].reshape(4,2),state[1][:8].reshape(4,2)[indices])

    def test_alignment_same_heartbeat_monitor_for_isacr_and_graphsage(self):
        import math
        times = []
        for cls, mode in ((simulation.Satellite_with_Computing, 'New_DQN'),
                          (GraphSAGESatellite, 'GNN_DQN')):
            env = simpy.Environment(initial_time=0.03)
            node = cls.__new__(cls)
            node.env, node.mode, node.active = env, mode, True
            node.neighbors = ['b']
            node.last_heartbeat = {'b': 0.03}
            node.last_heartbeat_sent_times = {'b': 0.03}
            node.heartbeat_timeout = 0.15
            node.state_update_period = 0.1
            expired = []
            def remove(neighbor):
                expired.append(env.now)
                node.neighbors.remove(neighbor)
                yield env.timeout(0)
            node.del_neighbor = remove
            env.process(node.monitor_single_neighbor('b'))
            deadline = 0.15  # Expiry follows the sender's global exchange cycle.
            env.run(until=deadline)
            self.assertEqual(expired, [])
            self.assertTrue(node.neighbor_is_alive('b'))
            env.run(until=deadline + 0.000001)
            self.assertEqual(expired, [math.nextafter(deadline, math.inf)])
            times.append(expired[0])
        self.assertEqual(times[0], times[1])

    def test_alignment_passive_wait_matches_isacr_retry_limit(self):
        sim, names = simulator(self.encoder)
        node = sim.satellites[names[0]]
        node.neighbor_hops = {}
        node.valid_action_mask = lambda *args, **kwargs: np.zeros(5, dtype=bool)
        packet = simulation.Packet(node.name, 'missing', 0, 1)
        packet.extra_information([True, 0, 0, 1, 0, None, None])
        packet.wait_count = 0
        packet.wait_retry_cycle = 1
        simulation.mark_queue_entry(packet, 'waiting_forward', sim.env.now)
        node.waiting_forward.append(packet)
        dropped = []
        node._drop_no_route = lambda *args, **kw: dropped.append((sim.env.now,kw['reason']))
        node._schedule_waiting_forward_retry()
        sim.env.run(until=0.54)
        self.assertEqual(packet.wait_count, 4)
        self.assertEqual(dropped, [])
        sim.env.run(until=0.56)
        self.assertEqual(packet.wait_count, 5)
        self.assertEqual(simulation.PASSIVE_WAIT_MAX_RETRIES, float('inf'))
        self.assertEqual(dropped, [])
        self.assertEqual(len(node.waiting_forward), 1)
        node.valid_action_mask = lambda *args, **kwargs: np.ones(5, dtype=bool)
        sim.env.run(until=.66)
        self.assertEqual(len(node.waiting_forward), 0)

    def test_alignment_actual_decision_and_replay_action_mapping(self):
        from unittest.mock import patch
        sim, names = simulator(self.encoder)
        sim.env.run(until=0.21)
        node = sim.satellites[names[0]]
        node.ppo_shuffle_func = self.encoder.shuffle_for_policy
        state = node.get_current_state(names[1],0,False,[0,0.5,0.5,0.5])
        indices = np.array([2,0,3,1])
        seen = []
        def qnet(tensor, mask):
            seen.append((tensor.clone(), np.asarray(mask).copy()))
            return torch.tensor([[0.,10.,0.,0.,-1.]])
        node.q_net = qnet
        with patch('numpy.random.permutation', return_value=indices):
            result = node.get_next_hop(state,names[1],packet_size=1,hops=0)
        environment_action, replay_action, replay_state = result
        self.assertEqual((environment_action,replay_action),(0,1))
        torch.testing.assert_close(seen[0][0][:,:60], replay_state[0].decision_embeddings)
        self.assertEqual(seen[0][1].tolist(), [False,True,False,False,True])
        mapped = node.replay_action_mask(node.valid_action_mask(names[1],1,False,0),indices)
        self.assertEqual(mapped.tolist(),seen[0][1].tolist())

    def test_alignment_timeout_fault_mask_and_probe_recovery(self):
        sim, names = simulator(self.encoder)
        node = sim.satellites[names[0]]
        slots = list(node.action_neighbors)
        propagator = node.propagator
        send_bundle, send_state = propagator.send_gnn_embedding_message, propagator.send_state
        def outage_bundle(*args):
            if 0.11 < sim.env.now < 0.4:
                yield sim.env.timeout(0)
            else:
                yield from send_bundle(*args)
        def outage_probe(*args):
            if 0.11 < sim.env.now < 0.4:
                yield sim.env.timeout(0)
            else:
                yield from send_state(*args)
        propagator.send_gnn_embedding_message = outage_bundle
        propagator.send_state = outage_probe
        for remote in sim.satellites.values():
            sim.env.process(remote.state_exchanger())
            for neighbor in list(remote.neighbors):
                sim.env.process(remote.monitor_single_neighbor(neighbor))
        sim.env.run(until=0.26)
        self.assertNotIn(names[1], node.neighbors)
        self.assertEqual(node.action_neighbors, slots)
        self.assertIn(names[1], node.pending_state_neighbors)
        self.assertFalse(node.valid_action_mask(names[1], 1, False, 0)[0])
        state = node.get_current_state(names[1], 0, False, [0,0.5,0.5,0.5])
        torch.testing.assert_close(state[0].x[0,:3], torch.tensor([1.,0.,0.]))
        self.assertIsNone(node._fresh_message(node.gnn_pipeline_messages, names[1], 2))
        # An old duplicate must not revive a neighbor before cache validation.
        old = node.gnn_pipeline_messages[names[1]][-1][2]
        sim.env.process(send_bundle(names[1],node.name,'pipeline',0,list(old)))
        sim.env.run(until=0.27)
        self.assertNotIn(names[1], node.neighbors)
        self.assertEqual(node.last_heartbeat[names[1]], .1)
        sim.env.run(until=0.51)
        self.assertIn(names[1], node.neighbors)
        self.assertEqual(node.action_neighbors, slots)
        self.assertNotIn(names[1], node.pending_state_neighbors)
        self.assertTrue(node.valid_action_mask(names[1], 1, False, 0)[0])
        self.assertIsNotNone(node._fresh_message(node.gnn_pipeline_messages, names[1], 4, False))

    def test_alignment_forward_packet_stores_policy_state_action_and_masks(self):
        from unittest.mock import patch
        sim, names = simulator(self.encoder)
        def paused_transmitter(neighbor):
            yield sim.env.timeout(1)
        sim.satellites[names[0]].transmit_packet = paused_transmitter
        sim.env.run(until=0.21)
        node = sim.satellites[names[0]]
        node.ppo_shuffle_func = self.encoder.shuffle_for_policy
        node.q_net = lambda tensor, mask: torch.tensor([[0.,10.,0.,0.,-1.]])
        previous = node.get_current_state(names[1],0,False,[0,0.5,0.5,0.5])
        packet = simulation.Packet(node.name,names[1],sim.env.now,1)
        packet.extra_information([False,0,1,1,sim.env.now,previous,1])
        packet.forward_action_mask = np.array([False,True,False,False,True])
        self.assertTrue(node.push_forward(packet))
        sim.env.process(node.forward_packet())
        with patch('numpy.random.permutation', return_value=np.array([2,0,3,1])):
            sim.env.run(until=0.22)
        stored_state, stored_action = packet.information[-2:]
        self.assertEqual(stored_action,1)
        self.assertIn(packet,node.transmission_queue[names[1]].items)
        self.assertEqual(packet.forward_action_mask.tolist(),[False,True,False,False,True])
        experience = node.propagator.experiences[-1]
        self.assertIs(experience[0],previous)
        self.assertEqual(experience[2],1)
        self.assertIs(experience[4],stored_state)
        np.testing.assert_array_equal(experience[6],packet.forward_action_mask)
        np.testing.assert_array_equal(experience[7],packet.forward_action_mask)
        torch.testing.assert_close(self.encoder(stored_state[0])[:4].reshape(1,-1),
                                   stored_state[0].decision_embeddings)

    def test_alignment_training_runner_disables_replay_permutation(self):
        from unittest.mock import patch
        import training_runner
        config = training_entry.build_config('graphsage')
        config['general']['phase'] = 'test'
        class Captured(Exception):
            pass
        captured = []
        def capture(config,agent,device,callback,**kwargs):
            captured.append((agent,callback))
            raise Captured()
        with patch.object(GNN_DDQN_Agent,'load_model'), \
             patch.object(training_runner,'build_environment',side_effect=capture):
            with self.assertRaises(Captured):
                training_runner.run_training(config)
        self.assertIsNone(captured[0][0].shuffle_func)
        self.assertIsNotNone(captured[0][1])

    def test_real_dueling_ddqn_update_trains_both_graphsage_layers(self):
        agent = GNN_DDQN_Agent(
            5, 5, 16, 2, 5, 18, 5, 10, 2, 0.99, 'cpu', -1, dueling=True,
            gnn_factory=GraphSAGEEncoder, shuffle_func=None,
            target_update_interval=1,
        )
        data = agent.online_gnn.replay_data(distributed_messages(agent.online_gnn))
        state = [data, np.zeros(18)]
        before = [conv.lin_l.weight.detach().clone() for conv in
                  (agent.online_gnn.conv1, agent.online_gnn.conv2)]
        mask = np.ones(5, dtype=bool)
        experiences = [(state, 0, i, 1.0, state, 0, mask, mask) for i in (0, 1)]
        self.assertEqual(agent.update(experiences), 1)
        for old, conv in zip(before, (agent.online_gnn.conv1, agent.online_gnn.conv2)):
            self.assertFalse(torch.equal(old, conv.lin_l.weight))
            self.assertTrue(torch.isfinite(conv.lin_l.weight.grad).all())
        self.assertEqual(agent.target_update_events, 1)


if __name__ == '__main__':
    unittest.main()
