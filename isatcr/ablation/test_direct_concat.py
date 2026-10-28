import unittest
from types import SimpleNamespace
from unittest.mock import patch

import simpy

from ablation.aggregation_models import DirectConcatEncoder
from ablation.training_entry import build_config
import training_runner
from SatelliteNetworkSimulator_Computing import Satellite_with_Computing


def satellite(env, name, neighbors):
    node = Satellite_with_Computing.__new__(Satellite_with_Computing)
    node.env = env
    node.name = name
    node.mode = 'GNN_DQN'
    node.gnn_node_feature_dim = 6
    node.neighbors = neighbors
    node.pending_state_neighbors = set()
    node.on_neighbor_state_received = lambda *args, **kwargs: None
    node.gnn = DirectConcatEncoder(6, 15, 2, 15)
    node.active = True
    node.state_update_period = 0.1
    node.heartbeat_timeout = 0.15
    node.fault_fill_multiplier = 2.0
    node.gnn_cycle_version = 0
    node.gnn_raw_snapshots = {}
    node.gnn_raw_messages = {}
    node.gnn_layer1_messages = {}
    node.gnn_layer2_messages = {}
    node.direct_concat_received_at = {}
    node.last_heartbeat = {}
    node.neighbor_states = {neighbor: [0, 1, 0, 0, 0, 0] for neighbor in neighbors}
    node.memory = 100
    node.max_hop = 10
    node.routing_tables = {'target': ('b', 1)}
    node.neighbor_hops = {neighbor: {'target': 1} for neighbor in neighbors}
    node.current_gnn_raw_state = lambda: [1, 0.8, 0.6, 0.2, 0.1, 0.5]
    node._normalized_compute_completion_time = lambda: node.current_gnn_raw_state()[5]
    return node


class DirectConcatRegressionTests(unittest.TestCase):
    def test_training_passes_no_gradient_clipping_threshold(self):
        config = build_config('direct_concat')
        config['general']['phase'] = 'test'
        self.assertFalse(config['agent']['use_gradient_clipping'])
        with patch.object(training_runner, 'GNN_DDQN_Agent', side_effect=RuntimeError('stop before training')) as factory:
            with self.assertRaisesRegex(RuntimeError, 'stop before training'):
                training_runner.run_training(config)
        self.assertIsNone(factory.call_args.kwargs['max_grad_norm'])

    def test_memory_mask_refreshes_and_preserves_forwarding_debits(self):
        env = simpy.Environment()
        node = satellite(env, 'a', ['b'])
        def mask():
            return node.valid_action_mask('target', 20, False, 0)[0]

        self.assertFalse(mask())  # No received capacity advertisement yet.
        low = [1, 0.1, 0.6, 0.2, 0.1, 0.5] * 21
        node.receive_gnn_embedding_message('b', 'layer2', 0, low)
        self.assertFalse(mask())
        high = [1, 0.3, 0.6, 0.2, 0.1, 0.5] * 21
        node.receive_gnn_embedding_message('b', 'layer2', 1, high)
        self.assertTrue(mask())
        node._update_neighbor_memory_after_forward('b', 20)
        self.assertFalse(mask())
        self.assertEqual(node.gnn_layer2_messages['b'][1][1], 0.3)
        node.receive_gnn_embedding_message('b', 'layer2', 1, high)
        self.assertFalse(mask())  # A duplicate cannot undo the debit.
        node.receive_gnn_embedding_message('b', 'layer2', 2, high)
        self.assertTrue(mask())  # A new advertisement refreshes capacity.
        env.run(until=0.16)
        self.assertFalse(mask())

    def test_updates_every_point_one_seconds_and_relays_three_hops(self):
        env = simpy.Environment()
        nodes = {name: satellite(env, name, neighbors) for name, neighbors in
                 [('a', ['b']), ('b', ['a', 'c']), ('c', ['b', 'd']), ('d', ['c'])]}
        nodes['d'].current_gnn_raw_state = lambda: [0, 0.7, 0.9, 0.3, 0.2, 0.4]
        sent = []

        def send(source, target, stage, version, value):
            self.assertIn(target, nodes[source].neighbors)
            self.assertEqual(stage, 'layer2')
            self.assertEqual(len(value), 126)
            sent.append((source, env.now, version))
            yield env.timeout(0.005)
            nodes[target].receive_gnn_embedding_message(source, stage, version, value)

        for node in nodes.values():
            node.propagator = SimpleNamespace(send_gnn_embedding_message=send)
            env.process(node.direct_concat_exchanger())
        env.run(until=0.31)
        self.assertEqual([version for source, _, version in sent if source == 'a'], [1, 2, 3])
        self.assertEqual(len(nodes['a'].direct_concat_local_view), 126)
        view = nodes['a'].fresh_direct_concat_view('b')
        # b's second-hop c->d slot exposes d to a, three links away.
        self.assertEqual(view[60:66], nodes['d'].current_gnn_raw_state())
        nodes['d'].current_gnn_raw_state = lambda: [0, 0.7, 0.4, 0.3, 0.2, 0.4]
        env.run(until=0.71)
        self.assertEqual(nodes['a'].fresh_direct_concat_view('b')[62], 0.4)

    def test_recent_previous_version_is_used_but_expired_view_is_filled(self):
        env = simpy.Environment()
        node = satellite(env, 'a', ['b'])
        view = [0, 0.8, 0.6, 0.2, 0.1, 0.5] * 21
        node.receive_gnn_embedding_message('b', 'layer2', 0, view)
        env.run(until=0.1)
        self.assertEqual(node.build_direct_concat_view([1] * 6)[6:12], view[:6])
        env.run(until=0.16)
        self.assertIsNone(node.fresh_direct_concat_view('b'))
        self.assertEqual(node.build_direct_concat_view([1] * 6)[6:12], [1, 0, 0, 1, 1, 0])

    def test_old_or_duplicate_message_does_not_revive_cache(self):
        env = simpy.Environment()
        node = satellite(env, 'a', ['b'])
        node.receive_gnn_embedding_message('b', 'layer2', 2, [0.5] * 126)
        env.run(until=0.2)
        for version in (1, 2):
            node.receive_gnn_embedding_message('b', 'layer2', version, [0.9] * 126)
        self.assertEqual(node.gnn_layer2_messages['b'], (2, [0.5] * 126))
        self.assertEqual(node.last_heartbeat['b'], 0)
        self.assertIsNone(node.fresh_direct_concat_view('b'))


if __name__ == '__main__':
    unittest.main()
