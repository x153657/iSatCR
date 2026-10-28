"""GAT/JKNet replay uses the actual distributed resource-message history."""
import numpy as np
import torch
from torch_geometric.data import Data

from ablation.aggregation_models import GATEncoder, JKNetEncoder, DirectConcatEncoder
from ablation.train_graphsage import GraphSAGEEncoder, GraphSAGESatellite, _message_encoder_apply
import SatelliteNetworkSimulator_Computing as simulation


class DirectConcatView(list):
    def __init__(self, values, node_names):
        super().__init__(values)
        self.node_names = tuple(node_names)


class MessageReplay:
    aligned_graphsage_policy = True
    satellite_class = GraphSAGESatellite
    communication_snapshot = GraphSAGEEncoder.communication_snapshot
    _apply = _message_encoder_apply
    shuffle_for_policy = staticmethod(GraphSAGEEncoder.shuffle_for_policy)

    def replay_data(self, messages):
        data = GraphSAGEEncoder.replay_data(messages)
        data.decision_embeddings = torch.tensor(
            [[value for message in messages for value in message[self.conv1.in_channels + self.conv1.out_channels:]]], dtype=torch.float)
        return data


class GATMessageEncoder(MessageReplay, GATEncoder):
    def aggregate_stage(self, x, edge_index, layer, edge_attr=None):
        if edge_attr is None:
            edge_attr = x.new_zeros((edge_index.shape[1], 2))
        return super().aggregate_stage(x, edge_index, layer, edge_attr)

    def forward(self, data):
        first = self.aggregate_stage(data.x, data.message_edge_index1, 1, data.message_edge_attr1)
        return self.aggregate_stage(first, data.message_edge_index2, 2, data.message_edge_attr2)


class JKNetMessageEncoder(MessageReplay, JKNetEncoder):
    def aggregate_stage(self, x, edge_index, layer, edge_attr=None):
        return (torch.nn.functional.leaky_relu(self.conv1(x, edge_index))
                if layer == 1 else self.conv2(x, edge_index))

    def forward(self, data):
        first = self.aggregate_stage(data.x, data.message_edge_index1, 1)
        second = self.aggregate_stage(first, data.message_edge_index2, 2)
        return self.combine_hops(data.x, first, second)


class DirectConcatSatellite(simulation.Satellite_with_Computing):
    aligned_graphsage_policy = True

    def valid_resource_message(self, stage, value):
        return stage == 'layer2' and len(value) == 126 and np.isfinite(value).all()

    def receive_gnn_embedding_message(self, neighbor, stage, version, value):
        if self.valid_resource_message(stage, value):
            super().receive_gnn_embedding_message(neighbor, stage, version, value)

    def construct_gnn_graph(self):
        return {}, []

    def encode_for_GNN(self, nodes, edges):
        fill = self.direct_concat_fault_feature(
            [view[:6] for neighbor in self.neighbors
             for view in [self.fresh_direct_concat_view(neighbor)] if view is not None])
        raw = [view[:6] if view is not None else fill
               for neighbor in self.action_neighbors
               for view in [self.fresh_direct_concat_view(neighbor)]]
        return Data(x=torch.tensor(raw, dtype=torch.float),
                    edge_index=torch.empty((2, 0), dtype=torch.long))


class AlignedDirectConcatEncoder(DirectConcatEncoder):
    aligned_graphsage_policy = True
    satellite_class = DirectConcatSatellite

    @staticmethod
    def shuffle_for_policy(state):
        data, other = state
        order = np.random.permutation(4)
        result = data.clone()
        result.x = data.x[order].clone()
        result.direct_concat_embeddings = data.direct_concat_embeddings[order].clone()
        result.decision_embeddings = data.decision_embeddings.view(4, -1)[order].reshape(1, -1)
        other = np.asarray(other).copy()
        other[:8] = other[:8].reshape(4, 2)[order].reshape(-1)
        return [result, other], order
