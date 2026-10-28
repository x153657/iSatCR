"""Aggregation/embedding variants for the iSatCR ablation experiments."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, SAGEConv


class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_dim, out_channels):
        super().__init__()
        self.output_dim = out_channels
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')

    def forward(self, data):
        x = F.leaky_relu(self.conv1(data.x, data.edge_index))
        return self.conv2(x, data.edge_index)


class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_dim, out_channels):
        super().__init__()
        self.output_dim = out_channels
        self.conv1 = GATConv(
            in_channels, hidden_channels, heads=1, concat=False, edge_dim=edge_dim
        )
        self.conv2 = GATConv(
            hidden_channels, out_channels, heads=1, concat=False, edge_dim=edge_dim
        )

    def forward(self, data):
        x = F.leaky_relu(self.conv1(data.x, data.edge_index, data.edge_attr))
        return self.conv2(x, data.edge_index, data.edge_attr)

    def aggregate_stage(self, x, edge_index, layer, edge_attr):
        if layer == 1:
            return F.leaky_relu(self.conv1(x, edge_index, edge_attr))
        return self.conv2(x, edge_index, edge_attr)


class JKNetEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_dim, out_channels):
        super().__init__()
        self.output_dim = out_channels
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        self.message_dim = hidden_channels
        self.output_projection = nn.Linear(in_channels + hidden_channels * 2, out_channels)

    def combine_hops(self, local, layer1, layer2):
        return self.output_projection(torch.cat([local, layer1, layer2], dim=-1))

    def forward(self, data):
        layer1 = F.leaky_relu(self.conv1(data.x, data.edge_index))
        layer2 = self.conv2(layer1, data.edge_index)
        return self.combine_hops(data.x, layer1, layer2)


class DirectConcatEncoder(nn.Module):
    """Return each neighbor's cached self/1-hop/2-hop raw-feature view."""

    def __init__(self, in_channels, hidden_channels, edge_dim, out_channels):
        super().__init__()
        self.raw_feature_dim = in_channels
        self.output_dim = in_channels * (1 + 4 + 16)
        self.direct_concat = True

    def forward(self, data):
        return data.direct_concat_embeddings


ENCODERS = {
    'graphsage': GraphSAGEEncoder,
    'gat': GATEncoder,
    'jknet': JKNetEncoder,
    'direct_concat': DirectConcatEncoder,
}


def make_encoder_factory(name):
    if name not in ENCODERS:
        choices = ', '.join(ENCODERS)
        raise ValueError(f'Unknown encoder {name!r}; choose one of: {choices}')

    encoder = ENCODERS[name]

    def factory(in_channels, hidden_channels, edge_dim, out_channels):
        if name == 'graphsage':
            # Import lazily: the replay encoder extends the base class above.
            from ablation.train_graphsage import GraphSAGEEncoder as ReplayGraphSAGEEncoder
            return ReplayGraphSAGEEncoder(in_channels, hidden_channels, edge_dim, out_channels)
        from ablation.message_aggregation import GATMessageEncoder, JKNetMessageEncoder, AlignedDirectConcatEncoder
        aligned = {'gat': GATMessageEncoder, 'jknet': JKNetMessageEncoder,
                   'direct_concat': AlignedDirectConcatEncoder}
        return aligned[name](in_channels, hidden_channels, edge_dim, out_channels)

    return factory
