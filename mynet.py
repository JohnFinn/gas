#!/usr/bin/env python3
from torch_geometric.data import Data, Batch
from torch_geometric.nn import NNConv, GraphConv, TransformerConv, SplineConv, SGConv, GatedGraphConv
from torch import nn
from torch.functional import F


class MyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 8, bias=True),
            nn.LeakyReLU(),
            nn.Linear(8, 8, bias=True),
            nn.LeakyReLU(),
            nn.Linear(8, 12, bias=True),
            nn.Softmax(dim=-1)
        )

    def forward(self, X):
        return self.net(X)


class MyNet2(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = [
            # GraphConv(1,1, aggr="mean"),
            # TransformerConv(1,1, edge_dim=1, aggr="mean"),
            NNConv(1,1, nn.Sequential(
                nn.Linear(1,1)
            )),
            # SplineConv(1,1,1,2),
            # GatedGraphConv(1,3, aggr="mean")
        ]
        self.net = nn.Sequential(
            nn.Linear(38, 4, bias=True),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(4, 4, bias=True),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(4, 12, bias=True),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Softmax(dim=-1)
        )

    def forward(self, batch: Batch):
        nodes = batch.x
        for layer in self.conv:
            nodes = layer(nodes, batch.edge_index, batch.edge_attr)
        return self.net(nodes.reshape(len(batch.batch.unique()), 38))


class MyNet3(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = TransformerConv(1,1, edge_dim=1)

    def forward(self, batch: Batch):
        pass