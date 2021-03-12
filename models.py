#!/usr/bin/env python3
from torch_geometric.data import Data, Batch
from torch_geometric.nn import NNConv, GraphConv, TransformerConv, SplineConv, SGConv, GatedGraphConv
from torch import nn
import torch
from torch.functional import F


def cycle_loss(a: torch.Tensor, b: torch.Tensor, mod: int) -> torch.TensorType:
    d = (a-b)
    return ((((d - 6) % 12) - 6) ** 2).mean()

class LambdaLayer(nn.Module):

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, batch):
        return self.func(batch)


class MyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 8, bias=True),
            nn.ReLU(),
            nn.Linear(8, 8, bias=True),
            nn.ReLU(),
            nn.Linear(8, 1, bias=True),
            nn.Sigmoid(),
            LambdaLayer(lambda batch: batch * 11)
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
            ), aggr="add"),
            # SplineConv(1,1,1,2),
            # GatedGraphConv(1,3, aggr="mean")
        ]
        self.net = nn.Sequential(
            nn.Linear(38, 4, bias=True),
            nn.ReLU(),
            nn.Linear(4, 4, bias=True),
            nn.ReLU(),
            nn.Linear(4, 1, bias=True),
            nn.Sigmoid(),
            LambdaLayer(lambda batch: batch * 11)
        )

    def forward(self, batch: Batch):
        nodes = batch.x
        for layer in self.conv:
            nodes = layer(nodes, batch.edge_index, batch.edge_attr)
        return self.net(nodes.reshape(len(batch.batch.unique()), 38))
