#!/usr/bin/env python3
from torch_geometric.data import Data, Batch
from torch_geometric.nn import NNConv, GraphConv, TransformerConv, SplineConv, SGConv, GatedGraphConv, ChebConv
from torch import nn
import torch
import inspect
from torch.functional import F


def cycle_loss(a: torch.Tensor, b: torch.Tensor, mod: int) -> torch.TensorType:
    d = (a-b)
    return ((((d - 6) % 12) - 6) ** 2).mean()

def cycle_dst2(a, b, mod):
    d = (a-b)
    return (((d - 6) % 12) - 6) ** 2


class LambdaLayer(nn.Module):

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, batch):
        return self.func(batch)

    def __repr__(self):
        return inspect.getsource(self.func)


class MyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 1, bias=True),
            LambdaLayer(lambda batch: batch * 11)
        )

    def forward(self, X):
        return self.net(X)


class MyNet2(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = NNConv(1,1, nn.Sequential(
                nn.Linear(1, 4, bias=True),
                nn.ReLU(),
                nn.Linear(4, 1, bias=True)
            ), aggr="add")
        # self.cheb_conv = ChebConv(1,1, 3)
        self.net = nn.Sequential(
            nn.Linear(37, 8, bias=True),
            nn.ReLU(),
            nn.Linear(8, 8, bias=True),
            nn.ReLU(),
            nn.Linear(8, 1, bias=True),
            # LambdaLayer(lambda batch: batch / 1000),
            nn.Sigmoid(),
            LambdaLayer(lambda batch: batch * 11)
        )

    def forward(self, batch: Batch):
        nodes = batch.x
        nodes = self.conv(nodes, batch.edge_index, batch.edge_attr)
        # nodes = self.cheb_conv(nodes, batch.edge_index, batch=batch.batch) # TODO find out what batch.edge_attr breaks
        batch_len = len(batch.batch.unique())
        return self.net(nodes.reshape(batch_len, len(nodes)//batch_len))
