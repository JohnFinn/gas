#!/usr/bin/env python3

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from locations import Coordinates

import torch
from torch import nn
import torch_geometric as tg
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, NNConv, EdgeConv

from torch_geometric.datasets import Planetoid

# dataset = Planetoid(root='/tmp/Cora', name='Cora')
from my_dataset import GasFlow

dataset = GasFlow()


class DenseNet(torch.nn.Module):

    def __init__(self, edge_features: int, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_features, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, in_channels * out_channels)
        )

    def forward(self, edge_attr):
        return self.net(edge_attr)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = NNConv(dataset.num_node_features, 3, nn = DenseNet(1, dataset.num_node_features, 3), root_weight=False)
        self.conv2 = NNConv(3, dataset.num_classes, nn = DenseNet(1, 3, dataset.num_classes), root_weight=False)
        # self.linear = nn.Linear(85176, 12)
        # self.conv1 = EdgeConv(DenseNet(1, dataset.num_node_features, 3))

    def forward(self, data):
        x, edge_index, edge_attr = data.x.unsqueeze(0), data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        # x = self.linear(x.flatten())

        return F.softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device).double()
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss = F.binary_cross_entropy(out[[0]].T, data.y.T.double())
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
# correct = int(pred[0].eq(data.y).sum().item())
# acc = correct / int(data.sum())
# print('Accuracy: {:.4f}'.format(acc))

# df : pd.DataFrame = pd.read_excel('Export_GTF_IEA.XLS')
# df = df[:195]
# df : pd.DataFrame = df[(df['Exit'] != 'Liquefied Natural Gas') & (df['Entry'] != 'Liquefied Natural Gas')]

# G = nx.DiGraph()
# G.add_edges_from(df[['Exit', 'Borderpoint']].to_numpy())
# G.add_edges_from(df[['Borderpoint', 'Entry']].to_numpy())
# G.add_edges_from(df[['Exit', 'Entry']].to_numpy())


# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='#e6e6e6'), zorder=0)



# nx.draw(G, pos=Coordinates(), with_labels=True)
# nx.draw(G.to_networkx(), pos=G.location_getter(), with_labels=True)
# plt.show()
