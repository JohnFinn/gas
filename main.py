#!/usr/bin/env python3

import pandas as pd
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import datetime as dt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from locations import Coordinates
from models import MyNet
from anim_plot import AnimPlot

import torch
from torch import nn
import torch_geometric as tg
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, NNConv, EdgeConv

from gf_dataset import GasFlow, GasFlowGraphs


dataset = GasFlow()
graph_dataset = GasFlowGraphs()
train, test = torch.utils.data.random_split(dataset, (120, 20))
train_graph, test_graph = torch.utils.data.random_split(graph_dataset, (120, 20))


ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='#e6e6e6'), zorder=0)

G = tg.utils.to_networkx(graph_dataset[0])
# G = nx.relabel_nodes(G, graph_dataset.dataset_.country_by_idx)
nx.draw(G, pos=graph_dataset.dataset_.location_getter(), labels=dataset.country_by_idx, with_labels=True)
plt.show()

pca = PCA(n_components=100)
data = dataset.df[[c for c in dataset.df.columns if isinstance(c, dt.datetime)]].astype(float).T
pca.fit(data)


animator = AnimPlot('train loss', 'test loss')
my_net = MyNet()

optimizer = torch.optim.Adam(my_net.parameters(), lr=0.01)


train_loader = torch.utils.data.DataLoader(train, 64)
test_loader = torch.utils.data.DataLoader(test, 10)

for epoch_no in range(256):

    with torch.no_grad():
        test_loss = 0.0
        for X, target in test_loader:

            transformed = torch.tensor(pca.transform(X), dtype=torch.float32)

            predicted = my_net(transformed)

            test_loss += loss.item()

        test_loss /= len(test)

    train_loss = 0
    for X, target in train_loader:
        transformed = torch.tensor(pca.transform(X), dtype=torch.float32)
        predicted = my_net(transformed)

        loss = criterion(predicted, target)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train)


    animator.add(train_loss, test_loss)
