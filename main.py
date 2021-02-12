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
from mynet import MyNet
from anim_plot import GraphAnimation

import torch
from torch import nn
import torch_geometric as tg
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, NNConv, EdgeConv

# from torch_geometric.datasets import Planetoid

# dataset = Planetoid(root='/tmp/Cora', name='Cora')
from my_dataset import GasFlow, GasFlowGraphs



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
exit()

pca = PCA(n_components=100)
data = dataset.df[[c for c in dataset.df.columns if isinstance(c, dt.datetime)]].astype(float).T
pca.fit(data)


animator = GraphAnimation('train loss', 'test loss')
my_net = MyNet()

optimizer = torch.optim.Adam(my_net.parameters(), lr=0.01)


train_loader = torch.utils.data.DataLoader(train, 64)
test_loader = torch.utils.data.DataLoader(test, 10)

for epoch_no in range(256):

    guessed_right_test = 0
    with torch.no_grad():
        test_loss = 0.0
        guessed_right_test = 0
        for X, target in test_loader:

            criterion = torch.nn.CrossEntropyLoss()
            transformed = torch.tensor(pca.transform(X), dtype=torch.float32)

            predicted = my_net(transformed)
            predicted_argmax = torch.argmax(predicted, axis=1)

            guessed_right_test += (predicted_argmax == target).sum().item()
            loss = criterion(predicted, target)
            test_loss += loss.item()

        test_loss /= len(test)

    guessed_right_train = 0
    train_loss = 0
    for X, target in train_loader:
        criterion = torch.nn.CrossEntropyLoss()
        transformed = torch.tensor(pca.transform(X), dtype=torch.float32)
        predicted = my_net(transformed)
        predicted_argmax = torch.argmax(predicted, axis=1)
        
        guessed_right_train += (predicted_argmax == target).sum().item()

        loss = criterion(predicted, target)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train)


    animator.extend_line1([epoch_no], [train_loss])
    animator.extend_line2([epoch_no], [test_loss])
    animator.redraw()

    up = '\033[1A'
    delete = '\033[K'
    print(delete + f'train loss: {train_loss}')
    print(delete + f'train accuracy: {guessed_right_train}/{len(train)}')
    print(delete + f'test loss: {test_loss}')
    print(delete + f'test accuracy: {guessed_right_test}/{len(test)}', end=up + up + up + '\r')

print('\n\n')
