#!/usr/bin/env python3

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
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

from torch_geometric.datasets import Planetoid

# dataset = Planetoid(root='/tmp/Cora', name='Cora')
from my_dataset import GasFlow

dataset = GasFlow()

# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='#e6e6e6'), zorder=0)

# nx.draw(tg.utils.to_networkx(dataset[0]), pos=dataset.location_getter(), with_labels=True)
# plt.show()


animator = GraphAnimation('train loss', 'test loss')
my_net = MyNet()

optimizer = torch.optim.Adam(my_net.parameters(), lr=0.01)

train, test = torch.utils.data.random_split(dataset, (110, 30))

print()
for epoch_no in range(300):

    guessed_right_test = 0
    with torch.no_grad():
        test_loss = 0.0
        guessed_right_test = 0
        for X in test:
            target = X.y.max(1)[1]

            criterion = torch.nn.CrossEntropyLoss()
            predicted = my_net(X)
            guessed_right_test += predicted.argmax().item() == target.item()
            loss = criterion(predicted, target)
            test_loss += loss.item()

    for X in train:
        target = X.y.max(1)[1]

        criterion = torch.nn.CrossEntropyLoss()
        predicted = my_net(X)

        loss = criterion(predicted, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    animator.extend_line1([epoch_no], [float(loss)])
    animator.extend_line2([epoch_no], [test_loss])

    up = '\033[1A'
    delete = '\033[K'
    print(delete + f'train loss: {float(loss)}')
    print(delete + f'test loss: {test_loss}')
    print(delete + f'accuracy: {guessed_right_test}/{len(test)}', end=up + up + '\r')

    animator.redraw()
