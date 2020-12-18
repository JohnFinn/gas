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
from my_dataset import GasFlow

dataset = GasFlow()

# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='#e6e6e6'), zorder=0)

# nx.draw(tg.utils.to_networkx(dataset[0]), pos=dataset.location_getter(), with_labels=True)
# plt.show()

pca = PCA(n_components=4)
data = dataset.df[[c for c in dataset.df.columns if isinstance(c, dt.datetime)]].astype(float).T
pca.fit(data)


animator = GraphAnimation('train loss', 'test loss')
my_net = MyNet()

optimizer = torch.optim.Adam(my_net.parameters(), lr=0.01)

train, test = torch.utils.data.random_split(dataset, (120, 20))

batch_size=10
train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size, shuffle=True)

for epoch_no in range(100):

    guessed_right_test = 0
    with torch.no_grad():
        test_loss = 0.0
        guessed_right_test = 0
        for X, y in test_loader:
            target = torch.argmax(y, axis=2)

            criterion = torch.nn.CrossEntropyLoss()
            transformed = torch.tensor([pca.transform(x.T) for x in X], dtype=torch.float32)

            predicted = my_net(transformed)
            predicted_argmax = torch.argmax(predicted, axis=2)

            guessed_right_test += (predicted_argmax == target).sum().item()
            loss = criterion(predicted.reshape(test_loader.batch_size,12), torch.argmax(y, axis=2).flatten())
            test_loss += loss.item()

        test_loss /= len(test)

    guessed_right_train = 0
    train_loss = 0
    for X, y in train_loader:
        target = torch.argmax(y, axis=2)
        criterion = torch.nn.CrossEntropyLoss()
        transformed = torch.tensor([pca.transform(x.T) for x in X], dtype=torch.float32)
        predicted = my_net(transformed)
        predicted_argmax = torch.argmax(predicted, axis=2)
        
        guessed_right_train += (predicted_argmax == target).sum().item()

        loss = criterion(predicted.reshape(train_loader.batch_size,12), torch.argmax(y, axis=2).flatten())
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