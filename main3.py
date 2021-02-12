#!/usr/bin/env python3

from my_dataset import GasFlowGraphs


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
from mynet import MyNet2


graph_dataset = GasFlowGraphs()
train_graphs, test_graphs = torch.utils.data.random_split(graph_dataset, (120, 20))


train_loader = tg.data.DataLoader(train_graphs, batch_size=len(train_graphs))
test_loader = tg.data.DataLoader(test_graphs, batch_size=len(test_graphs))


animator = GraphAnimation('train loss', 'test loss')
mynet = MyNet2()


with torch.no_grad():

    train_loss = 0.0
    for batch in train_loader:
        criterion = torch.nn.CrossEntropyLoss()
        predicted = mynet(batch)
        predicted_argmax = torch.argmax(predicted, axis=1)

        loss = criterion(predicted, batch.y)
        train_loss += loss.item()

    train_loss /= len(train_graphs)

    test_loss = 0.0
    for batch in test_loader:
        criterion = torch.nn.CrossEntropyLoss()
        predicted = mynet(batch)
        predicted_argmax = torch.argmax(predicted, axis=1)

        loss = criterion(predicted, batch.y)
        test_loss += loss.item()

    test_loss /= len(test_graphs)


    animator.extend_line1([0], [train_loss])
    animator.extend_line2([0], [test_loss])
    animator.redraw()


optimizer = torch.optim.Adam(mynet.parameters(), lr=0.001)

import random

for epoch in range(1000):

    guessed_right_train = 0
    train_loss = 0
    for batch in train_loader:
        criterion = torch.nn.CrossEntropyLoss()
        predicted = mynet(batch)
        predicted_argmax = torch.argmax(predicted, axis=1)
        guessed_right_train += (predicted_argmax == batch.y).sum().item()

        loss = criterion(predicted, batch.y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_graphs)


    guessed_right_test = 0
    with torch.no_grad():
        test_loss = 0.0
        guessed_right_test = 0
        for batch in test_loader:
            criterion = torch.nn.CrossEntropyLoss()
            predicted = mynet(batch)
            predicted_argmax = torch.argmax(predicted, axis=1)
            guessed_right_test += (predicted_argmax == batch.y).sum().item()

            loss = criterion(predicted, batch.y)
            test_loss += loss.item()

        test_loss /= len(test_graphs)

    if epoch % 25 == 0:
        animator.extend_line1([epoch], [train_loss])
        animator.extend_line2([epoch], [test_loss])
        animator.redraw()

    up = '\033[1A'
    delete = '\033[K'
    print(delete + f'epoch: {epoch}')
    print(delete + f'train loss: {train_loss}')
    print(delete + f'train accuracy: {guessed_right_train}/{len(train_graphs)}')
    print(delete + f'test loss: {test_loss}')
    print(delete + f'test accuracy: {guessed_right_test}/{len(test_graphs)}', end=up + up + up + up + '\r')

print('\n\n\n')