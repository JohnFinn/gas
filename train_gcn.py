#!/usr/bin/env python3
from gf_dataset import GasFlowGraphs

import pandas as pd
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import datetime as dt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from locations import Coordinates
from anim_plot import AnimPlot

import torch
from torch import nn
import torch_geometric as tg
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, NNConv, EdgeConv
from models import MyNet2, MyNet, cycle_loss


graph_dataset = GasFlowGraphs()
train_graphs, test_graphs = torch.utils.data.random_split(graph_dataset, (120, 20))


train_loader = tg.data.DataLoader(train_graphs, batch_size=len(train_graphs))
test_loader = tg.data.DataLoader(test_graphs, batch_size=len(test_graphs))


animator = AnimPlot('train loss', 'test loss')
mynet = MyNet2()

optimizer = torch.optim.Adam(mynet.parameters(), lr=0.001)

for epoch in range(1000):

    guessed_right_train = 0
    train_loss = 0
    for batch in train_loader:
        # criterion = torch.nn.MSELoss()
        predicted = mynet(batch)

        loss = cycle_loss(predicted.flatten(), batch.y.float(), 12)
        # loss = criterion(predicted, batch.y.float())
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    guessed_right_test = 0
    with torch.no_grad():
        test_loss = 0.0
        guessed_right_test = 0
        for batch in test_loader:
            predicted = mynet(batch)

            loss = cycle_loss(predicted.flatten(), batch.y.float(), 12)
            test_loss += loss.item()

    animator.add(train_loss, test_loss)


animator.process.stdin.close()
