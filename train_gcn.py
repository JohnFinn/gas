#!/usr/bin/env python3
from gf_dataset import GasFlowGraphs
import pandas as pd
import networkx as nx
import torch
from torch import nn
import torch_geometric as tg
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, NNConv, EdgeConv
import numpy as np
import os
from copy import copy, deepcopy
from sklearn.decomposition import PCA
from matplotlib import pyplot
import matplotlib
import seaborn as sns
import datetime as dt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from locations import Coordinates
from anim_plot import AnimPlot
from draw_tables import draw_tables
from models import MyNet2, MyNet, cycle_loss, cycle_dst2


graph_dataset = GasFlowGraphs()
torch.manual_seed(1) # reproducibility
mynet = MyNet2()
train_graphs, test_graphs = torch.utils.data.random_split(graph_dataset, (len(graph_dataset) - 20, 20))

train_loader = tg.data.DataLoader(train_graphs, batch_size=len(train_graphs))
test_loader = tg.data.DataLoader(test_graphs, batch_size=len(test_graphs))

optimizer = torch.optim.Adam(mynet.parameters(), lr=0.001)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
# optimizer = torch.optim.Adam(mynet.parameters(), lr=0.001)


def nxt_num() -> int:
    return sum((
        1
        for n in os.listdir('experiments')
        if n.startswith('exp-1')
    )) + 1

N = nxt_num()

animator = AnimPlot('train loss', 'test loss', output=f'experiments/exp-1-{N}.png')

def train_epochs():
    for epoch in range(10000):
        train_loss = 0
        for batch in train_loader:
            # criterion = torch.nn.MSELoss()
            predicted = mynet(batch)

            loss = cycle_loss(predicted.flatten(), batch.y[:,1].float(), 12)
            # loss = criterion(predicted, batch.y.float())
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
        train_loss /= len(train_loader)
        yield train_loss

min_test_loss = float('inf')
min_test_epoch = -1
for epoch_no, train_loss in enumerate(train_epochs()):
    with torch.no_grad():
        test_loss = 0.0
        for batch in test_loader:
            predicted = mynet(batch)

            loss = cycle_loss(predicted.flatten(), batch.y[:,1].float(), 12)
            test_loss += loss.item()
        test_loss /= len(test_loader)
        if test_loss < min_test_loss:
            min_test_loss = test_loss
            best = deepcopy(mynet)
            min_test_epoch = epoch_no

    animator.add(train_loss, test_loss)


fig: matplotlib.figure.Figure
ax1: matplotlib.axes.Axes
ax2: matplotlib.axes.Axes
ax3: matplotlib.axes.Axes
ax4: matplotlib.axes.Axes
fig, ((ax1, ax2), (ax3, ax4)) = pyplot.subplots(ncols=2, nrows=2, sharey=True)

draw_tables(fig, ax1, ax2, mynet, train_loader, test_loader)
draw_tables(fig, ax3, ax4, best, train_loader, test_loader)

ax1.title.set_text('last')
ax3.title.set_text(f'best {min_test_epoch}')

pyplot.legend()

fig.savefig(f'experiments/exp-2-{N}.png')
# TODO добавить вручную в этот файл информацию о том, что изменилось
with open(f'experiments/report3.md', 'at') as f:
    f.write(
f'''
# {N}
```
{mynet}
```
![exp-1](exp-1-{N}.png)
![exp-2](exp-2-{N}.png)


''')

pyplot.show()
animator.close()