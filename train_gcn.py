#!/usr/bin/env python3
from gf_dataset import GasFlowGraphs
from locations import Coordinates
from models import MyNet2, MyNet, cycle_loss, cycle_dst2

import numpy as np
import pandas as pd
import networkx as nx
import os
from copy import copy, deepcopy
from matplotlib import pyplot
import matplotlib
import seaborn as sns
import datetime as dt

import torch
import torch_geometric as tg

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

from anim_plot import AnimPlot


graph_dataset = GasFlowGraphs()
torch.manual_seed(1) # reproducibility
mynet = MyNet2()

# good seeds: 4, 6,
torch.manual_seed(100)
train_graphs, test_graphs = torch.utils.data.random_split(graph_dataset, (len(graph_dataset) - 20, 20))
# split = [(y.item(), m.item()) for (y, m), in map(lambda g: g.y, test_graphs)]

train_loader = tg.data.DataLoader(train_graphs, batch_size=len(train_graphs))
test_loader = tg.data.DataLoader(test_graphs, batch_size=len(test_graphs))

optimizer = torch.optim.Adam(mynet.parameters(), lr=0.001)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
# torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
# optimizer = torch.optim.Adam(mynet.parameters(), lr=0.001)


def nxt_num() -> int:
    return sum((
        1
        for n in os.listdir('experiments')
        if n.startswith('exp-1')
    )) + 1

N = nxt_num()

animator = AnimPlot('train loss', 'test loss', output=f'experiments/exp-1-{N}.png')

descision_tree = DecisionTreeClassifier(min_samples_leaf=6)
X = np.concatenate([ g.edge_attr.T for g in train_graphs ])
y = np.concatenate([ g.y for g in train_graphs ])[:,1]
descision_tree.fit(X, y)
predicted = descision_tree.predict(np.concatenate([ g.edge_attr.T for g in test_graphs ]))
target = np.array([g.y[0,1].item() for g in test_graphs])

tree_test_loss = cycle_loss(target, predicted, 12)
tree_train_loss = cycle_loss(y, descision_tree.predict(X), 12)
print('tree train loss', tree_train_loss)
print('tree test loss', tree_test_loss)


def train_epochs():
    for epoch in range(500):
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

def draw_tables(fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes, net: torch.nn.Module, data: tg.data.DataLoader):
    table = np.full((13, 12), np.nan)
    for batch in data:
        predicted = net(batch)
        Y = batch.y[:,0] - 2008
        M = batch.y[:,1]
        table[Y, M] = cycle_dst2(M.float(), predicted.flatten().detach().numpy(), 12) ** .5

    mshow = ax.matshow(table, vmin=0, vmax=6)
    ax.set_yticks(range(13))
    ax.set_yticklabels(range(2008, 2021))
    return mshow

mshow = draw_tables(fig, ax1, mynet, train_loader)
draw_tables(fig, ax2, mynet, test_loader)
draw_tables(fig, ax3, best, train_loader)
draw_tables(fig, ax4, best, test_loader)
# draw_tables(fig, ax5, lambda data: descision_tree.predict(data))

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(mshow, cax=cbar_ax)

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


class ReportWriter:

    def __init__(self, filename: str):
        self.filename = filename
        self.records = []

    def write(self):
        with open(self.filename, 'at') as f:
            f.write('\n'.join(map(lambda record: record.to_markdown(), self.records)) + '\n\n')


pyplot.show()
animator.close()
