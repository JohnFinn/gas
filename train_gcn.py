#!/usr/bin/env python3
from typing import Generator
import random
import os
import sys
import datetime as dt
from copy import copy, deepcopy

import click

import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot
import matplotlib
import seaborn as sns

import torch
import torch_geometric as tg

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from mpl_proc import MplProc, ProxyObject

from gf_dataset import GasFlowGraphs
from locations import Coordinates
from models import MyNet3, MyNet2, MyNet, cycle_loss, cycle_dst2
from models import cycle_loss
from report import FigRecord, StringRecord, Reporter

from seed_all import seed_all

from animator import Animator

class LineDrawer:

    def __init__(self, *, ax: matplotlib.axes.Axes, kw_reg, kw_min, kw_train, kw_test):
        self.min_diff = float('inf')
        self.ax = ax
        self.kw_reg = kw_reg
        self.kw_min = kw_min
        self.kw_train = kw_train

        class FakeHline:
            def set(self, *args, **kwargs):
                pass

        self.kw_test = kw_test
        self.min_train_hline, self.min_test_hline = FakeHline(), FakeHline()

    def append(self, *, train_loss: float, test_loss: float):
        crt_diff = abs(test_loss - train_loss)
        if crt_diff < self.min_diff:
            self.min_diff = crt_diff
            self.min_train_hline.set(**self.kw_reg)
            self.min_test_hline.set(**self.kw_reg)
            self.min_train_hline = self.ax.hlines(**self.kw_train, **self.kw_min, y=train_loss)
            self.min_test_hline = self.ax.hlines(**self.kw_test, **self.kw_min, y=test_loss)
        else:
            self.ax.hlines(**self.kw_reg, **self.kw_train, y=train_loss)
            self.ax.hlines(**self.kw_reg, **self.kw_test, y=test_loss)


@click.command()
@click.option('--seed', default=0, help='seed to use everywhere')
@click.option('--epochs', default=500, help='epochs to train')
@click.option('--num-splits', default=20, help='how many dataset splits to try when building baselines')
def train_gcn(seed, epochs, num_splits):

    print("[ Using Seed : ", seed, " ]")
    seed_all(seed)

    mpl_proc = MplProc()

    animator = Animator(mpl_proc)
    graph_dataset = GasFlowGraphs()
    lines = LineDrawer(ax=mpl_proc.proxy_ax,
                       kw_min=dict(),
                       kw_reg=dict(linewidth=0.3, color='gray'),
                       kw_train=dict(linestyle=':', xmin=300, xmax=400),
                       kw_test=dict(xmin=400, xmax=500)
                       )

    for seed in range(num_splits):
        # torch.manual_seed(seed)
        train_graphs, test_graphs = torch.utils.data.random_split(graph_dataset, (len(graph_dataset) - 20, 20))

        decision_tree = DecisionTreeClassifier(min_samples_leaf=6, max_depth=4, max_leaf_nodes=12)
        X = np.concatenate([ g.edge_attr.T for g in train_graphs ])
        y = np.concatenate([ g.y for g in train_graphs ])[:,1]
        decision_tree.fit(X, y)
        predicted = decision_tree.predict(np.concatenate([ g.edge_attr.T for g in test_graphs ]))
        target = np.array([g.y[0,1].item() for g in test_graphs])

        test_loss = cycle_loss(target, predicted)
        train_loss = cycle_loss(y, decision_tree.predict(X))

        if abs(test_loss - train_loss) < lines.min_diff:
            train_loader = tg.data.DataLoader(train_graphs, batch_size=len(train_graphs))
            test_loader = tg.data.DataLoader(test_graphs, batch_size=len(test_graphs))

        lines.append(
            test_loss=test_loss,
            train_loss=train_loss
        )

    lines = LineDrawer(ax=mpl_proc.proxy_ax,
                        kw_min = dict(),
                        kw_reg = dict(linewidth=0.3, color='gray'),
                        kw_train = dict(linestyle=':', xmin=100, xmax=200),
                        kw_test = dict(xmin=200, xmax=300)
            )


    for seed in range(num_splits):
        train_graphs, test_graphs = torch.utils.data.random_split(graph_dataset, (len(graph_dataset) - 20, 20))
        gnb = GaussianNB()
        X = np.concatenate([ g.edge_attr.T for g in train_graphs ])
        y = np.concatenate([ g.y for g in train_graphs ])[:,1]
        gnb.fit(X, y)
        predicted = gnb.predict(np.concatenate([ g.edge_attr.T for g in test_graphs ]))
        target = np.array([g.y[0,1].item() for g in test_graphs])

        lines.append(
            test_loss=cycle_loss(target, predicted),
            train_loss=cycle_loss(y, gnb.predict(X))
        )

    mynet = MyNet3()

# seed_all(seed)
    train_graphs, test_graphs = torch.utils.data.random_split(graph_dataset, (len(graph_dataset) - 20, 20))
    train_loader = tg.data.DataLoader(train_graphs, batch_size=len(train_graphs))
    test_loader = tg.data.DataLoader(test_graphs, batch_size=len(test_graphs))

    optimizer = torch.optim.Adam(mynet.parameters(), lr=0.001)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
# torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    def train_epochs():
        for epoch in range(epochs):
            train_loss = 0
            for batch in train_loader:
                # criterion = torch.nn.MSELoss()
                predicted = mynet(batch)

                loss = cycle_loss(predicted.flatten(), batch.y[:,1].float())
                # loss = criterion(predicted, batch.y.float())
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # lr_scheduler.step()
            train_loss /= len(train_loader)
            yield train_loss



    class IntersectionFinder:

        def __init__(self):
            self.old = (None, None)

        def intersects(self, a: float, b: float) -> bool:
            old_a, old_b = self.old
            self.old = a, b
            if old_a is None:
                return False
            if a == b:
                return True
            return (old_a > old_b) != (a > b)


    intersections = IntersectionFinder()

    min_test_loss = float('inf')
    min_test_epoch = -1
    for epoch_no, train_loss in enumerate(train_epochs()):
        with torch.no_grad():
            test_loss = 0.0
            for batch in test_loader:
                predicted = mynet(batch)

                loss = cycle_loss(predicted.flatten(), batch.y[:,1].float())
                test_loss += loss.item()
            test_loss /= len(test_loader)
            if test_loss < min_test_loss:
                min_test_loss = test_loss
                best = deepcopy(mynet)
                min_test_epoch = epoch_no
            if intersections.intersects(train_loss, test_loss):
                mpl_proc.proxy_ax.scatter(epoch_no, train_loss, s=100, marker='x', color='#3d89be')

        animator.add(train_loss, test_loss)

    fig: matplotlib.figure.Figure
    ax1: matplotlib.axes.Axes
    ax2: matplotlib.axes.Axes
    ax3: matplotlib.axes.Axes
    ax4: matplotlib.axes.Axes
    fig, ((ax1, ax2), (ax3, ax4)) = pyplot.subplots(ncols=2, nrows=2, sharey=True)

    def draw_tables(ax: matplotlib.axes.Axes, net: torch.nn.Module, data: tg.data.DataLoader):
        table = np.full((13, 12), np.nan)
        for batch in data:
            predicted = net(batch)
            Y = batch.y[:,0] - 2008
            M = batch.y[:,1]
            table[Y, M] = cycle_dst2(M.float(), predicted.flatten().detach().numpy()) ** .5

        mshow = ax.matshow(table, vmin=0, vmax=6)
        ax.set(yticks=range(13), yticklabels=range(2008, 2021))
        return mshow

    mshow = draw_tables(ax1, mynet, train_loader)
    draw_tables(ax2, mynet, test_loader)
    draw_tables(ax3, best, train_loader)
    draw_tables(ax4, best, test_loader)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(mshow, cax=cbar_ax)

    ax1.title.set_text('last')
    ax3.title.set_text(f'best {min_test_epoch}')


    def nxt_num() -> int:
        return sum((
            1
            for n in os.listdir('experiments')
            if n.startswith('exp-1')
        )) + 1

    N = nxt_num()


    reporter = Reporter('report4.md')
    reporter.append(StringRecord(f'# {N}'))
    reporter.append(StringRecord(f'''
    ```
    {mynet}
    ```
    '''))
    reporter.append(FigRecord(fig, 'exp-2', f'experiments/exp-2-{N}.png'))
    reporter.append(FigRecord(mpl_proc.proxy_fig, 'exp-1', f'experiments/exp-1-{N}.png'))

    reporter.write()

    pyplot.show()

if __name__ == '__main__':
    train_gcn()
