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

ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='#e6e6e6'), zorder=0)

nx.draw(tg.utils.to_networkx(dataset[0]), pos=dataset.location_getter(), with_labels=True)
plt.show()
