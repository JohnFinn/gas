#!/usr/bin/env python3

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from locations import Coordinates


# import torch_geometric as tg

df : pd.DataFrame = pd.read_excel('Export_GTF_IEA.XLS')
df = df[:195]
df : pd.DataFrame = df[(df['Exit'] != 'Liquefied Natural Gas') & (df['Entry'] != 'Liquefied Natural Gas')]

G = nx.DiGraph()
# G.add_edges_from(df[['Exit', 'Borderpoint']].to_numpy())
# G.add_edges_from(df[['Borderpoint', 'Entry']].to_numpy())
G.add_edges_from(df[['Exit', 'Entry']].to_numpy())


ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='#e6e6e6'), zorder=0)

nx.draw(G, pos=Coordinates(), with_labels=True)

plt.show()
