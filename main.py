#!/bin/python

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from locations import Coordinates

df : pd.DataFrame = pd.read_excel('Export_GTF_IEA.XLS')
df = df[:195]
df : pd.DataFrame = df[(df['Exit'] != 'Liquefied Natural Gas') & (df['Entry'] != 'Liquefied Natural Gas')]
verticies : pd.DataFrame = df[['Exit', 'Entry']]

G = nx.DiGraph()
G.add_edges_from(verticies.to_numpy())
G.add_nodes_from(df['Borderpoint'])

m = Basemap()
m.fillcontinents()
nx.draw(G, pos=Coordinates(), with_labels=True)
plt.show()
