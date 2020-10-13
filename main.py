#!/bin/python

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


df : pd.DataFrame = pd.read_excel('Export_GTF_IEA.XLS')
df = df[:195]
verticies : pd.DataFrame = df[['Exit', 'Entry']]

G = nx.DiGraph()
G.add_edges_from(verticies.to_numpy())

nx.draw(G, with_labels=True, font_weight='bold')
plt.show()
