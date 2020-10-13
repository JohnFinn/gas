#!/bin/python

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


df : pd.DataFrame = pd.read_excel('Export_GTF_IEA.XLS')
df = df[:195]
verticies : pd.DataFrame = df[['Exit', 'Entry']]
verticies : pd.DataFrame = verticies[(verticies['Exit'] != 'Liquefied Natural Gas') & (verticies['Entry'] != 'Liquefied Natural Gas')]

G = nx.DiGraph()
G.add_edges_from(verticies.to_numpy())

class Coordinates:

    def __init__(self):
        self.df: pd.DataFrame = pd.read_csv('countries.csv', sep='\t')

    def __getitem__(self, name: str) -> (float, float):
        data_point = self.df[self.df['name'] == name]
        if len(data_point) == 0:
            raise KeyError(f"No such country {name}")
        return tuple(data_point[['longitude', 'latitude']].values.reshape(2))

nx.draw(G, pos=Coordinates(), with_labels=True, font_weight='bold')
plt.show()
