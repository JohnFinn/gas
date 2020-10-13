#!/bin/python

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

df : pd.DataFrame = pd.read_excel('Export_GTF_IEA.XLS')
df = df[:195]
df : pd.DataFrame = df[(df['Exit'] != 'Liquefied Natural Gas') & (df['Entry'] != 'Liquefied Natural Gas')]
verticies : pd.DataFrame = df[['Exit', 'Entry']]

G = nx.DiGraph()
G.add_edges_from(verticies.to_numpy())
G.add_nodes_from(df['Borderpoint'])

class Coordinates:

    def __init__(self):
        from locations import locations
        self.locations = locations
        self.countries: pd.DataFrame = pd.read_csv('countries.csv', sep='\t')
        self.cities: pd.DataFrame = pd.read_csv('worldcities.csv')

    def __getitem__(self, name: str) -> (float, float):
        data_point = self.countries[self.countries['name'] == name]
        if len(data_point) != 0:
            return tuple(data_point[['longitude', 'latitude']].values.reshape(2))
        data_point = self.cities[self.cities['city'] == name]
        if len(data_point) != 0:
            return tuple(data_point.iloc[0][['lng', 'lat']].values.reshape(2))
        data_point = self.cities[self.cities['city_ascii'] == name]
        if len(data_point) != 0:
            return tuple(data_point.iloc[0][['lng', 'lat']].values.reshape(2))
        return self.locations[name]

m = Basemap()
m.fillcontinents()
nx.draw(G, pos=Coordinates(), with_labels=True)
plt.show()
