from datetime import datetime
import pandas as pd
import numpy as np
import torch

from locations import Coordinates

from torch.utils.data import Dataset
# from torch_geometric.data.dataset import Dataset
# import torch_geometric
# from torch_geometric.data import Data

class GasFlow(Dataset):

    num_classes = 12

    def __init__(self):
        df : pd.DataFrame = pd.read_excel('Export_GTF_IEA.XLS')
        df = df.replace('#N/A()', None)
        df = df[:195].dropna()
        self.df : pd.DataFrame = df[(df['Exit'] != 'Liquefied Natural Gas') & (df['Entry'] != 'Liquefied Natural Gas')]

        self.coordinates : Coordinates = Coordinates()

        self.idx_by_country = {name: idx for idx, name in enumerate(set(self.df['Entry']) | set(self.df['Exit']))}
        self.country_by_idx = {idx: name for name, idx in self.idx_by_country.items()}

        self.possible_dates = [col for col in self.df if isinstance(col, datetime)]

        self.edge_index = torch.tensor(self.df[['Exit', 'Entry']].replace(self.idx_by_country).values.T, dtype=torch.long)


    # def __getitem__(self, idx: int) -> Data:
    #     date = self.possible_dates[idx]
    #     one_hot_encoded_month = torch.zeros(1, 12, dtype=torch.long)
    #     one_hot_encoded_month[0, date.month] = 1
    #     return Data(
    #         edge_index=self.edge_index,
    #         edge_attr=torch.tensor(self.df[date].replace('#N/A()', -1).values[np.newaxis].T),
    #         x=torch.rand(len(self.idx_by_country), 1, dtype=torch.float), # no node features for now
    #         y=one_hot_encoded_month
    #     )
    
    def __getitem__(self, idx: int):
        date = self.possible_dates[idx]
        one_hot_encoded_month = torch.zeros(1, 12, dtype=torch.long)
        one_hot_encoded_month[0, date.month] = 1
        return torch.tensor(self.df[date].astype(float).values[np.newaxis].T), one_hot_encoded_month


    def __len__(self) -> int:
        return len(self.possible_dates)


    def location_getter(self):
        class Getter:
            def __getitem__(_not_this_self, idx: int):
                return self.coordinates.get_location(self.country_by_idx[idx])
        return Getter()
