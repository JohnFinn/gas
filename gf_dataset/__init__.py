from datetime import datetime
import pandas as pd
import numpy as np
import torch
from pathlib import Path

from locations import Coordinates

from torch.utils.data import Dataset
from torch_geometric.data.dataset import Dataset as tgDataset
import torch_geometric
from torch_geometric.data import Data

# https://www.iea.org/reports/gas-trade-flows

class GasFlow(Dataset):

    num_classes = 12

    def __init__(self):
        df : pd.DataFrame = pd.read_excel(Path(__file__).with_name('Export_GTF_IEA-leaked.xls'))
        df = df.replace('#N/A()', None)
        df = df[:195].dropna()
        self.df : pd.DataFrame = df[(df['Exit'] != 'Liquefied Natural Gas') & (df['Entry'] != 'Liquefied Natural Gas')]

        self.coordinates : Coordinates = Coordinates()

        self.idx_by_country = {name: idx for idx, name in enumerate(set(self.df['Entry']) | set(self.df['Exit']))}
        self.country_by_idx = {idx: name for name, idx in self.idx_by_country.items()}

        self.possible_dates = [col for col in self.df if isinstance(col, datetime)]

        self.edge_index = torch.tensor(self.df[['Exit', 'Entry']].replace(self.idx_by_country).values.T, dtype=torch.long)


    def __getitem__(self, idx: int):
        date = self.possible_dates[idx]
        return torch.tensor(self.df[date].astype(float).values), date.month - 1


    def __len__(self) -> int:
        return len(self.possible_dates)


    def location_getter(self):
        class Getter:
            def __getitem__(_not_this_self, idx: int):
                return self.coordinates.get_location(self.country_by_idx[idx])
        return Getter()


class GasFlowGraphs(tgDataset):

    def __init__(self):
        super().__init__()
        self.dataset_ = GasFlow()

    def __len__(self):
        return len(self.dataset_)

    def __getitem__(self, idx: int) -> Data:
        date = self.dataset_.possible_dates[idx]
        return Data(
            edge_index=self.dataset_.edge_index,
            edge_attr=torch.tensor(self.dataset_.df[date].replace('#N/A()', -1).values[np.newaxis].T.astype(float), dtype=torch.float),
            x=torch.ones(len(self.dataset_.idx_by_country),1, dtype=torch.float), # no node features
            y=torch.tensor([[date.year, (date.month - 1)]])
        )
