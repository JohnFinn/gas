# from torch_geometric.data import Data, Batch
from torch import nn
from torch.functional import F


class MyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 8, bias=True),
            nn.LeakyReLU(),
            nn.Linear(8, 8, bias=True),
            nn.LeakyReLU(),
            nn.Linear(8, 12, bias=True),
            nn.Softmax(dim=-1)
        )

    def forward(self, X):
        return self.net(X)

