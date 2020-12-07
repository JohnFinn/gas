# from torch_geometric.data import Data, Batch
from torch import nn
from torch.functional import F


class MyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16, bias=True),
            nn.Tanh(),
            nn.Linear(16, 8, bias=True),
            nn.Tanh(),
            nn.Linear(8, 12, bias=True),
            nn.Softmax()
        )

    def forward(self, X):
        return self.net(X)

