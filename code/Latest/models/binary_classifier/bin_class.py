from torch import nn


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(128, 256)
        self.layer_2 = nn.Linear(256, 64)
        self.layer_out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        # self.bn1 = nn.BatchNorm1d(128)
        self.sigm=nn.Sigmoid()

    def forward(self, x):
        # x = self.bn1(x)
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        x = self.relu(x)
        y = self.sigm(self.layer_out(x))
        return y
