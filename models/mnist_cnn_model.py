import torch
from torch.nn import Module, Sequential, Conv2d, ReLU, Linear

from utils import layer_initialization


class MNISTCNNModel(Module):

    def __init__(self):
        super(MNISTCNNModel, self).__init__()
        self.cnn1 = Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.cnn2 = Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.fnn = Linear(32 * 28 * 28, 10)
        self.act = ReLU()

        self.cnn = Sequential(
            self.cnn1,
            self.act,
            self.cnn2,
            self.act
        )

    def zero_initialization(self, mode: str = 'normal'):
        layer_initialization.zero_initialize_layer(self.cnn1, mode)
        layer_initialization.zero_initialize_layer(self.cnn2, mode)
        layer_initialization.zero_initialize_layer(self.fnn, mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.view(x.size(dim=0), -1)
        x = self.fnn(x)
        return x

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))
