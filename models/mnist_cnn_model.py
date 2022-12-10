import torch
from torch.nn import Module, Sequential, Conv2d, Linear, Flatten, ELU

from utils import zero_initialize_layer


class MNISTCNNModel(Module):

    def __init__(self, activation_fun=None):
        super(MNISTCNNModel, self).__init__()
        self.cnn1 = Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.cnn2 = Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.fnn = Linear(32 * 28 * 28, 10)
        self.act = activation_fun if activation_fun is not None else ELU(inplace=True)

        self.model = Sequential(
            self.cnn1,
            self.act,

            self.cnn2,
            self.act,

            Flatten(),
            self.fnn1
        )

    def zero_initialization(self, mode: str, factor: float):
        for layer in self.model:
            if isinstance(layer, Linear) or isinstance(layer, Conv2d):
                zero_initialize_layer(layer, mode, factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))
