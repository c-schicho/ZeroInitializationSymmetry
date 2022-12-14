import torch
from torch.nn import Module, Sequential, Conv2d, MaxPool2d, BatchNorm2d, Flatten, Linear, ELU

from utils import zero_initialize_layer


class CIFARCNNModel(Module):

    def __init__(self, activation_fun=None):
        super(CIFARCNNModel, self).__init__()
        self.cnn1 = Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.cnn2 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.cnn3 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.cnn4 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.cnn5 = Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.cnn6 = Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fnn1 = Linear(256 * 4 * 4, 1024)
        self.fnn2 = Linear(1024, 512)
        self.fnn3 = Linear(512, 10)
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.act = activation_fun if activation_fun is not None else ELU(inplace=True)

        self.model = Sequential(
            self.cnn1,
            self.act,
            self.cnn2,
            self.act,
            self.pool,
            BatchNorm2d(num_features=self.cnn2.out_channels),

            self.cnn3,
            self.act,
            self.cnn4,
            self.act,
            self.pool,
            BatchNorm2d(num_features=self.cnn4.out_channels),

            self.cnn5,
            self.act,
            self.cnn6,
            self.act,
            self.pool,
            BatchNorm2d(num_features=self.cnn6.out_channels),

            Flatten(),
            self.fnn1,
            self.act,
            self.fnn2,
            self.act,
            self.fnn3
        )

    def zero_initialization(self, mode: str, factor: float = 1.):
        for layer in self.model:
            if isinstance(layer, Linear) or isinstance(layer, Conv2d):
                zero_initialize_layer(layer, mode, factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))
