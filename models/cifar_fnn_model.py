import torch
from torch.nn import Module, Linear, Sequential, ReLU

from utils import zero_initialize_layer


class CIFARFNNModel(Module):

    def __init__(self):
        super(CIFARFNNModel, self).__init__()
        self.fnn1 = Linear(32 * 32 * 3, 512)
        self.fnn2 = Linear(512, 256)
        self.fnn3 = Linear(256, 128)
        self.fnn4 = Linear(128, 10)
        self.act = ReLU(True)

        self.model = Sequential(
            self.fnn1,
            self.act,
            self.fnn2,
            self.act,
            self.fnn3,
            self.act,
            self.fnn4
        )

    def zero_initialization(self, mode: str, factor: float):
        for layer in self.model:
            if isinstance(layer, Linear):
                zero_initialize_layer(layer, mode, factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))
