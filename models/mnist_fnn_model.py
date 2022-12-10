import torch
from torch.nn import Module, Linear, Sequential, ELU

from utils import zero_initialize_layer


class MNISTFNNModel(Module):

    def __init__(self, activation_fun=None):
        super(MNISTFNNModel, self).__init__()
        self.fnn1 = Linear(28 * 28, 500)
        self.fnn2 = Linear(500, 10)
        self.act = activation_fun if activation_fun is not None else ELU(inplace=True)

        self.model = Sequential(
            self.fnn1,
            self.act,
            self.fnn2
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
