import torch
from torch.nn import Module, Linear, ReLU, Sequential

from utils import layer_initialization


class MNISTFNNModel(Module):

    def __init__(self):
        super(MNISTFNNModel, self).__init__()
        self.fc1 = Linear(28 * 28, 500)
        self.fc2 = Linear(500, 10)
        self.act = ReLU(True)

        self.model = Sequential(
            self.fc1,
            self.act,
            self.fc2
        )

    def zero_initialization(self, mode: str = "normal"):
        layer_initialization.zero_initialize_layer(self.fc1, mode)
        layer_initialization.zero_initialize_layer(self.fc2, mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))
