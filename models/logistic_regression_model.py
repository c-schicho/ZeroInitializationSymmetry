import torch
from torch.nn import Module, Linear


class LogisticRegression(Module):

    def __init__(self, input_dim: int, n_classes: int):
        super(LogisticRegression, self).__init__()
        self.fnn1 = Linear(input_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fnn1(x)
