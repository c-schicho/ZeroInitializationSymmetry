from typing import Union

import torch
from torch.nn import Linear, Conv2d, Parameter


def zero_initialize_layer(layer: Union[Linear, Conv2d], mode: str):
    layer.weight = Parameter(torch.zeros_like(layer.weight, requires_grad=True))
    if mode == "normal":
        layer.bias = Parameter(torch.randn_like(layer.bias, requires_grad=True))
    elif mode == "uniform":
        layer.bias = Parameter(torch.rand_like(layer.bias, requires_grad=True))
    elif mode == "zero":
        layer.bias = Parameter(torch.zeros_like(layer.bias, requires_grad=True))
    else:
        raise ValueError(f"mode {mode} is not supported")
