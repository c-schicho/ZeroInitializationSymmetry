from dataclasses import dataclass
from typing import Optional, Callable

from torchvision.transforms import ToTensor


@dataclass
class TrainerConfig:
    model_name: str
    epochs: int
    batch_size: int
    initialization_mode: str
    initialization_factor: float = 1.
    optimizer: Optional = None
    transform_test: Optional[Callable] = ToTensor()
    transform_train: Optional[Callable] = ToTensor()
    activation_fun: Optional = None
