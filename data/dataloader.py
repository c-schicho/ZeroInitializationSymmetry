import random
from typing import Optional, Callable

import numpy as np
import torch
from torch import Generator
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from data.datasets import MNISTDataset, CIFAR10Dataset


def get_dataloader(
        dataset: str,
        train: bool,
        flatten: bool = False,
        batch_size: int = 16,
        shuffle: bool = False,
        num_workers: int = 1,
        transform: Optional[Callable] = ToTensor(),
        worker_init_fn: Optional[Callable] = None,
        generator: Optional[Generator] = None
) -> DataLoader:
    if dataset == "mnist":
        dataset = MNISTDataset(transform=transform, flatten=flatten, train=train)
    elif dataset == "cifar10":
        dataset = CIFAR10Dataset(transform=transform, flatten=flatten, train=train)
    else:
        raise ValueError(f"dataset {dataset} is not supported")

    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                      worker_init_fn=worker_init_fn, generator=generator)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
