import glob
import os
from typing import Optional, Callable
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor

DATA_PATH = "./data"
MNIST_PATH = os.path.join(DATA_PATH, "mnist")
CIFAR10_PATH = os.path.join(DATA_PATH, "cifar10")
TARGET_FILE = "targets.pkl"


class CustomImageDataset(Dataset):

    def __init__(self, path: str, transform: Optional[Callable], flatten: bool):
        super(CustomImageDataset, self).__init__()
        target_path = os.path.join(path, TARGET_FILE)
        self.img_paths = sorted(glob.glob(os.path.join(path, "**", "*.png"), recursive=True))
        self.targets = torch.load(target_path)
        self.transform = transform
        self.flatten = flatten

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.FloatType]:
        img = Image.open(self.img_paths[idx])
        target = self.targets[idx]

        if self.transform:
            img = self.transform(img)

        if self.flatten:
            img = img.reshape(-1)

        return img, target


class MNISTDataset(CustomImageDataset):

    def __init__(self, transform: Optional[Callable] = ToTensor(), flatten: bool = False, train: bool = True):
        directory = "train" if train else "test"
        path = os.path.join(MNIST_PATH, directory)

        if not os.path.exists(path):
            self.load_mnist_data(path, train)

        super().__init__(path=path, transform=transform, flatten=flatten)

    @staticmethod
    def load_mnist_data(path: str, train: bool):
        os.makedirs(path, exist_ok=True)
        dataset = MNIST(MNIST_PATH, train=train, download=True)

        targets = dataset.targets
        torch.save(targets, os.path.join(path, TARGET_FILE))

        for idx, data in enumerate(dataset):
            img, _ = data
            img.save(os.path.join(path, f"img_{idx:08d}.png"))


class CIFAR10Dataset(CustomImageDataset):

    def __init__(self, transform: Optional[Callable] = ToTensor(), train: bool = True, flatten: bool = False):
        directory = "train" if train else "test"
        path = os.path.join(CIFAR10_PATH, directory)

        if not os.path.exists(path):
            self.load_cifar10_data(path, train)

        super().__init__(path=path, transform=transform, flatten=flatten)

    @staticmethod
    def load_cifar10_data(path: str, train: bool):
        os.makedirs(path, exist_ok=True)
        dataset = CIFAR10(CIFAR10_PATH, train=train, download=True)

        targets = dataset.targets
        torch.save(targets, os.path.join(path, TARGET_FILE))

        for idx, data in enumerate(dataset):
            img, _ = data
            img.save(os.path.join(path, f"img_{idx:08d}.png"))
