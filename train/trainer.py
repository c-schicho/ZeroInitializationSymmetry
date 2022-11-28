import os.path
from typing import Optional

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import write_train_summary


class Trainer:

    def __init__(self, model, optimizer=None, loss_fun=None, writer: SummaryWriter = None, force_cpu: bool = False,
                 model_name: Optional[str] = None):
        self.device = torch.device('cuda' if (torch.cuda.is_available() and not force_cpu) else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer if optimizer is not None else Adam(self.model.parameters(), lr=0.001)
        self.loss_fun = loss_fun if loss_fun is not None else CrossEntropyLoss()
        writer_path = os.path.join("results", self.model.__class__.__name__) \
            if model_name is None else os.path.join("results", model_name)
        self.writer = writer if writer is not None else SummaryWriter(writer_path)

    def train(self, dataloader: DataLoader, num_epochs: int = 10):
        step = 0
        self.model.train()

        for epoch in range(1, num_epochs + 1):
            for data_batch in tqdm(dataloader, total=len(dataloader), ncols=90, desc=f"Epoch {epoch}/{num_epochs}"):
                img_data = data_batch[0].to(self.device)
                targets = data_batch[1].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(img_data)
                loss = self.loss_fun(outputs, targets)
                loss.backward()
                self.optimizer.step()

                step += targets.size(dim=0)
                write_train_summary(writer=self.writer, model=self.model, loss=loss, global_step=step)

    def test(self, dataloader: DataLoader):
        step = 0
        aggr_loss = 0.
        self.model.eval()

        for data_batch in tqdm(dataloader, total=len(dataloader), ncols=90, desc=f"Evaluating model"):
            img_data = data_batch[0].to(self.device)
            targets = data_batch[1].to(self.device)

            outputs = self.model(img_data)
            aggr_loss += self.loss_fun(outputs, targets)

            step += 1

        return print(f"Test loss = {aggr_loss / step}")
