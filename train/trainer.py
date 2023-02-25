import os.path
from typing import Optional, Dict, Union

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import calc_accuracy
from utils import write_train_summary, write_validation_summary, write_test_summary


class Trainer:

    def __init__(
            self,
            model,
            optimizer=None,
            loss_fun=None,
            writer: SummaryWriter = None,
            force_cpu: bool = False,
            model_name: Optional[str] = None
    ):
        self.device = torch.device("cuda" if (torch.cuda.is_available() and not force_cpu) else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer if optimizer is not None else Adam(self.model.parameters(), lr=0.001)
        self.loss_fun = loss_fun if loss_fun is not None else CrossEntropyLoss(reduction="sum")
        self.reduction = self.loss_fun.reduction

        writer_path = os.path.join("results", self.model.__class__.__name__) \
            if model_name is None else os.path.join("results", model_name)
        self.writer = writer if writer is not None else SummaryWriter(writer_path)

    def train(
            self,
            train_loader: DataLoader,
            validation_loader: Union[DataLoader, None] = None,
            test_loader: Union[DataLoader, None] = None,
            num_epochs: int = 10,
            train_summary: bool = False,
            validation_summary: bool = False,
            validation_summary_at: int = 10_000,
            validate_after_epoch: bool = False,
            test_after_epoch: bool = False
    ):
        update_step = 1
        self.model.train()

        for epoch in range(1, num_epochs + 1):
            for data_batch in tqdm(train_loader, total=len(train_loader), ncols=90, desc=f"Epoch {epoch}/{num_epochs}"):
                img_data = data_batch[0].to(self.device)
                targets = data_batch[1].to(self.device)
                batch_size = targets.size(dim=0)

                self.optimizer.zero_grad()
                outputs = self.model(img_data)
                loss = self.loss_fun(outputs, targets)
                loss.backward()
                self.optimizer.step()

                if train_summary:
                    if self.reduction == "sum":
                        loss /= batch_size

                    write_train_summary(writer=self.writer, model=self.model, loss=loss, global_step=update_step)

                if update_step % validation_summary_at == 0 and validation_loader is not None and validation_summary:
                    self.__calculate_write_validation_metrics(validation_loader, update_step)

                update_step += 1

            if validate_after_epoch and validation_loader is not None:
                self.__calculate_write_validation_metrics(validation_loader, epoch)

            if test_after_epoch and test_loader is not None:
                self.__calculate_write_test_metrics(test_loader, epoch)

    def __calculate_write_validation_metrics(self, dataloader: DataLoader, step: int):
        results = self.__calculate_metrics(dataloader)
        write_validation_summary(writer=self.writer, loss=results["loss"], accuracy=results["accuracy"],
                                 global_step=step)

    def __calculate_write_test_metrics(self, dataloader: DataLoader, step: int):
        results = self.__calculate_metrics(dataloader)
        write_test_summary(writer=self.writer, loss=results["loss"], accuracy=results["accuracy"], global_step=step)

    def __calculate_metrics(self, dataloader: DataLoader) -> Dict[str, float]:
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for data_batch in dataloader:
                img_data = data_batch[0].to(self.device)
                targets = data_batch[1].to(self.device)

                outputs = self.model(img_data)

                all_outputs.extend(outputs)
                all_targets.extend(targets)

            all_outputs = torch.stack(all_outputs)
            all_targets = torch.stack(all_targets)

            loss = self.loss_fun(all_outputs, all_targets)
            accuracy = calc_accuracy(all_outputs, all_targets)

            if self.reduction == "sum":
                loss /= len(dataloader.dataset)

        return {'accuracy': accuracy, 'loss': loss}

    def test(self, dataloader: DataLoader):
        all_outputs = []
        all_targets = []
        self.model.eval()

        for data_batch in tqdm(dataloader, total=len(dataloader), ncols=90, desc=f"Evaluating model"):
            img_data = data_batch[0].to(self.device)
            targets = data_batch[1].to(self.device)

            outputs = self.model(img_data)

            all_outputs.extend(outputs)
            all_targets.extend(targets)

        all_outputs = torch.stack(all_outputs)
        all_targets = torch.stack(all_targets)

        loss = self.loss_fun(all_outputs, all_targets)
        accuracy = calc_accuracy(all_outputs, all_targets)

        if self.reduction == "sum":
            loss /= len(dataloader.dataset)

        print(f"Test loss = {loss}")
        print(f"Test accuracy = {accuracy}")
