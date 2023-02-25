import pandas as pd
import torch
from tbparse import SummaryReader
from torch.utils.tensorboard import SummaryWriter


def write_train_summary(writer: SummaryWriter, model, loss: torch.Tensor, global_step: int):
    writer.add_scalar(tag="train loss", scalar_value=loss.cpu().item(), global_step=global_step)
    for name, param in model.named_parameters():
        writer.add_histogram(tag=f"trainable parameter {name}", values=param.cpu(), global_step=global_step)
        writer.add_histogram(tag=f"gradient of trainable parameter {name}", values=param.grad.cpu(),
                             global_step=global_step)


def write_validation_summary(writer: SummaryWriter, loss: float, accuracy: float, global_step: int):
    writer.add_scalar(tag="validation loss", scalar_value=loss, global_step=global_step)
    writer.add_scalar(tag="validation accuracy", scalar_value=accuracy, global_step=global_step)


def write_test_summary(writer: SummaryWriter, loss: float, accuracy: float, global_step: int):
    writer.add_scalar(tag="test loss", scalar_value=loss, global_step=global_step)
    writer.add_scalar(tag="test accuracy", scalar_value=accuracy, global_step=global_step)


def read_summary_files_to_df(summary_path: str, model_name: str, n_runs: int, n_epochs: int) -> pd.DataFrame:
    reader = SummaryReader(summary_path)
    df = reader.scalars
    df = df[df.tag == "test accuracy"]
    run_list = list(range(1, n_runs + 1)) * n_epochs
    df["run"] = run_list
    df["model"] = model_name
    return df

def read_gradient_summary_to_df(summary_path: str) -> pd.DataFrame:
    reader = SummaryReader(summary_path)
    df = reader.histograms
    df = df[df.tag.str.endswith("weight")]
    df = df[df.tag.str.startswith("gradient")]
    df.drop(columns=["counts"], inplace=True)
    df.tag = df.tag.map(lambda tag: f"Gradient of dense layer {tag[35]} weights")
    df = df.explode("limits")
    return df
