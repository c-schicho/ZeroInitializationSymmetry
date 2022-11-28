import torch


def calc_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    _, predictions = torch.max(outputs, dim=1)
    return torch.sum(predictions == targets).item() / len(predictions)
