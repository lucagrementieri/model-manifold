import torch
import torch.nn as nn
from torch.autograd.functional import jacobian


def local_data_matrix(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    training_state = model.training
    if training_state:
        model.eval()
    # noinspection PyTypeChecker
    j = jacobian(model, x.unsqueeze(0)).squeeze(0)
    j = j.reshape(j.size(0), -1)
    with torch.no_grad():
        p = torch.exp(model(x.unsqueeze(0)))
        jacobian_product = torch.bmm(j.unsqueeze(2), j.unsqueeze(1)).permute(1, 2, 0)
        g_matrix = torch.sum(p * jacobian_product, dim=-1)
    if training_state:
        model.train()
    return g_matrix


def local_data_matrix_trace(model: nn.Module, x: torch.Tensor) -> float:
    training_state = model.training
    if training_state:
        model.eval()
    # noinspection PyTypeChecker
    j = jacobian(model, x.unsqueeze(0)).squeeze(0)
    j = j.reshape(j.size(0), -1)
    with torch.no_grad():
        p = torch.exp(model(x.unsqueeze(0)))
        trace = torch.sum(p * torch.pow(j, 2).sum(dim=1))
    if training_state:
        model.train()
    return trace.item()


def batch_data_matrix_rank_and_trace(
    model: nn.Module, batch: torch.Tensor
) -> torch.Tensor:
    return torch.tensor([local_data_matrix_trace(model, x) for x in batch])
