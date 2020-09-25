from typing import Optional, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from torchvision import transforms
from tqdm import trange


def project_kernel(jac: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    kernel_basis = torch.qr(jac, some=False).Q[:, jac.size(1) - 1:]
    coefficients = torch.lstsq(direction.unsqueeze(1), kernel_basis).solution
    coefficients = coefficients[: kernel_basis.size(1), 0]
    displacement = torch.matmul(kernel_basis, coefficients)
    return displacement


def project_tangent(jac: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    coefficients = torch.lstsq(direction.unsqueeze(1), jac).solution
    coefficients = coefficients[: jac.size(1), 0]
    displacement = torch.matmul(jac, coefficients)
    return displacement


def constant_direction(
        model: nn.Module,
        start: torch.Tensor,
        direction: torch.Tensor,
        projection: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        step_size: float = 0.1,
        steps: int = 5000,
        post_processing: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    direction = direction.flatten()
    evolution = [start]
    point = start
    for i in trange(steps):
        # noinspection PyTypeChecker
        j = jacobian(model, start.unsqueeze(0)).squeeze(0)
        j = F.normalize(j.reshape(j.size(0), -1).T, dim=0)
        displacement = projection(j, direction)
        displacement = F.normalize(displacement, dim=-1).reshape(evolution[-1])
        point = post_processing(point + step_size * displacement)
        evolution.append(point)
    return torch.stack(evolution, dim=0)


def constant_direction_kernel(
        model: nn.Module,
        start: torch.Tensor,
        direction: torch.Tensor,
        step_size: float = 0.1,
        steps: int = 5000,
        post_processing: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    return constant_direction(
        model,
        start,
        direction,
        projection=project_kernel,
        step_size=step_size,
        steps=steps,
        post_processing=post_processing,
    )


def constant_direction_tangent(
        model: nn.Module,
        start: torch.Tensor,
        direction: torch.Tensor,
        step_size: float = 0.1,
        steps: int = 5000,
        post_processing: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    return constant_direction(
        model,
        start,
        direction,
        projection=project_tangent,
        step_size=step_size,
        steps=steps,
        post_processing=post_processing,
    )


def path(
        model: nn.Module,
        start: torch.Tensor,
        end: torch.Tensor,
        projection: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        step_size: float = 0.1,
        threshold: float = 1.0,
        post_processing: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    evolution = [start]
    point = start
    distance = torch.norm(end - point)
    print(f'Iteration {len(evolution) - 1:05d} - Distance {distance:.04f}\r', end='')
    while distance > threshold:
        # noinspection PyTypeChecker
        j = jacobian(model, start.unsqueeze(0)).squeeze(0)
        j = F.normalize(j.reshape(j.size(0), -1).T, dim=0)
        direction = (end - point).flatten()
        displacement = projection(j, direction)
        displacement = F.normalize(displacement, dim=-1).reshape(start.shape)
        point = post_processing(point + step_size * displacement)
        evolution.append(point)
        distance = torch.norm(end - point)
        print(
            f'Iteration {len(evolution) - 1:05d} - Distance {distance:.04f}\r', end=''
        )
    return torch.stack(evolution, dim=0)


def path_kernel(
        model: nn.Module,
        start: torch.Tensor,
        end: torch.Tensor,
        step_size: float = 0.01,
        threshold: float = 1.0,
        post_processing: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    return path(
        model,
        start,
        end,
        projection=project_kernel,
        step_size=step_size,
        threshold=threshold,
        post_processing=post_processing,
    )


def path_tangent(
        model: nn.Module,
        start: torch.Tensor,
        end: torch.Tensor,
        step_size: float = 0.01,
        threshold: float = 1.0,
        post_processing: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    return path(
        model,
        start,
        end,
        projection=project_tangent,
        step_size=step_size,
        threshold=threshold,
        post_processing=post_processing,
    )


def domain_projection(
        x: torch.Tensor,
        normalization: transforms.Normalize,
        domain: Tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    inf = torch.tensor(domain[0]).reshape(1, 1, 1)
    sup = torch.tensor(domain[1]).reshape(1, 1, 1)
    return torch.clamp(x, normalization(inf).item(), normalization(sup).item())
