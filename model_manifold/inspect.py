from typing import Optional, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from torchvision import transforms
from tqdm import trange


def project_kernel(jac: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    kernel_basis = torch.qr(jac, some=False).Q[:, jac.shape[1] - 1:]
    coefficients = torch.lstsq(direction.unsqueeze(1), kernel_basis).solution
    coefficients = coefficients[: kernel_basis.shape[1], 0]
    displacement = torch.matmul(kernel_basis, coefficients)
    return displacement


def project_tangent(jac: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    coefficients = torch.lstsq(direction.unsqueeze(1), jac).solution
    coefficients = coefficients[: jac.shape[1], 0]
    displacement = torch.matmul(jac, coefficients)
    return displacement


def constant_direction(
        model: nn.Module,
        start: torch.Tensor,
        direction: torch.Tensor,
        projection: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        step_size: float = 0.1,
        steps: int = 1000,
        post_processing: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    direction = torch.flatten(direction)
    points = [start]
    x = start
    p = torch.exp(model(x.unsqueeze(0)))
    probability, prediction = torch.max(p, dim=-1)
    probabilities = [probability.item()]
    predictions = [prediction.item()]
    for _ in trange(steps):
        # noinspection PyTypeChecker
        j = jacobian(model, x.unsqueeze(0)).squeeze(0)
        with torch.no_grad():
            j = F.normalize(j.reshape(j.shape[0], -1).T, dim=0)
            displacement = projection(j, direction)
            displacement = F.normalize(displacement, dim=-1).reshape(start.shape)
            x = post_processing(x + step_size * displacement)
            points.append(x.detach())
            p = torch.exp(model(x.unsqueeze(0)))
            probability, prediction = torch.max(p, dim=-1)
            probabilities.append(probability.item())
            predictions.append(prediction.item())
    points = torch.stack(points, dim=0)
    probabilities = torch.tensor(probabilities, device=start.device)
    predictions = torch.tensor(predictions, device=start.device)
    return points, probabilities, predictions


def constant_direction_kernel(
        model: nn.Module,
        start: torch.Tensor,
        direction: torch.Tensor,
        step_size: float = 0.1,
        steps: int = 1000,
        post_processing: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        steps: int = 1000,
        post_processing: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        steps: int = 10000,
        threshold: float = 1.0,
        post_processing: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points = [start]
    x = start
    p = torch.exp(model(x.unsqueeze(0)))
    probability, prediction = torch.max(p, dim=-1)
    probabilities = [probability.item()]
    predictions = [prediction.item()]
    distance = torch.norm(end - x)
    print(
        f'Iteration {len(points) - 1:05d} - Distance {distance:.04f} - '
        f'Predicted {predictions[-1]} with probability {probabilities[-1]:0.4f}'
    )
    while distance > threshold and len(points) < steps + 1:
        # noinspection PyTypeChecker
        j = jacobian(model, x.unsqueeze(0)).squeeze(0)
        with torch.no_grad():
            j = F.normalize(j.reshape(j.shape[0], -1).T, dim=0)
            direction = (end - x).flatten()
            displacement = projection(j, direction)
            displacement = F.normalize(displacement, dim=-1).reshape(start.shape)
            x = post_processing(x + step_size * displacement)
            points.append(x.detach())
            p = torch.exp(model(x.unsqueeze(0)))
            probability, prediction = torch.max(p, dim=-1)
            probabilities.append(probability.item())
            predictions.append(prediction.item())
            distance = torch.norm(end - x)
            print(
                f'Iteration {len(points) - 1:05d} - Distance {distance:.04f} - '
                f'Predicted {predictions[-1]} with probability {probabilities[-1]:0.4f}'
            )
    points = torch.stack(points, dim=0)
    probabilities = torch.tensor(probabilities, device=start.device)
    predictions = torch.tensor(predictions, device=start.device)
    return points, probabilities, predictions


def path_kernel(
        model: nn.Module,
        start: torch.Tensor,
        end: torch.Tensor,
        step_size: float = 0.1,
        steps: int = 10000,
        threshold: float = 1.0,
        post_processing: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return path(
        model,
        start,
        end,
        projection=project_kernel,
        step_size=step_size,
        steps=steps,
        threshold=threshold,
        post_processing=post_processing,
    )


def path_tangent(
        model: nn.Module,
        start: torch.Tensor,
        end: torch.Tensor,
        step_size: float = 0.1,
        steps: int = 10000,
        threshold: float = 1.0,
        post_processing: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return path(
        model,
        start,
        end,
        projection=project_tangent,
        step_size=step_size,
        steps=steps,
        threshold=threshold,
        post_processing=post_processing,
    )


def domain_projection(
        x: torch.Tensor,
        normalization: transforms.Normalize,
        domain: Tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    inf = torch.tensor(domain[0]).repeat(x.shape[0], 1, 1)
    sup = torch.tensor(domain[1]).repeat(x.shape[0], 1, 1)
    normalized_inf = normalization(inf).reshape(x.shape[0])
    normalized_sup = normalization(sup).reshape(x.shape[0])
    for i in range(x.shape[0]):
        x[i] = torch.clamp(x[i], normalized_inf[i], normalized_sup[i])
    return x
