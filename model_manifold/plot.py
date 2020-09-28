from pathlib import Path
from typing import Union, Optional

import imageio
import torch
from torchvision import transforms


def to_gif(images: torch.Tensor, output_path: Union[str, Path]) -> None:
    # noinspection PyArgumentList
    images = images.permute(0, 2, 3, 1).squeeze_(-1)
    images = torch.round(images * 255).to(torch.uint8)
    images = images.cpu()
    images = torch.unbind(images)
    imageio.mimsave(str(output_path), images)


def denormalize(x: torch.Tensor, normalization: transforms.Normalize) -> torch.Tensor:
    mean = torch.tensor(normalization.mean, device=x.device).view(-1, 1, 1)
    std = torch.tensor(normalization.std, device=x.device).view(-1, 1, 1)
    x *= std
    x += mean
    return x


def evolution_to_gif(
        evolution: torch.Tensor,
        output_path: Union[str, Path],
        normalization: Optional[transforms.Normalize] = None,
) -> None:
    if normalization is not None:
        evolution = denormalize(evolution, normalization)
    to_gif(evolution, output_path)
