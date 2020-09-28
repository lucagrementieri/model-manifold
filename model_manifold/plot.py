from pathlib import Path
from typing import Union

import imageio
import torch
import torch.nn.functional as F
from torchvision import transforms


def to_gif(
        images: torch.Tensor,
        output_path: Union[str, Path],
        step: int = 1,
        scale_factor: float = 1.0,
) -> None:
    # noinspection PyArgumentList
    images = F.interpolate(images, scale_factor=scale_factor)
    images = images.permute(0, 2, 3, 1).squeeze_(-1)
    images = torch.round(images * 255).to(torch.uint8)
    images = images.cpu()
    images = torch.unbind(images)
    imageio.mimsave(str(output_path), images[::step])


def denormalize(x: torch.Tensor, normalization: transforms.Normalize) -> torch.Tensor:
    mean = torch.tensor(normalization.mean, device=x.device).view(-1, 1, 1)
    std = torch.tensor(normalization.std, device=x.device).view(-1, 1, 1)
    x *= std
    x += mean
    return x
