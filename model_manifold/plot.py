from pathlib import Path
from typing import Union

import imageio
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms


def denormalize(x: torch.Tensor, normalization: transforms.Normalize) -> torch.Tensor:
    mean = torch.tensor(normalization.mean, device=x.device).view(-1, 1, 1)
    std = torch.tensor(normalization.std, device=x.device).view(-1, 1, 1)
    x *= std
    x += mean
    return x


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


def show_grid(images: torch.Tensor, rows: int, columns: int) -> None:
    steps = rows * columns
    plt.figure()
    plt.axis('off')
    for plot_idx, image_idx in enumerate(torch.linspace(0, images.shape[0], steps)):
        plt.subplot(rows, columns, plot_idx)
        image = images[round(image_idx)].cpu()
        plt.imshow(image, vmin=0, vmax=1)
    plt.show()
