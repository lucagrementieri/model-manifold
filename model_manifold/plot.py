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


def show_strip(
        images: torch.Tensor,
        probabilities: torch.Tensor,
        predictions: torch.Tensor,
        steps: int = 8
) -> None:
    images = images.permute(0, 2, 3, 1).squeeze_(-1)
    image_indices = torch.linspace(0, images.shape[0] - 1, steps).tolist()
    fig, axes = plt.subplots(1, steps, figsize=(10, 2))
    for plot_idx, image_idx in enumerate(image_indices):
        iteration = round(image_idx)
        image = images[iteration].cpu()
        axes[plot_idx].imshow(image, cmap='gray', vmin=0, vmax=1)
        axes[plot_idx].set_title(
            f'Iteration {iteration}:\n'
            f'predicted label {predictions[iteration]} with\n'
            f'probability {probabilities[iteration]:0.4f}',
            fontsize=7,
        )
        axes[plot_idx].axis('off')
    fig.tight_layout(pad=0.1)
    plt.show()
