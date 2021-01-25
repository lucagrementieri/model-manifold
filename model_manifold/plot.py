from pathlib import Path
from typing import Union

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    images = images[::step]
    images = images.cpu()
    images = F.interpolate(images, scale_factor=scale_factor)
    images = images.permute(0, 2, 3, 1).squeeze_(-1)
    images = torch.round(images * 255).to(torch.uint8)
    images = torch.unbind(images)
    images = [im.numpy() for im in images]
    imageio.mimsave(str(output_path), images)


def save_strip(
    images: torch.Tensor,
    output_path: Union[str, Path],
    probabilities: torch.Tensor,
    predictions: torch.Tensor,
    steps: int = 9,
) -> None:
    images = images.permute(0, 2, 3, 1).squeeze_(-1)
    image_indices = torch.linspace(0, images.shape[0] - 1, steps).tolist()
    fig, axes = plt.subplots(1, steps, figsize=(10, 1.8))
    for plot_idx, image_idx in enumerate(image_indices):
        iteration = round(image_idx)
        image = images[iteration].cpu()
        axes[plot_idx].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[plot_idx].set_title(
            f"Iteration {iteration}:\n"
            f"predicted label {predictions[iteration]} with\n"
            f"probability {probabilities[iteration]:0.4f}",
            fontsize=7,
        )
        axes[plot_idx].axis("off")
    fig.tight_layout()
    plt.savefig(str(output_path))


def save_traces(
    steps: np.ndarray, traces: np.ndarray, output_path: Union[str, Path]
) -> None:
    mean = np.mean(traces, axis=0)
    exponential_average = pd.Series(mean).ewm(alpha=0.05).mean().to_numpy()
    with plt.style.context("seaborn"):
        plt.plot(steps, mean, alpha=0.8)
        plt.plot(steps, exponential_average, color="r")
        plt.xlabel("Steps")
        plt.ylabel(r"Mean trace of $G(x, w)$")
        plt.title(r"Trace of $G(x, w)$ during training")
        plt.ylim(bottom=0)
    plt.savefig(str(output_path))
