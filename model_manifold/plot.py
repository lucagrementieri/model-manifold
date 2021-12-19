from pathlib import Path
from typing import Union

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import seaborn as sns


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


def save_ranks(
    steps: torch.Tensor, ranks: torch.Tensor, output_path: Union[str, Path]
) -> None:
    steps = steps.cpu().numpy()
    ranks = ranks.cpu().numpy()
    steps = np.tile(steps, (ranks.shape[0], 1))
    points = np.stack([steps, ranks], axis=-1).reshape(-1, 2)
    unique_points, counts = np.unique(points, axis=0, return_counts=True)
    data = {
        "Steps": unique_points[:, 0],
        r"Rank of $G(x, w)$": unique_points[:, 1],
        "Count": counts,
    }
    plt.clf()
    with plt.style.context("seaborn"):
        sns.scatterplot(
            data=data,
            x="Steps",
            y=r"Rank of $G(x, w)$",
            hue="Count",
            size="Count",
            sizes=(20, 200),
            legend="full",
            palette="coolwarm",
        )
        plt.yticks(range(np.max(ranks) + 2))
        plt.xlabel("Steps")
        plt.ylabel(r"Rank of $G(x, w)$")
        plt.title(r"Distribution of rank $G(x, w)$ during training")
    plt.tight_layout()
    plt.savefig(str(output_path))


def save_traces(
    steps: torch.Tensor, traces: torch.Tensor, output_path: Union[str, Path]
) -> None:
    steps = steps.cpu().numpy()
    traces = traces.cpu().numpy()
    colors = (
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    )
    plt.clf()
    with plt.style.context("seaborn"):
        for i in range(len(traces)):
            plt.plot(steps, traces[i], color=colors[i % len(colors)])
        plt.xlabel("Steps")
        plt.ylabel(r"Trace of $G(x, w)$")
        plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(str(output_path))
