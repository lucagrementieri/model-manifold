import argparse
import random
import sys
from functools import partial
from pathlib import Path
from typing import Union, Tuple

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import mnist_networks
from model_manifold.inspect import (
    constant_direction_kernel,
    path_tangent,
    domain_projection,
)
from model_manifold.plot import denormalize, to_gif, save_strip


def mnist_noise_path(
    checkpoint_path: Union[str, Path],
    start_idx: int = -1,
    end_idx: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    test_mnist = datasets.MNIST(
        "data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    network = mnist_networks.medium_cnn(checkpoint_path)
    device = next(network.parameters()).device

    if start_idx == -1:
        start_idx = random.randrange(len(test_mnist))
    start_image = test_mnist[start_idx][0].to(device)
    if end_idx == -1:
        end_idx = random.randrange(len(test_mnist))
    end_image = test_mnist[end_idx][0].to(device)

    print(f"Compute path from a noisy {start_idx} to {end_idx}.")
    v = torch.randn_like(start_image)

    # noinspection PyTypeChecker
    noise_path, _, _ = constant_direction_kernel(
        network,
        start_image,
        v,
        steps=250,
        post_processing=partial(domain_projection, normalization=normalize),
    )
    noisy_start = noise_path[-1]

    # noinspection PyTypeChecker
    data_path, prob_path, pred_path = path_tangent(
        network,
        noisy_start.to(device),
        end_image,
        steps=10000,
        post_processing=partial(domain_projection, normalization=normalize),
    )
    data_path = denormalize(data_path, normalize)

    return data_path, prob_path, pred_path, start_idx, end_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export the path on a noise leaf that minimize the distance "
        "from a valid image in the MNIST test set as a .gif",
        usage="python3 mnist_noise_leaf.py CHECKPOINT "
        "[--start START --end END --seed SEED --output-dir OUTPUT-DIR]",
    )
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint model")
    parser.add_argument(
        "--start", type=int, default=-1, help="Index of the starting image"
    )
    parser.add_argument("--end", type=int, default=-1, help="Index of ending image")
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory",
    )

    args = parser.parse_args(sys.argv[1:])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    image_path, probability_path, prediction_path, start, end = mnist_noise_path(
        args.checkpoint, args.start, args.end
    )
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{start:05d}_noise_{end:05d}"
    to_gif(
        image_path,
        output_dir / f"{filename}.gif",
        step=100,
        scale_factor=10.0,
    )
    save_strip(
        image_path, output_dir / f"{filename}.png", probability_path, prediction_path
    )
