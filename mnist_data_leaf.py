import argparse
import random
import sys
from functools import partial
from pathlib import Path
from typing import Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import mnist_networks
from model_manifold.inspect import path_tangent, domain_projection
from model_manifold.plot import denormalize, to_gif, show_strip


def random_idx(model: nn.Module, loader: Dataset, excluded: int):
    idx = random.randrange(len(loader))
    image = loader[idx][0]
    p = torch.exp(model(image.unsqueeze(0)))
    # noinspection PyTypeChecker
    while torch.any(p > 0.99) or idx == excluded:
        idx = random.randrange(len(loader))
        image = loader[idx][0]
        p = torch.exp(model(image.unsqueeze(0)))
    return idx


def mnist_path(
        checkpoint_path: Union[str, Path], start_idx: int = -1, end_idx: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    test_mnist = datasets.MNIST(
        'data',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    network = mnist_networks.small_cnn(checkpoint_path)

    if start_idx == -1:
        start_idx = random_idx(network, test_mnist, end_idx)
    else:
        start_image = test_mnist[start_idx][0]
        p = torch.exp(network(start_image.unsqueeze(0)))
        # noinspection PyTypeChecker
        if torch.any(p > 0.99):
            print(
                'Warning: the manifold path could be hard to find because '
                'the model is too confident on the selected start image.'
            )

    if end_idx == -1:
        end_idx = random_idx(network, test_mnist, start_idx)
    else:
        end_image = test_mnist[end_idx][0]
        p = torch.exp(network(end_image.unsqueeze(0)))
        # noinspection PyTypeChecker
        if torch.any(p > 0.99):
            print(
                'Warning: the manifold path could be hard to find because '
                'the model is too confident on the selected end image.'
            )

    print(f'Compute path from {start_idx} to {end_idx}')

    device = next(network.parameters()).device
    # noinspection PyTypeChecker
    data_path, prob_path, pred_path = path_tangent(
        network,
        test_mnist[start_idx][0].to(device),
        test_mnist[end_idx][0].to(device),
        steps=10000,
        post_processing=partial(domain_projection, normalization=normalize),
    )
    data_path = denormalize(data_path, normalize)

    return data_path, prob_path, pred_path, start_idx, end_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export the path on the data leaf connecting two images '
                    'from the MNIST test set as a .gif',
        usage='python3 mnist_data_leaf.py CHECKPOINT '
              '[--start START --end END --seed SEED --output-dir OUTPUT-DIR]',
    )
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint model')
    parser.add_argument(
        '--start', type=int, default=-1, help='Index of the starting image'
    )
    parser.add_argument('--end', type=int, default=-1, help='Index of ending image')
    parser.add_argument('--seed', type=int, default=5, help='Random seed')
    parser.add_argument(
        '--output-dir', type=str, default='outputs', help='Output directory',
    )

    args = parser.parse_args(sys.argv[1:])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    image_path, probability_path, prediction_path, start, end = mnist_path(
        args.checkpoint, args.start, args.end
    )
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    to_gif(
        image_path,
        output_dir / f'{start:05d}_{end:05d}.gif',
        step=100,
        scale_factor=10.0,
    )
    show_strip(image_path, probability_path, prediction_path)
