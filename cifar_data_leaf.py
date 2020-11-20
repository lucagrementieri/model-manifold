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

import cifar_vgg
from model_manifold.inspect import path_tangent, domain_projection
from model_manifold.plot import denormalize, to_gif, show_cifar_strip


def random_idx(model: nn.Module, loader: Dataset, excluded: int):
    idx = random.randrange(len(loader))
    """
    device = next(model.parameters()).device
    image = loader[idx][0].to(device)
    p = torch.exp(model(image.unsqueeze(0)))
    # noinspection PyTypeChecker
    while torch.any(p > 0.99) or idx == excluded:
        idx = random.randrange(len(loader))
        image = loader[idx][0].to(device)
        p = torch.exp(model(image.unsqueeze(0)))
    """
    return idx


def cifar_path(
        checkpoint_path: Union[str, Path], step_size=0.1, start_idx: int = -1, end_idx: int = -1, flip: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    test_cifar = datasets.CIFAR10(
        'data',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = cifar_vgg.VGG('VGG16')
    network = network.to(device)
    network.load_state_dict(torch.load(checkpoint_path, map_location=device)['net'])

    if start_idx == -1:
        start_idx = random_idx(network, test_cifar, end_idx)
    start_image = test_cifar[start_idx][0]
    if flip:
        start_image = torch.flip(start_image, dims=(-1,))
    start_image = start_image.to(device)
    p = torch.exp(network(start_image.unsqueeze(0)))
    # noinspection PyTypeChecker
    if torch.any(p > 0.99):
        print(
            'Warning: the manifold path could be hard to find because '
            'the model is too confident on the selected start image.'
        )

    if end_idx == -1:
        end_idx = random_idx(network, test_cifar, start_idx)
    end_image = test_cifar[end_idx][0]
    if flip:
        end_image = torch.flip(end_image, dims=(-1,))
    end_image = end_image.to(device)
    p = torch.exp(network(end_image.unsqueeze(0)))
    # noinspection PyTypeChecker
    if torch.any(p > 0.99):
        print(
            'Warning: the manifold path could be hard to find because '
            'the model is too confident on the selected end image.'
        )

    print(f'Compute path from {start_idx} to {end_idx}.')

    # noinspection PyTypeChecker
    data_path, prob_path, pred_path = path_tangent(
        network,
        start_image,
        end_image,
        step_size=step_size,
        steps=100000,
        post_processing=partial(domain_projection, normalization=normalize),
    )
    data_path = denormalize(data_path, normalize)

    return data_path, prob_path, pred_path, start_idx, end_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export the path on the data leaf connecting two images '
                    'from the CIFAR10 test set as a .gif',
        usage='python3 cifar_data_leaf.py CHECKPOINT '
              '[--start START --end END --seed SEED --output-dir OUTPUT-DIR --flip]',
    )
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint model')
    parser.add_argument(
        '--step-size', type=float, default=1, help='Step size'
    )
    parser.add_argument(
        '--start', type=int, default=-1, help='Index of the starting image'
    )
    parser.add_argument('--end', type=int, default=-1, help='Index of ending image')
    parser.add_argument('--seed', type=int, default=5, help='Random seed')
    parser.add_argument(
        '--output-dir', type=str, default='outputs', help='Output directory',
    )
    parser.add_argument(
        '--flip',
        action='store_true',
        help='Flip start image',
    )

    args = parser.parse_args(sys.argv[1:])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    image_path, probability_path, prediction_path, start, end = cifar_path(
        args.checkpoint, args.step_size, args.start, args.end, args.flip
    )
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    to_gif(
        image_path,
        output_dir / f'{start:05d}_{"flip_" if args.flip else ""}{end:05d}.gif',
        step=200,
        scale_factor=8.0,
    )
    show_cifar_strip(image_path, probability_path, prediction_path)
