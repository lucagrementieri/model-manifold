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
from model_manifold.inspect import constant_direction_kernel, domain_projection
from model_manifold.plot import denormalize, to_gif, show_strip


def mnist_kernel_direction(
        checkpoint_path: Union[str, Path], start_idx: int = -1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    test_mnist = datasets.MNIST(
        'data',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    network = mnist_networks.small_cnn(checkpoint_path)

    if start_idx == -1:
        start_idx = random.randrange(len(test_mnist))

    print(f'Evolve the image {start_idx} in the kernel of the local data matrix.')

    device = next(network.parameters()).device
    start_image = test_mnist[start_idx][0].to(device)
    v = torch.randn_like(start_image)

    # noinspection PyTypeChecker
    data_path, prob_path, pred_path = constant_direction_kernel(
        network,
        start_image,
        v,
        steps=1000,
        post_processing=partial(domain_projection, normalization=normalize),
    )
    data_path = denormalize(data_path, normalize)

    return data_path, prob_path, pred_path, start_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export the path obtained evolving a valid image along a '
                    'random direction in the kernel of the local data matrix as a .gif',
        usage='python3 mnist_kernel.py CHECKPOINT '
              '[--start START --seed SEED --output-dir OUTPUT-DIR]',
    )
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint model')
    parser.add_argument(
        '--start', type=int, default=-1, help='Index of the starting image'
    )
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument(
        '--output-dir', type=str, default='outputs', help='Output directory',
    )

    args = parser.parse_args(sys.argv[1:])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    image_path, probability_path, prediction_path, start = mnist_kernel_direction(
        args.checkpoint, args.start
    )
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    to_gif(
        image_path, output_dir / f'{start:05d}_noise.gif', step=100, scale_factor=10.0,
    )
    show_strip(image_path, probability_path, prediction_path)
