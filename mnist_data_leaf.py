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
from model_manifold.inspect import path_tangent, domain_projection
from model_manifold.plot import denormalize, to_gif, show_grid


def mnist_path(
        checkpoint_path: Union[str, Path],
        start_idx: int = -1,
        end_idx: int = -1,
) -> Tuple[torch.Tensor, int, int]:
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
        start_image = test_mnist[start_idx][0]
        p = torch.exp(network(start_image.unsqueeze(0)))
        # noinspection PyTypeChecker
        while torch.any(p > 0.99) or start_idx == end_idx:
            start_idx = random.randrange(len(test_mnist))
            start_image = test_mnist[start_idx][0]
            p = torch.exp(network(start_image.unsqueeze(0)))
    else:
        start_image = test_mnist[start_idx][0]
        p = torch.exp(network(start_image.unsqueeze(0)))
        # noinspection PyTypeChecker
        if torch.any(p > 0.99):
            print(
                'Warning: the manifold path could be hard to find because '
                'the model is too confident on the selected start image.'
            )

    print(
        f'The predicted label for the start image is '
        f'{p.argmax().item()} with probability {p.max().item():0.4f}.'
    )

    if end_idx == -1:
        end_idx = random.randrange(len(test_mnist))
        while end_idx == start_idx:
            end_idx = random.randrange(len(test_mnist))

    # noinspection PyTypeChecker
    data_path = path_tangent(
        network,
        test_mnist[start_idx][0],
        test_mnist[end_idx][0],
        steps=10000,
        post_processing=partial(domain_projection, normalization=normalize),
    )
    data_path = denormalize(data_path, normalize)

    return data_path, start_idx, end_idx


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
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument(
        '--output-dir', type=str, default='outputs', help='Output directory',
    )

    args = parser.parse_args(sys.argv[1:])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    image_path, start, end = mnist_path(args.checkpoint, args.output_dir, args.start, args.end)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    to_gif(
        image_path,
        output_dir / f'{start:05d}_{end:05d}.gif',
        step=100,
        scale_factor=10.0,
    )
    show_grid(image_path, 2, 5)