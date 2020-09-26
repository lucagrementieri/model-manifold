from functools import partial
import random

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import networks
from model_manifold.inspect import (
    path_tangent,
    domain_projection,
)

normalize = transforms.Normalize((0.1307,), (0.3081,))
test_mnist = datasets.MNIST(
    'data',
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), normalize]),
)

checkpoint = 'checkpoints/small_cnn_01.pt'
network = networks.small_cnn(checkpoint)

order = list(range(len(test_mnist)))

random.shuffle(order)

start_idx = order[0]
for i in order:
    image, label = test_mnist[i]
    p = torch.exp(network(image.unsqueeze(0)))[0, label].item()
    if p < 0.8:
        start_idx = i
        break

end_idx = random.choice(order)

# noinspection PyTypeChecker
joining_path = path_tangent(
    network,
    test_mnist[start_idx][0],
    test_mnist[end_idx][0],
    post_processing=partial(domain_projection, normalization=normalize),
)
