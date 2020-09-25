from functools import partial

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model_manifold.inspect import path_tangent, domain_projection
import networks

normalize = transforms.Normalize((0.1307,), (0.3081,))
test_mnist = datasets.MNIST(
    'data',
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), normalize]),
)

checkpoint = 'checkpoints/small_cnn_01.pt'
network = networks.small_cnn(checkpoint)

start_idx = 0
end_idx = 1

# noinspection PyTypeChecker
joining_path = path_tangent(
    network,
    test_mnist[start_idx][0],
    test_mnist[end_idx][0],
    post_processing=partial(domain_projection, normalization=normalize),
)
