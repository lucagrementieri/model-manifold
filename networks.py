import torch
import torch.nn as nn


def small_cnn() -> nn.Module:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = nn.Sequential(
        nn.Conv2d(1, 4, 3, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(4, 4, 3, 1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, stride=2),
        nn.Flatten(),
        nn.Linear(576, 32),
        nn.ReLU(inplace=True),
        nn.Linear(32, 10),
        nn.LogSoftmax(dim=1),
    )
    net = net.to(device)
    return net
