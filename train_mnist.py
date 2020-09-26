import argparse
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from networks import small_cnn


def train_epoch(
        model: nn.Module, loader: DataLoader, optimizer: Optimizer, epoch: int
) -> None:
    log_interval = len(loader) // 10
    device = next(model.parameters()).device
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model: nn.Module, loader: DataLoader) -> float:
    device = next(model.parameters()).device
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss,
            correct,
            len(loader.dataset),
            100.0 * correct / len(loader.dataset),
        )
    )
    return test_loss


def mnist_loader(batch_size: int, train: bool) -> DataLoader:
    loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            'data',
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=batch_size,
        shuffle=train,
        num_workers=1,
        pin_memory=True,
    )
    return loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a basic model on MNIST',
        usage='python3 train_mnist.py '
              '[--batch-size BATCH-SIZE --epochs EPOCHS --lr LR --seed SEED]',
    )
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')

    args = parser.parse_args(sys.argv[2:])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = small_cnn()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    train_loader = mnist_loader(args.batch_size, train=True)
    test_loader = mnist_loader(args.batch_size, train=False)

    for epoch in range(args.epochs):
        train_epoch(model, train_loader, optimizer, epoch + 1)
        test(model, test_loader)
        torch.save(model.state_dict(), f'checkpoints/small_cnn_{epoch + 1:02d}.pt')
