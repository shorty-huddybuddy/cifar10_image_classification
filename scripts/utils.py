from typing import Tuple
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def cifar10_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])


def get_dataloaders(batch_size: int = 128, data_root: str = './data') -> Tuple[DataLoader, DataLoader]:
    train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=cifar10_transforms(True))
    test_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=cifar10_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]
