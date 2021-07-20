import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms


def load_datasets(args):
    """Create data loaders for the CIFAR-10 dataset.
    Returns: Dict containing data loaders.
    """
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ])

    if args['cutout'] > 0:
        train_transform.transforms.append(Cutout(length=args['cutout']))

    valid_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ])

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = datasets.CIFAR10(root=args['data_path'],
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    valid_dataset = datasets.CIFAR10(root=args['data_path'],
                                     train=True,
                                     transform=valid_transform,
                                     download=True)

    test_dataset = datasets.CIFAR10(root=args['data_path'],
                                    train=False,
                                    transform=test_transform,
                                    download=True)

    train_indices = list(range(0, 45000))
    valid_indices = list(range(45000, 50000))
    train_subset = Subset(train_dataset, train_indices)
    valid_subset = Subset(valid_dataset, valid_indices)

    data_loaders = {}
    data_loaders['train_subset'] = torch.utils.data.DataLoader(
        dataset=train_subset,
        batch_size=args['batch_size'],
        shuffle=True,
        pin_memory=False,
        num_workers=2)

    data_loaders['valid_subset'] = torch.utils.data.DataLoader(
        dataset=valid_subset,
        batch_size=args['batch_size'],
        shuffle=True,
        pin_memory=False,
        num_workers=2,
        drop_last=True)

    data_loaders['train_dataset'] = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args['batch_size'],
        shuffle=True,
        pin_memory=False,
        num_workers=2)

    data_loaders['test_dataset'] = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        pin_memory=False,
        num_workers=2)

    return data_loaders