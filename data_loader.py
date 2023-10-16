import os
import torch
import torchvision
import torchvision.transforms as transforms


def fetch_dataloader(types, params):

    normalize_mean = (0.5070751592371323,
                      0.48654887331495095, 0.4409178433670343)
    normalize_std = (0.2673342858792401, 0.2564384629170883,
                     0.27615047132568404)

    data_folder = 'data'

    if params['augmentation'] == "yes":
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ])
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ])

    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])

    if params['dataset'] == 'cifar10':
        dataset_path = os.path.join(data_folder, 'data-cifar10')
        dataset_class = torchvision.datasets.CIFAR10
    elif params['dataset'] == 'cifar100':
        dataset_path = os.path.join(data_folder, 'data-cifar100')
        dataset_class = torchvision.datasets.CIFAR100

    trainset = dataset_class(
        root=dataset_path, train=True, download=True, transform=train_transformer)
    devset = dataset_class(root=dataset_path, train=False,
                           download=True, transform=dev_transformer)

    if params['dataset'] == 'tiny_imagenet':
        data_dir = os.path.join(data_folder, 'tiny-imagenet-200')
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [
                                     0.2302, 0.2265, 0.2262]),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [
                                     0.2302, 0.2265, 0.2262]),
            ])
        }
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'val', 'images')
        trainset = torchvision.datasets.ImageFolder(
            train_dir, data_transforms['train'])
        devset = torchvision.datasets.ImageFolder(
            test_dir, data_transforms['val'])

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'])
    devloader = torch.utils.data.DataLoader(
        devset, batch_size=params['batch_size'], shuffle=False, num_workers=params['num_workers'])

    return trainloader if types == 'train' else devloader
