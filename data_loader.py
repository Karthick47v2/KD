import torch
import torchvision
import torchvision.transforms as transforms


def fetch_dataloader(types, params):
    """
    Fetch and return train/dev dataloader with hyperparameters (params.subset_percent = 1.)
    """
    # using random crops and horizontal flip for train set
    if params.augmentation == "yes":
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.240, 0.243, 0.261))

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))])

    if params.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data/data-cifar10', train=True,
                                                download=True, transform=train_transformer)
        devset = torchvision.datasets.CIFAR10(root='./data/data-cifar10', train=False,
                                              download=True, transform=dev_transformer)
    elif params.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data/data-cifar100', train=True,
                                                 download=True, transform=train_transformer)
        devset = torchvision.datasets.CIFAR100(root='./data/data-cifar100', train=False,
                                               download=True, transform=dev_transformer)
    elif params.dataset == 'tiny_imagenet':
        data_dir = './data/tiny-imagenet-200/'
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
        train_dir = data_dir + 'train/'
        test_dir = data_dir + 'val/images/'
        trainset = torchvision.datasets.ImageFolder(
            train_dir, data_transforms['train'])
        devset = torchvision.datasets.ImageFolder(
            test_dir, data_transforms['val'])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
                                              shuffle=True, num_workers=params.num_workers)

    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
                                            shuffle=False, num_workers=params.num_workers)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl
