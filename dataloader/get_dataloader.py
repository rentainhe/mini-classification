import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from dataloader.get_transforms import get_transforms

def get_train_loader(__C):
    # data transforms
    train_transform, test_transform = get_transforms(__C)

    # get dataset
    if __C.dataset['name'] == "cifar10":
        trainset = datasets.CIFAR10(root=__C.dataset['dir'],
                                    train=True,
                                    download=True,
                                    transform=train_transform)
    elif __C.dataset['name'] == "cifar100":
        trainset = datasets.CIFAR100(root=__C.dataset['dir'],
                                    train=True,
                                    download=True,
                                    transform=train_transform)

    # get dataloader
    train_sampler = RandomSampler(trainset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=__C.dataset['batch_size'],
                              num_workers=__C.dataloader['num_workers'],
                              pin_memory=__C.dataloader['pin_memory'],
                              drop_last=__C.dataloader['drop_last'])
    return train_loader

def get_test_loader(__C):
    # data transforms
    train_transform, test_transform = get_transforms(__C)

    # get dataset
    if __C.dataset['name'] == "cifar10":
        testset = datasets.CIFAR10(root=__C.dataset['dir'],
                                    train=False,
                                    download=True,
                                    transform=test_transform)
    elif __C.dataset['name'] == "cifar100":
        testset = datasets.CIFAR100(root=__C.dataset['dir'],
                                    train=False,
                                    download=True,
                                    transform=test_transform)

    # get dataloader
    test_sampler = SequentialSampler(testset)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=__C.dataset['eval_batch_size'],
                             num_workers=__C.dataloader['num_workers'],
                             pin_memory=__C.dataloader['pin_memory'],
                             drop_last=False)
    return test_loader
