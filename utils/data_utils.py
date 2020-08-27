import pathlib

from torchvision import transforms
from torchvision.transforms import functional as F
from torch.utils.data import random_split

""" This file is used for various data-set utilities, e.g. generating a dataloader object.
"""


def load_dataset(dataset_name, **kwargs):
    """ Loads the specified dataset and returns a PyTorch dataset object.

    Applies the standard transformations for said dataset by default.
    """
    data_path = pathlib.Path('data').resolve()

    if dataset_name == 'cifar10':
        from torchvision.datasets import CIFAR10

        # This is the standard normalization transformation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # User can specify to load the training set; loads the test set by default.
        train = kwargs.pop('train', False)
        dataset = CIFAR10(data_path, train=train, transform=transform, download=True)
    elif dataset_name == 'cifar100':
        from torchvision.datasets import CIFAR100

        # This is the standard normalization transformation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # User can specify to load the training set; loads the test set by default.
        train = kwargs.pop('train', False)
        dataset = CIFAR100(data_path, train=train, transform=transform, download=True)
    else:
        raise NotImplementedError

    return dataset


def get_cal_eval_split(dataset_name, num_eval, **kwargs):
    """ Splits the given dataset into disjoint calibration / evaluation subsets.

    Args:
        dataset_name: str ;
        num_eval: int ; size of evaluation set
    """
    dataset = load_dataset(dataset_name)
    num_cal = len(dataset) - num_eval
    cal_dataset, eval_dataset = random_split(dataset, [num_cal, num_eval])

    return cal_dataset, eval_dataset

