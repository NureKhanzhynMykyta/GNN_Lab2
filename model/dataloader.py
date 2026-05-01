import torchvision.transforms as TF
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from configs import BaseConfig


def lam_func(t):
    return (t * 2) -1

def get_dataset(dataset_name='BaseConfig.DATASET'):
    transforms = TF.Compose(
        [
            TF.ToTensor(),
            TF.Resize((32, 32),
                      interpolation=TF.InterpolationMode.BICUBIC,
                      antialias=True),
            TF.RandomHorizontalFlip(),
            TF.Lambda(lam_func)
        ]
    )

    if dataset_name.upper() == "MNIST":
        dataset = datasets.MNIST(root="data", train=True, download=True, transform=transforms)
    elif dataset_name == "Cifar-10":
        dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transforms)
    elif dataset_name == "Cifar-100":
        dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transforms)
    elif dataset_name == "Flowers":
        dataset = datasets.ImageFolder(root="data/flowers", transform=transforms)

    return dataset

def inverse_transform(tensors):
    return ((tensors.clamp(-1,1) + 1.0) / 2.0) * 255.0

def get_dataloader(dataset_name=BaseConfig.DATASET,
                   batch_size=32,
                   pin_memory=False,
                   shuffle=True,
                   num_workers=0,
                   device=BaseConfig.DEVICE):
    dataset = get_dataset(dataset_name=dataset_name)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            pin_memory=pin_memory,
                            num_workers=num_workers,
                            shuffle=shuffle
                            )

    return dataloader
