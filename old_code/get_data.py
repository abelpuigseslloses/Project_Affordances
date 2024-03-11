
import torch
from torch.utils.data import Subset
from torchvision import transforms


def make_datasets(dataset, test_size):
    """ Receives a dataset and test_size (from 0 to 1, e.g.: 0.1)
    and returns training and testing datasets"""

    indices = torch.randperm(len(dataset)).tolist()  # Permute all indices so that order is randomized
    train_size = int(len(indices) * (1-test_size))  # 90 % Training Data    ### CHANGE INTO HYPERPARAMETER ARG

    train_dataset = torch.utils.data.Subset(dataset, indices[:train_size])
    test_dataset = torch.utils.data.Subset(dataset, indices[train_size:])

    return train_dataset, test_dataset


def apply_transforms(dataset, image_size=(224, 224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Apply transformations to a dataset.

    Parameters:
    - dataset: The dataset to transform.
    - image_size: A tuple of the desired image size.
    - mean: The mean for normalization.
    - std: The standard deviation for normalization.

    Returns:
    - None, but modifies the dataset's transform attribute in place.
    """
    dataset.transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def make_dataloaders(train_dataset, test_dataset, batch_size=128, num_workers=2):
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)

    return train_loader, test_loader