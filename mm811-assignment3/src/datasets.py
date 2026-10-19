"""Dataset loading helpers.

The assignment lets you work on a subset of MNIST (256 / 1024 / full).
`subset_size` controls how many *training* images are used; the test set
is always kept full so your reported accuracy/PSNR is a fair, stable number.
"""

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_mnist_loaders(
    batch_size: int = 128,
    data_dir: str = "./data",
    subset_size: int | None = None,
    num_workers: int = 2,
    pin_memory: bool = False,
    persistent_workers: bool = False,
):
    transform = transforms.Compose([transforms.ToTensor()])  # -> [0, 1] range

    train_set = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    if subset_size is not None:
        train_set = Subset(train_set, list(range(subset_size)))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    return train_loader, test_loader
