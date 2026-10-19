"""Part 1 training entry point.

Run from the project root:
    python -m src.part1_classification.train
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.datasets import get_mnist_loaders
from src.utils import get_device, save_checkpoint
from src.part1_classification.model import MNISTClassifier


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train(
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    subset_size: int = 256,
    data_dir: str = "./data",
    ckpt_path: str = "./checkpoints/part1_classifier.pt",
):
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader = get_mnist_loaders(
        batch_size=batch_size, data_dir=data_dir, subset_size=subset_size
    )

    model = MNISTClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False
        ):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        test_acc = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch:02d}: train_loss={train_loss:.4f}  test_acc={test_acc * 100:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model, ckpt_path)

    print(f"\nBest test accuracy: {best_acc * 100:.2f}%  (target: >90%)")
    return best_acc


if __name__ == "__main__":
    train()
