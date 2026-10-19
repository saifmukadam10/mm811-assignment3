"""Shared helpers used across all parts of the assignment."""

import os
import torch


def get_device() -> torch.device:
    """Pick GPU if available, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    """Save just the model weights (state_dict), creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(
    model: torch.nn.Module, path: str, device: torch.device
) -> torch.nn.Module:
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def psnr(mse: float, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio, derived directly from MSE.

    Images here are normalized to [0, 1] (torchvision's ToTensor), so max_val=1.0.
    PSNR = 10 * log10(MAX^2 / MSE)
    """
    if mse <= 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(max_val**2 / mse)).item()
