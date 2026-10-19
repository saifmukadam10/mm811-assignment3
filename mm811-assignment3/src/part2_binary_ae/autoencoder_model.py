import torch
import torch.nn as nn


# BinarizeFunction
class BinarizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.sign()  # outputs -1 or +1

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # straight-through estimator


def binarize(x: torch.Tensor) -> torch.Tensor:
    return BinarizeFunction.apply(x)


class AutoEncoder(nn.Module):
    def __init__(self, use_binary: bool = True):
        super().__init__()
        self.use_binary = use_binary

        # Encoder: 1×32×32 → 3×4×4 (no final tanh; decoder sees ±1 when use_binary=True)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),  # 1×32×32 → 32×32×32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),  # → 64×16×16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # → 128×8×8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),  # → 256×4×4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, 3, 1, 1),  # → 3×4×4 (latent logits)
        )

        # Decoder: 3×4×4 → 1×32×32
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 128, 4, 2, 1),  # → 128×8×8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # → 64×16×16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # → 32×32×32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1),  # → 1×32×32
            nn.Tanh(),  # output ∈ [−1, 1]
        )

    def forward(self, x):
        latent_logits = self.encoder(x)  # (N, 3, 4, 4), real-valued
        if self.use_binary:
            latent = binarize(latent_logits)  # strictly ±1
        else:
            latent = latent_logits  # warm-up (no binarization)
        recon = self.decoder(latent)
        # Return both for training utilities
        return recon, latent, latent_logits
