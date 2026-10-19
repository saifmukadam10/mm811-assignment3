"""
common.py - shared helpers for Part 4 (CIFAR-10 latent generative models).

Used by every script in this folder. Nothing to run here directly.

LOCAL USE: every output (checkpoints, latents, images, CIFAR-10 data) goes to
    <this folder>/outputs/
Override with the environment variable P4_OUT.  If you trained steps 01-03 somewhere else
(e.g. on Kaggle), just copy vqvae_best.pth, cont_ae_best.pth, vq_latents.pt, cont_latents.pt
into that outputs/ folder.
"""

import contextlib
import copy
import glob
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if os.environ.get("P4_OUT"):
    OUT_DIR = Path(os.environ["P4_OUT"])
elif Path("/kaggle/working").exists():
    OUT_DIR = Path("/kaggle/working")
else:
    OUT_DIR = _HERE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = os.environ.get("CIFAR_ROOT", str(OUT_DIR / "data"))


def find_file(name):
    """Find a file produced by an earlier step: OUT_DIR, current dir, this folder, then Kaggle inputs."""
    for d in (OUT_DIR, Path.cwd(), _HERE):
        if (d / name).exists():
            return d / name
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if hits:
        return Path(hits[0])
    raise FileNotFoundError(
        f"Could not find '{name}' (looked in {OUT_DIR}). Did you run the previous step, "
        f"or copy its output file into that folder?"
    )


# ----------------------------------------------------------------------------
# Reproducibility / device / mixed precision
# ----------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_ctx(device):
    """bf16 autocast on CUDA (RTX 40-series supports it, no GradScaler needed); no-op on CPU."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@contextlib.contextmanager
def fixed_rng(seed, device):
    """Temporarily fix the random seed (used so validation losses are comparable between epochs)."""
    devs = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devs):
        torch.manual_seed(seed)
        yield


# ----------------------------------------------------------------------------
# Training helpers
# ----------------------------------------------------------------------------
class EMA:
    """Exponential moving average of model weights. Use `ema.model` for evaluation / sampling."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.step = 0
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1 - d)
        for be, b in zip(self.model.buffers(), model.buffers()):
            be.copy_(b)


def warmup_cosine(warmup, total, min_ratio=0.01):
    """LR multiplier per optimiser step: linear warm-up, then cosine decay to min_ratio."""

    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total - warmup)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    return f


class TimeBudget:
    """Train for a fixed wall-clock time (minutes). The LR schedule (warm-up + cosine) follows the clock,
    so training always ends on time, whatever the GPU speed. Use via --time_budget in steps 04 / 05."""

    def __init__(self, minutes, warmup, base_lr, elapsed_prev=0.0, min_ratio=0.01):
        self.total = minutes * 60.0
        self.warmup, self.base_lr, self.min_ratio = warmup, base_lr, min_ratio
        self.prev, self.t0 = elapsed_prev, time.time()

    def elapsed(self):
        return self.prev + time.time() - self.t0

    def done(self):
        return self.elapsed() >= self.total

    def lr(self, step):
        if step < self.warmup:
            return self.base_lr * (step + 1) / self.warmup
        frac = min(1.0, self.elapsed() / self.total)
        return self.base_lr * (
            self.min_ratio + (1 - self.min_ratio) * 0.5 * (1 + math.cos(math.pi * frac))
        )


# ----------------------------------------------------------------------------
# Data  (images are scaled to [-1, 1] everywhere in Part 4)
# ----------------------------------------------------------------------------
def get_cifar10_loaders(batch_size=128, num_workers=2, augment=True):
    norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    train_tf = [transforms.RandomHorizontalFlip()] if augment else []
    train_tf = transforms.Compose(train_tf + [transforms.ToTensor(), norm])
    test_tf = transforms.Compose([transforms.ToTensor(), norm])

    train_set = datasets.CIFAR10(
        DATA_ROOT, train=True, download=True, transform=train_tf
    )
    test_set = datasets.CIFAR10(
        DATA_ROOT, train=False, download=True, transform=test_tf
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def get_cifar10_eval_loaders(batch_size=500, num_workers=2):
    """Ordered, non-augmented, nothing dropped. Used for latent extraction (03) and FID (06)."""
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_set = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=tf)
    mk = lambda ds: DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return mk(train_set), mk(test_set)


# ----------------------------------------------------------------------------
# Loading the autoencoders trained in steps 01 / 02
# ----------------------------------------------------------------------------
def load_vqvae(device):
    from models import VQVAE

    ck = torch.load(find_file("vqvae_best.pth"), map_location=device)
    m = VQVAE(**ck["config"]).to(device)
    m.load_state_dict(ck["model"])
    return m.eval()


def load_cont_ae(device):
    from models import ContinuousAE

    ck = torch.load(find_file("cont_ae_best.pth"), map_location=device)
    m = ContinuousAE(**ck["config"]).to(device)
    m.load_state_dict(ck["model"])
    return m.eval()


def discrete_ae_kind(kind="auto"):
    """'auto' -> 'fsq' if fsq_ae_best.pth exists, else 'vq'."""
    if kind != "auto":
        return kind
    try:
        find_file("fsq_ae_best.pth")
        return "fsq"
    except FileNotFoundError:
        return "vq"


def load_discrete_ae(device, kind="auto"):
    """Returns (model, kind). kind in {'auto', 'fsq', 'vq'}."""
    from models import FSQAE, VQVAE

    kind = discrete_ae_kind(kind)
    name, cls = (
        ("fsq_ae_best.pth", FSQAE) if kind == "fsq" else ("vqvae_best.pth", VQVAE)
    )
    ck = torch.load(find_file(name), map_location=device)
    m = cls(**ck["config"]).to(device)
    m.load_state_dict(ck["model"])
    return m.eval(), kind


def build_lpips(device):
    """Perceptual loss (LPIPS, VGG). Inputs in [-1, 1].  pip install lpips"""
    try:
        import lpips
    except ImportError as e:
        raise ImportError(
            "LPIPS is not installed: run  pip install lpips  (or pass --lpips_w 0)"
        ) from e
    m = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# ----------------------------------------------------------------------------
# Metrics / visualisation
# ----------------------------------------------------------------------------
def to_unit(x):
    """[-1, 1] -> [0, 1]"""
    return (x.clamp(-1, 1) + 1) / 2


def psnr_per_image(x, y):
    """PSNR (dB) per image. x, y in [-1, 1]. Computed on the [0, 1] scale (MAX = 1)."""
    mse = ((to_unit(x) - to_unit(y)) ** 2).flatten(1).mean(dim=1).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse)


def save_grid(tensor_m1_1, path, nrow=8):
    save_image(to_unit(tensor_m1_1), str(path), nrow=nrow, padding=2)
    print(f"Saved image grid -> {path}")
