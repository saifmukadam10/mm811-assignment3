# MM811 Programming Assignment 3

Deep learning assignment: classification, binary-latent autoencoder,
autoregressive models, image generation. Due Oct 19.

## Setup

```bash
pip install -r requirements.txt
```

## Part 1 — Image Classification (40%, target >90% accuracy)

```bash
python -m src.part1_classification.train
```

Trains a small CNN on an MNIST subset (default 256 images) and reports
test accuracy each epoch. Best checkpoint saved to
`checkpoints/part1_classifier.pt`.

To scale up (more images, more epochs), edit the call at the bottom of
`src/part1_classification/train.py` or import `train()` and pass your own
`subset_size`, `epochs`, etc.

## Part 2 — Binary Latent Autoencoder (30%, target PSNR>25)

```bash
python -m src.part2_binary_ae.train
```

Trains an autoencoder whose bottleneck is binarized to {-1, +1} via a
Straight-Through Estimator, latent shape `N x 3 x 4 x 4` as required by the
assignment. Training uses the full MNIST training set for up to 150 epochs
and reduces the learning rate when test loss plateaus. Reports test PSNR
each epoch. Best checkpoint saved to `checkpoints/part2_binary_ae.pt`.

## Notes

- Both scripts auto-download MNIST into `./data/` on first run.
- `subset_size` controls how many *training* images are used (the
  assignment allows 256 / 1024 / full); the test set is always kept full
  so accuracy/PSNR numbers are stable to report.
- Parts 3 and 4 are implemented in notebooks under `src/Part3/` and
  `src/Part4/`.

## Run the Part 4.1 continuous autoencoder locally (Windows)

From the repository root, install the CUDA-enabled PyTorch wheels into the
project virtual environment. These wheels match the project's PyTorch and
torchvision versions and the CUDA 13.0-capable NVIDIA driver:

```powershell
uv pip install --python .venv\Scripts\python.exe `
  --extra-index-url https://download.pytorch.org/whl/cu130 `
  "torch==2.14.0+cu130" "torchvision==0.29.1+cu130"
```

In VS Code, open `mm811-assignment3/src/Part4/02_train_continuous_ae.ipynb`,
select the project's `.venv` as the notebook kernel, then run all cells.
The notebook uses CUDA by default and stops with an explanatory error if
the selected PyTorch environment cannot access it. CIFAR-10 is downloaded
to `mm811-assignment3/data/CIFAR10` on first run; checkpoints and
reconstruction images are saved under `mm811-assignment3/results/part4`.
Set the `CIFAR_ROOT` environment variable before launching VS Code if you
already have CIFAR-10 stored elsewhere.

To verify GPU access in the selected kernel:

```python
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

## Run Part 2 in a Kaggle Notebook

Enable a GPU in the notebook's accelerator settings for faster training.
From a notebook cell, clone or otherwise place this project under
`/kaggle/working`, then run:

```python
%cd /kaggle/working/mm811-assignment3

from src.part2_binary_ae.train import train

best_psnr = train(
    epochs=150,
    batch_size=128,
    data_dir="/kaggle/working/data",
    ckpt_path="/kaggle/working/checkpoints/part2_binary_ae.pt",
    num_workers=2,
)
```

The training code automatically uses CUDA when available, enables pinned
memory for GPU transfers, and uses notebook-aware progress bars. MNIST is
downloaded to `data_dir`; if Kaggle Internet is disabled, enable Internet or
attach MNIST in torchvision's expected directory layout under that path.
Files saved under `/kaggle/working` are available from the notebook's Output
panel when the session finishes.

## Report

Keep `report/report_skeleton.md` updated as you finish each part, then
port it into the assignment's Overleaf/LaTeX template for submission.
