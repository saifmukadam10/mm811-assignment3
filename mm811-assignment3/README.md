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

  `src/Part4/`.
ssignment's Overleaf/LaTeX template for submission.
