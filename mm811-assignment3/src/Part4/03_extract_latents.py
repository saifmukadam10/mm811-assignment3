"""
03_extract_latents.py - encode CIFAR-10 once with both trained autoencoders and save the latents.

Needs (from steps 01b / 02, found automatically in part4/outputs):
    fsq_ae_best.pth (or vqvae_best.pth with --discrete vq)     and     cont_ae_best.pth

Run:   python 03_extract_latents.py

Outputs (part4/outputs):
    vq_latents.pt     {"train": (50000,8,8,S), "train_flip", "test", "vocab", "n_sub", "kind", "config"}
                      integer codes (uint8/int16 to save space; 04 converts to long). S = 10 for FSQ, none for VQ.
    cont_latents.pt   {"train": (50000,C,8,8) float, "train_flip", "test", "mean", "std", "config"}

"train_flip" = latents of horizontally flipped training images (free data augmentation).
It also prints the PSNR of "image -> latents -> decode", which is the best any generator can do.
"""
import argparse

import torch

from common import (OUT_DIR, get_cifar10_eval_loaders, get_device, load_cont_ae, load_discrete_ae,
                    psnr_per_image, set_seed)


@torch.no_grad()
def extract(loader, dae, cont, device, idx_dtype, flip=False, measure=False):
    idx_all, z_all = [], []
    d_psnr, c_psnr = [], []
    for x, _ in loader:
        x = x.to(device)
        if flip:
            x = torch.flip(x, dims=[3])
        idx = dae.encode(x)
        z = cont.encode(x)
        idx_all.append(idx.cpu().to(idx_dtype))
        z_all.append(z.cpu())
        if measure:   # round trip: image -> latent -> decoder
            d_psnr.append(psnr_per_image(x, dae.decode_codes(idx)).cpu())
            c_psnr.append(psnr_per_image(x, cont.decode(z)).cpu())
    out = (torch.cat(idx_all), torch.cat(z_all).float())
    if measure:
        return out + (torch.cat(d_psnr).mean().item(), torch.cat(c_psnr).mean().item())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--discrete", choices=["auto", "fsq", "vq"], default="auto")
    args, _ = p.parse_known_args()

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    dae, kind = load_discrete_ae(device, args.discrete)
    cont = load_cont_ae(device)
    print(f"Discrete AE: {kind} | vocab per entry {dae.vocab} | sub-tokens per position {dae.n_sub}")

    idx_dtype = torch.uint8 if dae.vocab <= 256 else torch.int16
    train_loader, test_loader = get_cifar10_eval_loaders(batch_size=500)

    print("Encoding train set ...")
    d_train, c_train = extract(train_loader, dae, cont, device, idx_dtype)
    print("Encoding flipped train set ...")
    d_flip, c_flip = extract(train_loader, dae, cont, device, idx_dtype, flip=True)
    print("Encoding test set ...")
    d_test, c_test, d_psnr, c_psnr = extract(test_loader, dae, cont, device, idx_dtype, measure=True)

    mean = c_train.mean(dim=(0, 2, 3))
    std = c_train.std(dim=(0, 2, 3))

    torch.save({"train": d_train, "train_flip": d_flip, "test": d_test, "vocab": dae.vocab,
                "num_codes": dae.vocab, "n_sub": dae.n_sub, "kind": kind, "config": dae.config},
               OUT_DIR / "vq_latents.pt")
    torch.save({"train": c_train, "train_flip": c_flip, "test": c_test,
                "mean": mean, "std": std, "config": cont.config},
               OUT_DIR / "cont_latents.pt")

    flat = d_train.long().reshape(-1, dae.n_sub)
    ppl = []
    for c in range(dae.n_sub):
        p_ = torch.bincount(flat[:, c], minlength=dae.vocab).float()
        p_ = p_ / p_.sum()
        ppl.append(torch.exp(-(p_ * torch.log(p_ + 1e-10)).sum()).item())

    print("=" * 60)
    print("SUMMARY (latent extraction)")
    print(f"vq_latents.pt   : train {tuple(d_train.shape)}, test {tuple(d_test.shape)}, "
          f"{kind}, vocab {dae.vocab} per entry, dtype {idx_dtype}")
    print(f"                  perplexity per sub-token (max {dae.vocab}): {[round(v, 1) for v in ppl]}")
    print(f"cont_latents.pt : train {tuple(c_train.shape)}, test {tuple(c_test.shape)}")
    print(f"                  per-channel mean {[round(v, 2) for v in mean.tolist()]}")
    print(f"                  per-channel std  {[round(v, 2) for v in std.tolist()]}")
    print(f"Round-trip test PSNR (image -> latent -> decode): "
          f"discrete {d_psnr:.2f} dB | continuous {c_psnr:.2f} dB")
    print("=" * 60)


if __name__ == "__main__":
    main()
