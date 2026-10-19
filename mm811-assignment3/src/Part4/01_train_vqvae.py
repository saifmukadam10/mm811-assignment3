"""
01_train_vqvae.py - Part 4.1a: discrete autoencoder (VQ-VAE) on CIFAR-10.

Latent: 8x8 grid of codes from a 1024-entry codebook  ->  8x8x10 bits  (assignment limit).
Goal  : test PSNR > 25 dB.

Run (Kaggle cell):
    !python 01_train_vqvae.py --epochs 60

Outputs (in /kaggle/working):
    vqvae_best.pth        best checkpoint (by test PSNR) -> used by 03, 04, 06
    vqvae_last.pth        resumable checkpoint (use --resume after a timeout)
    vqvae_recon.png       top rows: originals, bottom rows: reconstructions
"""
import argparse
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from common import OUT_DIR, get_cifar10_loaders, get_device, psnr_per_image, save_grid, set_seed
from models import VQVAE


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    psnrs, used = [], set()
    for x, _ in loader:
        x = x.to(device)
        recon, _, idx, _ = model(x)
        psnrs.append(psnr_per_image(x, recon).cpu())
        used.update(idx.unique().tolist())
    return torch.cat(psnrs).mean().item(), len(used)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num_codes", type=int, default=1024)   # 1024 = 10 bits -> 8x8x10
    p.add_argument("--code_dim", type=int, default=64)
    p.add_argument("--num_res", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    args, _ = p.parse_known_args()

    set_seed(42)
    device = get_device()
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    train_loader, test_loader = get_cifar10_loaders(args.batch_size)

    model = VQVAE(args.hidden, args.num_codes, args.code_dim, args.num_res).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    start_epoch, best_psnr = 1, 0.0
    last_path = OUT_DIR / "vqvae_last.pth"
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best_psnr = ck["epoch"] + 1, ck["best_psnr"]
        print(f"Resumed from epoch {ck['epoch']} (best PSNR {best_psnr:.2f})")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot_rec = tot_vq = tot_ppl = 0.0
        n = 0
        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x = x.to(device, non_blocking=True)
            recon, vq_loss, _, ppl = model(x)
            rec_loss = F.mse_loss(recon, x)
            loss = rec_loss + vq_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tot_rec += rec_loss.item()
            tot_vq += vq_loss.item()
            tot_ppl += ppl.item()
            n += 1
        sched.step()

        test_psnr, n_used = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:03d} | rec {tot_rec/n:.4f} | vq {tot_vq/n:.4f} | "
              f"perplexity {tot_ppl/n:.0f} | codes used (test) {n_used}/{args.num_codes} | "
              f"test PSNR {test_psnr:.2f} dB | {time.time()-t0:.0f}s")

        ckpt = {"model": model.state_dict(), "config": model.config,
                "epoch": epoch, "psnr": test_psnr}
        if test_psnr > best_psnr:
            best_psnr = test_psnr
            torch.save(ckpt, OUT_DIR / "vqvae_best.pth")
        torch.save({**ckpt, "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "best_psnr": best_psnr}, last_path)

    # ---- final report with the best checkpoint ----
    ck = torch.load(OUT_DIR / "vqvae_best.pth", map_location=device)
    model.load_state_dict(ck["model"])
    test_psnr, n_used = evaluate(model, test_loader, device)

    x, _ = next(iter(test_loader))
    x = x[:32].to(device)
    with torch.no_grad():
        recon = model(x)[0]
    save_grid(torch.cat([x, recon]), OUT_DIR / "vqvae_recon.png", nrow=8)

    bits = 8 * 8 * torch.log2(torch.tensor(float(args.num_codes))).item()
    print("=" * 60)
    print("SUMMARY (VQ-VAE)")
    print(f"Latent: 8x8 grid, {args.num_codes} codes -> {bits:.0f} bits (limit 8x8x10 = 640)")
    print(f"Best test PSNR: {test_psnr:.2f} dB (epoch {ck['epoch']})")
    print(f"Codes used: {n_used}/{args.num_codes}")
    print("PASS: PSNR > 25 and latent <= 8x8x10" if (test_psnr > 25 and bits <= 640)
          else "Below target: try --hidden 256 --epochs 100 (see notes)")
    print("=" * 60)


if __name__ == "__main__":
    main()
