"""
02_train_continuous_ae.py - Part 4.1b (v2): continuous autoencoder on CIFAR-10.

Latent: 8x8x5 floats = 320 values (assignment limit for continuous AEs).
Goal  : test PSNR > 25 dB.

v2 changes (the v1 AE had reconstruction FID 31.05, leaving only ~9 FID points for the generator):
  * --lpips_w   perceptual loss (LPIPS) -> sharper reconstructions -> lower FID
  * --noise_rel Gaussian noise added to the latent while training (relative to its std), so the decoder
                tolerates the small errors a generator makes -> better generated images

Needs   : pip install lpips      (or pass --lpips_w 0)
Run     : python 02_train_continuous_ae.py --epochs 60

Outputs (part4/outputs):  cont_ae_best.pth, cont_ae_last.pth (for --resume), cont_ae_recon.png
NOTE: this OVERWRITES cont_ae_best.pth from v1 -> copy your old outputs folder first.
"""
import argparse
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from common import (OUT_DIR, build_lpips, get_cifar10_loaders, get_device, psnr_per_image,
                    save_grid, set_seed)
from models import ContinuousAE


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    psnrs, zs = [], []
    for x, _ in loader:
        x = x.to(device)
        recon, z = model(x)
        psnrs.append(psnr_per_image(x, recon).cpu())
        zs.append(z.cpu())
    z = torch.cat(zs)
    return torch.cat(psnrs).mean().item(), z.mean().item(), z.std().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--latent_ch", type=int, default=5)       # 5 -> 8x8x5 = 320 (limit)
    p.add_argument("--num_res", type=int, default=2)
    p.add_argument("--latent_reg", type=float, default=1e-4)  # tiny L2 on latents: stops the scale from drifting
    p.add_argument("--lpips_w", type=float, default=0.1)      # 0 = plain MSE (v1 behaviour)
    p.add_argument("--noise_rel", type=float, default=0.1)    # train-time latent noise, relative to latent std
    p.add_argument("--resume", action="store_true")
    args, _ = p.parse_known_args()

    set_seed(42)
    device = get_device()
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    train_loader, test_loader = get_cifar10_loaders(args.batch_size)
    lp = build_lpips(device) if args.lpips_w > 0 else None

    model = ContinuousAE(args.hidden, args.latent_ch, args.num_res).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    start_epoch, best_psnr = 1, 0.0
    last_path = OUT_DIR / "cont_ae_last.pth"
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
        tot_mse = tot_lp = 0.0
        n = 0
        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x = x.to(device, non_blocking=True)
            z = model.encoder(x)
            z_in = z
            if args.noise_rel > 0:
                z_in = z + args.noise_rel * z.detach().std() * torch.randn_like(z)
            recon = model.decoder(z_in)

            mse = F.mse_loss(recon, x)
            loss = mse + args.latent_reg * z.pow(2).mean()
            if lp is not None:
                lp_loss = lp(recon, x).mean()
                loss = loss + args.lpips_w * lp_loss
                tot_lp += lp_loss.item()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tot_mse += mse.item()
            n += 1
        sched.step()

        test_psnr, z_mean, z_std = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:03d} | mse {tot_mse/n:.4f} | lpips {tot_lp/n:.4f} | "
              f"latent mean {z_mean:+.2f} std {z_std:.2f} | test PSNR {test_psnr:.2f} dB | {time.time()-t0:.0f}s")

        ckpt = {"model": model.state_dict(), "config": model.config,
                "epoch": epoch, "psnr": test_psnr}
        if test_psnr > best_psnr:
            best_psnr = test_psnr
            torch.save(ckpt, OUT_DIR / "cont_ae_best.pth")
        torch.save({**ckpt, "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "best_psnr": best_psnr}, last_path)

    # ---- final report with the best checkpoint ----
    ck = torch.load(OUT_DIR / "cont_ae_best.pth", map_location=device)
    model.load_state_dict(ck["model"])
    test_psnr, z_mean, z_std = evaluate(model, test_loader, device)

    x, _ = next(iter(test_loader))
    x = x[:32].to(device)
    with torch.no_grad():
        recon = model(x)[0]
    save_grid(torch.cat([x, recon]), OUT_DIR / "cont_ae_recon.png", nrow=8)

    size = args.latent_ch * 8 * 8
    print("=" * 60)
    print("SUMMARY (Continuous AE)")
    print(f"Latent: 8x8x{args.latent_ch} = {size} floats (limit 8x8x5 = 320)")
    print(f"Best test PSNR: {test_psnr:.2f} dB (epoch {ck['epoch']})")
    print(f"Latent stats: mean {z_mean:+.2f}, std {z_std:.2f}")
    print("PASS: PSNR > 25 and latent <= 8x8x5" if (test_psnr > 25 and size <= 320)
          else "Below target: lower --lpips_w / --noise_rel, or use --hidden 256 --epochs 100")
    print("Next: python 03_extract_latents.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
