"""
01b_train_fsq_ae.py - Part 4.1a (v2): discrete autoencoder with an 8x8x10 latent TENSOR,
every entry quantised to `levels` values (finite scalar quantisation, FSQ), plus a perceptual loss.

Why: the first VQ-VAE (1024 codes per position) reached 25.84 dB but its reconstruction FID was 47.6,
already above the FID < 40 target, so no generator on top of it could pass.

Latent  : 10 channels x 8 x 8, each entry in {0 .. levels-1}      (--levels 2 = pure binary, 640 bits)
Loss    : MSE + lpips_w * LPIPS   (LPIPS makes reconstructions sharper -> much lower FID)
Goal    : test PSNR > 25 dB

Needs   : pip install lpips      (downloads VGG weights once; or use --lpips_w 0 to skip)
Run     : python 01b_train_fsq_ae.py --epochs 60

Outputs (part4/outputs):  fsq_ae_best.pth, fsq_ae_last.pth (for --resume), fsq_ae_recon.png
"""

import argparse
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from common import (
    OUT_DIR,
    build_lpips,
    get_cifar10_loaders,
    get_device,
    psnr_per_image,
    save_grid,
    set_seed,
)
from models import FSQAE


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    psnrs = []
    for x, _ in loader:
        x = x.to(device)
        recon, _, idx, _ = model(x)
        psnrs.append(psnr_per_image(x, recon).cpu())
    return torch.cat(psnrs).mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument(
        "--channels", type=int, default=10
    )  # 10 -> latent tensor 8x8x10 (limit)
    p.add_argument("--levels", type=int, default=8)  # values per entry (2 = binary)
    p.add_argument("--num_res", type=int, default=2)
    p.add_argument("--lpips_w", type=float, default=0.1)  # 0 = plain MSE
    p.add_argument("--resume", action="store_true")
    args, _ = p.parse_known_args()

    set_seed(42)
    device = get_device()
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    train_loader, test_loader = get_cifar10_loaders(args.batch_size)
    lp = build_lpips(device) if args.lpips_w > 0 else None

    model = FSQAE(args.hidden, args.channels, args.levels, args.num_res).to(device)
    print(
        f"Parameters: {sum(p.numel() for p in model.parameters()):,} | "
        f"latent 8x8x{args.channels}, {args.levels} levels/entry = {model.bits_per_position:.0f} bits/position"
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-5
    )

    start_epoch, best_psnr = 1, 0.0
    last_path = OUT_DIR / "fsq_ae_last.pth"
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
        tot_mse = tot_lp = tot_ppl = 0.0
        n = 0
        for x, _ in tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False
        ):
            x = x.to(device, non_blocking=True)
            recon, _, _, ppl = model(x)
            mse = F.mse_loss(recon, x)
            loss = mse
            if lp is not None:
                lp_loss = lp(recon, x).mean()
                loss = loss + args.lpips_w * lp_loss
                tot_lp += lp_loss.item()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tot_mse += mse.item()
            tot_ppl += ppl.item()
            n += 1
        sched.step()

        test_psnr = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch:03d} | mse {tot_mse / n:.4f} | lpips {tot_lp / n:.4f} | "
            f"perplexity/channel {tot_ppl / n:.1f} of {args.levels} | test PSNR {test_psnr:.2f} dB | "
            f"{time.time() - t0:.0f}s"
        )

        ckpt = {
            "model": model.state_dict(),
            "config": model.config,
            "epoch": epoch,
            "psnr": test_psnr,
        }
        if test_psnr > best_psnr:
            best_psnr = test_psnr
            torch.save(ckpt, OUT_DIR / "fsq_ae_best.pth")
        torch.save(
            {
                **ckpt,
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "best_psnr": best_psnr,
            },
            last_path,
        )

    # ---- final report with the best checkpoint ----
    ck = torch.load(OUT_DIR / "fsq_ae_best.pth", map_location=device)
    model.load_state_dict(ck["model"])
    test_psnr = evaluate(model, test_loader, device)

    x, _ = next(iter(test_loader))
    x = x[:32].to(device)
    with torch.no_grad():
        recon = model(x)[0]
    save_grid(torch.cat([x, recon]), OUT_DIR / "fsq_ae_recon.png", nrow=8)

    print("=" * 60)
    print("SUMMARY (FSQ discrete AE)")
    print(
        f"Latent: 8x8x{args.channels} tensor, {args.levels} values per entry ({model.bits_per_position:.0f} bits/position)"
    )
    print(f"Best test PSNR: {test_psnr:.2f} dB (epoch {ck['epoch']})")
    print(
        "PASS: PSNR > 25 and latent tensor <= 8x8x10"
        if (test_psnr > 25 and args.channels <= 10)
        else "Below target: lower --lpips_w (e.g. 0.05) or train longer"
    )
    print("Next: python 03_extract_latents.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
