# %% [markdown]
# # Step 04 - Discrete latent diffusion (Part 4.2)
#
# Learns the distribution of the discrete latents (8x8 positions x 10 entries, each entry in 0..levels-1)
# with an absorbing-state ("masked") diffusion model:
#   * forward process : entries are randomly replaced by [MASK]
#   * denoiser        : bidirectional transformer over the 64 positions (the 10 entries of a position are
#                       embedded together) predicting the original entries
#   * sampling        : start from all [MASK], unmask the most confident entries step by step
#
# Needs  : vq_latents.pt (step 03), fsq_ae_best.pth / vqvae_best.pth (only for the preview images)
# Writes : ddiff_best.pth, ddiff_last.pth, ddiff_samples_epochXXX.png   (in part4/outputs)
#
# Works both ways:   python 04_train_discrete_diffusion.py --epochs 80
#                    or run the "# %%" cells one by one in VS Code (last cell calls main()).

# %%
import argparse
import math
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from common import (EMA, TimeBudget, OUT_DIR, amp_ctx, find_file, fixed_rng, get_device, load_discrete_ae,
                    save_grid, set_seed, warmup_cosine)
from models import MaskedTokenDenoiser


# %% [markdown]
# ## 1. Load the outputs of step 03

# %%
def load_step03_outputs():
    """Returns train (100000, 64, S) long [train + flipped train], test (10000, 64, S), meta dict."""
    lat = torch.load(find_file("vq_latents.pt"), map_location="cpu")
    S = lat["n_sub"]
    train = torch.cat([lat["train"], lat["train_flip"]]).long().reshape(-1, 64, S)
    test = lat["test"].long().reshape(-1, 64, S)
    meta = {"vocab": lat["vocab"], "n_sub": S, "kind": lat["kind"]}
    print(f"Loaded vq_latents.pt ({meta['kind']}): train {tuple(train.shape)}, test {tuple(test.shape)}, "
          f"vocab {meta['vocab']} per entry")
    return train, test, meta


# %% [markdown]
# ## 2. Forward (masking) process and evaluation

# %%
def mask_batch(x0, mask_id):
    """Mask a random number of ENTRIES per sample (MaskGIT cosine schedule).
    x0: (B, L, S). Returns the corrupted input and the boolean mask of corrupted entries (both (B, L, S))."""
    B = x0.size(0)
    flat = x0.reshape(B, -1)
    N = flat.size(1)
    u = torch.rand(B, device=x0.device)
    n_mask = (torch.cos(math.pi / 2 * u) * N).ceil().long().clamp(1, N)
    ranks = torch.rand(B, N, device=x0.device).argsort(dim=1).argsort(dim=1)
    mask = ranks < n_mask.unsqueeze(1)
    inp = torch.where(mask, torch.full_like(flat, mask_id), flat)
    return inp.view_as(x0), mask.view_as(x0)


@torch.no_grad()
def evaluate(model, tokens, device, bs=500):
    model.eval()
    tot_loss = tot_correct = tot_n = 0
    with fixed_rng(0, device):                       # same masks every epoch -> comparable numbers
        for i in range(0, tokens.size(0), bs):
            x0 = tokens[i:i + bs]
            inp, mask = mask_batch(x0, model.mask_id)
            with amp_ctx(device):
                logits = model(inp)
            lg, tgt = logits.float()[mask], x0[mask]
            tot_loss += F.cross_entropy(lg, tgt, reduction="sum").item()
            tot_correct += (lg.argmax(-1) == tgt).sum().item()
            tot_n += tgt.numel()
    return tot_loss / tot_n, tot_correct / tot_n


@torch.no_grad()
def preview(model, dae, device, path, steps=32):
    """Sample 64 latents, decode them with the autoencoder's decoder, save an image grid."""
    with amp_ctx(device):
        tokens = model.sample(64, steps=steps, device=device)
    save_grid(dae.decode_tokens(tokens), path, nrow=8)


# %% [markdown]
# ## 3. Training

# %%
def train(args):
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    train_tokens, test_tokens, meta = load_step03_outputs()
    train_tokens, test_tokens = train_tokens.to(device), test_tokens.to(device)
    dae, kind = load_discrete_ae(device, meta["kind"])

    model = MaskedTokenDenoiser(meta["vocab"], 64, args.d_model, args.nhead, args.layers,
                                dropout=args.dropout, n_sub=meta["n_sub"]).to(device)
    ema = EMA(model, args.ema)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    steps_per_epoch = train_tokens.size(0) // args.batch_size
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(args.warmup, args.epochs * steps_per_epoch))

    start_epoch, best_val = 1, float("inf")
    elapsed_prev = 0.0
    last_path = OUT_DIR / "ddiff_last.pth"
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        ema.model.load_state_dict(ck["ema"])
        ema.step = ck["ema_step"]
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best_val = ck["epoch"] + 1, ck["best_val"]
        elapsed_prev = ck.get("elapsed", 0.0)
        print(f"Resumed from epoch {ck['epoch']} (best val loss {best_val:.4f})")

    tb = TimeBudget(args.time_budget, args.warmup, args.lr, elapsed_prev) if args.time_budget > 0 else None
    stop = False
    for epoch in range(start_epoch, args.epochs + 1):
        if stop:
            break
        model.train()
        t0 = time.time()
        perm = torch.randperm(train_tokens.size(0), device=device)
        run_loss = 0.0
        n_done = 0
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for b in pbar:
            x0 = train_tokens[perm[b * args.batch_size:(b + 1) * args.batch_size]]
            if tb is not None:
                if tb.done():
                    stop = True
                    break
                for g in opt.param_groups:
                    g['lr'] = tb.lr(ema.step)
            inp, mask = mask_batch(x0, model.mask_id)
            with amp_ctx(device):
                logits = model(inp)
            loss = F.cross_entropy(logits.float()[mask], x0[mask])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if tb is None:
                sched.step()
            ema.update(model)

            run_loss += loss.item()
            n_done += 1
            if b % 50 == 0:
                pbar.set_postfix(loss=run_loss / (b + 1))

        if n_done == 0:
            break
        val_loss, val_acc = evaluate(ema.model, test_tokens, device)
        print(f"Epoch {epoch:03d} | train loss {run_loss/n_done:.4f} | "
              f"val loss (EMA) {val_loss:.4f} | val masked-entry acc {val_acc:.3f} | "
              f"lr {opt.param_groups[0]['lr']:.2e} | {time.time()-t0:.0f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": ema.model.state_dict(), "config": model.config, "kind": kind,
                        "epoch": epoch, "val_loss": val_loss}, OUT_DIR / "ddiff_best.pth")
        torch.save({"model": model.state_dict(), "ema": ema.model.state_dict(), "ema_step": ema.step,
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch,
                    "best_val": best_val, "elapsed": tb.elapsed() if tb is not None else 0.0, "config": model.config, "kind": kind}, last_path)

        if epoch % args.sample_every == 0 or epoch == args.epochs or stop:
            preview(ema.model, dae, device, OUT_DIR / f"ddiff_samples_epoch{epoch:03d}.png")

    print("=" * 60)
    print("SUMMARY (discrete latent diffusion)")
    print(f"Best val loss (EMA): {best_val:.4f}  -> {OUT_DIR / 'ddiff_best.pth'}")
    print("Next: run 06_generate_and_eval.py (after 05 is also finished)")
    print("=" * 60)


# %% [markdown]
# ## 4. Run

# %%
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)          # optimiser steps
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--sample_every", type=int, default=20)     # epochs between preview image grids
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--time_budget", type=float, default=0.0)   # minutes. >0: train for exactly this long (overrides --epochs)
    args, _ = p.parse_known_args()                              # parse_known_args: safe inside notebooks
    if args.time_budget > 0:
        args.epochs = 100000
    train(args)


if __name__ == "__main__":
    main()
