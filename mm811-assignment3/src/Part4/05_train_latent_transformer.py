# %% [markdown]
# # Step 05 - Continuous latent transformer (Part 4.3)
#
# Learns the distribution of the continuous 8x8x5 latents with an autoregressive transformer:
#   * the 8x8 grid is read in raster order, every position is one 5-dim vector
#   * a CAUSAL transformer turns the previous vectors into a condition h_i
#   * a small diffusion head models p(x_i | h_i)  (so we never regress to the blurry mean)
#
# Needs  : cont_latents.pt (step 03), cont_ae_best.pth (step 02, only for the preview images)
# Writes : ltrans_best.pth (also stores the latent mean/std), ltrans_last.pth, ltrans_samples_epochXXX.png
#
# Works both ways:   python 05_train_latent_transformer.py --epochs 100
#                    or run the "# %%" cells one by one in VS Code (last cell calls main()).

# %%
import argparse
import time

import torch
from tqdm import tqdm

from common import (EMA, TimeBudget, OUT_DIR, amp_ctx, find_file, fixed_rng, get_device, load_cont_ae,
                    save_grid, set_seed, warmup_cosine)
from models import ARLatentTransformer


# %% [markdown]
# ## 1. Load the outputs of step 03

# %%
def load_step03_outputs():
    """Returns normalised sequences train (100000, 64, C), test (10000, 64, C) plus mean/std (C,)."""
    lat = torch.load(find_file("cont_latents.pt"), map_location="cpu")
    mean, std = lat["mean"], lat["std"]

    def prep(z):                                   # (N, C, 8, 8) -> normalised (N, 64, C), raster order
        z = (z - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
        return z.permute(0, 2, 3, 1).reshape(z.size(0), 64, -1).contiguous()

    train = torch.cat([prep(lat["train"]), prep(lat["train_flip"])])
    test = prep(lat["test"])
    print(f"Loaded cont_latents.pt: train {tuple(train.shape)}, test {tuple(test.shape)}")
    print(f"Normalised train latents: mean {train.mean():+.3f}, std {train.std():.3f}")
    return train, test, mean, std


def to_images(x_norm, cont_ae, mean, std):
    """Generated normalised latents (n, 64, C) -> images in [-1, 1]."""
    n, _, C = x_norm.shape
    z = x_norm.view(n, 8, 8, C).permute(0, 3, 1, 2)
    z = z * std.view(1, -1, 1, 1).to(z.device) + mean.view(1, -1, 1, 1).to(z.device)
    return cont_ae.decode(z.float())


# %% [markdown]
# ## 2. Evaluation

# %%
@torch.no_grad()
def evaluate(model, seqs, device, bs=500, t_repeat=8):
    model.eval()
    total, n = 0.0, 0
    with fixed_rng(0, device):                       # same noise every epoch -> comparable numbers
        for i in range(0, seqs.size(0), bs):
            with amp_ctx(device):
                loss = model.loss(seqs[i:i + bs], t_repeat=t_repeat)
            total += loss.item()
            n += 1
    return total / n


@torch.no_grad()
def preview(model, cont_ae, mean, std, device, path, steps=50):
    with amp_ctx(device):
        x = model.sample(64, steps=steps, device=device)
    save_grid(to_images(x, cont_ae, mean, std), path, nrow=8)


# %% [markdown]
# ## 3. Training

# %%
def train(args):
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    train_seqs, test_seqs, mean, std = load_step03_outputs()
    train_seqs, test_seqs = train_seqs.to(device), test_seqs.to(device)
    cont_ae = load_cont_ae(device)

    model = ARLatentTransformer(latent_ch=train_seqs.size(-1), seq_len=64, d_model=args.d_model,
                                nhead=args.nhead, num_layers=args.layers, dropout=args.dropout,
                                head_width=args.head_width, head_depth=args.head_depth).to(device)
    ema = EMA(model, args.ema)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    steps_per_epoch = train_seqs.size(0) // args.batch_size
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(args.warmup, args.epochs * steps_per_epoch))

    start_epoch, best_val = 1, float("inf")
    elapsed_prev = 0.0
    last_path = OUT_DIR / "ltrans_last.pth"
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
        perm = torch.randperm(train_seqs.size(0), device=device)
        run_loss = 0.0
        n_done = 0
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for b in pbar:
            x = train_seqs[perm[b * args.batch_size:(b + 1) * args.batch_size]]
            if tb is not None:
                if tb.done():
                    stop = True
                    break
                for g in opt.param_groups:
                    g['lr'] = tb.lr(ema.step)
            with amp_ctx(device):
                loss = model.loss(x, t_repeat=args.t_repeat)

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
        val_loss = evaluate(ema.model, test_seqs, device, t_repeat=args.eval_t_repeat)
        print(f"Epoch {epoch:03d} | train loss {run_loss/n_done:.4f} | "
              f"val loss (EMA) {val_loss:.4f} | lr {opt.param_groups[0]['lr']:.2e} | {time.time()-t0:.0f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": ema.model.state_dict(), "config": model.config, "epoch": epoch,
                        "val_loss": val_loss, "mean": mean.cpu(), "std": std.cpu()},
                       OUT_DIR / "ltrans_best.pth")
        torch.save({"model": model.state_dict(), "ema": ema.model.state_dict(), "ema_step": ema.step,
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch,
                    "best_val": best_val, "elapsed": tb.elapsed() if tb is not None else 0.0, "config": model.config}, last_path)

        if epoch % args.sample_every == 0 or epoch == args.epochs or stop:
            preview(ema.model, cont_ae, mean, std, device, OUT_DIR / f"ltrans_samples_epoch{epoch:03d}.png")

    print("=" * 60)
    print("SUMMARY (continuous latent transformer)")
    print(f"Best val loss (EMA): {best_val:.4f}  -> {OUT_DIR / 'ltrans_best.pth'}")
    print("Next: run 06_generate_and_eval.py (after 04 is also finished)")
    print("=" * 60)


# %% [markdown]
# ## 4. Run

# %%
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)          # optimiser steps
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--head_width", type=int, default=512)
    p.add_argument("--head_depth", type=int, default=4)
    p.add_argument("--t_repeat", type=int, default=4)          # diffusion-head draws per token per step
    p.add_argument("--eval_t_repeat", type=int, default=2)     # same, for the validation loss (cheaper)
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
