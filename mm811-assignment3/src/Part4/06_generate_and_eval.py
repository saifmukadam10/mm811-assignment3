# %% [markdown]
# # Step 06 - Generate images and compute FID for both models
#
# Needs (all found automatically in part4/outputs):
#   fsq_ae_best.pth (or vqvae_best.pth), cont_ae_best.pth     (steps 01b/01, 02)
#   ddiff_best.pth                                            (step 04)
#   ltrans_best.pth                                           (step 05, also holds the latent mean/std)
# Writes: ddiff_final_samples.png, ltrans_final_samples.png, part4_results.json,
#         fid_real_stats_cifar10_train.pt  (cache: the real-image statistics are computed only once)
#
# One-time install (FID needs these):   pip install torchmetrics torch-fidelity
# The first FID run downloads the Inception weights, so it needs internet once.
#
# Works both ways:   python 06_generate_and_eval.py --num_gen 10000
#                    final numbers:   python 06_generate_and_eval.py --num_gen 50000
#                    sampler tuning:  python 06_generate_and_eval.py --tune
#                    autoencoders only (before training generators):  python 06_generate_and_eval.py --recon_only
#                    or run the "# %%" cells one by one in VS Code (last cell calls main()).

# %%
import argparse
import json
import time

import torch
from tqdm import tqdm

from common import (OUT_DIR, amp_ctx, find_file, get_cifar10_eval_loaders, get_device, load_cont_ae,
                    load_discrete_ae, save_grid, set_seed, to_unit)
from models import ARLatentTransformer, MaskedTokenDenoiser


# %% [markdown]
# ## 1. Load the outputs of steps 01b, 02, 04 and 05

# %%
def load_step_outputs(device):
    ck = torch.load(find_file("ddiff_best.pth"), map_location=device)
    ddiff = MaskedTokenDenoiser(**ck["config"]).to(device)
    ddiff.load_state_dict(ck["model"])
    ddiff.eval()
    kind = ck.get("kind", "vq")
    dae, _ = load_discrete_ae(device, kind)
    print(f"Discrete diffusion : epoch {ck['epoch']}, val loss {ck['val_loss']:.4f} (autoencoder: {kind})")

    cont = load_cont_ae(device)
    ck = torch.load(find_file("ltrans_best.pth"), map_location=device)
    ltrans = ARLatentTransformer(**ck["config"]).to(device)
    ltrans.load_state_dict(ck["model"])
    ltrans.eval()
    mean, std = ck["mean"].to(device), ck["std"].to(device)
    print(f"Latent transformer : epoch {ck['epoch']}, val loss {ck['val_loss']:.4f}")
    return dae, cont, ddiff, ltrans, mean, std


# %% [markdown]
# ## 2. Generators (yield batches of images in [-1, 1]) and FID

# %%
@torch.no_grad()
def gen_discrete(ddiff, dae, device, n, batch, **kw):
    done = 0
    while done < n:
        b = min(batch, n - done)
        with amp_ctx(device):
            tokens = ddiff.sample(b, device=device, **kw)
        yield dae.decode_tokens(tokens)
        done += b


@torch.no_grad()
def gen_continuous(ltrans, cont, mean, std, device, n, batch, **kw):
    done = 0
    while done < n:
        b = min(batch, n - done)
        with amp_ctx(device):
            x = ltrans.sample(b, device=device, **kw)
        z = x.float().view(b, 8, 8, -1).permute(0, 3, 1, 2)
        z = z * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
        yield cont.decode(z)
        done += b


@torch.no_grad()
def gen_reconstructions(ae_forward, loader, device, n):
    done = 0
    for x, _ in loader:
        x = x.to(device)[: n - done]
        yield ae_forward(x)
        done += x.size(0)
        if done >= n:
            break


REAL_STATE = ("real_features_sum", "real_features_cov_sum", "real_features_num_samples")
REAL_CACHE = OUT_DIR / "fid_real_stats_cifar10_train.pt"


def build_fid(device, train_loader):
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False).to(device)

    # --- diagnostic: is the Inception network really on the GPU, and how fast is it? ---
    net = getattr(fid, "inception", None)
    if net is not None:
        net.to(device)
        print(f"FID Inception network device: {next(net.parameters()).device}")
    dummy = torch.rand(250, 3, 32, 32, device=device)
    fid.update(dummy, real=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    fid.update(dummy, real=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"FID feature extraction speed: {250 / (time.time() - t0):.0f} images/s "
          f"(RTX 4050 should manage several hundred; ~20 means it is running on the CPU)")
    fid.reset()

    # --- real statistics (cached) ---
    if REAL_CACHE.exists() and all(hasattr(fid, k) for k in REAL_STATE):
        d = torch.load(REAL_CACHE, map_location=device)
        for k in REAL_STATE:
            getattr(fid, k).copy_(d[k].to(device))
        print(f"Loaded cached real-image statistics: {REAL_CACHE.name}")
        return fid
    for x, _ in tqdm(train_loader, desc="FID: real CIFAR-10 statistics"):
        fid.update(to_unit(x.to(device)), real=True)
    if all(hasattr(fid, k) for k in REAL_STATE):
        torch.save({k: getattr(fid, k).detach().cpu() for k in REAL_STATE}, REAL_CACHE)
        print(f"Cached real-image statistics -> {REAL_CACHE.name}")
    return fid


def fid_score(fid, batches, total, desc):
    fid.reset()                                   # keeps the real statistics
    with tqdm(total=total, desc=desc) as bar:
        for imgs in batches:
            fid.update(to_unit(imgs).float(), real=False)
            bar.update(imgs.size(0))
    return fid.compute().item()


# %% [markdown]
# ## 3. Optional: quick sampler tuning (small sample count)

# %%
def tune_discrete(fid, ddiff, dae, device, args):
    best = (float("inf"), None)
    for temp in (0.8, 0.9, 1.0):
        for ct in (1.0, 4.5):
            set_seed(args.seed)
            kw = dict(steps=args.d_steps, temperature=temp, choice_temperature=ct)
            f = fid_score(fid, gen_discrete(ddiff, dae, device, args.tune_n, args.batch, **kw),
                          args.tune_n, f"tune discrete T={temp} choice={ct}")
            print(f"  discrete  temperature={temp}  choice_temperature={ct}  ->  FID {f:.2f}")
            if f < best[0]:
                best = (f, kw)
    return best[1]


def tune_continuous(fid, ltrans, cont, mean, std, device, args):
    best = (float("inf"), None)
    for temp in (0.8, 0.9, 1.0):
        set_seed(args.seed)
        kw = dict(steps=args.c_steps, eta=args.c_eta, temperature=temp, x0_clip=args.c_clip)
        f = fid_score(fid, gen_continuous(ltrans, cont, mean, std, device, args.tune_n, args.batch, **kw),
                      args.tune_n, f"tune continuous T={temp}")
        print(f"  continuous temperature={temp}  ->  FID {f:.2f}")
        if f < best[0]:
            best = (f, kw)
    return best[1]


# %% [markdown]
# ## 4. Run

# %%
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_gen", type=int, default=10000)      # use 50000 for the final report
    p.add_argument("--batch", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tune", action="store_true")
    p.add_argument("--tune_n", type=int, default=5000)
    p.add_argument("--skip_recon", action="store_true")
    p.add_argument("--recon_only", action="store_true")   # only the autoencoders' reconstruction FID (no generators needed)
    # discrete diffusion sampler
    p.add_argument("--d_steps", type=int, default=32)
    p.add_argument("--d_temp", type=float, default=1.0)
    p.add_argument("--d_choice_temp", type=float, default=4.5)
    # continuous transformer sampler
    p.add_argument("--c_steps", type=int, default=50)
    p.add_argument("--c_temp", type=float, default=1.0)
    p.add_argument("--c_eta", type=float, default=1.0)
    p.add_argument("--c_clip", type=float, default=6.0)
    args, _ = p.parse_known_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    if args.recon_only:
        dae, _ = load_discrete_ae(device, "auto")
        cont = load_cont_ae(device)
        train_loader, _ = get_cifar10_eval_loaders(batch_size=500)
        fid = build_fid(device, train_loader)
        f_d = fid_score(fid, gen_reconstructions(lambda x: dae(x)[0], train_loader, device, args.num_gen),
                        args.num_gen, "FID: discrete AE reconstructions")
        f_c = fid_score(fid, gen_reconstructions(lambda x: cont(x)[0], train_loader, device, args.num_gen),
                        args.num_gen, "FID: continuous AE reconstructions")
        print("=" * 60)
        print(f"Reconstruction FID ({args.num_gen} images): discrete AE {f_d:.2f} | continuous AE {f_c:.2f}")
        print("These are the floors for the generators (aim for both well below 40, ideally < 30).")
        print("=" * 60)
        return

    dae, cont, ddiff, ltrans, mean, std = load_step_outputs(device)
    train_loader, _ = get_cifar10_eval_loaders(batch_size=500)
    fid = build_fid(device, train_loader)
    results = {"num_gen": args.num_gen}

    # -- FID of the autoencoder reconstructions: the best any generator on top of them can reach
    if not args.skip_recon:
        results["fid_recon_discrete_ae"] = fid_score(
            fid, gen_reconstructions(lambda x: dae(x)[0], train_loader, device, args.num_gen),
            args.num_gen, "FID: discrete AE reconstructions")
        results["fid_recon_cont_ae"] = fid_score(
            fid, gen_reconstructions(lambda x: cont(x)[0], train_loader, device, args.num_gen),
            args.num_gen, "FID: continuous AE reconstructions")
        print(f"Reconstruction FID  discrete AE {results['fid_recon_discrete_ae']:.2f} | "
              f"continuous AE {results['fid_recon_cont_ae']:.2f}   (lower bound for the generators)")

    d_kw = dict(steps=args.d_steps, temperature=args.d_temp, choice_temperature=args.d_choice_temp)
    c_kw = dict(steps=args.c_steps, eta=args.c_eta, temperature=args.c_temp, x0_clip=args.c_clip)
    if args.tune:
        print("Tuning samplers on", args.tune_n, "samples each ...")
        d_kw = tune_discrete(fid, ddiff, dae, device, args)
        c_kw = tune_continuous(fid, ltrans, cont, mean, std, device, args)
    results["discrete_sampler"], results["continuous_sampler"] = d_kw, c_kw

    # -- final evaluation + image grids
    set_seed(args.seed)
    t0 = time.time()
    results["fid_discrete_diffusion"] = fid_score(
        fid, gen_discrete(ddiff, dae, device, args.num_gen, args.batch, **d_kw),
        args.num_gen, "FID: discrete latent diffusion")
    results["time_discrete_s"] = round(time.time() - t0)

    t0 = time.time()
    results["fid_latent_transformer"] = fid_score(
        fid, gen_continuous(ltrans, cont, mean, std, device, args.num_gen, args.batch, **c_kw),
        args.num_gen, "FID: continuous latent transformer")
    results["time_continuous_s"] = round(time.time() - t0)

    set_seed(args.seed)
    save_grid(next(gen_discrete(ddiff, dae, device, 64, 64, **d_kw)), OUT_DIR / "ddiff_final_samples.png")
    save_grid(next(gen_continuous(ltrans, cont, mean, std, device, 64, 64, **c_kw)),
              OUT_DIR / "ltrans_final_samples.png")

    with open(OUT_DIR / "part4_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print(f"SUMMARY (FID with {args.num_gen} generated images vs CIFAR-10 train set; target < 40)")
    for name, key in (("Discrete latent diffusion", "fid_discrete_diffusion"),
                      ("Continuous latent transformer", "fid_latent_transformer")):
        v = results[key]
        print(f"{name:32s}: FID {v:6.2f}   {'PASS' if v < 40 else 'above target'}")
    print(f"Saved: {OUT_DIR / 'part4_results.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
